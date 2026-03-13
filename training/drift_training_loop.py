# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Prototype drift-only training loop."""

import copy
import json
import os
import pickle
import time

import numpy as np
import PIL.Image
import psutil
import torch

import dnnlib
import legacy
from metrics import metric_main
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix

from training.drift_loss import DriftLossConfig, cfg_alpha_to_unconditional_weight, grouped_drifting_stopgrad_loss
from training.drift_queue import ClassConditionalSampleQueue, QueueConfig, ensure_class_coverage
from training.training_loop import cosine_decay_with_warmup, remap_optimizer_state_dict, save_image_grid, setup_snapshot_image_grid

#----------------------------------------------------------------------------


def training_loop(
    run_dir                 = '.',
    training_set_kwargs     = {},
    data_loader_kwargs      = {},
    G_kwargs                = {},
    G_opt_kwargs            = {},
    lr_scheduler            = None,
    beta2_scheduler         = None,
    metrics                 = [],
    random_seed             = 0,
    num_gpus                = 1,
    rank                    = 0,
    batch_size              = 4,
    ema_scheduler           = None,
    total_kimg              = 25000,
    kimg_per_tick           = 4,
    image_snapshot_ticks    = 50,
    network_snapshot_ticks  = 50,
    snapshot_policy         = 'all',
    resume_pkl              = None,
    cudnn_benchmark         = True,
    abort_fn                = None,
    progress_fn             = None,
    negatives_per_group     = 4,
    positives_per_group     = 4,
    unconditional_per_group = 2,
    alpha_min               = 1.0,
    alpha_max               = 4.0,
    drift_temperature       = 0.05,
    queue_capacity_per_class = 256,
    queue_capacity_global   = 4096,
    queue_push_batch        = 128,
    queue_warmup_batches    = 4,
    drift_config            = None,
    **_unused_kwargs,
):
    # The shared launcher forwards the full config EasyDict to both trainers.
    # Drift training ignores GAN-only keys such as D_kwargs and augment settings.
    if drift_config is not None and str(getattr(drift_config, 'backbone', 'r3gan_conv')) == 'dit_like':
        from training import drift_research

        return drift_research.training_loop(
            run_dir=run_dir,
            training_set_kwargs=training_set_kwargs,
            data_loader_kwargs=data_loader_kwargs,
            G_kwargs=G_kwargs,
            G_opt_kwargs=G_opt_kwargs,
            lr_scheduler=lr_scheduler,
            beta2_scheduler=beta2_scheduler,
            metrics=metrics,
            random_seed=random_seed,
            num_gpus=num_gpus,
            rank=rank,
            batch_size=batch_size,
            total_kimg=total_kimg,
            kimg_per_tick=kimg_per_tick,
            image_snapshot_ticks=image_snapshot_ticks,
            network_snapshot_ticks=network_snapshot_ticks,
            resume_pkl=resume_pkl,
            cudnn_benchmark=cudnn_benchmark,
            abort_fn=abort_fn,
            progress_fn=progress_fn,
            drift_config=drift_config,
            **_unused_kwargs,
        )
    StartTime = time.time()
    Device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    if batch_size % num_gpus != 0:
        raise ValueError('batch_size must be divisible by num_gpus')
    LocalBatchSize = batch_size // num_gpus
    if LocalBatchSize % negatives_per_group != 0:
        raise ValueError('batch_size / num_gpus must be divisible by negatives_per_group')
    if queue_push_batch % num_gpus != 0:
        raise ValueError('queue_push_batch must be divisible by num_gpus')
    LocalQueuePushBatch = queue_push_batch // num_gpus
    Groups = LocalBatchSize // negatives_per_group

    if rank == 0:
        print('Loading training set...')
    TrainingSet = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    if not TrainingSet.has_labels:
        raise ValueError('drift trainer requires a labeled dataset')
    TrainingSetSampler = misc.InfiniteSampler(
        dataset=TrainingSet,
        rank=rank,
        num_replicas=num_gpus,
        seed=random_seed,
    )
    LoaderBatchSize = max(LocalBatchSize, LocalQueuePushBatch)
    TrainingSetIterator = iter(
        torch.utils.data.DataLoader(
            dataset=TrainingSet,
            sampler=TrainingSetSampler,
            batch_size=LoaderBatchSize,
            **data_loader_kwargs,
        )
    )
    if rank == 0:
        print()
        print('Num images: ', len(TrainingSet))
        print('Image shape:', TrainingSet.image_shape)
        print('Label shape:', TrainingSet.label_shape)
        print()

    if rank == 0:
        print('Constructing networks...')
    CommonKwargs = dict(c_dim=TrainingSet.label_dim, img_resolution=TrainingSet.resolution)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **CommonKwargs).train().requires_grad_(False).to(Device)
    G_ema = copy.deepcopy(G).eval()
    G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
    Queue = ClassConditionalSampleQueue(
        QueueConfig(
            num_classes=TrainingSet.label_dim,
            per_class_capacity=queue_capacity_per_class,
            global_capacity=queue_capacity_global,
            store_device='cpu',
            strict_without_replacement=False,
        )
    )

    ResumeData = None
    if resume_pkl is not None:
        with dnnlib.util.open_url(resume_pkl) as f:
            ResumeData = legacy.load_network_pkl(f)
        if rank == 0:
            print(f'Resuming from "{resume_pkl}"')
        for Name, Module in [('G', G), ('G_ema', G_ema)]:
            if ResumeData.get(Name) is not None:
                misc.copy_params_and_buffers(ResumeData[Name], Module, require_all=False)
        if 'G_opt_state' in ResumeData:
            G_opt.load_state_dict(remap_optimizer_state_dict(ResumeData['G_opt_state'], Device))
        if 'queue_state' in ResumeData and isinstance(ResumeData['queue_state'], dict):
            Queue.load_state_dict(ResumeData['queue_state'])

    if rank == 0:
        z = torch.empty([LocalBatchSize, G.z_dim], device=Device)
        c = torch.empty([LocalBatchSize, G.c_dim], device=Device)
        misc.print_module_summary(G, [z, c])

    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for Module in [G, G_ema]:
        if Module is not None and num_gpus > 1:
            for Param in misc.params_and_buffers(Module):
                torch.distributed.broadcast(Param, src=0)

    if rank == 0:
        print('Priming drift queue...')
    if ResumeData is None or 'queue_state' not in ResumeData:
        for _idx in range(queue_warmup_batches):
            WarmImages, WarmLabels = _next_queue_batch(
                training_set_iterator=TrainingSetIterator,
                device=Device,
                local_queue_push_batch=LocalQueuePushBatch,
            )
            Queue.push(WarmImages, WarmLabels)

    GridSize = None
    GridZ = None
    GridC = None
    if rank == 0:
        print('Exporting sample images...')
        GridSize, Images, Labels = setup_snapshot_image_grid(training_set=TrainingSet)
        save_image_grid(
            Images,
            os.path.join(run_dir, 'reals.png'),
            drange=[0, 255],
            grid_size=GridSize,
        )
        GridZ = torch.randn([Labels.shape[0], G.z_dim], device=Device).split(LocalBatchSize)
        GridC = torch.from_numpy(Labels).to(Device).split(LocalBatchSize)
        Images = torch.cat([G_ema(z, c).cpu() for z, c in zip(GridZ, GridC)]).to(torch.float).numpy()
        save_image_grid(
            Images,
            os.path.join(run_dir, 'fakes_init.png'),
            drange=[-1, 1],
            grid_size=GridSize,
        )

    if rank == 0:
        print('Initializing logs...')
    StatsCollector = training_stats.Collector(regex='.*')
    StatsMetrics = dict()
    StatsJsonl = None
    StatsTfevents = None
    if rank == 0:
        StatsJsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            StatsTfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    LatestSnapshotPkl = None
    LatestImagePng = None
    BestSnapshotPkl = None
    BestImagePng = None
    BestMetricName = metrics[0] if len(metrics) > 0 else None
    BestMetricValue = None

    def _is_better_metric(metric_name, candidate, reference):
        if metric_name.startswith('fid') or metric_name.startswith('kid'):
            return candidate < reference
        return candidate > reference

    def _safe_remove(path, protected):
        if path is None or path in protected:
            return
        if os.path.isfile(path):
            os.remove(path)

    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    CurNimg = ResumeData['cur_nimg'] if ResumeData is not None else 0
    CurTick = 0
    TickStartNimg = CurNimg
    TickStartTime = time.time()
    MaintenanceTime = TickStartTime - StartTime
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    StepStartEvent = None
    StepEndEvent = None
    if rank == 0:
        StepStartEvent = torch.cuda.Event(enable_timing=True)
        StepEndEvent = torch.cuda.Event(enable_timing=True)
        StepStartEvent.record(torch.cuda.current_stream(Device))
        StepEndEvent.record(torch.cuda.current_stream(Device))

    LossConfig = DriftLossConfig(temperature=drift_temperature)
    LastAlphaMean = 0.0
    LastDriftNorm = 0.0

    while True:
        if StepStartEvent is not None:
            StepStartEvent.record(torch.cuda.current_stream(Device))

        CurLr = cosine_decay_with_warmup(CurNimg, **lr_scheduler) if lr_scheduler is not None else G_opt.param_groups[0]['lr']
        CurBeta2 = cosine_decay_with_warmup(CurNimg, **beta2_scheduler) if beta2_scheduler is not None else G_opt.param_groups[0]['betas'][1]
        CurEmaNimg = cosine_decay_with_warmup(CurNimg, **ema_scheduler) if ema_scheduler is not None else float(batch_size)

        for Group in G_opt.param_groups:
            Group['lr'] = CurLr
            Group['betas'] = (0.0, float(CurBeta2))

        QueueImages, QueueLabels = _next_queue_batch(
            training_set_iterator=TrainingSetIterator,
            device=Device,
            local_queue_push_batch=LocalQueuePushBatch,
        )
        Queue.push(QueueImages, QueueLabels)

        GroupClassIds = torch.randint(0, G.c_dim, [Groups], device=Device)
        ensure_class_coverage(
            Queue,
            GroupClassIds,
            refill_fn=lambda: _next_queue_batch(
                training_set_iterator=TrainingSetIterator,
                device=Device,
                local_queue_push_batch=LocalQueuePushBatch,
            ),
        )

        PositivesGrouped = Queue.sample_positive_grouped(GroupClassIds, positives_per_group, Device)
        UnconditionalGrouped = Queue.sample_unconditional_grouped(Groups, unconditional_per_group, Device)
        AlphaGrouped = _sample_alpha(Groups, alpha_min, alpha_max, Device)
        AlphaWeights = torch.tensor(
            [
                cfg_alpha_to_unconditional_weight(
                    alpha=float(Value.item()),
                    n_generated_negatives=negatives_per_group,
                    n_unconditional_negatives=unconditional_per_group,
                )
                for Value in AlphaGrouped
            ],
            device=Device,
            dtype=torch.float32,
        )

        z = torch.randn([LocalBatchSize, G.z_dim], device=Device)
        c = _one_hot(GroupClassIds.repeat_interleave(negatives_per_group), G.c_dim, Device)
        AlphaFlat = AlphaGrouped.repeat_interleave(negatives_per_group)

        G_opt.zero_grad(set_to_none=True)
        G.requires_grad_(True)
        FakeImages = G(z, c, alpha=AlphaFlat)
        FakeGrouped = FakeImages.reshape(
            Groups,
            negatives_per_group,
            FakeImages.shape[1],
            FakeImages.shape[2],
            FakeImages.shape[3],
        )
        Loss, DriftStats = grouped_drifting_stopgrad_loss(
            x_grouped=FakeGrouped,
            y_pos_grouped=PositivesGrouped,
            unconditional_grouped=UnconditionalGrouped,
            unconditional_weight_grouped=AlphaWeights,
            config=LossConfig,
        )
        Loss.backward()
        G.requires_grad_(False)

        Params = [Param for Param in G.parameters() if Param.grad is not None]
        if len(Params) > 0:
            Flat = torch.cat([Param.grad.flatten() for Param in Params])
            if num_gpus > 1:
                torch.distributed.all_reduce(Flat)
                Flat /= num_gpus
            Grads = Flat.split([Param.numel() for Param in Params])
            for Param, Grad in zip(Params, Grads):
                Param.grad = Grad.reshape(Param.shape)
        G_opt.step()

        with torch.autograd.profiler.record_function('Gema'):
            EmaBeta = 0.5 ** (batch_size / max(CurEmaNimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, EmaBeta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        LastAlphaMean = float(AlphaGrouped.mean().item())
        LastDriftNorm = float(DriftStats['mean_drift_norm'])
        training_stats.report('Loss/G/loss', Loss.detach())
        training_stats.report('Loss/G/total', Loss.detach())
        training_stats.report('Loss/drift_norm', torch.as_tensor(DriftStats['mean_drift_norm'], device=Device))
        training_stats.report('Loss/drift_pos_norm', torch.as_tensor(DriftStats['mean_drift_pos_norm'], device=Device))
        training_stats.report('Loss/drift_neg_norm', torch.as_tensor(DriftStats['mean_drift_neg_norm'], device=Device))
        training_stats.report('Progress/alpha_mean', torch.as_tensor(LastAlphaMean, device=Device))
        training_stats.report('Progress/alpha_min', AlphaGrouped.min())
        training_stats.report('Progress/alpha_max', AlphaGrouped.max())
        training_stats.report('Progress/queue_global_count', torch.as_tensor(float(Queue.global_count()), device=Device))
        training_stats.report(
            'Progress/queue_covered_classes',
            torch.as_tensor(float(sum(1 for Count in Queue.class_counts() if Count > 0)), device=Device),
        )

        if StepEndEvent is not None:
            StepEndEvent.record(torch.cuda.current_stream(Device))

        CurNimg += batch_size
        Done = CurNimg >= total_kimg * 1000
        if (not Done) and (CurTick != 0) and (CurNimg < TickStartNimg + kimg_per_tick * 1000):
            continue

        TickEndTime = time.time()
        training_stats.report0('Progress/tick', CurTick)
        training_stats.report0('Progress/kimg', CurNimg / 1e3)
        training_stats.report0('Progress/lr', CurLr)
        training_stats.report0('Progress/ema_mimg', CurEmaNimg / 1e6)
        training_stats.report0('Progress/beta2', CurBeta2)
        training_stats.report0('Progress/alpha_mean', LastAlphaMean)
        training_stats.report0('Progress/drift_norm', LastDriftNorm)
        training_stats.report0('Progress/queue_global_count', float(Queue.global_count()))
        training_stats.report0('Timing/total_sec', TickEndTime - StartTime)
        training_stats.report0('Timing/sec_per_tick', TickEndTime - TickStartTime)
        training_stats.report0(
            'Timing/sec_per_kimg',
            (TickEndTime - TickStartTime) / max(CurNimg - TickStartNimg, 1) * 1e3,
        )
        training_stats.report0('Timing/maintenance_sec', MaintenanceTime)
        training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30)
        training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(Device) / 2**30)
        training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(Device) / 2**30)
        training_stats.report0('Timing/total_hours', (TickEndTime - StartTime) / (60 * 60))
        training_stats.report0('Timing/total_days', (TickEndTime - StartTime) / (24 * 60 * 60))
        torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            Fields = []
            Fields += [f"tick {CurTick:<5d}"]
            Fields += [f"kimg {CurNimg / 1e3:<8.1f}"]
            Fields += [f"time {dnnlib.util.format_time(TickEndTime - StartTime):<12s}"]
            Fields += [f"sec/tick {TickEndTime - TickStartTime:<7.1f}"]
            Fields += [f"sec/kimg {(TickEndTime - TickStartTime) / max(CurNimg - TickStartNimg, 1) * 1e3:<7.2f}"]
            Fields += [f"maintenance {MaintenanceTime:<6.1f}"]
            Fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"]
            Fields += [f"gpumem {torch.cuda.max_memory_allocated(Device) / 2**30:<6.2f}"]
            Fields += [f"reserved {torch.cuda.max_memory_reserved(Device) / 2**30:<6.2f}"]
            Fields += [f"alpha {LastAlphaMean:.3f}"]
            Fields += [f"drift {LastDriftNorm:.4f}"]
            Fields += [f"queue {Queue.global_count():<6d}"]
            print(' '.join(Fields))

        if (not Done) and (abort_fn is not None) and abort_fn():
            Done = True
            if rank == 0:
                print()
                print('Aborting...')

        SaveNetworkThisTick = (network_snapshot_ticks is not None) and (Done or CurTick % network_snapshot_ticks == 0)
        SaveImageThisTick = (rank == 0) and (
            ((image_snapshot_ticks is not None) and (Done or CurTick % image_snapshot_ticks == 0))
            or (snapshot_policy == 'latest-best' and SaveNetworkThisTick)
        )

        ImageSnapshotPath = None
        if SaveImageThisTick:
            Images = torch.cat([G_ema(z, c).cpu() for z, c in zip(GridZ, GridC)]).to(torch.float).numpy()
            ImageSnapshotPath = os.path.join(run_dir, f'fakes{CurNimg // 1000:09d}.png')
            save_image_grid(Images, ImageSnapshotPath, drange=[-1, 1], grid_size=GridSize)

        SnapshotPkl = None
        SnapshotData = None
        if SaveNetworkThisTick:
            SnapshotData = dict(
                G=G,
                D=None,
                G_ema=G_ema,
                training_set_kwargs=dict(training_set_kwargs),
                cur_nimg=CurNimg,
                trainer='drift',
                queue_state=Queue.state_dict(),
                G_opt_state=remap_optimizer_state_dict(G_opt.state_dict(), 'cpu'),
            )
            for Key, Value in list(SnapshotData.items()):
                if isinstance(Value, torch.nn.Module):
                    Value = copy.deepcopy(Value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(Value)
                        for Param in misc.params_and_buffers(Value):
                            torch.distributed.broadcast(Param, src=0)
                    SnapshotData[Key] = Value.cpu()
            SnapshotPkl = os.path.join(run_dir, f'network-snapshot-{CurNimg // 1000:09d}.pkl')
            if rank == 0:
                with open(SnapshotPkl, 'wb') as f:
                    pickle.dump(SnapshotData, f)

        SnapshotMetricResults = dict()
        if (SnapshotData is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for Metric in metrics:
                ResultDict = metric_main.calc_metric(
                    metric=Metric,
                    G=SnapshotData['G_ema'],
                    dataset_kwargs=training_set_kwargs,
                    num_gpus=num_gpus,
                    rank=rank,
                    device=Device,
                )
                if rank == 0:
                    metric_main.report_metric(ResultDict, run_dir=run_dir, snapshot_pkl=SnapshotPkl)
                StatsMetrics.update(ResultDict.results)
                SnapshotMetricResults.update(ResultDict.results)
        del SnapshotData

        if (rank == 0) and (snapshot_policy == 'latest-best') and (SnapshotPkl is not None):
            IsBest = False
            MetricValue = None
            if (BestMetricName is not None) and (BestMetricName in SnapshotMetricResults):
                MetricValue = SnapshotMetricResults[BestMetricName]
                if (BestMetricValue is None) or _is_better_metric(BestMetricName, MetricValue, BestMetricValue):
                    IsBest = True
            elif BestSnapshotPkl is None:
                IsBest = True

            PrevBestSnapshotPkl = BestSnapshotPkl
            PrevBestImagePng = BestImagePng
            if IsBest:
                BestSnapshotPkl = SnapshotPkl
                BestImagePng = ImageSnapshotPath
                if MetricValue is not None:
                    BestMetricValue = MetricValue
                    print(f'Updated best snapshot by {BestMetricName}: {BestMetricValue:.6f}')

            Protected = {SnapshotPkl, ImageSnapshotPath, BestSnapshotPkl, BestImagePng}
            _safe_remove(LatestSnapshotPkl, Protected)
            _safe_remove(LatestImagePng, Protected)
            _safe_remove(PrevBestSnapshotPkl, Protected)
            _safe_remove(PrevBestImagePng, Protected)
            LatestSnapshotPkl = SnapshotPkl
            LatestImagePng = ImageSnapshotPath

        StepTiming = []
        if StepStartEvent is not None and StepEndEvent is not None:
            StepEndEvent.synchronize()
            StepTiming = StepStartEvent.elapsed_time(StepEndEvent)
        training_stats.report0('Timing/G', StepTiming)
        StatsCollector.update()
        StatsDict = StatsCollector.as_dict()

        Timestamp = time.time()
        if StatsJsonl is not None:
            Fields = dict(StatsDict, timestamp=Timestamp)
            StatsJsonl.write(json.dumps(Fields) + '\n')
            StatsJsonl.flush()
        if StatsTfevents is not None:
            GlobalStep = int(CurNimg / 1e3)
            Walltime = Timestamp - StartTime
            for Name, Value in StatsDict.items():
                StatsTfevents.add_scalar(Name, Value.mean, global_step=GlobalStep, walltime=Walltime)
            for Name, Value in StatsMetrics.items():
                StatsTfevents.add_scalar(f'Metrics/{Name}', Value, global_step=GlobalStep, walltime=Walltime)
            StatsTfevents.flush()
        if progress_fn is not None:
            progress_fn(CurNimg // 1000, total_kimg)

        CurTick += 1
        TickStartNimg = CurNimg
        TickStartTime = time.time()
        MaintenanceTime = TickStartTime - TickEndTime
        if Done:
            break

    if rank == 0:
        print()
        print('Exiting...')


#----------------------------------------------------------------------------


def _next_queue_batch(training_set_iterator, device, local_queue_push_batch):
    Images, Labels = next(training_set_iterator)
    Images = Images[:local_queue_push_batch].detach().clone().to(device).to(torch.float32) / 127.5 - 1
    Labels = Labels[:local_queue_push_batch].detach().clone().to(device)
    return Images, _class_ids_from_labels(Labels)


def _class_ids_from_labels(labels):
    if labels.ndim == 1:
        return labels.long()
    if labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError('drift trainer requires one-hot or integer labels')
    return labels.argmax(dim=1).long()


def _one_hot(class_ids, num_classes, device):
    Out = torch.zeros([class_ids.shape[0], num_classes], device=device, dtype=torch.float32)
    Out.scatter_(1, class_ids.view(-1, 1), 1.0)
    return Out


def _sample_alpha(groups, alpha_min, alpha_max, device):
    if alpha_max < alpha_min:
        raise ValueError('alpha_max must be >= alpha_min')
    if alpha_max == alpha_min:
        return torch.full([groups], float(alpha_min), device=device, dtype=torch.float32)
    return torch.rand([groups], device=device, dtype=torch.float32) * (alpha_max - alpha_min) + alpha_min
