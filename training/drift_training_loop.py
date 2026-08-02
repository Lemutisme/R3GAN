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
from math import sqrt

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

from training.drift_field import DriftFieldConfig, cfg_alpha_to_unconditional_weight, cfg_alpha_to_unconditional_weight_vectorized, build_negative_log_weights
from training.drift_loss import DriftingLossConfig, FeatureDriftingConfig, drifting_stopgrad_loss
from training.drift_queue import ClassConditionalSampleQueue, QueueConfig, ensure_class_coverage
from training.drift_stage2 import GroupedDriftStepConfig, grouped_drift_training_step
from training.features.extractors import TinyFeatureEncoder, TinyFeatureEncoderConfig, freeze_module_parameters
from training.features.vectorize import FeatureVectorizationConfig
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
    use_bf16                = True,
    clip_grad_norm          = 2.0,
    **_unused_kwargs,
):
    # The shared launcher forwards the full config EasyDict to both trainers.
    # Drift training ignores GAN-only keys such as D_kwargs and augment settings.
    StartTime = time.time()
    Device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True
    UseBf16 = use_bf16 and torch.cuda.is_bf16_supported()

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
            store_device=Device,
            strict_without_replacement=False,
        )
    )

    # --- Feature loss setup ---
    _use_feature_loss = drift_config is not None and getattr(drift_config, 'use_feature_loss', False)
    FeatureExtractor = None
    FeatureStepConfig = None
    if _use_feature_loss:
        _feat_enc = str(getattr(drift_config, 'feature_encoder', 'tiny'))
        if _feat_enc != 'tiny':
            raise ValueError(f'Only "tiny" feature encoder is supported, got "{_feat_enc}"')
        _feat_in_ch = TrainingSet.image_shape[0]  # 3 for RGB
        _feat_base_ch = int(getattr(drift_config, 'feature_base_channels', 16))
        _feat_stages = int(getattr(drift_config, 'feature_stages', 3))
        FeatureExtractor = TinyFeatureEncoder(
            TinyFeatureEncoderConfig(
                in_channels=_feat_in_ch,
                base_channels=_feat_base_ch,
                stages=_feat_stages,
            )
        ).to(Device).eval()
        freeze_module_parameters(FeatureExtractor)

        _feat_temps_raw = getattr(drift_config, 'feature_temperatures', [0.02, 0.05, 0.2])
        _feat_temps = tuple(float(t) for t in _feat_temps_raw) if _feat_temps_raw else (drift_temperature,)
        _feat_selected = getattr(drift_config, 'feature_selected_stages', None)
        if _feat_selected is not None:
            _feat_selected = tuple(int(s) for s in _feat_selected)
        _feat_vec_cfg = FeatureVectorizationConfig(
            include_input_x2_mean=bool(getattr(drift_config, 'include_input_x2_mean', False)),
            include_patch4_stats=bool(getattr(drift_config, 'include_patch4_stats', True)),
            selected_stages=_feat_selected,
        )
        _feat_cfg = FeatureDriftingConfig(
            temperatures=_feat_temps,
            vectorization=_feat_vec_cfg,
            temperature_aggregation=str(getattr(drift_config, 'feature_temperature_aggregation', 'per_temperature_mse')),
            loss_term_reduction=str(getattr(drift_config, 'feature_loss_term_reduction', 'sum')),
            scale_temperature_by_sqrt_channels=not bool(getattr(drift_config, 'disable_feature_temperature_sqrt_scaling', False)),
            share_location_normalization=not bool(getattr(drift_config, 'disable_shared_location_normalization', False)),
            include_raw_drift_loss=bool(getattr(drift_config, 'feature_include_raw_drift_loss', False)),
            raw_drift_loss_weight=float(getattr(drift_config, 'feature_raw_drift_loss_weight', 1.0)),
        )
        _drift_field_cfg = DriftFieldConfig(temperature=drift_temperature)
        _base_loss_cfg = DriftingLossConfig(drift_field=_drift_field_cfg)
        FeatureStepConfig = GroupedDriftStepConfig(
            loss_config=_base_loss_cfg,
            feature_config=_feat_cfg,
            clip_grad_norm=clip_grad_norm if clip_grad_norm > 0 else None,
            run_optimizer_step=False,  # We handle optimizer step ourselves
        )
        if rank == 0:
            _feat_params = sum(p.numel() for p in FeatureExtractor.parameters())
            print(f'Feature encoder: {_feat_enc} ({_feat_params:,} params, frozen)')
            print(f'Feature temperatures: {_feat_temps}')
            print(f'Feature stages: {_feat_stages}, base_channels: {_feat_base_ch}')

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

    _DriftFieldCfg = DriftFieldConfig(temperature=drift_temperature)
    _DriftLossCfg = DriftingLossConfig(drift_field=_DriftFieldCfg)
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
        AlphaWeights = cfg_alpha_to_unconditional_weight_vectorized(
            AlphaGrouped, negatives_per_group, unconditional_per_group,
        )

        z = torch.randn([LocalBatchSize, G.z_dim], device=Device)
        c = _one_hot(GroupClassIds.repeat_interleave(negatives_per_group), G.c_dim, Device)
        AlphaFlat = AlphaGrouped.repeat_interleave(negatives_per_group)

        G_opt.zero_grad(set_to_none=True)
        G.requires_grad_(True)

        if _use_feature_loss:
            # --- Feature-space drift loss via drift_stage2 ---
            # Reshape noise into grouped 5D: [G, N_gen, C, H, W]
            NoiseGrouped = z.reshape(Groups, negatives_per_group, *TrainingSet.image_shape)
            ClassLabelsGrouped = GroupClassIds  # [G] integer class ids
            StepResult = grouped_drift_training_step(
                generator=G,
                optimizer=None,  # we handle optimizer step ourselves (for multi-GPU grad sync)
                noise_grouped=NoiseGrouped,
                class_labels_grouped=ClassLabelsGrouped,
                alpha_grouped=AlphaGrouped,
                positives_grouped=PositivesGrouped,
                style_indices_grouped=None,
                unconditional_grouped=UnconditionalGrouped,
                unconditional_weight_grouped=AlphaWeights,
                feature_extractor=FeatureExtractor,
                feature_input_transform=None,
                amp_dtype=torch.bfloat16 if UseBf16 else None,
                config=FeatureStepConfig,
                backward_when_no_step=True,
            )
            Loss = torch.tensor(StepResult['loss'], device=Device)
            DriftStats = {
                'mean_drift_norm': StepResult.get('mean_drift_norm', 0.0),
                'mean_drift_pos_norm': StepResult.get('drift_pos_norm', StepResult.get('mean_drift_pos_norm', 0.0)),
                'mean_drift_neg_norm': StepResult.get('drift_neg_norm', StepResult.get('mean_drift_neg_norm', 0.0)),
            }
        else:
            # --- Pixel-space drift loss (original path) ---
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=UseBf16):
                FakeImages = G(z, c, alpha=AlphaFlat)
            FakeImages = FakeImages.float()
            FakeGrouped = FakeImages.reshape(
                Groups, negatives_per_group,
                FakeImages.shape[1], FakeImages.shape[2], FakeImages.shape[3],
            )

            PixelDim = FakeImages.shape[1] * FakeImages.shape[2] * FakeImages.shape[3]
            x_grouped = FakeGrouped.reshape(Groups, negatives_per_group, PixelDim)
            y_pos_grouped = PositivesGrouped.reshape(Groups, positives_per_group, PixelDim)

            y_neg_grouped = x_grouped.detach()
            neg_log_weights_grouped = None
            if UnconditionalGrouped is not None:
                unc_grouped = UnconditionalGrouped.reshape(Groups, unconditional_per_group, PixelDim)
                y_neg_grouped = torch.cat([y_neg_grouped, unc_grouped], dim=1)
                gen_zeros = torch.zeros(Groups, negatives_per_group, device=Device, dtype=torch.float32)
                safe_weights = AlphaWeights.clone()
                zero_mask = safe_weights == 0.0
                safe_weights[zero_mask] = 1.0
                unc_log_w = torch.log(safe_weights).unsqueeze(1).expand(Groups, unconditional_per_group)
                if zero_mask.any():
                    unc_log_w = unc_log_w.clone()
                    unc_log_w[zero_mask] = torch.finfo(torch.float32).min
                neg_log_weights_grouped = torch.cat([gen_zeros, unc_log_w], dim=1)

            Loss, DriftStats = _batched_drifting_stopgrad_loss(
                x_grouped, y_pos_grouped, y_neg_grouped,
                neg_log_weights_grouped, _DriftLossCfg, negatives_per_group,
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
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=clip_grad_norm)
        G_opt.step()

        with torch.autograd.profiler.record_function('Gema'):
            EmaBeta = 0.5 ** (batch_size / max(CurEmaNimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, EmaBeta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        LastAlphaMean = AlphaGrouped.mean().detach()
        LastDriftNorm = DriftStats['mean_drift_norm']
        training_stats.report('Loss/G/loss', Loss.detach())
        training_stats.report('Loss/G/total', Loss.detach())
        training_stats.report('Loss/drift_norm', DriftStats['mean_drift_norm'])
        training_stats.report('Loss/drift_pos_norm', DriftStats['mean_drift_pos_norm'])
        training_stats.report('Loss/drift_neg_norm', DriftStats['mean_drift_neg_norm'])
        if 'feature_scale' in DriftStats:
            training_stats.report('Loss/feature_scale', DriftStats['feature_scale'])
        if 'drift_rms_scale' in DriftStats:
            training_stats.report('Loss/drift_rms_scale', DriftStats['drift_rms_scale'])
        training_stats.report('Progress/alpha_mean', LastAlphaMean)
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
            Fields += [f"alpha {float(LastAlphaMean):.3f}"]
            Fields += [f"drift {float(LastDriftNorm):.4f}"]
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
            # Build provenance metadata for reproducibility.
            _provenance = dict(
                drift_temperature=drift_temperature,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                negatives_per_group=negatives_per_group,
                positives_per_group=positives_per_group,
                unconditional_per_group=unconditional_per_group,
                queue_capacity_per_class=queue_capacity_per_class,
                queue_capacity_global=queue_capacity_global,
            )
            if drift_config is not None:
                _provenance['backbone'] = str(getattr(drift_config, 'backbone', 'r3gan_conv'))
                _provenance['use_feature_loss'] = bool(getattr(drift_config, 'use_feature_loss', False))
                _provenance['feature_encoder'] = str(getattr(drift_config, 'feature_encoder', 'none'))
            try:
                import subprocess
                _git_out = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True, text=True, cwd=os.path.dirname(__file__),
                )
                if _git_out.returncode == 0:
                    _provenance['git_commit'] = _git_out.stdout.strip()
            except Exception:
                pass

            SnapshotData = dict(
                G=G,
                D=None,
                G_ema=G_ema,
                training_set_kwargs=dict(training_set_kwargs),
                cur_nimg=CurNimg,
                trainer='drift',
                queue_state=Queue.state_dict(),
                G_opt_state=remap_optimizer_state_dict(G_opt.state_dict(), 'cpu'),
                provenance=_provenance,
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
    Images = Images[:local_queue_push_batch].to(device, non_blocking=True).to(torch.float32) / 127.5 - 1
    Labels = Labels[:local_queue_push_batch].to(device, non_blocking=True)
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


def _batched_drifting_stopgrad_loss(
    x_grouped, y_pos_grouped, y_neg_grouped,
    neg_log_weights_grouped, config, generated_negative_count,
    *, scale_temperature_by_sqrt_dim=True, normalize_features=True,
    normalize_drifts=True, normalization_eps=1e-8,
):
    """Vectorized drift loss over all groups simultaneously.

    Args:
        x_grouped: [G, N_gen, D] generated samples (flattened pixels)
        y_pos_grouped: [G, N_pos, D] positive samples
        y_neg_grouped: [G, N_neg, D] negative samples (gen detached + unconditional)
        neg_log_weights_grouped: [G, N_neg] or None
        config: DriftingLossConfig
        generated_negative_count: int, number of generated negatives in y_neg
        scale_temperature_by_sqrt_dim: scale temperature by sqrt(D) as in reference impl
        normalize_features: normalize features by mean pairwise distance before drift
        normalize_drifts: normalize drift vectors by RMS magnitude
        normalization_eps: epsilon for normalization clamping
    Returns:
        loss: scalar tensor (mean over groups)
        stats: dict with detached tensor stats
    """
    fc = config.drift_field

    # --- P0-B: Feature normalization (reference: drift_loss.py _normalize_features) ---
    # Scale all vectors by the mean pairwise distance so that temperature has
    # consistent behaviour regardless of raw pixel magnitude / dimensionality.
    feature_scale = None
    if normalize_features:
        # Mean pairwise distance between generated and positives, detached.
        # Flatten groups for distance computation: [G*N_gen, D] vs [G*N_pos, D]
        # Per-group is more accurate but expensive; use global mean for simplicity.
        with torch.no_grad():
            # Sample distances from x to y_pos across all groups
            dists_sample = torch.cdist(x_grouped, y_pos_grouped)  # [G, N_gen, N_pos]
            mean_dist = dists_sample.mean()
            dim = float(x_grouped.shape[-1])
            feature_scale = torch.clamp(mean_dist / sqrt(dim), min=normalization_eps)
        x_grouped = x_grouped / feature_scale
        y_pos_grouped = y_pos_grouped / feature_scale
        y_neg_grouped = y_neg_grouped / feature_scale

    # Batched pairwise distances: [G, N_gen, N_pos] and [G, N_gen, N_neg]
    dist_pos = torch.cdist(x_grouped, y_pos_grouped)
    dist_neg = torch.cdist(x_grouped, y_neg_grouped)

    # Self-mask: generated negatives are detached copies of x, mask diagonal
    if fc.mask_self_negatives and generated_negative_count > 0:
        diag_count = min(x_grouped.shape[1], generated_negative_count, y_neg_grouped.shape[1])
        if diag_count > 0:
            diagonal = torch.arange(diag_count, device=x_grouped.device)
            dist_neg = dist_neg.clone()
            dist_neg[:, diagonal, diagonal] = dist_neg[:, diagonal, diagonal] + fc.self_mask_value

    # --- P0-A: Temperature sqrt(D) scaling (reference: drift_loss.py:418-419) ---
    effective_temperature = fc.temperature
    if scale_temperature_by_sqrt_dim:
        effective_temperature = fc.temperature * sqrt(float(x_grouped.shape[-1]))

    logit_pos = -(dist_pos / effective_temperature)
    logit_neg = -(dist_neg / effective_temperature)
    if neg_log_weights_grouped is not None:
        logit_neg = logit_neg + neg_log_weights_grouped.unsqueeze(1)  # [G, 1, N_neg]

    logits = torch.cat([logit_pos, logit_neg], dim=2)  # [G, N_gen, N_pos+N_neg]
    row_affinity = torch.softmax(logits, dim=-1)

    if fc.normalize_over_x:
        col_affinity = torch.softmax(logits, dim=-2)
        affinity = torch.sqrt(torch.clamp(row_affinity * col_affinity, min=fc.eps))
    else:
        affinity = row_affinity

    n_pos = y_pos_grouped.shape[1]
    affinity_pos = affinity[:, :, :n_pos]
    affinity_neg = affinity[:, :, n_pos:]

    weight_pos = affinity_pos * affinity_neg.sum(dim=2, keepdim=True)
    weight_neg = affinity_neg * affinity_pos.sum(dim=2, keepdim=True)

    drift_pos = weight_pos @ y_pos_grouped  # [G, N_gen, D]
    drift_neg = weight_neg @ y_neg_grouped  # [G, N_gen, D]
    drift = (config.attraction_scale * drift_pos) - (config.repulsion_scale * drift_neg)

    # --- P0-C: Drift normalization (reference: drift_loss.py:550-575) ---
    # Normalize drift by its RMS magnitude so the loss/gradient scale is stable
    # regardless of how small or large the raw drift vectors are.
    drift_rms_scale = None
    if normalize_drifts:
        with torch.no_grad():
            dim = float(drift.shape[-1])
            drift_rms_scale = torch.sqrt(torch.mean(drift.pow(2).sum(dim=-1) / dim))
            drift_rms_scale = torch.clamp(drift_rms_scale, min=normalization_eps)
        drift = drift / drift_rms_scale

    target = x_grouped + drift
    if config.stopgrad_target:
        target = target.detach()
    # Per-group MSE, then mean over groups
    per_group_loss = (x_grouped - target).pow(2).mean(dim=(1, 2))  # [G]
    loss = per_group_loss.mean()

    # Un-normalize drift norms for logging (report in original scale)
    raw_drift = drift * drift_rms_scale if drift_rms_scale is not None else drift
    stats = {
        'mean_drift_norm': raw_drift.norm(dim=-1).mean().detach(),
        'mean_drift_pos_norm': drift_pos.norm(dim=-1).mean().detach(),
        'mean_drift_neg_norm': drift_neg.norm(dim=-1).mean().detach(),
    }
    if feature_scale is not None:
        stats['feature_scale'] = feature_scale.detach()
    if drift_rms_scale is not None:
        stats['drift_rms_scale'] = drift_rms_scale.detach()
    stats['effective_temperature'] = effective_temperature
    return loss, stats
