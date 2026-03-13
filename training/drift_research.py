from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import psutil
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import dnnlib
import legacy
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix

from training.drift_diagnostics import collect_generator_diagnostics
from training.drift_reference import (
    ClassConditionalSampleQueue,
    DriftFieldConfig,
    DriftingLossConfig,
    FeatureDriftingConfig,
    FeatureVectorizationConfig,
    GroupedDriftStepConfig,
    GroupedSamplingConfig,
    QueueConfig,
    RealBatchProvider,
    RealBatchProviderConfig,
    append_jsonl,
    attach_loss_scale_metrics,
    build_feature_extractor,
    build_lr_scheduler,
    build_periodic_eval_state,
    build_queue_warmup_report,
    build_real_provider_sanity_report,
    cfg_alpha_to_unconditional_weight,
    codebase_fingerprint,
    ensure_queue_has_labels,
    environment_fingerprint,
    environment_snapshot,
    grouped_drift_training_step,
    load_training_checkpoint,
    maybe_compile_callable,
    metric_jsonl_entry,
    payload_sha256,
    prime_queue,
    run_periodic_eval,
    sample_alpha,
    sample_grouped_real_batches,
    save_training_checkpoint,
    write_json,
)
from training.training_loop import cosine_decay_with_warmup, save_image_grid, setup_snapshot_image_grid


def training_loop(
    run_dir='.',
    training_set_kwargs={},
    data_loader_kwargs={},
    G_kwargs={},
    G_opt_kwargs={},
    lr_scheduler=None,
    beta2_scheduler=None,
    metrics=[],
    random_seed=0,
    num_gpus=1,
    rank=0,
    batch_size=4,
    total_kimg=25000,
    kimg_per_tick=4,
    image_snapshot_ticks=50,
    network_snapshot_ticks=50,
    resume_pkl=None,
    cudnn_benchmark=True,
    abort_fn=None,
    progress_fn=None,
    drift_config=None,
    **_unused_kwargs,
):
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * max(num_gpus, 1) + rank)
    torch.manual_seed(random_seed * max(num_gpus, 1) + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    drift = _build_drift_args(
        drift_config=drift_config,
        batch_size=batch_size,
        num_gpus=num_gpus,
        run_dir=run_dir,
    )

    if batch_size % num_gpus != 0:
        raise ValueError('batch_size must be divisible by num_gpus')
    local_batch_size = batch_size // num_gpus
    if local_batch_size % drift.negatives_per_group != 0:
        raise ValueError('batch_size / num_gpus must be divisible by negatives_per_group')
    local_groups = local_batch_size // drift.negatives_per_group

    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    if not training_set.has_labels:
        raise ValueError('drift trainer requires a labeled dataset')
    image_channels = int(training_set.image_shape[0])
    num_classes = int(training_set.label_dim)
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set,
        rank=rank,
        num_replicas=num_gpus,
        seed=random_seed,
    )
    loader_batch_size = max(local_batch_size, max(1, int(drift.real_loader_batch_size)), max(1, int(drift.queue_push_batch // max(num_gpus, 1))))
    loader_kwargs = dict(data_loader_kwargs)
    if str(drift.real_batch_source) == 'dataset_loader':
        loader_kwargs['num_workers'] = 0
        loader_kwargs.pop('prefetch_factor', None)
        loader_kwargs['persistent_workers'] = False
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set,
            sampler=training_set_sampler,
            batch_size=loader_batch_size,
            **loader_kwargs,
        )
    )

    provider = _build_real_batch_provider(
        drift=drift,
        training_set_iterator=training_set_iterator,
        training_set_kwargs=training_set_kwargs,
        image_channels=image_channels,
        image_size=int(training_set.resolution),
        num_classes=num_classes,
        device=device,
        rank=rank,
        num_gpus=num_gpus,
    )
    provider_manifest_fingerprint = getattr(provider, 'manifest_fingerprint', None)
    real_provider_sanity_report = None
    if drift.real_sanity_sample_batches > 0 and hasattr(provider, 'next_batch'):
        real_provider_sanity_report = build_real_provider_sanity_report(
            provider=provider,
            sample_batches=int(drift.real_sanity_sample_batches),
            num_classes=num_classes,
            device=device,
        )

    if rank == 0:
        print('Constructing drift generator...')
    g_kwargs = copy.deepcopy(G_kwargs)
    if drift.backbone == 'dit_like':
        g_kwargs['ImageChannels'] = image_channels
    g = dnnlib.util.construct_class_by_name(
        **g_kwargs,
        c_dim=num_classes,
        img_resolution=int(training_set.resolution),
    ).train().to(device)
    g.requires_grad_(True)
    g_ema = copy.deepcopy(g).eval().requires_grad_(False)
    optimizer = dnnlib.util.construct_class_by_name(params=g.parameters(), **G_opt_kwargs)
    train_generator = g
    if num_gpus > 1:
        train_generator = DDP(g, device_ids=[rank], broadcast_buffers=False)

    drift_args = _build_reference_args_namespace(
        drift=drift,
        run_dir=run_dir,
        image_channels=image_channels,
        image_size=int(training_set.resolution),
        num_classes=num_classes,
        provider_manifest_fingerprint=provider_manifest_fingerprint,
    )
    compile_info = _maybe_compile_generator_forward(train_generator, drift_args, device)

    queue = ClassConditionalSampleQueue(
        QueueConfig(
            num_classes=num_classes,
            per_class_capacity=int(drift.queue_capacity_per_class),
            global_capacity=int(drift.queue_capacity_global),
            store_device='cpu',
            strict_without_replacement=bool(drift.queue_strict_without_replacement),
        )
    )
    sampling_config = GroupedSamplingConfig(
        positives_per_group=int(drift.positives_per_group),
        unconditional_per_group=int(drift.unconditional_per_group),
    )
    step_config = GroupedDriftStepConfig(
        loss_config=DriftingLossConfig(
            drift_field=DriftFieldConfig(
                temperature=float(drift.drift_temperature),
                normalize_over_x=True,
                mask_self_negatives=True,
            ),
            attraction_scale=1.0,
            repulsion_scale=1.0,
            stopgrad_target=True,
        ),
        feature_config=_build_feature_config(drift),
        drift_temperatures=tuple(float(value) for value in drift.drift_temperatures),
        drift_temperature_reduction=str(drift.drift_temperature_reduction),
        clip_grad_norm=float(drift.clip_grad_norm),
        run_optimizer_step=True,
    )
    scheduler = None
    if lr_scheduler is None:
        scheduler = build_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=str(drift.scheduler),
            total_steps=max(1, _total_steps(total_kimg=total_kimg, batch_size=batch_size)),
            warmup_steps=int(drift.warmup_steps),
        )
    feature_extractor = None
    if drift.use_feature_loss:
        feature_extractor = build_feature_extractor(args=drift_args, device=device).eval()
        for parameter in feature_extractor.parameters():
            parameter.requires_grad = False

    resolved_config_hash = payload_sha256(_resolved_config_payload(drift=drift, training_set_kwargs=training_set_kwargs, image_channels=image_channels))
    resume_payload = None
    queue_resumed_from_checkpoint = False
    start_step = 0
    if resume_pkl is not None:
        resume_payload = _load_drift_resume_payload(
            resume_path=resume_pkl,
            model=g,
            optimizer=optimizer,
            scheduler=scheduler,
            drift=drift,
            resolved_config_hash=resolved_config_hash,
            device=device,
        )
        if resume_payload is not None:
            start_step = int(resume_payload.get('step', 0))
            queue_state = resume_payload.get('queue_state')
            if isinstance(queue_state, dict):
                queue.load_state_dict(queue_state)
                queue_resumed_from_checkpoint = True
            if 'ema_state_dict' in resume_payload and isinstance(resume_payload['ema_state_dict'], dict):
                g_ema.load_state_dict(resume_payload['ema_state_dict'], strict=False)

    if rank == 0 and provider is not None:
        print('Priming drift queue...')
    queue_warmup_report = None
    if not queue_resumed_from_checkpoint:
        prime_queue(
            queue=queue,
            provider=provider,
            num_classes=num_classes,
            sample_count=int(drift.queue_prime_samples),
            warmup_mode=str(drift.queue_warmup_mode),
            warmup_min_per_class=int(drift.queue_warmup_min_per_class),
            device=device,
        )
        queue_warmup_report = build_queue_warmup_report(
            queue=queue,
            warmup_mode=str(drift.queue_warmup_mode),
            samples_pushed=int(drift.queue_prime_samples),
            report_level=str(drift.queue_report_level),
        )
    else:
        queue_warmup_report = {
            'mode': 'resume_restore',
            'warmup_mode': 'resume_restore',
            'samples_pushed': 0.0,
            'global_count': float(queue.global_count()),
            'covered_classes': float(sum(1 for count in queue.class_counts() if count > 0)),
        }

    if rank == 0:
        _write_static_artifacts(run_dir=run_dir, provider_manifest_fingerprint=provider_manifest_fingerprint)

    grid_size = None
    grid_class_ids = None
    if rank == 0:
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size)
        grid_class_ids = torch.from_numpy(labels).to(device)
        if grid_class_ids.ndim == 2:
            grid_class_ids = grid_class_ids.argmax(dim=1).long()
        else:
            grid_class_ids = grid_class_ids.long()
        _save_fake_grid(
            generator=g_ema,
            drift=drift,
            grid_size=grid_size,
            class_ids=grid_class_ids,
            path=os.path.join(run_dir, 'fakes_init.png'),
            device=device,
        )

    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard

            stats_tfevents = tensorboard.SummaryWriter(run_dir)
            stats_tfevents.add_text('config/drift_args_json', json.dumps(vars(drift_args), indent=2, sort_keys=True))
            stats_tfevents.add_text('config/env_fingerprint_json', json.dumps(environment_fingerprint(), indent=2, sort_keys=True))
        except ImportError as err:
            print('Skipping tfevents export:', err)

    eval_state = None
    if drift.backbone == 'dit_like' and float(drift.eval_every_kimg) > 0:
        if drift.eval_reference_imagefolder_root is None:
            raise ValueError('--eval-reference-imagefolder-root is required when --eval-every-kimg > 0')
        eval_state = build_periodic_eval_state(args=drift_args, device=device)

    if rank == 0:
        noise_summary = torch.randn(local_batch_size, getattr(g, 'noise_channels', image_channels), int(training_set.resolution), int(training_set.resolution), device=device)
        class_summary = torch.zeros(local_batch_size, device=device, dtype=torch.long)
        alpha_summary = torch.full((local_batch_size,), float(drift.eval_alpha), device=device, dtype=torch.float32)
        style_summary = torch.zeros(local_batch_size, int(getattr(g, 'StyleTokenCount', drift.style_token_count)), device=device, dtype=torch.long)
        misc.print_module_summary(g, [noise_summary, class_summary, alpha_summary, style_summary])

    if rank == 0:
        print(f'Training drift parity backend "{drift.backbone}" for {total_kimg} kimg...')
        print()

    cur_nimg = 0 if resume_payload is None else int(resume_payload.get('generated_images_total', start_step * batch_size))
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    next_eval_generated_images = _initial_eval_threshold(
        start_generated_images=cur_nimg,
        interval_generated_images=_eval_interval_generated_images(drift),
    )
    logs = []
    periodic_evals = []
    latest_checkpoint_path = None

    for step in range(start_step, _total_steps(total_kimg=total_kimg, batch_size=batch_size)):
        step_start_time = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        if lr_scheduler is not None:
            cur_lr = cosine_decay_with_warmup(cur_nimg, **lr_scheduler)
            for group in optimizer.param_groups:
                group['lr'] = float(cur_lr)
        if beta2_scheduler is not None:
            cur_beta2 = cosine_decay_with_warmup(cur_nimg, **beta2_scheduler)
            for group in optimizer.param_groups:
                group['betas'] = (float(group['betas'][0]), float(cur_beta2))

        class_labels = torch.randint(0, num_classes, (local_groups,), device=device)
        alpha = sample_alpha(
            groups=int(local_groups),
            device=device,
            alpha_fixed=drift.alpha_fixed,
            alpha_min=float(drift.alpha_min),
            alpha_max=float(drift.alpha_max),
            alpha_dist=str(drift.alpha_dist),
            alpha_power=float(drift.alpha_power),
            alpha_point=float(drift.alpha_point),
            alpha_point_prob=float(drift.alpha_point_prob),
        )
        if drift.backbone == 'dit_like':
            noise_grouped = torch.randn(
                local_groups,
                int(drift.negatives_per_group),
                image_channels,
                int(training_set.resolution),
                int(training_set.resolution),
                device=device,
            )
            if int(drift.style_token_count) > 0:
                style_indices = torch.randint(
                    0,
                    int(drift.style_vocab_size),
                    (local_groups, int(drift.negatives_per_group), int(drift.style_token_count)),
                    device=device,
                )
            else:
                style_indices = torch.zeros(
                    local_groups,
                    int(drift.negatives_per_group),
                    0,
                    device=device,
                    dtype=torch.long,
                )
            should_refill = str(drift.queue_refill_policy) == 'per_step' or (
                str(drift.queue_refill_policy) == 'every_n_steps'
                and ((step - start_step) % max(1, int(drift.queue_refill_every)) == 0)
            )
            if should_refill:
                refill_images, refill_labels = _sample_real_batch(provider=provider, count=int(drift.queue_push_batch), device=device)
                queue.push(refill_images, refill_labels)
            backfilled = ensure_queue_has_labels(
                queue=queue,
                class_labels=class_labels,
                provider=provider,
                required_count=int(drift.positives_per_group if drift.queue_strict_without_replacement else 1),
                device=device,
            )
            positives_grouped, unconditional_grouped = sample_grouped_real_batches(
                queue=queue,
                class_labels=class_labels,
                config=sampling_config,
                device=device,
            )
            unconditional_weight_grouped = torch.tensor(
                [
                    cfg_alpha_to_unconditional_weight(
                        alpha=float(alpha[g_index].item()),
                        n_generated_negatives=int(drift.negatives_per_group),
                        n_unconditional_negatives=int(drift.unconditional_per_group),
                    )
                    for g_index in range(local_groups)
                ],
                device=device,
                dtype=torch.float32,
            )
            stats = grouped_drift_training_step(
                generator=train_generator,
                optimizer=optimizer,
                noise_grouped=noise_grouped,
                class_labels_grouped=class_labels,
                alpha_grouped=alpha,
                positives_grouped=positives_grouped,
                style_indices_grouped=style_indices,
                unconditional_grouped=unconditional_grouped,
                unconditional_weight_grouped=unconditional_weight_grouped,
                feature_extractor=feature_extractor,
                config=step_config,
            )
            attach_loss_scale_metrics(stats)
            stats['queue_underflow_backfilled'] = float(backfilled)
        else:
            stats = _run_r3gan_conv_step(
                generator=g,
                optimizer=optimizer,
                queue=queue,
                provider=provider,
                step=step,
                local_groups=local_groups,
                num_classes=num_classes,
                drift=drift,
                image_channels=image_channels,
                image_size=int(training_set.resolution),
                class_labels=class_labels,
                alpha=alpha,
                device=device,
                step_config=step_config,
                feature_extractor=feature_extractor,
            )

        if scheduler is not None:
            scheduler.step()
        _ema_update(src=g, dst=g_ema, decay=float(drift.ema_decay))

        step_time_s = time.perf_counter() - step_start_time
        generated_images_total = int((step + 1) * batch_size)
        stats['step_time_s'] = float(step_time_s)
        stats['generated_images_per_sec'] = float(batch_size / max(step_time_s, 1e-8))
        stats['generated_images_total'] = float(generated_images_total)
        stats['generated_kimg_total'] = float(generated_images_total / 1000.0)
        stats['lr'] = float(optimizer.param_groups[0]['lr'])
        stats['beta2'] = float(optimizer.param_groups[0]['betas'][1])
        stats['peak_cuda_mem_mb'] = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == 'cuda' else 0.0
        stats['queue_global_count'] = float(queue.global_count())
        stats['queue_covered_classes'] = float(sum(1 for count in queue.class_counts() if count > 0))
        stats['provider_manifest_fingerprint'] = provider_manifest_fingerprint

        training_stats.report('Loss/G/loss', torch.as_tensor(float(stats['loss']), device=device))
        training_stats.report('Loss/drift_norm', torch.as_tensor(float(stats.get('mean_drift_norm', stats.get('drift_norm', 0.0))), device=device))
        training_stats.report('Progress/alpha_mean', torch.as_tensor(float(stats.get('alpha_mean', alpha.mean().item())), device=device))
        training_stats.report('Progress/lr', torch.as_tensor(float(stats['lr']), device=device))
        training_stats.report('Progress/beta2', torch.as_tensor(float(stats['beta2']), device=device))
        training_stats.report('Progress/queue_global_count', torch.as_tensor(float(queue.global_count()), device=device))

        should_log_step = (step == start_step) or ((step + 1) % max(1, int(kimg_per_tick)) == 0)
        if rank == 0 and should_log_step:
            log_entry = {'step': float(step + 1), **stats}
            logs.append(log_entry)

        cur_nimg = generated_images_total
        done = cur_nimg >= total_kimg * 1000
        tick_boundary = done or (cur_nimg >= tick_start_nimg + kimg_per_tick * 1000)
        if not tick_boundary:
            continue

        tick_end_time = time.time()
        diagnostic_stats = None
        if rank == 0:
            diagnostic_stats = _collect_training_diagnostics(
                generator=g_ema,
                drift=drift,
                device=device,
                num_classes=num_classes,
            )
            _merge_diagnostic_stats(stats=stats, diagnostics=diagnostic_stats)
            _report_diagnostic_stats(diagnostics=diagnostic_stats)
        training_stats.report0('Progress/tick', cur_tick)
        training_stats.report0('Progress/kimg', cur_nimg / 1e3)
        training_stats.report0('Timing/total_sec', tick_end_time - start_time)
        training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time)
        training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / max(cur_nimg - tick_start_nimg, 1) * 1e3)
        training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30)
        training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0.0)

        if rank == 0:
            fields = [
                f'tick {cur_tick:<5d}',
                f'kimg {cur_nimg / 1e3:<8.1f}',
                f'sec/tick {tick_end_time - tick_start_time:<7.1f}',
                f'sec/kimg {(tick_end_time - tick_start_time) / max(cur_nimg - tick_start_nimg, 1) * 1e3:<7.2f}',
                f'loss {float(stats["loss"]):.4f}',
                f'drift {float(stats.get("mean_drift_norm", stats.get("drift_norm", 0.0))):.4f}',
                f'alpha {float(stats.get("alpha_mean", alpha.mean().item())):.3f}',
                f'queue {queue.global_count():<6d}',
            ]
            if diagnostic_stats is not None:
                if diagnostic_stats.get('pairwise_l2_mean') is not None:
                    fields.append(f'pairL2 {float(diagnostic_stats["pairwise_l2_mean"]):.2f}')
                if diagnostic_stats.get('diff_class_l2') is not None:
                    fields.append(f'clsL2 {float(diagnostic_stats["diff_class_l2"]):.2f}')
                if diagnostic_stats.get('diff_style_l2') is not None:
                    fields.append(f'styL2 {float(diagnostic_stats["diff_style_l2"]):.2f}')
            print(' '.join(fields))

        save_images = rank == 0 and (done or (image_snapshot_ticks is not None and cur_tick % image_snapshot_ticks == 0))
        save_network = done or (network_snapshot_ticks is not None and cur_tick % network_snapshot_ticks == 0)
        snapshot_path = None
        if save_images and grid_size is not None and grid_class_ids is not None:
            snapshot_image_path = os.path.join(run_dir, f'fakes{cur_nimg // 1000:09d}.png')
            _save_fake_grid(
                generator=g_ema,
                drift=drift,
                grid_size=grid_size,
                class_ids=grid_class_ids,
                path=snapshot_image_path,
                device=device,
            )
        if save_network:
            snapshot_path = _save_network_snapshot(
                run_dir=run_dir,
                generator=g,
                generator_ema=g_ema,
                training_set_kwargs=training_set_kwargs,
                cur_nimg=cur_nimg,
                queue_state=queue.state_dict(),
            )
            latest_checkpoint_path = _save_research_checkpoint(
                run_dir=run_dir,
                drift=drift,
                generator=g,
                generator_ema=g_ema,
                optimizer=optimizer,
                scheduler=scheduler,
                queue=queue,
                step=step + 1,
                generated_images_total=cur_nimg,
                resolved_config_hash=resolved_config_hash,
                provider_manifest_fingerprint=provider_manifest_fingerprint,
            )
            if rank == 0 and diagnostic_stats is not None:
                write_json(
                    _snapshot_diagnostics_path(snapshot_path),
                    {
                        'generated_images_total': int(cur_nimg),
                        'generated_kimg_total': float(cur_nimg / 1000.0),
                        'snapshot_path': str(snapshot_path),
                        **diagnostic_stats,
                    },
                )

        if eval_state is not None and rank == 0:
            interval_generated_images = _eval_interval_generated_images(drift)
            while interval_generated_images > 0 and cur_nimg >= next_eval_generated_images:
                eval_entry = run_periodic_eval(
                    args=drift_args,
                    generator=g_ema,
                    model_config=g_ema.ModelConfig,
                    device=device,
                    eval_state=eval_state,
                    step=int(step + 1),
                    generated_images_total=int(next_eval_generated_images),
                )
                periodic_evals.append(eval_entry)
                append_jsonl(path=Path(run_dir) / 'periodic_eval_history.jsonl', payload=eval_entry)
                _write_periodic_metric_jsonls(
                    run_dir=run_dir,
                    metrics=metrics,
                    eval_entry=eval_entry,
                )
                for metric_name in ('fid', 'fid50k_full', 'fid50k_fullb', 'inception_score_mean'):
                    if metric_name in eval_entry:
                        append_jsonl(
                            path=Path(run_dir) / 'metric-history.jsonl',
                            payload=metric_jsonl_entry(eval_entry=eval_entry, metric_name=metric_name),
                        )
                stats_metrics.update({k: v for k, v in eval_entry.items() if isinstance(v, (int, float))})
                next_eval_generated_images += interval_generated_images

        stats_collector.update()
        stats_dict = stats_collector.as_dict()
        timestamp = time.time()
        if rank == 0 and stats_jsonl is not None:
            stats_jsonl.write(json.dumps(dict(stats_dict, timestamp=timestamp)) + '\n')
            stats_jsonl.flush()
        if rank == 0 and stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                if isinstance(value, (int, float)):
                    stats_tfevents.add_scalar(f'eval/{name}', float(value), global_step=global_step, walltime=walltime)
            stats_tfevents.flush()

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        if (not done) and abort_fn is not None and abort_fn():
            done = True
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    if rank == 0:
        summary = {
            'trainer': 'drift',
            'drift_backbone': str(drift.backbone),
            'resolved_config_hash': resolved_config_hash,
            'provider_manifest_fingerprint': provider_manifest_fingerprint,
            'queue_warmup_report': queue_warmup_report,
            'compile_info': compile_info,
            'real_provider_sanity_report': real_provider_sanity_report,
            'resume_from': resume_pkl,
            'generated_images_total': int(cur_nimg),
            'generated_kimg_total': float(cur_nimg / 1000.0),
            'logs': logs[-200:],
            'periodic_evals': periodic_evals,
            'latest_checkpoint_path': latest_checkpoint_path,
            'maintenance_time_s': float(maintenance_time),
        }
        write_json(Path(run_dir) / 'drift_summary.json', summary)


def _build_drift_args(*, drift_config, batch_size, num_gpus, run_dir):
    drift = dict(
        backbone='dit_like',
        negatives_per_group=4,
        positives_per_group=4,
        unconditional_per_group=2,
        alpha_fixed=None,
        alpha_min=1.0,
        alpha_max=4.0,
        alpha_dist='uniform',
        alpha_power=3.0,
        alpha_point=1.0,
        alpha_point_prob=0.5,
        drift_temperature=0.05,
        drift_temperatures=[],
        drift_temperature_reduction='sum',
        learning_rate=2e-4,
        adam_beta1=0.0,
        adam_beta2=0.0,
        weight_decay=0.0,
        scheduler='none',
        warmup_steps=0,
        clip_grad_norm=2.0,
        ema_decay=0.999,
        compile_generator=False,
        compile_backend='inductor',
        compile_mode='reduce-overhead',
        compile_dynamic=False,
        compile_fullgraph=False,
        compile_fail_action='warn',
        use_feature_loss=False,
        feature_encoder='tiny',
        convnext_weights='none',
        convnextv2_weights='none',
        mae_encoder_path=None,
        mae_encoder_arch='resnet_unet',
        mae_input_patchify_size=1,
        feature_base_channels=16,
        feature_stages=3,
        feature_temperatures=[0.02, 0.05, 0.2],
        feature_temperature_aggregation='sum_drifts_then_mse',
        feature_loss_term_reduction='sum',
        feature_selected_stages=[],
        include_patch4_stats=False,
        include_input_x2_mean=False,
        disable_shared_location_normalization=False,
        disable_feature_temperature_sqrt_scaling=False,
        feature_include_raw_drift_loss=False,
        feature_raw_drift_loss_weight=1.0,
        feature_compile_drift_kernel=False,
        feature_compile_backend='inductor',
        feature_compile_mode='reduce-overhead',
        feature_compile_dynamic=False,
        feature_compile_fullgraph=False,
        feature_compile_fail_action='warn',
        queue_capacity_per_class=256,
        queue_capacity_global=4096,
        queue_push_batch=max(32, batch_size // max(num_gpus, 1)),
        queue_warmup_batches=4,
        queue_prime_samples=200,
        queue_warmup_mode='random',
        queue_warmup_min_per_class=1,
        queue_strict_without_replacement=False,
        queue_refill_policy='per_step',
        queue_refill_every=1,
        queue_report_level='basic',
        real_batch_source='dataset_loader',
        real_dataset_size=4096,
        real_loader_batch_size=max(32, batch_size // max(num_gpus, 1)),
        real_num_workers=0,
        disable_real_shuffle=False,
        real_pin_memory=False,
        real_persistent_workers=False,
        real_prefetch_factor=0,
        real_sanity_sample_batches=0,
        real_imagefolder_root=None,
        real_webdataset_urls=None,
        real_tensor_file_path=None,
        real_tensor_shards_manifest_path=None,
        real_transform_resize=None,
        disable_real_center_crop=False,
        real_horizontal_flip=False,
        real_transform_normalize=False,
        resume_model_only=False,
        resume_reset_scheduler=False,
        resume_reset_optimizer_lr=False,
        allow_resume_config_mismatch=False,
        save_every=0,
        checkpoint_dir=None,
        keep_last_k_checkpoints=0,
        eval_every_kimg=0.0,
        eval_reference_imagefolder_root=None,
        eval_reference_stats_path=None,
        eval_samples=50000,
        eval_sample_batch_size=128,
        eval_batch_size=128,
        eval_num_workers=0,
        eval_inception_weights='pretrained',
        eval_postprocess_mode='clamp_0_1',
        eval_alpha=1.0,
        patch_size=4,
        hidden_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        ffn_inner_dim=None,
        register_tokens=16,
        style_vocab_size=1,
        style_token_count=0,
        alpha_hidden_dim=128,
        norm_type='layernorm',
        use_qk_norm=False,
        use_rope=False,
        alpha_embedding_type='mlp',
        qk_norm_mode='auto',
        rope_mode='auto',
        disable_patch_positional_embedding=False,
        disable_rmsnorm_affine=False,
    )
    if drift_config is not None:
        drift.update(dict(drift_config))
    drift['output_dir'] = run_dir
    return SimpleNamespace(**drift)


def _build_reference_args_namespace(*, drift, run_dir, image_channels, image_size, num_classes, provider_manifest_fingerprint):
    return SimpleNamespace(
        output_dir=run_dir,
        checkpoint_path=str(Path(run_dir) / 'checkpoint.pt'),
        checkpoint_dir=str(Path(run_dir) / 'checkpoints') if drift.checkpoint_dir is None else str(drift.checkpoint_dir),
        keep_last_k_checkpoints=int(drift.keep_last_k_checkpoints),
        save_every=int(drift.save_every),
        groups=1,
        negatives_per_group=int(drift.negatives_per_group),
        positives_per_group=int(drift.positives_per_group),
        unconditional_per_group=int(drift.unconditional_per_group),
        alpha_fixed=drift.alpha_fixed,
        alpha_min=float(drift.alpha_min),
        alpha_max=float(drift.alpha_max),
        alpha_dist=str(drift.alpha_dist),
        alpha_power=float(drift.alpha_power),
        alpha_point=float(drift.alpha_point),
        alpha_point_prob=float(drift.alpha_point_prob),
        channels=int(image_channels),
        image_size=int(image_size),
        num_classes=int(num_classes),
        patch_size=int(drift.patch_size),
        hidden_dim=int(drift.hidden_dim),
        depth=int(drift.depth),
        num_heads=int(drift.num_heads),
        mlp_ratio=float(drift.mlp_ratio),
        ffn_inner_dim=drift.ffn_inner_dim,
        register_tokens=int(drift.register_tokens),
        norm_type=str(drift.norm_type),
        use_qk_norm=bool(drift.use_qk_norm),
        use_rope=bool(drift.use_rope),
        alpha_embedding_type=str(drift.alpha_embedding_type),
        qk_norm_mode=str(drift.qk_norm_mode),
        rope_mode=str(drift.rope_mode),
        disable_patch_positional_embedding=bool(drift.disable_patch_positional_embedding),
        disable_rmsnorm_affine=bool(drift.disable_rmsnorm_affine),
        feature_encoder=str(drift.feature_encoder),
        convnext_weights=str(drift.convnext_weights),
        convnextv2_weights=str(drift.convnextv2_weights),
        mae_encoder_path=drift.mae_encoder_path,
        mae_encoder_arch=str(drift.mae_encoder_arch),
        mae_input_patchify_size=int(drift.mae_input_patchify_size),
        feature_base_channels=int(drift.feature_base_channels),
        feature_stages=int(drift.feature_stages),
        feature_temperatures=list(drift.feature_temperatures),
        feature_temperature_aggregation=str(drift.feature_temperature_aggregation),
        feature_loss_term_reduction=str(drift.feature_loss_term_reduction),
        feature_selected_stages=list(drift.feature_selected_stages),
        include_patch4_stats=bool(drift.include_patch4_stats),
        include_input_x2_mean=bool(drift.include_input_x2_mean),
        disable_shared_location_normalization=bool(drift.disable_shared_location_normalization),
        disable_feature_temperature_sqrt_scaling=bool(drift.disable_feature_temperature_sqrt_scaling),
        feature_include_raw_drift_loss=bool(drift.feature_include_raw_drift_loss),
        feature_raw_drift_loss_weight=float(drift.feature_raw_drift_loss_weight),
        feature_compile_drift_kernel=bool(drift.feature_compile_drift_kernel),
        feature_compile_backend=str(drift.feature_compile_backend),
        feature_compile_mode=str(drift.feature_compile_mode),
        feature_compile_dynamic=bool(drift.feature_compile_dynamic),
        feature_compile_fullgraph=bool(drift.feature_compile_fullgraph),
        feature_compile_fail_action=str(drift.feature_compile_fail_action),
        compile_generator=bool(drift.compile_generator),
        compile_backend=str(drift.compile_backend),
        compile_mode=str(drift.compile_mode),
        compile_dynamic=bool(drift.compile_dynamic),
        compile_fullgraph=bool(drift.compile_fullgraph),
        compile_fail_action=str(drift.compile_fail_action),
        eval_every_kimg=float(drift.eval_every_kimg),
        eval_reference_imagefolder_root=drift.eval_reference_imagefolder_root,
        eval_reference_stats_path=drift.eval_reference_stats_path,
        eval_samples=int(drift.eval_samples),
        eval_sample_batch_size=int(drift.eval_sample_batch_size),
        eval_batch_size=int(drift.eval_batch_size),
        eval_num_workers=int(drift.eval_num_workers),
        eval_inception_weights=str(drift.eval_inception_weights),
        eval_alpha=float(drift.eval_alpha),
        eval_postprocess_mode=str(drift.eval_postprocess_mode),
        provider_manifest_fingerprint=provider_manifest_fingerprint,
    )


def _build_feature_config(drift):
    if not drift.use_feature_loss:
        return None
    return FeatureDriftingConfig(
        temperatures=tuple(float(v) for v in drift.feature_temperatures),
        vectorization=FeatureVectorizationConfig(
            include_per_location=True,
            include_global_stats=True,
            include_patch2_stats=True,
            include_patch4_stats=bool(drift.include_patch4_stats),
            include_input_x2_mean=bool(drift.include_input_x2_mean),
            selected_stages=None if len(drift.feature_selected_stages) == 0 else tuple(int(v) for v in drift.feature_selected_stages),
        ),
        normalize_features=True,
        normalize_drifts=True,
        temperature_aggregation=str(drift.feature_temperature_aggregation),
        loss_term_reduction=str(drift.feature_loss_term_reduction),
        scale_temperature_by_sqrt_channels=not bool(drift.disable_feature_temperature_sqrt_scaling),
        detach_positive_features=True,
        detach_negative_features=True,
        share_location_normalization=not bool(drift.disable_shared_location_normalization),
        include_raw_drift_loss=bool(drift.feature_include_raw_drift_loss),
        raw_drift_loss_weight=float(drift.feature_raw_drift_loss_weight),
        compile_drift_kernel=bool(drift.feature_compile_drift_kernel),
        compile_drift_backend=str(drift.feature_compile_backend),
        compile_drift_mode=str(drift.feature_compile_mode),
        compile_drift_dynamic=bool(drift.feature_compile_dynamic),
        compile_drift_fullgraph=bool(drift.feature_compile_fullgraph),
        compile_drift_fail_action=str(drift.feature_compile_fail_action),
    )


class _DatasetLoaderRealBatchProvider:
    def __init__(self, *, iterator, batch_size, training_set_kwargs, image_channels, image_size):
        self._iterator = iterator
        self._batch_size = int(batch_size)
        self._manifest_payload = {
            'source': 'dataset_loader',
            'path': training_set_kwargs.get('path'),
            'resolution': int(training_set_kwargs.get('resolution', image_size)),
            'use_labels': bool(training_set_kwargs.get('use_labels', True)),
            'xflip': bool(training_set_kwargs.get('xflip', False)),
            'image_channels': int(image_channels),
        }
        self.manifest_fingerprint = payload_sha256(self._manifest_payload)

    def next_batch(self, *, device):
        images, labels = next(self._iterator)
        images = images[: self._batch_size].detach().clone().to(device).to(torch.float32) / 127.5 - 1.0
        labels = labels[: self._batch_size].detach().clone().to(device)
        return images, _class_ids_from_labels(labels)


def _build_real_batch_provider(*, drift, training_set_iterator, training_set_kwargs, image_channels, image_size, num_classes, device, rank, num_gpus):
    if str(drift.real_batch_source) == 'dataset_loader':
        return _DatasetLoaderRealBatchProvider(
            iterator=training_set_iterator,
            batch_size=int(drift.real_loader_batch_size),
            training_set_kwargs=training_set_kwargs,
            image_channels=image_channels,
            image_size=image_size,
        )
    provider_config = RealBatchProviderConfig(
        source=str(drift.real_batch_source),
        dataset_size=int(drift.real_dataset_size),
        batch_size=int(drift.real_loader_batch_size),
        shuffle=not bool(drift.disable_real_shuffle),
        num_workers=int(drift.real_num_workers),
        pin_memory=bool(drift.real_pin_memory),
        persistent_workers=bool(drift.real_persistent_workers),
        prefetch_factor=None if int(drift.real_prefetch_factor) <= 0 else int(drift.real_prefetch_factor),
        seed=1337 + rank,
        channels=int(image_channels),
        image_size=int(image_size),
        num_classes=int(num_classes),
        imagefolder_root=drift.real_imagefolder_root,
        webdataset_urls=drift.real_webdataset_urls,
        tensor_file_path=drift.real_tensor_file_path,
        tensor_shards_manifest_path=drift.real_tensor_shards_manifest_path,
        transform_resize=drift.real_transform_resize,
        transform_center_crop=not bool(drift.disable_real_center_crop),
        transform_horizontal_flip=bool(drift.real_horizontal_flip),
        transform_normalize=bool(drift.real_transform_normalize),
        distributed_world_size=int(num_gpus),
        distributed_rank=int(rank),
    )
    return RealBatchProvider(provider_config)


def _maybe_compile_generator_forward(generator, drift_args, device):
    compiled_forward, compile_result = maybe_compile_callable(
        generator.forward,
        enabled=bool(drift_args.compile_generator),
        backend=str(drift_args.compile_backend),
        mode=str(drift_args.compile_mode),
        dynamic=bool(drift_args.compile_dynamic),
        fullgraph=bool(drift_args.compile_fullgraph),
        fail_action=str(drift_args.compile_fail_action),
        device=device,
        context='r3gan.drift.compile_generator',
    )
    if getattr(compile_result, 'enabled', False):
        generator.forward = compiled_forward
    return compile_result.to_dict() if hasattr(compile_result, 'to_dict') else {'enabled': False}


def _resolved_config_payload(*, drift, training_set_kwargs, image_channels):
    payload = dict(vars(drift))
    payload['training_set'] = {
        'path': training_set_kwargs.get('path'),
        'resolution': training_set_kwargs.get('resolution'),
        'use_labels': training_set_kwargs.get('use_labels'),
        'xflip': training_set_kwargs.get('xflip'),
        'image_channels': int(image_channels),
    }
    return payload


def _load_drift_resume_payload(*, resume_path, model, optimizer, scheduler, drift, resolved_config_hash, device):
    resume_path_obj = Path(str(resume_path))
    if resume_path_obj.suffix == '.pt':
        payload = torch.load(resume_path_obj, map_location=device)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        resume_hash = extra.get('resolved_config_hash') if isinstance(extra, dict) else None
        if (
            isinstance(resume_hash, str)
            and resume_hash != resolved_config_hash
            and not bool(drift.allow_resume_config_mismatch)
        ):
            raise ValueError(
                f'Resume config hash mismatch: expected {resolved_config_hash}, found {resume_hash}. '
                'Use --allow-resume-config-mismatch to override.'
            )
        if bool(drift.resume_model_only):
            model.load_state_dict(payload['model_state_dict'])
        else:
            payload = load_training_checkpoint(
                path=resume_path_obj,
                model=model,
                optimizer=optimizer,
                map_location=device,
                scheduler=None if bool(drift.resume_reset_scheduler) else scheduler,
            )
            if bool(drift.resume_reset_optimizer_lr):
                for group in optimizer.param_groups:
                    group['lr'] = float(drift.learning_rate)
        return payload

    with dnnlib.util.open_url(str(resume_path)) as handle:
        payload = legacy.load_network_pkl(handle)
    misc.copy_params_and_buffers(payload['G'], model, require_all=False)
    return {
        'step': int(payload.get('cur_nimg', 0) // max(1, model.img_resolution)),
        'generated_images_total': int(payload.get('cur_nimg', 0)),
        'queue_state': payload.get('queue_state'),
        'ema_state_dict': payload['G_ema'].state_dict() if payload.get('G_ema') is not None else None,
    }


def _save_network_snapshot(*, run_dir, generator, generator_ema, training_set_kwargs, cur_nimg, queue_state):
    snapshot_data = dict(
        G=copy.deepcopy(generator).eval().requires_grad_(False).cpu(),
        D=None,
        G_ema=copy.deepcopy(generator_ema).eval().requires_grad_(False).cpu(),
        training_set_kwargs=dict(training_set_kwargs),
        cur_nimg=int(cur_nimg),
        trainer='drift',
        queue_state=queue_state,
    )
    snapshot_path = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:09d}.pkl')
    with open(snapshot_path, 'wb') as handle:
        import pickle

        pickle.dump(snapshot_data, handle)
    return snapshot_path


def _save_research_checkpoint(*, run_dir, drift, generator, generator_ema, optimizer, scheduler, queue, step, generated_images_total, resolved_config_hash, provider_manifest_fingerprint):
    latest_path = Path(run_dir) / 'checkpoint.pt'
    checkpoint_dir = Path(run_dir) / 'checkpoints' if drift.checkpoint_dir is None else Path(drift.checkpoint_dir)
    extra = {
        'resolved_config_hash': resolved_config_hash,
        'provider_manifest_fingerprint': provider_manifest_fingerprint,
        'generated_images_total': int(generated_images_total),
        'drift_args': vars(drift),
    }
    save_training_checkpoint(
        path=latest_path,
        model=generator,
        optimizer=optimizer,
        step=int(step),
        extra=extra,
        queue_state=queue.state_dict(),
        scheduler=scheduler,
        ema_state_dict=generator_ema.state_dict(),
    )
    if int(drift.save_every) > 0 and step % int(drift.save_every) == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        step_path = checkpoint_dir / f'checkpoint_step_{step:08d}.pt'
        save_training_checkpoint(
            path=step_path,
            model=generator,
            optimizer=optimizer,
            step=int(step),
            extra=extra,
            queue_state=queue.state_dict(),
            scheduler=scheduler,
            ema_state_dict=generator_ema.state_dict(),
        )
        _prune_checkpoint_dir(checkpoint_dir=checkpoint_dir, keep_last_k=int(drift.keep_last_k_checkpoints))
    return str(latest_path)


def _prune_checkpoint_dir(*, checkpoint_dir, keep_last_k):
    if keep_last_k <= 0:
        return
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_step_*.pt'))
    if len(checkpoints) <= keep_last_k:
        return
    for stale_path in checkpoints[:-keep_last_k]:
        stale_path.unlink(missing_ok=True)


def _write_static_artifacts(*, run_dir, provider_manifest_fingerprint):
    repo_root = Path(__file__).resolve().parents[1]
    write_json(Path(run_dir) / 'env_snapshot.json', environment_snapshot(paths=[Path(run_dir)]))
    write_json(Path(run_dir) / 'codebase_fingerprint.json', codebase_fingerprint(repo_root=repo_root))
    write_json(
        Path(run_dir) / 'drift_runtime_metadata.json',
        {
            'provider_manifest_fingerprint': provider_manifest_fingerprint,
            'env_fingerprint': environment_fingerprint(),
        },
    )


def _collect_training_diagnostics(*, generator, drift, device, num_classes):
    return collect_generator_diagnostics(
        generator=generator,
        device=device,
        batch_size=8,
        num_classes=int(num_classes),
        eval_alpha=float(drift.eval_alpha),
        alpha_pair=(float(drift.alpha_min), float(drift.alpha_max)),
        seed=1234,
    )


def _merge_diagnostic_stats(*, stats, diagnostics):
    mapping = {
        'pixel_std_all': 'diag_pixel_std_all',
        'across_sample_std_mean': 'diag_across_sample_std_mean',
        'pairwise_l2_mean': 'diag_pairwise_l2_mean',
        'diff_class_l2': 'diag_diff_class_l2',
        'diff_alpha_l2': 'diag_diff_alpha_l2',
        'diff_style_l2': 'diag_diff_style_l2',
        'class_cond_norm_mean': 'diag_class_cond_norm_mean',
        'alpha_cond_norm_mean': 'diag_alpha_cond_norm_mean',
        'style_cond_norm_mean': 'diag_style_cond_norm_mean',
        'combined_cond_norm_mean': 'diag_combined_cond_norm_mean',
    }
    for src_name, dst_name in mapping.items():
        value = diagnostics.get(src_name)
        if value is not None:
            stats[dst_name] = float(value)


def _report_diagnostic_stats(*, diagnostics):
    mapping = {
        'pixel_std_all': 'Diag/pixel_std_all',
        'across_sample_std_mean': 'Diag/across_sample_std_mean',
        'pairwise_l2_mean': 'Diag/pairwise_l2_mean',
        'diff_class_l2': 'Diag/diff_class_l2',
        'diff_alpha_l2': 'Diag/diff_alpha_l2',
        'diff_style_l2': 'Diag/diff_style_l2',
        'class_cond_norm_mean': 'Diag/class_cond_norm_mean',
        'alpha_cond_norm_mean': 'Diag/alpha_cond_norm_mean',
        'style_cond_norm_mean': 'Diag/style_cond_norm_mean',
        'combined_cond_norm_mean': 'Diag/combined_cond_norm_mean',
    }
    for src_name, dst_name in mapping.items():
        value = diagnostics.get(src_name)
        if value is not None:
            training_stats.report0(dst_name, float(value))


def _snapshot_diagnostics_path(snapshot_path):
    snapshot_path = Path(snapshot_path)
    return snapshot_path.with_name(snapshot_path.name.replace('network-snapshot-', 'snapshot-diagnostics-').replace('.pkl', '.json'))


def _save_fake_grid(*, generator, drift, grid_size, class_ids, path, device):
    generator.eval()
    batch = int(class_ids.shape[0])
    noise = torch.randn(
        batch,
        int(getattr(generator, 'noise_channels', 3)),
        int(generator.img_resolution),
        int(generator.img_resolution),
        device=device,
    )
    alpha = torch.full((batch,), float(drift.eval_alpha), device=device, dtype=torch.float32)
    style_indices = torch.zeros(
        batch,
        int(getattr(generator, 'StyleTokenCount', drift.style_token_count)),
        device=device,
        dtype=torch.long,
    )
    images = generator(noise, class_ids.to(device=device, dtype=torch.long), alpha, style_indices).detach().cpu().to(torch.float).numpy()
    save_image_grid(images, path, drange=[-1, 1], grid_size=grid_size)


def _sample_real_batch(*, provider, count, device):
    images_chunks = []
    labels_chunks = []
    total = 0
    while total < count:
        images, labels = provider.next_batch(device=device)
        images_chunks.append(images)
        labels_chunks.append(labels)
        total += images.shape[0]
    return torch.cat(images_chunks, dim=0)[:count], torch.cat(labels_chunks, dim=0)[:count]


def _initial_eval_threshold(*, start_generated_images, interval_generated_images):
    if interval_generated_images <= 0:
        return 0
    return ((int(start_generated_images) // int(interval_generated_images)) + 1) * int(interval_generated_images)


def _eval_interval_generated_images(drift):
    if float(drift.eval_every_kimg) <= 0.0:
        return 0
    return int(math.ceil(float(drift.eval_every_kimg) * 1000.0))


def _periodic_metric_file_names(*, metrics, eval_entry):
    metric_names = []
    seen = set()
    for metric_name in list(metrics or []) + ['fid50k_full', 'fid50k_fullb']:
        if metric_name in eval_entry and metric_name not in seen:
            metric_names.append(str(metric_name))
            seen.add(metric_name)
    return tuple(metric_names)


def _write_periodic_metric_jsonls(*, run_dir, metrics, eval_entry):
    for metric_name in _periodic_metric_file_names(metrics=metrics, eval_entry=eval_entry):
        append_jsonl(
            path=Path(run_dir) / f'metric-{metric_name}.jsonl',
            payload=metric_jsonl_entry(eval_entry=eval_entry, metric_name=metric_name),
        )


def _total_steps(*, total_kimg, batch_size):
    return max(1, int(math.ceil(float(total_kimg) * 1000.0 / float(batch_size))))


def _ema_update(*, src, dst, decay):
    """Exponential moving average: dst = decay * dst + (1 - decay) * src."""
    with torch.no_grad():
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.lerp_(p_src, 1.0 - decay)
        for b_dst, b_src in zip(dst.buffers(), src.buffers()):
            b_dst.copy_(b_src)


def _class_ids_from_labels(labels):
    if labels.ndim == 1:
        return labels.long()
    if labels.ndim == 2:
        return labels.argmax(dim=1).long()
    raise ValueError(f'labels must be [B] or [B, C], got {tuple(labels.shape)}')


def _run_r3gan_conv_step(*, generator, optimizer, queue, provider, step, local_groups, num_classes, drift, image_channels, image_size, class_labels, alpha, device, step_config=None, feature_extractor=None):
    should_refill = str(drift.queue_refill_policy) == 'per_step' or (
        str(drift.queue_refill_policy) == 'every_n_steps'
        and (step % max(1, int(drift.queue_refill_every)) == 0)
    )
    if should_refill:
        refill_images, refill_labels = _sample_real_batch(provider=provider, count=int(drift.queue_push_batch), device=device)
        queue.push(refill_images, refill_labels)
    backfilled = ensure_queue_has_labels(
        queue=queue,
        class_labels=class_labels,
        provider=provider,
        required_count=int(drift.positives_per_group if drift.queue_strict_without_replacement else 1),
        device=device,
    )
    positives_grouped, unconditional_grouped = sample_grouped_real_batches(
        queue=queue,
        class_labels=class_labels,
        config=GroupedSamplingConfig(
            positives_per_group=int(drift.positives_per_group),
            unconditional_per_group=int(drift.unconditional_per_group),
        ),
        device=device,
    )
    unconditional_weight_grouped = torch.tensor(
        [
            cfg_alpha_to_unconditional_weight(
                alpha=float(alpha[g_index].item()),
                n_generated_negatives=int(drift.negatives_per_group),
                n_unconditional_negatives=int(drift.unconditional_per_group),
            )
            for g_index in range(local_groups)
        ],
        device=device,
        dtype=torch.float32,
    )

    negatives_per_group = int(drift.negatives_per_group)
    z = torch.randn([local_groups * negatives_per_group, generator.z_dim], device=device)
    one_hot = torch.zeros([z.shape[0], num_classes], device=device, dtype=torch.float32)
    one_hot.scatter_(1, class_labels.repeat_interleave(negatives_per_group).view(-1, 1), 1.0)
    fake_images = generator(z, one_hot, alpha=alpha.repeat_interleave(negatives_per_group))
    fake_grouped = fake_images.reshape(local_groups, negatives_per_group, image_channels, image_size, image_size)

    if step_config is None:
        step_config = GroupedDriftStepConfig(
            loss_config=DriftingLossConfig(
                drift_field=DriftFieldConfig(
                    temperature=float(drift.drift_temperature),
                    normalize_over_x=True,
                    mask_self_negatives=True,
                ),
                attraction_scale=1.0,
                repulsion_scale=1.0,
                stopgrad_target=True,
            ),
            feature_config=_build_feature_config(drift),
            drift_temperatures=tuple(float(v) for v in drift.drift_temperatures),
            drift_temperature_reduction=str(drift.drift_temperature_reduction),
            clip_grad_norm=float(drift.clip_grad_norm),
            run_optimizer_step=False,
        )

    from drifting_models.drift_field import build_negative_log_weights
    from drifting_models.drift_loss import (
        drifting_stopgrad_loss,
        drifting_stopgrad_loss_multi_temperature,
        feature_space_drifting_loss,
    )
    from drifting_models.features.vectorize import extract_feature_maps, vectorize_feature_maps

    losses = []
    drift_norms = []
    for g_idx in range(local_groups):
        gen_group = fake_grouped[g_idx]
        pos_group = positives_grouped[g_idx]
        unc_group = None if unconditional_grouped is None else unconditional_grouped[g_idx]
        unc_weight = float(unconditional_weight_grouped[g_idx].item())

        if step_config.feature_config is not None and feature_extractor is not None:
            gen_feats = vectorize_feature_maps(extract_feature_maps(feature_extractor, gen_group), step_config.feature_config.vectorization)
            pos_feats = vectorize_feature_maps(extract_feature_maps(feature_extractor, pos_group), step_config.feature_config.vectorization)
            unc_feats = None
            if unc_group is not None:
                unc_feats = vectorize_feature_maps(extract_feature_maps(feature_extractor, unc_group), step_config.feature_config.vectorization)
            loss, stats = feature_space_drifting_loss(
                generated_feature_vectors=gen_feats,
                positive_feature_vectors=pos_feats,
                unconditional_feature_vectors=unc_feats,
                base_loss_config=step_config.loss_config,
                feature_config=step_config.feature_config,
                unconditional_weight=unc_weight,
            )
        else:
            gen_vec = gen_group.reshape(gen_group.shape[0], -1)
            pos_vec = pos_group.reshape(pos_group.shape[0], -1)
            neg_vec = gen_vec
            neg_log_w = None
            if unc_group is not None:
                unc_vec = unc_group.reshape(unc_group.shape[0], -1)
                neg_vec = torch.cat([gen_vec, unc_vec], dim=0)
                neg_log_w = build_negative_log_weights(
                    n_generated_negatives=gen_vec.shape[0],
                    n_unconditional_negatives=unc_vec.shape[0],
                    unconditional_weight=unc_weight,
                    device=gen_vec.device,
                    dtype=gen_vec.dtype,
                )
            if step_config.drift_temperatures:
                loss, stats = drifting_stopgrad_loss_multi_temperature(
                    x=gen_vec, y_pos=pos_vec, y_neg=neg_vec,
                    temperatures=tuple(step_config.drift_temperatures),
                    config=step_config.loss_config,
                    negative_log_weights=neg_log_w,
                    generated_negative_count=gen_vec.shape[0],
                    reduction=str(step_config.drift_temperature_reduction),
                )
            else:
                loss, _, stats = drifting_stopgrad_loss(
                    x=gen_vec, y_pos=pos_vec, y_neg=neg_vec,
                    config=step_config.loss_config,
                    negative_log_weights=neg_log_w,
                    generated_negative_count=gen_vec.shape[0],
                )

        losses.append(loss)
        drift_norms.append(stats.get('mean_drift_norm', stats.get('drift_norm', 0.0)))

    total_loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = None
    clip = step_config.clip_grad_norm
    if clip is not None and clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
    optimizer.step()

    result = {
        'loss': float(total_loss.item()),
        'mean_drift_norm': float(sum(drift_norms) / max(len(drift_norms), 1)),
        'groups': local_groups,
        'negatives_per_group': negatives_per_group,
        'alpha_mean': float(alpha.mean().item()),
        'alpha_min': float(alpha.min().item()),
        'alpha_max': float(alpha.max().item()),
        'grad_norm': None if grad_norm is None else float(grad_norm.item()),
        'queue_underflow_backfilled': float(backfilled),
    }
    return result
