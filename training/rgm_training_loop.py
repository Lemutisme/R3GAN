"""Prototype RGM training loop."""

from dataclasses import asdict
import copy
import json
import os
import pickle
import time

import numpy as np
import psutil
import torch

import dnnlib
import legacy
from metrics import metric_main
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix

from training.drift_queue import ClassConditionalSampleQueue, QueueConfig, ensure_class_coverage
from training.rgm_loss import RGMLossConfig, grouped_rank_drifting_stopgrad_loss
from training.training_loop import cosine_decay_with_warmup, remap_optimizer_state_dict, save_image_grid, setup_snapshot_image_grid


def training_loop(
    run_dir=".",
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
    ema_scheduler=None,
    total_kimg=25000,
    kimg_per_tick=4,
    image_snapshot_ticks=50,
    network_snapshot_ticks=50,
    snapshot_policy="all",
    resume_pkl=None,
    cudnn_benchmark=True,
    abort_fn=None,
    progress_fn=None,
    negatives_per_group=4,
    positives_per_group=4,
    unconditional_per_group=2,
    queue_capacity_per_class=256,
    queue_capacity_global=4096,
    queue_push_batch=128,
    queue_warmup_batches=4,
    rank_levels=(),
    rgm_mode="drift_rank",
    loss_kwargs={},
    **_unused_kwargs,
):
    if rgm_mode != "drift_rank":
        raise ValueError(f"unsupported rgm_mode for v0: {rgm_mode}")

    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    rank_levels_tensor = torch.as_tensor(rank_levels, device=device, dtype=torch.float32)
    if rank_levels_tensor.ndim != 1 or rank_levels_tensor.numel() < 2:
        raise ValueError("rank_levels must contain at least two entries")
    if torch.any(rank_levels_tensor[1:] >= rank_levels_tensor[:-1]):
        raise ValueError("rank_levels must be strictly descending")

    if batch_size % num_gpus != 0:
        raise ValueError("batch_size must be divisible by num_gpus")
    local_batch_size = batch_size // num_gpus
    if local_batch_size % (negatives_per_group * rank_levels_tensor.numel()) != 0:
        raise ValueError("batch_size / num_gpus must be divisible by negatives_per_group * len(rank_levels)")
    if queue_push_batch % num_gpus != 0:
        raise ValueError("queue_push_batch must be divisible by num_gpus")
    local_queue_push_batch = queue_push_batch // num_gpus

    groups = local_batch_size // (negatives_per_group * rank_levels_tensor.numel())
    base_latent_count = groups * negatives_per_group

    if rank == 0:
        print("Loading training set...")
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    if not training_set.has_labels:
        raise ValueError("rgm trainer requires a labeled dataset")
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set,
        rank=rank,
        num_replicas=num_gpus,
        seed=random_seed,
    )
    loader_batch_size = max(base_latent_count, local_queue_push_batch)
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set,
            sampler=training_set_sampler,
            batch_size=loader_batch_size,
            **data_loader_kwargs,
        )
    )
    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print("Label shape:", training_set.label_shape)
        print()

    if rank == 0:
        print("Constructing networks...")
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()
    G_opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)
    queue = ClassConditionalSampleQueue(
        QueueConfig(
            num_classes=training_set.label_dim,
            per_class_capacity=queue_capacity_per_class,
            global_capacity=queue_capacity_global,
            store_device="cpu",
            strict_without_replacement=False,
        )
    )

    resume_data = None
    if resume_pkl is not None:
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        if rank == 0:
            print(f'Resuming from "{resume_pkl}"')
        for name, module in [("G", G), ("G_ema", G_ema)]:
            if resume_data.get(name) is not None:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        if "G_opt_state" in resume_data:
            G_opt.load_state_dict(remap_optimizer_state_dict(resume_data["G_opt_state"], device))
        if "queue_state" in resume_data and isinstance(resume_data["queue_state"], dict):
            queue.load_state_dict(resume_data["queue_state"])

    if rank == 0:
        z = torch.empty([local_batch_size, G.z_dim], device=device)
        c = torch.empty([local_batch_size, G.c_dim], device=device)
        misc.print_module_summary(G, [z, c])

    if rank == 0:
        print(f"Distributing across {num_gpus} GPUs...")
    for module in [G, G_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    if rank == 0:
        print("Priming RGM queue...")
    if resume_data is None or "queue_state" not in resume_data:
        for _idx in range(queue_warmup_batches):
            warm_images, warm_labels = _next_queue_batch(
                training_set_iterator=training_set_iterator,
                device=device,
                local_queue_push_batch=local_queue_push_batch,
            )
            queue.push(warm_images, warm_labels)

    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print("Exporting sample images...")
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(local_batch_size)
        grid_c = torch.from_numpy(labels).to(device).split(local_batch_size)
        images = torch.cat([G_ema(z, c).cpu() for z, c in zip(grid_z, grid_c)]).to(torch.float).numpy()
        save_image_grid(images, os.path.join(run_dir, "fakes_init.png"), drange=[-1, 1], grid_size=grid_size)

    if rank == 0:
        print("Initializing logs...")
    stats_collector = training_stats.Collector(regex=".*")
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        try:
            import torch.utils.tensorboard as tensorboard

            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print("Skipping tfevents export:", err)

    latest_snapshot_pkl = None
    latest_image_png = None
    best_snapshot_pkl = None
    best_image_png = None
    best_metric_name = metrics[0] if len(metrics) > 0 else None
    best_metric_value = None

    def _is_better_metric(metric_name, candidate, reference):
        if metric_name.startswith("fid") or metric_name.startswith("kid"):
            return candidate < reference
        return candidate > reference

    def _safe_remove(path, protected):
        if path is None or path in protected:
            return
        if os.path.isfile(path):
            os.remove(path)

    if rank == 0:
        print(f"Training for {total_kimg} kimg...")
        print()
    cur_nimg = resume_data["cur_nimg"] if resume_data is not None else 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    step_start_event = None
    step_end_event = None
    if rank == 0:
        step_start_event = torch.cuda.Event(enable_timing=True)
        step_end_event = torch.cuda.Event(enable_timing=True)
        step_start_event.record(torch.cuda.current_stream(device))
        step_end_event.record(torch.cuda.current_stream(device))

    loss_config = RGMLossConfig(**loss_kwargs)
    last_drift_norm = 0.0

    while True:
        if step_start_event is not None:
            step_start_event.record(torch.cuda.current_stream(device))

        cur_lr = cosine_decay_with_warmup(cur_nimg, **lr_scheduler) if lr_scheduler is not None else G_opt.param_groups[0]["lr"]
        cur_beta2 = cosine_decay_with_warmup(cur_nimg, **beta2_scheduler) if beta2_scheduler is not None else G_opt.param_groups[0]["betas"][1]
        cur_ema_nimg = cosine_decay_with_warmup(cur_nimg, **ema_scheduler) if ema_scheduler is not None else float(batch_size)

        for group in G_opt.param_groups:
            group["lr"] = cur_lr
            group["betas"] = (group["betas"][0], float(cur_beta2))

        queue_images, queue_labels = _next_queue_batch(
            training_set_iterator=training_set_iterator,
            device=device,
            local_queue_push_batch=local_queue_push_batch,
        )
        queue.push(queue_images, queue_labels)

        group_class_ids = torch.randint(0, G.c_dim, [groups], device=device)
        ensure_class_coverage(
            queue,
            group_class_ids,
            refill_fn=lambda: _next_queue_batch(
                training_set_iterator=training_set_iterator,
                device=device,
                local_queue_push_batch=local_queue_push_batch,
            ),
            required_count=max(positives_per_group, 1),
        )

        positives_grouped = queue.sample_positive_grouped(group_class_ids, positives_per_group, device)
        unconditional_grouped = queue.sample_unconditional_grouped(groups, unconditional_per_group, device)
        unconditional_weights = torch.ones([groups], device=device, dtype=torch.float32)

        z_base = torch.randn([base_latent_count, G.z_dim], device=device)
        c_base = _one_hot(group_class_ids.repeat_interleave(negatives_per_group), G.c_dim, device)
        flat_z = z_base.unsqueeze(1).expand(-1, rank_levels_tensor.numel(), -1).reshape(local_batch_size, G.z_dim)
        flat_c = c_base.unsqueeze(1).expand(-1, rank_levels_tensor.numel(), -1).reshape(local_batch_size, G.c_dim)
        flat_rank = rank_levels_tensor.view(1, -1).expand(base_latent_count, -1).reshape(local_batch_size)

        G_opt.zero_grad(set_to_none=True)
        G.requires_grad_(True)
        fake_images = G(flat_z, flat_c, rank=flat_rank)
        channels, height, width = fake_images.shape[1:]
        fake_grouped = fake_images.reshape(base_latent_count, rank_levels_tensor.numel(), channels, height, width)
        fake_grouped = fake_grouped.reshape(groups, negatives_per_group, rank_levels_tensor.numel(), channels, height, width)
        fake_grouped = fake_grouped.permute(0, 2, 1, 3, 4, 5).contiguous()
        loss, drift_stats = grouped_rank_drifting_stopgrad_loss(
            x_grouped=fake_grouped,
            y_pos_grouped=positives_grouped,
            unconditional_grouped=unconditional_grouped,
            unconditional_weight_grouped=unconditional_weights,
            rank_levels=rank_levels_tensor,
            config=loss_config,
        )
        loss.backward()
        G.requires_grad_(False)

        params = [param for param in G.parameters() if param.grad is not None]
        if len(params) > 0:
            flat = torch.cat([param.grad.flatten() for param in params])
            if num_gpus > 1:
                torch.distributed.all_reduce(flat)
                flat /= num_gpus
            grads = flat.split([param.numel() for param in params])
            for param, grad in zip(params, grads):
                param.grad = grad.reshape(param.shape)
        G_opt.step()

        with torch.autograd.profiler.record_function("Gema"):
            ema_beta = 0.5 ** (batch_size / max(cur_ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        last_drift_norm = float(drift_stats["mean_drift_norm"])
        training_stats.report("Loss/G/loss", loss.detach())
        training_stats.report("Loss/G/transport", torch.as_tensor(drift_stats["transport_loss"], device=device))
        training_stats.report("Loss/G/order", torch.as_tensor(drift_stats["order_loss"], device=device))
        training_stats.report("Loss/G/eq", torch.as_tensor(drift_stats["eq_loss"], device=device))
        training_stats.report("Loss/drift_norm", torch.as_tensor(drift_stats["mean_drift_norm"], device=device))
        training_stats.report("Loss/drift_norm_best", torch.as_tensor(drift_stats["best_rank_drift_norm"], device=device))
        training_stats.report("Loss/drift_norm_worst", torch.as_tensor(drift_stats["worst_rank_drift_norm"], device=device))
        training_stats.report("Progress/queue_global_count", torch.as_tensor(float(queue.global_count()), device=device))
        training_stats.report("Progress/rank_best", rank_levels_tensor[-1])
        training_stats.report("Progress/rank_worst", rank_levels_tensor[0])

        if step_end_event is not None:
            step_end_event.record(torch.cuda.current_stream(device))

        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        training_stats.report0("Progress/tick", cur_tick)
        training_stats.report0("Progress/kimg", cur_nimg / 1e3)
        training_stats.report0("Progress/lr", cur_lr)
        training_stats.report0("Progress/ema_mimg", cur_ema_nimg / 1e6)
        training_stats.report0("Progress/beta2", cur_beta2)
        training_stats.report0("Progress/drift_norm", last_drift_norm)
        training_stats.report0("Progress/queue_global_count", float(queue.global_count()))
        training_stats.report0("Timing/total_sec", tick_end_time - start_time)
        training_stats.report0("Timing/sec_per_tick", tick_end_time - tick_start_time)
        training_stats.report0("Timing/sec_per_kimg", (tick_end_time - tick_start_time) / max(cur_nimg - tick_start_nimg, 1) * 1e3)
        training_stats.report0("Timing/maintenance_sec", maintenance_time)
        training_stats.report0("Resources/cpu_mem_gb", psutil.Process(os.getpid()).memory_info().rss / 2**30)
        training_stats.report0("Resources/peak_gpu_mem_gb", torch.cuda.max_memory_allocated(device) / 2**30)
        training_stats.report0("Resources/peak_gpu_mem_reserved_gb", torch.cuda.max_memory_reserved(device) / 2**30)
        training_stats.report0("Timing/total_hours", (tick_end_time - start_time) / (60 * 60))
        training_stats.report0("Timing/total_days", (tick_end_time - start_time) / (24 * 60 * 60))
        torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            fields = []
            fields += [f"tick {cur_tick:<5d}"]
            fields += [f"kimg {cur_nimg / 1e3:<8.1f}"]
            fields += [f"time {dnnlib.util.format_time(tick_end_time - start_time):<12s}"]
            fields += [f"sec/tick {tick_end_time - tick_start_time:<7.1f}"]
            fields += [f"sec/kimg {(tick_end_time - tick_start_time) / max(cur_nimg - tick_start_nimg, 1) * 1e3:<7.2f}"]
            fields += [f"maintenance {maintenance_time:<6.1f}"]
            fields += [f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"]
            fields += [f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"]
            fields += [f"reserved {torch.cuda.max_memory_reserved(device) / 2**30:<6.2f}"]
            fields += [f"drift {last_drift_norm:.4f}"]
            fields += [f"queue {queue.global_count():<6d}"]
            print(" ".join(fields))

        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print("Aborting...")

        save_network_this_tick = (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0)
        save_image_this_tick = (rank == 0) and (
            ((image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0))
            or (snapshot_policy == "latest-best" and save_network_this_tick)
        )

        image_snapshot_path = None
        if save_image_this_tick:
            images = torch.cat([G_ema(z, c).cpu() for z, c in zip(grid_z, grid_c)]).to(torch.float).numpy()
            image_snapshot_path = os.path.join(run_dir, f"fakes{cur_nimg // 1000:09d}.png")
            save_image_grid(images, image_snapshot_path, drange=[-1, 1], grid_size=grid_size)

        snapshot_pkl = None
        snapshot_data = None
        if save_network_this_tick:
            snapshot_data = dict(
                G=G,
                D=None,
                G_ema=G_ema,
                training_set_kwargs=dict(training_set_kwargs),
                cur_nimg=cur_nimg,
                trainer="rgm",
                rgm_mode=rgm_mode,
                rank_levels=[float(value) for value in rank_levels_tensor.detach().cpu().tolist()],
                condition_schema=asdict(G.ConditionSchema),
                queue_state=queue.state_dict(),
                G_opt_state=remap_optimizer_state_dict(G_opt.state_dict(), "cpu"),
                transport_state=None,
            )
            for key, value in list(snapshot_data.items()):
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value)
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
            snapshot_pkl = os.path.join(run_dir, f"network-snapshot-{cur_nimg // 1000:09d}.pkl")
            if rank == 0:
                with open(snapshot_pkl, "wb") as f:
                    pickle.dump(snapshot_data, f)

        snapshot_metric_results = dict()
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print("Evaluating metrics...")
            for metric in metrics:
                result_dict = metric_main.calc_metric(
                    metric=metric,
                    G=snapshot_data["G_ema"],
                    dataset_kwargs=training_set_kwargs,
                    num_gpus=num_gpus,
                    rank=rank,
                    device=device,
                )
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
                snapshot_metric_results.update(result_dict.results)
        del snapshot_data

        if (rank == 0) and (snapshot_policy == "latest-best") and (snapshot_pkl is not None):
            is_best = False
            metric_value = None
            if (best_metric_name is not None) and (best_metric_name in snapshot_metric_results):
                metric_value = snapshot_metric_results[best_metric_name]
                if (best_metric_value is None) or _is_better_metric(best_metric_name, metric_value, best_metric_value):
                    is_best = True
            elif best_snapshot_pkl is None:
                is_best = True

            prev_best_snapshot_pkl = best_snapshot_pkl
            prev_best_image_png = best_image_png
            if is_best:
                best_snapshot_pkl = snapshot_pkl
                best_image_png = image_snapshot_path
                if metric_value is not None:
                    best_metric_value = metric_value
                    print(f"Updated best snapshot by {best_metric_name}: {best_metric_value:.6f}")

            protected = {snapshot_pkl, image_snapshot_path, best_snapshot_pkl, best_image_png}
            _safe_remove(latest_snapshot_pkl, protected)
            _safe_remove(latest_image_png, protected)
            _safe_remove(prev_best_snapshot_pkl, protected)
            _safe_remove(prev_best_image_png, protected)
            latest_snapshot_pkl = snapshot_pkl
            latest_image_png = image_snapshot_path

        step_timing = []
        if step_start_event is not None and step_end_event is not None:
            step_end_event.synchronize()
            step_timing = step_start_event.elapsed_time(step_end_event)
        training_stats.report0("Timing/G", step_timing)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + "\n")
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f"Metrics/{name}", value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    if rank == 0:
        print()
        print("Exiting...")


def _next_queue_batch(training_set_iterator, device, local_queue_push_batch):
    images, labels = next(training_set_iterator)
    images = images[:local_queue_push_batch].detach().clone().to(device).to(torch.float32) / 127.5 - 1
    labels = labels[:local_queue_push_batch].detach().clone().to(device)
    return images, _class_ids_from_labels(labels)


def _class_ids_from_labels(labels):
    if labels.ndim == 1:
        return labels.long()
    if labels.ndim != 2 or labels.shape[1] == 0:
        raise ValueError("rgm trainer requires one-hot or integer labels")
    return labels.argmax(dim=1).long()


def _one_hot(class_ids, num_classes, device):
    out = torch.zeros([class_ids.shape[0], num_classes], device=device, dtype=torch.float32)
    out.scatter_(1, class_ids.view(-1, 1), 1.0)
    return out
