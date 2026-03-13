from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path

import torch

import dnnlib
import legacy

#----------------------------------------------------------------------------


def _generator_device(generator):
    try:
        return next(generator.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _one_hot(class_ids, num_classes, device):
    if num_classes <= 0:
        return torch.zeros(class_ids.shape[0], 0, device=device, dtype=torch.float32)
    cond = torch.zeros(class_ids.shape[0], num_classes, device=device, dtype=torch.float32)
    cond.scatter_(1, class_ids.view(-1, 1), 1.0)
    return cond


def _supports_kwarg(generator, name):
    try:
        signature = inspect.signature(generator.forward)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def _sample_noise(generator, *, batch_size, device):
    if hasattr(generator, 'noise_channels'):
        return torch.randn(
            batch_size,
            int(getattr(generator, 'noise_channels')),
            int(getattr(generator, 'img_resolution')),
            int(getattr(generator, 'img_resolution')),
            device=device,
        )
    return torch.randn(batch_size, int(getattr(generator, 'z_dim')), device=device)


def _zero_style_indices(generator, *, batch_size, device):
    return torch.zeros(
        batch_size,
        int(getattr(generator, 'StyleTokenCount', 0)),
        device=device,
        dtype=torch.long,
    )


def _call_generator(generator, *, noise, class_ids, alpha=None, style_indices=None):
    device = noise.device
    cond = _one_hot(class_ids=class_ids, num_classes=int(getattr(generator, 'c_dim', 0)), device=device)
    attempts = [
        {key: value for key, value in {'alpha': alpha, 'style_indices': style_indices}.items() if value is not None},
        {key: value for key, value in {'alpha': alpha}.items() if value is not None},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return generator(noise, cond, **kwargs).to(torch.float32)
        except TypeError as err:
            last_error = err
    if last_error is not None:
        raise last_error
    raise RuntimeError('generator invocation unexpectedly failed')


def _mean_pairwise_l2(images):
    if images.shape[0] < 2:
        return 0.0
    flat = images.reshape(images.shape[0], -1)
    distances = torch.cdist(flat, flat, p=2)
    mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    if not torch.any(mask):
        return 0.0
    return float(distances[mask].mean().item())


def _mean_sample_l2(lhs, rhs):
    return float((lhs.reshape(lhs.shape[0], -1) - rhs.reshape(rhs.shape[0], -1)).norm(dim=1).mean().item())


def _conditioning_norms(generator, *, class_ids, alpha, style_indices):
    model = getattr(generator, 'Model', None)
    config = getattr(model, 'config', None)
    if model is None or config is None:
        return {}
    if not all(hasattr(model, name) for name in ('class_embedding', 'alpha_embedding', 'style_embedding')):
        return {}

    class_cond = model.class_embedding(class_ids)
    alpha_cond = model.alpha_embedding(alpha)
    style_cond = model.style_embedding(style_indices).sum(dim=1)
    if style_indices.shape[1] > 1:
        style_cond = style_cond / math.sqrt(float(style_indices.shape[1]))
    combined = class_cond + alpha_cond + style_cond
    return {
        'class_cond_norm_mean': float(class_cond.norm(dim=1).mean().item()),
        'alpha_cond_norm_mean': float(alpha_cond.norm(dim=1).mean().item()),
        'style_cond_norm_mean': float(style_cond.norm(dim=1).mean().item()),
        'combined_cond_norm_mean': float(combined.norm(dim=1).mean().item()),
    }


def collect_generator_diagnostics(
    *,
    generator,
    device=None,
    batch_size=8,
    num_classes=None,
    eval_alpha=None,
    alpha_pair=(1.0, 4.0),
    seed=0,
):
    if device is None:
        device = _generator_device(generator)
    device = torch.device(device)
    batch_size = int(batch_size)
    num_classes = int(getattr(generator, 'c_dim', 0) if num_classes is None else num_classes)
    eval_alpha = float(getattr(generator, 'EvalAlpha', 1.0) if eval_alpha is None else eval_alpha)
    alpha_pair = tuple(float(v) for v in alpha_pair)

    generator_was_training = bool(generator.training)
    generator.eval()
    rng_devices = []
    if device.type == 'cuda':
        rng_devices = [torch.cuda.current_device() if device.index is None else int(device.index)]
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(int(seed))
        noise = _sample_noise(generator, batch_size=batch_size, device=device)
        class_ids = torch.arange(batch_size, device=device, dtype=torch.long) % max(num_classes, 1)
        alpha = torch.full((batch_size,), eval_alpha, device=device, dtype=torch.float32)
        style_zero = _zero_style_indices(generator, batch_size=batch_size, device=device)

        with torch.no_grad():
            base_images = _call_generator(
                generator,
                noise=noise,
                class_ids=class_ids,
                alpha=alpha if _supports_kwarg(generator, 'alpha') or hasattr(generator, 'EvalAlpha') else None,
                style_indices=style_zero if _supports_kwarg(generator, 'style_indices') else None,
            )

            diagnostics = {
                'pixel_std_all': float(base_images.std().item()),
                'across_sample_std_mean': float(base_images.std(dim=0).mean().item()),
                'pairwise_l2_mean': _mean_pairwise_l2(base_images[: min(batch_size, 8)]),
                'diff_class_l2': None,
                'diff_alpha_l2': None,
                'diff_style_l2': None,
            }

            if num_classes > 1:
                alt_class_ids = (class_ids + 1) % num_classes
                alt_class_images = _call_generator(
                    generator,
                    noise=noise,
                    class_ids=alt_class_ids,
                    alpha=alpha if _supports_kwarg(generator, 'alpha') or hasattr(generator, 'EvalAlpha') else None,
                    style_indices=style_zero if _supports_kwarg(generator, 'style_indices') else None,
                )
                diagnostics['diff_class_l2'] = _mean_sample_l2(base_images, alt_class_images)

            if _supports_kwarg(generator, 'alpha') or hasattr(generator, 'EvalAlpha'):
                alpha_low = torch.full((batch_size,), alpha_pair[0], device=device, dtype=torch.float32)
                alpha_high = torch.full((batch_size,), alpha_pair[1], device=device, dtype=torch.float32)
                alpha_low_images = _call_generator(
                    generator,
                    noise=noise,
                    class_ids=class_ids,
                    alpha=alpha_low,
                    style_indices=style_zero if _supports_kwarg(generator, 'style_indices') else None,
                )
                alpha_high_images = _call_generator(
                    generator,
                    noise=noise,
                    class_ids=class_ids,
                    alpha=alpha_high,
                    style_indices=style_zero if _supports_kwarg(generator, 'style_indices') else None,
                )
                diagnostics['diff_alpha_l2'] = _mean_sample_l2(alpha_low_images, alpha_high_images)

            if _supports_kwarg(generator, 'style_indices') and int(getattr(generator, 'StyleTokenCount', 0)) > 0:
                style_random = torch.randint(
                    0,
                    max(1, int(getattr(generator, 'style_vocab_size', 1))),
                    style_zero.shape,
                    device=device,
                )
                styled_images = _call_generator(
                    generator,
                    noise=noise,
                    class_ids=class_ids,
                    alpha=alpha if _supports_kwarg(generator, 'alpha') or hasattr(generator, 'EvalAlpha') else None,
                    style_indices=style_random,
                )
                diagnostics['diff_style_l2'] = _mean_sample_l2(base_images, styled_images)
                diagnostics.update(
                    _conditioning_norms(
                        generator,
                        class_ids=class_ids,
                        alpha=alpha,
                        style_indices=style_random,
                    )
                )
            else:
                diagnostics.update(
                    _conditioning_norms(
                        generator,
                        class_ids=class_ids,
                        alpha=alpha,
                        style_indices=style_zero,
                    )
                )

    if generator_was_training:
        generator.train()
    return diagnostics


def infer_snapshot_kimg(snapshot_path):
    match = re.search(r'network-snapshot-(\d+)\.pkl$', str(snapshot_path))
    return None if match is None else int(match.group(1))


def load_snapshot_drift_norm(snapshot_path):
    run_dir = Path(snapshot_path).resolve().parent
    stats_path = run_dir / 'stats.jsonl'
    snapshot_kimg = infer_snapshot_kimg(snapshot_path)
    if not stats_path.is_file() or snapshot_kimg is None:
        return {'drift_norm': None, 'drift_norm_progress_kimg': None}

    best_entry = None
    best_distance = None
    with stats_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            progress = entry.get('Progress/kimg', {})
            drift = entry.get('Loss/drift_norm', {})
            if not isinstance(progress, dict) or not isinstance(drift, dict):
                continue
            progress_kimg = progress.get('mean')
            drift_norm = drift.get('mean')
            if progress_kimg is None or drift_norm is None:
                continue
            distance = abs(float(progress_kimg) - float(snapshot_kimg))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_entry = {
                    'drift_norm': float(drift_norm),
                    'drift_norm_progress_kimg': float(progress_kimg),
                }
    if best_entry is None:
        return {'drift_norm': None, 'drift_norm_progress_kimg': None}
    return best_entry


def load_generator_from_snapshot(*, snapshot_path, device='cpu', use_ema=True):
    snapshot_path = Path(snapshot_path)
    with dnnlib.util.open_url(str(snapshot_path)) as handle:
        payload = legacy.load_network_pkl(handle)
    key = 'G_ema' if bool(use_ema) and payload.get('G_ema') is not None else 'G'
    generator = payload[key]
    generator = generator.eval().requires_grad_(False).to(torch.device(device))
    return generator


def analyze_snapshot(
    *,
    snapshot_path,
    device='cpu',
    batch_size=8,
    eval_alpha=None,
    alpha_pair=(1.0, 4.0),
    seed=0,
    use_ema=True,
):
    snapshot_path = Path(snapshot_path).resolve()
    generator = load_generator_from_snapshot(snapshot_path=snapshot_path, device=device, use_ema=use_ema)
    diagnostics = collect_generator_diagnostics(
        generator=generator,
        device=device,
        batch_size=batch_size,
        num_classes=int(getattr(generator, 'c_dim', 0)),
        eval_alpha=eval_alpha,
        alpha_pair=alpha_pair,
        seed=seed,
    )
    drift_norm_info = load_snapshot_drift_norm(snapshot_path)
    conditioning_norms = {
        'class': diagnostics.pop('class_cond_norm_mean', None),
        'alpha': diagnostics.pop('alpha_cond_norm_mean', None),
        'style': diagnostics.pop('style_cond_norm_mean', None),
        'combined': diagnostics.pop('combined_cond_norm_mean', None),
    }
    return {
        'snapshot_path': str(snapshot_path),
        'snapshot_kimg': infer_snapshot_kimg(snapshot_path),
        'generator_class': generator.__class__.__name__,
        'used_ema_generator': bool(use_ema),
        **drift_norm_info,
        **diagnostics,
        'conditioning_norms': conditioning_norms,
    }
