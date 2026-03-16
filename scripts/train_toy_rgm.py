#!/usr/bin/env python3

"""Train toy 2D drifting and rank-drift RGM experiments."""

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.rgm_loss import RGMLossConfig, grouped_rank_drifting_stopgrad_loss
from training.toy_datasets import sample_checkerboard, sample_four_gaussians, sample_swiss_roll
from training.toy_models import ToyMLP, ToyRankMLP


SAMPLERS = {
    "checkerboard": sample_checkerboard,
    "swiss_roll": sample_swiss_roll,
    "four_gaussians": sample_four_gaussians,
}


def main():
    parser = argparse.ArgumentParser(description="Train toy RGM experiments")
    parser.add_argument("--dataset", choices=sorted(SAMPLERS.keys()), default="checkerboard")
    parser.add_argument("--toy-mode", choices=["drift", "rank_drift"], default="rank_drift")
    parser.add_argument("--rank-levels", type=str, default="1.0,0.5,0.0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--plot-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sampler = SAMPLERS[args.dataset]
    rank_levels = [float(value) for value in args.rank_levels.split(",") if value]
    if args.toy_mode == "rank_drift" and len(rank_levels) < 2:
        raise ValueError("rank_drift requires at least two rank levels")

    if args.toy_mode == "drift":
        model = ToyMLP().to(device)
    else:
        model = ToyRankMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_config = RGMLossConfig(temperature=args.temp)

    z_vis = torch.randn(256, 32, device=device)
    history = []
    for step in range(1, args.steps + 1):
        positives = sampler(1024, seed=args.seed + step).to(device)
        z = torch.randn(512, 32, device=device)
        if args.toy_mode == "drift":
            generated = model(z).reshape(1, 1, z.shape[0], 2, 1, 1)
        else:
            generated = torch.stack(
                [model(z, torch.full([z.shape[0]], rank, device=device)) for rank in rank_levels],
                dim=1,
            ).reshape(1, len(rank_levels), z.shape[0], 2, 1, 1)
        positives_grouped = positives.reshape(1, positives.shape[0], 2, 1, 1)

        optimizer.zero_grad(set_to_none=True)
        loss, stats = grouped_rank_drifting_stopgrad_loss(
            x_grouped=generated,
            y_pos_grouped=positives_grouped,
            unconditional_grouped=None,
            unconditional_weight_grouped=None,
            rank_levels=rank_levels if args.toy_mode == "rank_drift" else [1.0],
            config=loss_config,
        )
        loss.backward()
        optimizer.step()

        history.append({"step": step, **stats})
        if step == 1 or step % args.plot_every == 0 or step == args.steps:
            _save_visuals(
                model=model,
                sampler=sampler,
                z_vis=z_vis,
                rank_levels=rank_levels,
                step=step,
                toy_mode=args.toy_mode,
                outdir=outdir,
                device=device,
            )

    summary = {
        "dataset": args.dataset,
        "toy_mode": args.toy_mode,
        "rank_levels": rank_levels,
        "steps": args.steps,
        "lr": args.lr,
        "temp": args.temp,
        "seed": args.seed,
        "final": history[-1],
    }
    if args.toy_mode == "rank_drift":
        summary["diagnostics"] = _compute_rank_diagnostics(
            model=model,
            z_vis=z_vis,
            rank_levels=rank_levels,
            device=device,
        )
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _save_visuals(model, sampler, z_vis, rank_levels, step, toy_mode, outdir, device):
    target = sampler(2048, seed=step).cpu().numpy()
    if toy_mode == "drift":
        outputs = [model(z_vis).detach().cpu().numpy()]
        labels = ["drift"]
    else:
        outputs = [
            model(z_vis, torch.full([z_vis.shape[0]], rank, device=device)).detach().cpu().numpy()
            for rank in rank_levels
        ]
        labels = [f"rank={rank:g}" for rank in rank_levels]

    bounds = _compute_bounds([target, *outputs])
    panels = [_render_scatter(target, color=(0, 0, 0), bounds=bounds)]
    panels.extend(_render_scatter(output, color=(230, 126, 34), bounds=bounds) for output in outputs)
    scatter_strip = Image.new("RGB", (panels[0].width * len(panels), panels[0].height), "white")
    draw = ImageDraw.Draw(scatter_strip)
    for panel_index, panel in enumerate(panels):
        scatter_strip.paste(panel, (panel_index * panel.width, 0))
        label = "target" if panel_index == 0 else f"{labels[panel_index - 1]} @ {step}"
        draw.text((panel_index * panel.width + 8, 8), label, fill=(20, 20, 20))
    scatter_strip.save(outdir / f"scatter_step_{step:06d}.png")

    if toy_mode == "rank_drift":
        palette = [(217, 72, 65), (43, 140, 190), (49, 163, 84), (117, 107, 177)]
        stacked = []
        for rank in rank_levels:
            points = model(z_vis, torch.full([z_vis.shape[0]], rank, device=device)).detach().cpu().numpy()
            stacked.append(points)
        image = _render_path_panel(
            target=target,
            stacked=stacked,
            rank_levels=rank_levels,
            palette=palette,
            bounds=bounds,
        )
        image.save(outdir / f"path_step_{step:06d}.png")


def _render_scatter(points, color, bounds, size=512):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for x, y in _scale_points(points, size=size, bounds=bounds):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return image


def _render_path_panel(target, stacked, rank_levels, palette, bounds, size=512):
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    for x, y in _scale_points(target, size=size, bounds=bounds):
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(0, 0, 0))

    scaled_stacked = [_scale_points(points, size=size, bounds=bounds) for points in stacked]
    subset = min(32, len(scaled_stacked[0]))
    for point_index in range(subset):
        trajectory = [points[point_index] for points in scaled_stacked]
        draw.line(trajectory, fill=(120, 120, 120), width=1)
    for rank_index, points in enumerate(scaled_stacked):
        color = palette[rank_index % len(palette)]
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.text((8, 8 + 16 * rank_index), f"rank={rank_levels[rank_index]:g}", fill=color)
    return image


def _compute_bounds(point_sets):
    stacked = torch.cat([torch.as_tensor(points, dtype=torch.float32) for points in point_sets], dim=0)
    mins = stacked.min(dim=0).values
    maxs = stacked.max(dim=0).values
    return mins, maxs


def _scale_points(points, size, bounds):
    points = torch.as_tensor(points, dtype=torch.float32)
    mins, maxs = bounds
    span = torch.clamp(maxs - mins, min=1e-6)
    normalized = (points - mins) / span
    xs = 24 + normalized[:, 0] * float(size - 48)
    ys = 24 + (1.0 - normalized[:, 1]) * float(size - 48)
    return list(zip(xs.tolist(), ys.tolist()))


def _compute_rank_diagnostics(model, z_vis, rank_levels, device):
    with torch.no_grad():
        outputs = [
            model(z_vis, torch.full([z_vis.shape[0]], rank, device=device)).detach().cpu()
            for rank in rank_levels
        ]
    gaps = []
    for index in range(len(outputs) - 1):
        gaps.append(float((outputs[index] - outputs[index + 1]).norm(dim=-1).mean().item()))
    return {
        "mean_pairwise_rank_gap": float(sum(gaps) / len(gaps)) if gaps else 0.0,
        "pairwise_rank_gaps": gaps,
        "rank_levels_evaluated": [float(rank) for rank in rank_levels],
    }


if __name__ == "__main__":
    main()
