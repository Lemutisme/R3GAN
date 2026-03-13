#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_root_on_path() -> None:
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze a drift or GAN snapshot for collapse indicators.')
    parser.add_argument('snapshot', type=str, help='Path to a network snapshot pickle')
    parser.add_argument('--device', type=str, default='cpu', help='Torch device used for diagnostic sampling')
    parser.add_argument('--batch-size', type=int, default=8, help='Sampling batch size for diagnostics')
    parser.add_argument('--eval-alpha', type=float, default=None, help='Override eval alpha used for sampling')
    parser.add_argument('--alpha-low', type=float, default=1.0, help='Low alpha used for alpha sensitivity')
    parser.add_argument('--alpha-high', type=float, default=4.0, help='High alpha used for alpha sensitivity')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for deterministic diagnostics')
    parser.add_argument('--use-g', action='store_true', help='Use G instead of G_ema from the snapshot')
    parser.add_argument('--output-json', type=str, default=None, help='Optional path to write the diagnostic JSON')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_repo_root_on_path()

    from training.drift_diagnostics import analyze_snapshot

    payload = analyze_snapshot(
        snapshot_path=args.snapshot,
        device=args.device,
        batch_size=int(args.batch_size),
        eval_alpha=args.eval_alpha,
        alpha_pair=(float(args.alpha_low), float(args.alpha_high)),
        seed=int(args.seed),
        use_ema=not bool(args.use_g),
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
