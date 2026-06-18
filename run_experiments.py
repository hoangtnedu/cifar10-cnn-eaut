from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import load_yaml
from train import run_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all CIFAR-10 experiments from config.")
    parser.add_argument("--config", type=str, default="configs/cifar10_5seeds.yaml")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Optional model names to run.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Optional seeds to run.")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_names = args.models if args.models else cfg["models"]["names"]
    seeds = args.seeds if args.seeds else cfg["reproducibility"]["seeds"]

    print("=" * 80)
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Models: {model_names}")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    for model_name in model_names:
        for seed in seeds:
            run_one(args.config, model_name=model_name, seed=int(seed), resume_override=False if args.no_resume else None)

    print("\nAll requested experiments finished. Run aggregate_results.py to summarize.")


if __name__ == "__main__":
    main()
