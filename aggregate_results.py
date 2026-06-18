from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

from src.utils import ensure_dir, load_yaml


METRICS = [
    "test_acc",
    "test_macro_f1",
    "params_million",
    "flops_gmac",
    "latency_ms_mean",
]


def read_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: List[float]) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def safe_std(values: List[float]) -> float | None:
    values = [v for v in values if v is not None]
    return stdev(values) if len(values) >= 2 else 0.0 if len(values) == 1 else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_mean_std(m: float | None, s: float | None, digits: int = 4) -> str:
    if m is None:
        return ""
    return f"{m:.{digits}f} ± {s:.{digits}f}" if s is not None else f"{m:.{digits}f}"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results into mean ± std tables.")
    parser.add_argument("--config", type=str, default="configs/cifar10_5seeds.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    exp_name = cfg["experiment"]["name"]
    output_root = Path(cfg["experiment"].get("output_dir", "outputs")) / exp_name
    aggregate_dir = ensure_dir(output_root / "aggregate")

    summaries = sorted(output_root.glob("*/seed_*/run_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No run_summary.json files found under {output_root}")

    all_runs = [read_summary(p) for p in summaries]
    all_runs_csv = aggregate_dir / "all_runs.csv"
    write_csv(all_runs_csv, all_runs)

    model_names = cfg["models"]["names"]
    summary_rows: List[Dict[str, Any]] = []
    for model in model_names:
        model_runs = [r for r in all_runs if r.get("model") == model]
        row: Dict[str, Any] = {"model": model, "n_runs": len(model_runs)}
        for metric in METRICS:
            values = [r.get(metric) for r in model_runs if r.get(metric) is not None]
            values = [float(v) for v in values]
            m = safe_mean(values)
            s = safe_std(values)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_mean_std"] = fmt_mean_std(m, s, digits=4)
        summary_rows.append(row)

    summary_csv = aggregate_dir / "summary_mean_std.csv"
    write_csv(summary_csv, summary_rows)

    md_path = aggregate_dir / "summary_markdown.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Summary: mean ± std\n\n")
        f.write("| Model | n | Accuracy | F1-macro | Params (M) | FLOPs (GMac) | CPU latency (ms/img) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in summary_rows:
            f.write(
                f"| {r['model']} | {r['n_runs']} | "
                f"{r['test_acc_mean_std']} | "
                f"{r['test_macro_f1_mean_std']} | "
                f"{r['params_million_mean_std']} | "
                f"{r['flops_gmac_mean_std']} | "
                f"{r['latency_ms_mean_mean_std']} |\n"
            )

    print(f"Saved all runs: {all_runs_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved markdown: {md_path}")
    print("\nPreview:")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
