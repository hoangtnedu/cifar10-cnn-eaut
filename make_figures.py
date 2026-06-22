from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - shown to users at runtime
    raise RuntimeError(
        "matplotlib is required to generate figures. Install it with: pip install matplotlib"
    ) from exc

from src.utils import ensure_dir, load_json, load_yaml


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PAPER_MODEL_NAMES = {
    "SimpleCNN": "SimpleCNN",
    "VGG11TinyBN": "VGG-11-Tiny-BN",
    "ResNet18CIFAR": "ResNet18-CIFAR",
    "MobileNetV2CIFAR": "MobileNetV2",
    "EfficientNetB0CIFAR": "EfficientNet-B0",
}

SUMMARY_METRICS = [
    "test_acc",
    "test_macro_f1",
    "params_million",
    "flops_gmac",
    "latency_ms_mean",
]


# Matplotlib defaults usually support Vietnamese characters through DejaVu Sans.
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"none", "nan", "null"}:
            return None
        # CSV files may contain strings such as "0.9123 ± 0.0010".
        if "±" in value:
            value = value.split("±", 1)[0].strip()
    try:
        out = float(value)
    except Exception:
        return None
    return None if math.isnan(out) else out


def safe_mean(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return mean(xs) if xs else None


def safe_std(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return stdev(xs) if len(xs) >= 2 else 0.0


def as_percent(value: float | None, reference: float | None = None) -> float | None:
    """Convert accuracy/F1 from ratio to percent when values look like [0, 1]."""
    if value is None:
        return None
    ref = value if reference is None else reference
    return value * 100.0 if abs(ref) <= 1.5 else value


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_mean_std(m: float | None, s: float | None, digits: int = 4, percent: bool = False) -> str:
    if m is None:
        return ""
    mm = as_percent(m) if percent else m
    ss = as_percent(s, reference=m) if percent else s
    if ss is None:
        return f"{mm:.{digits}f}"
    return f"{mm:.{digits}f} ± {ss:.{digits}f}"


def model_label(model_name: str) -> str:
    return PAPER_MODEL_NAMES.get(model_name, model_name)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---" for _ in headers]) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def resolve_output_root(cfg: Dict[str, Any], output_root_arg: str | None) -> Path:
    if output_root_arg:
        return Path(output_root_arg)
    exp_cfg = cfg["experiment"]
    return Path(exp_cfg.get("output_dir", "outputs")) / exp_cfg.get("name", "experiment")


def load_all_runs(output_root: Path, aggregate_dir: Path) -> List[Dict[str, Any]]:
    all_runs_csv = aggregate_dir / "all_runs.csv"
    if all_runs_csv.exists():
        return read_csv_rows(all_runs_csv)

    summaries = sorted(output_root.glob("*/seed_*/run_summary.json"))
    if not summaries:
        raise FileNotFoundError(
            f"No all_runs.csv or run_summary.json files found under: {output_root}\n"
            "Run training first, then run aggregate_results.py or run this script again."
        )
    return [load_json(p) for p in summaries]


def aggregate_summary_rows(all_runs: Sequence[Dict[str, Any]], model_names: Sequence[str]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for model in model_names:
        model_runs = [r for r in all_runs if str(r.get("model")) == model]
        row: Dict[str, Any] = {"model": model, "n_runs": len(model_runs)}
        for metric in SUMMARY_METRICS:
            values = [safe_float(r.get(metric)) for r in model_runs]
            m = safe_mean(values)
            s = safe_std(values)
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_mean_std"] = fmt_mean_std(m, s, digits=4)
        summary_rows.append(row)
    return summary_rows


def load_summary_rows(
    output_root: Path,
    aggregate_dir: Path,
    all_runs: Sequence[Dict[str, Any]],
    model_names: Sequence[str],
) -> List[Dict[str, Any]]:
    summary_csv = aggregate_dir / "summary_mean_std.csv"
    if not summary_csv.exists():
        return aggregate_summary_rows(all_runs, model_names)

    rows = read_csv_rows(summary_csv)
    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        row: Dict[str, Any] = dict(r)
        row["n_runs"] = int(safe_float(row.get("n_runs")) or 0)
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = safe_float(row.get(f"{metric}_mean"))
            row[f"{metric}_std"] = safe_float(row.get(f"{metric}_std"))
        cleaned.append(row)
    return cleaned


def sort_rows_by_config(rows: Sequence[Dict[str, Any]], model_names: Sequence[str]) -> List[Dict[str, Any]]:
    order = {name: i for i, name in enumerate(model_names)}
    return sorted(rows, key=lambda r: order.get(str(r.get("model")), 10_000))


def save_current_figure(path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_paper_tables(summary_rows: Sequence[Dict[str, Any]], tables_dir: Path) -> List[Path]:
    ensure_dir(tables_dir)
    outputs: List[Path] = []

    main_rows: List[Dict[str, Any]] = []
    main_md_rows: List[List[str]] = []
    for r in summary_rows:
        model = str(r["model"])
        row = {
            "Model": model_label(model),
            "n": str(r.get("n_runs", "")),
            "Accuracy (%)": fmt_mean_std(
                safe_float(r.get("test_acc_mean")), safe_float(r.get("test_acc_std")), digits=2, percent=True
            ),
            "F1-macro (%)": fmt_mean_std(
                safe_float(r.get("test_macro_f1_mean")), safe_float(r.get("test_macro_f1_std")), digits=2, percent=True
            ),
            "Params (M)": fmt_mean_std(
                safe_float(r.get("params_million_mean")), safe_float(r.get("params_million_std")), digits=3
            ),
            "FLOPs (GMac)": fmt_mean_std(
                safe_float(r.get("flops_gmac_mean")), safe_float(r.get("flops_gmac_std")), digits=3
            ),
            "CPU latency (ms/img)": fmt_mean_std(
                safe_float(r.get("latency_ms_mean_mean")), safe_float(r.get("latency_ms_mean_std")), digits=2
            ),
        }
        main_rows.append(row)
        main_md_rows.append([row[h] for h in row.keys()])

    csv_path = tables_dir / "table_1_main_results_for_paper.csv"
    md_path = tables_dir / "table_1_main_results_for_paper.md"
    write_csv(csv_path, main_rows)
    write_markdown_table(md_path, list(main_rows[0].keys()), main_md_rows)
    outputs.extend([csv_path, md_path])

    complexity_rows: List[Dict[str, Any]] = []
    complexity_md_rows: List[List[str]] = []
    for r in summary_rows:
        model = str(r["model"])
        row = {
            "Model": model_label(model),
            "Params (M)": fmt_mean_std(
                safe_float(r.get("params_million_mean")), safe_float(r.get("params_million_std")), digits=3
            ),
            "FLOPs (GMac)": fmt_mean_std(
                safe_float(r.get("flops_gmac_mean")), safe_float(r.get("flops_gmac_std")), digits=3
            ),
            "CPU latency (ms/img)": fmt_mean_std(
                safe_float(r.get("latency_ms_mean_mean")), safe_float(r.get("latency_ms_mean_std")), digits=2
            ),
        }
        complexity_rows.append(row)
        complexity_md_rows.append([row[h] for h in row.keys()])

    csv_path = tables_dir / "table_2_model_complexity_for_paper.csv"
    md_path = tables_dir / "table_2_model_complexity_for_paper.md"
    write_csv(csv_path, complexity_rows)
    write_markdown_table(md_path, list(complexity_rows[0].keys()), complexity_md_rows)
    outputs.extend([csv_path, md_path])

    return outputs


def get_metric_arrays(summary_rows: Sequence[Dict[str, Any]], metric: str, percent: bool = False):
    labels = [model_label(str(r["model"])) for r in summary_rows]
    means = [safe_float(r.get(f"{metric}_mean")) for r in summary_rows]
    stds = [safe_float(r.get(f"{metric}_std")) for r in summary_rows]
    if percent:
        means = [as_percent(v) for v in means]
        stds = [as_percent(s, reference=safe_float(r.get(f"{metric}_mean"))) for s, r in zip(stds, summary_rows)]
    return labels, np.array([np.nan if v is None else v for v in means]), np.array([0.0 if v is None else v for v in stds])


def plot_accuracy_f1_bars(summary_rows: Sequence[Dict[str, Any]], figures_dir: Path, dpi: int) -> Path:
    labels, acc_mean, acc_std = get_metric_arrays(summary_rows, "test_acc", percent=True)
    _, f1_mean, f1_std = get_metric_arrays(summary_rows, "test_macro_f1", percent=True)

    x = np.arange(len(labels))
    width = 0.38
    plt.figure(figsize=(10, 5.4))
    plt.bar(x - width / 2, acc_mean, width, yerr=acc_std, capsize=4, label="Accuracy")
    plt.bar(x + width / 2, f1_mean, width, yerr=f1_std, capsize=4, label="F1-macro")
    plt.ylabel("Giá trị trung bình (%)")
    plt.xlabel("Mô hình")
    plt.title("So sánh Accuracy và F1-macro trên CIFAR-10")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(max(0, np.nanmin([acc_mean, f1_mean]) - 5), min(100, np.nanmax([acc_mean, f1_mean]) + 3))
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    path = figures_dir / "fig_01_accuracy_f1_bar.png"
    save_current_figure(path, dpi)
    return path


def plot_single_metric_bar(
    summary_rows: Sequence[Dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    figures_dir: Path,
    dpi: int,
) -> Path:
    labels, means, stds = get_metric_arrays(summary_rows, metric, percent=False)
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5.0))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.ylabel(ylabel)
    plt.xlabel("Mô hình")
    plt.title(title)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    path = figures_dir / filename
    save_current_figure(path, dpi)
    return path


def plot_tradeoff_scatter(
    summary_rows: Sequence[Dict[str, Any]],
    x_metric: str,
    xlabel: str,
    title: str,
    filename: str,
    figures_dir: Path,
    dpi: int,
    log_x: bool = False,
) -> Path:
    labels = [model_label(str(r["model"])) for r in summary_rows]
    xs = np.array([safe_float(r.get(f"{x_metric}_mean")) or np.nan for r in summary_rows])
    ys = np.array([as_percent(safe_float(r.get("test_acc_mean"))) or np.nan for r in summary_rows])

    plt.figure(figsize=(7.6, 5.4))
    plt.scatter(xs, ys, s=70)
    for label, x, y in zip(labels, xs, ys):
        if not np.isnan(x) and not np.isnan(y):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy trung bình (%)")
    plt.title(title)
    if log_x and np.all(xs[np.isfinite(xs)] > 0):
        plt.xscale("log")
    plt.grid(alpha=0.25)
    path = figures_dir / filename
    save_current_figure(path, dpi)
    return path


def get_pareto_front(summary_rows: Sequence[Dict[str, Any]], cost_metric: str) -> List[Dict[str, Any]]:
    rows = [r for r in summary_rows if safe_float(r.get(f"{cost_metric}_mean")) is not None]
    rows = sorted(rows, key=lambda r: safe_float(r.get(f"{cost_metric}_mean")) or float("inf"))
    front: List[Dict[str, Any]] = []
    best_acc = -float("inf")
    for r in rows:
        acc = safe_float(r.get("test_acc_mean"))
        if acc is not None and acc > best_acc:
            front.append(r)
            best_acc = acc
    return front


def plot_pareto_latency(summary_rows: Sequence[Dict[str, Any]], figures_dir: Path, dpi: int) -> Path:
    labels = [model_label(str(r["model"])) for r in summary_rows]
    xs = np.array([safe_float(r.get("latency_ms_mean_mean")) or np.nan for r in summary_rows])
    ys = np.array([as_percent(safe_float(r.get("test_acc_mean"))) or np.nan for r in summary_rows])
    front = get_pareto_front(summary_rows, "latency_ms_mean")
    front_x = [safe_float(r.get("latency_ms_mean_mean")) for r in front]
    front_y = [as_percent(safe_float(r.get("test_acc_mean"))) for r in front]

    plt.figure(figsize=(7.6, 5.4))
    plt.scatter(xs, ys, s=70, label="Mô hình")
    if front_x and front_y:
        plt.plot(front_x, front_y, marker="o", label="Biên Pareto")
    for label, x, y in zip(labels, xs, ys):
        if not np.isnan(x) and not np.isnan(y):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xlabel("Độ trễ CPU trung bình (ms/ảnh)")
    plt.ylabel("Accuracy trung bình (%)")
    plt.title("Trade-off Accuracy và độ trễ suy luận CPU")
    plt.grid(alpha=0.25)
    plt.legend()
    path = figures_dir / "fig_08_pareto_accuracy_latency.png"
    save_current_figure(path, dpi)
    return path


def make_summary_figures(summary_rows: Sequence[Dict[str, Any]], figures_dir: Path, dpi: int) -> List[Path]:
    outputs = [plot_accuracy_f1_bars(summary_rows, figures_dir, dpi)]
    outputs.append(
        plot_single_metric_bar(
            summary_rows,
            metric="params_million",
            ylabel="Số tham số (triệu)",
            title="So sánh số tham số của các mô hình",
            filename="fig_02_params_bar.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
    )
    outputs.append(
        plot_single_metric_bar(
            summary_rows,
            metric="flops_gmac",
            ylabel="FLOPs (GMac)",
            title="So sánh FLOPs của các mô hình",
            filename="fig_03_flops_bar.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
    )
    outputs.append(
        plot_single_metric_bar(
            summary_rows,
            metric="latency_ms_mean",
            ylabel="Độ trễ CPU (ms/ảnh)",
            title="So sánh độ trễ suy luận trên CPU",
            filename="fig_04_latency_bar.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
    )
    outputs.append(
        plot_tradeoff_scatter(
            summary_rows,
            x_metric="params_million",
            xlabel="Số tham số (triệu)",
            title="Accuracy theo số tham số",
            filename="fig_05_accuracy_vs_params.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
    )
    outputs.append(
        plot_tradeoff_scatter(
            summary_rows,
            x_metric="flops_gmac",
            xlabel="FLOPs (GMac)",
            title="Accuracy theo FLOPs",
            filename="fig_06_accuracy_vs_flops.png",
            figures_dir=figures_dir,
            dpi=dpi,
            log_x=True,
        )
    )
    outputs.append(
        plot_tradeoff_scatter(
            summary_rows,
            x_metric="latency_ms_mean",
            xlabel="Độ trễ CPU trung bình (ms/ảnh)",
            title="Accuracy theo độ trễ suy luận CPU",
            filename="fig_07_accuracy_vs_latency.png",
            figures_dir=figures_dir,
            dpi=dpi,
        )
    )
    outputs.append(plot_pareto_latency(summary_rows, figures_dir, dpi))
    return outputs


def load_history_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_csv_rows(path)
    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        row: Dict[str, Any] = {"epoch": int(safe_float(r.get("epoch")) or 0)}
        for key in ["train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1"]:
            row[key] = safe_float(r.get(key))
        if row["epoch"] > 0:
            cleaned.append(row)
    return cleaned


def aggregate_histories(histories: Sequence[List[Dict[str, Any]]]) -> Dict[int, Dict[str, float]]:
    by_epoch: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for history in histories:
        for row in history:
            epoch = int(row["epoch"])
            for key, value in row.items():
                if key == "epoch" or value is None:
                    continue
                by_epoch[epoch][key].append(float(value))

    out: Dict[int, Dict[str, float]] = {}
    for epoch in sorted(by_epoch):
        out[epoch] = {key: mean(values) for key, values in by_epoch[epoch].items() if values}
    return out


def plot_model_learning_curves(model: str, aggregated: Dict[int, Dict[str, float]], figures_dir: Path, dpi: int) -> List[Path]:
    if not aggregated:
        return []
    epochs = np.array(sorted(aggregated.keys()))
    outputs: List[Path] = []

    train_acc = np.array([as_percent(aggregated[e].get("train_acc")) or np.nan for e in epochs])
    val_acc = np.array([as_percent(aggregated[e].get("val_acc")) or np.nan for e in epochs])
    if np.isfinite(train_acc).any() or np.isfinite(val_acc).any():
        plt.figure(figsize=(8.2, 5.0))
        plt.plot(epochs, train_acc, label="Train Accuracy")
        plt.plot(epochs, val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"Đường cong Accuracy - {model_label(model)}")
        plt.grid(alpha=0.25)
        plt.legend()
        path = figures_dir / f"fig_learning_accuracy_{safe_filename(model)}.png"
        save_current_figure(path, dpi)
        outputs.append(path)

    train_loss = np.array([aggregated[e].get("train_loss", np.nan) for e in epochs], dtype=float)
    val_loss = np.array([aggregated[e].get("val_loss", np.nan) for e in epochs], dtype=float)
    if np.isfinite(train_loss).any() or np.isfinite(val_loss).any():
        plt.figure(figsize=(8.2, 5.0))
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Đường cong Loss - {model_label(model)}")
        plt.grid(alpha=0.25)
        plt.legend()
        path = figures_dir / f"fig_learning_loss_{safe_filename(model)}.png"
        save_current_figure(path, dpi)
        outputs.append(path)

    return outputs


def make_learning_curve_figures(output_root: Path, model_names: Sequence[str], figures_dir: Path, dpi: int) -> List[Path]:
    outputs: List[Path] = []
    combined: Dict[str, Dict[int, Dict[str, float]]] = {}

    for model in model_names:
        history_paths = sorted((output_root / model).glob("seed_*/history.csv"))
        histories = [load_history_rows(p) for p in history_paths if p.exists()]
        histories = [h for h in histories if h]
        if not histories:
            print(f"[WARN] No history.csv files found for model={model}; skipping learning curves.")
            continue
        aggregated = aggregate_histories(histories)
        combined[model] = aggregated
        outputs.extend(plot_model_learning_curves(model, aggregated, figures_dir, dpi))

    if combined:
        plt.figure(figsize=(9.2, 5.6))
        for model, aggregated in combined.items():
            epochs = np.array(sorted(aggregated.keys()))
            val_acc = np.array([as_percent(aggregated[e].get("val_acc")) or np.nan for e in epochs])
            plt.plot(epochs, val_acc, label=model_label(model))
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy trung bình (%)")
        plt.title("So sánh đường cong Validation Accuracy của các mô hình")
        plt.grid(alpha=0.25)
        plt.legend()
        path = figures_dir / "fig_09_validation_accuracy_curves_all_models.png"
        save_current_figure(path, dpi)
        outputs.append(path)

    return outputs


def write_confusion_matrix_csv(cm: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *CIFAR10_CLASSES])
        for label, row in zip(CIFAR10_CLASSES, cm):
            writer.writerow([label, *row.tolist()])


def plot_confusion_matrix(
    cm: np.ndarray,
    model: str,
    seed: int,
    figures_dir: Path,
    dpi: int,
    normalize: bool = True,
) -> Path:
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        display = cm.astype(float) / denom * 100.0
        value_fmt = ".1f"
        title_suffix = "chuẩn hóa theo lớp thật (%)"
        filename = f"fig_confusion_matrix_normalized_{safe_filename(model)}_seed_{seed}.png"
    else:
        display = cm.astype(float)
        value_fmt = ".0f"
        title_suffix = "số mẫu"
        filename = f"fig_confusion_matrix_raw_{safe_filename(model)}_seed_{seed}.png"

    plt.figure(figsize=(8.0, 6.8))
    ax = plt.gca()
    im = ax.imshow(display, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_yticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Lớp dự đoán")
    ax.set_ylabel("Lớp thật")
    ax.set_title(f"Ma trận nhầm lẫn - {model_label(model)} - seed {seed} ({title_suffix})")

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            ax.text(j, i, format(display[i, j], value_fmt), ha="center", va="center", fontsize=7)

    path = figures_dir / filename
    save_current_figure(path, dpi)
    return path


def choose_best_runs(all_runs: Sequence[Dict[str, Any]], model_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for model in model_names:
        candidates = [r for r in all_runs if str(r.get("model")) == model and safe_float(r.get("test_acc")) is not None]
        if candidates:
            out[model] = max(candidates, key=lambda r: safe_float(r.get("test_acc")) or -1.0)
    return out


def compute_confusion_matrix_for_run(cfg: Dict[str, Any], output_root: Path, model: str, seed: int, device_name: str | None) -> np.ndarray:
    import torch

    from src.data import get_cifar10_loaders
    from src.metrics import confusion_matrix_np
    from src.models import create_model
    from src.utils import load_checkpoint

    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    num_classes = int(cfg.get("models", {}).get("num_classes", 10))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))

    _, _, test_loader = get_cifar10_loaders(
        data_dir=exp_cfg.get("data_dir", "data"),
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=int(train_cfg.get("num_workers", 2)),
        seed=seed,
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
    )
    model_obj = create_model(model, num_classes=num_classes).to(device)
    ckpt_path = output_root / model / f"seed_{seed}" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        ckpt_path = output_root / model / f"seed_{seed}" / "checkpoint_last.pt"
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model_obj.load_state_dict(ckpt["model_state"])
    model_obj.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model_obj(images)
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.numpy().tolist())

    return confusion_matrix_np(np.asarray(y_true), np.asarray(y_pred), num_classes=num_classes)


def make_confusion_matrix_figures(
    cfg: Dict[str, Any],
    output_root: Path,
    all_runs: Sequence[Dict[str, Any]],
    model_names: Sequence[str],
    figures_dir: Path,
    tables_dir: Path,
    dpi: int,
    device_name: str | None,
) -> List[Path]:
    outputs: List[Path] = []
    best_runs = choose_best_runs(all_runs, model_names)
    if not best_runs:
        print("[WARN] No completed runs found for confusion matrices.")
        return outputs

    for model, run in best_runs.items():
        seed_value = safe_float(run.get("seed"))
        if seed_value is None:
            print(f"[WARN] Cannot infer seed for model={model}; skipping confusion matrix.")
            continue
        seed = int(seed_value)
        try:
            print(f"Computing confusion matrix for model={model}, seed={seed}...")
            cm = compute_confusion_matrix_for_run(cfg, output_root, model, seed, device_name)
        except Exception as exc:
            print(f"[WARN] Failed to compute confusion matrix for {model}, seed={seed}: {exc}")
            continue

        csv_path = tables_dir / f"confusion_matrix_{safe_filename(model)}_seed_{seed}.csv"
        write_confusion_matrix_csv(cm, csv_path)
        outputs.append(csv_path)
        outputs.append(plot_confusion_matrix(cm, model, seed, figures_dir, dpi, normalize=True))
        outputs.append(plot_confusion_matrix(cm, model, seed, figures_dir, dpi, normalize=False))

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready figures and tables from CIFAR-10 CNN experiment outputs. "
            "Run aggregate_results.py first for the cleanest tables."
        )
    )
    parser.add_argument("--config", type=str, default="configs/cifar10_5seeds_colab_drive.yaml")
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional direct path to the experiment output root, e.g. /content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds",
    )
    parser.add_argument("--figures-dir", type=str, default=None, help="Optional output directory for PNG figures.")
    parser.add_argument("--tables-dir", type=str, default=None, help="Optional output directory for CSV/Markdown tables.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--skip-confusion-matrix",
        action="store_true",
        help="Skip confusion matrix generation from checkpoints. Use this if you only need aggregate plots quickly.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for confusion-matrix inference, e.g. cuda, cpu. Default: cuda if available else cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    model_names = list(cfg["models"]["names"])
    output_root = resolve_output_root(cfg, args.output_root)
    aggregate_dir = output_root / "aggregate"
    figures_dir = ensure_dir(Path(args.figures_dir) if args.figures_dir else output_root / "figures")
    tables_dir = ensure_dir(Path(args.tables_dir) if args.tables_dir else output_root / "paper_tables")

    all_runs = load_all_runs(output_root, aggregate_dir)
    summary_rows = load_summary_rows(output_root, aggregate_dir, all_runs, model_names)
    summary_rows = sort_rows_by_config(summary_rows, model_names)

    print(f"Output root: {output_root}")
    print(f"Figures dir: {figures_dir}")
    print(f"Tables dir: {tables_dir}")

    outputs: List[Path] = []
    outputs.extend(make_paper_tables(summary_rows, tables_dir))
    outputs.extend(make_summary_figures(summary_rows, figures_dir, args.dpi))
    outputs.extend(make_learning_curve_figures(output_root, model_names, figures_dir, args.dpi))

    if args.skip_confusion_matrix:
        print("Skipping confusion matrices (--skip-confusion-matrix).")
    else:
        outputs.extend(
            make_confusion_matrix_figures(
                cfg=cfg,
                output_root=output_root,
                all_runs=all_runs,
                model_names=model_names,
                figures_dir=figures_dir,
                tables_dir=tables_dir,
                dpi=args.dpi,
                device_name=args.device,
            )
        )

    print("\nGenerated files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
