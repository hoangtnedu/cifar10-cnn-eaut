from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from src.data import get_cifar10_loaders
from src.engine import evaluate, train_one_epoch
from src.measure import measure_model
from src.models import create_model
from src.utils import (
    ensure_dir,
    get_device,
    load_checkpoint,
    load_yaml,
    save_checkpoint,
    save_json,
    set_seed,
)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "SGD")).lower()
    lr = float(cfg.get("lr", 0.1))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(cfg.get("nesterov", False)),
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    name = str(cfg.get("scheduler", "CosineAnnealingLR")).lower()
    epochs = int(cfg.get("epochs", 100))
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name in {"none", "null", ""}:
        return None
    raise ValueError(f"Unsupported scheduler: {name}")


def append_history(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_one(config_path: str | Path, model_name: str, seed: int, resume_override: bool | None = None) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    exp_cfg = cfg["experiment"]
    train_cfg = cfg["training"]
    repro_cfg = cfg.get("reproducibility", {})
    measure_cfg = cfg.get("measurement", {})
    num_classes = int(cfg.get("models", {}).get("num_classes", 10))

    set_seed(
        seed,
        deterministic=bool(repro_cfg.get("deterministic", False)),
        benchmark=bool(repro_cfg.get("benchmark", True)),
    )

    device = get_device()
    exp_name = exp_cfg.get("name", "experiment")
    output_root = Path(exp_cfg.get("output_dir", "outputs")) / exp_name
    run_dir = ensure_dir(output_root / model_name / f"seed_{seed}")
    last_ckpt_path = run_dir / "checkpoint_last.pt"
    best_ckpt_path = run_dir / "checkpoint_best.pt"
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "run_summary.json"

    resume = bool(exp_cfg.get("resume", True)) if resume_override is None else resume_override

    print(f"\n=== Running model={model_name}, seed={seed}, device={device}, resume={resume} ===")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=exp_cfg.get("data_dir", "data"),
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=int(train_cfg.get("num_workers", 2)),
        seed=seed,
        val_ratio=float(train_cfg.get("val_ratio", 0.1)),
    )

    model = create_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    start_epoch = 1
    best_val_acc = -1.0
    best_epoch = 0

    if resume and last_ckpt_path.exists():
        print(f"Resuming from: {last_ckpt_path}")
        ckpt = load_checkpoint(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        best_epoch = int(ckpt.get("best_epoch", best_epoch))

    epochs = int(train_cfg.get("epochs", 100))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if start_epoch > epochs:
        print(f"Run already completed: epoch {start_epoch - 1}/{epochs}")
    else:
        for epoch in range(start_epoch, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp)
            val_metrics = evaluate(model, val_loader, criterion, device, num_classes=num_classes, desc="val")

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                scheduler.step()

            improved = val_metrics["acc"] > best_val_acc
            if improved:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_name": model_name,
                        "seed": seed,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "best_val_acc": best_val_acc,
                        "best_epoch": best_epoch,
                        "config": cfg,
                    },
                    best_ckpt_path,
                )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "seed": seed,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "config": cfg,
                },
                last_ckpt_path,
            )

            row = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
            }
            append_history(history_path, row)

            if epoch % int(exp_cfg.get("print_every", 1)) == 0:
                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"train_acc={train_metrics['acc']:.4f} | "
                    f"val_acc={val_metrics['acc']:.4f} | "
                    f"val_f1={val_metrics['macro_f1']:.4f} | "
                    f"best={best_val_acc:.4f}@{best_epoch}"
                )

    # Đánh giá test bằng checkpoint tốt nhất.
    if best_ckpt_path.exists():
        ckpt = load_checkpoint(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_epoch = int(ckpt.get("best_epoch", best_epoch))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))

    test_metrics = evaluate(model, test_loader, criterion, device, num_classes=num_classes, desc="test")

    measurements: Dict[str, Any] = {}
    if bool(measure_cfg.get("enabled", True)):
        measurements = measure_model(model, measure_cfg)

    summary = {
        "experiment": exp_name,
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "device": str(device),
        "epochs": epochs,
        "batch_size": int(train_cfg.get("batch_size", 128)),
        "optimizer": train_cfg.get("optimizer", "SGD"),
        "lr": float(train_cfg.get("lr", 0.1)),
        "scheduler": train_cfg.get("scheduler", "CosineAnnealingLR"),
        "label_smoothing": float(train_cfg.get("label_smoothing", 0.0)),
        **measurements,
    }
    save_json(summary, summary_path)
    print(f"Saved summary: {summary_path}")
    print(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one CIFAR-10 CNN model for one seed.")
    parser.add_argument("--config", type=str, default="configs/cifar10_5seeds.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_one(args.config, args.model, args.seed, resume_override=False if args.no_resume else None)


if __name__ == "__main__":
    main()
