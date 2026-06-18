from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops_thop(model: nn.Module, input_size: Sequence[int]) -> int | None:
    try:
        from thop import profile
    except Exception:
        return None

    device = next(model.parameters()).device
    dummy = torch.randn(*input_size, device=device)
    was_training = model.training
    model.eval()
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return int(flops)
    except Exception:
        return None
    finally:
        model.train(was_training)


@torch.inference_mode()
def measure_cpu_latency(
    model: nn.Module,
    input_size: Sequence[int] = (1, 3, 32, 32),
    warmup: int = 30,
    iters: int = 100,
    cpu_threads: int = 1,
) -> Dict[str, float]:
    old_threads = torch.get_num_threads()
    if cpu_threads and cpu_threads > 0:
        torch.set_num_threads(cpu_threads)

    cpu_model = model.to("cpu")
    cpu_model.eval()
    dummy = torch.randn(*input_size, device="cpu")

    for _ in range(warmup):
        _ = cpu_model(dummy)

    times_ms: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = cpu_model(dummy)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    torch.set_num_threads(old_threads)
    return {
        "latency_ms_mean": float(np.mean(times_ms)),
        "latency_ms_std": float(np.std(times_ms, ddof=1)) if len(times_ms) > 1 else 0.0,
    }


def measure_model(model: nn.Module, cfg: Dict[str, Any]) -> Dict[str, Any]:
    input_size = cfg.get("input_size", [1, 3, 32, 32])
    warmup = int(cfg.get("latency_warmup", 30))
    iters = int(cfg.get("latency_iters", 100))
    cpu_threads = int(cfg.get("cpu_threads", 1))

    # FLOPs đo trên CPU để tránh chiếm GPU; kết quả FLOPs là lý thuyết.
    model_cpu = model.to("cpu")
    params = count_parameters(model_cpu)
    flops = estimate_flops_thop(model_cpu, input_size)
    latency = measure_cpu_latency(model_cpu, input_size=input_size, warmup=warmup, iters=iters, cpu_threads=cpu_threads)

    return {
        "params": params,
        "params_million": params / 1e6,
        "flops": flops,
        "flops_gmac": (flops / 1e9) if flops is not None else None,
        **latency,
    }
