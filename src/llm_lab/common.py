"""跨模块共用的测量原语。

设计原则：测量代码不参与被测量的逻辑。
所有计时都用 CUDA event（而不是 time.time），因为 kernel 是异步的，
用墙钟会把 launch 开销算进计算时间，在小 kernel 上误差可达数倍。
"""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

# ---------------------------------------------------------------- 环境

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def device_info() -> Dict[str, Any]:
    """写进每份 results/*.json 的环境指纹。

    没有这一段的实验数字是不可复现的——同一段代码在 4050 和 A100 上
    可以差 20 倍，报告里必须能追溯到底跑在哪张卡上。
    """
    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_built": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": DEVICE,
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # pragma: no cover - triton 缺失时不影响别的实验
        info["triton"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": props.name,
                "sm": f"{props.major}.{props.minor}",
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "sm_count": props.multi_processor_count,
            }
        )
    return info


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 计时


@contextmanager
def cuda_timer():
    """用 CUDA event 包一段同步区间，yield 一个把耗时写进 .elapsed_ms 的对象。"""

    class _T:
        elapsed_ms: float = 0.0

    t = _T()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        yield t
        end.record()
        torch.cuda.synchronize()
        t.elapsed_ms = start.elapsed_time(end)
    else:
        s = time.perf_counter()
        yield t
        t.elapsed_ms = (time.perf_counter() - s) * 1000.0


def bench(fn: Callable[[], Any], warmup: int = 10, rep: int = 50) -> Dict[str, float]:
    """返回 median / mean / min / p90 毫秒。

    用 median 而不是 mean：GPU 上偶尔会有别的进程抢 SM，
    均值会被个别离群点拉高，median 更接近"稳定态"。
    """
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(rep):
        with cuda_timer() as t:
            fn()
        samples.append(t.elapsed_ms)

    samples_sorted = sorted(samples)
    p90_idx = min(len(samples_sorted) - 1, int(0.9 * len(samples_sorted)))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": samples_sorted[0],
        "p90_ms": samples_sorted[p90_idx],
        "rep": rep,
    }


def peak_memory_mb() -> float:
    """自进程启动以来的显存峰值（MB）。"""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


# ---------------------------------------------------------------- 结果落盘

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def save_result(name: str, payload: Dict[str, Any]) -> str:
    """把一次实验的原始数据写成 results/<name>.json。

    这些 json 是要提交进 git 的：它们就是"这些图是我自己跑出来的"的证据链。
    每份都带 device_info 和 git sha，别人 checkout 后能对上。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        "experiment": name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "env": device_info(),
        "data": payload,
    }
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    return path


# ---------------------------------------------------------------- 格式化


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} PB"


def print_table(rows: Iterable[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    """等宽文本表，用于终端里直接看结果（不需要 pandas）。"""
    rows = list(rows)
    if not rows:
        print("(empty)")
        return
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(_fmt(r.get(c))) for r in rows)) for c in columns}
    header = "  ".join(str(c).ljust(widths[c]) for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}"
    return str(v)