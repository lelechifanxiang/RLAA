from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import pytest


TResult = TypeVar("TResult")


@dataclass(slots=True)
class BenchmarkResult:
    avg_ms: float
    last_result: TResult


@pytest.fixture
def benchmark_runner():
    """提供轻量 benchmark runner，未安装 pytest-benchmark 时仍可运行性能烟雾测试。"""

    def run_benchmark(
        func: Callable[[], TResult],
        *,
        warmup: int = 0,
        iterations: int = 1,
    ) -> BenchmarkResult:
        for _ in range(warmup):
            func()

        started_at = time.perf_counter()
        last_result = None
        for _ in range(iterations):
            last_result = func()
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        return BenchmarkResult(
            avg_ms=elapsed_ms / max(iterations, 1),
            last_result=last_result,
        )

    return run_benchmark
