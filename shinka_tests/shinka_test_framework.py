"""Shinka Test Framework — timing, logging, profiling for JAX C++ kernel mutations.

Usage:
    python -m pytest shinka_tests/test_target_01.py -v --shinka-benchmark
    python shinka_tests/bench_target_01.py --n 32 64 128 --dtype f32 f64 --reps 300

Returns JSON output with μs precision, GC pressure, and cross-platform metrics.
"""

import json
import sys
import time
import os
import platform
import psutil
import gc
from dataclasses import dataclass, asdict
from typing import Any, Callable, List, Optional, Dict
from contextlib import contextmanager
import numpy as np


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""
    target: str                      # e.g., "jaxlib/tridiagonal_solve_perturbed.h"
    operation: str                   # e.g., "tridiagonal_solve"
    n: int
    dtype_str: str
    platform: str                    # cpu, gpu, tpu
    device_name: str
    mean_us: Optional[float]
    std_us: Optional[float]
    min_us: Optional[float]
    max_us: Optional[float]
    median_us: Optional[float]
    repetitions: int
    warmup_reps: int
    compile_time_us: Optional[float]
    gc_collections: int
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    os_info: str
    cpu_info: str
    timestamp: str
    mutation_id: Optional[str]      # None for baseline, "gen-68" etc for mutant
    correctness_passed: Optional[bool]
    build_time_s: Optional[float]
    tags: List[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class SystemInfo:
    """Capture system state for background quality metrics."""
    os: str
    arch: str
    cpu_count: int
    cpu_freq_mhz: Optional[float]
    total_ram_gb: float
    python_version: str
    jax_version: str
    jaxlib_version: str
    gpu_name: Optional[str]
    gpu_memory_gb: Optional[float]
    cuda_version: Optional[str]
    openblas_version: Optional[str]
    mkl_version: Optional[str]
    compiler: Optional[str]
    build_flags: Optional[str]

    @staticmethod
    def capture() -> "SystemInfo":
        import jax
        gpu_name = None
        gpu_mem = None
        cuda_ver = None
        try:
            for dev in jax.devices():
                if dev.platform == "gpu":
                    gpu_name = dev.device_kind
                    # Try to get memory via nvidia-smi or similar later
                    break
        except Exception:
            pass
        
        return SystemInfo(
            os=f"{platform.system()} {platform.release()}",
            arch=platform.machine(),
            cpu_count=os.cpu_count() or 0,
            cpu_freq_mhz=psutil.cpu_freq().current if psutil.cpu_freq() else None,
            total_ram_gb=round(psutil.virtual_memory().total / (1024**3), 2),
            python_version=platform.python_version(),
            jax_version=getattr(jax, "__version__", "unknown"),
            jaxlib_version=getattr(jax, "_jaxlib_version", getattr(jax, "__version__", "unknown")),
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_mem,
            cuda_version=cuda_ver,
            openblas_version=None,  # Can be fetched from scipy or env
            mkl_version=None,
            compiler="clang" if os.system("which clang 2>/dev/null") == 0 else "gcc",
            build_flags=None,
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class ShinkaBenchmark:
    """Benchmark harness with μs precision, GC control, and memory tracking."""
    
    def __init__(self, target: str, operation: str, mutation_id: Optional[str] = None):
        self.target = target
        self.operation = operation
        self.mutation_id = mutation_id
        self.results: List[BenchmarkResult] = []
        self.system_info = SystemInfo.capture()
        
    @contextmanager
    def _gc_control(self):
        """Disable GC during benchmark, record collections."""
        gc_old = gc.isenabled()
        gc_collections = gc.get_count()
        if gc_old:
            gc.disable()
        try:
            yield
        finally:
            new_collections = gc.get_count()
            if gc_old:
                gc.enable()
            self._gc_collections = sum(new_collections) - sum(gc_collections)
    
    def measure(
        self,
        fn: Callable,
        n: int,
        dtype_str: str,
        platform: str = "cpu",
        reps: int = 300,
        warmup: int = 10,
        tags: Optional[List[str]] = None,
    ) -> BenchmarkResult:
        """Measure fn() with μs precision."""
        import jax
        
        # Force platform
        if platform == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        elif platform == "gpu":
            jax.config.update("jax_platform_name", "gpu")
        
        device = jax.devices(platform)[0]
        device_name = device.device_kind if hasattr(device, "device_kind") else str(device)
        
        # Memory tracking
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        mem_peak = mem_before
        
        # Warmup
        for _ in range(warmup):
            out = fn()
            jax.block_until_ready(out)
        
        # Compile timing (first call after JIT)
        jf = jax.jit(fn)
        t_compile0 = time.perf_counter()
        _ = jax.block_until_ready(jf())
        t_compile1 = time.perf_counter()
        compile_time_us = round((t_compile1 - t_compile0) * 1e6, 2)
        
        # Actual measurement
        times: List[float] = []
        with self._gc_control():
            for _ in range(reps):
                t0 = time.perf_counter()
                out = jax.block_until_ready(jf())
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1e6)  # microseconds
                # Track memory every 10 reps
                if _ % 10 == 0:
                    mem_current = process.memory_info().rss / (1024 * 1022)
                    mem_peak = max(mem_peak, mem_current)
        
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        arr = np.array(times)
        result = BenchmarkResult(
            target=self.target,
            operation=self.operation,
            n=n,
            dtype_str=dtype_str,
            platform=platform,
            device_name=device_name,
            mean_us=round(float(np.mean(arr)), 2),
            std_us=round(float(np.std(arr)), 2),
            min_us=round(float(np.min(arr)), 2),
            max_us=round(float(np.max(arr)), 2),
            median_us=round(float(np.median(arr)), 2),
            repetitions=reps,
            warmup_reps=warmup,
            compile_time_us=compile_time_us,
            gc_collections=getattr(self, "_gc_collections", 0),
            memory_before_mb=round(mem_before, 2),
            memory_after_mb=round(mem_after, 2),
            memory_peak_mb=round(mem_peak, 2),
            os_info=self.system_info.os,
            cpu_info=f"{self.system_info.cpu_count} cores @ {self.system_info.cpu_freq_mhz or 'unknown'} MHz",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            mutation_id=self.mutation_id,
            correctness_passed=None,  # Set by caller after pytest
            build_time_s=None,
            tags=tags or [],
        )
        
        self.results.append(result)
        return result
    
    def save(self, filepath: str):
        """Save all results to JSON."""
        data = {
            "system_info": asdict(self.system_info),
            "benchmarks": [asdict(r) for r in self.results],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[shinka] Saved {len(self.results)} results to {filepath}")
    
    def print_summary(self):
        """Pretty-print summary table."""
        print("\n" + "=" * 80)
        print(f"Shinka Benchmark: {self.operation} @ {self.target}")
        print(f"Mutation: {self.mutation_id or 'BASELINE'}")
        print(f"System: {self.system_info.os} | {self.system_info.cpu_count} cores | RAM {self.system_info.total_ram_gb} GB")
        print("=" * 80)
        print(f"{'Operation':<25} {'n':>6} {'dtype':>6} {'Mean μs':>10} {'Std μs':>8} {'Min μs':>8} {'Mem Δ MB':>10}")
        print("-" * 80)
        for r in self.results:
            mem_delta = r.memory_after_mb - r.memory_before_mb
            print(f"{r.operation:<25} {r.n:>6} {r.dtype_str:>6} {r.mean_us or '-':>10} {r.std_us or '-':>8} {r.min_us or '-':>8} {mem_delta:>+10.2f}")
        print("=" * 80)


def shinka_main(benchmark_fn, args=None):
    """CLI entry point for benchmark scripts."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Shinka JAX Kernel Benchmark")
    parser.add_argument("--n", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024],
                        help="Matrix sizes to benchmark")
    parser.add_argument("--dtype", nargs="+", default=["f32", "f64", "c64", "c128"],
                        help="Dtypes to test")
    parser.add_argument("--reps", type=int, default=300,
                        help="Repetitions per measurement")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup repetitions")
    parser.add_argument("--platform", default="cpu", choices=["cpu", "gpu", "tpu"],
                        help="JAX platform")
    parser.add_argument("--output", default="shinka_bench_results.json",
                        help="Output JSON path")
    parser.add_argument("--mutation-id", default=None,
                        help="Mutation identifier (e.g., gen-68)")
    
    parsed = parser.parse_args(args)
    benchmark_fn(parsed)
