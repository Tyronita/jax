#!/usr/bin/env python3
"""ShinkaEvolve Benchmark — Target 05: blas_kernels_ffi.h

GEMM benchmarks at various sizes — the most important performance metric in ML.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax, jax.numpy as jnp
from shinka_test_framework import ShinkaBenchmark
import argparse


def bench_blas(parsed_args):
    target = "jaxlib/cpu/blas_kernels_ffi.h"
    benchmark = ShinkaBenchmark(
        target=target,
        operation="gemm",
        mutation_id=parsed_args.mutation_id,
    )
    
    key = jax.random.PRNGKey(42)
    
    for n in parsed_args.n:
        for dtype_str in parsed_args.dtype:
            dtype_map = {"f32": jnp.float32, "f64": jnp.float64}
            dtype = dtype_map[dtype_str]
            
            # Square GEMM
            key, sk = jax.random.split(key)
            a = jax.random.normal(sk, (n, n), dtype=dtype)
            key, sk = jax.random.split(key)
            b = jax.random.normal(sk, (n, n), dtype=dtype)
            
            def gemm_fn():
                return jnp.dot(a, b)
            
            print(f"[bench] GEMM n={n} dtype={dtype_str} ...")
            benchmark.measure(gemm_fn, n, dtype_str, parsed_args.platform,
                            parsed_args.reps, parsed_args.warmup,
                            tags=["gemm", "blas", "square"])
    
    benchmark.print_summary()
    benchmark.save(parsed_args.output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument("--dtype", nargs="+", default=["f32", "f64"])
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--platform", default="cpu")
    p.add_argument("--output", default="results/target_05_baseline.json")
    p.add_argument("--mutation-id", default=None)
    bench_blas(p.parse_args())
