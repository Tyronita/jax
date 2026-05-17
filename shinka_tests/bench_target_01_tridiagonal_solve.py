#!/usr/bin/env python3
"""ShinkaEvolve Benchmark — Target 01: tridiagonal_solve_perturbed.h

Extensive μs-precision benchmarks with background quality metrics logging.

Usage:
    python bench_target_01_tridiagonal_solve.py \
        --n 32 64 128 256 512 1024 2048 \
        --dtype f32 f64 c64 c128 \
        --reps 300 \
        --output results/target_01_baseline.json

Produces JSON with:
    - per-size, per-dtype μs timing (mean, std, min, max, median)
    - compile time (JIT first call)
    - memory delta (before/after/peak in MB)
    - GC collections during measurement
    - system info (CPU, RAM, OS, compiler)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from shinka_test_framework import ShinkaBenchmark, shinka_main


def bench_tridiagonal_solve(parsed_args):
    """Benchmark tridiagonal solve for each (n, dtype, platform)."""
    
    target = "jaxlib/tridiagonal_solve_perturbed.h"
    operation = "tridiagonal_solve"
    mutation_id = parsed_args.mutation_id
    
    benchmark = ShinkaBenchmark(
        target=target,
        operation=operation,
        mutation_id=mutation_id,
    )
    
    key = jax.random.PRNGKey(42)
    
    for n in parsed_args.n:
        for dtype_str in parsed_args.dtype:
            dtype_map = {
                "f32": jnp.float32,
                "f64": jnp.float64,
                "c64": jnp.complex64,
                "c128": jnp.complex128,
            }
            dtype = dtype_map[dtype_str]
            
            # Generate test data
            key, subkey = jax.random.split(key)
            dl = jax.random.uniform(subkey, (n,), dtype=dtype) * 0.5 + 0.25
            key, subkey = jax.random.split(key)
            d = jax.random.uniform(subkey, (n,), dtype=dtype) * 2.0 + 1.0
            key, subkey = jax.random.split(key)
            du = jax.random.uniform(subkey, (n,), dtype=dtype) * 0.5 + 0.25
            key, subkey = jax.random.split(key)
            b = jax.random.normal(subkey, (n, 10), dtype=dtype)
            
            # Function to benchmark
            def solve_fn():
                return jax.lax.linalg.tridiagonal_solve(
                    dl, d, du, b, perturb_singular=True
                )
            
            print(f"[bench] n={n} dtype={dtype_str} reps={parsed_args.reps} ...")
            
            result = benchmark.measure(
                fn=solve_fn,
                n=n,
                dtype_str=dtype_str,
                platform=parsed_args.platform,
                reps=parsed_args.reps,
                warmup=parsed_args.warmup,
                tags=["tridiagonal", "perturbed", "linalg"],
            )
            
            print(f"  mean={result.mean_us}μs std={result.std_us}μs "
                  f"min={result.min_us}μs max={result.max_us}μs "
                  f"compile={result.compile_time_us}μs "
                  f"mem_delta={result.memory_after_mb - result.memory_before_mb:+.1f}MB")
    
    benchmark.print_summary()
    benchmark.save(parsed_args.output)
    print(f"\n[shinka] Results saved to {parsed_args.output}")


if __name__ == '__main__':
    shinka_main(bench_tridiagonal_solve)
