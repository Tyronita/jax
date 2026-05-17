#!/usr/bin/env python3
"""ShinkaEvolve Benchmark — Target 02: lapack_kernels.h

Benchmarks LU, QR, SVD, Eigh, Cholesky, Matrix Inverse, Triangular Solve.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from shinka_test_framework import ShinkaBenchmark
import argparse


def random_well_conditioned_matrix(n, dtype, key):
    a = jax.random.normal(key, (n, n), dtype=dtype)
    u, s, vt = jnp.linalg.svd(a)
    s = jnp.clip(s, 0.1, None)
    return (u * s) @ vt


def bench_lapack_kernels(parsed_args):
    target = "jaxlib/cpu/lapack_kernels.h"
    benchmark = ShinkaBenchmark(
        target=target,
        operation="lapack_kernels",
        mutation_id=parsed_args.mutation_id,
    )
    
    key = jax.random.PRNGKey(42)
    
    for n in parsed_args.n:
        for dtype_str in parsed_args.dtype:
            dtype_map = {"f32": jnp.float32, "f64": jnp.float64, "c64": jnp.complex64, "c128": jnp.complex128}
            dtype = dtype_map[dtype_str]
            
            # LU
            key, subkey = jax.random.split(key)
            a = random_well_conditioned_matrix(n, dtype, subkey)
            def lu_fn():
                return jax.lax.linalg.lu(a)
            print(f"[bench] LU n={n} dtype={dtype_str} ...")
            benchmark.measure(lu_fn, n, dtype_str, parsed_args.platform, parsed_args.reps, parsed_args.warmup,
                            tags=["lu", "lapack"])
            
            # QR
            key, subkey = jax.random.split(key)
            a = jax.random.normal(subkey, (n, n//2), dtype=dtype)
            def qr_fn():
                return jnp.linalg.qr(a)
            print(f"[bench] QR n={n} dtype={dtype_str} ...")
            benchmark.measure(qr_fn, n, dtype_str, parsed_args.platform, parsed_args.reps, parsed_args.warmup,
                            tags=["qr", "lapack"])
            
            # SVD
            key, subkey = jax.random.split(key)
            a = jax.random.normal(subkey, (n, n//2), dtype=dtype)
            def svd_fn():
                return jnp.linalg.svd(a, full_matrices=False)
            print(f"[bench] SVD n={n} dtype={dtype_str} ...")
            benchmark.measure(svd_fn, n, dtype_str, parsed_args.platform, parsed_args.reps, parsed_args.warmup,
                            tags=["svd", "lapack"])
            
            # Cholesky
            key, subkey = jax.random.split(key)
            a_base = jax.random.normal(subkey, (n, n), dtype=dtype)
            a = a_base @ a_base.T + n * jnp.eye(n, dtype=dtype)
            def chol_fn():
                return jnp.linalg.cholesky(a)
            print(f"[bench] Cholesky n={n} dtype={dtype_str} ...")
            benchmark.measure(chol_fn, n, dtype_str, parsed_args.platform, parsed_args.reps, parsed_args.warmup,
                            tags=["cholesky", "lapack"])
            
            # Matrix Inverse
            key, subkey = jax.random.split(key)
            a = random_well_conditioned_matrix(n, dtype, subkey)
            def inv_fn():
                return jnp.linalg.inv(a)
            print(f"[bench] Inverse n={n} dtype={dtype_str} ...")
            benchmark.measure(inv_fn, n, dtype_str, parsed_args.platform, parsed_args.reps, parsed_args.warmup,
                            tags=["inverse", "lapack"])
    
    benchmark.print_summary()
    benchmark.save(parsed_args.output)
    print(f"\n[shinka] Results saved to {parsed_args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--dtype", nargs="+", default=["f32", "f64", "c64", "c128"])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--platform", default="cpu")
    parser.add_argument("--output", default="results/target_02_baseline.json")
    parser.add_argument("--mutation-id", default=None)
    args = parser.parse_args()
    bench_lapack_kernels(args)
