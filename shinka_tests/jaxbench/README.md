# JaxBench-aligned test cases

Extra correctness tests for the ShinkaEvolve mutation targets, kept in the fork so
they run in CI against the **built** `jaxlib` / `jax-cuda-plugin` wheels. They mirror
the correctness gate of [JaxBench](https://github.com/Tyronita/JaxBench): property
residuals over an N-sweep and multiple dtypes, on CPU and (auto-detected) GPU.

| file | covers | targets |
|---|---|---|
| `test_linalg_sweeps.py` | LU/QR/SVD/eigh/cholesky/solve/inv, N∈{64..1024}, f32/f64/c64/c128 | `solver_kernels_ffi.cc`, `lapack_kernels.cc`, `linalg_kernels.*` |
| `test_tridiagonal.py` | tridiagonal solve, N up to 16384, multi-RHS | `tridiagonal_solve_perturbed.h`, `linalg_kernels.cu.cc` |
| `test_prng.py` | Threefry determinism + distribution | `prng_kernels.cu.cc` |
| `test_integration.py` | composite pipelines (LU→solve, normal equations, SVD pinv, eigh A^½) | multiple |

## Run

```bash
JAX_ENABLE_X64=1 pytest shinka_tests/jaxbench -q          # CPU; GPU auto-skips
JAX_ENABLE_X64=1 pytest shinka_tests/jaxbench -q -k gpu   # GPU (needs the plugin)
```

A ShinkaEvolve candidate must keep these green (correctness gate) before its speedup
counts. See the JaxBench `docs/PR_PLAYBOOK.md` for the path to an upstream PR.
