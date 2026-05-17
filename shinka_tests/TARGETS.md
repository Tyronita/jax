# ShinkaEvolve Target Directory — Top 25 JAX C++ Kernel Files for Optimization

**Repository:** `github.com/Tyronita/jax` (our fork)  
**Upstream:** `github.com/google/jax`  
**Mission:** Systematic C++ performance optimization via EVOLVE-BLOCK mutation markers  
**Date:** 2026-05-17  
**Version:** 1.0  

---

## Philosophy

Each target file is a **Shinka island** — an independent optimization domain. Within each file, we insert `// EVOLVE-BLOCK-START` / `// EVOLVE-BLOCK-END` markers around hot loops and algorithmic sections. ShinkaEvolve generates mutations within these blocks, and each mutation is evaluated via:

1. **Correctness**: `pytest tests/...` — JAX API-level tests (CPU/GPU/TPU)
2. **Performance**: Microbenchmarks with μs-level timing, GC/memory pressure logging
3. **Cross-platform**: Every candidate is tested on Linux (Azure VM), Windows (local dev), and macOS (Apple Silicon)
4. **Background quality metrics**: Profile-guided logging (cache miss rates, branch mispredictions where available)

---

## The 25 Targets

### Tier 1 — Hot Paths in Dense Linear Algebra (High Impact, Wide Usage)

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 1 | `tridiagonal_solve_perturbed.h` | `jaxlib/tridiagonal_solve_perturbed.h` | Proven - our gen-68 hit +41.73%. Used in `jax.lax.linalg.tridiagonal_solve`, time-series PDE solvers, ODE integration. | Very High | `tests/linalg_test.py` |
| 2 | `lapack_kernels.h` | `jaxlib/cpu/lapack_kernels.h` | Backend for `lu`, `qr`, `svd`, `eigh`, `solve`, `inv`. Every ML training loop. | Very High | `tests/linalg_test.py` |
| 3 | `tridiagonal_solve_kernels.h` | `jaxlib/cpu/tridiagonal_solve_kernels.h` | Non-perturbed tridiagonal solve. Used in signal processing, banded matrices. | High | `tests/linalg_test.py` |
| 4 | `linalg_kernels.h` (GPU) | `jaxlib/gpu/linalg_kernels.h` | GPU LU, QR, SVD. Speeds up JAX on NVIDIA T4/A100/H100. | Very High | `tests/linalg_test.py` |
| 5 | `blas_kernels_ffi.h` | `jaxlib/cpu/blas_kernels_ffi.h` | GEMM backend — the most-called kernel in all of ML. | Critical | `tests/linalg_test.py` |

### Tier 2 — Sparse & Structured Matrix Operations

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 6 | `sparse_kernels.h` (CPU) | `jaxlib/cpu/sparse_kernels.h` | CSR/CSC sparse matmul, sparse Cholesky, topological sort. Scientific computing graphs. | High | `tests/sparse_test.py` |
| 7 | `sparse_kernels.h` (GPU) | `jaxlib/gpu/sparse_kernels.h` | cuSPARSE wrappers — GNNs, physics sims on GPU. | High | `tests/sparse_test.py` |
| 8 | `prng_kernels.h` (GPU) | `jaxlib/gpu/prng_kernels.h` | JAX's Threefry/random kernels. Called billions of times. | Critical | `tests/random_test.py` |
| 9 | `householder_kernels.h` | `jaxlib/gpu/householder_kernels.h` | QR decomposition via Householder reflections. Scientific computing. | High | `tests/linalg_test.py` |
| 10 | `rnn_kernels.h` (GPU) | `jaxlib/gpu/rnn_kernels.h` | cuDNN RNN wrappers. Used in deep sequence models. | High | `tests/lax_test.py` |

### Tier 3 — FFT, Transform, and Signal Processing

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 11 | `fft_kernels.h` | `jaxlib/cpu/fft_kernels.h` | FFT, IFFT, rFFT, irFFT — signal processing, audio ML. | High | `tests/fft_test.py` |
| 12 | `fft_kernels.h` (GPU) | `jaxlib/gpu/fft_kernels.h` | cuFFT wrappers. GPU-accelerated spectral methods. | High | `tests/fft_test.py` |
| 13 | `pocketfft_kernels.h` | `jaxlib/cpu/pocketfft_kernels.h` | PocketFFT fallback on CPU. Always used when no MKL. | High | `tests/fft_test.py` |

### Tier 4 — Random Number Generation & Sampling (CPU)

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 14 | `prng_kernels.h` (CPU) | `jaxlib/cpu/prng_kernels.h` | Threefry, Philox PRNG. Entire JAX random module. | Critical | `tests/random_test.py` |
| 15 | `random_gamma_kernels.h` | `jaxlib/cpu/random_gamma_kernels.h` | Gamma distribution sampling (slow, iterative). MCMC. | High | `tests/random_test.py` |
| 16 | `random_beta_kernels.h` | `jaxlib/cpu/random_beta_kernels.h` | Beta distribution. Variational inference, sampling. | Medium | `tests/random_test.py` |
| 17 | `random_poisson_kernels.h` | `jaxlib/cpu/random_poisson_kernels.h` | Poisson distribution. Count data, NLP. | Medium | `tests/random_test.py` |

### Tier 5 — Sorting, Search, and Special Functions

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 18 | `sort_kernels.h` | `jaxlib/cpu/sort_kernels.h` | `jax.lax.sort`, top-k, argsort. Used everywhere. | Very High | `tests/lax_test.py` |
| 19 | `sort_gpu_kernels.h` | `jaxlib/gpu/sort_gpu_kernels.h` | GPU sort/bitonic sort wrappers. | High | `tests/lax_test.py` |
| 20 | `special_kernels.h` | `jaxlib/cpu/special_kernels.h` | Bessel, Gamma, Erf, Zeta. Scientific computing. | Medium | `tests/scipy_special_test.py` |
| 21 | `special_kernels.h` (GPU) | `jaxlib/gpu/special_kernels.h` | cuSpecial wrappers. High-performance scientific sims. | Medium | `tests/scipy_special_test.py` |

### Tier 6 — Image Processing & Convolution

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 22 | `convolution_kernels.h` | `jaxlib/cpu/convolution_kernels.h` | `jax.lax.conv_general_dilated` on CPU. Last-mile inference. | High | `tests/lax_test.py` |
| 23 | `convolution_kernels.h` (GPU) | `jaxlib/gpu/convolution_kernels.h` | cuDNN conv wrappers. Every CV model. | Very High | `tests/lax_test.py` |

### Tier 7 — Reductions and Collectives

| # | File | Path | Why Optimize | Approx Users | Test File |
|---|------|------|-------------|-------------|-----------|
| 24 | `reduce_kernels.h` | `jaxlib/cpu/reduce_kernels.h` | `sum`, `mean`, `prod`, `min`, `max`, `argmin`, `softmax`. Every forward/backward pass. | Critical | `tests/lax_test.py` |
| 25 | `collective_kernels.h` | `jaxlib/gpu/collective_kernels.h` | AllReduce, AllGather, AllToAll. Multi-GPU training. | Critical | `tests/multidevice_test.py` |

---

## File Selection Rationale

### Why these 25?

1. **Proven impact**: `tridiagonal_solve_perturbed.h` (Target #1) already shows +41.73% via ShinkaEvolve
2. **Usage frequency**: Every file here is on a hot path for at least one major ML subdomain
3. **Self-contained algorithmic headers**: Most targets are `.h` files with tight loops — ideal for mutation
4. **Build footprint**: Templates/Self-contained = small rebuild; shared infra (like `xla/`) = massive rebuild — we AVOID the latter
5. **Cross-platform scope**: Mix of CPU and GPU targets ensures Shinka finds optimization on all hardware

### What we DON'T target (yet):

- **XLA compiler infrastructure** — `xla/` directory changes trigger 10,000+ action rebuilds (too slow per mutation)
- **Python frontend** — Not C++, not performance-critical
- **Mosaic GPU TPU** — Too niche for general optimization
- **Build system files (BUILD, .bazelrc)** — Outside optimization scope

---

## Per-Island Shinka Configuration

Each target gets its own `island_*.json` config:

```json
{
  "island_id": 1,
  "target_file": "jaxlib/tridiagonal_solve_perturbed.h",
  "evolve_blocks": ["forward_elimination", "back_substitution", "pivot_logic"],
  "test_filter": "tests/linalg_test.py::LaxLinalgTest::test_tridiagonal_solve",
  "benchmark_sizes": [32, 64, 128, 256, 512, 1024],
  "dtypes": ["f32", "f64", "c64", "c128"],
  "platforms": ["cpu", "gpu"],
  "expected_build_time_min": 2,
  "mutation_budget": 500
}
```

---

## Commit Strategy for JAX Fork

We commit to our fork in branches:

```
shinka/target-01-tridiagonal-solve
shinka/target-02-lapack-kernels
...
shinka/target-25-collective-kernels
```

Each branch contains:
1. EVOLVE-BLOCK markers in the `.h` files
2. `shinka_tests/test_target_*.py` — extensive test suite
3. `shinka_tests/bench_*.py` — performance benchmarks
4. `TARGETS.md` (this file)
5. `shinka_tests/PROFILE_LOG.md` — per-run profiling data

Upstream PRs are only submitted for mutations that pass ALL tiers (correctness + performance + cross-platform).

---

## Test Infrastructure

See `shinka_tests/` directory for:
- `shinka_test_framework.py` — timing + logging + profiling base class
- `test_target_01_tridiagonal_solve.py` — correctness + regression tests
- `bench_target_01_tridiagonal_solve.py` — μs-precision benchmarks
- `test_target_02_lapack_kernels.py` — LU/QR/SVD robustness suite
- ... (one per target)

---

## Next Steps

1. Insert EVOLVE-BLOCK markers into each target file
2. Run ShinkaEvolve baseline (no mutations) to establish reference timing
3. Begin generational mutation for each island
4. Collect JSON results and rank mutations
5. PR the best to upstream google/jax

**Status:** In progress — cold Bazel build running on `evan` (Azure VM, Standard_NC4as_T4_v3) to create warm-cache Docker image
