# ShinkaEvolve Multi-Platform Test Strategy

**Document:** SHINKA-TEST-STRATEGY.md  
**Version:** 1.0  
**Date:** 2026-05-17  
**Scope:** Test execution across OS × Device × Compiler combinations for all 25 JAX targets

---

## Goal

Every mutation must pass correctness and performance thresholds on **at least 3 platforms** before upstream PR:

1. **Linux x86_64 + CPU** (Azure VM `evan` — Standard_NC4as_T4_v3)
2. **Linux x86_64 + GPU** (same VM, Tesla T4, CUDA 12.8)
3. **Windows 11 x86_64 + CPU** (local dev machine — Evan-HP-Envy-17)
4. **macOS ARM64 (Apple Silicon) + CPU** (GitHub Actions or local M1/M2/M3)
5. **Linux ARM64 + CPU** (AWS Graviton3 or GitHub Actions)

---

## Platform Matrix

| OS | Arch | Device | Compiler | Test Level | CI | Priority |
|----|------|--------|----------|-----------|----|----------|
| Ubuntu 24.04 | x86_64 | CPU | Clang 18 | Full | Azure VM | **P0** |
| Ubuntu 24.04 | x86_64 | GPU (Tesla T4) | Clang 18 + NVCC | Full | Azure VM | **P0** |
| Windows 11 | x86_64 | CPU | MSVC 2022 | Full | Local + GH Actions | **P1** |
| macOS 15 | ARM64 | CPU (M3) | Clang 18 | Full | GitHub Actions | **P1** |
| Amazon Linux | ARM64 | CPU | GCC 13 | Core only | AWS/GH Actions | **P2** |
| Ubuntu 24.04 | x86_64 | GPU (A100/H100) | Clang 18 + NVCC | Performance only | Colab/GCP | **P2** |

---

## Test Levels

### Level 1 — Core Correctness (all platforms)
- Run subset: `test_target_*.py` with sizes `[32, 128, 512]`
- Target: < 5 minutes per platform
- Gates: all assertions pass, no crash, finite outputs

### Level 2 — Full Correctness (P0 + P1 platforms)
- Run full parameterized test matrix
- All sizes: `[2, 5, 10, 33, 64, 127, 256, 512, 1024]`
- All dtypes: `f32, f64, c64, c128`
- Target: < 30 minutes per platform
- Gates: Level 1 + edge cases + vmap/batch + gradients

### Level 3 — Performance Benchmark (P0 platforms)
- Run `bench_target_*.py` with statistical rigor
- Repetitions: 300 for μs precision
- Sizes: `[32, 64, 128, 256, 512, 1024, 2048, 4096]`
- Log: mean, std, min, max, median, compile time, memory delta, GC collections
- Target: < 1 hour per target
- Gates: no regression > 10% from baseline on any size

### Level 4 — Stress & Stability (P0 platforms, nightly)
- Run 10,000 iterations on random matrices
- Vary: ill-conditioned, near-singular, extreme values
- Log: crash rate, NaN rate, numerical drift over time
- Target: overnight batch
- Gates: 0 crashes, NaN rate < 0.01%

---

## Per-Platform Build Instructions

### Ubuntu 24.04 x86_64 (Primary)
```bash
# Already set up on evan
bazelisk build --config=opt --wheels=jaxlib
pip install dist/jaxlib-*.whl
pip install -e .
pip install pytest pytest-xdist psutil numpy scipy
pytest shinka_tests/ -v --timeout=600
```

### Windows 11 x86_64
```powershell
# Requires Bazelisk, MSVC 2022
python build\build.py build --wheels=jaxlib
pip install dist\jaxlib-*.whl
pip install -e .
pip install pytest psutil numpy scipy
pytest shinka_tests\ -v
```

### macOS ARM64
```bash
# Requires Xcode Command Line Tools, Bazelisk
bazelisk build --config=opt --wheels=jaxlib --cpu=darwin_arm64
pip install dist/jaxlib-*.whl
pip install -e .
pip install pytest psutil numpy scipy
pytest shinka_tests/ -v
```

---

## Background Quality Metrics

Every benchmark run logs:

```json
{
  "system_info": {
    "os": "Ubuntu 24.04",
    "arch": "x86_64",
    "cpu_count": 4,
    "cpu_freq_mhz": 2400,
    "total_ram_gb": 28.0,
    "python_version": "3.12.3",
    "jax_version": "0.4.38",
    "jaxlib_version": "0.4.38",
    "gpu_name": "Tesla T4",
    "gpu_memory_gb": 16.0,
    "cuda_version": "12.8",
    "compiler": "clang 18.1.8"
  },
  "benchmarks": [
    {
      "target": "jaxlib/tridiagonal_solve_perturbed.h",
      "operation": "tridiagonal_solve",
      "n": 256,
      "dtype_str": "f64",
      "mean_us": 145.32,
      "std_us": 12.45,
      "compile_time_us": 2341.0,
      "gc_collections": 0,
      "memory_before_mb": 245.1,
      "memory_after_mb": 247.8,
      "memory_peak_mb": 251.3,
      "mutation_id": "gen-68"
    }
  ]
}
```

---

## Anomaly Detection

Per-run heuristics flag suspicious results:

1. **Regression**: mean_time > 1.10 × baseline → flag
2. **High variance**: std/mean > 0.30 → flag (GC interference or noisy neighbor)
3. **Memory leak**: memory_after > memory_before + 50MB → flag
4. **Compile anomaly**: compile_time > 5s for simple functions → flag
5. **NaN/Inf**: any non-finite output → immediately fail

---

## CI/CD Integration

### GitHub Actions Workflow (`.github/workflows/shinka-tests.yml`)

```yaml
name: Shinka Kernel Tests
on: [push, pull_request]
jobs:
  linux-cpu:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - name: Build jaxlib
        run: bazelisk build --config=opt --wheels=jaxlib
      - name: Run Shinka Tests
        run: |
          pip install dist/jaxlib-*.whl
          pip install -e . pytest psutil numpy scipy
          pytest shinka_tests/ --timeout=600 -v
      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: linux-cpu-results
          path: results/*.json

  macos-arm64:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - name: Install deps
        run: brew install bazelisk
      - name: Build and Test
        run: |
          bazelisk build --config=opt --wheels=jaxlib --cpu=darwin_arm64
          pip install dist/jaxlib-*.whl
          pip install -e . pytest psutil numpy scipy
          pytest shinka_tests/ --timeout=600 -v

  windows-cpu:
    runs-on: windows-2022
    steps:
      - uses: actions/checkout@v4
      - name: Build and Test
        run: |
          python build\build.py build --wheels=jaxlib
          pip install dist\jaxlib-*.whl
          pip install -e . pytest psutil numpy scipy
          pytest shinka_tests\ -v
```

---

## Mutation Acceptance Criteria

For a mutation to be accepted into upstream `google/jax`:

| Criterion | Tier 1 (Tridiag) | Tier 2 (Lapack) | Tier 3+ |
|-----------|-----------------|-----------------|---------|
| Linux CPU pass | Required | Required | Required |
| Linux GPU pass | Required | Required | Skip if no GPU |
| Windows CPU pass | Required | Required | Required |
| macOS ARM64 pass | Required | Required | Required |
| Speedup > 5% on at least 1 size | Required | Required | Required |
| No regression > 2% on any size | Required | Required | Required |
| Correctness test coverage | All tests | LU/QR/SVD subset | Core tests |
| Upstream PR ready | Yes | Partial | No |

---

## Current Status

- ✅ Framework written (`shinka_test_framework.py`)
- ✅ Tier 1 tests complete (Tridiag, Lapack, BLAS)
- ✅ Tier 2-7 tests complete (Sparse, PRNG, FFT, Sort, Special, Conv, Reduction)
- 🔄 Branch `shinka/targets-v1` pushed to `Tyronita/jax`
- ⏳ Docker warm-cache build in progress on `evan` (15,257/25,033)
- ⏳ EVOLVE-BLOCK markers to be inserted into each target file
- ⏳ Baseline benchmarks to be run once Bazel build finishes

---

## Contact

**Maintainer:** Evan O'Leary  
**Fork:** `https://github.com/Tyronita/jax`  
**Branch:** `shinka/targets-v1`  
**VM:** `evan` @ `20.102.40.196` (Azure EVAN_GROUP)
