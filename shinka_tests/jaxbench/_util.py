"""Shared helpers + config: dtype mapping, input generation, residual norm, devices.

Constants live here (not in conftest.py) because the jax repo already has a top-level
conftest.py; importing `from conftest import ...` would resolve to the wrong module.
"""
import os
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

NP = {"f32": np.float32, "f64": np.float64, "c64": np.complex64, "c128": np.complex128}
RE = {"f32": np.float32, "f64": np.float64, "c64": np.float32, "c128": np.float64}

DTYPES_REAL = ["f32", "f64"]
DTYPES_ALL = ["f32", "f64", "c64", "c128"]
SIZES = [64, 128, 256, 512, 1024]
TOL = {"f32": 2e-4, "f64": 1e-10, "c64": 2e-4, "c128": 1e-10}


def _has_gpu():
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


HAS_GPU = _has_gpu()
DEVICES = ["cpu"] + (["gpu"] if HAS_GPU else [])


def rng(seed=0):
    return np.random.default_rng(seed)


def randmat(r, n, dt):
    d = NP[dt]
    if dt.startswith("c"):
        rr = RE[dt]
        return (r.standard_normal((n, n)).astype(rr) + 1j * r.standard_normal((n, n)).astype(rr)).astype(d)
    return r.standard_normal((n, n)).astype(d)


def spd(r, n, dt):
    a = randmat(r, n, dt)
    return (a @ a.conj().T + n * np.eye(n, dtype=NP[dt])).astype(NP[dt])


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1.0))


def to_np(x):
    import jax
    x = jax.block_until_ready(x)
    if isinstance(x, tuple):
        return tuple(to_np(e) for e in x)
    return np.asarray(x)
