"""Dense linear-algebra correctness across N-sweeps, dtypes, and devices.

Covers the cuSOLVER/LAPACK-backed ops (solver_kernels_ffi.cc / lapack_kernels.cc) and
the GPU custom kernels (linalg_kernels). Each check is a defining residual, robust to
factorisation non-uniqueness.
"""
import numpy as np
import pytest

from _util import DEVICES, DTYPES_REAL, DTYPES_ALL, SIZES, TOL
from _util import rng, randmat, spd, rel, to_np

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.scipy.linalg as jsl


def _dev(x, device):
    return jax.device_put(jnp.asarray(x), jax.devices(device)[0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", SIZES)
def test_lu(device, dt, n):
    a = randmat(rng(n), n, dt)
    P, L, U = to_np(jax.jit(jsl.lu)(_dev(a, device)))
    assert rel(P @ L @ U, a) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", SIZES)
def test_qr(device, dt, n):
    a = randmat(rng(n), n, dt)
    Q, R = to_np(jax.jit(jnp.linalg.qr)(_dev(a, device)))
    assert rel(Q @ R, a) <= TOL[dt]
    assert rel(Q.conj().T @ Q, np.eye(n, dtype=Q.dtype)) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", SIZES)
def test_svd(device, dt, n):
    a = randmat(rng(n), n, dt)
    U, s, Vh = to_np(jax.jit(lambda m: jnp.linalg.svd(m, full_matrices=False))(_dev(a, device)))
    assert rel((U * s) @ Vh, a) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", SIZES)
def test_eigh(device, dt, n):
    a = spd(rng(n), n, dt)
    w, V = to_np(jax.jit(jnp.linalg.eigh)(_dev(a, device)))
    assert rel(a @ V, V @ np.diag(w)) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", SIZES)
def test_cholesky(device, dt, n):
    a = spd(rng(n), n, dt)
    L = to_np(jax.jit(jnp.linalg.cholesky)(_dev(a, device)))
    assert rel(L @ L.conj().T, a) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", SIZES)
def test_solve(device, dt, n):
    from _util import NP
    r = rng(n)
    a = randmat(r, n, dt) + n * np.eye(n, dtype=NP[dt])
    b = randmat(r, n, dt)[:, :1]
    x = to_np(jax.jit(jnp.linalg.solve)(_dev(a, device), _dev(b, device)))
    assert rel(a @ x, b) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_ALL)
@pytest.mark.parametrize("n", [64, 256, 512])
def test_inv(device, dt, n):
    from _util import NP
    a = randmat(rng(n), n, dt) + n * np.eye(n, dtype=NP[dt])
    inv = to_np(jax.jit(jnp.linalg.inv)(_dev(a, device)))
    assert rel(a @ inv, np.eye(n, dtype=NP[dt])) <= TOL[dt]
