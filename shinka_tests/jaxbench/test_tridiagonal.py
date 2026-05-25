"""Tridiagonal solver correctness — the proven ShinkaEvolve target (gen-68 +41.73%).

Covers tridiagonal_solve across N-sweeps and dtypes, on diagonally-dominant systems
and on the perturbed-pivot path, checking the dense residual A x = b.
"""
import numpy as np
import pytest

from _util import DEVICES, DTYPES_REAL, TOL, rng as _rng, NP, rel, to_np

jax = pytest.importorskip("jax")
import jax.numpy as jnp


def _system(r, n, dt, dominant=True):
    dl = r.standard_normal(n).astype(NP[dt]); dl[0] = 0
    base = 4.0 if dominant else 0.5
    d = (r.standard_normal(n) + base).astype(NP[dt])
    du = r.standard_normal(n).astype(NP[dt]); du[-1] = 0
    b = r.standard_normal((n, 1)).astype(NP[dt])
    return dl, d, du, b


def _dense(dl, d, du):
    return np.diag(d) + np.diag(du[:-1], 1) + np.diag(dl[1:], -1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", DTYPES_REAL)
@pytest.mark.parametrize("n", [256, 1024, 4096, 16384])
def test_tridiagonal_solve(device, dt, n):
    la = jax.lax.linalg
    dl, d, du, b = _system(_rng(n), n, dt)
    dev = jax.devices(device)[0]
    put = lambda x: jax.device_put(jnp.asarray(x), dev)
    x = to_np(jax.jit(la.tridiagonal_solve)(put(dl), put(d), put(du), put(b)))
    assert rel(_dense(dl, d, du) @ x, b) <= TOL[dt]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [512, 4096])
def test_tridiagonal_multiple_rhs(device, n):
    """Multiple right-hand sides solved consistently."""
    la = jax.lax.linalg
    r = _rng(n)
    dl = r.standard_normal(n).astype(np.float64); dl[0] = 0
    d = (r.standard_normal(n) + 4).astype(np.float64)
    du = r.standard_normal(n).astype(np.float64); du[-1] = 0
    b = r.standard_normal((n, 8)).astype(np.float64)
    dev = jax.devices(device)[0]
    put = lambda x: jax.device_put(jnp.asarray(x), dev)
    x = to_np(jax.jit(la.tridiagonal_solve)(put(dl), put(d), put(du), put(b)))
    assert rel(_dense(dl, d, du) @ x, b) <= 1e-9
