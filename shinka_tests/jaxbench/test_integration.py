"""Composite integration tests: chains of kernels as real workloads use them.

Catches mutations that pass a single-op gate but break a realistic pipeline.
"""
import numpy as np
import pytest

from _util import DEVICES

jax = pytest.importorskip("jax")
import jax.numpy as jnp
import jax.scipy.linalg as jsl


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [64, 256, 1024])
def test_lu_factor_then_solve(device, n):
    with jax.default_device(jax.devices(device)[0]):
        r = np.random.default_rng(n)
        A = jnp.asarray(r.standard_normal((n, n)) + n * np.eye(n))
        b = jnp.asarray(r.standard_normal((n, 4)))
        x1 = jnp.linalg.solve(A, b)
        x2 = jsl.lu_solve(jsl.lu_factor(A), b)
    np.testing.assert_allclose(np.asarray(x1), np.asarray(x2), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [128, 512])
def test_normal_equations_via_cholesky(device, n):
    """Least-squares via Cholesky of AᵀA matches lstsq."""
    with jax.default_device(jax.devices(device)[0]):
        r = np.random.default_rng(n)
        A = jnp.asarray(r.standard_normal((n, n // 2)))
        b = jnp.asarray(r.standard_normal((n, 1)))
        AtA = A.T @ A + 1e-3 * jnp.eye(n // 2)
        x_chol = jsl.cho_solve(jsl.cho_factor(AtA), A.T @ b)
        x_lstsq = jnp.linalg.lstsq(A, b)[0]
    np.testing.assert_allclose(np.asarray(x_chol), np.asarray(x_lstsq), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [128, 512])
def test_svd_pseudoinverse_solves(device, n):
    """SVD-based pseudo-inverse reproduces a consistent solution."""
    with jax.default_device(jax.devices(device)[0]):
        r = np.random.default_rng(n)
        A = jnp.asarray(r.standard_normal((n, n)) + n * np.eye(n))
        b = jnp.asarray(r.standard_normal((n, 1)))
        U, s, Vh = jnp.linalg.svd(A, full_matrices=False)
        x = Vh.conj().T @ ((U.conj().T @ b) / s[:, None])
    np.testing.assert_allclose(np.asarray(A @ x), np.asarray(b), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_eigh_matrix_function(device):
    """Build A^{1/2} via eigh and verify (A^{1/2})² ≈ A for an SPD matrix."""
    n = 256
    with jax.default_device(jax.devices(device)[0]):
        r = np.random.default_rng(0)
        M = r.standard_normal((n, n))
        A = jnp.asarray(M @ M.T + n * np.eye(n))
        w, V = jnp.linalg.eigh(A)
        sqrtA = (V * jnp.sqrt(w)) @ V.conj().T
    np.testing.assert_allclose(np.asarray(sqrtA @ sqrtA), np.asarray(A), rtol=1e-5, atol=1e-4)
