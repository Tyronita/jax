"""ShinkaEvolve Test Suite — Target 02: lapack_kernels.h

Tests LU, QR, SVD, Eigendecomposition, Cholesky, Triangular solve.
These are the most-used kernels in all of machine learning.
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


def random_well_conditioned_matrix(n, dtype, key, min_sv=0.1):
    """Generate random well-conditioned matrix."""
    a = jax.random.normal(key, (n, n), dtype=dtype)
    u, s, vt = jnp.linalg.svd(a)
    s = jnp.clip(s, min_sv, None)
    return (u * s) @ vt


def random_pd_matrix(n, dtype, key):
    """Generate random positive-definite matrix."""
    a = jax.random.normal(key, (n, n), dtype=dtype)
    return a @ a.T + n * jnp.eye(n, dtype=dtype)


class ShinkaLapackKernelsTest(jtu.JaxTestCase):
    """Comprehensive correctness suite for lapack_kernels.h."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    # ---- LU Decomposition ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [2, 5, 10, 33, 64, 127, 256, 512]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_lu_decomposition_correctness(self, n, dtype):
        """LU: verify P @ A == L @ U."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "lu")))
        a = random_well_conditioned_matrix(n, dtype, key)
        
        lu, pivots = lax_linalg.lu(a)
        
        # Extract L and U
        l = jnp.tril(lu, -1) + jnp.eye(n, dtype=dtype)
        u = jnp.triu(lu)
        
        # Construct permutation from pivots
        p = jnp.eye(n, dtype=dtype)
        for i in range(n):
            p = p.at[[i, int(pivots[i])], :].set(p[[int(pivots[i]), i], :])
        
        # Check P @ A ≈ L @ U
        reconstructed = p @ a
        lu_product = l @ u
        self.assertAllClose(reconstructed, lu_product, atol=1e-4 if dtype == np.float32 else 1e-12)

    # ---- QR Decomposition ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_dtype={dtype.__name__}',
            'm': m,
            'n': n,
            'dtype': dtype,
        } for m, n in [(10, 5), (50, 30), (128, 64), (256, 128), (512, 256)]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_qr_decomposition(self, m, n, dtype):
        """QR: verify Q^H @ Q == I and Q @ R == A."""
        key = jax.random.fold_in(self.key, hash((m, n, str(dtype), "qr")))
        a = jax.random.normal(key, (m, n), dtype=dtype)
        
        q, r = jnp.linalg.qr(a)
        
        # Q is orthogonal (for real) or unitary (for complex)
        qhq = q.T.conj() @ q if jnp.iscomplexobj(q) else q.T @ q
        self.assertAllClose(qhq, jnp.eye(min(m, n), dtype=dtype),
                           atol=1e-4 if dtype == np.float32 else 1e-11)
        
        # Q @ R == A
        self.assertAllClose(q @ r, a, atol=1e-4 if dtype == np.float32 else 1e-11)

    # ---- SVD ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_dtype={dtype.__name__}',
            'm': m,
            'n': n,
            'dtype': dtype,
        } for m, n in [(10, 8), (64, 32), (128, 128), (256, 128), (512, 256)]
          for dtype in [np.float32, np.float64])
    )
    def test_svd_reconstruction(self, m, n, dtype):
        """SVD: verify U @ diag(S) @ V^H == A."""
        key = jax.random.fold_in(self.key, hash((m, n, str(dtype), "svd")))
        a = jax.random.normal(key, (m, n), dtype=dtype)
        
        u, s, vh = jnp.linalg.svd(a, full_matrices=False)
        
        # Reconstruct A
        reconstructed = (u * s) @ vh
        self.assertAllClose(reconstructed, a, atol=1e-4 if dtype == np.float32 else 1e-11)
        
        # U and V are unitary
        uh_u = u.T @ u
        self.assertAllClose(uh_u, jnp.eye(min(m, n), dtype=dtype),
                           atol=1e-4 if dtype == np.float32 else 1e-11)

    # ---- Eigendecomposition ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [5, 10, 33, 64, 128]
          for dtype in [np.float32, np.float64])
    )
    def test_eigendecomposition(self, n, dtype):
        """eigh: verify A @ V == V @ diag(λ) for symmetric matrices."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "eig")))
        a = random_pd_matrix(n, dtype, key)  # Symmetric positive-definite
        
        w, v = jnp.linalg.eigh(a)
        
        # A @ v_i == λ_i @ v_i for each eigenvector
        for i in range(min(5, n)):
            lhs = a @ v[:, i]
            rhs = w[i] * v[:, i]
            self.assertAllClose(lhs, rhs, atol=1e-4 if dtype == np.float32 else 1e-10)

    # ---- Cholesky ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [5, 10, 33, 64, 128, 256, 512]
          for dtype in [np.float32, np.float64])
    )
    def test_cholesky(self, n, dtype):
        """Cholesky: verify L @ L^H == A."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "chol")))
        a = random_pd_matrix(n, dtype, key)
        
        l = jnp.linalg.cholesky(a)
        
        # L @ L^H
        reconstructed = l @ l.T.conj()
        self.assertAllClose(reconstructed, a, atol=1e-4 if dtype == np.float32 else 1e-11)

    # ---- Matrix Inverse ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [5, 10, 33, 64, 128]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_matrix_inverse(self, n, dtype):
        """inv: verify A @ A^{-1} == I."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "inv")))
        a = random_well_conditioned_matrix(n, dtype, key)
        
        a_inv = jnp.linalg.inv(a)
        identity = a @ a_inv
        
        self.assertAllClose(identity, jnp.eye(n, dtype=dtype),
                           atol=1e-3 if dtype == np.float32 else 1e-10)

    # ---- Triangular Solve ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_nrhs={nrhs}_dtype={dtype.__name__}',
            'n': n,
            'nrhs': nrhs,
            'dtype': dtype,
        } for n in [32, 64, 128, 256]
          for nrhs in [1, 5, 10]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_triangular_solve(self, n, nrhs, dtype):
        """Triangular solve: verify L @ x == b."""
        key = jax.random.fold_in(self.key, hash((n, nrhs, str(dtype), "tri")))
        # Lower triangular matrix
        l = jnp.tril(jax.random.normal(key, (n, n), dtype=dtype)) + jnp.eye(n, dtype=dtype)
        b = jax.random.normal(key, (n, nrhs), dtype=dtype)
        
        x = jax.lax.linalg.triangular_solve(l, b, left_side=True, lower=True)
        
        residual = l @ x - b
        self.assertLess(jnp.linalg.norm(residual) / (jnp.linalg.norm(b) + 1e-12),
                       1e-4 if dtype == np.float32 else 1e-11)

    # ---- Batch Operations ----

    def test_batch_lu(self):
        """LU on batched matrices."""
        batch = (4, 3)
        n = 32
        dtype = np.float64
        key = self.key
        
        a = jax.random.normal(key, (*batch, n, n), dtype=dtype)
        lu, pivots = lax_linalg.lu(a)
        self.assertEqual(lu.shape, (*batch, n, n))
        self.assertTrue(jnp.all(jnp.isfinite(lu)))

    def test_batch_cholesky(self):
        """Cholesky on batched positive-definite matrices."""
        batch = (8,)
        n = 64
        dtype = np.float64
        key = self.key
        
        # Create batch of PD matrices
        a_base = jax.random.normal(key, (*batch, n, n), dtype=dtype)
        a = jnp.einsum('...ij,...kj->...ik', a_base, a_base) + n * jnp.eye(n, dtype=dtype)
        
        l = jnp.linalg.cholesky(a)
        self.assertEqual(l.shape, (*batch, n, n))
        
        # Verify each in batch
        for i in range(batch[0]):
            reconstructed = l[i] @ l[i].T
            self.assertAllClose(reconstructed, a[i], atol=1e-10)

    # ---- Gradients ----

    def test_lu_gradient(self):
        """Verify LU decomposition is differentiable."""
        n = 16
        dtype = np.float64
        key = self.key
        a = random_well_conditioned_matrix(n, dtype, key)
        
        def loss(a):
            lu, _ = lax_linalg.lu(a)
            return jnp.sum(lu ** 2)
        
        val, grad = jax.value_and_grad(loss)(a)
        self.assertTrue(jnp.isfinite(val))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))

    def test_svd_gradient(self):
        """Verify SVD is differentiable."""
        n = 16
        dtype = np.float64
        key = self.key
        a = jax.random.normal(key, (n, n), dtype=dtype)
        
        def loss(a):
            u, s, vh = jnp.linalg.svd(a)
            return jnp.sum(s)
        
        val, grad = jax.value_and_grad(loss)(a)
        self.assertTrue(jnp.isfinite(val))
        self.assertTrue(jnp.all(jnp.isfinite(grad)))


if __name__ == '__main__':
    absltest.main()
