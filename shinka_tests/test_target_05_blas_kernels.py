"""ShinkaEvolve Test Suite — Target 05: blas_kernels_ffi.h

GEMM (General Matrix Multiply) and BLAS Level 2/3 operations.
The single most performance-critical kernel in all machine learning.
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class ShinkaBlasKernelsTest(jtu.JaxTestCase):
    """GEMM and BLAS suite."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    # ---- GEMM ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_k={k}_dtype={dtype.__name__}',
            'm': m,
            'n': n,
            'k': k,
            'dtype': dtype,
        } for m, n, k in [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 128, 256), (512, 512, 512)]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_gemm_correctness(self, m, n, k, dtype):
        """General matrix multiply: verify against numpy reference."""
        key = jax.random.fold_in(self.key, hash((m, n, k, str(dtype), "gemm")))
        
        a = jax.random.normal(key, (m, k), dtype=dtype)
        b = jax.random.normal(key, (k, n), dtype=dtype)
        
        c = jnp.dot(a, b)
        c_np = np.dot(np.array(a), np.array(b))
        
        self.assertAllClose(np.array(c), c_np,
                           atol=1e-4 if dtype in (np.float32, np.complex64) else 1e-12)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_k={k}',
            'm': m,
            'n': n,
            'k': k,
        } for m, n, k in [(64, 64, 64), (128, 256, 128), (512, 512, 512), (1024, 512, 1024)]
          if m * n * k <= 1024 * 1024 * 512)  # Don't OOM
    )
    def test_gemm_various_shapes(self, m, n, k):
        """GEMM with non-square matrices."""
        dtype = np.float64
        key = jax.random.fold_in(self.key, hash((m, n, k, "shape")))
        
        a = jax.random.normal(key, (m, k), dtype=dtype)
        b = jax.random.normal(key, (k, n), dtype=dtype)
        
        c = jnp.dot(a, b)
        c_np = np.dot(np.array(a), np.array(b))
        self.assertAllClose(np.array(c), c_np, atol=1e-11)

    # ---- Batch GEMM ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_batch={batch}_n={n}',
            'batch': batch,
            'n': n,
        } for batch in [(4,), (4, 3), (8, 4)]
          for n in [32, 64, 128])
    )
    def test_batch_gemm(self, batch, n):
        """Batched matrix multiply."""
        dtype = np.float64
        key = jax.random.fold_in(self.key, hash((batch, n, "batch_gemm")))
        
        a = jax.random.normal(key, (*batch, n, n), dtype=dtype)
        b = jax.random.normal(key, (*batch, n, n), dtype=dtype)
        
        c = jnp.matmul(a, b)
        
        # Verify each batch element
        for idx in np.ndindex(batch):
            c_np = np.dot(np.array(a[idx]), np.array(b[idx]))
            self.assertAllClose(np.array(c[idx]), c_np, atol=1e-11)

    # ---- GEMV ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_dtype={dtype.__name__}',
            'm': m,
            'n': n,
            'dtype': dtype,
        } for m, n in [(64, 64), (128, 256), (512, 512), (1024, 1024)]
          for dtype in [np.float32, np.float64])
    )
    def test_gemv(self, m, n, dtype):
        """Matrix-vector multiply."""
        key = jax.random.fold_in(self.key, hash((m, n, str(dtype), "gemv")))
        
        a = jax.random.normal(key, (m, n), dtype=dtype)
        x = jax.random.normal(key, (n,), dtype=dtype)
        
        y = jnp.dot(a, x)
        y_np = np.dot(np.array(a), np.array(x))
        
        self.assertAllClose(np.array(y), y_np,
                           atol=1e-4 if dtype == np.float32 else 1e-12)

    # ---- SYMM (Symmetric multiply) ----

    def test_symmetric_multiply(self):
        """A @ A^T for symmetric result."""
        n = 128
        dtype = np.float64
        key = self.key
        
        a = jax.random.normal(key, (n, n), dtype=dtype)
        a_sym = a @ a.T
        
        # Cholesky should work on positive-definite
        l = jnp.linalg.cholesky(a_sym)
        reconstructed = l @ l.T
        
        self.assertAllClose(reconstructed, a_sym, atol=1e-10)

    # ---- Alpha/Beta scaling in GEMM ----

    def test_gemm_alpha_beta(self):
        """GEMM with alpha and beta coefficients."""
        m, n, k = 128, 128, 128
        dtype = np.float64
        key = self.key
        
        a = jax.random.normal(key, (m, k), dtype=dtype)
        b = jax.random.normal(key, (k, n), dtype=dtype)
        c = jax.random.normal(key, (m, n), dtype=dtype)
        
        alpha = 2.0
        beta = 0.5
        
        # JAX equivalent of BLAS GEMM
        result = alpha * jnp.dot(a, b) + beta * c
        
        expected = alpha * np.dot(np.array(a), np.array(b)) + beta * np.array(c)
        self.assertAllClose(np.array(result), expected, atol=1e-11)

    # ---- Transpose variants ----

    def test_gemm_transpose(self):
        """GEMM with transposed inputs."""
        m, n, k = 128, 128, 128
        dtype = np.float64
        key = self.key
        
        a = jax.random.normal(key, (k, m), dtype=dtype)
        b = jax.random.normal(key, (k, n), dtype=dtype)
        
        # C = A^T @ B
        c = jnp.dot(a.T, b)
        expected = np.dot(np.array(a).T, np.array(b))
        
        self.assertAllClose(np.array(c), expected, atol=1e-11)

    # ---- Edge cases ----

    def test_gemm_zero_matrix(self):
        """GEMM with zero matrix should return zero."""
        n = 128
        dtype = np.float64
        
        a = jnp.zeros((n, n), dtype=dtype)
        b = jax.random.normal(self.key, (n, n), dtype=dtype)
        
        c = jnp.dot(a, b)
        self.assertAllClose(c, jnp.zeros_like(c), atol=1e-15)

    def test_gemm_identity(self):
        """GEMM with identity matrix."""
        n = 128
        dtype = np.float64
        
        a = jnp.eye(n, dtype=dtype)
        b = jax.random.normal(self.key, (n, n), dtype=dtype)
        
        c = jnp.dot(a, b)
        self.assertAllClose(c, b, atol=1e-15)

    # ---- Gradient (autodiff through GEMM) ----

    def test_gemm_gradient(self):
        """Verify GEMM is differentiable."""
        m, n, k = 64, 64, 64
        dtype = np.float64
        key = self.key
        
        a = jax.random.normal(key, (m, k), dtype=dtype)
        b = jax.random.normal(key, (k, n), dtype=dtype)
        
        def loss(a, b):
            c = jnp.dot(a, b)
            return jnp.sum(c ** 2)
        
        val, grads = jax.value_and_grad(loss, argnums=(0, 1))(a, b)
        
        self.assertTrue(jnp.isfinite(val))
        for g in grads:
            self.assertTrue(jnp.all(jnp.isfinite(g)))


if __name__ == '__main__':
    absltest.main()
