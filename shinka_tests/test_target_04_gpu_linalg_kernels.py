"""ShinkaEvolve Test Suite — Target 04: gpu/linalg_kernels.h

GPU-accelerated LU, QR, SVD, Cholesky, and Triangular Solve via CUDA/cuSOLVER.
These are the performance-critical paths for all GPU-based JAX workloads.
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


def has_gpu():
    """Check if GPU is available."""
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


class ShinkaGpuLinalgKernelsTest(jtu.JaxTestCase):
    """GPU linear algebra correctness suite."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [32, 64, 128, 256]
          for dtype in [np.float32, np.float64])
    )
    def test_gpu_lu(self, n, dtype):
        """GPU LU decomposition."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "gpu_lu")))
        a = jax.random.normal(key, (n, n), dtype=dtype)
        
        # Force GPU placement
        a = jax.device_put(a, jax.devices("gpu")[0])
        
        lu, pivots = jax.lax.linalg.lu(a)
        
        # Verify on CPU for convenience
        lu_cpu = np.array(lu)
        self.assertTrue(np.all(np.isfinite(lu_cpu)))

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_dtype={dtype.__name__}',
            'm': m,
            'n': n,
            'dtype': dtype,
        } for m, n in [(64, 32), (128, 64), (256, 128)]
          for dtype in [np.float32, np.float64])
    )
    def test_gpu_qr(self, m, n, dtype):
        """GPU QR decomposition."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        key = jax.random.fold_in(self.key, hash((m, n, str(dtype), "gpu_qr")))
        a = jax.random.normal(key, (m, n), dtype=dtype)
        a = jax.device_put(a, jax.devices("gpu")[0])
        
        q, r = jnp.linalg.qr(a)
        q_cpu = np.array(q)
        r_cpu = np.array(r)
        
        # Verify Q is orthogonal
        qhq = q_cpu.T @ q_cpu
        self.assertAllClose(qhq, np.eye(min(m, n), dtype=dtype), atol=1e-3)
        
        # Verify Q @ R == A
        self.assertAllClose(q_cpu @ r_cpu, np.array(a), atol=1e-3)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_m={m}_n={n}_dtype={dtype.__name__}',
            'm': m,
            'n': n,
            'dtype': dtype,
        } for m, n in [(64, 32), (128, 64)]
          for dtype in [np.float32, np.float64])
    )
    def test_gpu_svd(self, m, n, dtype):
        """GPU SVD."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        key = jax.random.fold_in(self.key, hash((m, n, str(dtype), "gpu_svd")))
        a = jax.random.normal(key, (m, n), dtype=dtype)
        a = jax.device_put(a, jax.devices("gpu")[0])
        
        u, s, vh = jnp.linalg.svd(a, full_matrices=False)
        
        # Verify reconstruction
        reconstructed = (u * s) @ vh
        self.assertAllClose(np.array(reconstructed), np.array(a), atol=1e-3)

    def test_gpu_cholesky(self):
        """GPU Cholesky decomposition."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        n = 128
        dtype = np.float64
        key = self.key
        
        a_base = jax.random.normal(key, (n, n), dtype=dtype)
        a = a_base @ a_base.T + n * jnp.eye(n, dtype=dtype)
        a = jax.device_put(a, jax.devices("gpu")[0])
        
        l = jnp.linalg.cholesky(a)
        l_cpu = np.array(l)
        
        reconstructed = l_cpu @ l_cpu.T
        self.assertAllClose(reconstructed, np.array(a), atol=1e-10)

    def test_gpu_matrix_inverse(self):
        """GPU matrix inverse."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        n = 128
        dtype = np.float64
        key = self.key
        
        a = jax.random.normal(key, (n, n), dtype=dtype)
        # Make it well-conditioned
        a = a @ a.T + n * jnp.eye(n, dtype=dtype)
        a = jax.device_put(a, jax.devices("gpu")[0])
        
        a_inv = jnp.linalg.inv(a)
        identity = a @ a_inv
        
        self.assertAllClose(np.array(identity), np.eye(n, dtype=dtype), atol=1e-5)

    def test_gpu_triangular_solve(self):
        """GPU triangular solve."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        n = 256
        dtype = np.float64
        key = self.key
        
        l = jnp.tril(jax.random.normal(key, (n, n), dtype=dtype)) + jnp.eye(n, dtype=dtype)
        b = jax.random.normal(key, (n, 5), dtype=dtype)
        
        l = jax.device_put(l, jax.devices("gpu")[0])
        b = jax.device_put(b, jax.devices("gpu")[0])
        
        x = jax.lax.linalg.triangular_solve(l, b, left_side=True, lower=True)
        
        residual = np.array(l) @ np.array(x) - np.array(b)
        self.assertLess(np.linalg.norm(residual), 1e-8)


if __name__ == '__main__':
    absltest.main()
