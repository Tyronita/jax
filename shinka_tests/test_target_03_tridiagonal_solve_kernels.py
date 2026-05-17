"""ShinkaEvolve Test Suite — Target 03: cpu/tridiagonal_solve_kernels.h

Tests non-perturbed tridiagonal solver. Used in banded matrix ops, signal processing.
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class ShinkaTridiagonalSolveKernelsTest(jtu.JaxTestCase):
    """Test suite for non-perturbed tridiagonal solve CPU kernels."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(123)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [2, 5, 10, 33, 64, 128, 256, 512]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_basic_solve(self, n, dtype):
        """Basic tridiagonal solve: verify A @ x == b."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype))))
        
        dl = jax.random.uniform(key, (n - 1,), dtype=dtype) * 0.5 + 0.1
        d = jax.random.uniform(key, (n,), dtype=dtype) * 2.0 + 1.0
        du = jax.random.uniform(key, (n - 1,), dtype=dtype) * 0.5 + 0.1
        b = jax.random.normal(key, (n, 5), dtype=dtype)
        
        # Build dense tridiagonal
        a = jnp.diag(d) + jnp.diag(dl, -1) + jnp.diag(du, 1)
        
        x = jax.lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=False)
        
        residual = jnp.linalg.norm(a @ x - b)
        self.assertLess(residual / (jnp.linalg.norm(b) + 1e-12),
                       1e-5 if dtype == np.float32 else 1e-12)

    def test_n_equals_2(self):
        """n=2 is a special case."""
        n = 2
        dtype = np.float64
        key = self.key
        
        dl = jnp.array([0.5], dtype=dtype)
        d = jnp.array([2.0, 3.0], dtype=dtype)
        du = jnp.array([0.3], dtype=dtype)
        b = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
        
        a = jnp.diag(d) + jnp.diag(dl, -1) + jnp.diag(du, 1)
        x = jax.lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=False)
        
        self.assertAllClose(a @ x, b, atol=1e-12)

    def test_vmap_batch(self):
        """Batched tridiagonal solves."""
        batch = 8
        n = 64
        dtype = np.float64
        key = self.key
        
        dl = jax.random.uniform(key, (batch, n - 1), dtype=dtype)
        d = jax.random.uniform(key, (batch, n), dtype=dtype) * 2.0 + 1.0
        du = jax.random.uniform(key, (batch, n - 1), dtype=dtype)
        b = jax.random.normal(key, (batch, n, 3), dtype=dtype)
        
        solve_batched = jax.vmap(
            lambda dl, d, du, b: jax.lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=False)
        )
        x = solve_batched(dl, d, du, b)
        self.assertEqual(x.shape, (batch, n, 3))
        self.assertTrue(jnp.all(jnp.isfinite(x)))


if __name__ == '__main__':
    absltest.main()
