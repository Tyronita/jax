"""ShinkaEvolve Test Suite — Target 01: tridiagonal_solve_perturbed.h

Extensive correctness, regression, edge-case, and numerical stability tests.
Runs via: pytest shinka_tests/test_target_01_tridiagonal_solve.py -v
"""

import functools
import itertools

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import lax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.lax import linalg as lax_linalg

jax.config.parse_flags_with_absl()


# ----- Helper: generate random tridiagonal system -----

def random_tridiag(n, dtype, key):
    """Generate well-conditioned random tridiagonal system."""
    dl = jax.random.uniform(key, (n,), dtype=dtype) * 0.5 + 0.25
    d = jax.random.uniform(key, (n,), dtype=dtype) * 2.0 + 1.0  # strong diagonal
    du = jax.random.uniform(key, (n,), dtype=dtype) * 0.5 + 0.25
    nrhs = 10
    b = jax.random.normal(key, (n, nrhs), dtype=dtype)
    return dl, d, du, b


def random_ill_conditioned_tridiag(n, dtype, key):
    """Generate near-singular tridiagonal system."""
    dl = jax.random.uniform(key, (n,), dtype=dtype) * 0.01
    d = jax.random.uniform(key, (n,), dtype=dtype) * 0.02 + 1e-8  # tiny diagonal
    du = jax.random.uniform(key, (n,), dtype=dtype) * 0.01
    b = jax.random.normal(key, (n, 5), dtype=dtype)
    return dl, d, du, b


def make_tridiagonal_matrix(dl, d, du):
    """Dense tridiagonal matrix from bands."""
    n = len(d)
    a = jnp.diag(d) + jnp.diag(dl[1:], -1) + jnp.diag(du[:-1], 1)
    return a


# ----- Test Class -----

class ShinkaTridiagonalSolveTest(jtu.JaxTestCase):
    """Comprehensive correctness suite for tridiagonal_solve_perturbed.h."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    # ---- Basic Correctness ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [2, 3, 5, 10, 33, 64, 127, 256, 512, 1024]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_random_system_correctness(self, n, dtype):
        """Random tridiagonal system: verify A @ x == b."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype))))
        dl, d, du, b = random_tridiag(n, dtype, key)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        
        # Reconstruct A and verify
        a = make_tridiagonal_matrix(dl, d, du)
        residual = jnp.linalg.norm(a @ x - b)
        
        # Relative residual check
        b_norm = jnp.linalg.norm(b)
        self.assertLess(residual / (b_norm + 1e-12), 1e-5 if dtype == np.float32 else 1e-12)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [2, 5, 33, 128, 512]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_no_perturbation(self, n, dtype):
        """Test perturb_singular=False: exact solve for nonsingular A."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "no_perturb")))
        dl, d, du, b = random_tridiag(n, dtype, key)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=False)
        a = make_tridiagonal_matrix(dl, d, du)
        residual = jnp.linalg.norm(a @ x - b)
        b_norm = jnp.linalg.norm(b)
        self.assertLess(residual / (b_norm + 1e-12), 1e-5 if dtype == np.float32 else 1e-12)

    # ---- Edge Cases ----

    def test_zero_rhs(self):
        """Zero RHS should return zero solution."""
        n = 64
        dtype = np.float64
        key = self.key
        dl, d, du, _ = random_tridiag(n, dtype, key)
        b = jnp.zeros((n, 1), dtype=dtype)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        self.assertAllClose(x, jnp.zeros_like(x), atol=1e-12)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}',
            'n': n,
        } for n in [2, 3, 4])
    )
    def test_small_n_corner_cases(self, n):
        """Test n=2,3,4 which may have special code paths."""
        dtype = np.float64
        key = jax.random.fold_in(self.key, n)
        dl, d, du, b = random_tridiag(n, dtype, key)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        a = make_tridiagonal_matrix(dl, d, du)
        self.assertAllClose(a @ x, b, atol=1e-10)

    def test_perturbation_activates(self):
        """Force perturbation: create exactly singular tridiagonal matrix."""
        n = 64
        dtype = np.float64
        # Singular matrix: repeated rows
        dl = jnp.ones(n, dtype=dtype) * 0.5
        d = jnp.ones(n, dtype=dtype)  # This creates singularity
        du = jnp.ones(n, dtype=dtype) * 0.5
        b = jnp.ones((n, 3), dtype=dtype)
        
        # With perturbation, should succeed
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        # Just verify it doesn't crash and returns finite values
        self.assertTrue(jnp.all(jnp.isfinite(x)))

    def test_pivoting_required(self):
        """Matrix requiring partial pivoting: |sub-diagonal| > |diagonal| at some point."""
        n = 128
        dtype = np.float64
        key = self.key
        
        # Create matrix where pivoting is definitely needed
        dl = jax.random.uniform(key, (n,), dtype=dtype) * 10.0  # Large sub-diagonal
        d = jax.random.uniform(key, (n,), dtype=dtype) * 0.1 + 0.01  # Small diagonal
        du = jax.random.uniform(key, (n,), dtype=dtype) * 0.5
        b = jax.random.normal(key, (n, 5), dtype=dtype)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        # Verify finite
        self.assertTrue(jnp.all(jnp.isfinite(x)))

    # ---- Multiple RHS ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_nrhs={nrhs}_dtype={dtype.__name__}',
            'n': n,
            'nrhs': nrhs,
            'dtype': dtype,
        } for n in [32, 128, 512]
          for nrhs in [1, 3, 10, 50]
          for dtype in [np.float32, np.float64])
    )
    def test_multiple_rhs(self, n, nrhs, dtype):
        """Test varying number of right-hand sides."""
        key = jax.random.fold_in(self.key, hash((n, nrhs, str(dtype))))
        dl, d, du, _ = random_tridiag(n, dtype, key)
        b = jax.random.normal(key, (n, nrhs), dtype=dtype)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        a = make_tridiagonal_matrix(dl, d, du)
        
        for i in range(nrhs):
            residual = jnp.linalg.norm(a @ x[:, i] - b[:, i])
            b_norm = jnp.linalg.norm(b[:, i])
            self.assertLess(residual / (b_norm + 1e-12), 1e-4 if dtype == np.float32 else 1e-11)

    # ---- JIT Compilation & Gradients ----

    def test_jit_compiles(self):
        """Verify JIT compilation works and is deterministic."""
        n = 256
        dtype = np.float64
        dl, d, du, b = random_tridiag(n, dtype, self.key)
        
        solve_fn = lambda dl, d, du, b: lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        jitted = jax.jit(solve_fn)
        
        x1 = jitted(dl, d, du, b)
        x2 = jitted(dl, d, du, b)
        self.assertAllClose(x1, x2, atol=0)

    def test_gradients_exist(self):
        """Verify autodiff works: value_and_grad should produce finite results."""
        n = 64
        dtype = np.float64
        dl, d, du, b = random_tridiag(n, dtype, self.key)
        
        def loss(dl, d, du, b):
            x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
            return jnp.sum(x ** 2)
        
        val, grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))(dl, d, du, b)
        
        for g in grads:
            self.assertTrue(jnp.all(jnp.isfinite(g)))

    # ---- Vmap / Batch dimensions ----

    def test_vmap_batch(self):
        """Test vmap over batch dimension of tridiagonal systems."""
        batch = 16
        n = 64
        dtype = np.float64
        key = self.key
        
        dl = jax.random.uniform(key, (batch, n), dtype=dtype)
        d = jax.random.uniform(key, (batch, n), dtype=dtype) * 2.0 + 1.0
        du = jax.random.uniform(key, (batch, n), dtype=dtype)
        b = jax.random.normal(key, (batch, n, 5), dtype=dtype)
        
        batched_solve = jax.vmap(
            lambda dl, d, du, b: lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        )
        x = batched_solve(dl, d, du, b)
        self.assertEqual(x.shape, (batch, n, 5))
        self.assertTrue(jnp.all(jnp.isfinite(x)))

    # ---- Numerical Stability: ill-conditioned ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [32, 128, 512]
          for dtype in [np.float64])  # Only float64 for stability tests
    )
    def test_ill_conditioned_system(self, n, dtype):
        """Near-singular system: verify solution is as accurate as possible."""
        key = jax.random.fold_in(self.key, hash((n, "ill")))
        dl, d, du, b = random_ill_conditioned_tridiag(n, dtype, key)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        a = make_tridiagonal_matrix(dl, d, du)
        
        residual = jnp.linalg.norm(a @ x - b)
        # For ill-conditioned systems, just check it's finite and residual is small-ish
        self.assertTrue(jnp.all(jnp.isfinite(x)))
        self.assertLess(residual, 1e-6)

    # ---- Regression Tests from ShinkaEvolve History ----

    def test_regression_gen5_loop_unrolling(self):
        """Regression: gen-5 loop unrolling in MaybePerturbPivot. Ensure no crash."""
        n = 256
        dtype = np.float32
        dl, d, du, b = random_tridiag(n, dtype, self.key)
        
        # Run many times to stress the perturbation loop
        for _ in range(100):
            x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
            self.assertTrue(jnp.all(jnp.isfinite(x)))

    def test_regression_gen18_reciprocal_precompute(self):
        """Regression: gen-18 reciprocal precompute. Verify numerical unchanged."""
        n = 512
        dtype = np.float64
        key = self.key
        dl, d, du, b = random_tridiag(n, dtype, key)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        # Check solution is well-conditioned (not NaN/inf)
        self.assertTrue(jnp.all(jnp.isfinite(x)))
        # Verify residual
        a = make_tridiagonal_matrix(dl, d, du)
        residual = jnp.linalg.norm(a @ x - b)
        self.assertLess(residual / jnp.linalg.norm(b), 1e-12)

    # ---- Dtype Coverage ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_dtype={dtype.__name__}',
            'dtype': dtype,
        } for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_all_dtypes_basic(self, dtype):
        """Smoke test for all supported dtypes."""
        n = 64
        key = self.key
        dl, d, du, b = random_tridiag(n, dtype, key)
        
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        self.assertEqual(x.dtype, dtype)
        self.assertTrue(jnp.all(jnp.isfinite(x)))

    # ---- Performance Hint Tests (verify function still callable) ----

    def test_large_n_performance_hint(self):
        """Large n test: just verify it doesn't OOM or hang."""
        n = 8192
        dtype = np.float64
        key = self.key
        dl, d, du, b = random_tridiag(n, dtype, key)
        
        # Single call — don't benchmark here, just correctness
        x = lax.linalg.tridiagonal_solve(dl, d, du, b, perturb_singular=True)
        self.assertTrue(jnp.all(jnp.isfinite(x)))
        self.assertEqual(x.shape, (n, 10))


if __name__ == '__main__':
    absltest.main()
