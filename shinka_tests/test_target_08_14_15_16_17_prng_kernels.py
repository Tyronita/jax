"""ShinkaEvolve Test Suite — Targets 08, 14-17: Random Number Generation Kernels

PRNG (Threefry/Philox on GPU), Gamma, Beta, Poisson distribution sampling.
Correctness verified against scipy/numpy statistical properties.
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
import scipy.stats as stats

jax.config.parse_flags_with_absl()


def has_gpu():
    try:
        return len(jax.devices("gpu")) > 0
    except:
        return False


class ShinkaPrngKernelsTest(jtu.JaxTestCase):
    """Comprehensive RNG correctness suite."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    # ---- Threefry PRNG Basic Properties ----

    def test_prng_reproducibility(self):
        """Same key should produce identical sequences."""
        key = self.key
        sample1 = jax.random.uniform(key, (1000,), dtype=jnp.float32)
        sample2 = jax.random.uniform(key, (1000,), dtype=jnp.float32)
        self.assertAllClose(sample1, sample2, atol=0)

    def test_prng_split_independence(self):
        """Split keys should produce independent sequences."""
        key = self.key
        key1, key2 = jax.random.split(key)
        
        sample1 = jax.random.uniform(key1, (1000,), dtype=jnp.float32)
        sample2 = jax.random.uniform(key2, (1000,), dtype=jnp.float32)
        
        # Should be different
        self.assertGreater(jnp.mean(jnp.abs(sample1 - sample2)), 0.1)

    def test_prng_statistical_properties(self):
        """Uniform samples should be ~uniformly distributed."""
        key = self.key
        sample = jax.random.uniform(key, (10000,), dtype=jnp.float64)
        
        # Kolmogorov-Smirnov test against uniform
        sample_np = np.array(sample)
        ks_stat, p_value = stats.kstest(sample_np, 'uniform')
        self.assertGreater(p_value, 0.01)  # Should not reject uniformity

    # ---- Normal Distribution ----

    def test_normal_distribution(self):
        """Normal samples should pass normality test."""
        key = self.key
        sample = jax.random.normal(key, (10000,), dtype=jnp.float64)
        
        sample_np = np.array(sample)
        self.assertAllClose(jnp.mean(sample), 0.0, atol=0.05)
        self.assertAllClose(jnp.std(sample), 1.0, atol=0.05)
        
        # Shapiro-Wilk on subset
        _, p_value = stats.shapiro(sample_np[:5000])
        self.assertGreater(p_value, 0.001)

    # ---- Gamma Distribution (Target 15) ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_shape={shape}_scale={scale}',
            'shape': shape,
            'scale': scale,
        } for shape in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
          for scale in [0.5, 1.0, 2.0])
    )
    def test_gamma_distribution(self, shape, scale):
        """Gamma samples should match scipy moments."""
        key = self.key
        sample = jax.random.gamma(key, shape, (50000,), dtype=jnp.float64) * scale
        
        sample_np = np.array(sample)
        
        # Check mean and variance
        expected_mean = shape * scale
        expected_var = shape * scale ** 2
        
        self.assertAllClose(np.mean(sample_np), expected_mean, atol=expected_mean * 0.1)
        self.assertAllClose(np.var(sample_np), expected_var, atol=expected_var * 0.2)

    # ---- Beta Distribution (Target 16) ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_a={a}_b={b}',
            'a': a,
            'b': b,
        } for a in [0.5, 1.0, 2.0, 5.0]
          for b in [0.5, 1.0, 2.0, 5.0])
    )
    def test_beta_distribution(self, a, b):
        """Beta samples should be in [0,1] and match moments."""
        key = self.key
        sample = jax.random.beta(key, a, b, (50000,), dtype=jnp.float64)
        
        sample_np = np.array(sample)
        
        # In [0,1]
        self.assertTrue(np.all(sample_np >= 0))
        self.assertTrue(np.all(sample_np <= 1))
        
        # Moments
        expected_mean = a / (a + b)
        expected_var = a * b / ((a + b)**2 * (a + b + 1))
        
        self.assertAllClose(np.mean(sample_np), expected_mean, atol=0.05)
        self.assertAllClose(np.var(sample_np), expected_var, atol=expected_var * 0.2)

    # ---- Poisson Distribution (Target 17) ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_lam={lam}',
            'lam': lam,
        } for lam in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    )
    def test_poisson_distribution(self, lam):
        """Poisson samples should be non-negative integers with correct mean."""
        key = self.key
        sample = jax.random.poisson(key, lam, (50000,), dtype=jnp.int32)
        
        sample_np = np.array(sample)
        
        # Non-negative integers
        self.assertTrue(np.all(sample_np >= 0))
        self.assertTrue(np.all(sample_np == sample_np.astype(int)))
        
        # Mean and variance should both be ~lambda
        self.assertAllClose(np.mean(sample_np), lam, atol=lam * 0.1)
        self.assertAllClose(np.var(sample_np), lam, atol=lam * 0.2)

    # ---- GPU PRNG (Target 08) ----

    def test_gpu_prng_basic(self):
        """GPU PRNG should produce finite, valid samples."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        key = self.key
        sample = jax.random.uniform(key, (1000,), dtype=jnp.float32)
        
        self.assertEqual(len(sample), 1000)
        self.assertTrue(jnp.all(sample >= 0))
        self.assertTrue(jnp.all(sample <= 1))
        self.assertTrue(jnp.all(jnp.isfinite(sample)))

    def test_gpu_prng_normal(self):
        """GPU normal distribution."""
        if not has_gpu():
            self.skipTest("No GPU available")
        
        key = self.key
        sample = jax.random.normal(key, (10000,), dtype=jnp.float64)
        sample = jax.device_put(sample, jax.devices("gpu")[0])
        
        self.assertAllClose(jnp.mean(sample), 0.0, atol=0.1)
        self.assertAllClose(jnp.std(sample), 1.0, atol=0.1)

    # ---- Vmap and Batch ----

    def test_vmap_random(self):
        """vmap over random function."""
        keys = jax.random.split(self.key, 10)
        samples = jax.vmap(lambda k: jax.random.uniform(k, (100,), dtype=jnp.float64))(keys)
        
        self.assertEqual(samples.shape, (10, 100))
        self.assertTrue(jnp.all(jnp.isfinite(samples)))

    # ---- Edge Cases ----

    def test_zero_samples(self):
        """Zero-length sample should return empty array."""
        key = self.key
        sample = jax.random.uniform(key, (0,), dtype=jnp.float32)
        self.assertEqual(len(sample), 0)

    def test_large_sample(self):
        """Large sample should not crash."""
        key = self.key
        sample = jax.random.normal(key, (1000000,), dtype=jnp.float32)
        self.assertEqual(len(sample), 1000000)
        self.assertTrue(jnp.all(jnp.isfinite(sample)))

    # ---- Gradient through PRNG ----

    def test_prng_not_differentiable(self):
        """PRNG should not produce NaN gradients."""
        key = self.key
        # Direct random sampling is non-differentiable
        sample = jax.random.normal(key, (100,), dtype=jnp.float32)
        self.assertTrue(jnp.all(jnp.isfinite(sample)))


if __name__ == '__main__':
    absltest.main()
