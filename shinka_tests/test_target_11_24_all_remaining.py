"""ShinkaEvolve Test Suite — Targets 11-13, 18-19, 20-21, 22-23, 24-25

FFT, Sort, Special Functions, Convolution, Reduction, Collective ops.
Covers the remaining high-value kernels.
"""

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
import scipy.special as sp_special
import scipy.fft as sp_fft

jax.config.parse_flags_with_absl()


class ShinkaRemainingKernelsTest(jtu.JaxTestCase):
    """Comprehensive suite for remaining kernel targets."""

    def setUp(self):
        super().setUp()
        self.key = jax.random.PRNGKey(42)

    # ---- FFT (Targets 11-13) ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [16, 32, 64, 128, 256, 512, 1024, 2048]
          for dtype in [np.float32, np.float64, np.complex64, np.complex128])
    )
    def test_fft_roundtrip(self, n, dtype):
        """FFT roundtrip: ifft(fft(x)) == x."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "fft")))
        x = jax.random.normal(key, (n,), dtype=dtype)
        
        y = jnp.fft.fft(x)
        x_reconstructed = jnp.fft.ifft(y)
        
        self.assertAllClose(x, x_reconstructed,
                           atol=1e-4 if dtype in (np.float32, np.complex64) else 1e-12)

    def test_fft_2d(self):
        """2D FFT for image-like data."""
        shape = (128, 128)
        dtype = np.float64
        key = self.key
        
        x = jax.random.normal(key, shape, dtype=dtype)
        y = jnp.fft.fft2(x)
        x_rec = jnp.fft.ifft2(y)
        
        self.assertAllClose(x, x_rec, atol=1e-11)

    def test_rfft_irfft(self):
        """Real FFT roundtrip."""
        n = 1024
        dtype = np.float64
        key = self.key
        
        x = jax.random.normal(key, (n,), dtype=dtype)
        y = jnp.fft.rfft(x)
        x_rec = jnp.fft.irfft(y, n=n)
        
        self.assertAllClose(x, x_rec, atol=1e-11)

    def test_fft_shift(self):
        """FFT shift properties."""
        n = 256
        x = jnp.zeros(n)
        x = x.at[128].set(1.0)
        
        y = jnp.fft.fft(x)
        # DC at index 0
        self.assertGreater(jnp.abs(y[0]), 0.9)
        # Shifted DC to center
        y_shifted = jnp.fft.fftshift(y)
        self.assertGreater(jnp.abs(y_shifted[128]), 0.9)

    # ---- Sort (Targets 18-19) ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_n={n}_dtype={dtype.__name__}',
            'n': n,
            'dtype': dtype,
        } for n in [16, 64, 256, 1024, 4096]
          for dtype in [np.float32, np.float64, np.int32, np.int64])
    )
    def test_sort_ascending(self, n, dtype):
        """Sort should produce monotonically increasing sequence."""
        key = jax.random.fold_in(self.key, hash((n, str(dtype), "sort")))
        x = jax.random.normal(key, (n,), dtype=dtype if 'float' in str(dtype) else jnp.int32)
        
        sorted_x = jnp.sort(x)
        
        # Monotonically increasing
        diffs = sorted_x[1:] - sorted_x[:-1]
        self.assertTrue(jnp.all(diffs >= 0))

    def test_argsort(self):
        """Argsort indices should reconstruct sorted array."""
        n = 1024
        key = self.key
        x = jax.random.normal(key, (n,), dtype=jnp.float64)
        
        idx = jnp.argsort(x)
        sorted_x = x[idx]
        
        # Should be same as direct sort
        self.assertAllClose(sorted_x, jnp.sort(x), atol=0)

    def test_topk(self):
        """Top-k should return k largest elements."""
        n = 1024
        k = 10
        key = self.key
        x = jax.random.normal(key, (n,), dtype=jnp.float64)
        
        topk_vals, topk_idx = jax.lax.top_k(x, k)
        
        self.assertEqual(len(topk_vals), k)
        # All top-k should be >= all others
        min_topk = jnp.min(topk_vals)
        self.assertTrue(jnp.all(x <= min_topk) | (jnp.sum(x > min_topk) <= k))

    def test_sort_axis(self):
        """Sort along specific axis."""
        shape = (10, 256)
        key = self.key
        x = jax.random.normal(key, shape, dtype=jnp.float64)
        
        sorted_axis0 = jnp.sort(x, axis=0)
        sorted_axis1 = jnp.sort(x, axis=1)
        
        # Each column should be sorted in axis=0
        diffs = sorted_axis0[1:, :] - sorted_axis0[:-1, :]
        self.assertTrue(jnp.all(diffs >= 0))
        
        # Each row should be sorted in axis=1
        diffs = sorted_axis1[:, 1:] - sorted_axis1[:, :-1]
        self.assertTrue(jnp.all(diffs >= 0))

    # ---- Special Functions (Targets 20-21) ----

    def test_erf(self):
        """Error function: erf(0) == 0, erf(inf) == 1."""
        self.assertAllClose(jax.scipy.special.erf(0.0), 0.0, atol=1e-15)
        self.assertAllClose(jax.scipy.special.erf(jnp.inf), 1.0, atol=1e-10)
        self.assertAllClose(jax.scipy.special.erf(-jnp.inf), -1.0, atol=1e-10)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_x={x}',
            'x': x,
        } for x in [0.5, 1.0, 2.0, 5.0, 10.0])
    )
    def test_gamma(self, x):
        """Gamma function: verify against scipy."""
        result = jax.scipy.special.gamma(x)
        expected = sp_special.gamma(x)
        self.assertAllClose(result, expected, atol=1e-10)

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_x={x}',
            'x': x,
        } for x in [0.5, 1.0, 2.0, 5.0])
    )
    def test_bessel_j0(self, x):
        """Bessel function J0: verify against scipy."""
        result = jax.scipy.special.jv(0, x)
        expected = sp_special.jv(0, x)
        self.assertAllClose(result, expected, atol=1e-10)

    def test_logit(self):
        """Logit function properties."""
        x = jnp.array([0.25, 0.5, 0.75])
        result = jax.scipy.special.logit(x)
        # logit(0.5) = 0
        self.assertAllClose(result[1], 0.0, atol=1e-15)
        # logit(0.25) < 0, logit(0.75) > 0
        self.assertLess(result[0], 0)
        self.assertGreater(result[2], 0)

    # ---- Convolution (Targets 22-23) ----

    def test_conv_1d(self):
        """1D convolution."""
        n = 256
        kernel_size = 5
        key = self.key
        
        x = jax.random.normal(key, (n,), dtype=jnp.float64)
        k = jax.random.normal(key, (kernel_size,), dtype=jnp.float64)
        
        result = jax.lax.conv(
            x[jnp.newaxis, jnp.newaxis, :],
            k[jnp.newaxis, jnp.newaxis, :],
            window_strides=(1,),
            padding='SAME'
        )
        
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        self.assertEqual(result.shape[2], n)

    def test_conv_2d(self):
        """2D convolution (image filter)."""
        h, w = 64, 64
        kh, kw = 3, 3
        key = self.key
        
        x = jax.random.normal(key, (1, 1, h, w), dtype=jnp.float64)
        k = jax.random.normal(key, (1, 1, kh, kw), dtype=jnp.float64)
        
        result = jax.lax.conv(
            x,
            k,
            window_strides=(1, 1),
            padding='SAME'
        )
        
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        self.assertEqual(result.shape, (1, 1, h, w))

    # ---- Reductions (Target 24) ----

    @parameterized.named_parameters(
        jtu.cases_from_list({
            'testcase_name': f'_shape={shape}_axis={axis}',
            'shape': shape,
            'axis': axis,
        } for shape in [(128,), (64, 64), (32, 32, 32)]
          for axis in [None, 0, -1])
    )
    def test_sum_reduction(self, shape, axis):
        """Sum reduction correctness."""
        key = self.key
        x = jax.random.normal(key, shape, dtype=jnp.float64)
        
        result = jnp.sum(x, axis=axis)
        expected = np.sum(np.array(x), axis=axis)
        
        self.assertAllClose(result, expected, atol=1e-12)

    def test_mean_std(self):
        """Mean and std correctness."""
        key = self.key
        x = jax.random.normal(key, (10000,), dtype=jnp.float64)
        
        self.assertAllClose(jnp.mean(x), 0.0, atol=0.1)
        self.assertAllClose(jnp.std(x), 1.0, atol=0.1)

    def test_softmax(self):
        """Softmax sums to 1."""
        key = self.key
        x = jax.random.normal(key, (128,), dtype=jnp.float64)
        
        result = jax.nn.softmax(x)
        self.assertAllClose(jnp.sum(result), 1.0, atol=1e-12)
        self.assertTrue(jnp.all(result >= 0))
        self.assertTrue(jnp.all(result <= 1))

    def test_argmax(self):
        """Argmax returns valid index."""
        key = self.key
        x = jax.random.normal(key, (1024,), dtype=jnp.float64)
        
        idx = jnp.argmax(x)
        self.assertTrue(0 <= idx < len(x))
        self.assertEqual(x[idx], jnp.max(x))

    # ---- Collective Ops (Target 25) ----
    # (These require multiple devices, so we only define the test structure)

    def test_collective_ops_exist(self):
        """Verify collective operations are importable."""
        # psum, pmean, pmax, pmin, all_gather, all_to_all
        from jax.lax import psum, pmean, pmax, pmin
        self.assertTrue(callable(psum))
        self.assertTrue(callable(pmean))
        self.assertTrue(callable(pmax))
        self.assertTrue(callable(pmin))

    # ---- Edge Cases ----

    def test_fft_zero_input(self):
        """FFT of zeros should be zeros."""
        x = jnp.zeros(128)
        y = jnp.fft.fft(x)
        self.assertAllClose(y, jnp.zeros_like(y), atol=1e-15)

    def test_sort_single_element(self):
        """Sort of single element."""
        x = jnp.array([5.0])
        self.assertAllClose(jnp.sort(x), x, atol=0)

    def test_conv_identity(self):
        """Convolution with delta function."""
        n = 64
        x = jax.random.normal(self.key, (1, 1, n), dtype=jnp.float64)
        k = jnp.zeros((1, 1, 3))
        k = k.at[0, 0, 1].set(1.0)  # Delta at center
        
        result = jax.lax.conv(x, k, window_strides=(1,), padding='SAME')
        self.assertAllClose(result, x, atol=1e-12)


if __name__ == '__main__':
    absltest.main()
