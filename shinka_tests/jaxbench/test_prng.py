"""Threefry PRNG correctness: determinism + distribution sanity.

Targets prng_kernels (GPU). A PRNG mutation must stay deterministic (same key ->
same bits) and keep the intended distribution.
"""
import numpy as np
import pytest

from _util import DEVICES

jax = pytest.importorskip("jax")
import jax.numpy as jnp


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n", [1 << 12, 1 << 16, 1 << 20])
def test_determinism(device, n):
    dev = jax.devices(device)[0]
    with jax.default_device(dev):
        k = jax.random.PRNGKey(0)
        a = jax.random.uniform(k, (n,))
        b = jax.random.uniform(k, (n,))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("device", DEVICES)
def test_uniform_distribution(device):
    dev = jax.devices(device)[0]
    with jax.default_device(dev):
        x = np.asarray(jax.random.uniform(jax.random.PRNGKey(1), (1 << 20,)))
    assert abs(x.mean() - 0.5) < 1e-2
    assert abs(x.var() - 1 / 12) < 1e-2
    assert x.min() >= 0.0 and x.max() < 1.0


@pytest.mark.parametrize("device", DEVICES)
def test_normal_distribution(device):
    dev = jax.devices(device)[0]
    with jax.default_device(dev):
        x = np.asarray(jax.random.normal(jax.random.PRNGKey(2), (1 << 20,)))
    assert abs(x.mean()) < 1e-2
    assert abs(x.var() - 1.0) < 2e-2


@pytest.mark.parametrize("device", DEVICES)
def test_distinct_keys_differ(device):
    dev = jax.devices(device)[0]
    with jax.default_device(dev):
        a = jax.random.uniform(jax.random.PRNGKey(0), (1 << 16,))
        b = jax.random.uniform(jax.random.PRNGKey(1), (1 << 16,))
    assert not np.array_equal(np.asarray(a), np.asarray(b))
