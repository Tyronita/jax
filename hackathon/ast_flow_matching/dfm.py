"""Masked (absorbing) discrete flow matching.

This is the standard absorbing-state construction (a.k.a. masked diffusion /
discrete flow matching). Per slot, independently:

    p_t(x_t | x_1):   x_t = x_1  with prob t,   x_t = [MASK]  with prob 1 - t.

At ``t = 0`` everything is masked (the source); at ``t = 1`` we recover the
data. The network is trained as an ``x_1``-predictor on the masked slots, and
generation runs the absorbing process forward in time, unmasking slots at the
rate ``dt / (1 - t)`` and drawing the revealed token from the predicted
``p(x_1 | x_t)``.

The "known geometry" enters through ``logit_bias``: a ``[L, V]`` additive bias
that is ``0`` for grammar-legal (slot, token) pairs and a large negative number
otherwise. Pass the grammar bias to constrain the flow to the program manifold;
pass zeros for the unconstrained sequence-only baseline. Everything else --
architecture, data, optimizer, init -- is identical between the two, so the
bias isolates exactly what injecting the known structure buys you.

``apply`` is the model forward, partially applied with ``n_heads`` so it has the
signature ``apply(params, x, t) -> logits[B, L, V]``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random

from grammar import MASK


def corrupt(key, x1, t):
    """Forward / masking process: ``x_t`` given data ``x1`` and time ``t[B]``."""
    keep = random.uniform(key, x1.shape) < t[:, None]
    return jnp.where(keep, x1, MASK)


def loss(apply, params, key, x1, logit_bias):
    """Masked cross-entropy ``x_1``-prediction loss, averaged over masked slots."""
    B, _ = x1.shape
    kt, kc = random.split(key)
    t = random.uniform(kt, (B,), minval=1e-3, maxval=1.0)
    xt = corrupt(kc, x1, t)
    logits = apply(params, xt, t) + logit_bias[None]
    logp = jax.nn.log_softmax(logits, axis=-1)
    tgt = jnp.take_along_axis(logp, x1[..., None], axis=-1)[..., 0]   # [B, L]
    masked = (xt == MASK).astype(jnp.float32)
    return -(tgt * masked).sum() / jnp.clip(masked.sum(), 1.0)


def unmask_step(apply, params, x, t, dt, key, logit_bias):
    """One absorbing-process step: reveal some masked slots at time ``t``."""
    B, _ = x.shape
    logits = apply(params, x, jnp.full((B,), t)) + logit_bias[None]
    ks, ku = random.split(key)
    proposal = random.categorical(ks, logits, axis=-1)
    is_masked = x == MASK
    unmask_prob = jnp.clip(dt / (1.0 - t), 0.0, 1.0)
    do_unmask = is_masked & (random.uniform(ku, x.shape) < unmask_prob)
    return jnp.where(do_unmask, proposal, x)
