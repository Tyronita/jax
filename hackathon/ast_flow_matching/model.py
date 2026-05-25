"""A tiny Transformer denoiser, implemented in pure JAX.

The network is the ``x_1``-predictor for discrete flow matching: given a
partially-masked tree ``x_t`` and a time ``t``, it outputs per-slot logits
over the real vocabulary. It is deliberately small (a couple of attention
blocks) and dependency-free -- parameters are a plain pytree of arrays and
the optimizer is hand-rolled in ``train.py`` -- so the whole prototype runs
on CPU in a couple of minutes.

Note the attention is *non-causal*: this is not an autoregressive language
model. Every slot attends to every other slot, which is what lets the
denoiser reason about the whole (partial) tree at once.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random


def _glorot(key, shape):
    fan_in = shape[0]
    return random.normal(key, shape) * (1.0 / jnp.sqrt(fan_in))


def init_params(key, L, vocab_in, v_out, d=64, n_layers=2, d_ff=128):
    keys = list(random.split(key, 6 + n_layers * 6))
    kp = iter(keys)

    def nxt():
        return next(kp)

    params = {
        "tok_emb": random.normal(nxt(), (vocab_in, d)) * 0.02,
        "pos_emb": random.normal(nxt(), (L, d)) * 0.02,
        "t1": (_glorot(nxt(), (1, d)), jnp.zeros((d,))),
        "t2": (_glorot(nxt(), (d, d)), jnp.zeros((d,))),
        "blocks": [],
        "lnf": (jnp.ones((d,)), jnp.zeros((d,))),
        "head": (_glorot(nxt(), (d, v_out)), jnp.zeros((v_out,))),
    }
    for _ in range(n_layers):
        params["blocks"].append({
            "ln1": (jnp.ones((d,)), jnp.zeros((d,))),
            "Wq": _glorot(nxt(), (d, d)),
            "Wk": _glorot(nxt(), (d, d)),
            "Wv": _glorot(nxt(), (d, d)),
            "Wo": _glorot(nxt(), (d, d)),
            "ln2": (jnp.ones((d,)), jnp.zeros((d,))),
            "ff1": (_glorot(nxt(), (d, d_ff)), jnp.zeros((d_ff,))),
            "ff2": (_glorot(nxt(), (d_ff, d)), jnp.zeros((d,))),
        })
    return params


def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def forward(params, x, t, n_heads):
    """``x``: ``[B, L]`` int tokens. ``t``: ``[B]`` float in [0, 1].

    Returns logits ``[B, L, v_out]``.
    """
    B, L = x.shape
    d = params["tok_emb"].shape[1]
    dh = d // n_heads

    h = params["tok_emb"][x] + params["pos_emb"][None]            # [B, L, d]
    te = jax.nn.gelu(t[:, None] @ params["t1"][0] + params["t1"][1])
    te = te @ params["t2"][0] + params["t2"][1]                   # [B, d]
    h = h + te[:, None, :]

    for blk in params["blocks"]:
        a = _layernorm(h, *blk["ln1"])
        q = (a @ blk["Wq"]).reshape(B, L, n_heads, dh).transpose(0, 2, 1, 3)
        k = (a @ blk["Wk"]).reshape(B, L, n_heads, dh).transpose(0, 2, 1, 3)
        v = (a @ blk["Wv"]).reshape(B, L, n_heads, dh).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(dh)
        attn = jax.nn.softmax(scores, axis=-1)
        o = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, d) @ blk["Wo"]
        h = h + o
        f = _layernorm(h, *blk["ln2"])
        f = jax.nn.gelu(f @ blk["ff1"][0] + blk["ff1"][1])
        f = f @ blk["ff2"][0] + blk["ff2"][1]
        h = h + f

    h = _layernorm(h, *params["lnf"])
    return h @ params["head"][0] + params["head"][1]
