"""Train two identical denoisers and compare: grammar-aware vs. sequence-only.

Both models share architecture, data, initial weights, and optimizer. The
*only* difference is whether the known program geometry (the grammar's
positional type-map) is injected as an additive logit bias during training
and sampling:

  * geometry-aware : flow constrained to grammar-legal tokens per slot
  * sequence-only  : unconstrained -- must learn validity from data

We periodically sample from each and report:

  * validity : fraction of samples that are syntactically valid trees
  * marg-TV  : mean per-slot total-variation distance between the sampled
               token marginals and the true data marginals (lower is better;
               it penalises both invalid tokens and wrong frequencies)

The headline: the geometry-aware model is valid by construction (100% at every
budget) and reaches a lower marginal-TV with far less training, because none of
its capacity or gradient is spent learning what the grammar already tells us.

Run from *outside* the JAX source tree so ``import jax`` resolves to an
installed jaxlib rather than this checkout, e.g.::

    cd hackathon/ast_flow_matching && python train.py
"""

from __future__ import annotations

import argparse
import functools

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util

import grammar
import model
import dfm

N_HEADS = 4


# --- hand-rolled Adam (keeps the demo dependency-free) ----------------------
def adam_init(params):
    z = tree_util.tree_map(jnp.zeros_like, params)
    return (z, tree_util.tree_map(jnp.zeros_like, params))


def adam_update(params, grads, state, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m, v = state
    m = tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    t = step + 1
    bc1 = 1 - jnp.power(b1, t)
    bc2 = 1 - jnp.power(b2, t)
    params = tree_util.tree_map(
        lambda p, m_, v_: p - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + eps),
        params, m, v)
    return params, (m, v)


def evaluate(samples, height, true_marg):
    samples = np.asarray(samples)
    validity = grammar.is_valid(samples, height).mean()
    n, L = samples.shape
    emp = np.stack([(samples == j).mean(axis=0) for j in range(grammar.V)], axis=1)
    tv = 0.5 * np.abs(emp - true_marg).sum(axis=1).mean()
    return float(validity), float(tv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=3)        # L = 2^(h+1)-1 = 15
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--sample-steps", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    H = args.height
    L = grammar.num_nodes(H)
    true_marg = grammar.true_marginals(H)

    # Known geometry -> additive logit bias (0 = legal, -1e9 = illegal).
    allowed = grammar.allowed_mask(H)
    bias_geo = jnp.where(jnp.asarray(allowed), 0.0, -1e9)
    bias_base = jnp.zeros((L, grammar.V))

    apply = functools.partial(model.forward, n_heads=N_HEADS)

    @jax.jit
    def train_step(params, opt_state, key, x1, bias, step):
        loss_val, grads = jax.value_and_grad(
            lambda p: dfm.loss(apply, p, key, x1, bias))(params)
        params, opt_state = adam_update(params, grads, opt_state, step, args.lr)
        return params, opt_state, loss_val

    jit_unmask = jax.jit(
        lambda p, x, t, dt, k, bias: dfm.unmask_step(apply, p, x, t, dt, k, bias))

    def run_sample(params, key, bias):
        x = jnp.full((args.n_eval, L), grammar.MASK, dtype=jnp.int32)
        dt = jnp.float32(1.0 / args.sample_steps)
        for i in range(args.sample_steps):
            key, sk = random.split(key)
            x = jit_unmask(params, x, jnp.float32(i / args.sample_steps), dt, sk, bias)
        return x

    key = random.PRNGKey(args.seed)
    key, ik = random.split(key)
    p0 = model.init_params(ik, L, grammar.VOCAB_IN, grammar.V)
    # Identical initial weights for a clean controlled comparison.
    params = {"geo": p0, "base": tree_util.tree_map(lambda a: a, p0)}
    opt = {"geo": adam_init(p0), "base": adam_init(p0)}
    bias = {"geo": bias_geo, "base": bias_base}

    data_rng = np.random.default_rng(args.seed)

    print(f"AST discrete flow matching | height={H} L={L} vocab={grammar.V} "
          f"batch={args.batch} steps={args.steps}\n")
    header = (f"{'step':>6} | {'loss_geo':>9} {'loss_base':>9} | "
              f"{'valid_geo':>9} {'valid_base':>10} | {'TV_geo':>7} {'TV_base':>7}")
    print(header)
    print("-" * len(header))

    for step in range(args.steps + 1):
        x1 = jnp.asarray(grammar.sample_data(data_rng, args.batch, H))
        step_arr = jnp.int32(step)
        key, kg, kb = random.split(key, 3)
        params["geo"], opt["geo"], lg = train_step(
            params["geo"], opt["geo"], kg, x1, bias["geo"], step_arr)
        params["base"], opt["base"], lb = train_step(
            params["base"], opt["base"], kb, x1, bias["base"], step_arr)

        if step % args.eval_every == 0:
            key, sg_k, sb_k = random.split(key, 3)
            sg = run_sample(params["geo"], sg_k, bias["geo"])
            sb = run_sample(params["base"], sb_k, bias["base"])
            vg, tg = evaluate(sg, H, true_marg)
            vb, tb = evaluate(sb, H, true_marg)
            print(f"{step:>6} | {float(lg):>9.4f} {float(lb):>9.4f} | "
                  f"{vg:>9.1%} {vb:>10.1%} | {tg:>7.4f} {tb:>7.4f}")

    # --- show a few generated programs -------------------------------------
    key, sg_k, sb_k = random.split(key, 3)
    sg = np.asarray(run_sample(params["geo"], sg_k, bias["geo"]))
    sb = np.asarray(run_sample(params["base"], sb_k, bias["base"]))
    print("\ngeometry-aware samples (valid by construction):")
    for row in sg[:6]:
        print("   ", grammar.to_expr(row, H))
    print("\nsequence-only samples (✓ = valid, ✗ = grammar violation):")
    valid_b = grammar.is_valid(sb[:6], H)
    for row, ok in zip(sb[:6], valid_b):
        print(f"   {'✓' if ok else '✗'} ", grammar.to_expr(row, H))


if __name__ == "__main__":
    main()
