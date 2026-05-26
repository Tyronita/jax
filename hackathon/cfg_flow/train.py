"""Train three denoisers on the same program-DAG data and compare.

  seq        flat sequence baseline      (no structure at all)
  seq+mask   knows node arities          (valid, but blind to the graph)
  graph      full known geometry         (arities + data/control-flow edges)

Metrics, evaluated by sampling programs onto held-out scaffolds:

  valid   fraction of generated programs that are structurally valid
  exec    fraction that run without error on random inputs
  dfg-TV  how far the generated *data-flow-conditioned* op distribution is from
          the true one (mean TV of P(child_op | parent ops); lower is better) --
          this is the signal that lives on the graph edges

Expected story: arity-masking buys validity (seq -> seq+mask); the edge bias
buys the data-flow distribution (seq+mask -> graph).

Runs on GPU automatically if available. Presets:
  python train.py --preset cpu     # tiny, for a laptop / smoke test
  python train.py --preset a100    # ~30M params, single A100
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import data
from model import Config, GraphTransformer
import dfm

PRESETS = {
    "cpu":  dict(N=16, d=128, n_layers=3, n_heads=4, d_ff=256,
                 steps=1500, batch=128, n_train=8000,  eval_every=300),
    "a100": dict(N=64, d=512, n_layers=8, n_heads=8, d_ff=2048,
                 steps=20000, batch=512, n_train=200000, eval_every=1000),
}

VARIANTS = {
    "seq":      dict(use_type=False, use_rel=False, use_mask=False),
    "seq+mask": dict(use_type=True,  use_rel=False, use_mask=True),
    "graph":    dict(use_type=True,  use_rel=True,  use_mask=True),
}


def dfg_conditional_tv(gen_op, parents, node_type, cond_true, min_support=20):
    """Mean TV between generated and true P(child_op | parent0_op, parent1_op)."""
    bin_ids = data.LEGAL_OPS[data.BINARY]
    id2bin = {o: k for k, o in enumerate(bin_ids)}
    counts = np.zeros((data.V, data.V, len(bin_ids)))
    is_bin = node_type == data.BINARY
    for s in range(gen_op.shape[0]):
        for i in np.where(is_bin[s])[0]:
            # condition on the OPS at the two parent nodes (not their indices)
            o0, o1 = gen_op[s, parents[s, i, 0]], gen_op[s, parents[s, i, 1]]
            c = gen_op[s, i]
            if c in id2bin and 0 <= o0 < data.V and 0 <= o1 < data.V:
                counts[o0, o1, id2bin[c]] += 1
    tv_sum = w_sum = 0.0
    for p0 in range(data.V):
        for p1 in range(data.V):
            tot = counts[p0, p1].sum()
            if tot >= min_support:
                phat = counts[p0, p1] / tot
                tv = 0.5 * np.abs(phat - cond_true[p0, p1]).sum()
                tv_sum += tv * tot
                w_sum += tot
    return tv_sum / w_sum if w_sum else float("nan")


def evaluate(model, eval_batch, scaffold_np, dist, sample_steps, topo):
    gen = dfm.sample(model, eval_batch, steps=sample_steps, topo=topo).cpu().numpy()
    op_np, type_np, par_np = scaffold_np
    valid = data.is_valid(gen, type_np)
    validity = float(valid.mean())
    # executability on a subset of structurally valid programs
    n_check = min(128, gen.shape[0])
    ok = 0
    for s in range(n_check):
        if not valid[s]:
            continue
        try:
            data.execute(gen[s], par_np[s], type_np[s],
                         inputs=[3, 5, 7, 11, 13, 17, 19, 23])
            ok += 1
        except Exception:
            pass
    n_valid = int(valid[:n_check].sum())
    execu = ok / n_valid if n_valid else 0.0
    tv = dfg_conditional_tv(gen, par_np, type_np, dist["cond"])
    return validity, execu, tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-steps", type=int, default=48)
    ap.add_argument("--n-eval", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.preset]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"device={device} preset={args.preset} cfg={cfg}\n")

    dist = data.make_distribution(args.seed)
    legal_mask = torch.as_tensor(data.legal_mask(), device=device)
    N = cfg["N"]

    # training set + a held-out eval set of scaffolds
    ops, types, pars = data.sample_dataset(args.seed, cfg["n_train"], N, dist)
    e_ops, e_types, e_pars = data.sample_dataset(args.seed + 1, args.n_eval, N, dist)
    eval_batch = dfm.make_batch(e_ops, e_types, e_pars, legal_mask, device)
    eval_scaffold = (e_ops, e_types, e_pars)

    models, opts = {}, {}
    for name, flags in VARIANTS.items():
        m = GraphTransformer(Config(N=N, d=cfg["d"], n_layers=cfg["n_layers"],
                                    n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], **flags)).to(device)
        models[name] = m
        opts[name] = torch.optim.AdamW(m.parameters(), lr=3e-4)
    print("params: " + "  ".join(f"{n}={models[n].n_params()/1e6:.1f}M" for n in VARIANTS) + "\n")

    hdr = f"{'step':>6} | " + " | ".join(f"{n:>22}" for n in VARIANTS)
    sub = f"{'':>6} | " + " | ".join(f"{'valid':>6} {'exec':>5} {'dfgTV':>7}" for _ in VARIANTS)
    print(hdr); print(sub); print("-" * len(sub))

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for step in range(cfg["steps"] + 1):
        idx = rng.integers(0, cfg["n_train"], size=cfg["batch"])
        batch = dfm.make_batch(ops[idx], types[idx], pars[idx], legal_mask, device)
        for name in VARIANTS:
            opts[name].zero_grad()
            l = dfm.loss(models[name], batch)
            l.backward()
            opts[name].step()

        if step % cfg["eval_every"] == 0:
            cells = []
            for name in VARIANTS:
                models[name].eval()
                v, e, tv = evaluate(models[name], eval_batch, eval_scaffold,
                                    dist, args.sample_steps, topo=VARIANTS[name]["use_rel"])
                models[name].train()
                cells.append(f"{v:>6.1%} {e:>5.0%} {tv:>7.3f}")
            print(f"{step:>6} | " + " | ".join(cells))

    print(f"\ntrained in {time.time() - t0:.0f}s\n")

    # show generated programs from the full-geometry model
    g = dfm.sample(models["graph"], eval_batch, steps=args.sample_steps, topo=True).cpu().numpy()
    print("graph-model sample (valid + runnable by construction):")
    print(data.to_code(g[0], e_pars[0], e_types[0]))
    s = dfm.sample(models["seq"], eval_batch, steps=args.sample_steps).cpu().numpy()
    bad = np.where(~data.is_valid(s, e_types))[0]
    if len(bad):
        print("\nseq-baseline sample (structure violations):")
        print(data.to_code(s[bad[0]], e_pars[bad[0]], e_types[bad[0]]))


if __name__ == "__main__":
    main()
