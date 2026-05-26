# Flow matching over program graphs — beyond the syntax tree

> Poolside hackathon. The novel angle: discrete flow matching whose "known
> geometry" is a program's **data-flow and control-flow graph**, not just its
> syntax tree.

This is the second, more ambitious prototype. The first (`../ast_flow_matching`)
showed the idea on a syntax tree, where the only structure is "this slot is an
operator, that slot is a value." But **AST diffusion has already been done**
([Diffusion on Syntax Trees, Kapur et al., 2024](https://arxiv.org/abs/2405.20519);
grammar VAEs). The research gap is the *semantic* graph: nobody has done
flow-matching / diffusion that generates over **CFG/DFG/CPG** structure for code.
This prototype targets exactly that gap.

## The setup

Programs are small **computation DAGs** — a tiny SSA-style IR with genuine
structure that an AST alone doesn't capture:

- **data flow:** every non-leaf node consumes the values of earlier nodes (its
  *parents*). Those edges are the data-flow graph.
- **control flow:** `SELECT(cond, a, b)` picks a branch based on a predicate
  (the SSA / phi-function encoding of an `if`), so its cond-edge is a control
  dependency.

A program's *scaffold* — node arities plus the wiring (with operand roles) — is
the **known geometry**. It's computable, it constrains which op is legal at each
node, and it says exactly which earlier nodes each node depends on. The task:
**given a scaffold, flow-match the operations onto it.**

## The experiment: what does each piece of known geometry buy you?

Three denoisers, *identical* architecture/data/optimizer, differing only in how
much of the geometry they get (`model.py`):

| variant | knows arities | sees edges | decoding order |
|---|---|---|---|
| `seq` | ✗ | ✗ | random |
| `seq+mask` | ✓ (legal-op mask) | ✗ | random |
| `graph` | ✓ | ✓ (edge-biased attention) | **topological** (parents first) |

Two metrics, measured by sampling programs onto held-out scaffolds:

- **validity** — fraction structurally valid (op matches each node's arity)
- **dfg-TV** — distance between the generated and true **data-flow-conditioned**
  op distribution `P(child_op | parent ops)`. This signal lives *on the edges*:
  a node's op depends on its parents' ops. Lower is better.

## Result (CPU preset, ~5 min, ~0.4M params each)

```
  step |          seq        |       seq+mask      |        graph
       | valid  exec  dfgTV  | valid  exec  dfgTV  | valid  exec  dfgTV
     0 |  0.0%   0%   0.496  | 100%  100%   0.495  | 100%  100%   0.493
   900 |  0.4%   0%   0.479  | 100%  100%   0.470  | 100%  100%   0.456
  1500 |  0.4%   0%   0.488  | 100%  100%   0.468  | 100%  100%   0.338  ↓
```

The decomposition is clean:

1. **Arity mask → validity.** `seq` ≈ 0% valid (it can't know which of the
   varying-per-program slots are leaves); `seq+mask` is 100% by construction.
2. **Edges + topological decoding → data flow.** `seq+mask` is valid but its
   dfg-TV stays flat (~0.47): with no edges it can't tell which earlier nodes
   are a node's parents, so it can only reproduce the *marginal* op frequencies.
   `graph` drives dfg-TV down (0.49 → 0.34, still falling at 1500 steps; a
   larger/longer run reaches ~0.13) because it reads the data flow and reveals
   parents before children.

**Topological decoding turned out to be essential** — and it's a nice
instantiation of the thesis. With random-order unmasking even the graph model
fails on dfg-TV, because a node is often generated *before* its parents and so
can't condition on them. Using the known DAG to unmask in dependency order
("pathfinding over the mesh") is what makes the structure pay off. Only the
graph model has the edges to do it.

## Files

```
data.py    the DSL: computation DAGs, the data-flow-conditioned distribution,
           executor, validity check, SSA pretty-printer (pure NumPy)
model.py   relational graph Transformer; edges enter as per-head attention bias
dfm.py     masked discrete flow matching: corrupt / loss / topological sampler
train.py   trains the 3 variants, prints the comparison, shows samples
```

## Running it

```bash
pip install -r requirements.txt
python train.py --preset cpu      # ~5 min on a laptop; shows the trend
python train.py --preset a100     # ~30M params, single A100, full separation
```

It auto-selects CUDA when available (`--device` to override).

## Honest limitations & next steps

- **Scaffold is given.** We generate ops onto a sampled graph skeleton ("fix the
  mesh, flow-match the contents"). Co-generating the *topology* (DiGress-style,
  with edge/arity constraints) is the harder open problem.
- **Synthetic distribution.** The data-flow-conditioned rule is hand-designed so
  the metric is exact. The real test is mining DAGs from actual code
  (tree-sitter for ASTs, Joern for CPGs) on a narrow language/DSL.
- **Toy scale.** ~0.4M (cpu) / ~30M (a100) params. The research suggests this is
  the right zone — CodeFusion matched far larger AR models at 75M on constrained
  domains — but it is not a general code model.

## Positioning vs. prior art

- *vs.* token-sequence code diffusion (DiffuCoder, Dream-Coder, Seed Diffusion):
  those denoise flat tokens and discard the graph; here the graph is the prior.
- *vs.* AST diffusion ([Diffusion on Syntax Trees](https://arxiv.org/abs/2405.20519))
  and AST-masked token diffusion ([TreeDiff](https://arxiv.org/abs/2508.01473)):
  those stop at the syntax tree; here we use data/control-flow edges.
- *vs.* graph diffusion / discrete flow on graphs ([DiGress](https://arxiv.org/abs/2209.14734),
  [DeFoG](https://arxiv.org/abs/2410.04263)): mature for molecules, never applied
  to program graphs — this is that application.
