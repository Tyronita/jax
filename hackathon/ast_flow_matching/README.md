# Flow matching over program graphs — a minimal prototype

> Poolside hackathon idea. *"When geometry is knowable, flow-matched
> diffusion looks more like flow-field pathfinding."*

## The idea

Standard diffusion / flow-matching models learn the geometry of the data
manifold **implicitly**, because for images or text that geometry is unknown
and continuous. Code is different: it has explicit, *computable* structure —
ASTs, control-flow graphs, data-flow graphs, unified as the **Code Property
Graph** (CPG). The bet is that when the geometry is knowable, you should not
make the model relearn it. You hand it over as a fixed scaffold, and
generation turns into **constrained pathfinding over a known mesh** rather
than blind denoising.

Most code-diffusion work today (CodeFusion, DiffuCoder, …) still denoises a
*token sequence* and throws the graph away. This prototype is the smallest
honest test of the opposite: inject known program geometry into a discrete
flow and measure what it buys you.

## What this prototype is

A toy but complete instance of the thesis, in pure JAX:

| Thesis concept              | Here                                                            |
| --------------------------- | -------------------------------------------------------------- |
| Program space               | Perfect binary arithmetic-expression trees (heap layout)       |
| Known geometry              | Positional type-map: each slot is an *operator* or *value* slot |
| Generative process          | Masked (absorbing) **discrete flow matching**                  |
| "Pathfinding over the mesh" | Sampler restricted to grammar-legal tokens per slot            |
| Sequence-only baseline      | Identical model, no constraint — must learn validity from data |

Because the tree topology is fixed, every array slot has a known type, so the
geometry collapses to a `[L, V]` mask of which tokens are legal where. That
mask enters the flow as an additive logit bias (`0` legal, `-1e9` illegal)
during **both** training and sampling. The geometry-aware and sequence-only
runs are otherwise byte-for-byte identical — same architecture, data, initial
weights, optimizer — so the bias isolates exactly what the prior contributes.

```
grammar.py   problem definition: vocab, positional type-map, data dist, validity, pretty-printer
model.py     tiny non-causal Transformer x_1-predictor (pure JAX pytree params)
dfm.py       masked discrete flow matching: corrupt / loss / unmask step
train.py     trains both models, compares, prints a table + sample programs
```

## Running it

This lives inside the JAX **source** tree, which has no built `jaxlib`. Install
a released CPU JAX and run from *this* directory so `import jax` resolves to the
installed package rather than the checkout:

```bash
pip install "jax[cpu]"
cd hackathon/ast_flow_matching
python train.py                       # ~2 min on CPU
python train.py --height 4 --steps 800   # taller trees: the gap widens
```

## What you see

Two metrics over training budget:

* **validity** — fraction of samples that are syntactically valid trees
* **marg-TV**  — mean per-slot total-variation distance between the sampled
  token marginals and the true data marginals (lower is better; penalises both
  illegal tokens and wrong frequencies)

```
  step |  loss_geo loss_base | valid_geo valid_base |  TV_geo TV_base
---------------------------------------------------------------------
     0 |    1.8041    2.6887 |    100.0%       0.0% |  0.2734  0.5672
   250 |    1.1790    1.1698 |    100.0%      99.6% |  0.0380  0.0447
  1500 |    1.1666    1.1567 |    100.0%      99.8% |  0.0351  0.0379
```

* **Geometry-aware is valid by construction** — 100% at *every* budget,
  including step 0 with random weights. It never emits `(+ 1 *)`.
* **The baseline must learn validity.** With a tiny budget it is 0% valid
  (operators land in leaf slots and vice-versa); it climbs as it trains.
* **The advantage compounds with program size.** Validity needs *every* slot
  to be type-correct, so at `--height 4` (31 slots) the baseline plateaus a
  point or two below 100% even after training, while the geometry-aware run is
  unaffected.

The honest read: for this easy toy task the baseline mostly catches up given
enough budget. The win is concentrated where it matters for real code — the
low-data / large-structure regime — and it is *free*: the geometry-aware model
spends zero capacity and zero gradient learning what the grammar already states.

## From here to real program graphs

This deliberately uses the simplest geometry (a static positional type-map).
The path toward the actual thesis:

1. **Variable topology** — real ASTs are not perfect trees. Generate the shape
   too (e.g. a `[STOP]`/expand token), or flow-match over a separately sampled
   skeleton.
2. **Grammar-as-geometry** — replace the per-slot mask with a full
   context-free grammar, masking transitions by the legal productions at each
   step. The constraint becomes the parser.
3. **Real graph edges** — attend over actual CFG/DFG edges instead of a dense
   sequence, i.e. graph attention on the CPG. This is the "fix the mesh,
   flow-match the contents" framing from the note.
4. **Semantics** — push past syntax to data-flow / type constraints, so the
   known geometry encodes more than shape.

The claim to validate at scale: the richer the *known* structure you bake in,
the more of the model's budget is freed for the genuinely hard, learned part of
the distribution.
