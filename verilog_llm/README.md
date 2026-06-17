# A tiny LLM in Verilog

A complete, working **GPT-style language model implemented in synthesizable
Verilog**. It loads trained weights, runs a transformer forward pass in
fixed-point arithmetic, and autoregressively generates text — all in RTL,
verified **bit-exact** against a Python golden model.

```
  generated: "the lazy dog. the quick "
  PASS: hardware output is bit-exact with the golden model (24 tokens).
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — model dataflow, hardware block
  diagram, and the bit-exact build/verify flow (with diagrams).
- [docs/FPGA_TO_ASIC.md](docs/FPGA_TO_ASIC.md) — how to get this onto an FPGA and
  then an ASIC, and what the RTL needs before it's silicon-ready.
- [docs/COST.md](docs/COST.md) — FPGA/ASIC cost breakdown and an honest
  "is it worth it?" verdict.
- [PUBLISH.md](PUBLISH.md) — turn this directory into its own standalone repo.

## What it is

A decoder-only transformer (the same architecture family as GPT), shrunk so it
fits in a simulator and overfits a small corpus:

| | |
|---|---|
| Architecture | token + positional embeddings → `N`× (pre-norm self-attention + ReLU MLP) → final LayerNorm → output head |
| Dimensions | `d_model=32`, `layers=2`, `heads=1`, `context=24`, `mlp=64`, `vocab=28` |
| Tokenizer | character-level |
| Arithmetic | **Q16.16 fixed-point** (32-bit), 64-bit accumulators, saturating |
| Decoding | greedy (argmax), autoregressive, one token per clock |

Every non-trivial hardware operation is built from scratch in fixed point:

- **matrix–vector products** with wide accumulators,
- **LayerNorm** — mean/variance, integer `sqrt` (bit-by-bit), reciprocal,
- **softmax** — max-subtraction, `exp` via `2^x` with a cubic polynomial, divide,
- **causal attention**, **ReLU**, **residual** connections.

## Why you can trust it

The hardware isn't "approximately" a transformer — it's **bit-exact**. The flow:

1. `ref/model.py` trains the model in float (numpy, hand-written backprop,
   gradient-checked to ~1e-7).
2. Weights are quantized to Q16.16 and exported as `$readmemh` hex files.
3. `ref/fixed_infer.py` is a **bit-accurate** fixed-point reference — its integer
   ops match the Verilog functions in `rtl/fixed_pkg.vh` one-for-one.
4. `rtl/gpt.v` runs in Icarus Verilog and the testbench asserts the generated
   token stream equals the golden stream exactly.

## Run it

```bash
./run.sh train     # train, quantize, export weights, compile, simulate
./run.sh           # recompile + simulate using existing weights
```

Requirements: `iverilog` (Icarus Verilog) and `python3` + `numpy`.

Generate from an arbitrary prompt using the same fixed-point math the hardware
uses:

```bash
cd ref && python3 gen.py "brown fox "
# -> 'brown fox jumps over the'
```

The model was trained only on *"the quick brown fox jumps over the lazy dog."*
and learned to continue it from any position — evidence it's a real (if tiny)
language model, not a stored string.

## Layout

```
ref/
  fixed.py            fixed-point primitives (golden semantics)
  model.py            float transformer + manual backprop (training)
  fixed_infer.py      bit-accurate fixed-point forward pass (golden inference)
  train_and_export.py train → quantize → export weights/config/testvectors
  gen.py              generate text from exported weights
rtl/
  fixed_pkg.vh        fmul/fdiv/fsqrt/frsqrt/fexp — Verilog ports of fixed.py
  gpt.v               the model: memories, forward pass, generation FSM
  config.vh           auto-generated dimensions & constants
  load_weights.vh     auto-generated $readmemh statements
  id2asc.vh           auto-generated token→ASCII decoder (for printing)
tb/
  tb_gpt.v            runs the model, prints text, checks bit-exactness
weights/              exported .hex weights + prompt/golden vectors
run.sh                build + simulate
```

## Notes / scope

This favors clarity and verifiability over hardware efficiency: the forward pass
is described procedurally (loops over the matmuls), so a full token is computed
per clock in simulation. It elaborates as RTL (memories → BRAM, loops unroll),
but a real accelerator would pipeline the MACs and stream weights rather than
unrolling. The point here is a *correct, end-to-end, readable* LLM in Verilog.
