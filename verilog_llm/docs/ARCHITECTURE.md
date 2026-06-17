# Architecture

Three views: the **model** (what it computes), the **hardware** (how the RTL is
organized), and the **build/verify flow** (how we know it's correct).

## 1. Model dataflow

A decoder-only transformer (GPT family). One token id in → probability of the
next token out. Greedy decoding feeds the argmax back in (autoregression).

```mermaid
flowchart TB
    tok["token ids (seq, len T)"] --> emb
    pos["positional index 0..T-1"] --> emb
    subgraph emb["embeddings"]
        te["token_emb lookup (V×D)"]
        pe["pos_emb (T×D)"]
        te --> add0((+))
        pe --> add0
    end
    add0 --> x0["residual stream x  (T×D)"]

    x0 --> blk
    subgraph blk["transformer block  ×N layers"]
        direction TB
        ln1["LayerNorm"] --> attn
        subgraph attn["causal self-attention (1 head)"]
            qkv["Q,K,V = xWq,xWk,xWv"] --> sc["scores = QKᵀ / √D"]
            sc --> mask["causal mask + softmax"]
            mask --> av["A·V"] --> proj["·Wo"]
        end
        attn --> r1((+))
        ln2["LayerNorm"] --> mlp
        subgraph mlp["feed-forward"]
            f1["·W1 (D→M)"] --> relu["ReLU"] --> f2["·W2 (M→D)"]
        end
        mlp --> r2((+))
    end
    blk --> lnf["final LayerNorm"]
    lnf --> head["output head ·Wh (D→V)"]
    head --> logits["logits (V)"] --> argmax["argmax → next token"]
    argmax -. append, repeat .-> tok
```

Default dims: `D=32, N=2, heads=1, T=24, M=64, V=28` (character-level).

## 2. Hardware organization (`rtl/gpt.v`)

Weights live in on-chip memories loaded at init via `$readmemh`. A small FSM
drives one **full forward pass per generated token**; the datapath is built
entirely from the fixed-point primitives in `rtl/fixed_pkg.vh`.

```mermaid
flowchart LR
    subgraph mem["weight memories (BRAM)"]
        tokm[tok/pos]
        wm["Wq/Wk/Wv/Wo · W1/W2 · Wh"]
        bm["biases · LayerNorm γ/β"]
    end

    subgraph dp["fixed-point datapath (Q16.16)"]
        mac["MAC / matrix-vector<br/>(64-bit accumulators)"]
        ln["LayerNorm<br/>mean·var·isqrt·recip"]
        sm["softmax<br/>max · exp(2^x) · divide"]
        relu[ReLU]
    end

    subgraph ctrl["control"]
        fsm["FSM: IDLE → GEN → DONE<br/>(one token / clock in sim)"]
        amax["argmax"]
        seq["seq buffer (T tokens)"]
    end

    mem --> mac --> ln --> sm --> relu --> dp
    dp --> logits[(logits)]
    logits --> amax --> seq
    seq --> mac
    fsm --- dp
    fsm --- seq
```

### Fixed-point primitives (`fixed_pkg.vh`)

Everything is signed **Q16.16** (16 integer + 16 fractional bits) in 32-bit
words, with 64-bit accumulators and saturation. Each function is a bit-for-bit
port of `ref/fixed.py`:

| function | op | hardware technique |
|---|---|---|
| `fmul` | Q×Q | 64-bit product, arithmetic shift, saturate |
| `fadd/fsub` | Q±Q | saturating |
| `fdiv` | Q÷Q | signed divide (truncating) |
| `isqrt`/`fsqrt` | √ | classic bit-by-bit integer square root |
| `frsqrt` | 1/√ | `fsqrt` then reciprocal (for LayerNorm) |
| `fexp_neg` | eˣ, x≤0 | `2^(x·log₂e)`, split int/frac, cubic poly for `2^frac` |

## 3. Build & verify flow

The hardware isn't *approximately* a transformer — it is **bit-exact** with a
software golden model. That equivalence is the whole point.

```mermaid
flowchart LR
    train["ref/model.py<br/>float train + backprop<br/>(grad-checked ~1e-7)"] --> quant["quantize → Q16.16"]
    quant --> hex["weights/*.hex<br/>config.vh · load_weights.vh"]
    quant --> golden["ref/fixed_infer.py<br/>bit-accurate fixed model"]
    golden --> gtok["weights/golden.hex<br/>(golden token stream)"]
    hex --> sim["iverilog: rtl/gpt.v + tb"]
    gtok --> tb["tb_gpt.v compares<br/>hardware vs golden"]
    sim --> tb
    tb --> verdict{{"bit-exact?<br/>PASS / FAIL"}}
```

`ref/fixed.py` and `rtl/fixed_pkg.vh` implement the *same integer arithmetic*,
so the RTL reproduces the golden token stream exactly (verified for all 24
generated tokens).
