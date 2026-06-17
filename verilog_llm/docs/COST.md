# Cost & "is it worth it?"

All figures are **rough 2025–2026 industry ballparks** for planning only — real
quotes vary widely by foundry, node, area, volume, region and negotiation.
Currency is USD. "NRE" = non-recurring engineering (one-time design/mask cost).

The framing is a generic engineering organization deciding whether to push this
design — or a real LLM accelerator built from it — onto hardware.

---

## TL;DR verdict

| Path | One-time cost | Worth it? |
|---|---|---|
| **Run in simulation** (today) | ~$0 | ✅ Already done. Best ROI in the repo. |
| **FPGA demo** | $150 – $5k hardware + weeks of labor | ✅ **Yes** — for learning, portfolio, interviews, and as a scaffold for a real accelerator. |
| **ASIC, hobby tapeout** (TinyTapeout) | ~$300 – $1k + your time | 🟡 Only for the trophy / to learn the flow. No product value. |
| **ASIC, MPW prototype** | ~$20k – $150k + a small team | ❌ Not for *this* model. Only if prototyping real silicon IP. |
| **ASIC, full production** | **$1M – $5M** (mature node) up to **$50M – $1B+** (leading edge) | ❌ Absolutely not for a toy. Only viable for a competitive, real-model accelerator. |

**Bottom line:** the *toy model* is worth taking to **FPGA** and no further. The
*design pattern* is worth a lot more — as a teaching artifact and as the seed of
a real transformer accelerator — but turning that into competitive silicon is a
$100M-scale endeavor against well-funded incumbents.

---

## 1. FPGA cost breakdown

Dominated by labor, not parts.

| Item | Low | High |
|---|---|---|
| Dev board | $130 (Arty A7) | $5k (Alveo, only if scaling up) |
| Tools (EDA) | $0 (Yosys/nextpnr, or Vivado free tier) | ~$3–5k/yr (full Vivado seat) |
| Engineering: micro-arch rewrite + bring-up | ~3 weeks | ~3 months |
| — at a ~$180k/yr loaded engineer (~$15k/mo) | ~$10k | ~$45k |
| **Total (realistic working demo)** | **~$10k** | **~$50k** |

A solo hobbyist with a $150 board and open tools does it for **~the price of the
board** plus their own time.

---

## 2. ASIC cost breakdown

Two independent buckets: **mask set** (fixed, per node) and **design NRE**
(team + tools + IP). MPW shuttles let you share the mask to slash the first.

### Mask set (full reticle, order-of-magnitude)

| Node | Mask set (NRE) | Notes |
|---|---|---|
| 180 nm | $50k – $100k | open-tool friendly, cheap |
| 130 nm (Sky130) | $100k – $200k | open PDK; MPW makes it far cheaper |
| 65 nm | $250k – $400k | |
| 28 nm | $1M – $2M | sweet spot for cost/perf on mature nodes |
| 16/14 nm | $3M – $5M | |
| 7 nm | $10M – $18M | |
| 5 nm | $15M – $25M | |
| 3 nm | $20M – $40M (mask alone) | full SoC programs run $500M – $1.5B |

### Design NRE (to do it *properly*, small chip, mature node)

| Item | Ballpark |
|---|---|
| Commercial EDA flow (Synopsys/Cadence/Siemens) | $0.5M – $2M / yr per bundle |
| IP licensing (SRAM compilers, PLL, I/O, PHYs) | $0.1M – $1M+ |
| Design + verification team (3–10 eng, 12–18 mo) | $1M – $5M |
| Packaging, test program, bring-up, qual | $0.2M – $1M |

### Cheap on-ramps (share the mask)

| Route | Cost | What you get |
|---|---|---|
| TinyTapeout | ~$300 – $1k | a shared tile on Sky130/GF180; tiny area |
| Open MPW (Sky130/IHP, academic) | ~$5k – $25k | a few mm², tens of chips |
| Commercial MPW (28 nm via Europractice/MOSIS) | ~$50k – $150k | real node, small block, tens of chips |

> Note: **Efabless** (the well-known Sky130 shuttle broker) wound down in 2025 —
> verify a current broker (Europractice, MOSIS, IHP) before committing.

---

## 3. So, is it worth it?

**As a product: no.** This model has a 28-character vocabulary and was trained to
memorize a single sentence. It has *zero* commercial utility. Spending mask money
on it would be lighting cash on fire.

**As engineering value: yes, cheaply, on FPGA.** For the cost of a dev board (or
nothing, in simulation) you get:
- a genuinely impressive, *verified* end-to-end demo,
- hands-on coverage of the entire flow — training → quantization → RTL →
  synthesis → silicon — which is rare and interview-gold,
- a clean, bit-exact scaffold you can grow.

**The real question is the *real* accelerator.** If "is it worth it" means *should
a firm build custom LLM inference silicon*, the honest answer:

- The economics only close at **scale** — you need a real (billions-of-params)
  model and high inference volume to beat renting GPUs.
- A competitive inference ASIC at an advanced node is a **$50M – $300M+** program
  over 2–3 years, against entrenched, well-funded players (Nvidia, plus Groq,
  Etched, Tenstorrent, Cerebras, AWS Trainium/Inferentia, Google TPU).
- For almost every firm, the rational path is **rent GPUs/TPUs in the cloud**, or
  buy merchant accelerators — *not* tape out your own.
- Custom silicon makes sense only with (a) a durable, high-volume internal
  workload, (b) a specific power/latency/cost wall that commodity parts can't
  clear, and (c) the capital and team to sustain multi-generation development.

**Recommendation:** keep this as an open-source FPGA demo and learning vehicle.
Take it to a board for ~$150. Tape out via TinyTapeout only if you want the
souvenir. Do **not** fund a custom ASIC for this model; revisit the ASIC question
only if you have a real model and a real, high-volume workload that pencils out.
