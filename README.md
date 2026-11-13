# algebraic-supplementary

Hybrid Architectures and Pure Algebraic Cubic Hermite Schedule Experiments for the 125M Algebraic Transformer.

This repository contains:
- **14 hybrid architectures** — combinatorial permutations of standard and algebraic transformer components
- **Pure-ALU Cubic Hermite schedule** — standard and algebraic transformers trained with AdamW + pure algebraic Smoothstep polynomial decay schedule replacing transcendental Cosine
- **125M Benchmark Results** — comprehensive 2.5B token comparisons on FineWeb-Edu demonstrating pure-ALU polynomial schedule beating transcendental Cosine decay
- **Pure XLA Execution** — fully portable, hardware-accelerated JAX XLA implementations for all architectures across TPU, GPU, and CPU

---

## 125M Main Benchmark Results (2.5B Tokens, Seed 42)

Evaluated on held-out FineWeb-Edu validation window (8M tokens, 2048 context):

| Run / Architecture | LR Schedule | Schedule Math | Eval NLL | Eval PPL | vs. Cosine Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Algebraic + Cubic (Ours)** | **Pure Algebraic Cubic** | **$S(u) = (1 - u)^2(1 + 2u)$ (Pure ALU)** | **`2.8614`** | **`17.49`** | **`-0.0147` (Outperforms Cosine)** |
| **Algebraic + Cosine** | Cosine | $\frac{1}{2}(1 + \cos(\pi u))$ (Transcendental) | `2.8761` | `17.74` | *baseline* |
| | | | | | |
| **Standard + Cubic (Ours)** | **Pure Algebraic Cubic** | **$S(u) = (1 - u)^2(1 + 2u)$ (Pure ALU)** | **`2.9142`** | **`18.43`** | **`-0.0033` (Outperforms Cosine)** |
| **Standard + Cosine** | Cosine | $\frac{1}{2}(1 + \cos(\pi u))$ (Transcendental) | `2.9175` | `18.49` | *baseline* |

> **Key Finding**: The pure algebraic cubic Hermite schedule requires only addition and multiplication (zero transcendental trigonometric calls) while matching or beating transcendental Cosine decay on both Standard and Algebraic transformer architectures.

---

## Scale & Setup

All experiments target **125M parameters** on **2.5B tokens of FineWeb-Edu**, single seed (42).

| Setting | Value |
|---|---|
| Layers | 12 |
| d_model | 768 |
| Heads | 12 |
| d_ff | 2048 |
| Context | 2048 tokens |
| Vocab | 32,768 (BPE) |
| Batch | 512 seqs (1,048,576 tokens/step) |
| Micro | 4 seqs/device |
| Steps | 2,385 |
| Seed | 42 |

---

## Pure Algebraic Cubic Schedule (`src/algebraic/train.py`)

The pure ALU Cubic Hermite schedule (Smoothstep) replaces standard Cosine decay with a zero-derivative polynomial curve:

```
S(u) = (1 - u)^2 * (1 + 2u) = 1 - 3u^2 + 2u^3
lr(u) = floor + (peak - floor) * S(u)
```

Where:
* $u = \frac{t - t_{\text{warm}}}{T - t_{\text{warm}}} \in [0, 1]$
* $\text{floor} = \text{peak} \times 0.1$
* $S'(0) = 0$ (smooth departure from peak learning rate)
* $S'(1) = 0$ (smooth, zero-derivative terminal convergence at minimum floor)
* $S(0.5) = 0.5$ (exact inflection symmetry matching cosine)
* **Zero transcendental calls**: strictly uses only basic ALU arithmetic (multiplication and addition).

---

## Hybrid Architectures (`hybrid/`)

14 non-trivial permutations across 4 axes:

| Axis | Standard | Algebraic |
|---|---|---|
| Positional | RoPE (`rope`) | Geometric Cayley (`cayley_geo`) |
| FFN | SwiGLU (`swiglu`) | ALU-GLU β=0.5 (`alu_glu`) |
| Attention | Softmax (`softmax`) | Asinh-power + QK-norm + sink (`asinh`) |
| Loss Head | Cross-Entropy (`ce`) | Algebraic likelihood (`alg_ce`) |

Softmax-attention hybrids: **110,120,448 parameters**.
Asinh-attention hybrids: **110,122,128 parameters** (Δ = +1,680 for QK-norm gains and sink logits).

### All 14 hybrids

```
hybrid_rope_swiglu_softmax_alg_ce     hybrid_rope_swiglu_asinh_ce
hybrid_rope_swiglu_asinh_alg_ce       hybrid_rope_alu_softmax_ce
hybrid_rope_alu_softmax_alg_ce        hybrid_rope_alu_asinh_ce
hybrid_rope_alu_asinh_alg_ce          hybrid_cayley_swiglu_softmax_ce
hybrid_cayley_swiglu_softmax_alg_ce   hybrid_cayley_swiglu_asinh_ce
hybrid_cayley_swiglu_asinh_alg_ce     hybrid_cayley_alu_softmax_ce
hybrid_cayley_alu_softmax_alg_ce      hybrid_cayley_alu_asinh_ce
```

### Python API

```python
from algebraic.config import build, PRESETS
from hybrid.registry import list_presets, get_config

# Build Cubic presets
cfg_alg_cubic = build("algebraic_cubic", size="125m")
cfg_std_cubic = build("standard_cubic", size="125m")

# Build Hybrid configs
cfg_hybrid = get_config("hybrid_rope_alu_asinh_alg_ce", size="125m")
```

---

## Pure XLA Execution (`src/algebraic/`)

All primitives are implemented in pure XLA JAX without external kernel dependencies:
- **Attention**: Pure XLA implementations of Softmax attention and Asinh-power attention ($\rho(s/c)^n$ with learned sink logits).
- **Normalization**: Pure XLA learned RMS-norm and QK RMS-norm.
- **Feed-Forward**: Pure XLA SwiGLU and ALU-GLU activations.
- **Objectives**: Pure XLA Cross-Entropy and Algebraic likelihood with exact analytical gradients.

---

## Smoke Tests

Verify all 16 architectures (14 hybrids + 2 cubic presets) with 1-step synthetic AdamW:

```bash
# Unified 16-architecture smoke runner
python scripts/smoke_all.py --all --tiny --steps 1 --device cpu

# Single preset smoke test
python scripts/smoke_all.py --preset algebraic_cubic --steps 1 --batch 4
```

Run test suites:

```bash
# Pure-ALU Cubic schedule mathematical identity, derivative and preset tests
pytest tests/test_cubic.py -v

# Hybrid parameter counts, forward/backward passes
pytest tests/test_hybrids.py -v

# Unified 16-architecture smoke suite
pytest tests/test_smoke_all.py -v

# Algebraic component unit tests
pytest tests/test_components.py -v
```

---

## Requirements

```
jax>=0.4.30
flax
optax
numpy
```

Install:
```bash
pip install -r requirements.txt
```

---

## Repository Layout

```
hybrid/          14 hybrid model configs and registry
src/algebraic/
  algebraic.py   Algebraic primitives (rho, alu, cayley, asinh-attention, alg-likelihood)
  standard.py    Standard primitives (silu, rope, softmax-attention, cross-entropy)
  model.py       Unified configurable transformer (LMConfig)
  config.py      125m and proxy presets (standard, algebraic, cubic, cosine)
  train.py       AdamW training loop with cubic/cosine/wsd schedules
results/
  hparam_sweep/  600M hyperparameter sweep results and selections
  main_125m/     2.5B comparison results, summaries, and per-run metrics
scripts/         run_cubic_pipeline.py, run_slice.py, smoke_all.py, setup_workers.sh
tests/           test_cubic.py, test_hybrids.py, test_smoke_all.py, test_components.py
```
