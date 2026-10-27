# algebraic-supplementary

Hybrid and ARDS experiments for the 125M algebraic transformer.

This repository contains:
- **14 hybrid architectures** — combinatorial permutations of standard and algebraic transformer components
- **ARDS pipelines** — standard and algebraic transformers trained with AdamW + Algebraic Rational Decay Schedule
- **Pure XLA Execution** — fully portable, hardware-accelerated JAX XLA implementations for all architectures across TPU, GPU, and CPU

No pure-standard or pure-algebraic models live here. For the base implementations see [`algebraic-transformer`](https://github.com/tasmaikeni13/algebraic-transformer).

---

## Scale

All experiments target **125M parameters** on **2.5B tokens of FineWeb-Edu**, single seed (42).

| Setting | Value |
|---|---|
| Layers | 12 |
| d_model | 768 |
| Heads | 12 |
| d_ff | 2048 |
| Context | 2048 tokens |
| Vocab | 32,768 (BPE) |
| Batch | 512 seqs |
| Micro | 4 seqs/device |
| Steps | ~2,385 |
| Seed | 42 |

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
from hybrid.registry import list_presets, get_config

# List all 14 hybrids
names = list_presets()

# Build a config
cfg = get_config("hybrid_rope_alu_asinh_alg_ce", size="125m")
```

### CLI

```bash
python -m hybrid list
python -m hybrid info hybrid_cayley_swiglu_asinh_alg_ce
```

---

## ARDS & Comparison Pipelines (`ards/`)

Dedicated configurations and training pipelines for the **4 canonical comparison runs**:
`{Standard, Algebraic} x {Cosine, ARDS}`

| Preset | Architecture | Optimizer | Schedule | Description |
|---|---|---|---|---|
| `standard_cosine` | Standard | AdamW | Cosine | Standard (RoPE, SwiGLU, Softmax, CE) + Cosine decay |
| `algebraic_cosine` | Algebraic | AdamW | Cosine | Pure Algebraic (Cayley, ALU-GLU, Asinh, Alg-CE) + Cosine decay |
| `standard_ards` | Standard | AdamW | ARDS | Standard (RoPE, SwiGLU, Softmax, CE) + Rational decay |
| `algebraic_ards` | Algebraic | AdamW | ARDS | Pure Algebraic (Cayley, ALU-GLU, Asinh, Alg-CE) + Rational decay |

### Rational Decay Schedule (ARDS)
```
lr(t) = peak / sqrt(1 + (t - t_warm)^2 / (0.8 T)^2)    for t > t_warm
lr(t) = peak * t / t_warm                                 for t <= t_warm
```

Defaults for 125M / 2.5B tokens: `peak=0.003`, `T=2385`, `t_warm=48`, `min_lr≈0.0019`.

### Python API

```python
from ards import (
    standard_ards,
    algebraic_ards,
    standard_cosine,
    algebraic_cosine,
    get_training_config,
)

# 4 Comparison Runs
cfg_std_ards = standard_ards()         # Standard + ARDS
cfg_alg_ards = algebraic_ards()        # Algebraic + ARDS
cfg_std_cos  = standard_cosine()       # Standard + Cosine
cfg_alg_cos  = algebraic_cosine()      # Algebraic + Cosine

# Training recipes
recipe_std_cos = get_training_config("standard_cosine")
recipe_alg_ards = get_training_config("algebraic_ards")
```

### Training (single-chip smoke)

```bash
# ARDS run
bash scripts/tpu_python.sh 0 -m ards.train \
  --arm standard_ards --smoke --out /tmp/smoke_std_ards

# Cosine comparison run
bash scripts/tpu_python.sh 0 -m ards.train \
  --arm standard_cosine --smoke --out /tmp/smoke_std_cosine
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

Verify all 16 architectures (14 hybrids + 2 ARDS variants) with 1-step synthetic AdamW:

```bash
# Unified 16-architecture smoke runner
python scripts/smoke_all.py --all --tiny --steps 1 --device cpu

# Single preset smoke test
python scripts/smoke_all.py --preset hybrid_rope_alu_asinh_alg_ce --steps 1 --batch 4
```

Run the test suites:

```bash
# Hybrid parameter counts, forward/backward passes
pytest tests/test_hybrids.py -v

# ARDS and comparison schedule precision, single-step AdamW
pytest tests/test_ards.py -v

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
ards/            4 comparison runs and ARDS pipelines
src/algebraic/
  algebraic.py   Algebraic primitives (rho, alu, cayley, asinh-attention, alg-likelihood)
  standard.py    Standard primitives (silu, rope, softmax-attention, cross-entropy)
  model.py       Unified configurable transformer (LMConfig)
  config.py      125m and proxy presets
  train.py       AdamW training loop with ARDS/cosine/WSD schedules
configs/         comparison_125m.json
scripts/         smoke_all.py, smoke_comparison.py, smoke_benchmarks.py, tpu_python.sh
tests/           test_hybrids.py, test_ards.py, test_smoke_all.py, test_components.py
```
