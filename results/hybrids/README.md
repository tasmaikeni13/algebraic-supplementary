# Factorial Ablation: 14 Hybrid Architectures on 2.5B Tokens (125M Scale)

Comprehensive empirical comparison of all 16 architecture combinations (2 pure baselines + 14 hybrids) 
trained under identical conditions on FineWeb-Edu using AdamW with Cosine decay (seed 42).

## Results Table

| ID | Preset Name | Position | FFN | Attention | Head | Params | Eval NLL | Eval PPL | $\Delta$ vs Standard | $\Delta$ vs Algebraic | Throughput | Train Time |
|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BASE** | `standard_cosine` | RoPE | SwiGLU | Softmax | CE | 110.12M | **2.9136** | 18.42 | 0.0000 | +0.0379 | ~760k tok/s | ~55 min |
| **H01** | `hybrid_cayley_swiglu_softmax_ce` | cayley_geo | swiglu | softmax | ce | 110.12M | **2.9177** | 18.50 | +0.0041 | +0.0420 | 757,405 tok/s | 55.2 min |
| **H02** | `hybrid_rope_alu_softmax_ce` | rope | alu_glu | softmax | ce | 110.12M | **2.9164** | 18.47 | +0.0028 | +0.0407 | 742,373 tok/s | 56.3 min |
| **H03** | `hybrid_rope_swiglu_asinh_ce` | rope | swiglu | asinh | ce | 110.12M | **2.8974** | 18.13 | -0.0162 | +0.0217 | 533,830 tok/s | 78.3 min |
| **H04** | `hybrid_rope_swiglu_softmax_alg_ce` | rope | swiglu | softmax | alg_ce | 110.12M | **2.9013** | 18.20 | -0.0123 | +0.0256 | 756,918 tok/s | 55.2 min |
| **H05** | `hybrid_cayley_alu_softmax_ce` | cayley_geo | alu_glu | softmax | ce | 110.12M | **2.9160** | 18.47 | +0.0024 | +0.0403 | 742,355 tok/s | 56.3 min |
| **H06** | `hybrid_cayley_swiglu_asinh_ce` | cayley_geo | swiglu | asinh | ce | 110.12M | **2.8968** | 18.12 | -0.0168 | +0.0211 | 533,828 tok/s | 78.3 min |
| **H07** | `hybrid_cayley_swiglu_softmax_alg_ce` | cayley_geo | swiglu | softmax | alg_ce | 110.12M | **2.8966** | 18.11 | -0.0170 | +0.0209 | 756,786 tok/s | 55.2 min |
| **H08** | `hybrid_rope_alu_asinh_ce` | rope | alu_glu | asinh | ce | 110.12M | **2.8968** | 18.12 | -0.0168 | +0.0211 | 527,445 tok/s | 79.2 min |
| **H09** | `hybrid_rope_alu_softmax_alg_ce` | rope | alu_glu | softmax | alg_ce | 110.12M | **2.8959** | 18.10 | -0.0177 | +0.0202 | 741,000 tok/s | 56.4 min |
| **H10** | `hybrid_rope_swiglu_asinh_alg_ce` | rope | swiglu | asinh | alg_ce | 110.12M | **2.8752** | 17.73 | -0.0384 | -0.0005 | 533,756 tok/s | 78.3 min |
| **H11** | `hybrid_rope_alu_asinh_alg_ce` | rope | alu_glu | asinh | alg_ce | 110.12M | **2.8750** | 17.73 | -0.0386 | -0.0007 | 527,642 tok/s | 79.2 min |
| **H12** | `hybrid_cayley_swiglu_asinh_alg_ce` | cayley_geo | swiglu | asinh | alg_ce | 110.12M | **2.8746** | 17.72 | -0.0390 | -0.0011 | 533,762 tok/s | 78.3 min |
| **H13** | `hybrid_cayley_alu_softmax_alg_ce` | cayley_geo | alu_glu | softmax | alg_ce | 110.12M | **2.8978** | 18.13 | -0.0158 | +0.0221 | 741,104 tok/s | 56.4 min |
| **H14** | `hybrid_cayley_alu_asinh_ce` | cayley_geo | alu_glu | asinh | ce | 110.12M | **2.8968** | 18.12 | -0.0168 | +0.0211 | 527,475 tok/s | 79.2 min |
| **BASE** | `algebraic_cosine` | Cayley | ALU-GLU | Asinh | Alg-CE | 110.12M | **2.8757** | 17.74 | -0.0379 | 0.0000 | ~530k tok/s | ~78 min |

## Component Key
- **Position**: `RoPE` (Standard transcendental) vs `Cayley` (Pure algebraic $SO(n)$ rational rotation)
- **FFN**: `SwiGLU` (Standard transcendental $\sigma(x)$) vs `ALU-GLU` (Pure algebraic rational curve $x / \sqrt{1 + x^2}$)
- **Attention**: `Softmax` (Standard transcendental $\exp(x)$) vs `Asinh` (Pure algebraic algebraic power attention $\rho(s/8)^{16}$)
- **Head**: `CE` (Standard cross-entropy) vs `Alg-CE` (Pure algebraic gradient of rational likelihood $\rho(z/2)^8$)

*Report auto-generated at 2026-11-14 02:02:40 IST*
