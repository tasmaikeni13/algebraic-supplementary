"""14 combinatorial hybrid configurations spanning standard and algebraic axes.

The 4 orthogonal component axes are:
1. Positional Encoding (`pos`): trigonometric `rope` vs geometric `cayley_geo`
2. Feed-Forward Network (`ffn`): standard `swiglu` vs algebraic `alu_glu` (beta=0.5)
3. Attention Mechanism (`attn`): standard `softmax` (sink="none", qk_norm=False)
   vs algebraic `asinh` (n=16, c=8.0, sink="learned", sink_init=0.0, qk_norm=True)
4. Training Head (`head`): standard `ce` vs algebraic `alg_ce` (n=8, c=2.0)

Pure standard (rope + swiglu + softmax + ce) and pure algebraic
(cayley_geo + alu_glu + asinh + alg_ce) are strictly excluded from hybrid/.
"""

from dataclasses import replace
from typing import Any, Dict

from algebraic.config import SIZES
from algebraic.model import LMConfig


def make_hybrid_config(
    pos: str,
    ffn: str,
    attn: str,
    head: str,
    size: str = "125m",
    **overrides: Any,
) -> LMConfig:
    """Build an LMConfig for a hybrid model across the 4 component axes.

    Normalizes alias names and enforces exclusion of pure standard and pure algebraic models.
    """
    # Normalize pos
    pos_norm = pos.lower()
    if pos_norm in ("cayley", "cayley_geo"):
        pos_norm = "cayley_geo"
    elif pos_norm == "rope":
        pos_norm = "rope"
    else:
        raise ValueError(f"unknown pos axis {pos!r}; expected 'rope' or 'cayley_geo'")

    # Normalize ffn
    ffn_norm = ffn.lower()
    if ffn_norm in ("alu", "aluglu", "alu_glu"):
        ffn_norm = "alu_glu"
    elif ffn_norm == "swiglu":
        ffn_norm = "swiglu"
    else:
        raise ValueError(f"unknown ffn axis {ffn!r}; expected 'swiglu' or 'alu_glu'")

    # Normalize attn
    attn_norm = attn.lower()
    if attn_norm not in ("softmax", "asinh"):
        raise ValueError(f"unknown attn axis {attn!r}; expected 'softmax' or 'asinh'")

    # Normalize head
    head_norm = head.lower()
    if head_norm in ("alg_ce", "algce"):
        head_norm = "alg_ce"
    elif head_norm == "ce":
        head_norm = "ce"
    else:
        raise ValueError(f"unknown head axis {head!r}; expected 'ce' or 'alg_ce'")

    # Strictly prohibit pure standard and pure algebraic presets
    if pos_norm == "rope" and ffn_norm == "swiglu" and attn_norm == "softmax" and head_norm == "ce":
        raise ValueError("Pure standard preset is excluded from hybrid/ (use algebraic.config.standard)")
    if pos_norm == "cayley_geo" and ffn_norm == "alu_glu" and attn_norm == "asinh" and head_norm == "alg_ce":
        raise ValueError("Pure algebraic preset is excluded from hybrid/ (use algebraic.config.algebraic)")

    if size not in SIZES:
        raise KeyError(f"unknown model size {size!r}; available: {list(SIZES.keys())}")

    cfg_kwargs: Dict[str, Any] = dict(
        **SIZES[size],
        norm="rms",
        pos=pos_norm,
        ffn=ffn_norm,
        attn=attn_norm,
        head=head_norm,
    )

    if ffn_norm == "alu_glu":
        cfg_kwargs["alu_beta"] = 0.5

    if attn_norm == "asinh":
        cfg_kwargs.update(
            qk_norm=True,
            attn_n=16,
            attn_c=8.0,
            sink="learned",
            sink_init=0.0,
        )
    else:
        cfg_kwargs.update(
            qk_norm=False,
            sink="none",
        )

    if head_norm == "alg_ce":
        cfg_kwargs.update(
            head_n=8,
            head_c=2.0,
        )

    cfg = LMConfig(**cfg_kwargs)
    return replace(cfg, **overrides)


# -----------------------------------------------------------------------------
# 14 Canonical Hybrid Preset Definitions (H01 - H14)
# -----------------------------------------------------------------------------


def hybrid_cayley_swiglu_softmax_ce(size: str = "125m", **overrides) -> LMConfig:
    """H01: CayleyGeo + SwiGLU + Softmax + CE."""
    return make_hybrid_config("cayley_geo", "swiglu", "softmax", "ce", size=size, **overrides)


def hybrid_rope_alu_softmax_ce(size: str = "125m", **overrides) -> LMConfig:
    """H02: RoPE + ALU-GLU(0.5) + Softmax + CE."""
    return make_hybrid_config("rope", "alu_glu", "softmax", "ce", size=size, **overrides)


def hybrid_rope_swiglu_asinh_ce(size: str = "125m", **overrides) -> LMConfig:
    """H03: RoPE + SwiGLU + Asinh(16, 8.0, QK-norm, sink) + CE."""
    return make_hybrid_config("rope", "swiglu", "asinh", "ce", size=size, **overrides)


def hybrid_rope_swiglu_softmax_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H04: RoPE + SwiGLU + Softmax + Alg-CE(8, 2.0)."""
    return make_hybrid_config("rope", "swiglu", "softmax", "alg_ce", size=size, **overrides)


def hybrid_cayley_alu_softmax_ce(size: str = "125m", **overrides) -> LMConfig:
    """H05: CayleyGeo + ALU-GLU(0.5) + Softmax + CE."""
    return make_hybrid_config("cayley_geo", "alu_glu", "softmax", "ce", size=size, **overrides)


def hybrid_cayley_swiglu_asinh_ce(size: str = "125m", **overrides) -> LMConfig:
    """H06: CayleyGeo + SwiGLU + Asinh(16, 8.0, QK-norm, sink) + CE."""
    return make_hybrid_config("cayley_geo", "swiglu", "asinh", "ce", size=size, **overrides)


def hybrid_cayley_swiglu_softmax_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H07: CayleyGeo + SwiGLU + Softmax + Alg-CE(8, 2.0)."""
    return make_hybrid_config("cayley_geo", "swiglu", "softmax", "alg_ce", size=size, **overrides)


def hybrid_rope_alu_asinh_ce(size: str = "125m", **overrides) -> LMConfig:
    """H08: RoPE + ALU-GLU(0.5) + Asinh(16, 8.0, QK-norm, sink) + CE."""
    return make_hybrid_config("rope", "alu_glu", "asinh", "ce", size=size, **overrides)


def hybrid_rope_alu_softmax_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H09: RoPE + ALU-GLU(0.5) + Softmax + Alg-CE(8, 2.0)."""
    return make_hybrid_config("rope", "alu_glu", "softmax", "alg_ce", size=size, **overrides)


def hybrid_rope_swiglu_asinh_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H10: RoPE + SwiGLU + Asinh(16, 8.0, QK-norm, sink) + Alg-CE(8, 2.0)."""
    return make_hybrid_config("rope", "swiglu", "asinh", "alg_ce", size=size, **overrides)


def hybrid_rope_alu_asinh_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H11: RoPE + ALU-GLU(0.5) + Asinh(16, 8.0, QK-norm, sink) + Alg-CE(8, 2.0)."""
    return make_hybrid_config("rope", "alu_glu", "asinh", "alg_ce", size=size, **overrides)


def hybrid_cayley_swiglu_asinh_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H12: CayleyGeo + SwiGLU + Asinh(16, 8.0, QK-norm, sink) + Alg-CE(8, 2.0)."""
    return make_hybrid_config("cayley_geo", "swiglu", "asinh", "alg_ce", size=size, **overrides)


def hybrid_cayley_alu_softmax_alg_ce(size: str = "125m", **overrides) -> LMConfig:
    """H13: CayleyGeo + ALU-GLU(0.5) + Softmax + Alg-CE(8, 2.0)."""
    return make_hybrid_config("cayley_geo", "alu_glu", "softmax", "alg_ce", size=size, **overrides)


def hybrid_cayley_alu_asinh_ce(size: str = "125m", **overrides) -> LMConfig:
    """H14: CayleyGeo + ALU-GLU(0.5) + Asinh(16, 8.0, QK-norm, sink) + CE."""
    return make_hybrid_config("cayley_geo", "alu_glu", "asinh", "ce", size=size, **overrides)
