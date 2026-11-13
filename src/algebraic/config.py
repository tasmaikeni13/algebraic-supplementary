"""Matched standard and pure algebraic Transformer presets."""

from dataclasses import replace
from typing import Dict

from algebraic.model import LMConfig

SIZES: Dict[str, Dict[str, int]] = {
    "proxy": dict(d_model=384, n_layers=8, n_heads=6, d_ff=1024),
    "125m": dict(d_model=768, n_layers=12, n_heads=12, d_ff=2048),
}


def standard(size: str = "proxy", **overrides) -> LMConfig:
    """Learned RMS gains, SwiGLU, RoPE, softmax attention, cross-entropy."""
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="swiglu", pos="rope",
                   attn="softmax", sink="none", head="ce")
    return replace(cfg, **overrides)


def algebraic(size: str = "proxy", **overrides) -> LMConfig:
    """Pure algebraic model: arithmetic, comparisons, and reciprocal roots.

    Learned RMS gains; ALU-GLU with beta=1/2; geometric Cayley rotations;
    QK normalization; rho(s/8)^16 attention with a learned per-head sink;
    rho(z/2)^8 likelihood with the algebraic gradient of exact log loss.
    The primitive audit covers the model objective and its gradient.
    """
    cfg = LMConfig(**SIZES[size], norm="rms", ffn="alu_glu", alu_beta=0.5,
                   pos="cayley_geo", qk_norm=True, attn="asinh", attn_n=16, attn_c=8.0,
                   sink="learned", sink_init=0.0, head="alg_ce", head_n=8, head_c=2.0)
    return replace(cfg, **overrides)


def standard_cubic(size: str = "proxy", **overrides) -> LMConfig:
    """125M Standard Transformer with pure Algebraic Cubic schedule."""
    overrides.pop("schedule", None)
    return standard(size, **overrides)


def algebraic_cubic(size: str = "proxy", **overrides) -> LMConfig:
    """125M Pure Algebraic Transformer with pure Algebraic Cubic schedule."""
    overrides.pop("schedule", None)
    return algebraic(size, **overrides)


def standard_cosine(size: str = "proxy", **overrides) -> LMConfig:
    """125M Standard Transformer with Cosine schedule."""
    overrides.pop("schedule", None)
    return standard(size, **overrides)


def algebraic_cosine(size: str = "proxy", **overrides) -> LMConfig:
    """125M Pure Algebraic Transformer with Cosine schedule."""
    overrides.pop("schedule", None)
    return algebraic(size, **overrides)


PRESETS = {
    "standard": standard,
    "algebraic": algebraic,
    "standard_cubic": standard_cubic,
    "algebraic_cubic": algebraic_cubic,
    "standard_cosine": standard_cosine,
    "algebraic_cosine": algebraic_cosine,
}


def build(preset: str, size: str = "125m", **overrides) -> LMConfig:
    """Build an LMConfig for a standard, algebraic, cubic, or hybrid preset."""
    if preset in PRESETS:
        return PRESETS[preset](size, **overrides)
    try:
        from hybrid.registry import HYBRID_PRESETS, HYBRID_ALIASES
        if preset in HYBRID_PRESETS:
            return HYBRID_PRESETS[preset](size, **overrides)
        if preset in HYBRID_ALIASES:
            canonical = HYBRID_ALIASES[preset]
            return HYBRID_PRESETS[canonical](size, **overrides)
    except ImportError:
        pass
    raise KeyError(f"unknown preset {preset!r}")

