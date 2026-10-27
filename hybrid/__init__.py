"""Hybrid Transformer Architecture Package.

Provides all 14 non-trivial combinatorial permutations of standard and algebraic
transformer components across 4 orthogonal component axes:
- Positional Encoding (pos): rope vs cayley_geo
- Feed-Forward Network (ffn): swiglu vs alu_glu (beta=0.5)
- Attention Mechanism (attn): softmax vs asinh (n=16, c=8.0, qk_norm, learned sink)
- Training Head (head): ce vs alg_ce (n=8, c=2.0)

Pure standard and pure algebraic configurations are strictly excluded.
"""

from hybrid.config import (
    hybrid_cayley_alu_asinh_ce,
    hybrid_cayley_alu_softmax_alg_ce,
    hybrid_cayley_alu_softmax_ce,
    hybrid_cayley_swiglu_asinh_alg_ce,
    hybrid_cayley_swiglu_asinh_ce,
    hybrid_cayley_swiglu_softmax_alg_ce,
    hybrid_cayley_swiglu_softmax_ce,
    hybrid_rope_alu_asinh_alg_ce,
    hybrid_rope_alu_asinh_ce,
    hybrid_rope_alu_softmax_alg_ce,
    hybrid_rope_alu_softmax_ce,
    hybrid_rope_swiglu_asinh_alg_ce,
    hybrid_rope_swiglu_asinh_ce,
    hybrid_rope_swiglu_softmax_alg_ce,
    make_hybrid_config,
)
from hybrid.model import (
    HybridModel,
    build_hybrid_model,
    count_hybrid_params,
    get_model_parameter_count,
    init_hybrid_params,
    verify_hybrid_param_count,
)
from hybrid.registry import (
    HYBRID_ALIASES,
    HYBRID_METADATA,
    HYBRID_PRESETS,
    build_hybrid,
    describe_hybrids,
    get_hybrid,
    get_hybrid_preset,
    list_hybrids,
)

__all__ = [
    # Registry & lookup
    "HYBRID_PRESETS",
    "HYBRID_ALIASES",
    "HYBRID_METADATA",
    "get_hybrid",
    "get_hybrid_preset",
    "list_hybrids",
    "build_hybrid",
    "describe_hybrids",
    "make_hybrid_config",
    # Model container & functions
    "HybridModel",
    "build_hybrid_model",
    "init_hybrid_params",
    "count_hybrid_params",
    "verify_hybrid_param_count",
    "get_model_parameter_count",
    # 14 Canonical Presets (H01 -> H14)
    "hybrid_cayley_swiglu_softmax_ce",
    "hybrid_rope_alu_softmax_ce",
    "hybrid_rope_swiglu_asinh_ce",
    "hybrid_rope_swiglu_softmax_alg_ce",
    "hybrid_cayley_alu_softmax_ce",
    "hybrid_cayley_swiglu_asinh_ce",
    "hybrid_cayley_swiglu_softmax_alg_ce",
    "hybrid_rope_alu_asinh_ce",
    "hybrid_rope_alu_softmax_alg_ce",
    "hybrid_rope_swiglu_asinh_alg_ce",
    "hybrid_rope_alu_asinh_alg_ce",
    "hybrid_cayley_swiglu_asinh_alg_ce",
    "hybrid_cayley_alu_softmax_alg_ce",
    "hybrid_cayley_alu_asinh_ce",
]
