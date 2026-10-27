"""Registry and lookup utilities for the 14 hybrid presets."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from algebraic.model import LMConfig
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
)


@dataclass(frozen=True)
class HybridMetadata:
    index: str
    name: str
    pos: str
    ffn: str
    attn: str
    head: str
    params_125m: int
    params_proxy: int


# The 14 canonical preset functions mapped by their canonical name
_CANONICAL_PRESETS: Dict[str, Callable[..., LMConfig]] = {
    "hybrid_cayley_swiglu_softmax_ce": hybrid_cayley_swiglu_softmax_ce,      # H01
    "hybrid_rope_alu_softmax_ce": hybrid_rope_alu_softmax_ce,                # H02
    "hybrid_rope_swiglu_asinh_ce": hybrid_rope_swiglu_asinh_ce,              # H03
    "hybrid_rope_swiglu_softmax_alg_ce": hybrid_rope_swiglu_softmax_alg_ce,  # H04
    "hybrid_cayley_alu_softmax_ce": hybrid_cayley_alu_softmax_ce,            # H05
    "hybrid_cayley_swiglu_asinh_ce": hybrid_cayley_swiglu_asinh_ce,          # H06
    "hybrid_cayley_swiglu_softmax_alg_ce": hybrid_cayley_swiglu_softmax_alg_ce,  # H07
    "hybrid_rope_alu_asinh_ce": hybrid_rope_alu_asinh_ce,                    # H08
    "hybrid_rope_alu_softmax_alg_ce": hybrid_rope_alu_softmax_alg_ce,        # H09
    "hybrid_rope_swiglu_asinh_alg_ce": hybrid_rope_swiglu_asinh_alg_ce,      # H10
    "hybrid_rope_alu_asinh_alg_ce": hybrid_rope_alu_asinh_alg_ce,            # H11
    "hybrid_cayley_swiglu_asinh_alg_ce": hybrid_cayley_swiglu_asinh_alg_ce,  # H12
    "hybrid_cayley_alu_softmax_alg_ce": hybrid_cayley_alu_softmax_alg_ce,    # H13
    "hybrid_cayley_alu_asinh_ce": hybrid_cayley_alu_asinh_ce,                # H14
}

# Ordered list of canonical preset names (H01 -> H14)
ORDERED_HYBRID_NAMES: List[str] = [
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

HYBRID_METADATA: Dict[str, HybridMetadata] = {
    "hybrid_cayley_swiglu_softmax_ce": HybridMetadata("H01", "hybrid_cayley_swiglu_softmax_ce", "cayley_geo", "swiglu", "softmax", "ce", 110120448, 26745600),
    "hybrid_rope_alu_softmax_ce": HybridMetadata("H02", "hybrid_rope_alu_softmax_ce", "rope", "alu_glu", "softmax", "ce", 110120448, 26745600),
    "hybrid_rope_swiglu_asinh_ce": HybridMetadata("H03", "hybrid_rope_swiglu_asinh_ce", "rope", "swiglu", "asinh", "ce", 110122128, 26746672),
    "hybrid_rope_swiglu_softmax_alg_ce": HybridMetadata("H04", "hybrid_rope_swiglu_softmax_alg_ce", "rope", "swiglu", "softmax", "alg_ce", 110120448, 26745600),
    "hybrid_cayley_alu_softmax_ce": HybridMetadata("H05", "hybrid_cayley_alu_softmax_ce", "cayley_geo", "alu_glu", "softmax", "ce", 110120448, 26745600),
    "hybrid_cayley_swiglu_asinh_ce": HybridMetadata("H06", "hybrid_cayley_swiglu_asinh_ce", "cayley_geo", "swiglu", "asinh", "ce", 110122128, 26746672),
    "hybrid_cayley_swiglu_softmax_alg_ce": HybridMetadata("H07", "hybrid_cayley_swiglu_softmax_alg_ce", "cayley_geo", "swiglu", "softmax", "alg_ce", 110120448, 26745600),
    "hybrid_rope_alu_asinh_ce": HybridMetadata("H08", "hybrid_rope_alu_asinh_ce", "rope", "alu_glu", "asinh", "ce", 110122128, 26746672),
    "hybrid_rope_alu_softmax_alg_ce": HybridMetadata("H09", "hybrid_rope_alu_softmax_alg_ce", "rope", "alu_glu", "softmax", "alg_ce", 110120448, 26745600),
    "hybrid_rope_swiglu_asinh_alg_ce": HybridMetadata("H10", "hybrid_rope_swiglu_asinh_alg_ce", "rope", "swiglu", "asinh", "alg_ce", 110122128, 26746672),
    "hybrid_rope_alu_asinh_alg_ce": HybridMetadata("H11", "hybrid_rope_alu_asinh_alg_ce", "rope", "alu_glu", "asinh", "alg_ce", 110122128, 26746672),
    "hybrid_cayley_swiglu_asinh_alg_ce": HybridMetadata("H12", "hybrid_cayley_swiglu_asinh_alg_ce", "cayley_geo", "swiglu", "asinh", "alg_ce", 110122128, 26746672),
    "hybrid_cayley_alu_softmax_alg_ce": HybridMetadata("H13", "hybrid_cayley_alu_softmax_alg_ce", "cayley_geo", "alu_glu", "softmax", "alg_ce", 110120448, 26745600),
    "hybrid_cayley_alu_asinh_ce": HybridMetadata("H14", "hybrid_cayley_alu_asinh_ce", "cayley_geo", "alu_glu", "asinh", "ce", 110122128, 26746672),
}

# Construct alias mappings
HYBRID_ALIASES: Dict[str, str] = {}
for i, name in enumerate(ORDERED_HYBRID_NAMES, start=1):
    idx_upper = f"H{i:02d}"
    idx_lower = f"h{i:02d}"
    HYBRID_ALIASES[idx_upper] = name
    HYBRID_ALIASES[idx_lower] = name

    # Add variations like alu_glu -> alu, cayley_geo -> cayley, alg_ce -> algce
    alt1 = name.replace("_alu_", "_aluglu_")
    alt2 = name.replace("_alu_", "_alu_glu_")
    alt3 = name.replace("_alg_ce", "_algce")
    alt4 = name.replace("_cayley_", "_cayley_geo_")
    for alt in (alt1, alt2, alt3, alt4):
        if alt != name:
            HYBRID_ALIASES[alt] = name


class HybridPresetsRegistry(dict):
    """Registry dictionary mapping canonical names to presets, with alias fallback."""

    def __init__(self, canonical: Dict[str, Callable[..., LMConfig]], aliases: Dict[str, str]):
        super().__init__(canonical)
        self._aliases = aliases

    def __getitem__(self, key: str) -> Callable[..., LMConfig]:
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self._aliases:
            return super().__getitem__(self._aliases[key])
        if key in ("standard", "algebraic", "hybrid_rope_swiglu_softmax_ce", "hybrid_cayley_alu_asinh_alg_ce"):
            raise KeyError(f"Pure preset {key!r} is excluded from hybrid/ (use algebraic.config.{key})")
        raise KeyError(f"unknown hybrid preset {key!r}")

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str) and key in self._aliases:
            return True
        return False

    def get(self, key: str, default: Optional[Callable[..., LMConfig]] = None) -> Optional[Callable[..., LMConfig]]:
        try:
            return self[key]
        except KeyError:
            return default


# The public registry containing exactly 14 canonical entries in its keys()
HYBRID_PRESETS = HybridPresetsRegistry(_CANONICAL_PRESETS, HYBRID_ALIASES)


def list_hybrids() -> List[str]:
    """Return the 14 canonical hybrid preset names in index order (H01 - H14)."""
    return list(ORDERED_HYBRID_NAMES)


def get_hybrid(name: str) -> Callable[..., LMConfig]:
    """Resolve a hybrid preset by canonical name or alias (e.g. 'H01')."""
    return HYBRID_PRESETS[name]


get_hybrid_preset = get_hybrid


def build_hybrid(name: str, size: str = "125m", **overrides: Any) -> LMConfig:
    """Build an LMConfig for the given hybrid preset name or alias."""
    preset_fn = get_hybrid(name)
    return preset_fn(size=size, **overrides)


def describe_hybrids() -> List[Dict[str, Any]]:
    """Return structured metadata describing all 14 hybrid presets."""
    desc = []
    for name in ORDERED_HYBRID_NAMES:
        meta = HYBRID_METADATA[name]
        desc.append({
            "index": meta.index,
            "name": meta.name,
            "pos": meta.pos,
            "ffn": meta.ffn,
            "attn": meta.attn,
            "head": meta.head,
            "params_125m": meta.params_125m,
            "params_proxy": meta.params_proxy,
        })
    return desc
