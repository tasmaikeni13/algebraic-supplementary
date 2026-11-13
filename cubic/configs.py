"""Model presets for pure algebraic cubic configurations."""

from typing import Any, Dict
from algebraic.config import build
from algebraic.model import LMConfig


def standard_cubic(size: str = "125m", **overrides: Any) -> LMConfig:
    """125M Standard Transformer preset with pure Algebraic Cubic schedule."""
    return build("standard_cubic", size=size, **overrides)


def algebraic_cubic(size: str = "125m", **overrides: Any) -> LMConfig:
    """125M Pure Algebraic Transformer preset with pure Algebraic Cubic schedule."""
    return build("algebraic_cubic", size=size, **overrides)


CUBIC_PRESETS: Dict[str, Any] = {
    "standard_cubic": standard_cubic,
    "algebraic_cubic": algebraic_cubic,
}
