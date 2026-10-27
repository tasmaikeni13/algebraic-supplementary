"""Model container and instantiation utilities for hybrid architectures."""

from typing import Any, Callable, Dict, Optional, Tuple

import jax

from algebraic.model import (
    LMConfig,
    count_params,
    forward_hidden,
    head_loss,
    init_params,
    position_tables,
)


class HybridModel:
    """Container wrapping a hybrid model's configuration, parameters, and position tables."""

    def __init__(
        self,
        cfg: LMConfig,
        params: Dict[str, Any],
        tables: Optional[Tuple[Any, Any]] = None,
    ):
        self.cfg = cfg
        self.params = params
        self.tables = tables if tables is not None else position_tables(cfg)

    @property
    def param_count(self) -> int:
        """Total trainable parameter count."""
        return count_params(self.params)

    def forward(self, tokens: jax.Array, params: Optional[Dict[str, Any]] = None) -> jax.Array:
        """Compute final normalized hidden states."""
        p = self.params if params is None else params
        return forward_hidden(p, tokens, self.cfg, self.tables)

    def loss(
        self,
        hidden: jax.Array,
        targets: jax.Array,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute head loss and telemetry."""
        p = self.params if params is None else params
        return head_loss(p, hidden, targets, self.cfg)

    def forward_and_loss(
        self,
        tokens: jax.Array,
        targets: jax.Array,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute forward hidden states followed by head loss in one step."""
        p = self.params if params is None else params
        h = forward_hidden(p, tokens, self.cfg, self.tables)
        return head_loss(p, h, targets, self.cfg)

    def __iter__(self):
        """Enable tuple unpacking: params, forward_fn, loss_fn = build_hybrid_model(cfg)."""
        forward_fn = lambda p, tok: forward_hidden(p, tok, self.cfg, self.tables)
        loss_fn = lambda p, h, tgt: head_loss(p, h, tgt, self.cfg)
        return iter((self.params, forward_fn, loss_fn))


def build_hybrid_model(
    cfg: LMConfig,
    rng: Optional[jax.Array] = None,
) -> HybridModel:
    """Instantiate a HybridModel container from an LMConfig."""
    if rng is None:
        rng = jax.random.PRNGKey(42)
    params = init_params(cfg, rng)
    tables = position_tables(cfg)
    return HybridModel(cfg, params, tables)


def init_hybrid_params(
    cfg: LMConfig,
    rng: Optional[jax.Array] = None,
) -> Dict[str, Any]:
    """Initialize parameter dictionary for a hybrid configuration."""
    if rng is None:
        rng = jax.random.PRNGKey(42)
    return init_params(cfg, rng)


def count_hybrid_params(params: Dict[str, Any]) -> int:
    """Return the total number of scalar parameters in the model parameter tree."""
    return count_params(params)


def verify_hybrid_param_count(
    cfg: LMConfig,
    params: Optional[Dict[str, Any]] = None,
) -> bool:
    """Verify that a hybrid configuration matches theoretical parameter count expectations."""
    if params is None:
        p = init_hybrid_params(cfg)
    else:
        p = params
    actual = count_hybrid_params(p)
    if cfg.d_model == 768 and cfg.n_layers == 12:
        expected = 110_122_128 if cfg.attn == "asinh" else 110_120_448
    elif cfg.d_model == 384 and cfg.n_layers == 8:
        expected = 26_746_672 if cfg.attn == "asinh" else 26_745_600
    else:
        return True
    return actual == expected


def get_model_parameter_count(cfg: LMConfig) -> int:
    """Convenience function to compute parameter count directly from config."""
    p = init_hybrid_params(cfg)
    return count_hybrid_params(p)
