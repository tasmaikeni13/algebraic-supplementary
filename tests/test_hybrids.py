"""Comprehensive test suite for the 14 hybrid transformer permutations.

Verifies:
1. Exclusion of pure standard and pure algebraic presets from hybrid registry.
2. algebraic.config.PRESETS purity and delegation.
3. Exact parameter count verification across 125M and proxy scales (110,120,448 vs 110,122,128).
4. Forward pass, loss calculation, and finite gradients on CPU.
5. Forward pass, loss calculation, and finite gradients on TPU.
6. HybridModel container and API interfaces.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import algebraic.config as cfg_module
from algebraic.model import (
    count_params,
    eval_logprobs,
    forward_hidden,
    head_loss,
    init_params,
    position_tables,
)
from hybrid import (
    HYBRID_ALIASES,
    HYBRID_METADATA,
    HYBRID_PRESETS,
    HybridModel,
    build_hybrid,
    build_hybrid_model,
    describe_hybrids,
    get_hybrid,
    get_hybrid_preset,
    list_hybrids,
    make_hybrid_config,
    verify_hybrid_param_count,
)


@pytest.fixture(autouse=True)
def float32_context():
    """Guard TPU tests from global x64 pollution."""
    enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    with enable_x64(False):
        yield


# -----------------------------------------------------------------------------
# Test Vector 1: Pure Preset Exclusion
# -----------------------------------------------------------------------------


def test_pure_presets_not_in_hybrid_registry():
    """Ensure registry contains exactly 14 non-trivial presets and excludes pure variants."""
    assert len(HYBRID_PRESETS) == 14
    assert len(list_hybrids()) == 14

    # Pure presets must NOT be in HYBRID_PRESETS
    assert "standard" not in HYBRID_PRESETS
    assert "algebraic" not in HYBRID_PRESETS
    assert "hybrid_rope_swiglu_softmax_ce" not in HYBRID_PRESETS
    assert "hybrid_cayley_alu_asinh_alg_ce" not in HYBRID_PRESETS

    # get_hybrid must raise KeyError for pure models
    with pytest.raises(KeyError):
        get_hybrid("standard")
    with pytest.raises(KeyError):
        get_hybrid("algebraic")
    with pytest.raises(KeyError):
        get_hybrid("hybrid_rope_swiglu_softmax_ce")
    with pytest.raises(KeyError):
        get_hybrid("hybrid_cayley_alu_asinh_alg_ce")

    # Direct config constructor must raise ValueError for pure combinations
    with pytest.raises(ValueError, match="Pure standard preset is excluded"):
        make_hybrid_config("rope", "swiglu", "softmax", "ce")
    with pytest.raises(ValueError, match="Pure algebraic preset is excluded"):
        make_hybrid_config("cayley_geo", "alu_glu", "asinh", "alg_ce")


# -----------------------------------------------------------------------------
# Test Vector 2: Delegation & algebraic.config.PRESETS Purity
# -----------------------------------------------------------------------------


def test_algebraic_config_delegation_and_presets_purity():
    """Ensure PRESETS remains strictly standard/algebraic while build() delegates to hybrids."""
    # PRESETS must remain strictly {"standard", "algebraic"}
    assert set(cfg_module.PRESETS.keys()) == {"standard", "algebraic"}

    # Delegation succeeds for all 14 canonical hybrid presets
    for preset_name in HYBRID_PRESETS:
        cfg = cfg_module.build(preset_name, "125m")
        assert cfg.vocab_size == 32768
        assert cfg.d_model == 768
        assert cfg.n_layers == 12

    # Delegation succeeds for aliases H01 - H14
    for i in range(1, 15):
        alias = f"H{i:02d}"
        cfg = cfg_module.build(alias, "125m")
        assert cfg.vocab_size == 32768
        assert cfg.d_model == 768

    # Unknown preset raises KeyError
    with pytest.raises(KeyError):
        cfg_module.build("nonexistent_hybrid_preset_xyz")


# -----------------------------------------------------------------------------
# Test Vector 3: Parameter Count Exactness (All 14 Hybrids)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(HYBRID_PRESETS.keys()))
def test_hybrid_exact_parameter_counts(preset_name):
    """Verify exact theoretical parameter counts on 125M and proxy scales."""
    key = jax.random.PRNGKey(0)

    # 125M scale check
    cfg_125m = build_hybrid(preset_name, size="125m")
    p_125m = count_params(init_params(cfg_125m, key))
    if cfg_125m.attn == "asinh":
        assert p_125m == 110_122_128, f"{preset_name} 125M params {p_125m} != 110,122,128"
    else:
        assert p_125m == 110_120_448, f"{preset_name} 125M params {p_125m} != 110,120,448"
    assert verify_hybrid_param_count(cfg_125m)

    # Proxy scale check
    cfg_proxy = build_hybrid(preset_name, size="proxy")
    p_proxy = count_params(init_params(cfg_proxy, key))
    if cfg_proxy.attn == "asinh":
        assert p_proxy == 26_746_672, f"{preset_name} proxy params {p_proxy} != 26,746,672"
    else:
        assert p_proxy == 26_745_600, f"{preset_name} proxy params {p_proxy} != 26,745,600"
    assert verify_hybrid_param_count(cfg_proxy)


# -----------------------------------------------------------------------------
# Test Vector 4: Parametrized CPU Forward, Loss, and Finite Gradients
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", sorted(HYBRID_PRESETS.keys()))
def test_hybrid_cpu_forward_loss_and_finite_gradients(preset_name):
    """Run forward pass and compute gradients on CPU to confirm finite values and no NaNs."""
    cfg = build_hybrid(
        preset_name,
        size="proxy",
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=96,
        vocab_size=97,
        seq_len=16,
        dtype="float32",
        remat=False,
    )
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    tok = jax.random.randint(key, (2, 16), 0, 97)
    tables = position_tables(cfg, 16)

    def loss_fn(p):
        h = forward_hidden(p, tok, cfg, tables)
        return head_loss(p, h, tok, cfg)[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(loss), f"Loss for {preset_name} is non-finite: {loss}"
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in leaves), f"Gradients for {preset_name} contain non-finite values"
    assert not any(np.isnan(g).any() for g in leaves), f"Gradients for {preset_name} contain NaNs"

    lp = eval_logprobs(params, tok, tok, cfg, tables)
    assert lp.shape == tok.shape
    assert np.all(lp <= 0), f"Log-probabilities for {preset_name} exceed 0"


# -----------------------------------------------------------------------------
# Test Vector 5: Parametrized TPU Forward, Loss, and Gradients with Pure XLA Execution
# -----------------------------------------------------------------------------


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("preset_name", sorted(HYBRID_PRESETS.keys()))
def test_hybrid_tpu_forward_loss_and_gradients(preset_name):
    """Run forward, loss, and backward pass on TPU with pure XLA execution."""
    cfg = build_hybrid(
        preset_name,
        size="125m",
        seq_len=512,
        attn_impl="dense",
        head_impl="dense",
        flash_block=512,
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
        remat=False,
    )
    key = jax.random.PRNGKey(101)
    params = init_params(cfg, key)
    tok = jax.random.randint(key, (2, 512), 0, cfg.vocab_size)
    tables = position_tables(cfg, 512)

    def step(p, x):
        def loss_fn(parameters):
            h = forward_hidden(parameters, x, cfg, tables)
            return head_loss(parameters, h, x, cfg)[0]

        return jax.value_and_grad(loss_fn)(p)

    loss, grads = jax.jit(step)(params, tok)
    assert np.isfinite(loss), f"TPU loss for {preset_name} is non-finite: {loss}"
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in leaves), f"TPU gradients for {preset_name} contain non-finite values"
    assert not any(np.isnan(g).any() for g in leaves), f"TPU gradients for {preset_name} contain NaNs"


# -----------------------------------------------------------------------------
# Test Vector 6: HybridModel Container & Helpers
# -----------------------------------------------------------------------------


def test_hybrid_model_container():
    """Verify HybridModel container methods and tuple unpacking interface."""
    cfg = build_hybrid("H01", size="proxy", d_model=64, n_layers=2, n_heads=2, d_ff=96, vocab_size=97, seq_len=16)
    key = jax.random.PRNGKey(42)

    # Test object methods
    model = build_hybrid_model(cfg, key)
    assert isinstance(model, HybridModel)
    assert model.param_count > 0

    tok = jax.random.randint(key, (1, 16), 0, 97)
    h = model.forward(tok)
    assert h.shape == (1, 16, 64)

    loss, telem = model.loss(h, tok)
    assert np.isfinite(loss)

    fwd_loss, _ = model.forward_and_loss(tok, tok)
    assert np.isfinite(fwd_loss)

    # Test tuple unpacking
    params, forward_fn, loss_fn = build_hybrid_model(cfg, key)
    h2 = forward_fn(params, tok)
    loss2, _ = loss_fn(params, h2, tok)
    assert np.isclose(float(loss), float(loss2), atol=1e-5)


def test_describe_hybrids_and_aliases():
    """Verify metadata and alias resolutions."""
    hybrids = describe_hybrids()
    assert len(hybrids) == 14
    for h in hybrids:
        assert h["index"].startswith("H")
        assert h["params_125m"] in (110_120_448, 110_122_128)
        assert h["params_proxy"] in (26_745_600, 26_746_672)

    # Test alias resolution
    assert get_hybrid("H01") == get_hybrid("hybrid_cayley_swiglu_softmax_ce")
    assert get_hybrid("H14") == get_hybrid("hybrid_cayley_alu_asinh_ce")
    assert get_hybrid("hybrid_rope_aluglu_softmax_ce") == get_hybrid("hybrid_rope_alu_softmax_ce")
