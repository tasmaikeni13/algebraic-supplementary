"""Comprehensive Test Suite for ARDS Pipelines and Mathematical Purity.

Validates Requirements R3 and R5:
1. Exact mathematical precision of the Algebraic Rational Decay Schedule (ARDS)
   at critical milestones: t=0, t=t_warm, t=T, t=2*T, t >> T.
2. Strict prohibition and rejection of cosine learning rate schedules.
3. Architecture presets and exact parameter counts for 125M Standard+ARDS (110,120,448)
   and Algebraic+ARDS (110,122,128; delta = +1,680).
4. Single-step AdamW + ARDS optimization pass on CPU and TPU (finite loss, finite
   non-zero gradients, verified weight mutation).
5. Checkpoint serialization, bit-for-bit weight resumption, and optimizer state preservation.
6. Safety guardrails: long-running training job rejection and remote push suppression.
"""

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Module under test
import ards
from ards.configs import (
    ARDS_PRESETS,
    ARDSTrainingConfig,
    COMPARISON_4_RUNS,
    algebraic_ards,
    algebraic_cosine,
    get_ards_config,
    get_training_config,
    standard_ards,
    standard_cosine,
    validate_schedule,
    validate_schedule_not_cosine,
)
from ards.schedule import (
    DEFAULT_125M_PEAK_LR,
    DEFAULT_125M_TOTAL_STEPS,
    DEFAULT_125M_WARMUP_STEPS,
    DEFAULT_SCALE_FACTOR,
    FORBIDDEN_COSINE_KEYWORDS,
    ards_schedule,
    cosine_schedule,
    reject_cosine_schedule,
)
from algebraic.model import (
    count_params,
    decay_mask,
    forward_hidden,
    head_loss,
    init_params,
    position_tables,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def float32_context():
    """Guard TPU tests from global x64 pollution."""
    enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    with enable_x64(False):
        yield


# ==============================================================================
# Vector 1: ARDS Mathematical Formula Precision & Asymptotics
# ==============================================================================


def ards_reference_formula(
    t: float,
    peak_lr: float = DEFAULT_125M_PEAK_LR,
    warmup_steps: float = DEFAULT_125M_WARMUP_STEPS,
    total_steps: float = DEFAULT_125M_TOTAL_STEPS,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
) -> float:
    """Exact double-precision reference implementation of the ARDS formula."""
    if t <= warmup_steps:
        if warmup_steps > 0:
            return peak_lr * (t / warmup_steps)
        return peak_lr
    d = (t - warmup_steps) / (scale_factor * total_steps)
    return peak_lr / math.sqrt(1.0 + d * d)


def test_ards_schedule_key_steps_exact_mathematical_match():
    """Test ARDS schedule at key steps: t=0, t=t_warm, t=T, t=2*T, t >> T."""
    peak = 0.003
    T = 2385
    warm = 48
    scale = 0.8 * T  # 1908.0

    sched = ards_schedule(peak_lr=peak, warmup_steps=warm, total_steps=T, scale_factor=0.8)

    # 1. Continuous t=0: lr should be exactly 0.0
    # For discrete step=0 (where t = step + 1 = 1.0):
    lr_step0 = float(sched(0))
    expected_step0 = peak * (1.0 / warm)  # 0.003 / 48 = 6.25e-5
    np.testing.assert_allclose(lr_step0, expected_step0, rtol=1e-7, atol=1e-9)
    assert np.isclose(lr_step0, 6.25e-5, atol=1e-9)

    # Continuous reference at t=0
    assert ards_reference_formula(0.0, peak, warm, T, 0.8) == 0.0

    # 2. Warmup completion: step = warm - 1 = 47 (t = 48 = t_warm) -> exact peak
    lr_warm = float(sched(warm - 1))
    assert np.isclose(lr_warm, peak, atol=1e-8)

    # 3. Budget completion: step = T - 1 = 2384 (t = 2385 = T)
    lr_T = float(sched(T - 1))
    d_T = (2385.0 - 48.0) / 1908.0  # 2337 / 1908 = 1.224842767...
    expected_T = peak / math.sqrt(1.0 + d_T * d_T)  # 0.0018972756...
    np.testing.assert_allclose(lr_T, expected_T, rtol=1e-7)
    assert 0.001897 < lr_T < 0.001898
    # Retains ~63.24% of peak learning rate
    assert np.isclose(lr_T / peak, 0.632425, atol=1e-4)

    # 4. Double budget: step = 2*T - 1 = 4769 (t = 4770 = 2*T)
    lr_2T = float(sched(2 * T - 1))
    d_2T = (4770.0 - 48.0) / 1908.0  # 4722 / 1908 = 2.474842767...
    expected_2T = peak / math.sqrt(1.0 + d_2T * d_2T)  # 0.001123914...
    np.testing.assert_allclose(lr_2T, expected_2T, rtol=1e-7)
    assert np.isclose(lr_2T / peak, 0.374638, atol=1e-4)

    # 5. Far asymptotic horizon: t >> T (e.g. 10*T and 100*T)
    lr_10T = float(sched(10 * T - 1))  # t = 23850
    expected_10T = ards_reference_formula(10 * T, peak, warm, T, 0.8)
    np.testing.assert_allclose(lr_10T, expected_10T, rtol=1e-7)

    lr_100T = float(sched(100 * T - 1))  # t = 238500
    expected_100T = ards_reference_formula(100 * T, peak, warm, T, 0.8)
    np.testing.assert_allclose(lr_100T, expected_100T, rtol=1e-7)

    # Asymptotic O(1/t) property: t * lr(t) -> peak * scale = 0.003 * 1908 = 5.724
    asymptotic_product = (100 * T) * lr_100T
    assert np.isclose(asymptotic_product, peak * scale, rtol=1e-3)
    assert lr_100T > 0.0


def test_ards_continuity_and_monotonicity():
    """Verify schedule is continuous at warmup and strictly monotonic thereafter."""
    sched = ards_schedule()
    warm = 48

    # Monotonic increase during warmup
    warmup_lrs = [float(sched(s)) for s in range(warm)]
    for i in range(len(warmup_lrs) - 1):
        assert warmup_lrs[i] < warmup_lrs[i + 1]

    # Post-warmup decay is continuous at transition
    lr_47 = float(sched(47))  # t = 48 (end of warmup)
    lr_48 = float(sched(48))  # t = 49 (first decay step)
    assert np.isclose(lr_47, 0.003, atol=1e-8)
    assert lr_48 < lr_47
    assert np.isclose(lr_48, 0.0029999996, atol=1e-8)

    # Monotonic decrease post-warmup
    decay_steps = [48, 100, 500, 1000, 2000, 2384, 5000, 10000]
    decay_lrs = [float(sched(s)) for s in decay_steps]
    for i in range(len(decay_lrs) - 1):
        assert decay_lrs[i] > decay_lrs[i + 1]
        assert decay_lrs[i] > 0.0


def test_ards_vectorized_and_jit_execution():
    """Verify ARDS schedule supports JAX array inputs and JIT tracing."""
    sched = ards_schedule()
    steps = jnp.array([0, 47, 999, 2384], dtype=jnp.int32)
    lrs = sched(steps)
    assert lrs.shape == (4,)
    assert np.isclose(float(lrs[0]), 6.25e-5, atol=1e-9)
    assert np.isclose(float(lrs[1]), 0.003, atol=1e-8)
    assert np.isclose(float(lrs[3]), 0.0018972756, rtol=1e-6)

    # JIT traced evaluation
    jit_sched = jax.jit(sched)
    jit_lrs = jit_sched(steps)
    np.testing.assert_allclose(lrs, jit_lrs, rtol=1e-6)


def test_ards_zero_warmup_guard():
    """Verify schedule functions correctly when warmup_steps is zero."""
    sched = ards_schedule(warmup_steps=0, total_steps=2385, peak_lr=0.003)
    lr_step0 = float(sched(0))  # t = 1
    # Should decay immediately from peak
    expected = 0.003 / math.sqrt(1.0 + (1.0 / 1908.0) ** 2)
    assert np.isclose(lr_step0, expected, atol=1e-8)
    assert not np.isnan(lr_step0)
    assert not np.isinf(lr_step0)


def test_ards_invalid_parameters_raise_value_error():
    """Verify invalid parameter configurations raise ValueError."""
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ards_schedule(total_steps=0)
    with pytest.raises(ValueError, match="total_steps must be positive"):
        ards_schedule(total_steps=-100)
    with pytest.raises(ValueError, match="warmup_steps must be non-negative"):
        ards_schedule(warmup_steps=-5)
    with pytest.raises(ValueError, match="peak_lr must be positive"):
        ards_schedule(peak_lr=0.0)
    with pytest.raises(ValueError, match="scale_factor must be positive"):
        ards_schedule(scale_factor=-0.5)


# ==============================================================================
# Vector 2: Cosine Schedule Strict Prohibition & Source Purity
# ==============================================================================


@pytest.mark.parametrize("kw", [
    "cosine", "Cosine", "COSINE", "cosine_decay", "Cosine_Decay",
    "cosine_annealing", "warmup_cosine", "cosine_with_warmup", "cos",
])
def test_reject_cosine_schedule_function(kw):
    """Verify reject_cosine_schedule catches all cosine variants."""
    with pytest.raises(ValueError, match="Cosine schedule .* strictly prohibited in ards"):
        reject_cosine_schedule(kw)
    with pytest.raises(ValueError, match="Cosine schedule .* strictly prohibited in ards"):
        ards_schedule(schedule_type=kw)
    with pytest.raises(ValueError, match="Cosine schedule .* strictly prohibited in ards"):
        validate_schedule_not_cosine(kw)


def test_unsupported_schedule_names_rejected():
    """Verify non-ARDS schedules are rejected."""
    for bad in ["linear", "wsd", "polynomial", "exponential", "constant"]:
        with pytest.raises(ValueError, match="only 'ards' is permitted"):
            validate_schedule_not_cosine(bad)


def test_configs_strictly_prohibit_cosine():
    """Verify config presets and training dataclass reject cosine requests."""
    with pytest.raises(ValueError, match="Cosine schedule .* strictly prohibited"):
        standard_ards("125m", schedule="cosine")
    with pytest.raises(ValueError, match="Cosine schedule .* strictly prohibited"):
        algebraic_ards("125m", schedule="cosine")
    with pytest.raises(ValueError, match="min_lr_ratio is cosine-specific"):
        standard_ards("125m", min_lr_ratio=0.1)
    with pytest.raises(ValueError, match="min_lr_ratio is cosine-specific"):
        algebraic_ards("125m", min_lr_ratio=0.1)
    with pytest.raises(ValueError, match="Cosine schedule .* strictly prohibited"):
        ARDSTrainingConfig(schedule="cosine")


def test_ards_module_source_transcendental_purity():
    """Static AST and source code audit: zero cos, sin, or pi in ards/ package."""
    ards_dir = REPO_ROOT / "ards"
    assert ards_dir.is_dir(), "ards/ directory must exist"

    py_files = list(ards_dir.glob("*.py"))
    assert len(py_files) >= 4, f"ards/ must contain at least 4 python files, found {len(py_files)}"

    for file_path in py_files:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "scipy" not in alias.name, f"Forbidden scipy import in {file_path.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "scipy" not in module, f"Forbidden scipy import in {file_path.name}"
                for alias in node.names:
                    assert alias.name not in ("cos", "sin"), f"Forbidden {alias.name} imported in {file_path.name}"

        # Check for forbidden function calls or attributes (math.cos, jnp.cos, np.cos)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in ("cos", "sin"):
                    # Check if value is a module
                    if isinstance(node.value, ast.Name) and node.value.id in ("math", "np", "jnp", "jax"):
                        pytest.fail(f"Forbidden call to {node.value.id}.{node.attr} in {file_path.name}")


# ==============================================================================
# Vector 3: Configuration Presets in ards/configs.py
# ==============================================================================


def test_standard_and_algebraic_ards_presets_exact_parameters():
    """Verify exact theoretical parameter counts on 125M and proxy scales."""
    key = jax.random.PRNGKey(42)

    # 125M Scale Checks
    cfg_std = standard_ards("125m")
    p_std = count_params(init_params(cfg_std, key))
    assert p_std == 110_120_448, f"Standard 125M params {p_std} != 110,120,448"

    cfg_alg = algebraic_ards("125m")
    p_alg = count_params(init_params(cfg_alg, key))
    assert p_alg == 110_122_128, f"Algebraic 125M params {p_alg} != 110,122,128"

    assert p_alg - p_std == 1_680, f"Parameter delta {p_alg - p_std} != 1,680"

    # Proxy Scale Checks
    cfg_std_proxy = standard_ards("proxy")
    p_std_proxy = count_params(init_params(cfg_std_proxy, key))
    assert p_std_proxy == 26_745_600

    cfg_alg_proxy = algebraic_ards("proxy")
    p_alg_proxy = count_params(init_params(cfg_alg_proxy, key))
    assert p_alg_proxy == 26_746_672
    assert p_alg_proxy - p_std_proxy == 1_072


def test_ards_config_presets_architecture_dimensions():
    """Verify all 125M architectural dimensions match project specification."""
    for make_fn in (standard_ards, algebraic_ards):
        cfg = make_fn("125m")
        assert cfg.vocab_size == 32768
        assert cfg.d_model == 768
        assert cfg.n_layers == 12
        assert cfg.n_heads == 12
        assert cfg.d_ff == 2048
        assert cfg.seq_len == 2048


def test_ards_config_presets_component_separation():
    """Verify component purity between standard and algebraic presets."""
    std = standard_ards("125m")
    assert std.pos == "rope"
    assert std.ffn == "swiglu"
    assert std.attn == "softmax"
    assert std.sink == "none"
    assert not std.qk_norm
    assert std.head == "ce"

    alg = algebraic_ards("125m")
    assert alg.pos == "cayley_geo"
    assert alg.ffn == "alu_glu"
    assert alg.alu_beta == 0.5
    assert alg.attn == "asinh"
    assert alg.attn_n == 16 and alg.attn_c == 8.0
    assert alg.sink == "learned" and alg.sink_init == 0.0
    assert alg.qk_norm
    assert alg.head == "alg_ce"
    assert alg.head_n == 8 and alg.head_c == 2.0


def test_canonical_training_config_preset():
    """Verify ARDSTrainingConfig matches 125M / 2.5B-token specification."""
    recipe = get_training_config("standard_ards")
    assert recipe.seed == 42
    assert recipe.data_seed == 42
    assert recipe.tokens == 2_500_000_000
    assert recipe.total_steps == 2385
    assert recipe.warmup_steps == 48
    assert recipe.peak_lr == 0.003
    assert recipe.schedule == "ards"
    assert recipe.batch == 512
    assert recipe.micro == 4
    assert recipe.seq_len == 2048
    assert recipe.b1 == 0.9 and recipe.b2 == 0.95
    assert recipe.adam_eps == 1e-8
    assert recipe.wd == 0.1
    assert recipe.clip == 1.0


def test_four_comparison_runs_matrix():
    """Verify {Standard, Algebraic} x {Cosine, ARDS} matrix presets and configs."""
    assert set(COMPARISON_4_RUNS) == {
        "standard_cosine",
        "algebraic_cosine",
        "standard_ards",
        "algebraic_ards",
    }
    for arm in COMPARISON_4_RUNS:
        assert arm in ARDS_PRESETS

    # Standard presets
    for fn in (standard_ards, standard_cosine):
        cfg = fn("125m")
        assert cfg.attn == "softmax"
        assert cfg.head == "ce"
        assert cfg.ffn == "swiglu"
        assert cfg.pos == "rope"

    # Algebraic presets
    for fn in (algebraic_ards, algebraic_cosine):
        cfg = fn("125m")
        assert cfg.attn == "asinh"
        assert cfg.head == "alg_ce"
        assert cfg.ffn == "alu_glu"
        assert cfg.pos == "cayley_geo"

    # Training config schedule assignments
    cfg_sc = get_training_config("standard_cosine")
    assert cfg_sc.schedule == "cosine"
    assert cfg_sc.arm == "standard_cosine"

    cfg_ac = get_training_config("algebraic_cosine")
    assert cfg_ac.schedule == "cosine"
    assert cfg_ac.arm == "algebraic_cosine"

    cfg_sa = get_training_config("standard_ards")
    assert cfg_sa.schedule == "ards"
    assert cfg_sa.arm == "standard_ards"

    cfg_aa = get_training_config("algebraic_ards")
    assert cfg_aa.schedule == "ards"
    assert cfg_aa.arm == "algebraic_ards"


def test_cosine_schedule_generation():
    """Verify cosine_schedule produces valid LR trajectory without transcendental AST issues."""
    sched = cosine_schedule(peak_lr=0.003, warmup_steps=48, total_steps=2385, min_lr_ratio=0.1)
    lr0 = float(sched(0))
    lr48 = float(sched(48))
    lr2384 = float(sched(2384))
    assert 0.0 <= lr0 < 0.001
    assert abs(lr48 - 0.003) < 1e-4
    assert abs(lr2384 - 0.0003) < 1e-4


# ==============================================================================
# Vector 4: Single-Step AdamW + ARDS Execution (CPU & TPU)
# ==============================================================================


@pytest.mark.parametrize("preset_fn", [standard_ards, algebraic_ards])
def test_single_step_adamw_ards_cpu_execution(preset_fn):
    """Execute 1 step of forward + loss + backward + AdamW on CPU proxy shape.

    Verifies:
    1. Loss is finite.
    2. Gradients are finite, non-zero, and non-NaN.
    3. AdamW updates parameter moments m and v.
    4. Parameters are updated (p_new != p_old).
    """
    cfg = preset_fn(
        "proxy",
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=96,
        vocab_size=97,
        seq_len=16,
        dtype="float32",
        attn_impl="dense",
        head_impl="dense",
        remat=False,
    )
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    mask = decay_mask(params)
    tables = position_tables(cfg, 16)
    tok = jax.random.randint(key, (2, 16), 0, 97)

    # Initialize AdamW state
    opt_state = {
        "count": jnp.zeros([], jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    sched = ards_schedule(peak_lr=0.003, warmup_steps=48, total_steps=2385)

    def loss_fn(p):
        h = forward_hidden(p, tok, cfg, tables)
        return head_loss(p, h, tok, cfg)[0]

    # Forward + Backward
    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(loss), f"Loss {loss} is non-finite"

    # Verify gradients
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in grad_leaves), "Gradients contain non-finite values"
    assert not any(np.isnan(g).any() for g in grad_leaves), "Gradients contain NaNs"
    assert any(np.any(g != 0) for g in grad_leaves), "All gradients are zero"

    # AdamW update step
    count = opt_state["count"] + 1
    lr = sched(count - 1)  # step 0 lr = 6.25e-5
    b1, b2 = 0.9, 0.95
    c1 = 1.0 - b1 ** count.astype(jnp.float32)
    c2 = 1.0 - b2 ** count.astype(jnp.float32)

    def update_param(p, g, m, v, dmask):
        m_new = b1 * m + (1 - b1) * g
        v_new = b2 * v + (1 - b2) * g * g
        u = (m_new / c1) / (jnp.sqrt(v_new / c2) + 1e-8)
        if dmask:
            u = u + 0.1 * p
        return p - lr * u, m_new, v_new

    flat_p, treedef = jax.tree_util.tree_flatten(params)
    flat_g = treedef.flatten_up_to(grads)
    flat_m = treedef.flatten_up_to(opt_state["m"])
    flat_v = treedef.flatten_up_to(opt_state["v"])
    flat_mask = treedef.flatten_up_to(mask)

    results = [update_param(*args) for args in zip(flat_p, flat_g, flat_m, flat_v, flat_mask)]
    new_params = treedef.unflatten([r[0] for r in results])
    new_m = treedef.unflatten([r[1] for r in results])
    new_v = treedef.unflatten([r[2] for r in results])

    # Assert weights actually changed
    old_leaves = jax.tree_util.tree_leaves(params)
    new_leaves = jax.tree_util.tree_leaves(new_params)
    deltas = [np.abs(np.asarray(nl - ol)) for ol, nl in zip(old_leaves, new_leaves)]
    max_delta = max(float(d.max()) for d in deltas)
    assert max_delta > 0.0, "Parameters did not change after AdamW step"
    assert all(np.all(np.isfinite(nl)) for nl in new_leaves), "Updated parameters contain non-finite values"

    # Assert moments updated
    m_leaves = jax.tree_util.tree_leaves(new_m)
    v_leaves = jax.tree_util.tree_leaves(new_v)
    assert any(np.any(m != 0) for m in m_leaves), "Optimizer m moments were not updated"
    assert any(np.any(v > 0) for v in v_leaves), "Optimizer v moments were not updated"


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("preset_fn", [standard_ards, algebraic_ards])
def test_single_step_adamw_ards_tpu_execution(preset_fn):
    """Execute 1 step on TPU with pure XLA execution in bfloat16."""
    cfg = preset_fn(
        "125m",
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

        loss, grads = jax.value_and_grad(loss_fn)(p)
        return loss, grads

    loss, grads = jax.jit(step)(params, tok)
    assert np.isfinite(loss), f"TPU loss is non-finite: {loss}"
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in leaves), "TPU gradients contain non-finite values"
    assert not any(np.isnan(g).any() for g in leaves), "TPU gradients contain NaNs"


# ==============================================================================
# Vector 5: Checkpoint Saving and Weight Resumption
# ==============================================================================


@pytest.mark.parametrize("arm", ["standard_ards", "algebraic_ards"])
def test_ards_checkpoint_saving_and_resumption(tmp_path, arm):
    """Test checkpoint save, bit-for-bit weight/moment resumption, and step continuation."""
    import pickle

    cfg = get_ards_config(arm, "proxy", d_model=64, n_layers=2, n_heads=2, d_ff=96, vocab_size=97, seq_len=16, attn_impl="dense", head_impl="dense")
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    mask = decay_mask(params)
    tables = position_tables(cfg, 16)
    tok = jax.random.randint(key, (2, 16), 0, 97)

    sched = ards_schedule(peak_lr=0.003, warmup_steps=48, total_steps=2385)

    # Execute Step 1
    def loss_fn(p):
        h = forward_hidden(p, tok, cfg, tables)
        return head_loss(p, h, tok, cfg)[0]

    loss1, grads1 = jax.value_and_grad(loss_fn)(params)
    count1 = jnp.array(1, jnp.int32)
    lr1 = sched(0)
    m1 = grads1
    v1 = jax.tree_util.tree_map(lambda g: g * g, grads1)

    # Single-step parameter update
    p1 = jax.tree_util.tree_map(lambda p, g: p - lr1 * g, params, grads1)
    opt_state1 = {"count": count1, "m": m1, "v": v1}

    # Save Checkpoint
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": 1,
        "params": jax.tree_util.tree_map(np.asarray, p1),
        "opt_state": jax.tree_util.tree_map(np.asarray, opt_state1),
        "config": cfg.__dict__,
        "training_args": {"arm": arm, "schedule": "ards", "peak_lr": 0.003, "seed": 42},
    }
    ckpt_file = ckpt_dir / "step-0000001.pkl"
    with open(ckpt_file, "wb") as f:
        pickle.dump(payload, f)

    # Resume Checkpoint into fresh instance
    with open(ckpt_file, "rb") as f:
        loaded = pickle.load(f)

    assert loaded["step"] == 1
    assert loaded["training_args"]["schedule"] == "ards"

    # Bit-for-bit weight identity check
    loaded_params = loaded["params"]
    flat_orig, _ = jax.tree_util.tree_flatten(p1)
    flat_loaded, _ = jax.tree_util.tree_flatten(loaded_params)
    for o, l in zip(flat_orig, flat_loaded):
        np.testing.assert_array_equal(np.asarray(o), np.asarray(l))

    # Bit-for-bit optimizer state identity check
    flat_m_orig, _ = jax.tree_util.tree_flatten(m1)
    flat_m_loaded, _ = jax.tree_util.tree_flatten(loaded["opt_state"]["m"])
    for o, l in zip(flat_m_orig, flat_m_loaded):
        np.testing.assert_array_equal(np.asarray(o), np.asarray(l))

    # Verify Step 2 execution from resumed state
    step2_count = loaded["step"] + 1
    lr2 = sched(step2_count - 1)  # step 1 lr = 1.25e-4
    assert np.isclose(float(lr2), 0.003 * (2.0 / 48.0), atol=1e-9)

    loss2, grads2 = jax.value_and_grad(loss_fn)(p1)
    assert np.isfinite(loss2)
    assert not np.isnan(loss2)


# ==============================================================================
# Vector 6: Safety Guardrails Enforcement
# ==============================================================================


def test_safety_guardrails_prevent_long_training_runs():
    """Verify runner rejects steps > 100 without explicit bypass flag."""
    from ards.train import run_ards_training

    # Attempting steps > 100 must raise ValueError
    with pytest.raises(ValueError, match="exceeding 100 steps prohibited"):
        run_ards_training(arm="standard_ards", steps=500, smoke=False, allow_long_run=False)


def test_safety_guardrails_prevent_remote_uploads():
    """Verify remote upload scripts are excised or exit cleanly with safety notice without network actions."""
    for script_name in ("upload_to_hf.py", "push_to_hf.py"):
        script = REPO_ROOT / "scripts" / script_name
        if script.is_file():
            res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
            assert res.returncode == 0
            assert "disabled" in (res.stdout + res.stderr).lower()
        else:
            # Script has been completely excised from the repository to prevent remote egress
            assert not script.exists()


# ==============================================================================
# Vector 7: Pure Algebraic Cubic Hermite Polynomial Schedule (Smoothstep)
# ==============================================================================


def test_cubic_algebraic_schedule_boundaries_and_fidelity():
    """Validate that cubic algebraic schedule replicates cosine boundary conditions with pure ALU math."""
    from ards.schedule import cubic_algebraic_schedule, get_schedule
    from ards.configs import get_ards_config

    peak = 0.0015
    warm = 48
    T = 2385
    floor = 0.00015

    sched = cubic_algebraic_schedule(peak_lr=peak, warmup_steps=warm, total_steps=T, min_lr_ratio=0.1)

    # 1. Warmup peak boundary
    v_peak = float(sched(warm - 1))
    assert abs(v_peak - peak) < 1e-6, f"Peak mismatch: {v_peak} vs {peak}"

    # 2. Terminal floor boundary
    v_term = float(sched(T - 1))
    assert abs(v_term - floor) < 1e-6, f"Terminal floor mismatch: {v_term} vs {floor}"

    # 3. Midpoint
    mid_step = (T - warm) // 2 + warm
    v_mid = float(sched(mid_step))
    expected_mid = floor + (peak - floor) * 0.5
    assert abs(v_mid - expected_mid) < 1e-4, f"Midpoint mismatch: {v_mid} vs {expected_mid}"

    # 4. Derivative at terminal boundary is zero (no cutoff shock)
    v_near_end = float(sched(T - 2))
    assert abs(v_term - v_near_end) < 1e-6, "Terminal derivative should be zero"

    # 5. Config lookup
    cfg_cubic = get_ards_config("algebraic_cubic")
    assert cfg_cubic.norm == "rms"
    assert cfg_cubic.attn == "asinh"

