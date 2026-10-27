"""Comprehensive Requirement-Driven Test Suite for 16-Architecture Smoke Verification.

Validates Requirement R5 and Milestone 4 Acceptance Criteria:
1. Exact parameter counts for all 16 models (14 hybrid presets + 2 ARDS variants):
   - 8 Softmax attention models: exact 110,120,448 parameters (125M)
   - 8 Asinh attention models: exact 110,122,128 parameters (125M, +1,680 delta)
2. Single-step synthetic AdamW optimization pass on CPU (proxy shapes, fast execution).
3. Single-step synthetic AdamW optimization pass on TPU with pure XLA:
   - dense attention (softmax and asinh)
   - loss head (cross-entropy and algebraic likelihood)
   - qk_rms_norm (RMS-norm hook for asinh models)
   - Strictly finite positive loss, non-zero finite bfloat16 gradients, weight mutation.
4. CLI smoke runner scripts/smoke_all.py verification (options: --preset, --all, --out, --chip, --eval-smoke).
5. Fast smoke evaluation routes for zero-shot benchmarks (8 tasks) and recall benchmarks (4 tasks).
6. Safety isolation: remote upload stubs exit 0 without network access; long-training guard intercepts >100 steps.
7. Regression test for ards/train.py --tiny duplicate flash_block keyword argument.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic.model import (
    count_params,
    decay_mask,
    forward_hidden,
    head_loss,
    init_params,
    position_tables,
)
from ards.configs import ARDS_PRESETS, get_ards_config
from hybrid.registry import HYBRID_PRESETS, get_hybrid

REPO_ROOT = Path(__file__).resolve().parent.parent

# 14 Hybrid Presets + 2 ARDS Presets = 16 Architectures
HYBRID_NAMES = sorted(HYBRID_PRESETS.keys())
ARDS_NAMES = ["standard_ards", "algebraic_ards"]
ALL_16_PRESETS = HYBRID_NAMES + ARDS_NAMES


@pytest.fixture(autouse=True)
def float32_context():
    """Guard TPU Mosaic lowering from global x64 pollution."""
    enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    with enable_x64(False):
        yield


def get_model_config(preset: str, size: str = "125m", **overrides):
    """Helper to load config from hybrid or ards registries."""
    if preset in HYBRID_PRESETS:
        return get_hybrid(preset)(size=size, **overrides)
    elif preset in ARDS_PRESETS:
        return get_ards_config(preset, size=size, **overrides)
    raise KeyError(f"Unknown preset {preset!r}")


# ==============================================================================
# Vector 1: Exact Parameter Counts (All 16 Architectures)
# ==============================================================================


@pytest.mark.parametrize("preset", ALL_16_PRESETS)
def test_all_16_architectures_exact_parameter_counts_125m(preset):
    """Verify exact parameter counts at 125M scale for all 16 architectures."""
    cfg = get_model_config(preset, size="125m")
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    n_params = count_params(params)

    if cfg.attn == "softmax":
        assert n_params == 110_120_448, f"{preset} (softmax) expected 110,120,448, got {n_params}"
    elif cfg.attn == "asinh":
        assert n_params == 110_122_128, f"{preset} (asinh) expected 110,122,128, got {n_params}"
    else:
        pytest.fail(f"Unexpected attention mechanism: {cfg.attn}")


def test_asinh_delta_is_exactly_1680_parameters():
    """Verify that asinh attention adds exactly +1,680 parameters over softmax."""
    key = jax.random.PRNGKey(0)
    std_cfg = get_model_config("standard_ards", size="125m")
    alg_cfg = get_model_config("algebraic_ards", size="125m")

    n_std = count_params(init_params(std_cfg, key))
    n_alg = count_params(init_params(alg_cfg, key))

    assert n_std == 110_120_448
    assert n_alg == 110_122_128
    assert n_alg - n_std == 1680, f"Expected delta of 1,680, got {n_alg - n_std}"


@pytest.mark.parametrize("preset", ALL_16_PRESETS)
def test_all_16_architectures_exact_parameter_counts_proxy(preset):
    """Verify exact parameter counts at proxy scale for all 16 architectures."""
    cfg = get_model_config(preset, size="proxy")
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    n_params = count_params(params)

    if cfg.attn == "softmax":
        assert n_params == 26_745_600, f"{preset} (proxy softmax) expected 26,745,600, got {n_params}"
    elif cfg.attn == "asinh":
        assert n_params == 26_746_672, f"{preset} (proxy asinh) expected 26,746,672, got {n_params}"


# ==============================================================================
# Vector 2: CPU Synthetic Forward, Loss, and Gradients (Proxy Scale)
# ==============================================================================


@pytest.mark.parametrize("preset", ALL_16_PRESETS)
def test_cpu_synthetic_adamw_pass_proxy(preset):
    """Execute 1-step synthetic AdamW pass on CPU proxy shape across all 16 models."""
    cfg = get_model_config(
        preset,
        size="proxy",
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

    opt_state = {
        "count": jnp.zeros([], jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    def loss_fn(p):
        h = forward_hidden(p, tok, cfg, tables)
        return head_loss(p, h, tok, cfg)[0]

    loss, grads = jax.value_and_grad(loss_fn)(params)
    assert np.isfinite(loss) and loss > 0.0, f"Loss is non-finite or non-positive: {loss}"

    # Verify gradients
    g_leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in g_leaves)
    assert not any(np.isnan(g).any() for g in g_leaves)
    assert any(np.any(g != 0) for g in g_leaves)

    # AdamW update
    count = opt_state["count"] + 1
    lr = 0.003
    b1, b2 = 0.9, 0.95
    c1 = 1.0 - b1 ** count.astype(jnp.float32)
    c2 = 1.0 - b2 ** count.astype(jnp.float32)

    def upd(p, g, m, v, dmask):
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

    results = [upd(*args) for args in zip(flat_p, flat_g, flat_m, flat_v, flat_mask)]
    new_params = treedef.unflatten([r[0] for r in results])

    old_leaves = jax.tree_util.tree_leaves(params)
    new_leaves = jax.tree_util.tree_leaves(new_params)
    deltas = [np.abs(np.asarray(nl - ol)) for ol, nl in zip(old_leaves, new_leaves)]
    assert max(float(d.max()) for d in deltas) > 0.0, "Weights did not update after AdamW step"


# ==============================================================================
# Vector 3: TPU Pure XLA 1-Step AdamW Pass (All 16 Architectures)
# ==============================================================================


@pytest.mark.skipif(jax.devices()[0].platform != "tpu", reason="needs a TPU")
@pytest.mark.parametrize("preset", ALL_16_PRESETS)
def test_tpu_synthetic_adamw_step_all_16(preset):
    """Execute 1-step synthetic AdamW pass on TPU with pure XLA execution in bfloat16."""
    cfg = get_model_config(
        preset,
        size="125m",
        seq_len=512,
        attn_impl="dense",
        head_impl="dense",
        flash_block=512,
        head_block_n=128,
        head_block_v=512,
        dtype="bfloat16",
        remat=True,
    )
    key = jax.random.PRNGKey(101)
    params = init_params(cfg, key)
    mask = decay_mask(params)
    tok = jax.random.randint(key, (2, 512), 0, cfg.vocab_size)
    tables = position_tables(cfg, 512)

    opt_state = {
        "count": jnp.zeros([], jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    lr = 3e-3
    b1, b2 = 0.9, 0.95
    adam_eps = 1e-8
    wd = 0.1
    clip = 1.0

    def step(p_state, o_state, x, y):
        def loss_fn(p):
            h = forward_hidden(p, x, cfg, tables)
            return head_loss(p, h, y, cfg)[0]

        loss, grads = jax.value_and_grad(loss_fn)(p_state)
        gnorm = jnp.sqrt(sum(jnp.sum(a * a) for a in jax.tree_util.tree_leaves(grads)))
        scale = jnp.minimum(1.0, clip / jnp.maximum(gnorm, 1e-12))
        count = o_state["count"] + 1
        c1 = 1.0 - b1 ** count.astype(jnp.float32)
        c2 = 1.0 - b2 ** count.astype(jnp.float32)

        def upd(p, gi, m_val, v_val, dmask):
            gi = gi * scale
            m_new = b1 * m_val + (1.0 - b1) * gi
            v_new = b2 * v_val + (1.0 - b2) * gi * gi
            u = (m_new / c1) / (jnp.sqrt(v_new / c2) + adam_eps)
            if dmask:
                u = u + wd * p
            return p - lr * u, m_new, v_new

        flat_p, tdef = jax.tree_util.tree_flatten(p_state)
        flat_g = tdef.flatten_up_to(grads)
        flat_m = tdef.flatten_up_to(o_state["m"])
        flat_v = tdef.flatten_up_to(o_state["v"])
        flat_mask = tdef.flatten_up_to(mask)
        new_vals = [upd(*z) for z in zip(flat_p, flat_g, flat_m, flat_v, flat_mask)]
        p_updated = tdef.unflatten([a for a, _, _ in new_vals])
        o_updated = {
            "count": count,
            "m": tdef.unflatten([b for _, b, _ in new_vals]),
            "v": tdef.unflatten([c for _, _, c in new_vals]),
        }
        return p_updated, o_updated, {"loss": loss, "gnorm": gnorm, "grads": grads}

    p_new, o_new, met = jax.jit(step)(params, opt_state, tok, tok)
    jax.block_until_ready(met["loss"])

    loss_val = float(met["loss"])
    assert np.isfinite(loss_val) and loss_val > 0.0, f"TPU loss is non-finite or non-positive: {loss_val}"

    # Verify gradients
    g_leaves = jax.tree_util.tree_leaves(met["grads"])
    assert all(np.all(np.isfinite(np.asarray(g))) for g in g_leaves), "TPU gradients contain non-finite values"
    assert not any(np.isnan(np.asarray(g)).any() for g in g_leaves), "TPU gradients contain NaNs"
    assert any(np.any(np.asarray(g) != 0) for g in g_leaves), "TPU gradients are all zero"

    # Verify parameter mutation
    old_leaves = jax.tree_util.tree_leaves(params)
    new_leaves = jax.tree_util.tree_leaves(p_new)
    deltas = [np.max(np.abs(np.asarray(nl) - np.asarray(ol))) for ol, nl in zip(old_leaves, new_leaves)]
    assert max(deltas) > 0.0, "Weights did not update after TPU AdamW step"


# ==============================================================================
# Vector 4: CLI Smoke Runner Interface (scripts/smoke_all.py)
# ==============================================================================


def test_smoke_all_cli_single_preset(tmp_path):
    """Test scripts/smoke_all.py running a single preset via CLI."""
    out_file = tmp_path / "smoke_single.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_all.py"),
        "--preset", "H01",
        "--tiny",
        "--steps", "1",
        "--device", "cpu",
        "--out", str(out_file),
        "--no-reexec",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed:\n{res.stderr}\n{res.stdout}"
    assert out_file.exists(), "Output report file was not created"

    report = json.loads(out_file.read_text())
    assert report["all_passed"] is True
    assert report["total_passed"] == 1
    assert "hybrid_cayley_swiglu_softmax_ce" in report["results"]
    entry = report["results"]["hybrid_cayley_swiglu_softmax_ce"]
    assert entry["passed"] is True
    assert entry["final_loss"] > 0.0
    assert entry["max_param_delta"] > 0.0


def test_smoke_all_cli_invalid_preset(tmp_path):
    """Test scripts/smoke_all.py properly fails on invalid preset name."""
    out_file = tmp_path / "smoke_invalid.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_all.py"),
        "--preset", "nonexistent_model_xyz",
        "--tiny",
        "--device", "cpu",
        "--out", str(out_file),
        "--no-reexec",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode != 0, "CLI should fail on nonexistent preset"


def test_smoke_all_cli_eval_smoke(tmp_path):
    """Test scripts/smoke_all.py running with --eval-smoke."""
    bench_meta = Path.home() / "data" / "evals" / "metadata.json"
    if not bench_meta.exists():
        pytest.skip("Frozen evaluation benchmarks not present on local machine")

    out_file = tmp_path / "smoke_eval.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_all.py"),
        "--preset", "standard_ards",
        "--tiny",
        "--steps", "1",
        "--device", "cpu",
        "--eval-smoke",
        "--eval-limit", "1",
        "--out", str(out_file),
        "--no-reexec",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI execution failed:\n{res.stderr}\n{res.stdout}"
    assert out_file.exists(), "Output report file was not created"

    report = json.loads(out_file.read_text())
    assert report["all_passed"] is True
    assert "evaluation_smoke" in report
    eval_res = report["evaluation_smoke"]
    assert eval_res["status"] == "passed"
    assert len(eval_res["zero_shot"]["tasks"]) == 8
    assert len(eval_res["recall"]["tasks"]) == 4


# ==============================================================================
# Vector 5: Fast Zero-Shot & Recall Evaluator Smoke
# ==============================================================================


def test_zero_shot_and_recall_smoke_execution():
    """Verify scripts/smoke_benchmarks.py runs evaluator smoke without missing tasks."""
    from scripts.evaluate_zero_shot import find_public_benchmarks
    from scripts.evaluate_recall import find_recall_benchmarks

    benchmarks_dir = find_public_benchmarks()
    recall_dir = find_recall_benchmarks()

    if not benchmarks_dir.exists() or not recall_dir.exists():
        pytest.skip("Frozen evaluation benchmarks not present on local machine")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_benchmarks.py"),
        "--benchmarks", str(benchmarks_dir),
        "--recall-benchmarks", str(recall_dir),
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Benchmark smoke failed:\n{res.stderr}\n{res.stdout}"


# ==============================================================================
# Vector 6: Safety Isolation Enforcement
# ==============================================================================


def test_safety_guardrails_prevent_remote_uploads():
    """Verify upload scripts are safely stubbed or excised to prevent remote egress."""
    for script_name in ("upload_to_hf.py", "push_to_hf.py"):
        script_path = REPO_ROOT / "scripts" / script_name
        if script_path.is_file():
            res = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
            assert res.returncode == 0, f"{script_name} failed with code {res.returncode}"
            assert "disabled" in res.stdout.lower() or "safely stubbed" in res.stdout.lower() or "safety" in res.stdout.lower()
        else:
            assert not script_path.exists()


def test_safety_guardrails_intercept_long_runs():
    """Verify ARDS training runner intercepts runs exceeding 100 steps."""
    from ards.train import run_ards_training
    with pytest.raises(ValueError, match="Training runs exceeding 100 steps prohibited"):
        run_ards_training(arm="standard_ards", steps=101, allow_long_run=False)


def test_safety_guardrails_cli_intercept_long_runs(tmp_path):
    """Verify scripts/smoke_all.py intercepts >100 steps unless --allow-long-run is passed."""
    # 1. Prohibited run (>100 steps without --allow-long-run)
    cmd_prohibited = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_all.py"),
        "--steps", "101",
        "--tiny",
        "--device", "cpu",
        "--no-reexec",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    res_fail = subprocess.run(cmd_prohibited, env=env, capture_output=True, text=True)
    assert res_fail.returncode != 0, "CLI execution should fail when --steps > 100 without --allow-long-run"
    assert "Training runs exceeding 100 steps prohibited" in (res_fail.stderr + res_fail.stdout)

    # 2. Allowed run (>100 steps with --allow-long-run)
    out_file = tmp_path / "smoke_allowed.json"
    cmd_allowed = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "smoke_all.py"),
        "--preset", "H01",
        "--steps", "101",
        "--allow-long-run",
        "--tiny",
        "--device", "cpu",
        "--out", str(out_file),
        "--no-reexec",
    ]
    res_pass = subprocess.run(cmd_allowed, env=env, capture_output=True, text=True)
    assert res_pass.returncode == 0, f"CLI execution should succeed with --allow-long-run:\n{res_pass.stderr}\n{res_pass.stdout}"
    assert out_file.exists(), "Output report file was not created"
    data = json.loads(out_file.read_text())
    assert data["all_passed"] is True
    assert data["steps"] == 101


# ==============================================================================
# Vector 7: Regression Test for ards/train.py --tiny Finding
# ==============================================================================


def test_regression_ards_train_tiny_flash_block(tmp_path):
    """Verify that ards/train.py with --tiny executes without multiple values for 'flash_block'."""
    from ards.train import run_ards_training

    # Must complete step 1 without TypeError
    out_std = tmp_path / "std_tiny"
    res_std = run_ards_training(arm="standard_ards", tiny=True, steps=1, out_dir=str(out_std))
    assert res_std["status"] == "passed"
    assert res_std["completed_steps"] == 1

    out_alg = tmp_path / "alg_tiny"
    res_alg = run_ards_training(arm="algebraic_ards", tiny=True, steps=1, out_dir=str(out_alg))
    assert res_alg["status"] == "passed"
    assert res_alg["completed_steps"] == 1


# ==============================================================================
# Vector 8: 16-Architecture Verification Artifact Integrity
# ==============================================================================


def test_smoke_all_verification_artifact_integrity():
    """Assert results/verification/smoke_all.json contains all 16 architectures, total_passed == 16, and valid metrics."""
    artifact_path = REPO_ROOT / "results" / "verification" / "smoke_all.json"
    if not artifact_path.exists():
        pytest.skip("results/verification/smoke_all.json does not exist")

    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert data.get("all_passed") is True
    assert data.get("total_passed") == 16
    assert data.get("total_targets") == 16

    results = data.get("results", {})
    assert len(results) == 16, f"Expected 16 results, got {len(results)}"

    from scripts.smoke_all import ALL_16_PRESETS
    assert set(results.keys()) == set(ALL_16_PRESETS)

    for preset_name, res in results.items():
        assert res.get("passed") is True, f"{preset_name} did not pass"
        assert res.get("param_match") is True, f"{preset_name} parameter match failed"
        assert res.get("parameters", 0) > 0, f"{preset_name} invalid parameters"
        loss = res.get("final_loss")
        assert loss is not None and not math.isnan(loss) and not math.isinf(loss) and loss > 0, f"{preset_name} invalid loss: {loss}"
        gnorm = res.get("final_gnorm")
        assert gnorm is not None and not math.isnan(gnorm) and not math.isinf(gnorm) and gnorm > 0, f"{preset_name} invalid gnorm: {gnorm}"
        delta = res.get("max_param_delta")
        assert delta is not None and not math.isnan(delta) and not math.isinf(delta) and delta > 0, f"{preset_name} invalid delta: {delta}"
        assert res.get("attn_impl") == "flash"
        assert res.get("head_impl") == "fused"

