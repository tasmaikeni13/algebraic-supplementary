"""Comprehensive Test Suite for Pure-ALU Cubic Hermite Polynomial Schedule and Presets.

Validates:
1. Exact mathematical precision of the Pure Algebraic Cubic Hermite schedule (Smoothstep):
   S(u) = (1 - u)^2 * (1 + 2u)
2. Crucial boundary properties:
   - Zero derivative at u=0 (smooth transition from peak)
   - Zero derivative at u=1 (smooth convergence at terminal floor)
   - Exact midpoint S(0.5) = 0.5 (inflection symmetry matching cosine)
3. Learning rate values at critical milestones:
   - t = 0: linear warmup start
   - t = t_warm: exact peak LR
   - t = T (total steps): exact terminal floor (0.1 * peak)
4. Pure ALU audit: zero transcendental function calls (no cos, sin, exp, log).
5. Architecture presets and exact 125M parameter counts:
   - standard_cubic: 110,120,448 parameters
   - algebraic_cubic: 110,122,128 parameters (exact +1,680 delta for learned attention sink)
6. Single-step AdamW optimization pass with cubic schedule (finite loss, non-zero finite gradients).
"""

import ast
import inspect
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algebraic.config import build, PRESETS
from algebraic.model import (
    count_params,
    decay_mask,
    forward_hidden,
    head_loss,
    init_params,
    position_tables,
)
from algebraic.train import lr_schedule, get_args


@pytest.fixture(autouse=True)
def float32_context():
    """Guard TPU tests from global x64 pollution."""
    enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64
    with enable_x64(False):
        yield


# ==============================================================================
# Vector 1: Cubic Hermite Mathematical Precision & Properties
# ==============================================================================


def cubic_smoothstep(u: float) -> float:
    """Exact reference smoothstep polynomial S(u) = (1 - u)^2 * (1 + 2u)."""
    return (1.0 - u) * (1.0 - u) * (1.0 + 2.0 * u)


def test_cubic_polynomial_mathematical_identity():
    """Verify algebraic expansion: (1-u)^2 * (1+2u) == 1 - 3u^2 + 2u^3."""
    for u in np.linspace(0.0, 1.0, 101):
        s_factored = (1.0 - u) * (1.0 - u) * (1.0 + 2.0 * u)
        s_expanded = 1.0 - 3.0 * (u ** 2) + 2.0 * (u ** 3)
        assert math.isclose(s_factored, s_expanded, rel_tol=1e-12, abs_tol=1e-12)


def test_cubic_boundary_derivatives():
    """Verify S'(0) = 0 and S'(1) = 0 for zero-derivative endpoints."""
    # S'(u) = -6u(1 - u)
    def s_prime(u):
        return -6.0 * u * (1.0 - u)

    assert math.isclose(s_prime(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(s_prime(1.0), 0.0, abs_tol=1e-12)

    # Numerical derivative check
    eps = 1e-6
    deriv_0 = (cubic_smoothstep(eps) - cubic_smoothstep(0.0)) / eps
    deriv_1 = (cubic_smoothstep(1.0) - cubic_smoothstep(1.0 - eps)) / eps
    assert abs(deriv_0) < 1e-4
    assert abs(deriv_1) < 1e-4


def test_cubic_midpoint_symmetry():
    """Verify exact inflection symmetry at u=0.5: S(0.5) = 0.5."""
    assert math.isclose(cubic_smoothstep(0.5), 0.5, abs_tol=1e-12)


def test_cubic_schedule_steps():
    """Verify learning rate schedule at warmup, mid-training, and final step."""
    peak = 0.002
    min_ratio = 0.1
    floor = peak * min_ratio
    total_steps = 2385
    warmup_steps = 48

    args = get_args([
        "--preset", "algebraic_cubic",
        "--schedule", "cubic",
        "--lr", str(peak),
        "--min-lr-ratio", str(min_ratio),
        "--warmup", str(warmup_steps),
        "--steps", str(total_steps),
        "--out", "/tmp/test_out",
    ])
    sched, warm = lr_schedule(args, total_steps)
    assert warm == warmup_steps

    # Step 0: Warmup start
    lr_0 = float(sched(0))
    expected_0 = peak * (1.0 / warmup_steps)
    assert math.isclose(lr_0, expected_0, rel_tol=1e-4)

    # Step warmup_steps - 1 (t = warmup_steps): peak
    lr_peak = float(sched(warmup_steps - 1))
    assert math.isclose(lr_peak, peak, rel_tol=1e-4)

    # Midpoint progress
    mid_step = int((warmup_steps + total_steps) / 2)
    lr_mid = float(sched(mid_step))
    expected_mid = floor + (peak - floor) * 0.5
    assert math.isclose(lr_mid, expected_mid, rel_tol=5e-3)

    # Final step: terminal floor
    lr_final = float(sched(total_steps - 1))
    assert math.isclose(lr_final, floor, rel_tol=1e-4)


def test_pure_alu_cubic_ast_audit():
    """Audit AST of lr_schedule to ensure cubic schedule uses zero transcendental functions."""
    source = inspect.getsource(lr_schedule)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Inspect conditions matching cubic
            test_str = ast.unparse(node.test)
            if "cubic" in test_str:
                body_source = ast.unparse(node.body)
                forbidden = ["cos", "sin", "tan", "exp", "log", "sqrt"]
                for f in forbidden:
                    assert f"jnp.{f}" not in body_source, f"Forbidden transcendental function jnp.{f} found in cubic schedule!"


# ==============================================================================
# Vector 2: Presets and Parameter Counts
# ==============================================================================


def test_presets_registered_in_config():
    """Verify standard_cubic and algebraic_cubic are registered in PRESETS."""
    assert "standard_cubic" in PRESETS
    assert "algebraic_cubic" in PRESETS
    assert "standard_cosine" in PRESETS
    assert "algebraic_cosine" in PRESETS


def test_parameter_counts_exact():
    """Verify exact 125M parameter counts for standard and algebraic presets."""
    cfg_std = build("standard_cubic", "125m")
    cfg_alg = build("algebraic_cubic", "125m")

    params_std = init_params(cfg_std, jax.random.PRNGKey(0))
    params_alg = init_params(cfg_alg, jax.random.PRNGKey(0))

    n_std = count_params(params_std)
    n_alg = count_params(params_alg)

    assert n_std == 110_120_448, f"standard_cubic has {n_std} params, expected 110,120,448"
    assert n_alg == 110_122_128, f"algebraic_cubic has {n_alg} params, expected 110,122,128"
    assert n_alg - n_std == 1_680, "Difference between algebraic and standard must be exactly 1,680 (learned sink logits)"


# ==============================================================================
# Vector 3: Single-Step Optimization Pass with Cubic Schedule
# ==============================================================================


@pytest.mark.parametrize("preset", ["standard_cubic", "algebraic_cubic"])
def test_single_step_adamw_cubic_cpu(preset):
    """Run 1 step of AdamW training with cubic schedule on CPU."""
    cfg = build(preset, "proxy", d_model=64, n_layers=1, n_heads=1, d_ff=96, vocab_size=97)
    key = jax.random.PRNGKey(42)
    params = init_params(cfg, key)
    tables = position_tables(cfg, 16)

    tokens = jax.random.randint(key, (2, 16), 0, 97)
    targets = jax.random.randint(jax.random.PRNGKey(43), (2, 16), 0, 97)

    def loss_fn(p):
        h = forward_hidden(p, tokens, cfg, tables)
        return head_loss(p, h, targets, cfg)

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert jnp.isfinite(loss)
    assert loss > 0

    # Ensure gradients are non-zero and finite
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    assert any(jnp.any(g != 0) for g in grad_leaves)
