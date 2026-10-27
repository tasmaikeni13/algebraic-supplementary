#!/usr/bin/env python3
"""Unified Fast Smoke Test Runner for All 16 Architectures.

Verifies:
1. All 14 hybrid presets from hybrid/registry.py (H01 - H14).
2. Comparison presets from ards/configs.py (standard_ards, algebraic_ards, standard_cosine, algebraic_cosine).
3. 1-step synthetic AdamW optimization pass with pure XLA execution.
4. Finite positive loss and non-zero finite gradients.
5. Exact parameter count:
   - 110,120,448 for softmax attention (125M)
   - 110,122,128 for asinh attention (125M, +1,680 delta)
6. Optional evaluation smoke (--eval-smoke) across:
   - 8 Public Zero-Shot benchmark tasks (arc_easy, arc_challenge, hellaswag, piqa, openbookqa, winogrande, boolq, lambada)
   - 4 Bounded-Context Recall tasks (needle_passkey, associative_recall, multi_needle, latest_value)
7. Generates comprehensive summary JSON report.
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# The 16 canonical architecture names
CANONICAL_HYBRIDS: List[str] = [
    "hybrid_cayley_swiglu_softmax_ce",       # H01
    "hybrid_rope_alu_softmax_ce",             # H02
    "hybrid_rope_swiglu_asinh_ce",           # H03
    "hybrid_rope_swiglu_softmax_alg_ce",     # H04
    "hybrid_cayley_alu_softmax_ce",           # H05
    "hybrid_cayley_swiglu_asinh_ce",         # H06
    "hybrid_cayley_swiglu_softmax_alg_ce",   # H07
    "hybrid_rope_alu_asinh_ce",               # H08
    "hybrid_rope_alu_softmax_alg_ce",         # H09
    "hybrid_rope_swiglu_asinh_alg_ce",       # H10
    "hybrid_rope_alu_asinh_alg_ce",           # H11
    "hybrid_cayley_swiglu_asinh_alg_ce",     # H12
    "hybrid_cayley_alu_softmax_alg_ce",       # H13
    "hybrid_cayley_alu_asinh_ce",             # H14
]

CANONICAL_ARDS: List[str] = [
    "standard_ards",
    "algebraic_ards",
]

ALL_16_PRESETS: List[str] = CANONICAL_HYBRIDS + CANONICAL_ARDS

EXPECTED_PARAMS_125M = {
    "softmax": 110_120_448,
    "asinh": 110_122_128,
}

EXPECTED_PARAMS_PROXY = {
    "softmax": 26_745_600,
    "asinh": 26_746_672,
}


def resolve_preset_name(name: str) -> Tuple[str, str]:
    """Resolve preset name to (canonical_name, category) where category is 'hybrid' or 'ards'."""
    from hybrid.registry import HYBRID_ALIASES, HYBRID_PRESETS, ORDERED_HYBRID_NAMES
    from ards.configs import ARDS_PRESETS

    name_clean = name.strip()
    if name_clean in HYBRID_ALIASES:
        return HYBRID_ALIASES[name_clean], "hybrid"
    if name_clean in ORDERED_HYBRID_NAMES or name_clean in HYBRID_PRESETS:
        return name_clean, "hybrid"
    if name_clean in ARDS_PRESETS:
        if name_clean in ("standard_cosine", "algebraic_cosine"):
            return name_clean, "ards"
        canonical = "standard_ards" if name_clean in ("standard", "standard_ards") else "algebraic_ards"
        return canonical, "ards"

    # Check case-insensitive
    name_lower = name_clean.lower()
    for k, v in HYBRID_ALIASES.items():
        if k.lower() == name_lower:
            return v, "hybrid"
    for k in ORDERED_HYBRID_NAMES:
        if k.lower() == name_lower:
            return k, "hybrid"
    for k in ARDS_PRESETS:
        if k.lower() == name_lower:
            if k in ("standard_cosine", "algebraic_cosine"):
                return k, "ards"
            canonical = "standard_ards" if k in ("standard", "standard_ards") else "algebraic_ards"
            return canonical, "ards"

    raise ValueError(f"Unknown preset {name!r}. Available: {ALL_16_PRESETS}")


def load_model_config(
    canonical_name: str,
    category: str,
    size: str = "125m",
    seq_len: int = 2048,
    batch: int = 4,
    attn_impl: str = "dense",
    head_impl: str = "dense",
    flash_block: int = 512,
    head_block_n: int = 128,
    head_block_v: int = 512,
    dtype: str = "bfloat16",
    tiny: bool = False,
):
    """Construct LMConfig for the given preset."""
    from hybrid.registry import get_hybrid
    from ards.configs import get_ards_config

    overrides: Dict[str, Any] = {
        "seq_len": seq_len,
        "attn_impl": attn_impl,
        "head_impl": head_impl,
        "flash_block": flash_block,
        "head_block_n": head_block_n,
        "head_block_v": head_block_v,
        "dtype": dtype,
        "remat": True,
    }

    if tiny:
        overrides.update(
            d_model=128,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            flash_block=128,
            seq_len=128,
            remat=False,
        )

    if category == "hybrid":
        return get_hybrid(canonical_name)(size=size, **overrides)
    else:
        return get_ards_config(canonical_name, size=size, **overrides)


def execute_single_smoke_step(
    preset_name: str,
    size: str = "125m",
    batch: int = 4,
    seq_len: int = 2048,
    steps: int = 1,
    seed: int = 42,
    tiny: bool = False,
    is_tpu: bool = True,
) -> Dict[str, Any]:
    """Execute 1-step synthetic AdamW optimization pass on a single architecture."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from algebraic.model import (
        count_params,
        decay_mask,
        forward_hidden,
        head_loss,
        init_params,
        position_tables,
    )

    canonical_name, category = resolve_preset_name(preset_name)

    # Pure XLA dense execution
    attn_impl = "dense"
    head_impl = "dense"
    dtype = "bfloat16" if is_tpu else "float32"
    effective_seq = 128 if tiny else seq_len
    effective_batch = 1 if tiny else batch
    flash_block = 128 if tiny else 512

    cfg = load_model_config(
        canonical_name=canonical_name,
        category=category,
        size=size,
        seq_len=effective_seq,
        batch=effective_batch,
        attn_impl=attn_impl,
        head_impl=head_impl,
        flash_block=flash_block,
        dtype=dtype,
        tiny=tiny,
    )

    # 1. Parameter initialization & parameter count verification
    key = jax.random.PRNGKey(seed)
    params = init_params(cfg, key)
    n_params = count_params(params)

    expected_params = None
    if not tiny:
        expected_lookup = EXPECTED_PARAMS_125M if size == "125m" else EXPECTED_PARAMS_PROXY
        expected_params = expected_lookup[cfg.attn]
        assert n_params == expected_params, (
            f"Parameter count mismatch for {canonical_name}: "
            f"got {n_params}, expected {expected_params} (attn={cfg.attn})"
        )

    # 2. Synthetic token data generation
    data_key, _ = jax.random.split(key)
    tokens = jax.random.randint(data_key, (effective_batch, effective_seq), 0, cfg.vocab_size)
    tables = position_tables(cfg, effective_seq)
    mask = decay_mask(params)

    # 3. AdamW State Initialization
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

    # 4. Define Step Function
    def train_step(p_state, o_state, x, y):
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

    jitted_step = jax.jit(train_step)

    # 5. Execute Steps
    t0 = time.monotonic()
    curr_params = params
    curr_opt = opt_state
    last_metrics = None

    for _ in range(steps):
        curr_params, curr_opt, last_metrics = jitted_step(curr_params, curr_opt, tokens, tokens)
        jax.block_until_ready(last_metrics["loss"])

    step_duration = time.monotonic() - t0

    # 6. Verify Numerical Stability
    final_loss = float(last_metrics["loss"])
    final_gnorm = float(last_metrics["gnorm"])

    assert math.isfinite(final_loss), f"Loss for {canonical_name} is non-finite: {final_loss}"
    assert final_loss > 0.0, f"Loss for {canonical_name} is non-positive: {final_loss}"

    # Gradient checks
    g_leaves = jax.tree_util.tree_leaves(last_metrics["grads"])
    assert all(np.all(np.isfinite(np.asarray(g))) for g in g_leaves), f"Gradients contain non-finite values for {canonical_name}"
    assert not any(np.any(np.isnan(np.asarray(g))) for g in g_leaves), f"Gradients contain NaNs for {canonical_name}"
    assert any(np.any(np.asarray(g) != 0) for g in g_leaves), f"Gradients are all zero for {canonical_name}"

    # Parameter update check
    old_leaves = jax.tree_util.tree_leaves(params)
    new_leaves = jax.tree_util.tree_leaves(curr_params)
    deltas = [np.max(np.abs(np.asarray(nl) - np.asarray(ol))) for ol, nl in zip(old_leaves, new_leaves)]
    max_delta = float(max(deltas))
    assert max_delta > 0.0, f"Weights did not mutate after AdamW step for {canonical_name}"

    result = {
        "preset": canonical_name,
        "category": category,
        "pos": cfg.pos,
        "ffn": cfg.ffn,
        "attn": cfg.attn,
        "head": cfg.head,
        "parameters": n_params,
        "expected_parameters": expected_params,
        "param_match": (n_params == expected_params) if expected_params else True,
        "final_loss": final_loss,
        "final_gnorm": final_gnorm,
        "max_param_delta": max_delta,
        "completed_steps": steps,
        "seq_len": effective_seq,
        "batch": effective_batch,
        "attn_impl": cfg.attn_impl,
        "head_impl": cfg.head_impl,
        "dtype": cfg.dtype,
        "step_seconds": step_duration,
        "passed": True,
    }

    # Free memory
    del params, curr_params, curr_opt, last_metrics
    jax.clear_caches()

    return result


def run_evaluation_smoke(
    benchmarks_dir: Optional[Path] = None,
    recall_dir: Optional[Path] = None,
    limit: int = 1,
    batch: int = 1,
) -> Dict[str, Any]:
    """Execute fast 8-task zero-shot + 4-task recall smoke evaluation on representative model."""
    import jax
    from algebraic.benchmarking import PUBLIC_TASKS, Scorer
    from algebraic.config import build
    from algebraic.model import init_params
    from algebraic.recall_benchmarks import TASKS as RECALL_TASKS
    from scripts.evaluate_zero_shot import evaluate as evaluate_zero_shot, find_public_benchmarks
    from scripts.evaluate_recall import evaluate as evaluate_recall, find_recall_benchmarks

    benchmarks_dir = find_public_benchmarks(benchmarks_dir)
    recall_dir = find_recall_benchmarks(recall_dir)

    print(f"\n[eval_smoke] Running evaluation smoke (limit={limit})...", flush=True)
    print(f"[eval_smoke] Zero-shot directory: {benchmarks_dir}", flush=True)
    print(f"[eval_smoke] Recall directory:    {recall_dir}", flush=True)

    is_tpu = (jax.default_backend() == "tpu")
    cfg = build(
        "standard",
        "125m",
        d_model=128,
        n_layers=1,
        n_heads=2,
        d_ff=256,
        seq_len=2048,
        dtype="bfloat16" if is_tpu else "float32",
        attn_impl="dense",
    )
    scorer = Scorer(cfg, init_params(cfg, jax.random.PRNGKey(42)), batch=batch)

    # 1. Zero-shot 8 public tasks
    t_zs = time.monotonic()
    zs_report, zs_preds = evaluate_zero_shot(scorer, benchmarks_dir, list(PUBLIC_TASKS), limit=limit)
    zs_duration = time.monotonic() - t_zs

    assert set(zs_report["results"].keys()) == set(PUBLIC_TASKS), "Missing tasks in zero-shot smoke"
    assert len(zs_preds) == len(PUBLIC_TASKS) * limit, "Mismatch in zero-shot prediction count"

    # 2. Recall 4 bounded-context tasks
    t_rc = time.monotonic()
    rc_report, rc_preds = evaluate_recall(scorer, recall_dir, list(RECALL_TASKS), limit=limit)
    rc_duration = time.monotonic() - t_rc

    assert set(rc_report["results"].keys()) == set(RECALL_TASKS), "Missing tasks in recall smoke"
    assert len(rc_preds) == len(RECALL_TASKS) * limit, "Mismatch in recall prediction count"
    assert not any(r["evidence_truncated"] for r in rc_preds), "Evidence truncated in recall smoke"

    del scorer
    jax.clear_caches()

    return {
        "status": "passed",
        "limit": limit,
        "zero_shot": {
            "tasks": list(PUBLIC_TASKS),
            "results": {k: v["accuracy"] for k, v in zs_report["results"].items()},
            "seconds": zs_duration,
        },
        "recall": {
            "tasks": list(RECALL_TASKS),
            "results": {k: [cell["exact_match"] for cell in v] for k, v in rc_report["results"].items()},
            "evidence_truncated": False,
            "seconds": rc_duration,
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default=None, help="run specific preset (canonical or alias, e.g. H01)")
    p.add_argument("--all", action="store_true", help="run all 16 architectures")
    p.add_argument("--out", type=Path, default=None, help="output report JSON or directory path")
    p.add_argument("--chip", type=int, default=int(os.environ.get("TPU_VISIBLE_CHIPS") or 0), choices=(0, 1, 2, 3), help="target TPU chip (0-3, default from TPU_VISIBLE_CHIPS or 0)")
    p.add_argument("--size", default="125m", choices=("125m", "proxy"), help="model scale (default 125m)")
    p.add_argument("--batch", type=int, default=4, help="synthetic global batch (default 4)")
    p.add_argument("--steps", type=int, default=1, help="optimizer steps per model (default 1)")
    p.add_argument("--allow-long-run", action="store_true", default=False, help="allow long training runs exceeding 100 steps")
    p.add_argument("--seq-len", type=int, default=2048, help="sequence context length (default 2048)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--tiny", action="store_true", help="use tiny shapes for rapid CPU smoke")
    p.add_argument("--eval-smoke", action="store_true", help="run fast benchmark evaluation smoke (8 zero-shot + 4 recall tasks)")
    p.add_argument("--eval-limit", type=int, default=1, help="row limit for evaluation smoke (default 1)")
    p.add_argument("--device", choices=("auto", "tpu", "cpu"), default="auto", help="target compute device (default auto)")
    p.add_argument("--no-reexec", action="store_true", help="disable automatic tpu_python.sh re-exec")
    args = p.parse_args(argv)

    # Safety Guardrails
    if args.steps > 100 and not args.allow_long_run:
        raise ValueError(
            f"Training runs exceeding 100 steps prohibited in test/development mode (got {args.steps} steps). "
            f"Use --steps <= 100 or --allow-long-run."
        )

    # Device configuration
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    # 1. Automatic TPU Chip Environment Re-Execution
    if args.device != "cpu" and not args.no_reexec and (ROOT / "scripts/tpu_python.sh").exists():
        expected_chip_str = str(args.chip)
        current_visible = os.environ.get("TPU_VISIBLE_CHIPS")
        if current_visible != expected_chip_str:
            reexec_cmd = [
                "bash",
                str(ROOT / "scripts/tpu_python.sh"),
                expected_chip_str,
                str(Path(__file__).resolve()),
                *(argv if argv is not None else sys.argv[1:]),
                "--no-reexec",
            ]
            print(f"[smoke_all] Re-executing via: {' '.join(reexec_cmd)}", flush=True)
            res = subprocess.run(reexec_cmd)
            return res.returncode

    # 2. JAX Import & Hardware Detection
    import jax
    jax.config.update("jax_enable_x64", False)

    devices = jax.devices()
    backend = jax.default_backend()
    is_tpu = (backend == "tpu")
    print(f"[smoke_all] Backend: {backend} | Devices: {devices} | Scale: {args.size}", flush=True)

    # 3. Determine Presets to Test
    if args.preset:
        canonical_name, _ = resolve_preset_name(args.preset)
        target_presets = [canonical_name]
    else:
        target_presets = list(ALL_16_PRESETS)

    print(f"[smoke_all] Executing fast 1-step AdamW smoke test across {len(target_presets)} architecture(s)...", flush=True)

    results: Dict[str, Any] = {}
    total_passed = 0
    t_start_all = time.monotonic()

    for idx, preset_name in enumerate(target_presets, start=1):
        print(f"[{idx:02d}/{len(target_presets):02d}] Testing {preset_name}...", end=" ", flush=True)
        try:
            res = execute_single_smoke_step(
                preset_name=preset_name,
                size=args.size,
                batch=args.batch,
                seq_len=args.seq_len,
                steps=args.steps,
                seed=args.seed,
                tiny=args.tiny,
                is_tpu=is_tpu,
            )
            results[res["preset"]] = res
            total_passed += 1
            print(
                f"PASSED (loss={res['final_loss']:.4f}, gnorm={res['final_gnorm']:.4e}, "
                f"delta={res['max_param_delta']:.4f}, params={res['parameters']:,}, {res['step_seconds']:.2f}s)"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            results[preset_name] = {
                "preset": preset_name,
                "passed": False,
                "error": str(e),
            }

    total_duration = time.monotonic() - t_start_all
    all_passed = (total_passed == len(target_presets))

    # Optional Evaluation Smoke
    eval_smoke_res = None
    if args.eval_smoke:
        try:
            eval_smoke_res = run_evaluation_smoke(limit=args.eval_limit)
        except Exception as e:
            print(f"[eval_smoke] FAILED: {e}", flush=True)
            eval_smoke_res = {
                "status": "failed",
                "error": str(e),
            }
            all_passed = False

    # 4. Construct Structured Verification Report
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "16-architecture unified synthetic smoke verification",
        "all_passed": all_passed,
        "total_passed": total_passed,
        "total_targets": len(target_presets),
        "total_duration_seconds": total_duration,
        "backend": backend,
        "devices": [str(d) for d in devices],
        "scale": args.size,
        "batch": args.batch,
        "seq_len": 128 if args.tiny else args.seq_len,
        "steps": args.steps,
        "pure_xla_execution": True,
        "results": results,
    }
    if eval_smoke_res:
        report["evaluation_smoke"] = eval_smoke_res

    # 5. Write Report JSON (support file path or directory)
    if args.out is not None:
        out_path = args.out
        if out_path.is_dir() or not out_path.suffix:
            out_name = f"smoke_{args.preset}.json" if args.preset else "smoke_all.json"
            out_path = out_path / out_name
    elif args.preset:
        preset_name = args.preset
        out_path = Path(f"/tmp/smoke_{preset_name}.json")
    else:
        out_path = ROOT / "results/verification/smoke_all.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, indent=2)
    out_path.write_text(report_json)
    if args.preset and args.out is None:
        canonical_name, _ = resolve_preset_name(args.preset)
        if canonical_name != args.preset:
            alias_path = Path(f"/tmp/smoke_{canonical_name}.json")
            alias_path.write_text(report_json)
    print(f"\n[smoke_all] Report written to: {out_path}", flush=True)

    # 6. Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'#':<3} {'Preset Name':<38} {'Attn':<8} {'Params':<12} {'Loss':<10} {'Status':<8}")
    print("-" * 90)
    for i, p_name in enumerate(target_presets, start=1):
        info = results.get(p_name, {})
        if not info or not info.get("passed", False):
            # Try resolved name
            try:
                c_name, _ = resolve_preset_name(p_name)
                info = results.get(c_name, {})
            except Exception:
                pass
        status = "PASSED" if info.get("passed", False) else "FAILED"
        attn = info.get("attn", "N/A")
        params_str = f"{info.get('parameters', 0):,}" if "parameters" in info else "N/A"
        loss_str = f"{info.get('final_loss', 0.0):.4f}" if "final_loss" in info else "N/A"
        print(f"{i:<3} {p_name:<38} {attn:<8} {params_str:<12} {loss_str:<10} {status:<8}")
    print("=" * 90)
    print(f"Summary: {total_passed}/{len(target_presets)} passed in {total_duration:.2f}s.\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
