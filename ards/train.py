#!/usr/bin/env python3
"""Dedicated Single-Seed AdamW + ARDS Training Pipeline.

Trains 125M Standard or Pure Algebraic Transformer models using:
- Single seed 42
- ARDS (Algebraic Rational Decay Schedule)
- Decoupled AdamW optimizer
- Pure XLA execution (dense attention, loss head, QK RMS-norm)
- Fast synthetic smoke testing via --smoke / --steps 1
- Safety guardrails preventing accidental 2.5B-token long runs.

Example (smoke test on single TPU chip):
  bash scripts/tpu_python.sh 0 -m ards.train --arm standard_ards --smoke --out /tmp/smoke_std_test
"""

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Ensure repository root and src/ are in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ards.configs import (
    ARDS_PRESETS,
    ARDSTrainingConfig,
    COMPARISON_4_RUNS,
    get_ards_config,
    validate_schedule,
    validate_schedule_not_cosine,
)
from ards.schedule import ards_schedule, cosine_schedule, get_schedule


class SyntheticWindowSampler:
    """In-memory synthetic token batch generator for fast smoke verification."""

    def __init__(self, seq_len: int, global_batch: int, seed: int, vocab_size: int = 32768):
        self.T = seq_len
        self.B = global_batch
        self.seed = seed
        self.vocab_size = vocab_size

    def batch(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng([self.seed, step])
        tokens = rng.integers(0, self.vocab_size, size=(self.B, self.T + 1), dtype=np.int32)
        return tokens[:, :-1], tokens[:, 1:]


def save_checkpoint(directory: Path, step: int, payload: dict, keep: int = 2) -> None:
    """Atomically write training state checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"step-{step:07d}.pkl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    for old in sorted(directory.glob("step-*.pkl"))[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def latest_checkpoint(directory: Path) -> Optional[Path]:
    """Find the highest-numbered checkpoint in directory."""
    found = sorted(Path(directory).glob("step-*.pkl"))
    return found[-1] if found else None


def run_ards_training(
    arm: str = "standard_ards",
    size: str = "125m",
    seed: int = 42,
    steps: int = 1,
    tokens: float = 0,
    seq_len: int = 2048,
    batch: int = 512,
    micro: int = 4,
    lr: float = 3e-3,
    warmup: float = 48,
    total_steps: int = 2385,
    wd: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
    adam_eps: float = 1e-8,
    clip: float = 1.0,
    attn_impl: str = "dense",
    head_impl: str = "dense",
    flash_block: int = 512,
    head_block_n: int = 128,
    head_block_v: int = 512,
    train_path: Optional[str] = None,
    valid_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    smoke: bool = False,
    tiny: bool = False,
    save_params: bool = True,
    resume: bool = False,
    allow_long_run: bool = False,
    **overrides: Any,
) -> Dict[str, Any]:
    """Execute AdamW training run for one of the 4 comparison arms using pure XLA execution."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental.shard_map import shard_map
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from algebraic.model import (
        count_params,
        decay_mask,
        eval_logprobs,
        forward_hidden,
        head_loss,
        init_params,
        position_tables,
        telemetry_is_probability,
    )
    from algebraic.train import WindowSampler

    # 1. Purity & Schedule Validation
    sched_name = overrides.get("schedule")
    if sched_name is None:
        sched_name = "cosine" if "cosine" in arm else "ards"
    validate_schedule(sched_name)
    if arm in ("standard_ards", "algebraic_ards") and "cos" in str(sched_name).lower():
        raise ValueError(
            f"Cosine schedule is strictly prohibited for {arm}; "
            f"use pure ARDS algebraic rational decay, or arm='standard_cosine' / 'algebraic_cosine'."
        )

    # 2. Safety Guardrails (R5)
    planned_steps = steps if steps > 0 else (math.ceil(tokens / (batch * seq_len)) if tokens > 0 else 1)
    if planned_steps > 100 and not allow_long_run:
        raise ValueError(
            f"Training runs exceeding 100 steps prohibited in test/development mode (got {planned_steps} steps). "
            f"Use --smoke, --steps <= 100, or explicitly pass allow_long_run=True."
        )

    # 3. Model Configuration
    if tiny:
        overrides.update(d_model=128, n_layers=1, n_heads=2, d_ff=256)
        flash_block = 128
        seq_len = 128
        batch = 1
        micro = 1

    # CPU compatibility fallback: if not on TPU, fallback to dense kernels
    is_tpu = jax.devices()[0].platform == "tpu"
    if not is_tpu:
        attn_impl = "dense"
        head_impl = "dense"

    cfg = get_ards_config(
        arm,
        size=size,
        seq_len=seq_len,
        attn_impl=attn_impl,
        head_impl=head_impl,
        flash_block=flash_block,
        head_block_n=head_block_n,
        head_block_v=head_block_v,
        **overrides,
    )

    # 4. Device Mesh & Topology
    devices = jax.devices()
    n_dev = len(devices)
    pidx, pcount = jax.process_index(), jax.process_count()
    is_main = pidx == 0
    mesh = Mesh(np.array(devices), ("data",))

    if batch % (n_dev * micro) != 0:
        raise ValueError(f"batch {batch} not divisible by devices*micro {n_dev}*{micro}")
    accum = batch // (n_dev * micro)

    out = Path(out_dir or tempfile.mkdtemp(prefix="ards-run-"))
    if is_main:
        out.mkdir(parents=True, exist_ok=True)

    # 5. Model Parameters & Position Tables
    tables = position_tables(cfg, seq_len)
    params = init_params(cfg, jax.random.PRNGKey(seed))
    mask = decay_mask(params)
    n_params = count_params(params)
    is_prob = telemetry_is_probability(cfg)

    # 6. Scheduler (ARDS or Cosine for the 4 comparison runs)
    warm_steps = int(round(warmup * total_steps)) if warmup < 1 else int(warmup)
    if "cos" in sched_name.lower():
        min_ratio = float(overrides.get("min_lr_ratio", 0.1))
        sched_fn = cosine_schedule(peak_lr=lr, warmup_steps=warm_steps, total_steps=total_steps, min_lr_ratio=min_ratio)
    else:
        sched_fn = ards_schedule(peak_lr=lr, warmup_steps=warm_steps, total_steps=total_steps)

    opt_state = {
        "count": jnp.zeros([], jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }

    start_step = 0
    if resume:
        latest = latest_checkpoint(out)
        if latest is not None:
            with open(latest, "rb") as fh:
                loaded = pickle.load(fh)
            t_args = loaded.get("training_args", {})
            ckpt_sched = t_args.get("schedule", "ards")
            if ckpt_sched != sched_name:
                raise ValueError(f"Cannot resume checkpoint trained with different schedule: {ckpt_sched} != {sched_name}")
            if t_args.get("peak_lr") != lr or t_args.get("seed") != seed:
                raise ValueError("checkpoint training recipe does not match this run")
            params = jax.tree_util.tree_map(jnp.asarray, loaded["params"])
            opt_state = jax.tree_util.tree_map(jnp.asarray, loaded["opt_state"])
            start_step = loaded["step"]

    # 7. Optimizer Step Definition
    def local_step(p_state, o_state, xs, ys):
        def micro_fn(carry, batch_data):
            gsum, lsum = carry
            x, y = batch_data

            def loss_fn(p):
                h = forward_hidden(p, x, cfg, tables)
                return head_loss(p, h, y, cfg)

            (_, tel), g = jax.value_and_grad(loss_fn, has_aux=True)(p_state)
            nll = -jnp.log(jnp.maximum(tel, 1e-30)) if is_prob else tel
            gsum = jax.tree_util.tree_map(lambda a, b: a + b.astype(jnp.float32), gsum, g)
            return (gsum, lsum + jnp.mean(nll)), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, p_state)
        (g, lsum), _ = lax.scan(micro_fn, (zeros, jnp.zeros([], jnp.float32)), (xs, ys))
        g = jax.tree_util.tree_map(lambda a: lax.pmean(a / accum, "data"), g)
        loss = lax.pmean(lsum / accum, "data")
        gnorm = jnp.sqrt(sum(jnp.sum(a * a) for a in jax.tree_util.tree_leaves(g)))
        scale = jnp.minimum(1.0, clip / jnp.maximum(gnorm, 1e-12))
        count = o_state["count"] + 1
        current_lr = sched_fn(count - 1)
        c1 = 1.0 - b1 ** count.astype(jnp.float32)
        c2 = 1.0 - b2 ** count.astype(jnp.float32)

        def upd(p, gi, m_val, v_val, dmask):
            gi = gi * scale
            m_new = b1 * m_val + (1.0 - b1) * gi
            v_new = b2 * v_val + (1.0 - b2) * gi * gi
            u = (m_new / c1) / (jnp.sqrt(v_new / c2) + adam_eps)
            if dmask:
                u = u + wd * p
            return p - current_lr * u, m_new, v_new

        flat_p, tdef = jax.tree_util.tree_flatten(p_state)
        flat_g = tdef.flatten_up_to(g)
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
        return p_updated, o_updated, {"loss": loss, "gnorm": gnorm, "lr": current_lr}

    rep = P()
    data_spec = P(None, "data", None)
    step_fn = jax.jit(
        shard_map(
            local_step,
            mesh=mesh,
            in_specs=(rep, rep, data_spec, data_spec),
            out_specs=(rep, rep, rep),
            check_rep=False,
        ),
        donate_argnums=(0, 1),
    )

    # 8. Data Sampler (Synthetic fallback if files missing or in smoke mode)
    use_synthetic = smoke or not train_path or not os.path.exists(train_path)
    if use_synthetic:
        sampler = SyntheticWindowSampler(seq_len, batch, seed, cfg.vocab_size)
    else:
        sampler = WindowSampler(train_path, seq_len, batch, seed, pidx, pcount)

    rep_sharding = NamedSharding(mesh, rep)
    batch_sharding = NamedSharding(mesh, data_spec)
    params = jax.device_put(params, rep_sharding)
    opt_state = jax.device_put(opt_state, rep_sharding)

    def put_global(local_arr, sharding, global_shape):
        return jax.make_array_from_process_local_data(sharding, local_arr, global_shape)

    local_rows = batch // pcount
    global_shape = (accum, batch // accum, seq_len)

    # 9. Training Loop
    history = []
    t_start = time.time()
    for step in range(start_step, planned_steps):
        x_np, y_np = sampler.batch(step)
        x_np = x_np.reshape(local_rows // accum, accum, seq_len).transpose(1, 0, 2)
        y_np = y_np.reshape(local_rows // accum, accum, seq_len).transpose(1, 0, 2)
        x = put_global(np.ascontiguousarray(x_np), batch_sharding, global_shape)
        y = put_global(np.ascontiguousarray(y_np), batch_sharding, global_shape)

        params, opt_state, metrics = step_fn(params, opt_state, x, y)
        if step == start_step:
            jax.block_until_ready(metrics["loss"])

        m = {k: float(v) for k, v in metrics.items()}
        history.append({"step": step + 1, **m})

        if not all(math.isfinite(v) for v in m.values()):
            raise FloatingPointError(f"Non-finite metrics at step {step + 1}: {m}")

    jax.block_until_ready(params)
    train_seconds = time.time() - t_start

    result = {
        "status": "passed",
        "arm": arm,
        "seed": seed,
        "parameters": n_params,
        "completed_steps": planned_steps,
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "final_gnorm": history[-1]["gnorm"] if history else float("nan"),
        "final_lr": history[-1]["lr"] if history else float("nan"),
        "train_seconds": train_seconds,
        "config": asdict(cfg),
        "synthetic": use_synthetic,
    }

    if is_main:
        if save_params:
            host = jax.tree_util.tree_map(lambda a: np.asarray(jax.device_get(a)), params)
            flat, _ = jax.tree_util.tree_flatten_with_path(host)
            np.savez(out / "params.npz", **{jax.tree_util.keystr(k): v for k, v in flat})
        (out / "result.json").write_text(json.dumps(result, indent=2))
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "args": {
                        "arm": arm,
                        "seed": seed,
                        "steps": planned_steps,
                        "schedule": sched_name,
                        "peak_lr": lr,
                        "batch": batch,
                        "seq_len": seq_len,
                    },
                    "n_params": n_params,
                    "total_steps": planned_steps,
                },
                indent=2,
            )
        )
        # Save step checkpoint for resumption
        payload = {
            "step": planned_steps,
            "params": jax.tree_util.tree_map(lambda a: np.asarray(jax.device_get(a)), params),
            "opt_state": jax.tree_util.tree_map(lambda a: np.asarray(jax.device_get(a)), opt_state),
            "config": asdict(cfg),
            "training_args": {
                "arm": arm,
                "schedule": sched_name,
                "peak_lr": lr,
                "seed": seed,
            },
        }
        save_checkpoint(out, planned_steps, payload, keep=2)

    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="standard_ards", choices=list(ARDS_PRESETS.keys()))
    p.add_argument("--size", default="125m", choices=("125m", "proxy"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--tokens", type=float, default=0)
    p.add_argument("--smoke", action="store_true", help="fast 1-step synthetic smoke run")
    p.add_argument("--tiny", action="store_true", help="reduced dimensions for CPU smoke")
    p.add_argument("--schedule", default=None, choices=("ards", "cosine"), help="schedule: 'ards' or 'cosine'")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=float, default=48)
    p.add_argument("--total-steps", type=int, default=2385)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--train", dest="train_path", default=None)
    p.add_argument("--valid", dest="valid_path", default=None)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--save-params", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-long-run", action="store_true", help="bypass >100 steps safety guard")
    args = p.parse_args(argv)

    if args.schedule is None:
        args.schedule = "cosine" if "cosine" in args.arm else "ards"
    validate_schedule(args.schedule)
    steps = 1 if args.smoke else (args.steps or 1)

    res = run_ards_training(
        arm=args.arm,
        size=args.size,
        seed=args.seed,
        steps=steps,
        tokens=args.tokens,
        seq_len=args.seq_len,
        batch=args.batch,
        micro=args.micro,
        lr=args.lr,
        warmup=args.warmup,
        total_steps=args.total_steps,
        train_path=args.train_path,
        valid_path=args.valid_path,
        out_dir=args.out_dir,
        smoke=args.smoke,
        tiny=args.tiny,
        save_params=args.save_params,
        resume=args.resume,
        allow_long_run=args.allow_long_run,
    )
    print(json.dumps({k: res[k] for k in ("arm", "status", "parameters", "completed_steps", "final_loss", "final_lr")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
