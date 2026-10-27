"""CLI tool for listing, inspecting, and verifying hybrid transformer presets."""

import argparse
import sys
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from hybrid.model import build_hybrid_model, count_hybrid_params, init_hybrid_params
from hybrid.registry import (
    HYBRID_METADATA,
    HYBRID_PRESETS,
    ORDERED_HYBRID_NAMES,
    build_hybrid,
    describe_hybrids,
    get_hybrid,
    list_hybrids,
)


def print_list():
    """Print formatted table of all 14 hybrid presets."""
    hybrids = describe_hybrids()
    header = f"{'Index':<6} {'Preset Name':<38} {'Pos':<12} {'FFN':<10} {'Attn':<10} {'Head':<8} {'125M Params':<14} {'Proxy Params':<14}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for h in hybrids:
        print(
            f"{h['index']:<6} {h['name']:<38} {h['pos']:<12} {h['ffn']:<10} "
            f"{h['attn']:<10} {h['head']:<8} {h['params_125m']:<14,d} {h['params_proxy']:<14,d}"
        )
    print(sep)
    print(f"Total hybrid presets: {len(hybrids)}")


def print_info(preset_name: str, size: str = "125m"):
    """Print detailed configuration of a hybrid preset."""
    cfg = build_hybrid(preset_name, size=size)
    print(f"Preset: {preset_name} (Size: {size})")
    print(f"  Positional: {cfg.pos}")
    print(f"  FFN:        {cfg.ffn} (alu_beta={cfg.alu_beta})")
    print(f"  Attention:  {cfg.attn} (n={cfg.attn_n}, c={cfg.attn_c}, qk_norm={cfg.qk_norm}, sink={cfg.sink})")
    print(f"  Head:       {cfg.head} (n={cfg.head_n}, c={cfg.head_c})")
    print(f"  Dimensions: d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads}, d_ff={cfg.d_ff}, vocab={cfg.vocab_size}")

    key = jax.random.PRNGKey(0)
    params = init_hybrid_params(cfg, key)
    total_params = count_hybrid_params(params)
    print(f"  Parameter Count: {total_params:,d}")


def verify_params() -> bool:
    """Verify exact parameter counts for all 14 presets across 125M and proxy scales."""
    print("Verifying parameter count exactness across all 14 hybrid presets...")
    key = jax.random.PRNGKey(0)
    all_passed = True

    for name in ORDERED_HYBRID_NAMES:
        meta = HYBRID_METADATA[name]

        # Check 125M
        cfg_125m = build_hybrid(name, size="125m")
        p_125m = count_hybrid_params(init_hybrid_params(cfg_125m, key))
        exp_125m = meta.params_125m
        if p_125m != exp_125m:
            print(f"FAIL [125M] {name} ({meta.index}): actual {p_125m:,d} != expected {exp_125m:,d}")
            all_passed = False
        else:
            print(f"PASS [125M] {name} ({meta.index}): {p_125m:,d}")

        # Check proxy
        cfg_proxy = build_hybrid(name, size="proxy")
        p_proxy = count_hybrid_params(init_hybrid_params(cfg_proxy, key))
        exp_proxy = meta.params_proxy
        if p_proxy != exp_proxy:
            print(f"FAIL [proxy] {name} ({meta.index}): actual {p_proxy:,d} != expected {exp_proxy:,d}")
            all_passed = False
        else:
            print(f"PASS [proxy] {name} ({meta.index}): {p_proxy:,d}")

    if all_passed:
        print("All 14 hybrid presets match exact parameter count specifications on both 125M and proxy scales!")
    return all_passed


def run_smoke_step(preset_name: str, size: str = "proxy") -> bool:
    """Execute a single synthetic training step (forward + loss + grad) for a preset."""
    print(f"Running synthetic smoke step for preset {preset_name} (size: {size})...")
    cfg = build_hybrid(preset_name, size=size, remat=False)
    key = jax.random.PRNGKey(42)
    k_init, k_data = jax.random.split(key)

    model = build_hybrid_model(cfg, k_init)
    batch_size = 2
    seq_len = min(cfg.seq_len, 64)
    tokens = jax.random.randint(k_data, (batch_size, seq_len), 0, cfg.vocab_size)
    targets = jax.random.randint(k_data, (batch_size, seq_len), 0, cfg.vocab_size)

    def loss_fn(params):
        loss, _ = model.forward_and_loss(tokens, targets, params=params)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(model.params)
    loss_val = float(loss)
    leaves = jax.tree_util.tree_leaves(grads)
    all_finite = np.isfinite(loss_val) and all(np.all(np.isfinite(g)) for g in leaves)

    if all_finite:
        print(f"Smoke step SUCCESS: loss = {loss_val:.4f}, finite gradients verified across {len(leaves)} arrays.")
        return True
    else:
        print(f"Smoke step FAILED: non-finite loss or gradients encountered.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Hybrid Transformer Presets CLI")
    parser.add_argument("--list", action="store_true", help="List all 14 hybrid presets")
    parser.add_argument("--info", type=str, metavar="PRESET", help="Show info for a preset")
    parser.add_argument("--size", type=str, default="125m", help="Size for info/smoke (125m or proxy)")
    parser.add_argument("--verify-params", action="store_true", help="Verify exact parameter counts for all presets")
    parser.add_argument("--smoke-step", type=str, metavar="PRESET", help="Run 1-step synthetic training on a preset")

    args = parser.parse_args()

    if args.list:
        print_list()
        return

    if args.info:
        print_info(args.info, size=args.size)
        return

    if args.verify_params:
        success = verify_params()
        sys.exit(0 if success else 1)

    if args.smoke_step:
        success = run_smoke_step(args.smoke_step, size=args.size)
        sys.exit(0 if success else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
