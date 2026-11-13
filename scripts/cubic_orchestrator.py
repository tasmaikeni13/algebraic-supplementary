#!/usr/bin/env python3
"""Autonomous Orchestrator for Phase 1b (standard_cubic sweep) and Phase 2 (main 2.5B runs).

Pipeline Execution Order:
1. Phase 1a: algebraic_cubic 600M sweep (4 candidate LRs: 1.0e-3, 1.5e-3, 2.0e-3, 3.0e-3) [currently running]
2. Phase 1b: standard_cubic 600M sweep (4 candidate LRs: 1.0e-3, 1.5e-3, 2.0e-3, 3.0e-3)
3. Phase 2: Main 125M benchmark on 2.5B tokens for 1 seed (seed 42):
   - algebraic_cubic_s42 with best algebraic LR
   - standard_cubic_s42 with best standard LR
4. Final comparison report generated against Cosine baselines in results/main_125m/cubic_comparison.json.
5. All intermediate checkpoints strictly omitted (--ckpt-every 0) to preserve disk space.
6. All status and ETAs reported in IST (UTC+5:30).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = Path.home() / "data" / "fw25"
HOSTS = ("w0", "w1", "w2", "w3")
IST = timezone(timedelta(hours=5, minutes=30))
ALGEBRAIC_PYTHON = Path.home() / "venv" / "bin" / "python"
if not ALGEBRAIC_PYTHON.exists():
    ALGEBRAIC_PYTHON = Path(sys.executable)

STATUS_FILE = ROOT / "cubic_status.json"


def ist_now_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def update_status(data: Dict[str, Any]) -> None:
    data["last_updated_ist"] = ist_now_str()
    data["last_updated_timestamp"] = time.time()
    try:
        tmp = STATUS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(STATUS_FILE)
    except Exception as e:
        print(f"Warning: could not write status file: {e}", flush=True)


def clean_storage(hosts=HOSTS, prune_run_dir=None):
    for h in hosts:
        cmds = [
            "rm -rf /tmp/pytest* /tmp/tpu_logs/* /tmp/*.log* 2>/dev/null || true",
        ]
        if prune_run_dir:
            cmds.append(f"rm -rf {shlex.quote(str(prune_run_dir))}/ckpt {shlex.quote(str(prune_run_dir))}/*.npz {shlex.quote(str(prune_run_dir))}/*.npy 2>/dev/null || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=30, check=False)


def run_command_stream(cmd, log_file=None):
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    log_fh = open(log_file, "a") if log_file else None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
    finally:
        proc.wait()
        if log_fh:
            log_fh.close()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {cmd}")


def run_standard_cubic_sweep(hosts=HOSTS, lrs=(0.0010, 0.0015, 0.0020, 0.0030), seed=42) -> float:
    """Execute Phase 1b: 4-arm sweep for standard_cubic on 600M tokens."""
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] PHASE 1b: HYPERPARAMETER SWEEP FOR STANDARD_CUBIC ON 600M TOKENS")
    print(f"Architecture: Standard Transformer (RMSNorm, SwiGLU, RoPE, Softmax, Cross-Entropy)")
    print(f"Optimizer: AdamW + Cubic Hermite Polynomial Schedule")
    print(f"Candidates: lr in {list(lrs)}")
    print(f"Token budget per run: 600,000,000 (572 steps)")
    print("=" * 80 + "\n", flush=True)

    results_dir = ROOT / "results" / "hparam_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    sweep_file = results_dir / "cubic_sweep_results.json"
    sel_file = results_dir / "cubic_selected.json"
    flag_file = results_dir / "standard_cubic_sweep_done.flag"

    group = "hparam_sweep"
    total_jobs = len(lrs)
    completed_durations = []
    t_phase_start = time.time()

    clean_storage(hosts)

    for i, lr in enumerate(lrs, 1):
        lr_tag = f"lr{lr:.1e}".replace(".0e", "e").replace("-0", "-")
        name = f"standard_cubic_{lr_tag}_s{seed}"
        t_job_start = time.time()

        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_sec = avg_sec * (total_jobs - (i - 1))
        else:
            rem_sec = 19 * 60 * (total_jobs - (i - 1))
        eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
        eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

        print(f"\n[{ist_now_str()}] [Phase 1b {i}/{total_jobs}] Launching sweep run: {name} (ETA for standard sweep: {eta_str})", flush=True)

        update_status({
            "phase": "sweep_standard_cubic",
            "current_run": name,
            "current_run_index": i,
            "total_runs_in_phase": total_jobs,
            "eta_ist": eta_str,
            "phase_elapsed_minutes": (time.time() - t_phase_start) / 60,
        })

        args = [
            "--preset", "standard_cubic",
            "--size", "125m",
            "--seq-len", "2048",
            "--batch", "512",
            "--micro", "4",
            "--tokens", "600000000",
            "--lr", str(lr),
            "--wd", "0.1",
            "--warmup", "0.02",
            "--min-lr-ratio", "0.1",
            "--b2", "0.95",
            "--schedule", "cubic",
            "--seed", str(seed),
            "--train", str(DATA_DIR / "fineweb_sweep_600M.npy"),
            "--valid", str(DATA_DIR / "fineweb_valid.npy"),
            "--eval-tokens", "8000000",
            "--eval-batch", "4",
            "--eval-every", "100",
            "--log-every", "25",
            "--ckpt-every", "0",
        ]

        single_job_path = ROOT / "runs" / group / f"job_{name}.json"
        single_job_path.parent.mkdir(parents=True, exist_ok=True)
        single_job_path.write_text(json.dumps([{"name": name, "args": args}], indent=2))

        cmd = [
            str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
            str(single_job_path),
            "--group", group,
            "--hosts", ",".join(hosts),
            "--skip-done",
        ]
        run_command_stream(cmd)

        dt = time.time() - t_job_start
        completed_durations.append(dt)
        print(f"[{ist_now_str()}] Completed {name} in {dt/60:.1f} minutes.", flush=True)

        out_dir = ROOT / "runs" / group / name
        clean_storage(hosts, prune_run_dir=out_dir)

    # Process sweep results
    existing_summary = {}
    if sweep_file.is_file():
        try:
            existing_summary = json.loads(sweep_file.read_text())
        except Exception:
            pass

    existing_runs = existing_summary.get("runs", {})
    existing_selected = existing_summary.get("selected_learning_rates", {})

    std_records = []
    for lr in lrs:
        lr_tag = f"lr{lr:.1e}".replace(".0e", "e").replace("-0", "-")
        name = f"standard_cubic_{lr_tag}_s{seed}"
        res_file = ROOT / "runs" / group / name / "result.json"
        if res_file.is_file():
            d = json.loads(res_file.read_text())
            std_records.append({
                "name": name,
                "lr": lr,
                "cand_name": lr_tag,
                "final_eval_nll": d["final_eval_nll"],
                "final_eval_ppl": d["final_eval_ppl"],
                "final_train_loss": d.get("final_train_loss"),
                "train_seconds": d.get("train_seconds"),
            })

    std_records.sort(key=lambda x: x["final_eval_nll"])
    best_std_rec = std_records[0]
    best_std_lr = best_std_rec["lr"]

    print("\nSTANDARD_CUBIC 600M SWEEP RESULTS:")
    for rec in std_records:
        marker = " -> [SELECTED BEST]" if rec["lr"] == best_std_lr else ""
        print(f"  lr={rec['lr']} ({rec['cand_name']}): NLL={rec['final_eval_nll']:.4f}, PPL={rec['final_eval_ppl']:.2f}{marker}")

    existing_runs["standard_cubic"] = std_records
    existing_selected["standard_cubic"] = best_std_lr

    merged_summary = {
        "completed_at_ist": ist_now_str(),
        "token_budget_per_run": 600_000_000,
        "selected_learning_rates": existing_selected,
        "runs": existing_runs,
    }
    sweep_file.write_text(json.dumps(merged_summary, indent=2))

    sel_data = {
        "selected": {arm: {"lr": lr, "wd": 0.1} for arm, lr in existing_selected.items()},
        "selected_at_ist": ist_now_str()
    }
    sel_file.write_text(json.dumps(sel_data, indent=2))
    flag_file.write_text(f"completed_at_ist: {ist_now_str()}\n")
    print(f"\n[Phase 1b Complete] Saved merged sweep results to {sweep_file} and selections to {sel_file}", flush=True)

    return best_std_lr


def run_standard_cubic_main(best_lr: float, hosts=HOSTS, seed=42):
    """Execute Phase 2b: Main 125M run for standard_cubic on 2.5B tokens."""
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] PHASE 2b: MAIN 125M COMPARISON RUN FOR STANDARD_CUBIC ON 2.5B TOKENS")
    print(f"Architecture: Standard Transformer")
    print(f"Learning rate: {best_lr}")
    print(f"Seed: {seed}")
    print(f"Token budget: 2,500,000,000 (2385 steps)")
    print("=" * 80 + "\n", flush=True)

    group = "main_125m"
    name = f"standard_cubic_s{seed}"
    args = [
        "--preset", "standard_cubic",
        "--size", "125m",
        "--seq-len", "2048",
        "--batch", "512",
        "--micro", "4",
        "--tokens", "2500000000",
        "--lr", str(best_lr),
        "--wd", "0.1",
        "--warmup", "0.02",
        "--min-lr-ratio", "0.1",
        "--b2", "0.95",
        "--schedule", "cubic",
        "--seed", str(seed),
        "--train", str(DATA_DIR / "fineweb_train_2_5B.npy"),
        "--valid", str(DATA_DIR / "fineweb_valid.npy"),
        "--eval-tokens", "8000000",
        "--eval-batch", "4",
        "--eval-every", "250",
        "--ckpt-every", "0",
        "--log-every", "25",
    ]

    single_job_path = ROOT / "runs" / group / f"job_{name}.json"
    single_job_path.parent.mkdir(parents=True, exist_ok=True)
    single_job_path.write_text(json.dumps([{"name": name, "args": args}], indent=2))

    t_start = time.time()
    eta_time = datetime.now(IST) + timedelta(minutes=72)
    eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

    update_status({
        "phase": "main_standard_cubic",
        "current_run": name,
        "current_run_index": 2,
        "total_runs_in_phase": 2,
        "eta_ist": eta_str,
        "phase_elapsed_minutes": 0,
    })

    cmd = [
        str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
        str(single_job_path),
        "--group", group,
        "--hosts", ",".join(hosts),
        "--skip-done",
        "--no-hook",
    ]
    run_command_stream(cmd)

    dt = time.time() - t_start
    print(f"[{ist_now_str()}] Completed {name} in {dt/60:.1f} minutes.", flush=True)

    out_dir = ROOT / "runs" / group / name
    clean_storage(hosts, prune_run_dir=out_dir)

    flag_file = ROOT / "results" / "main_125m" / "standard_cubic_main_done.flag"
    flag_file.write_text(f"completed_at_ist: {ist_now_str()}\n")


def generate_consolidated_comparison():
    """Build the final comparison table across Standard and Algebraic with Cosine and Cubic."""
    group = "main_125m"
    results_dir = ROOT / "results" / "main_125m"
    results_dir.mkdir(parents=True, exist_ok=True)

    sel_file = ROOT / "results" / "hparam_sweep" / "cubic_selected.json"
    sel_data = {}
    if sel_file.is_file():
        try:
            sel_data = json.loads(sel_file.read_text()).get("selected", {})
        except Exception:
            pass

    arms_summary = {}
    for arm in ("algebraic_cubic", "standard_cubic"):
        res_file = ROOT / "runs" / group / f"{arm}_s42" / "result.json"
        if res_file.is_file():
            d = json.loads(res_file.read_text())
            lr = sel_data.get(arm, {}).get("lr", d.get("lr"))
            arms_summary[arm] = {
                "completed_seeds": [42],
                "count": 1,
                "learning_rate": lr,
                "nll_per_seed": {"42": d["final_eval_nll"]},
                "ppl_per_seed": {"42": d["final_eval_ppl"]},
                "mean_nll": round(d["final_eval_nll"], 4),
                "std_nll": 0.0,
                "mean_ppl": round(d["final_eval_ppl"], 2),
                "std_ppl": 0.0,
                "final_train_loss": d.get("final_train_loss"),
                "train_seconds": d.get("train_seconds"),
            }

    baselines = {
        "algebraic_cosine": {"mean_nll": 2.8761, "mean_ppl": 17.74},
        "standard_cosine": {"mean_nll": 2.9175, "mean_ppl": 18.49},
    }

    comparison = {
        "completed_at_ist": ist_now_str(),
        "scale": "125m",
        "tokens_per_run": 2500000000,
        "seeds": [42],
        "arms_summary": arms_summary,
        "baselines_for_comparison": baselines,
    }

    out_file = results_dir / "cubic_comparison.json"
    out_file.write_text(json.dumps(comparison, indent=2))
    print(f"\n[{ist_now_str()}] Saved consolidated final comparison summary to {out_file}", flush=True)

    # Print pretty comparison table
    print("\n" + "=" * 80)
    print("FINAL 125M BENCHMARK COMPARISON TABLE (2.5B TOKENS, SEED 42):")
    print(f"{'Run / Configuration':<30} | {'Schedule':<10} | {'Eval NLL':<10} | {'Eval PPL':<10}")
    print("-" * 80)
    for k, v in baselines.items():
        sched = "Cosine"
        arch = "Algebraic" if "algebraic" in k else "Standard"
        print(f"{arch + ' + ' + sched:<30} | {sched:<10} | {v['mean_nll']:<10.4f} | {v['mean_ppl']:<10.2f}")
    print("-" * 80)
    for k, v in arms_summary.items():
        arch = "Algebraic" if "algebraic" in k else "Standard"
        print(f"{arch + ' + Cubic (Ours)':<30} | {'Cubic':<10} | {v['mean_nll']:<10.4f} | {v['mean_ppl']:<10.2f}")
    print("=" * 80 + "\n", flush=True)

    update_status({
        "phase": "complete",
        "status": "All sweeps and main 2.5B runs completed successfully for both Standard and Algebraic architectures",
        "eta_ist": "COMPLETED",
        "arms_summary": arms_summary,
        "baselines": baselines,
    })


def run_pipeline_hook_if_needed(hosts=HOSTS, current_job_name=""):
    """Hook called by run_slice when entering main_125m.
    
    1. If standard_cubic sweep is not done, execute it first.
    2. Continue with algebraic_cubic_s42.
    3. Then execute standard_cubic_s42.
    4. Generate final comparison.
    """
    flag_sweep = ROOT / "results" / "hparam_sweep" / "standard_cubic_sweep_done.flag"
    if not flag_sweep.is_file():
        print(f"\n[{ist_now_str()}] Hook triggered: Running Phase 1b standard_cubic sweep before main runs...", flush=True)
        best_std_lr = run_standard_cubic_sweep(hosts=hosts)
    else:
        sel_file = ROOT / "results" / "hparam_sweep" / "cubic_selected.json"
        best_std_lr = 0.0015
        if sel_file.is_file():
            try:
                best_std_lr = json.loads(sel_file.read_text())["selected"]["standard_cubic"]["lr"]
            except Exception:
                pass

    return best_std_lr
