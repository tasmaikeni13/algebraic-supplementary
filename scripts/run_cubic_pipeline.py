#!/usr/bin/env python3
"""Unified Autonomous Pipeline for 125M Cubic Hyperparameter Sweep and Main Comparison Runs.

Target Hardware: Google Cloud TPU v4-32 (16 chips across 4 workers: w0, w1, w2, w3).

Workflow:
1. Phase 1: Hyperparameter Sweep on 600M tokens per arm:
   - Arms: algebraic_cubic (and standard_cubic if specified)
   - Candidates: lr in {1.0e-3, 1.5e-3, 2.5e-3}
   - Evaluated on held-out validation window (8M tokens) every 100 steps
   - Zero checkpoints saved during sweep to preserve disk space (--ckpt-every 0)
   - Automatic selection of best learning rate based on validation NLL
2. Phase 2: Main 125M Comparison Run on 2.5B tokens for 1 Seed (seed 42):
   - Using best hyperparameter selected from Phase 1
   - Evaluated on held-out validation tokens every 250 steps
   - Zero checkpoints saved to maintain clean storage (--ckpt-every 0)
3. Live Status Tracking & ETA:
   - cubic_status.json updated after and during runs with live ETA in IST (UTC+5:30)
   - Final summary comparing against cosine (2.8761) baseline
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
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


def make_sweep_jobs(arms=("algebraic_cubic",), lrs=(0.0010, 0.0015, 0.0025), seed=42):
    jobs = []
    for lr in lrs:
        lr_tag = f"lr{lr:.1e}".replace(".0e", "e").replace("-0", "-")
        for arm in arms:
            name = f"{arm}_{lr_tag}_s{seed}"
            args = [
                "--preset", arm,
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
            jobs.append({
                "name": name,
                "arm": arm,
                "cand_name": lr_tag,
                "lr": lr,
                "seed": seed,
                "args": args,
            })
    return jobs


def make_main_jobs(selected_lrs: Dict[str, float], seeds=(42,)):
    jobs = []
    for seed in seeds:
        for arm, lr in selected_lrs.items():
            name = f"{arm}_s{seed}"
            args = [
                "--preset", arm,
                "--size", "125m",
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", "4",
                "--tokens", "2500000000",
                "--lr", str(lr),
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
            jobs.append({
                "name": name,
                "arm": arm,
                "lr": lr,
                "seed": seed,
                "args": args,
            })
    return jobs


def run_phase1_sweep(arms=("algebraic_cubic",), lrs=(0.0010, 0.0015, 0.0025), seed=42) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] PHASE 1: HYPERPARAMETER SWEEP FOR CUBIC ON 600M TOKENS")
    print(f"Arms: {list(arms)}")
    print(f"Candidates: lr in {list(lrs)}")
    print(f"Token budget per run: 600,000,000 (572 steps)")
    print("=" * 80 + "\n", flush=True)

    results_dir = ROOT / "results" / "hparam_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)
    sel_file = results_dir / "cubic_selected.json"
    sweep_file = results_dir / "cubic_sweep_results.json"

    group = "hparam_sweep"
    sweep_jobs = make_sweep_jobs(arms=arms, lrs=lrs, seed=seed)
    total_sweep_jobs = len(sweep_jobs)
    print(f"Total sweep jobs to execute: {total_sweep_jobs}", flush=True)

    subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *HOSTS], cwd=ROOT, check=True)

    t_phase_start = time.time()
    completed_durations = []

    for i, j in enumerate(sweep_jobs, 1):
        name = j["name"]
        t_job_start = time.time()

        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_jobs = total_sweep_jobs - (i - 1)
            rem_sec = avg_sec * rem_jobs
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            # Baseline estimate ~18 minutes per 600M run
            rem_sec = 18 * 60 * (total_sweep_jobs - (i - 1))
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

        print(f"\n[{ist_now_str()}] [{i}/{total_sweep_jobs}] Launching sweep run: {name} (ETA for sweep: {eta_str})", flush=True)

        update_status({
            "phase": "sweep",
            "current_run": name,
            "current_run_index": i,
            "total_runs_in_phase": total_sweep_jobs,
            "eta_ist": eta_str,
            "phase_elapsed_minutes": (time.time() - t_phase_start) / 60,
        })

        single_job_path = ROOT / "runs" / group / f"job_{name}.json"
        single_job_path.parent.mkdir(parents=True, exist_ok=True)
        single_job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]}], indent=2))

        cmd = [
            str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
            str(single_job_path),
            "--group", group,
            "--hosts", ",".join(HOSTS),
            "--skip-done",
        ]
        run_command_stream(cmd)
        dt = time.time() - t_job_start
        completed_durations.append(dt)
        print(f"[{ist_now_str()}] Completed {name} in {dt/60:.1f} minutes.", flush=True)

        out_dir = ROOT / "runs" / group / name
        clean_storage(HOSTS, prune_run_dir=out_dir)

    all_results = {}
    selected_lrs = {}

    for arm in arms:
        arm_runs = [j for j in sweep_jobs if j["arm"] == arm]
        records = []
        for r in arm_runs:
            name = r["name"]
            res_file = ROOT / "runs" / group / name / "result.json"
            if res_file.is_file():
                d = json.loads(res_file.read_text())
                records.append({
                    "name": name,
                    "lr": r["lr"],
                    "cand_name": r["cand_name"],
                    "final_eval_nll": d["final_eval_nll"],
                    "final_eval_ppl": d["final_eval_ppl"],
                    "final_train_loss": d.get("final_train_loss"),
                    "train_seconds": d.get("train_seconds"),
                })
        records.sort(key=lambda x: x["final_eval_nll"])
        all_results[arm] = records
        best = records[0]
        selected_lrs[arm] = best["lr"]
        print(f"\n{arm.upper()} 600M SWEEP RESULTS:")
        for rec in records:
            marker = " -> [SELECTED BEST]" if rec["lr"] == best["lr"] else ""
            print(f"  lr={rec['lr']} ({rec['cand_name']}): NLL={rec['final_eval_nll']:.4f}, PPL={rec['final_eval_ppl']:.2f}{marker}")

    summary = {
        "completed_at_ist": ist_now_str(),
        "token_budget_per_run": 600_000_000,
        "selected_learning_rates": selected_lrs,
        "runs": all_results,
    }
    sweep_file.write_text(json.dumps(summary, indent=2))
    sel_file.write_text(json.dumps({
        "selected": {arm: {"lr": lr, "wd": 0.1} for arm, lr in selected_lrs.items()},
        "selected_at_ist": ist_now_str()
    }, indent=2))

    print(f"\nSaved sweep summary to {sweep_file} and selections to {sel_file}")
    return selected_lrs


def run_phase2_main(selected_lrs: Dict[str, float], seeds=(42,)):
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] PHASE 2: MAIN 125M CUBIC COMPARISON RUN ON 2.5B TOKENS")
    print(f"Using selected learning rates: {selected_lrs}")
    print(f"Seeds: {list(seeds)}")
    print(f"Token budget per run: 2,500,000,000 (2385 steps)")
    print("=" * 80 + "\n", flush=True)

    group = "main_125m"
    main_jobs = make_main_jobs(selected_lrs, seeds=seeds)
    total_main_jobs = len(main_jobs)
    print(f"Total main jobs to execute: {total_main_jobs}", flush=True)

    t_phase_start = time.time()
    completed_durations = []

    for i, j in enumerate(main_jobs, 1):
        name = j["name"]
        t_job_start = time.time()

        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_jobs = total_main_jobs - (i - 1)
            rem_sec = avg_sec * rem_jobs
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            rem_sec = 75 * 60 * (total_main_jobs - (i - 1))
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

        print(f"\n[{ist_now_str()}] [{i}/{total_main_jobs}] Launching main run: {name} (ETA for all main runs: {eta_str})", flush=True)

        update_status({
            "phase": "main",
            "current_run": name,
            "current_run_index": i,
            "total_runs_in_phase": total_main_jobs,
            "eta_ist": eta_str,
            "phase_elapsed_minutes": (time.time() - t_phase_start) / 60,
        })

        single_job_path = ROOT / "runs" / group / f"job_{name}.json"
        single_job_path.parent.mkdir(parents=True, exist_ok=True)
        single_job_path.write_text(json.dumps([{"name": j["name"], "args": j["args"]}], indent=2))

        cmd = [
            str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
            str(single_job_path),
            "--group", group,
            "--hosts", ",".join(HOSTS),
            "--skip-done",
        ]
        run_command_stream(cmd)
        dt = time.time() - t_job_start
        completed_durations.append(dt)
        print(f"[{ist_now_str()}] Completed {name} in {dt/60:.1f} minutes.", flush=True)

        out_dir = ROOT / "runs" / group / name
        clean_storage(HOSTS, prune_run_dir=out_dir)

    results_dir = ROOT / "results" / "main_125m"
    results_dir.mkdir(parents=True, exist_ok=True)

    arms = list(selected_lrs.keys())
    main_records = {}

    for arm in arms:
        nlls = []
        ppls = []
        for s in seeds:
            res_file = ROOT / "runs" / group / f"{arm}_s{s}" / "result.json"
            if res_file.is_file():
                d = json.loads(res_file.read_text())
                nlls.append(d["final_eval_nll"])
                ppls.append(d["final_eval_ppl"])
        mean_nll = sum(nlls) / len(nlls) if nlls else float("nan")
        std_nll = (sum((x - mean_nll) ** 2 for x in nlls) / max(1, len(nlls) - 1)) ** 0.5 if len(nlls) > 1 else 0.0
        mean_ppl = sum(ppls) / len(ppls) if ppls else float("nan")
        std_ppl = (sum((x - mean_ppl) ** 2 for x in ppls) / max(1, len(ppls) - 1)) ** 0.5 if len(ppls) > 1 else 0.0

        main_records[arm] = {
            "completed_seeds": list(seeds),
            "count": len(nlls),
            "learning_rate": selected_lrs[arm],
            "nll_per_seed": {str(s): nll for s, nll in zip(seeds, nlls)},
            "ppl_per_seed": {str(s): ppl for s, ppl in zip(seeds, ppls)},
            "mean_nll": round(mean_nll, 4),
            "std_nll": round(std_nll, 4),
            "mean_ppl": round(mean_ppl, 2),
            "std_ppl": round(std_ppl, 2),
        }
        print(f"\n{arm.upper()} MAIN RUN SUMMARY: NLL = {mean_nll:.4f} ± {std_nll:.4f}, PPL = {mean_ppl:.2f} ± {std_ppl:.2f}")

    comparison = {
        "completed_at_ist": ist_now_str(),
        "scale": "125m",
        "tokens_per_run": 2500000000,
        "seeds": list(seeds),
        "arms_summary": main_records,
        "baselines_for_comparison": {
            "algebraic_cosine": {"mean_nll": 2.8761, "mean_ppl": 17.74},
            "standard_cosine": {"mean_nll": 2.9175, "mean_ppl": 18.49},
        }
    }
    out_file = results_dir / "cubic_comparison.json"
    out_file.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved final comparison summary to {out_file}", flush=True)

    update_status({
        "phase": "complete",
        "status": "Cubic sweep (600M) and main run (2.5B) completed successfully",
        "eta_ist": "COMPLETED",
        "total_elapsed_minutes": (time.time() - t_phase_start) / 60,
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["all", "sweep", "main"], default="all")
    p.add_argument("--arms", nargs="+", default=["algebraic_cubic"])
    p.add_argument("--lrs", type=float, nargs="+", default=[0.0010, 0.0015, 0.0020, 0.0030])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = p.parse_args()

    clean_storage(HOSTS)

    if args.phase in ("all", "sweep"):
        selected = run_phase1_sweep(arms=args.arms, lrs=args.lrs, seed=args.seeds[0])
    else:
        sel_file = ROOT / "results" / "hparam_sweep" / "cubic_selected.json"
        if not sel_file.is_file():
            p.error(f"Cannot run main without sweep selections: {sel_file} not found")
        data = json.loads(sel_file.read_text()).get("selected", {})
        selected = {arm: data[arm]["lr"] for arm in args.arms if arm in data}

    if args.phase in ("all", "main"):
        run_phase2_main(selected, seeds=args.seeds)


if __name__ == "__main__":
    main()
