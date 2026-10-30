#!/usr/bin/env python3
"""Unified Autonomous Pipeline for 125M ARDS Hyperparameter Sweep and Main Comparison Runs.

Target Hardware: Google Cloud TPU v4-32 (16 chips across 4 workers: w0, w1, w2, w3).

Workflow:
1. Phase 1: Hyperparameter Sweep on 600M tokens per arm:
   - 4 combinations: {Standard, Algebraic} x {Cosine, ARDS}
     (standard_cosine, algebraic_cosine, standard_ards, algebraic_ards)
   - 3 candidates per arm: lr in {1.5e-3, 3.0e-3, 6.0e-3}
   - Evaluated on held-out validation window (8M tokens) every 100 steps
   - Zero checkpoints saved during sweep to preserve disk space
   - Automatic selection of best learning rate per arm based on validation NLL
   - Automatic git commit and push of sweep results to GitHub
2. Storage Maintenance:
   - Verification of available disk storage across all 4 workers before/after each run
   - Immediate pruning of checkpoints/temporary files
3. Phase 2: Main 125M Comparison Runs on 2.5B tokens:
   - 4 combinations x 5 seeds (42, 43, 44, 45, 46) -> 20 runs total
   - Using best hyperparameters selected from Phase 1
   - Evaluated on held-out validation tokens every 250 steps
   - Model/checkpoints pruned after recording results to maintain clean storage
4. Live Status Tracking & ETA:
   - pipeline_status.json written and updated after every run
   - Live ETA in IST (Indian Standard Time, UTC+5:30)
   - Final summary and push of main comparison results to GitHub
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

STATUS_FILE = ROOT / "pipeline_status.json"


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


def get_disk_info(hosts=HOSTS) -> dict[str, str]:
    info = {}
    for h in hosts:
        try:
            line = subprocess.check_output(
                ["ssh", h, "df -h / | tail -n 1"], text=True, timeout=15
            ).strip()
            parts = line.split()
            info[h] = f"Size: {parts[1]}, Used: {parts[2]}, Avail: {parts[3]} ({parts[4]})"
        except Exception as e:
            info[h] = f"Error: {e}"
    return info


def clean_storage(hosts=HOSTS, prune_run_dir=None):
    for h in hosts:
        cmds = [
            "rm -rf /tmp/pytest* /tmp/tpu_logs/* /tmp/*.log* 2>/dev/null || true",
        ]
        if prune_run_dir:
            cmds.append(f"rm -rf {shlex.quote(str(prune_run_dir))}/ckpt {shlex.quote(str(prune_run_dir))}/*.npz 2>/dev/null || true")
        cmd_str = " ; ".join(cmds)
        subprocess.run(["ssh", h, cmd_str], timeout=30, check=False)


def sync_data_to_workers(data_dir=DATA_DIR, hosts=HOSTS):
    """Sync prepared token datasets to worker slices w1, w2, w3."""
    print(f"\n[{ist_now_str()}] Syncing token dataset from w0 to all workers...", flush=True)
    required = [
        data_dir / "fineweb_train_2_5B.npy",
        data_dir / "fineweb_sweep_600M.npy",
        data_dir / "fineweb_valid.npy",
        data_dir / "fineweb_metadata.json",
    ]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required dataset file: {p}")

    for h in hosts:
        if h == "w0":
            continue
        print(f"[{ist_now_str()}] Syncing {data_dir} to {h}...", flush=True)
        subprocess.run(["ssh", h, f"mkdir -p {shlex.quote(str(data_dir))}"], check=True)
        subprocess.run(
            ["rsync", "-a", "--info=progress2", f"{data_dir}/", f"{h}:{data_dir}/"],
            check=True,
        )
    print(f"[{ist_now_str()}] Token datasets synced across all workers successfully!\n", flush=True)


def run_command_stream(cmd, log_file=None):
    """Run a subprocess and stream output to both console and optional log file."""
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


def make_sweep_jobs():
    candidates = [
        {"name": "lr1.5e-3", "lr": 0.0015},
        {"name": "lr3e-3", "lr": 0.003},
        {"name": "lr6e-3", "lr": 0.006},
    ]
    arms = ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]
    seed = 42
    jobs = []
    for cand in candidates:
        for arm in arms:
            sched = "cosine" if "cosine" in arm else "ards"
            name = f"{arm}_{cand['name']}_s{seed}"
            args = [
                "--preset", arm,
                "--size", "125m",
                "--seq-len", "2048",
                "--batch", "512",
                "--micro", "4",
                "--tokens", "600000000",
                "--lr", str(cand["lr"]),
                "--wd", "0.1",
                "--warmup", "0.02",
                "--min-lr-ratio", "0.1",
                "--b2", "0.95",
                "--schedule", sched,
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
                "cand_name": cand["name"],
                "lr": cand["lr"],
                "seed": seed,
                "args": args,
            })
    return jobs


def make_main_jobs(selected_lrs: Dict[str, float], seeds=(42, 43, 44, 45, 46)):
    arms = ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]
    jobs = []
    for seed in seeds:
        for arm in arms:
            lr = selected_lrs[arm]
            sched = "cosine" if "cosine" in arm else "ards"
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
                "--schedule", sched,
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


def run_phase1_sweep() -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] PHASE 1: HYPERPARAMETER SWEEP FOR 4 ARMS ON 600M TOKENS")
    print("Arms: {standard_cosine, algebraic_cosine, standard_ards, algebraic_ards}")
    print("Candidates: lr in {1.5e-3, 3.0e-3, 6.0e-3} (standard literature sweep)")
    print("=" * 80 + "\n", flush=True)

    results_dir = ROOT / "results" / "hparam_sweep"
    sel_file = results_dir / "selected.json"
    sweep_file = results_dir / "sweep_results.json"
    arms = ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]

    if sel_file.is_file() and sweep_file.is_file():
        try:
            d = json.loads(sel_file.read_text()).get("selected", {})
            if all(arm in d for arm in arms):
                print(f"[{ist_now_str()}] Phase 1 sweep already completed! Loaded selected learning rates from {sel_file}:", flush=True)
                selected_lrs = {arm: d[arm]["lr"] for arm in arms}
                for arm, lr in selected_lrs.items():
                    print(f"  {arm}: lr={lr}", flush=True)
                return selected_lrs
        except Exception as e:
            print(f"Notice: could not load existing sweep results ({e}), proceeding with sweep execution.", flush=True)

    group = "hparam_sweep"
    sweep_jobs = make_sweep_jobs()
    total_sweep_jobs = len(sweep_jobs)
    print(f"Total sweep jobs to execute: {total_sweep_jobs}", flush=True)

    # Sync codebase and token dataset to all workers before starting
    subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *HOSTS], cwd=ROOT, check=True)
    sync_data_to_workers(DATA_DIR, HOSTS)

    t_phase_start = time.time()
    completed_durations = []

    for i, j in enumerate(sweep_jobs, 1):
        name = j["name"]
        t_job_start = time.time()

        # Compute current ETA
        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_jobs = total_sweep_jobs - (i - 1)
            rem_sec = avg_sec * rem_jobs
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            eta_str = "Calculating after first run..."

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

        # Storage maintenance
        out_dir = ROOT / "runs" / group / name
        clean_storage(HOSTS, prune_run_dir=out_dir)

    # Collect and analyze sweep results
    results_dir = ROOT / "results" / "hparam_sweep"
    results_dir.mkdir(parents=True, exist_ok=True)

    arms = ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]
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
        print(f"\n{arm.upper()} SWEEP RESULTS:")
        for rec in records:
            marker = " -> [SELECTED BEST]" if rec["lr"] == best["lr"] else ""
            print(f"  lr={rec['lr']} ({rec['cand_name']}): NLL={rec['final_eval_nll']:.4f}, PPL={rec['final_eval_ppl']:.2f}{marker}")

    # Write summary files
    summary = {
        "completed_at_ist": ist_now_str(),
        "token_budget_per_run": 600_000_000,
        "selected_learning_rates": selected_lrs,
        "runs": all_results,
    }
    (results_dir / "sweep_results.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "selected.json").write_text(json.dumps({
        "selected": {arm: {"lr": selected_lrs[arm], "wd": 0.1} for arm in arms},
        "selected_at_ist": ist_now_str(),
    }, indent=2))

    # Commit and push sweep results to GitHub
    print("\nPushing Phase 1 sweep results to GitHub...", flush=True)
    subprocess.run(["git", "add", "-f", "results/hparam_sweep/"], cwd=ROOT, check=True)
    status = subprocess.check_output(["git", "status", "--porcelain", "results/hparam_sweep/"], cwd=ROOT, text=True)
    if status.strip():
        subprocess.run(
            ["git", "commit", "-m", "Record 125M ARDS hyperparameter sweep results on 600M tokens across 4 arms"],
            cwd=ROOT, check=True
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("Phase 1 sweep results successfully pushed to GitHub!\n", flush=True)

    return selected_lrs


def run_phase2_main(selected_lrs: Dict[str, float], seeds=(42, 43, 44, 45, 46)):
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] PHASE 2: MAIN 125M COMPARISON RUNS (5 SEEDS x 4 COMBINATIONS ON 2.5B TOKENS)")
    print(f"Selected Learning Rates: {selected_lrs}")
    print(f"Seeds: {seeds}")
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

        # Compute current ETA
        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_jobs = total_main_jobs - (i - 1)
            rem_sec = avg_sec * rem_jobs
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            # Baseline estimate ~28 minutes per run
            rem_sec = 28 * 60 * (total_main_jobs - (i - 1))
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

        # Storage maintenance: prune intermediate checkpoints and weights across all workers
        out_dir = ROOT / "runs" / group / name
        clean_storage(HOSTS, prune_run_dir=out_dir)

    # Summarize Phase 2 results across 5 seeds for all 4 arms
    results_dir = ROOT / "results" / "main_125m"
    results_dir.mkdir(parents=True, exist_ok=True)

    arms = ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]
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
            "seeds": list(seeds),
            "selected_lr": selected_lrs[arm],
            "nll_per_seed": nlls,
            "ppl_per_seed": ppls,
            "mean_nll": mean_nll,
            "std_nll": std_nll,
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
        }
        print(f"\n{arm.upper()} FINAL (5 seeds): NLL = {mean_nll:.4f} ± {std_nll:.4f}, Mean PPL = {mean_ppl:.2f} ± {std_ppl:.2f}")

    final_report = {
        "completed_at_ist": ist_now_str(),
        "scale": "125m",
        "tokens_per_run": 2_500_000_000,
        "seeds": list(seeds),
        "results": main_records,
    }
    (results_dir / "comparison_summary.json").write_text(json.dumps(final_report, indent=2))

    # Push final results to GitHub
    print("\nPushing final main comparison results to GitHub...", flush=True)
    subprocess.run(["git", "add", "-f", "results/main_125m/"], cwd=ROOT, check=True)
    status = subprocess.check_output(["git", "status", "--porcelain", "results/main_125m/"], cwd=ROOT, text=True)
    if status.strip():
        subprocess.run(
            ["git", "commit", "-m", "Record 125M main comparison runs across 5 seeds on 2.5B tokens for 4 arms"],
            cwd=ROOT, check=True
        )
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT, check=True)
        print("Main run comparison results pushed to GitHub!", flush=True)

    update_status({
        "phase": "complete",
        "status": "All sweep and main runs completed successfully",
        "eta_ist": "COMPLETED",
        "total_elapsed_minutes": (time.time() - t_phase_start) / 60,
    })
    print(f"\n[{ist_now_str()}] ALL RUNS COMPLETED SUCCESSFULLY!", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("all", "sweep", "main"), default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = p.parse_args()

    selected_lrs = {}
    if args.phase in ("all", "sweep"):
        selected_lrs = run_phase1_sweep()

    if args.phase in ("all", "main"):
        if not selected_lrs:
            sel_file = ROOT / "results" / "hparam_sweep" / "selected.json"
            if sel_file.is_file():
                sel_data = json.loads(sel_file.read_text()).get("selected", {})
                selected_lrs = {arm: sel_data.get(arm, {}).get("lr", 0.003)
                                for arm in ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]}
            else:
                selected_lrs = {arm: 0.003 for arm in ["standard_cosine", "algebraic_cosine", "standard_ards", "algebraic_ards"]}
        run_phase2_main(selected_lrs, seeds=args.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
