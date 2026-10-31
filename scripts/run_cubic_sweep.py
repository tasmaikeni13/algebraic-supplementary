#!/usr/bin/env python3
"""Controlled 3-Arm Sweep for Cubic Algebraic Schedule on 125M Models (2.5B Tokens x 2 Seeds).

Evaluates the proposed pure-ALU Cubic Hermite Polynomial Schedule:
    S(u) = (1 - u)^2 * (1 + 2u)
    lr(t) = floor + (peak - floor) * S(u)

Across 3 learning rate candidate arms:
    lr in {1.0e-3, 1.5e-3, 2.5e-3}
Across 2 random seeds:
    seeds: [42, 43]
Total: 6 benchmark runs on 2.5B tokens each.

Strictly storage-aware (checkpoints pruned after metrics extraction).
NO remote pushes to GitHub (local reporting only).
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
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
STATUS_FILE = ROOT / "cubic_status.json"


def ist_now_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def update_status(data: Dict[str, Any]) -> None:
    data["last_updated_ist"] = ist_now_str()
    data["last_updated_timestamp"] = time.time()
    try:
        STATUS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Warning writing {STATUS_FILE}: {e}", flush=True)


def clean_storage(hosts=HOSTS, prune_run_dir: Path = None) -> None:
    """Strict storage preservation across the cluster."""
    if prune_run_dir:
        for h in hosts:
            subprocess.run(
                ["ssh", h, f"rm -rf {shlex.quote(str(prune_run_dir / 'ckpt'))} 2>/dev/null || true"],
                check=False
            )
            subprocess.run(
                ["ssh", h, f"rm -f {shlex.quote(str(prune_run_dir))}/*.npy {shlex.quote(str(prune_run_dir))}/*.ckpt 2>/dev/null || true"],
                check=False
            )


def make_cubic_jobs(lrs=(0.0010, 0.0015, 0.0025), seeds=(42, 43)) -> List[Dict[str, Any]]:
    jobs = []
    for lr in lrs:
        lr_tag = f"lr{lr:.1e}".replace(".0e", "e").replace("-0", "-")
        for seed in seeds:
            name = f"algebraic_cubic_{lr_tag}_s{seed}"
            args = [
                "--preset", "algebraic_cubic",
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
                "lr": lr,
                "lr_tag": lr_tag,
                "seed": seed,
                "args": args,
            })
    return jobs


def run_command_stream(cmd: List[str]) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.0010, 0.0015, 0.0025])
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = p.parse_args()

    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] CONTROLLED 3-ARM SWEEP: PURE-ALU CUBIC ALGEBRAIC SCHEDULE")
    print(f"Candidate LRs: {args.lrs}")
    print(f"Seeds: {args.seeds}")
    print(f"Token Budget: 2.5 Billion tokens per run (2385 steps)")
    print(f"Remote push to GitHub: DISABLED")
    print("=" * 80 + "\n", flush=True)

    group = "cubic_sweep_125m"
    jobs = make_cubic_jobs(lrs=args.lrs, seeds=args.seeds)
    total_jobs = len(jobs)
    print(f"Total benchmark jobs: {total_jobs}", flush=True)

    # Sync code to all workers
    subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *HOSTS], cwd=ROOT, check=True)

    t_phase_start = time.time()
    completed_durations = []

    for i, j in enumerate(jobs, 1):
        name = j["name"]
        t_job_start = time.time()

        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_jobs = total_jobs - (i - 1)
            rem_sec = avg_sec * rem_jobs
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            # Baseline estimate: algebraic 125M takes ~78 minutes per run
            rem_sec = 78 * 60 * (total_jobs - (i - 1))
            eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            eta_str = eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

        print(f"\n[{ist_now_str()}] [{i}/{total_jobs}] Launching: {name} (ETA for all runs: {eta_str})", flush=True)

        update_status({
            "phase": "cubic_sweep",
            "current_run": name,
            "current_run_index": i,
            "total_runs": total_jobs,
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

        # Storage cleanup: remove intermediate checkpoints immediately
        out_dir = ROOT / "runs" / group / name
        clean_storage(HOSTS, prune_run_dir=out_dir)

    # Summarize results
    results_dir = ROOT / "results" / "cubic_sweep_125m"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_records = {}
    for lr in args.lrs:
        lr_tag = f"lr{lr:.1e}".replace(".0e", "e").replace("-0", "-")
        nlls = []
        ppls = []
        for s in args.seeds:
            res_file = ROOT / "runs" / group / f"algebraic_cubic_{lr_tag}_s{s}" / "result.json"
            if res_file.is_file():
                d = json.loads(res_file.read_text())
                nlls.append(d["final_eval_nll"])
                ppls.append(d["final_eval_ppl"])
        mean_nll = sum(nlls) / len(nlls) if nlls else float("nan")
        std_nll = (sum((x - mean_nll) ** 2 for x in nlls) / max(1, len(nlls) - 1)) ** 0.5 if len(nlls) > 1 else 0.0
        mean_ppl = sum(ppls) / len(ppls) if ppls else float("nan")
        std_ppl = (sum((x - mean_ppl) ** 2 for x in ppls) / max(1, len(ppls) - 1)) ** 0.5 if len(ppls) > 1 else 0.0
        summary_records[lr_tag] = {
            "lr": lr,
            "seeds": args.seeds,
            "nll_per_seed": nlls,
            "ppl_per_seed": ppls,
            "mean_nll": round(mean_nll, 4),
            "std_nll": round(std_nll, 4),
            "mean_ppl": round(mean_ppl, 2),
            "std_ppl": round(std_ppl, 2),
        }
        print(f"\nLR={lr} ({lr_tag}) 2-SEED SUMMARY: NLL = {mean_nll:.4f} ± {std_nll:.4f}, PPL = {mean_ppl:.2f} ± {std_ppl:.2f}")

    final_report = {
        "completed_at_ist": ist_now_str(),
        "schedule": "cubic_hermite_algebraic",
        "formula": "S(u) = (1 - u)^2 * (1 + 2u)",
        "scale": "125m",
        "tokens_per_run": 2_500_000_000,
        "seeds": list(args.seeds),
        "results": summary_records,
    }
    (results_dir / "cubic_sweep_summary.json").write_text(json.dumps(final_report, indent=2))
    print(f"\nSaved sweep summary to {results_dir / 'cubic_sweep_summary.json'}")

    update_status({
        "phase": "complete",
        "status": "Cubic sweep completed successfully across 2 seeds on 2.5B tokens",
        "eta_ist": "COMPLETED",
        "total_elapsed_minutes": (time.time() - t_phase_start) / 60,
    })
    print(f"\n[{ist_now_str()}] ALL 6 RUNS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
