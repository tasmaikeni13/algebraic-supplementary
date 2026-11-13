#!/usr/bin/env python3
"""Unified Autonomous Pipeline for Evaluating All 14 Hybrid Transformer Architectures.

Executes a complete factorial ablation of all 14 Hybrid Transformer architectures
(H01 through H14) at 125M parameter scale on 2.5B tokens of FineWeb-Edu each,
using AdamW with Cosine decay (seed 42) for scientific fairness and parity.

Target Hardware: Google Cloud TPU v4-32 (16 chips across 4 workers: w0, w1, w2, w3).

Key Guarantees:
1. Strict storage safety: --ckpt-every 0 ensures zero intermediate checkpoints saved.
2. Remote host cleanup: cleans scratch/tmp files across w0, w1, w2, w3 between runs.
3. Live status tracking in hybrid_status.json with progress and IST timestamps (UTC+5:30).
4. Continuous compilation of results/hybrids/hybrid_comparison.json and README.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hybrid.registry import ORDERED_HYBRID_NAMES, HYBRID_METADATA

DATA_DIR = Path.home() / "data" / "fw25"
HOSTS = ("w0", "w1", "w2", "w3")
IST = timezone(timedelta(hours=5, minutes=30))
ALGEBRAIC_PYTHON = Path.home() / "venv" / "bin" / "python"
if not ALGEBRAIC_PYTHON.exists():
    ALGEBRAIC_PYTHON = Path(sys.executable)

STATUS_FILE = ROOT / "hybrid_status.json"
RESULTS_DIR = ROOT / "results" / "hybrids"

BASELINE_RESULTS = {
    "standard_cosine": {
        "nll": 2.9175,
        "ppl": 18.49,
        "nll_seed42": 2.9136,
        "ppl_seed42": 18.42,
        "components": {"pos": "rope", "ffn": "swiglu", "attn": "softmax", "head": "ce"},
        "params": 110_120_448,
        "schedule": "cosine",
    },
    "algebraic_cosine": {
        "nll": 2.8761,
        "ppl": 17.74,
        "nll_seed42": 2.8757,
        "ppl_seed42": 17.74,
        "components": {"pos": "cayley_geo", "ffn": "alu_glu", "attn": "asinh", "head": "alg_ce"},
        "params": 110_122_128,
        "schedule": "cosine",
    },
    "standard_cubic": {
        "nll": 2.9142,
        "ppl": 18.43,
        "components": {"pos": "rope", "ffn": "swiglu", "attn": "softmax", "head": "ce"},
        "params": 110_120_448,
        "schedule": "cubic",
    },
    "algebraic_cubic": {
        "nll": 2.8614,
        "ppl": 17.49,
        "components": {"pos": "cayley_geo", "ffn": "alu_glu", "attn": "asinh", "head": "alg_ce"},
        "params": 110_122_128,
        "schedule": "cubic",
    },
}


def ist_now_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def update_status_file(data: Dict[str, Any]) -> None:
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
            cmds.append(
                f"rm -rf {shlex.quote(str(prune_run_dir))}/ckpt "
                f"{shlex.quote(str(prune_run_dir))}/*.npz "
                f"{shlex.quote(str(prune_run_dir))}/*.npy 2>/dev/null || true"
            )
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


class RunMonitorThread(threading.Thread):
    """Monitors runs/<group>/<name>/log.jsonl and periodically updates status."""

    def __init__(self, run_dir: Path, status_payload: Dict[str, Any], interval: float = 10.0):
        super().__init__(daemon=True)
        self.run_dir = run_dir
        self.payload = status_payload
        self.interval = interval
        self.stop_event = threading.Event()

    def run(self):
        log_jsonl = self.run_dir / "log.jsonl"
        while not self.stop_event.is_set():
            if log_jsonl.is_file():
                try:
                    with log_jsonl.open("r") as f:
                        lines = f.readlines()
                    if lines:
                        last = json.loads(lines[-1])
                        step = last.get("step")
                        total_steps = self.payload.get("total_steps", 2385)
                        if step is not None:
                            self.payload["current_step"] = step
                            self.payload["progress_pct"] = round(step / total_steps * 100, 1)
                            self.payload["current_loss"] = last.get("loss")
                            self.payload["current_gnorm"] = last.get("gnorm")
                            self.payload["current_lr"] = last.get("lr")
                            self.payload["current_tok_per_sec"] = last.get("tok_per_sec", 0)

                            # Estimate remaining time for this run
                            t_start = self.payload.get("run_start_time", time.time())
                            elapsed = time.time() - t_start
                            self.payload["elapsed_minutes"] = round(elapsed / 60, 1)
                            if step > 0:
                                sec_per_step = elapsed / step
                                rem_steps = total_steps - step
                                rem_sec = sec_per_step * rem_steps
                                eta_arm = datetime.now(IST) + timedelta(seconds=rem_sec)
                                self.payload["arm_eta_ist"] = eta_arm.strftime("%Y-%m-%d %H:%M:%S IST")

                            # Write live status
                            update_status_file(self.payload["global_status"])
                except Exception:
                    pass
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()


def compile_hybrid_summary(completed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "hybrid_comparison.json"
    readme_file = RESULTS_DIR / "README.md"

    std_cosine_nll = BASELINE_RESULTS["standard_cosine"]["nll_seed42"]
    alg_cosine_nll = BASELINE_RESULTS["algebraic_cosine"]["nll_seed42"]

    summary_data = {
        "title": "Factorial Ablation of 14 Hybrid Transformer Architectures vs Baselines",
        "scale": "125m",
        "tokens_per_run": 2500000000,
        "optimizer": "AdamW (lr=0.0015, wd=0.1, b2=0.95)",
        "schedule": "cosine",
        "seed": 42,
        "completed_at_ist": ist_now_str(),
        "baselines": BASELINE_RESULTS,
        "completed_count": len(completed_runs),
        "total_hybrids": len(ORDERED_HYBRID_NAMES),
        "hybrids": completed_runs,
    }

    summary_file.write_text(json.dumps(summary_data, indent=2))

    # Generate Markdown Table
    md = [
        "# Factorial Ablation: 14 Hybrid Architectures on 2.5B Tokens (125M Scale)",
        "",
        "Comprehensive empirical comparison of all 16 architecture combinations (2 pure baselines + 14 hybrids) ",
        "trained under identical conditions on FineWeb-Edu using AdamW with Cosine decay (seed 42).",
        "",
        "## Results Table",
        "",
        "| ID | Preset Name | Position | FFN | Attention | Head | Params | Eval NLL | Eval PPL | $\\Delta$ vs Standard | $\\Delta$ vs Algebraic | Throughput | Train Time |",
        "|:---|:---|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **BASE** | `standard_cosine` | RoPE | SwiGLU | Softmax | CE | 110.12M | **{BASELINE_RESULTS['standard_cosine']['nll_seed42']:.4f}** | {BASELINE_RESULTS['standard_cosine']['ppl_seed42']:.2f} | 0.0000 | +{BASELINE_RESULTS['standard_cosine']['nll_seed42'] - alg_cosine_nll:.4f} | ~760k tok/s | ~55 min |",
    ]

    for rec in completed_runs:
        idx = rec.get("index", "")
        name = rec.get("name", "")
        pos = rec.get("pos", "")
        ffn = rec.get("ffn", "")
        attn = rec.get("attn", "")
        head = rec.get("head", "")
        params = f"{rec.get('n_params', 0) / 1e6:.2f}M"
        nll = rec.get("final_eval_nll", float("nan"))
        ppl = rec.get("final_eval_ppl", float("nan"))
        d_std = nll - std_cosine_nll if math.isfinite(nll) else float("nan")
        d_alg = nll - alg_cosine_nll if math.isfinite(nll) else float("nan")
        tok_s = f"{rec.get('mean_tok_per_s', 0):,.0f} tok/s"
        t_min = f"{rec.get('train_seconds', 0) / 60:.1f} min"
        d_std_str = f"{d_std:+.4f}" if math.isfinite(d_std) else "N/A"
        d_alg_str = f"{d_alg:+.4f}" if math.isfinite(d_alg) else "N/A"

        md.append(
            f"| **{idx}** | `{name}` | {pos} | {ffn} | {attn} | {head} | {params} | **{nll:.4f}** | {ppl:.2f} | {d_std_str} | {d_alg_str} | {tok_s} | {t_min} |"
        )

    md.extend([
        f"| **BASE** | `algebraic_cosine` | Cayley | ALU-GLU | Asinh | Alg-CE | 110.12M | **{BASELINE_RESULTS['algebraic_cosine']['nll_seed42']:.4f}** | {BASELINE_RESULTS['algebraic_cosine']['ppl_seed42']:.2f} | {alg_cosine_nll - std_cosine_nll:.4f} | 0.0000 | ~530k tok/s | ~78 min |",
        "",
        "## Component Key",
        "- **Position**: `RoPE` (Standard transcendental) vs `Cayley` (Pure algebraic $SO(n)$ rational rotation)",
        "- **FFN**: `SwiGLU` (Standard transcendental $\\sigma(x)$) vs `ALU-GLU` (Pure algebraic rational curve $x / \\sqrt{1 + x^2}$)",
        "- **Attention**: `Softmax` (Standard transcendental $\\exp(x)$) vs `Asinh` (Pure algebraic algebraic power attention $\\rho(s/8)^{16}$)",
        "- **Head**: `CE` (Standard cross-entropy) vs `Alg-CE` (Pure algebraic gradient of rational likelihood $\\rho(z/2)^8$)",
        "",
        f"*Report auto-generated at {ist_now_str()}*",
    ])

    readme_file.write_text("\n".join(md) + "\n")
    return summary_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrids", nargs="*", default=None, help="Subset of hybrid names or aliases (default: all 14)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--lr", type=float, default=0.0015, help="Learning rate (default: 0.0015)")
    parser.add_argument("--tokens", type=int, default=2500000000, help="Tokens per arm (default: 2,500,000,000)")
    parser.add_argument("--group", default="hybrids", help="Runs group subfolder (default: hybrids)")
    parser.add_argument("--hosts", default="w0,w1,w2,w3", help="TPU worker hosts")
    parser.add_argument("--no-sync", action="store_true", help="Skip setup_workers.sh sync")
    parser.add_argument("--skip-done", action="store_true", default=True, help="Skip already validated complete runs")
    args = parser.parse_args()

    hosts = args.hosts.split(",")
    seed = args.seed
    tokens = args.tokens
    lr = args.lr
    group = args.group

    # Resolve target hybrids
    if args.hybrids:
        from hybrid.registry import HYBRID_ALIASES
        target_names = []
        for h in args.hybrids:
            if h in ORDERED_HYBRID_NAMES:
                target_names.append(h)
            elif h in HYBRID_ALIASES:
                target_names.append(HYBRID_ALIASES[h])
            else:
                raise ValueError(f"Unknown hybrid preset or alias: {h}")
    else:
        target_names = list(ORDERED_HYBRID_NAMES)

    total_hybrids = len(target_names)
    print("=" * 80)
    print(f"[{ist_now_str()}] LAUNCHING FACTORIAL ABLATION: {total_hybrids} HYBRID ARCHITECTURES")
    print(f"Scale: 125M | Tokens: {tokens:,} | Optimizer: AdamW (lr={lr}, wd=0.1, b2=0.95)")
    print(f"Schedule: cosine | Seed: {seed} | TPU Slice: {hosts}")
    print("=" * 80 + "\n", flush=True)

    if not args.no_sync:
        print(f"[{ist_now_str()}] Syncing code repository across TPU workers ({hosts})...", flush=True)
        subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *hosts], cwd=ROOT, check=True)
        print("Workers synced successfully.\n", flush=True)

    # Initial disk cleanup
    clean_storage(hosts)

    t_pipeline_start = time.time()
    completed_durations: List[float] = []
    completed_runs: List[Dict[str, Any]] = []

    # Check already completed runs if skip-done
    root_runs = ROOT / "runs" / group
    root_runs.mkdir(parents=True, exist_ok=True)

    for i, canonical_name in enumerate(target_names, 1):
        meta = HYBRID_METADATA[canonical_name]
        idx = meta.index
        job_name = f"{idx}_{canonical_name}_s{seed}"
        run_dir = root_runs / job_name

        job_args = [
            "--preset", canonical_name,
            "--size", "125m",
            "--seq-len", "2048",
            "--batch", "512",
            "--micro", "4",
            "--tokens", str(tokens),
            "--lr", str(lr),
            "--wd", "0.1",
            "--warmup", "0.02",
            "--min-lr-ratio", "0.1",
            "--b2", "0.95",
            "--schedule", "cosine",
            "--seed", str(seed),
            "--train", str(DATA_DIR / "fineweb_train_2_5B.npy"),
            "--valid", str(DATA_DIR / "fineweb_valid.npy"),
            "--eval-tokens", "8000000",
            "--eval-batch", "4",
            "--eval-every", "250",
            "--ckpt-every", "0",
            "--log-every", "25",
        ]
        job = {"name": job_name, "args": job_args}

        # Check if already complete
        from scripts.run_slice import validate_completed_job
        is_already_done = False
        if args.skip_done and (run_dir / "result.json").is_file():
            try:
                res = validate_completed_job(run_dir, job)
                print(f"[{ist_now_str()}] [{i}/{total_hybrids}] Reusing complete run {job_name} (NLL: {res['final_eval_nll']:.4f})", flush=True)
                rec = {
                    "index": idx,
                    "name": canonical_name,
                    "job_name": job_name,
                    "pos": meta.pos,
                    "ffn": meta.ffn,
                    "attn": meta.attn,
                    "head": meta.head,
                    "n_params": res.get("n_params", meta.params_125m),
                    "final_eval_nll": res["final_eval_nll"],
                    "final_eval_ppl": res["final_eval_ppl"],
                    "final_train_loss": res.get("final_train_loss"),
                    "train_seconds": res.get("train_seconds"),
                    "mean_tok_per_s": res.get("mean_tok_per_s"),
                }
                completed_runs.append(rec)
                completed_durations.append(res.get("train_seconds", 3600.0))
                compile_hybrid_summary(completed_runs)
                is_already_done = True
            except Exception as e:
                print(f"[{ist_now_str()}] Validation failed for {job_name} ({e}); will execute run.", flush=True)

        if is_already_done:
            continue

        # Calculate pipeline ETA
        if completed_durations:
            avg_sec = sum(completed_durations) / len(completed_durations)
            rem_jobs = total_hybrids - (i - 1)
            rem_sec = avg_sec * rem_jobs
            pipeline_eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            pipeline_eta_str = pipeline_eta_time.strftime("%Y-%m-%d %H:%M:%S IST")
        else:
            # Estimate: softmax ~55 min, asinh ~75 min -> avg 65 min
            rem_sec = 65 * 60 * (total_hybrids - (i - 1))
            pipeline_eta_time = datetime.now(IST) + timedelta(seconds=rem_sec)
            pipeline_eta_str = pipeline_eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

        # Estimate individual arm runtime
        expected_arm_sec = 75 * 60 if meta.attn == "asinh" else 55 * 60
        arm_eta_time = datetime.now(IST) + timedelta(seconds=expected_arm_sec)
        arm_eta_str = arm_eta_time.strftime("%Y-%m-%d %H:%M:%S IST")

        print("\n" + "-" * 80)
        print(f"[{ist_now_str()}] [{i}/{total_hybrids}] STARTING ARM: {idx} ({canonical_name})")
        print(f"Components: Pos={meta.pos}, FFN={meta.ffn}, Attn={meta.attn}, Head={meta.head}")
        print(f"Arm ETA: {arm_eta_str} | Overall Pipeline ETA: {pipeline_eta_str}")
        print("-" * 80, flush=True)

        status_dict = {
            "pipeline_name": "14 Hybrid Transformer Architectures Evaluation",
            "scale": "125m",
            "tokens_per_run": tokens,
            "current_arm_index": i,
            "total_arms": total_hybrids,
            "current_arm": {
                "index": idx,
                "name": canonical_name,
                "job_name": job_name,
                "components": {"pos": meta.pos, "ffn": meta.ffn, "attn": meta.attn, "head": meta.head},
                "params": meta.params_125m,
                "current_step": 0,
                "total_steps": 2385,
                "progress_pct": 0.0,
                "arm_eta_ist": arm_eta_str,
            },
            "pipeline_overall_eta_ist": pipeline_eta_str,
            "completed_arms_count": len(completed_runs),
            "completed_arms": completed_runs,
            "baseline_references": BASELINE_RESULTS,
        }
        update_status_file(status_dict)

        # Write single job spec
        single_job_path = root_runs / f"job_{job_name}.json"
        single_job_path.write_text(json.dumps([job], indent=2))

        # Start live monitor thread
        monitor_payload = {
            "global_status": status_dict,
            "run_start_time": time.time(),
            "total_steps": 2385,
        }
        monitor = RunMonitorThread(run_dir, monitor_payload, interval=10.0)
        monitor.start()

        t_job_start = time.time()
        try:
            cmd = [
                str(ALGEBRAIC_PYTHON), str(ROOT / "scripts/run_slice.py"),
                str(single_job_path),
                "--group", group,
                "--hosts", ",".join(hosts),
                "--skip-done",
            ]
            run_command_stream(cmd)
        finally:
            monitor.stop()

        dt = time.time() - t_job_start
        completed_durations.append(dt)
        print(f"[{ist_now_str()}] Completed {job_name} in {dt/60:.1f} minutes.", flush=True)

        # Validate result
        res_file = run_dir / "result.json"
        if not res_file.is_file():
            raise RuntimeError(f"Missing result.json for {job_name} at {res_file}")

        d = json.loads(res_file.read_text())
        rec = {
            "index": idx,
            "name": canonical_name,
            "job_name": job_name,
            "pos": meta.pos,
            "ffn": meta.ffn,
            "attn": meta.attn,
            "head": meta.head,
            "n_params": d.get("n_params", meta.params_125m),
            "final_eval_nll": d["final_eval_nll"],
            "final_eval_ppl": d["final_eval_ppl"],
            "final_train_loss": d.get("final_train_loss"),
            "train_seconds": d.get("train_seconds", dt),
            "mean_tok_per_s": d.get("mean_tok_per_s"),
        }
        completed_runs.append(rec)

        # Immediately prune scratch and intermediate checkpoints across all hosts
        clean_storage(hosts, prune_run_dir=run_dir)

        # Update summary and markdown report
        compile_hybrid_summary(completed_runs)

        # Update status
        status_dict["completed_arms_count"] = len(completed_runs)
        status_dict["completed_arms"] = completed_runs
        update_status_file(status_dict)

    # All runs completed!
    print("\n" + "=" * 80)
    print(f"[{ist_now_str()}] ALL {total_hybrids} HYBRID ARCHITECTURES COMPLETED SUCCESSFULLY!")
    print(f"Total time elapsed: {(time.time() - t_pipeline_start) / 3600:.2f} hours")
    print(f"Final report saved to: {RESULTS_DIR / 'README.md'}")
    print(f"Final JSON saved to: {RESULTS_DIR / 'hybrid_comparison.json'}")
    print("=" * 80 + "\n", flush=True)

    status_dict["status"] = "completed"
    status_dict["completed_at_ist"] = ist_now_str()
    update_status_file(status_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
