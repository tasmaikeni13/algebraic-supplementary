#!/usr/bin/env python3
"""Run resumable data-parallel jobs across a TPU slice from worker zero.

The repository and Python environment must exist at the same paths on every
worker. Dataset arrays also need identical paths. --skip-done validates
completed outputs against --config before skipping a job.
"""

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

ALGEBRAIC_PYTHON = Path.home() / "venv" / "bin" / "python"
if not ALGEBRAIC_PYTHON.exists():
    ALGEBRAIC_PYTHON = Path(sys.executable)


def validate_completed_job(run, job):
    """Require the requested recipe, full step budget, and finite measurements."""
    from dataclasses import asdict
    from algebraic.config import build
    from algebraic.train import get_args, parse_overrides
    from algebraic.benchmarking import load_jsonl
    args = get_args([*job["args"], "--distributed", "--out", str(run)])
    expected_args = vars(args)
    required = ["meta.json", "result.json", "log.jsonl"] + (["params.npz"] if args.save_params else [])
    if any(not (run / name).is_file() for name in required):
        raise ValueError(f"incomplete run: {run}")
    meta = json.loads((run / "meta.json").read_text())
    result = json.loads((run / "result.json").read_text())
    ignore = {"out", "resume", "log_every", "ckpt_every", "keep_ckpts"}
    if any(meta["args"].get(k) != v for k, v in expected_args.items() if k not in ignore):
        raise ValueError(f"completed run used a different recipe: {run}")
    config = asdict(build(args.preset, args.size, **{**parse_overrides(args.set), "seq_len": args.seq_len}))
    steps = args.steps or math.ceil(args.tokens / (args.batch * args.seq_len))
    losses = [row for row in load_jsonl(run / "log.jsonl") if "loss" in row]
    if (meta["config"] != config or result["config"] != config or not losses
            or losses[-1]["step"] != steps or not math.isfinite(losses[-1]["loss"])
            or meta["total_steps"] != steps or result.get("completed_steps") != steps
            or not math.isfinite(result["final_eval_nll"])):
        raise ValueError(f"incomplete or incompatible training result: {run}")
    return result


def remote_command(job, group, python=ALGEBRAIC_PYTHON):
    if not group or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in group):
        raise ValueError("group must be a simple directory name")
    name = job["name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+-" for c in name):
        raise ValueError("job name must be a simple directory name")
    out = ROOT / "runs" / group / name
    command = [str(python), "-u", "-m", "algebraic.train", "--distributed", *job["args"], "--out", str(out)]
    return (f"mkdir -p {shlex.quote(str(out.parent))} && cd {shlex.quote(str(ROOT))} && "
            f"JAX_PLATFORMS=tpu JAX_ENABLE_X64=0 PYTHONPATH={shlex.quote(str(ROOT / 'src'))}:{shlex.quote(str(ROOT))} "
            f"{shlex.join(command)} > {shlex.quote(str(out))}.\"$(hostname)\".log 2>&1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jobs", type=Path)
    p.add_argument("--group", required=True)
    p.add_argument("--hosts", default="w0,w1,w2,w3")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--skip-done", action="store_true")
    a = p.parse_args()
    hosts = a.hosts.split(",")
    jobs = json.loads(a.jobs.read_text())
    if not hosts or len(hosts) != len(set(hosts)) or any(not h or h.startswith('-') for h in hosts):
        p.error("hosts must be distinct SSH host names")
    names = [j["name"] for j in jobs]
    if len(names) != len(set(names)):
        p.error("job names must be unique")
    commands = [remote_command(job, a.group) for job in jobs]
    if not a.no_sync:
        subprocess.run(["bash", str(ROOT / "scripts/setup_workers.sh"), *hosts], cwd=ROOT, check=True)
    root = ROOT / "runs" / a.group
    root.mkdir(parents=True, exist_ok=True)
    for job, command in zip(jobs, commands):
        name = job["name"]
        if a.skip_done and (root / name / "result.json").exists():
            try:
                validate_completed_job(root / name, job)
                print(f"Reuse complete training run {name}", flush=True)
                continue
            except Exception as e:
                print(f"Validation failed for {name} ({e}); re-running...", flush=True)
        t0 = time.monotonic()
        processes = [subprocess.Popen(["ssh", h, command]) for h in hosts]
        try:
            while any(pr.poll() is None for pr in processes):
                failed = [pr.returncode for pr in processes if pr.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f"{name}: a worker failed with exit codes {failed}; see runs/{a.group}/*.log")
                time.sleep(1)
            codes = [pr.wait() for pr in processes]
            if any(codes):
                raise RuntimeError(f"{name}: worker exit codes {codes}")
        finally:
            for pr in processes:
                if pr.poll() is None:
                    pr.terminate()
            for pr in processes:
                pr.wait()

        # Collect outputs from the worker that wrote result.json (e.g. w2)
        out_dir = root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "result.json").is_file():
            for h in hosts:
                if h == "w0":
                    continue
                check_res = subprocess.run(
                    ["ssh", h, f"test -f {shlex.quote(str(out_dir / 'result.json'))}"],
                    check=False
                )
                if check_res.returncode == 0:
                    subprocess.run(
                        ["rsync", "-a", f"{h}:{shlex.quote(str(out_dir))}/", f"{shlex.quote(str(out_dir))}/"],
                        check=True
                    )
                    break

        # Prune intermediate checkpoints across all workers to strictly preserve disk space
        for h in hosts:
            subprocess.run(
                ["ssh", h, f"rm -rf {shlex.quote(str(out_dir / 'ckpt'))} 2>/dev/null || true"],
                check=False
            )

        print(f"{name}: completed in {(time.monotonic()-t0)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
