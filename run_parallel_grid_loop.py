#!/usr/bin/env python3
"""
Run one-at-a-time sweeps for VQC_LAYERS and REUPLOADING using run.sh defaults.

For each value, it launches the 4 dataset/device tasks in parallel, waits,
then reports acc/auroc vs thresholds. No early stopping; it completes the sweep.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Spec:
    device: str
    dataset: str
    image_size: int


SPECS: List[Spec] = [
    Spec("cuda:0", "mnist", 8),
    Spec("cuda:1", "mnist", 28),
    Spec("cuda:2", "fmnist", 28),
    Spec("cuda:3", "cifar10", 32),
]

# thresholds: acc, auroc
THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "mnist_8": (0.7250, 0.8852),
    "mnist_28": (0.8375, 0.9700),
    "fmnist_28": (0.7500, 0.9631),
    "cifar10_32": (0.2250, 0.6446),
}


def parse_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    last = None
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "Test -> acc:" in line:
                last = line.strip()
    if not last:
        return None

    def _get(key: str) -> float:
        m = re.search(rf"{key}:\s*([0-9.]+|nan)", last)
        if not m:
            return float("nan")
        val = m.group(1)
        return float("nan") if val == "nan" else float(val)

    return {"acc": _get("acc"), "auroc": _get("auroc")}


def run_batch(
    run_sh: Path,
    log_dir: Path,
    run_tag: str,
    overrides: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    jobs = []
    for spec in SPECS:
        run_name = f"{run_tag}-{spec.dataset}-img{spec.image_size}"
        log_path = log_dir / f"{run_name}.log"
        env = os.environ.copy()
        env.update(overrides)
        env.update(
            {
                "DEVICE": spec.device,
                "DATASET_CHOICE": spec.dataset,
                "IMAGE_SIZE": str(spec.image_size),
                "RUN_NAME": run_name,
            }
        )
        print(
            f"[start] {run_name} device={spec.device} dataset={spec.dataset} "
            f"image_size={spec.image_size} log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(
            ["sh", str(run_sh)],
            cwd=str(run_sh.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        jobs.append((proc, spec, run_name, log_path))

    failed = False
    for proc, spec, run_name, log_path in jobs:
        code = proc.wait()
        print(f"[end] {run_name} exit={code} log={log_path}", flush=True)
        if code != 0:
            failed = True
    if failed:
        print(f"[WARN] One or more runs failed for {run_tag}.")

    results: Dict[str, Dict[str, float]] = {}
    for spec, run_name, log_path in [(s, rn, lp) for _, s, rn, lp in jobs]:
        metrics = parse_metrics(log_path) or {"acc": float("nan"), "auroc": float("nan")}
        key = f"{spec.dataset}_{spec.image_size}"
        results[key] = metrics
    return results


def print_summary(run_tag: str, results: Dict[str, Dict[str, float]]) -> None:
    print(f"=== Results for {run_tag} ===")
    for key, metrics in results.items():
        thr = THRESHOLDS.get(key, (float("nan"), float("nan")))
        acc_ok = metrics["acc"] > thr[0]
        auroc_ok = metrics["auroc"] > thr[1]
        status = "PASS" if acc_ok and auroc_ok else "FAIL"
        print(
            f"{key}: acc={metrics['acc']:.4f} (>{thr[0]:.4f}) "
            f"auroc={metrics['auroc']:.4f} (>{thr[1]:.4f}) -> {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-at-a-time sweep for VQC_LAYERS then REUPLOADING.")
    parser.add_argument("--run-sh", type=Path, default=Path(__file__).resolve().parent / "run.sh")
    parser.add_argument("--vqc-layers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--reuploading", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--summary", action="store_true", help="Print acc/auroc summary after each sweep value.")
    args = parser.parse_args()

    run_sh = args.run_sh.resolve()
    if not run_sh.exists():
        raise FileNotFoundError(f"run.sh not found at {run_sh}")

    log_dir = Path(os.environ.get("LOG_DIR", "results/logs"))
    if not log_dir.is_absolute():
        log_dir = run_sh.parent / log_dir

    # Sweep VQC_LAYERS first
    for v in args.vqc_layers:
        ts = time.strftime("%Y%m%d-%H%M%S")
        tag = f"loop-vqc{v}-{ts}"
        results = run_batch(run_sh, log_dir, tag, {"VQC_LAYERS": str(v)})
        if args.summary:
            print_summary(tag, results)

    # Then sweep REUPLOADING
    for r in args.reuploading:
        ts = time.strftime("%Y%m%d-%H%M%S")
        tag = f"loop-reup{r}-{ts}"
        results = run_batch(run_sh, log_dir, tag, {"REUPLOADING": str(r)})
        if args.summary:
            print_summary(tag, results)


if __name__ == "__main__":
    main()
