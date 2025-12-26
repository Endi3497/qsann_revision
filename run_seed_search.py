#!/usr/bin/env python3
"""
Search seeds that beat (acc, auroc) thresholds on 4 datasets in parallel GPUs.
Uses run.sh defaults, with overrides matching the 2025-12-24 settings.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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

METRIC_KEYS: Tuple[str, ...] = ("acc", "auroc", "precision", "recall", "f1", "auprc")

# Override run.sh defaults to match the provided experiment settings.
DEFAULT_OVERRIDES: Dict[str, str] = {
    "MODEL_MODULE": "model_revision",
    "DATASET_LABELS": "0 1 2 3 4 5 6 7 8 9",
    "SAMPLES_PER_LABEL": "40",
    "IMAGE_SIZE": "28",
    "PATCH_SIZE": "4",
    "TRAIN_RATIO": "0.8",
    "VAL_RATIO": "0.0",
    "TEST_RATIO": "0.2",
    "NUM_QUBITS": "8",
    "VQC_LAYERS": "2",
    "REUPLOADING": "3",
    "MEASUREMENT": "correlations",
    "BACKEND_DEVICE": "gpu",
    "USE_TORCH_AUTOGRAD": "--use-torch-autograd",
    "QKV_MODE": "separate",
    "QKV_DIM": "64",
    "ATTN_TYPE": "rbf",
    "ATTN_LAYERS": "1",
    "RBF_GAMMA": "1.0",
    "AGG_MODE": "concat",
    "HIDDEN_DIMS": "__none__",
    "DROPOUT": "0.3",
    "EPOCHS": "30",
    "BATCH_SIZE": "64",
    "LR": "0.01",
    "NUM_WORKERS": "4",
    "EARLY_STOP": "",
    "EARLY_STOP_PATIENCE": "10",
    "EARLY_STOP_MIN_DELTA": "0.0",
    "NO_POS_WEIGHT": "",
    "NO_BALANCE_SAMPLER": "--no-balance-sampler",
    "NUM_CLASSES": "10",
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

    return {key: _get(key) for key in METRIC_KEYS}


def beats_threshold(key: str, metrics: Dict[str, float]) -> bool:
    acc_thr, auroc_thr = THRESHOLDS[key]
    return metrics["acc"] > acc_thr and metrics["auroc"] > auroc_thr


def format_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for key in METRIC_KEYS:
        val = metrics.get(key, float("nan"))
        if val != val:
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={val:.4f}")
    return " ".join(parts)


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
        env.update(DEFAULT_OVERRIDES)
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


def iter_seeds(args: argparse.Namespace) -> Iterable[int]:
    if args.seeds:
        return args.seeds
    return range(args.seed_start, args.seed_end + 1, args.seed_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed search for QSANN_revision across 4 datasets/GPUs.")
    parser.add_argument("--run-sh", type=Path, default=Path(__file__).resolve().parent / "run.sh")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=200)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="*", help="Explicit seed list (overrides range).")
    parser.add_argument("--log-dir", type=Path, help="Override LOG_DIR for run.sh.")
    parser.add_argument("--summary", action="store_true", help="Print per-seed pass/fail summaries.")
    parser.add_argument("--keep-going", action="store_true", help="Do not stop when all targets are found.")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        help="Extra run.sh environment overrides as KEY=VALUE (can repeat).",
    )
    args = parser.parse_args()

    run_sh = args.run_sh.resolve()
    if not run_sh.exists():
        raise FileNotFoundError(f"run.sh not found at {run_sh}")

    log_dir = args.log_dir or Path(os.environ.get("LOG_DIR", "results/logs"))
    if not log_dir.is_absolute():
        log_dir = run_sh.parent / log_dir

    extra_overrides: Dict[str, str] = {}
    for item in args.sets:
        if "=" not in item:
            raise ValueError(f"--set requires KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        extra_overrides[key] = val

    found_per_dataset: Dict[str, Dict[str, float]] = {}
    found_seed_per_dataset: Dict[str, int] = {}
    passing_records: Dict[str, List[Tuple[int, Dict[str, float]]]] = {key: [] for key in THRESHOLDS.keys()}
    passing_seeds_set: Dict[str, set[int]] = {key: set() for key in THRESHOLDS.keys()}
    found_global_seed: Optional[int] = None
    found_global_metrics: Optional[Dict[str, Dict[str, float]]] = None

    seeds = list(iter_seeds(args))
    print(f"[info] Searching {len(seeds)} seeds: {seeds[0]}..{seeds[-1]}")

    for seed in seeds:
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_tag = f"seed{seed}-{ts}"
        overrides = {"SEED": str(seed), **extra_overrides}
        results = run_batch(run_sh, log_dir, run_tag, overrides)

        for key, metrics in results.items():
            if beats_threshold(key, metrics):
                if seed not in passing_seeds_set[key]:
                    passing_records[key].append((seed, metrics))
                    passing_seeds_set[key].add(seed)
                if key not in found_per_dataset:
                    found_per_dataset[key] = metrics
                    found_seed_per_dataset[key] = seed
                    print(
                        f"[found] {key} seed={seed} {format_metrics(metrics)}",
                        flush=True,
                    )

        all_pass = all(
            key in results and beats_threshold(key, results[key]) for key in THRESHOLDS.keys()
        )
        if all_pass and found_global_seed is None:
            found_global_seed = seed
            found_global_metrics = results
            print(f"[found] GLOBAL seed={seed} passed all thresholds.", flush=True)

        if args.summary:
            print(f"=== Seed {seed} results ===")
            for key in sorted(THRESHOLDS.keys()):
                metrics = results.get(key, {"acc": float("nan"), "auroc": float("nan")})
                status = "PASS" if beats_threshold(key, metrics) else "FAIL"
                print(
                    f"{key}: acc={metrics['acc']:.4f} auroc={metrics['auroc']:.4f} -> {status}"
                )

        if all_pass and not args.keep_going:
            print("[info] Global seed found; stopping early.")
            break

    print("=== Summary ===")
    for key in sorted(THRESHOLDS.keys()):
        seed = found_seed_per_dataset.get(key)
        metrics = found_per_dataset.get(key)
        if seed is None or metrics is None:
            print(f"{key}: not found")
        else:
            print(f"{key}: seed={seed} {format_metrics(metrics)}")
    if found_global_seed is None:
        print("GLOBAL: not found")
    else:
        print(f"GLOBAL: seed={found_global_seed}")
        if found_global_metrics is not None:
            for key in sorted(THRESHOLDS.keys()):
                metrics = found_global_metrics.get(key, {"acc": float("nan"), "auroc": float("nan")})
                print(f"  {key}: {format_metrics(metrics)}")
    print("=== Passing seeds per dataset ===")
    for key in sorted(THRESHOLDS.keys()):
        records = passing_records.get(key, [])
        if not records:
            print(f"{key}: none")
        else:
            for seed, metrics in records:
                print(f"{key}: seed={seed} {format_metrics(metrics)}")


if __name__ == "__main__":
    main()
