#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def run_eval(executable: str, dataset_dir: str, overlap: int, output_path: Path):
    cmd = [
        executable,
        "evaluate-dataset",
        dataset_dir,
        "--subtitle-overlap-seconds",
        str(overlap),
        "--output",
        str(output_path),
        "--show-failures",
        "10000",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with output_path.open("r", encoding="utf-8") as report_in:
        return json.load(report_in)


def confusion_map(report: dict) -> dict[tuple[str, str], int]:
    confusions = report.get("overall", {}).get("top1_confusions", [])
    return {
        (row["expected"], row["predicted"]): int(row["count"])
        for row in confusions
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run overlap A/B evaluate-dataset and print metric/confusion deltas.",
    )
    parser.add_argument("dataset_dir", help="Dataset directory")
    parser.add_argument(
        "--executable",
        default=".venv/bin/mkv-match",
        help="mkv-match executable path (default: .venv/bin/mkv-match)",
    )
    parser.add_argument("--baseline-overlap", type=int, default=0)
    parser.add_argument("--candidate-overlap", type=int, default=5)
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for temporary report outputs",
    )
    args = parser.parse_args()

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="mkv_ab_eval_")).resolve()

    baseline_path = work_dir / f"baseline_o{args.baseline_overlap}.json"
    candidate_path = work_dir / f"candidate_o{args.candidate_overlap}.json"

    baseline = run_eval(args.executable, args.dataset_dir, args.baseline_overlap, baseline_path)
    candidate = run_eval(args.executable, args.dataset_dir, args.candidate_overlap, candidate_path)

    keys = ["top_1", "top_3", "top_5"]
    print("\nMetric Deltas (candidate - baseline)")
    for key in keys:
        b = float(baseline["overall"]["accuracy"][key])
        c = float(candidate["overall"]["accuracy"][key])
        print(f"{key}: {b:.4f} -> {c:.4f} (delta {c - b:+.4f})")
    b_mrr = float(baseline["overall"]["mrr"])
    c_mrr = float(candidate["overall"]["mrr"])
    print(f"mrr: {b_mrr:.4f} -> {c_mrr:.4f} (delta {c_mrr - b_mrr:+.4f})")

    b_filter = float(baseline["overall"].get("filtered_low_info_ratio", 0.0))
    c_filter = float(candidate["overall"].get("filtered_low_info_ratio", 0.0))
    print(f"filtered_low_info_ratio: {b_filter:.4f} -> {c_filter:.4f} (delta {c_filter - b_filter:+.4f})")

    conf_b = confusion_map(baseline)
    conf_c = confusion_map(candidate)
    deltas = []
    all_pairs = set(conf_b.keys()) | set(conf_c.keys())
    for pair in all_pairs:
        delta = conf_c.get(pair, 0) - conf_b.get(pair, 0)
        if delta:
            deltas.append((abs(delta), delta, pair))
    deltas.sort(reverse=True)

    print("\nTop confusion deltas")
    for _, delta, (expected, predicted) in deltas[:20]:
        print(
            f"{expected} -> {predicted}: "
            f"{conf_b.get((expected, predicted), 0)} -> {conf_c.get((expected, predicted), 0)} "
            f"(delta {delta:+d})"
        )

    print(f"\nReports written to: {baseline_path} and {candidate_path}")


if __name__ == "__main__":
    main()
