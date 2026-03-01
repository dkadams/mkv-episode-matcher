#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import itertools
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    low_info_min_words: int
    low_info_cue_ratio: float
    support_window_bonus: float
    support_offset_penalty: float


def parse_int_list(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def parse_float_list(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def run_eval(
    executable: str,
    dataset_dir: str,
    overlap: int,
    candidate: Candidate,
    output_path: Path,
) -> dict:
    cmd = [
        executable,
        "evaluate-dataset",
        dataset_dir,
        "--subtitle-overlap-seconds",
        str(overlap),
        "--window-expansion-mode",
        "always",
        "--low-info-filter",
        "--low-info-min-words",
        str(candidate.low_info_min_words),
        "--low-info-cue-ratio",
        str(candidate.low_info_cue_ratio),
        "--support-window-bonus",
        str(candidate.support_window_bonus),
        "--support-offset-penalty",
        str(candidate.support_offset_penalty),
        "--output",
        str(output_path),
        "--show-failures",
        "10000",
    ]
    subprocess.run(cmd, check=True)
    with output_path.open("r", encoding="utf-8") as report_in:
        return json.load(report_in)


def meets_gate(candidate_report: dict, baseline_report: dict) -> bool:
    c = candidate_report["overall"]
    b = baseline_report["overall"]
    return (
        c["accuracy"]["top_1"] >= b["accuracy"]["top_1"]
        and c["accuracy"]["top_3"] > b["accuracy"]["top_3"]
        and c["accuracy"]["top_5"] > b["accuracy"]["top_5"]
        and c["mrr"] > b["mrr"]
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic offline grid-search for evaluate-dataset knobs. "
            "Primary objective: satisfy release gate. Secondary: maximize MRR."
        )
    )
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--executable",
        default=".venv/bin/mkv-match",
        help="mkv-match executable path (default: .venv/bin/mkv-match)",
    )
    parser.add_argument("--baseline-overlap", type=int, default=0)
    parser.add_argument("--candidate-overlap", type=int, default=5)
    parser.add_argument("--low-info-min-words", default="6,8,10")
    parser.add_argument("--low-info-cue-ratio", default="0.2,0.25,0.3")
    parser.add_argument("--support-window-bonus", default="0.008,0.012,0.016")
    parser.add_argument("--support-offset-penalty", default="0.001,0.003,0.005")
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for intermediate output reports",
    )
    args = parser.parse_args()

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="mkv_grid_eval_")).resolve()

    candidates = [
        Candidate(a, b, c, d)
        for a, b, c, d in itertools.product(
            parse_int_list(args.low_info_min_words),
            parse_float_list(args.low_info_cue_ratio),
            parse_float_list(args.support_window_bonus),
            parse_float_list(args.support_offset_penalty),
        )
    ]
    candidates.sort(
        key=lambda c: (
            c.low_info_min_words,
            c.low_info_cue_ratio,
            c.support_window_bonus,
            c.support_offset_penalty,
        )
    )

    # Baseline uses the first candidate values for low-info/support knobs to keep comparison deterministic.
    baseline_candidate = candidates[0]
    baseline_report = run_eval(
        args.executable,
        args.dataset_dir,
        args.baseline_overlap,
        baseline_candidate,
        work_dir / f"baseline_o{args.baseline_overlap}.json",
    )

    print(
        "Baseline:",
        f"top1={baseline_report['overall']['accuracy']['top_1']:.4f}",
        f"top3={baseline_report['overall']['accuracy']['top_3']:.4f}",
        f"top5={baseline_report['overall']['accuracy']['top_5']:.4f}",
        f"mrr={baseline_report['overall']['mrr']:.4f}",
    )

    best_report = None
    best_candidate = None
    best_gate = False

    for candidate in candidates:
        label = (
            f"mw{candidate.low_info_min_words}"
            f"_cr{candidate.low_info_cue_ratio}"
            f"_sb{candidate.support_window_bonus}"
            f"_op{candidate.support_offset_penalty}"
        )
        report = run_eval(
            args.executable,
            args.dataset_dir,
            args.candidate_overlap,
            candidate,
            work_dir / f"candidate_o{args.candidate_overlap}_{label}.json",
        )
        gate = meets_gate(report, baseline_report)
        metrics = report["overall"]
        print(
            label,
            f"gate={'PASS' if gate else 'FAIL'}",
            f"top1={metrics['accuracy']['top_1']:.4f}",
            f"top3={metrics['accuracy']['top_3']:.4f}",
            f"top5={metrics['accuracy']['top_5']:.4f}",
            f"mrr={metrics['mrr']:.4f}",
        )

        if best_report is None:
            best_report = report
            best_candidate = candidate
            best_gate = gate
            continue

        current_key = (
            1 if gate else 0,
            metrics["mrr"],
            metrics["accuracy"]["top_1"],
        )
        best_metrics = best_report["overall"]
        best_key = (
            1 if best_gate else 0,
            best_metrics["mrr"],
            best_metrics["accuracy"]["top_1"],
        )
        if current_key > best_key:
            best_report = report
            best_candidate = candidate
            best_gate = gate

    print("\nBest candidate:")
    print(best_candidate)
    print(
        f"gate={'PASS' if best_gate else 'FAIL'}",
        f"top1={best_report['overall']['accuracy']['top_1']:.4f}",
        f"top3={best_report['overall']['accuracy']['top_3']:.4f}",
        f"top5={best_report['overall']['accuracy']['top_5']:.4f}",
        f"mrr={best_report['overall']['mrr']:.4f}",
    )
    print(f"Reports written under: {work_dir}")


if __name__ == "__main__":
    main()
