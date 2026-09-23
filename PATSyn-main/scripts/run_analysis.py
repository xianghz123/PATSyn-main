#!/usr/bin/env python3

import argparse
from pathlib import Path

from patsyn.experiments import run_study
from patsyn.utils import (
    load_config,
    save_json,
    set_seed,
)


VALID_MODES = [
    "ablation",
    "sensitivity",
    "length",
    "efficiency",
    "scalability",
    "all",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PATSyn analysis experiments."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/porto.yaml",
        help="Dataset configuration file.",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default.yaml",
        help="Default configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=VALID_MODES,
        default="all",
        help="Analysis mode.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Privacy budget for analysis experiments.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed list.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one seed for a quick analysis run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing analysis results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        default_path=args.default_config,
        dataset_path=args.config,
    )

    epsilon = (
        args.epsilon
        if args.epsilon is not None
        else float(config["experiment"]["default_epsilon"])
    )

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    seeds = (
        args.seeds
        if args.seeds is not None
        else list(config["experiment"]["seeds"])
    )

    if args.quick:
        seeds = [seeds[0]]

    set_seed(seeds[0])

    dataset_name = config["dataset"]["name"]
    result_root = Path(config["paths"]["result_dir"])

    analysis_dir = (
        result_root
        / "analysis"
        / args.mode
    )

    analysis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = analysis_dir / "summary.json"

    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists. "
            "Use --overwrite to replace it."
        )

    print(f"[PATSyn] Dataset: {dataset_name}")
    print(f"[PATSyn] Analysis: {args.mode}")
    print(f"[PATSyn] epsilon: {epsilon}")
    print(f"[PATSyn] Seeds: {seeds}")

    summary = run_study(
        mode=args.mode,
        config=config,
        config_path=args.config,
        default_config_path=args.default_config,
        epsilon=epsilon,
        seeds=seeds,
        output_dir=analysis_dir,
        overwrite=args.overwrite,
        quick=args.quick,
    )

    output = {
        "dataset": dataset_name,
        "mode": args.mode,
        "epsilon": epsilon,
        "seeds": seeds,
        "summary": summary,
    }

    save_json(
        output,
        summary_path,
    )

    print("[PATSyn] Analysis complete.")
    print(f"[PATSyn] Saved: {analysis_dir}")


if __name__ == "__main__":
    main()