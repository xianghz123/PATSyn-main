#!/usr/bin/env python3

import argparse
from pathlib import Path

from patsyn.preprocessing import (
    prepare_dataset,
    save_processed_dataset,
)
from patsyn.utils import (
    load_config,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare trajectories for PATSyn."
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
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic cohort selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        default_path=args.default_config,
        dataset_path=args.config,
    )

    seed = (
        args.seed
        if args.seed is not None
        else int(config["project"]["seed"])
    )
    set_seed(seed)

    dataset_name = config["dataset"]["name"]
    processed_dir = Path(config["paths"]["processed_dir"])
    trajectory_path = processed_dir / "trajectories.pkl"
    dataset_stats_path = processed_dir / "dataset_stats.json"
    admission_stats_path = processed_dir / "admission_stats.json"

    processed_dir.mkdir(parents=True, exist_ok=True)

    if trajectory_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{trajectory_path} already exists. "
            "Use --overwrite to replace it."
        )

    print(f"[PATSyn] Preparing dataset: {dataset_name}")
    print(f"[PATSyn] Seed: {seed}")

    trajectories, dataset_stats, admission_stats = prepare_dataset(
        config=config,
        seed=seed,
    )

    save_processed_dataset(
        trajectories=trajectories,
        output_path=trajectory_path,
    )

    save_json(dataset_stats, dataset_stats_path)
    save_json(admission_stats, admission_stats_path)

    print(f"[PATSyn] Admitted trajectories: {len(trajectories):,}")
    print(f"[PATSyn] Saved: {trajectory_path}")
    print(f"[PATSyn] Saved: {dataset_stats_path}")
    print(f"[PATSyn] Saved: {admission_stats_path}")


if __name__ == "__main__":
    main()