#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patsyn.preprocessing import prepare_dataset, save_processed_dataset
from patsyn.utils import load_config, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare trajectories for PATSyn.")
    parser.add_argument("--config", default="configs/porto.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.default_config, args.config)

    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    set_seed(seed)

    processed_dir = Path(config["paths"]["processed_dir"])
    trajectory_path = processed_dir / "trajectories.pkl"
    dataset_stats_path = processed_dir / "dataset_stats.json"
    admission_stats_path = processed_dir / "admission_stats.json"

    processed_dir.mkdir(parents=True, exist_ok=True)

    if trajectory_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{trajectory_path} already exists. Use --overwrite to replace it."
        )

    print(f"[PATSyn] Preparing dataset: {config['dataset']['name']}")

    trajectories, dataset_stats, admission_stats = prepare_dataset(
        config=config,
        seed=seed,
    )

    save_processed_dataset(trajectories, trajectory_path)
    save_json(dataset_stats, dataset_stats_path)
    save_json(admission_stats, admission_stats_path)

    print(f"[PATSyn] Admitted trajectories: {len(trajectories):,}")
    print(f"[PATSyn] Saved: {trajectory_path}")


if __name__ == "__main__":
    main()
