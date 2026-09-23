#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patsyn.domain import load_public_domain
from patsyn.metrics import evaluate_all
from patsyn.preprocessing import load_processed_dataset
from patsyn.synthesis import load_sampled_conditions, load_synthetic_dataset
from patsyn.utils import load_config, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PATSyn outputs.")
    parser.add_argument("--config", default="configs/porto.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic", default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def flatten_dict(data, prefix=""):
    output = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_dict(value, name))
        else:
            output[name] = value
    return output


def save_metrics_csv(metrics, metadata, output_path):
    row = {**metadata, **flatten_dict(metrics)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    config = load_config(args.default_config, args.config)

    epsilon = (
        args.epsilon
        if args.epsilon is not None
        else float(config["experiment"]["default_epsilon"])
    )
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
    set_seed(seed)

    processed_dir = Path(config["paths"]["processed_dir"])
    cache_dir = Path(config["paths"]["cache_dir"])
    output_root = Path(config["paths"]["output_dir"])
    result_root = Path(config["paths"]["result_dir"])

    run_dir = output_root / f"eps_{epsilon}" / f"seed_{seed}"
    result_dir = result_root / f"eps_{epsilon}" / f"seed_{seed}"

    real_path = processed_dir / "trajectories.pkl"
    domain_path = cache_dir / "public_domain.pkl"
    synthetic_path = Path(args.synthetic) if args.synthetic else run_dir / "synthetic.pkl"
    conditions_path = (
        Path(args.conditions) if args.conditions else run_dir / "sampled_conditions.npz"
    )

    for path in [real_path, domain_path, synthetic_path, conditions_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file does not exist: {path}")

    metrics_json_path = result_dir / "metrics.json"
    metrics_csv_path = result_dir / "metrics.csv"
    if metrics_json_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{metrics_json_path} already exists. Use --overwrite to replace it."
        )

    real_trajectories = load_processed_dataset(real_path)
    synthetic_trajectories = load_synthetic_dataset(synthetic_path)
    sampled_conditions = load_sampled_conditions(conditions_path)
    public_domain = load_public_domain(domain_path)

    metrics = evaluate_all(
        real_trajectories=real_trajectories,
        synthetic_trajectories=synthetic_trajectories,
        sampled_conditions=sampled_conditions,
        public_domain=public_domain,
        config=config,
        seed=seed,
    )

    output = {
        "dataset": config["dataset"]["name"],
        "epsilon": epsilon,
        "seed": seed,
        "num_real": len(real_trajectories),
        "num_synthetic": len(synthetic_trajectories),
        "metrics": metrics,
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    save_json(output, metrics_json_path)
    save_metrics_csv(
        metrics,
        {
            "dataset": config["dataset"]["name"],
            "epsilon": epsilon,
            "seed": seed,
            "num_real": len(real_trajectories),
            "num_synthetic": len(synthetic_trajectories),
        },
        metrics_csv_path,
    )

    print(f"[PATSyn] Saved: {metrics_json_path}")
    print(f"[PATSyn] Saved: {metrics_csv_path}")


if __name__ == "__main__":
    main()
