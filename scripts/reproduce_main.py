#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pctsyn.utils import load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the main PCTSyn experiments.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/porto.yaml"],
        help="Dataset configuration files.",
    )
    parser.add_argument("--default-config", default="configs/default.yaml")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")

    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_command(command):
    print("[PCTSyn] Running:", " ".join(str(item) for item in command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def flatten_dict(data, prefix=""):
    output = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_dict(value, name))
        else:
            output[name] = value
    return output


def ensure_prepared(config_path, default_config_path, config, overwrite):
    python_exec = sys.executable
    trajectory_path = REPO_ROOT / config["paths"]["processed_dir"] / "trajectories.pkl"
    domain_path = REPO_ROOT / config["paths"]["cache_dir"] / "public_domain.pkl"

    if overwrite or not trajectory_path.exists():
        command = [
            python_exec,
            "scripts/prepare_data.py",
            "--config",
            config_path,
            "--default-config",
            default_config_path,
        ]
        if overwrite:
            command.append("--overwrite")
        run_command(command)

    if overwrite or not domain_path.exists():
        command = [
            python_exec,
            "scripts/build_domain.py",
            "--config",
            config_path,
            "--default-config",
            default_config_path,
        ]
        if overwrite:
            command.append("--overwrite")
        run_command(command)


def run_one(config_path, default_config_path, config, epsilon, seed, num_synthetic, overwrite):
    python_exec = sys.executable
    output_root = REPO_ROOT / config["paths"]["output_dir"]
    result_root = REPO_ROOT / config["paths"]["result_dir"]
    run_dir = output_root / f"eps_{epsilon}" / f"seed_{seed}"
    result_dir = result_root / f"eps_{epsilon}" / f"seed_{seed}"

    if overwrite or not (run_dir / "manifest.json").exists():
        command = [
            python_exec,
            "scripts/run_pctsyn.py",
            "--config",
            config_path,
            "--default-config",
            default_config_path,
            "--epsilon",
            str(epsilon),
            "--seed",
            str(seed),
            "--num-synthetic",
            str(num_synthetic),
        ]
        if overwrite:
            command.append("--overwrite")
        run_command(command)

    if overwrite or not (result_dir / "metrics.json").exists():
        command = [
            python_exec,
            "scripts/evaluate.py",
            "--config",
            config_path,
            "--default-config",
            default_config_path,
            "--epsilon",
            str(epsilon),
            "--seed",
            str(seed),
        ]
        if overwrite:
            command.append("--overwrite")
        run_command(command)


def aggregate_results(config, epsilons, seeds):
    result_root = REPO_ROOT / config["paths"]["result_dir"]
    rows = []

    for epsilon in epsilons:
        for seed in seeds:
            path = result_root / f"eps_{epsilon}" / f"seed_{seed}" / "metrics.json"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            row = {
                "dataset": data["dataset"],
                "epsilon": data["epsilon"],
                "seed": data["seed"],
                "num_real": data["num_real"],
                "num_synthetic": data["num_synthetic"],
            }
            row.update(flatten_dict(data["metrics"]))
            rows.append(row)

    if not rows:
        return None

    output_path = result_root / "main_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main():
    args = parse_args()
    mode = "full" if args.full else "quick"
    summary = {"project": "PCTSyn", "mode": mode, "datasets": []}

    for config_path in args.configs:
        config = load_config(args.default_config, config_path)

        if not args.skip_prepare:
            ensure_prepared(
                config_path,
                args.default_config,
                config,
                args.overwrite,
            )

        if mode == "quick":
            epsilons = [float(config["experiment"]["default_epsilon"])]
            seeds = [int(config["project"]["seed"])]
            num_synthetic = min(10000, int(config["dataset"]["num_synthetic"]))
        else:
            epsilons = [float(value) for value in config["privacy"]["epsilons"]]
            seeds = [int(value) for value in config["experiment"]["seeds"]]
            num_synthetic = int(config["dataset"]["num_synthetic"])

        for epsilon in epsilons:
            for seed in seeds:
                run_one(
                    config_path,
                    args.default_config,
                    config,
                    epsilon,
                    seed,
                    num_synthetic,
                    args.overwrite,
                )

        main_results = aggregate_results(config, epsilons, seeds)
        summary["datasets"].append(
            {
                "dataset": config["dataset"]["name"],
                "epsilons": epsilons,
                "seeds": seeds,
                "num_synthetic": num_synthetic,
                "main_results": str(main_results) if main_results else None,
            }
        )

    summary_path = REPO_ROOT / "results" / "reproduction_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, summary_path)
    print(f"[PCTSyn] Summary: {summary_path}")


if __name__ == "__main__":
    main()
