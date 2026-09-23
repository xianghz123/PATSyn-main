#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from patsyn.utils import (
    load_config,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce the PATSyn experimental pipeline."
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["configs/porto.yaml"],
        help="Dataset configuration files.",
    )
    parser.add_argument(
        "--default-config",
        type=str,
        default="configs/default.yaml",
        help="Default configuration file.",
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Run a lightweight reproducibility check.",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Run the full PATSyn experiments.",
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip preprocessing and public-domain construction.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip additional analysis experiments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def run_command(command, repo_root):
    print(
        "[PATSyn] Running:",
        " ".join(str(item) for item in command),
    )

    subprocess.run(
        command,
        cwd=repo_root,
        check=True,
    )


def flatten_dict(data, prefix=""):
    flat = {}

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(
                flatten_dict(
                    value,
                    name,
                )
            )
        else:
            flat[name] = value

    return flat


def prepare_dataset(
    repo_root,
    python_exec,
    config_path,
    default_config_path,
    config,
    overwrite,
):
    processed_path = (
        repo_root
        / config["paths"]["processed_dir"]
        / "trajectories.pkl"
    )

    if processed_path.exists() and not overwrite:
        print(
            f"[PATSyn] Using existing dataset: "
            f"{processed_path}"
        )
        return

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

    run_command(
        command,
        repo_root,
    )


def build_domain(
    repo_root,
    python_exec,
    config_path,
    default_config_path,
    config,
    overwrite,
):
    domain_path = (
        repo_root
        / config["paths"]["cache_dir"]
        / "public_domain.pkl"
    )

    if domain_path.exists() and not overwrite:
        print(
            f"[PATSyn] Using existing domain: "
            f"{domain_path}"
        )
        return

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

    run_command(
        command,
        repo_root,
    )


def run_single_experiment(
    repo_root,
    python_exec,
    config_path,
    default_config_path,
    config,
    epsilon,
    seed,
    num_synthetic,
    overwrite,
):
    output_root = (
        repo_root
        / config["paths"]["output_dir"]
    )

    result_root = (
        repo_root
        / config["paths"]["result_dir"]
    )

    run_dir = (
        output_root
        / f"eps_{epsilon}"
        / f"seed_{seed}"
    )

    result_dir = (
        result_root
        / f"eps_{epsilon}"
        / f"seed_{seed}"
    )

    manifest_path = run_dir / "manifest.json"
    metrics_path = result_dir / "metrics.json"

    if not manifest_path.exists() or overwrite:
        command = [
            python_exec,
            "scripts/run_patsyn.py",
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

        run_command(
            command,
            repo_root,
        )
    else:
        print(
            f"[PATSyn] Using existing run: "
            f"{run_dir}"
        )

    if not metrics_path.exists() or overwrite:
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

        run_command(
            command,
            repo_root,
        )
    else:
        print(
            f"[PATSyn] Using existing metrics: "
            f"{metrics_path}"
        )


def aggregate_results(
    repo_root,
    config,
    epsilons,
    seeds,
):
    result_root = (
        repo_root
        / config["paths"]["result_dir"]
    )

    rows = []

    for epsilon in epsilons:
        for seed in seeds:
            metrics_path = (
                result_root
                / f"eps_{epsilon}"
                / f"seed_{seed}"
                / "metrics.json"
            )

            if not metrics_path.exists():
                continue

            with metrics_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            row = {
                "dataset": data["dataset"],
                "epsilon": data["epsilon"],
                "seed": data["seed"],
                "num_real": data["num_real"],
                "num_synthetic": data["num_synthetic"],
            }

            row.update(
                flatten_dict(
                    data["metrics"]
                )
            )

            rows.append(row)

    if not rows:
        return None

    summary_path = (
        result_root
        / "main_results.csv"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def run_additional_analysis(
    repo_root,
    python_exec,
    config_path,
    default_config_path,
    overwrite,
):
    command = [
        python_exec,
        "scripts/run_analysis.py",
        "--config",
        config_path,
        "--default-config",
        default_config_path,
        "--mode",
        "all",
    ]

    if overwrite:
        command.append("--overwrite")

    run_command(
        command,
        repo_root,
    )


def main():
    args = parse_args()

    repo_root = Path(
        __file__
    ).resolve().parents[1]

    python_exec = sys.executable

    mode = (
        "full"
        if args.full
        else "quick"
    )

    artifact_summary = {
        "project": "PATSyn",
        "mode": mode,
        "datasets": [],
    }

    for config_path in args.configs:
        config = load_config(
            default_path=args.default_config,
            dataset_path=config_path,
        )

        dataset_name = config["dataset"]["name"]

        print("=" * 60)
        print(f"[PATSyn] Dataset: {dataset_name}")
        print(f"[PATSyn] Mode: {mode}")
        print("=" * 60)

        if not args.skip_prepare:
            prepare_dataset(
                repo_root=repo_root,
                python_exec=python_exec,
                config_path=config_path,
                default_config_path=args.default_config,
                config=config,
                overwrite=args.overwrite,
            )

            build_domain(
                repo_root=repo_root,
                python_exec=python_exec,
                config_path=config_path,
                default_config_path=args.default_config,
                config=config,
                overwrite=args.overwrite,
            )

        if mode == "quick":
            epsilons = [
                float(
                    config["experiment"][
                        "default_epsilon"
                    ]
                )
            ]

            seeds = [
                int(
                    config["project"]["seed"]
                )
            ]

            num_synthetic = min(
                10000,
                int(
                    config["dataset"][
                        "num_synthetic"
                    ]
                ),
            )

        else:
            epsilons = [
                float(value)
                for value
                in config["privacy"]["epsilons"]
            ]

            seeds = [
                int(value)
                for value
                in config["experiment"]["seeds"]
            ]

            num_synthetic = int(
                config["dataset"]["num_synthetic"]
            )

        for epsilon in epsilons:
            for seed in seeds:
                run_single_experiment(
                    repo_root=repo_root,
                    python_exec=python_exec,
                    config_path=config_path,
                    default_config_path=args.default_config,
                    config=config,
                    epsilon=epsilon,
                    seed=seed,
                    num_synthetic=num_synthetic,
                    overwrite=args.overwrite,
                )

        summary_path = aggregate_results(
            repo_root=repo_root,
            config=config,
            epsilons=epsilons,
            seeds=seeds,
        )

        dataset_summary = {
            "dataset": dataset_name,
            "epsilons": epsilons,
            "seeds": seeds,
            "num_synthetic": num_synthetic,
            "main_results": (
                str(summary_path)
                if summary_path is not None
                else None
            ),
        }

        artifact_summary["datasets"].append(
            dataset_summary
        )

        if (
            mode == "full"
            and not args.skip_analysis
        ):
            run_additional_analysis(
                repo_root=repo_root,
                python_exec=python_exec,
                config_path=config_path,
                default_config_path=args.default_config,
                overwrite=args.overwrite,
            )

    summary_path = (
        repo_root
        / "results"
        / "reproduction_summary.json"
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        artifact_summary,
        summary_path,
    )

    print("=" * 60)
    print("[PATSyn] Reproduction complete.")
    print(f"[PATSyn] Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()