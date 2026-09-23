#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

from patsyn.domain import load_public_domain
from patsyn.preprocessing import load_processed_dataset
from patsyn.recovery import (
    recover_model,
    save_model,
)
from patsyn.reporting import (
    generate_private_reports,
    save_reports,
)
from patsyn.synthesis import (
    generate_synthetic_trajectories,
    save_sampled_conditions,
    save_synthetic_dataset,
)
from patsyn.utils import (
    load_config,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PATSyn synthesis pipeline."
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
        "--epsilon",
        type=float,
        default=None,
        help="Total trajectory-level privacy budget.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-synthetic",
        type=int,
        default=None,
        help="Number of synthetic trajectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
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
    seed = (
        args.seed
        if args.seed is not None
        else int(config["project"]["seed"])
    )
    num_synthetic = (
        args.num_synthetic
        if args.num_synthetic is not None
        else int(config["dataset"]["num_synthetic"])
    )

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if num_synthetic <= 0:
        raise ValueError("num_synthetic must be positive.")

    set_seed(seed)

    dataset_name = config["dataset"]["name"]
    rho = float(config["privacy"]["rho"])
    num_stages = int(config["model"]["R"])
    lambda_value = float(config["model"]["lambda"])

    epsilon_q = rho * epsilon
    epsilon_x = (1.0 - rho) * epsilon

    processed_dir = Path(config["paths"]["processed_dir"])
    cache_dir = Path(config["paths"]["cache_dir"])
    output_root = Path(config["paths"]["output_dir"])

    trajectory_path = processed_dir / "trajectories.pkl"
    domain_path = cache_dir / "public_domain.pkl"

    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"{trajectory_path} does not exist. "
            "Run scripts/prepare_data.py first."
        )

    if not domain_path.exists():
        raise FileNotFoundError(
            f"{domain_path} does not exist. "
            "Run scripts/build_domain.py first."
        )

    run_dir = (
        output_root
        / f"eps_{epsilon}"
        / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    reports_path = run_dir / "reports.npz"
    model_path = run_dir / "model.pkl"
    synthetic_path = run_dir / "synthetic.pkl"
    conditions_path = run_dir / "sampled_conditions.npz"
    manifest_path = run_dir / "manifest.json"

    expected_outputs = [
        reports_path,
        model_path,
        synthetic_path,
        conditions_path,
        manifest_path,
    ]

    if (
        not args.overwrite
        and any(path.exists() for path in expected_outputs)
    ):
        raise FileExistsError(
            f"Outputs already exist in {run_dir}. "
            "Use --overwrite to replace them."
        )

    print(f"[PATSyn] Dataset: {dataset_name}")
    print(f"[PATSyn] epsilon: {epsilon}")
    print(f"[PATSyn] epsilon_q: {epsilon_q}")
    print(f"[PATSyn] epsilon_x: {epsilon_x}")
    print(f"[PATSyn] R: {num_stages}")
    print(f"[PATSyn] lambda: {lambda_value}")
    print(f"[PATSyn] Seed: {seed}")
    print(f"[PATSyn] Synthetic trajectories: {num_synthetic:,}")

    start_time = time.perf_counter()

    trajectories = load_processed_dataset(
        trajectory_path
    )

    public_domain = load_public_domain(
        domain_path
    )

    print(
        f"[PATSyn] Loaded admitted trajectories: "
        f"{len(trajectories):,}"
    )

    report_start = time.perf_counter()

    reports = generate_private_reports(
        trajectories=trajectories,
        public_domain=public_domain,
        epsilon=epsilon,
        rho=rho,
        num_stages=num_stages,
        seed=seed,
    )

    save_reports(
        reports=reports,
        output_path=reports_path,
    )

    report_time = time.perf_counter() - report_start

    # Server-side processing uses only released reports and public data.
    num_input = len(trajectories)
    del trajectories

    recovery_start = time.perf_counter()

    model = recover_model(
        reports=reports,
        public_domain=public_domain,
        epsilon=epsilon,
        rho=rho,
        num_stages=num_stages,
        lambda_value=lambda_value,
    )

    save_model(
        model=model,
        output_path=model_path,
    )

    recovery_time = time.perf_counter() - recovery_start

    synthesis_start = time.perf_counter()

    synthetic, sampled_conditions = generate_synthetic_trajectories(
        model=model,
        public_domain=public_domain,
        num_trajectories=num_synthetic,
        seed=seed,
    )

    save_synthetic_dataset(
        trajectories=synthetic,
        output_path=synthetic_path,
    )

    save_sampled_conditions(
        conditions=sampled_conditions,
        output_path=conditions_path,
    )

    synthesis_time = time.perf_counter() - synthesis_start
    total_time = time.perf_counter() - start_time

    domain_stats = getattr(
        public_domain,
        "stats",
        {},
    )

    manifest = {
        "project": "PATSyn",
        "dataset": dataset_name,
        "epsilon": epsilon,
        "rho": rho,
        "epsilon_q": epsilon_q,
        "epsilon_x": epsilon_x,
        "R": num_stages,
        "L_max": int(config["model"]["L_max"]),
        "lambda": lambda_value,
        "seed": seed,
        "num_input": num_input,
        "num_synthetic": num_synthetic,
        "omega_q_size": domain_stats.get(
            "omega_q_size",
            None,
        ),
        "omega_x_size": domain_stats.get(
            "omega_x_size",
            None,
        ),
        "runtime_seconds": {
            "reporting": report_time,
            "recovery": recovery_time,
            "synthesis": synthesis_time,
            "total": total_time,
        },
    }

    save_json(
        manifest,
        manifest_path,
    )

    print(f"[PATSyn] Reporting: {report_time:.2f} s")
    print(f"[PATSyn] Recovery: {recovery_time:.2f} s")
    print(f"[PATSyn] Synthesis: {synthesis_time:.2f} s")
    print(f"[PATSyn] Total: {total_time:.2f} s")
    print(f"[PATSyn] Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()