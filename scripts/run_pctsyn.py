#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pctsyn.domain import load_public_domain
from pctsyn.preprocessing import load_processed_dataset
from pctsyn.recovery import recover_model, save_model
from pctsyn.reporting import generate_private_reports, save_reports
from pctsyn.synthesis import (
    generate_synthetic_trajectories,
    save_sampled_conditions,
    save_synthetic_dataset,
)
from pctsyn.utils import load_config, peak_rss_mib, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the PCTSyn synthesis pipeline.")
    parser.add_argument("--config", default="configs/porto.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-synthetic", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.default_config, args.config)

    epsilon = (
        args.epsilon
        if args.epsilon is not None
        else float(config["experiment"]["default_epsilon"])
    )
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])
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
            f"{trajectory_path} does not exist. Run scripts/prepare_data.py first."
        )
    if not domain_path.exists():
        raise FileNotFoundError(
            f"{domain_path} does not exist. Run scripts/build_domain.py first."
        )

    run_dir = output_root / f"eps_{epsilon}" / f"seed_{seed}"
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
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError(
            f"Outputs already exist in {run_dir}. Use --overwrite to replace them."
        )

    trajectories = load_processed_dataset(trajectory_path)
    public_domain = load_public_domain(domain_path)
    num_input = len(trajectories)

    print(f"[PCTSyn] Dataset: {config['dataset']['name']}")
    print(f"[PCTSyn] epsilon: {epsilon}")
    print(f"[PCTSyn] Seed: {seed}")
    print(f"[PCTSyn] Input trajectories: {num_input:,}")
    print(f"[PCTSyn] Synthetic trajectories: {num_synthetic:,}")

    reporting_start = time.perf_counter()
    reports = generate_private_reports(
        trajectories=trajectories,
        public_domain=public_domain,
        epsilon=epsilon,
        rho=rho,
        num_stages=num_stages,
        seed=seed,
    )
    save_reports(reports, reports_path)
    reporting_time = time.perf_counter() - reporting_start

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
    save_model(model, model_path)
    recovery_time = time.perf_counter() - recovery_start

    synthesis_start = time.perf_counter()
    synthetic, sampled_conditions = generate_synthetic_trajectories(
        model=model,
        public_domain=public_domain,
        num_trajectories=num_synthetic,
        seed=seed,
    )
    save_synthetic_dataset(synthetic, synthetic_path)
    save_sampled_conditions(sampled_conditions, conditions_path)
    synthesis_time = time.perf_counter() - synthesis_start

    method_time = reporting_time + recovery_time + synthesis_time
    domain_stats = public_domain.stats

    manifest = {
        "project": "PCTSyn",
        "dataset": config["dataset"]["name"],
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
        "omega_q_size": domain_stats["omega_q_size"],
        "omega_x_size": domain_stats["omega_x_size"],
        "runtime_seconds": {
            "reporting": reporting_time,
            "recovery": recovery_time,
            "synthesis": synthesis_time,
            "method_total": method_time,
        },
        "peak_rss_mib": peak_rss_mib(),
    }
    save_json(manifest, manifest_path)

    print(f"[PCTSyn] Reporting: {reporting_time:.2f} s")
    print(f"[PCTSyn] Recovery: {recovery_time:.2f} s")
    print(f"[PCTSyn] Synthesis: {synthesis_time:.2f} s")
    print(f"[PCTSyn] Method total: {method_time:.2f} s")
    print(f"[PCTSyn] Saved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
