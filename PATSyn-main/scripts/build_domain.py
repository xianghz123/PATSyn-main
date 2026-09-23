#!/usr/bin/env python3

import argparse
from pathlib import Path

from patsyn.domain import (
    build_public_domain,
    save_public_domain,
)
from patsyn.utils import (
    load_config,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the public domain for PATSyn."
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
        "--overwrite",
        action="store_true",
        help="Overwrite an existing public-domain cache.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        default_path=args.default_config,
        dataset_path=args.config,
    )

    dataset_name = config["dataset"]["name"]
    cache_dir = Path(config["paths"]["cache_dir"])
    domain_path = cache_dir / "public_domain.pkl"
    stats_path = cache_dir / "domain_stats.json"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if domain_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{domain_path} already exists. "
            "Use --overwrite to rebuild it."
        )

    print(f"[PATSyn] Building public domain: {dataset_name}")

    public_domain, domain_stats = build_public_domain(
        config=config,
    )

    save_public_domain(
        public_domain=public_domain,
        output_path=domain_path,
    )

    save_json(domain_stats, stats_path)

    print(
        "[PATSyn] Cells: "
        f"{domain_stats.get('num_cells', 'N/A')}"
    )
    print(
        "[PATSyn] Legal transitions: "
        f"{domain_stats.get('num_edges', 'N/A')}"
    )
    print(
        "[PATSyn] |Omega_q|: "
        f"{domain_stats.get('omega_q_size', 'N/A')}"
    )
    print(f"[PATSyn] Saved: {domain_path}")
    print(f"[PATSyn] Saved: {stats_path}")


if __name__ == "__main__":
    main()