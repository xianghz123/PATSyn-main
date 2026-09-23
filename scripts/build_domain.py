#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patsyn.domain import build_public_domain, save_public_domain
from patsyn.utils import load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build the public domain for PATSyn.")
    parser.add_argument("--config", default="configs/porto.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.default_config, args.config)

    cache_dir = Path(config["paths"]["cache_dir"])
    domain_path = cache_dir / "public_domain.pkl"
    stats_path = cache_dir / "domain_stats.json"

    cache_dir.mkdir(parents=True, exist_ok=True)

    if domain_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{domain_path} already exists. Use --overwrite to rebuild it."
        )

    print(f"[PATSyn] Building public domain: {config['dataset']['name']}")

    public_domain, domain_stats = build_public_domain(config)
    save_public_domain(public_domain, domain_path)
    save_json(domain_stats, stats_path)

    print(f"[PATSyn] Cells: {domain_stats['num_cells']}")
    print(f"[PATSyn] Legal transitions: {domain_stats['num_edges']}")
    print(f"[PATSyn] |Omega_q|: {domain_stats['omega_q_size']}")
    print(f"[PATSyn] Saved: {domain_path}")


if __name__ == "__main__":
    main()
