# PATSyn

This repository provides the main experimental implementation for:

**PATSyn: Progress-Aligned Trajectory Synthesis under Local Differential Privacy**

PATSyn uses a fixed two-report trajectory-level LDP mechanism to recover progress-aligned mobility statistics and generates synthetic trajectories through exact-step destination-conditioned sampling.

## Environment

We recommend Python 3.10.

```bash
conda create -n patsyn python=3.10 -y
conda activate patsyn
pip install -r requirements.txt
```

## Data

This repository uses the Porto taxi trajectory dataset as the example dataset.

Place the raw Porto CSV file at:

```text
data/raw/porto/train.csv
```

The main configuration files are:

```text
configs/default.yaml
configs/porto.yaml
```

Before running the pipeline, fill the public grid, length-bin, and domain-bound settings in `configs/porto.yaml` with the exact configuration used in the paper experiments.

## Run PATSyn

Run the following scripts from the repository root:

```bash
python scripts/prepare_data.py --config configs/porto.yaml
python scripts/build_domain.py --config configs/porto.yaml
python scripts/run_patsyn.py --config configs/porto.yaml --epsilon 1.0 --seed 42
python scripts/evaluate.py --config configs/porto.yaml --epsilon 1.0 --seed 42
```

For a quick end-to-end run:

```bash
python scripts/reproduce_main.py --quick
```

For the main multi-budget and multi-seed experiments:

```bash
python scripts/reproduce_main.py --full
```

## Code Structure

```text
configs/
    default.yaml
    porto.yaml

scripts/
    prepare_data.py
    build_domain.py
    run_patsyn.py
    evaluate.py
    reproduce_main.py

patsyn/
    preprocessing.py
    domain.py
    hcr.py
    reporting.py
    recovery.py
    synthesis.py
    metrics.py
    utils.py
```

The core PATSyn implementation is contained in:

```text
patsyn/reporting.py
patsyn/recovery.py
patsyn/synthesis.py
```

## Outputs

Synthetic trajectories and intermediate outputs are stored under:

```text
outputs/porto/
```

Evaluation results are stored under:

```text
results/porto/
```

## Notes

- The public grid, legal transitions, exact-step reachability, and feasible-length sets are public information.
- Each admitted user releases two privatized categorical reports and one trajectory-independent stage index.
- Raw trajectories and unrandomized representations remain on the client side.
- Trajectories outside the admitted domain are excluded rather than truncated.
