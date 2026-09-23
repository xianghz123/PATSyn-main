# PATSyn

This repository provides the main experimental implementation for:

**PATSyn: Progress-Aligned Trajectory Synthesis under Local Differential Privacy**

PATSyn applies a fixed two-report trajectory-level LDP mechanism to recover progress-aligned mobility statistics and generates synthetic trajectories using exact-step destination-conditioned sampling.

## Environment

We recommend Python 3.10.

```bash
conda create -n patsyn python=3.10 -y
conda activate patsyn
pip install -r requirements.txt
```

## Data

This repository uses the Porto taxi trajectory dataset as the example dataset.

Raw data are not included. Please place the Porto CSV file at:

```text
data/raw/porto/train.csv
```

The main configuration files are:

```text
configs/default.yaml
configs/porto.yaml
```

Please make sure that the dataset paths and public-domain settings in `configs/porto.yaml` are correctly configured before running the experiments.

## Run PATSyn

Run the main pipeline from the repository root:

```bash
python scripts/prepare_data.py \
  --config configs/porto.yaml

python scripts/build_domain.py \
  --config configs/porto.yaml

python scripts/run_patsyn.py \
  --config configs/porto.yaml \
  --epsilon 1.0 \
  --seed 42

python scripts/evaluate.py \
  --config configs/porto.yaml \
  --epsilon 1.0 \
  --seed 42
```

For a quick end-to-end run:

```bash
python scripts/reproduce_paper.py --quick
```

For the full multi-budget and multi-seed experiments:

```bash
python scripts/reproduce_paper.py --full
```

Additional experiments can be executed with:

```bash
python scripts/run_analysis.py \
  --config configs/porto.yaml \
  --mode all
```

Supported analysis modes include:

```text
ablation
sensitivity
length
efficiency
scalability
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
    run_analysis.py
    reproduce_paper.py

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

A typical single-run output directory is:

```text
outputs/porto/eps_1.0/seed_42/
```

and contains the privatized reports, recovered model, synthetic trajectories, sampled conditions, and run configuration.

## Notes

- The grid graph, legal transitions, exact-step reachability, and feasible-length sets are treated as public information.
- Each admitted user releases two privatized categorical reports and one trajectory-independent stage index.
- Raw trajectories and unrandomized trajectory representations remain on the client side.
- Trajectories outside the admitted domain are excluded rather than truncated.
- Raw datasets, large intermediate files, logs, and external baseline repositories are not included in this repository.