import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from patsyn.domain import (
    PublicDomain,
    TripPattern,
)
from patsyn.recovery import (
    RecoveredModel,
)


Trajectory = List[int]


def _sample_probability_vector(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> int:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    total = float(
        probabilities.sum()
    )

    if total <= 0:
        raise ValueError(
            "Probability vector has zero mass."
        )

    probabilities = (
        probabilities / total
    )

    return int(
        rng.choice(
            len(probabilities),
            p=probabilities,
        )
    )


def _stage_index(
    transition_position: int,
    trajectory_length: int,
    num_stages: int,
) -> int:
    return (
        num_stages
        * transition_position
        - 1
    ) // trajectory_length


def sample_trip_condition(
    model: RecoveredModel,
    public_domain: PublicDomain,
    rng: np.random.Generator,
) -> TripPattern:
    q_index = (
        _sample_probability_vector(
            model.p_q,
            rng,
        )
    )

    return public_domain.omega_q[
        q_index
    ]


def sample_exact_length(
    q: TripPattern,
    public_domain: PublicDomain,
    rng: np.random.Generator,
) -> int:
    source, destination, bin_index = q

    lengths = (
        public_domain.get_feasible_lengths(
            source=source,
            destination=destination,
            bin_index=bin_index,
        )
    )

    if len(lengths) == 0:
        raise ValueError(
            f"No feasible exact length for {q}."
        )

    index = int(
        rng.integers(
            0,
            len(lengths),
        )
    )

    return int(
        lengths[index]
    )


def compute_backward_reachability(
    model: RecoveredModel,
    public_domain: PublicDomain,
    destination: int,
    length: int,
) -> np.ndarray:
    if length <= 0:
        raise ValueError(
            "length must be positive."
        )

    if length > public_domain.L_max:
        raise ValueError(
            "length exceeds L_max."
        )

    num_cells = (
        public_domain.grid.num_cells
    )

    num_stages = (
        model.num_stages
    )

    h = np.zeros(
        (
            length + 1,
            num_cells,
        ),
        dtype=np.float64,
    )

    h[
        length,
        destination,
    ] = 1.0

    for t in range(
        length,
        0,
        -1,
    ):
        stage = _stage_index(
            transition_position=t,
            trajectory_length=length,
            num_stages=num_stages,
        )

        kernel = (
            model.stabilized_kernels[
                stage
            ]
        )

        h[
            t - 1
        ] = (
            kernel @ h[t]
        )

    return h


def generate_one_trajectory(
    model: RecoveredModel,
    public_domain: PublicDomain,
    rng: np.random.Generator,
):
    q = sample_trip_condition(
        model=model,
        public_domain=public_domain,
        rng=rng,
    )

    source, destination, bin_index = q

    length = sample_exact_length(
        q=q,
        public_domain=public_domain,
        rng=rng,
    )

    h = compute_backward_reachability(
        model=model,
        public_domain=public_domain,
        destination=destination,
        length=length,
    )

    if h[0, source] <= 0:
        raise RuntimeError(
            "The sampled condition has zero "
            "backward reachability."
        )

    trajectory = [
        int(source)
    ]

    current = int(
        source
    )

    for t in range(
        1,
        length + 1,
    ):
        stage = _stage_index(
            transition_position=t,
            trajectory_length=length,
            num_stages=model.num_stages,
        )

        neighbors = (
            public_domain.grid.neighbors(
                current
            )
        )

        if len(neighbors) == 0:
            raise RuntimeError(
                "Current state has no legal successor."
            )

        neighbors_array = np.asarray(
            neighbors,
            dtype=np.int64,
        )

        base = (
            model.stabilized_kernels[
                stage,
                current,
                neighbors_array,
            ]
        )

        weights = (
            base
            * h[
                t,
                neighbors_array,
            ]
        )

        total = float(
            weights.sum()
        )

        if total <= 0:
            raise RuntimeError(
                "No feasible conditioned successor."
            )

        probabilities = (
            weights / total
        )

        local_index = int(
            rng.choice(
                len(neighbors_array),
                p=probabilities,
            )
        )

        current = int(
            neighbors_array[
                local_index
            ]
        )

        trajectory.append(
            current
        )

    if trajectory[0] != source:
        raise AssertionError(
            "Start constraint violated."
        )

    if trajectory[-1] != destination:
        raise AssertionError(
            "Destination constraint violated."
        )

    if (
        len(trajectory) - 1
        != length
    ):
        raise AssertionError(
            "Exact-length constraint violated."
        )

    for left, right in zip(
        trajectory[:-1],
        trajectory[1:],
    ):
        if not public_domain.grid.is_legal_edge(
            left,
            right,
        ):
            raise AssertionError(
                "Adjacency constraint violated."
            )

    condition = (
        int(source),
        int(destination),
        int(bin_index),
        int(length),
    )

    return (
        trajectory,
        condition,
    )


def generate_synthetic_trajectories(
    model: RecoveredModel,
    public_domain: PublicDomain,
    num_trajectories: int,
    seed: int,
):
    if num_trajectories <= 0:
        raise ValueError(
            "num_trajectories must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    trajectories: List[
        Trajectory
    ] = []

    conditions = np.empty(
        (
            num_trajectories,
            4,
        ),
        dtype=np.int64,
    )

    for index in range(
        num_trajectories
    ):
        trajectory, condition = (
            generate_one_trajectory(
                model=model,
                public_domain=(
                    public_domain
                ),
                rng=rng,
            )
        )

        trajectories.append(
            trajectory
        )

        conditions[
            index,
            :,
        ] = condition

    return (
        trajectories,
        conditions,
    )


def save_synthetic_dataset(
    trajectories: Sequence[
        Sequence[int]
    ],
    output_path: Path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "wb"
    ) as file:
        pickle.dump(
            [
                list(trajectory)
                for trajectory
                in trajectories
            ],
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_synthetic_dataset(
    input_path: Path,
) -> List[Trajectory]:
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Synthetic dataset does not exist: {input_path}"
        )

    with input_path.open(
        "rb"
    ) as file:
        trajectories = (
            pickle.load(file)
        )

    return trajectories


def save_sampled_conditions(
    conditions: np.ndarray,
    output_path: Path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = np.asarray(
        conditions,
        dtype=np.int64,
    )

    if (
        conditions.ndim != 2
        or conditions.shape[1] != 4
    ):
        raise ValueError(
            "conditions must have shape (N, 4)."
        )

    np.savez_compressed(
        output_path,
        source=conditions[:, 0],
        destination=conditions[:, 1],
        bin=conditions[:, 2],
        length=conditions[:, 3],
    )


def load_sampled_conditions(
    input_path: Path,
) -> Dict[str, np.ndarray]:
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Condition file does not exist: {input_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:
        return {
            "source": data[
                "source"
            ].astype(
                np.int64
            ),
            "destination": data[
                "destination"
            ].astype(
                np.int64
            ),
            "bin": data[
                "bin"
            ].astype(
                np.int64
            ),
            "length": data[
                "length"
            ].astype(
                np.int64
            ),
        }