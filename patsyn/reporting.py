from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from patsyn.domain import (
    Edge,
    PublicDomain,
)
from patsyn.hcr import (
    batch_encode_indices,
)


Trajectory = Sequence[int]


@dataclass
class PrivateReports:
    q_j: np.ndarray
    q_sigma: np.ndarray
    stage: np.ndarray
    x_j: np.ndarray
    x_sigma: np.ndarray
    epsilon_q: float
    epsilon_x: float
    num_stages: int
    omega_q_size: int
    omega_x_size: int

    @property
    def num_users(self) -> int:
        return len(
            self.stage
        )


def stage_index(
    transition_position: int,
    trajectory_length: int,
    num_stages: int,
) -> int:
    if not (
        1 <= transition_position
        <= trajectory_length
    ):
        raise ValueError(
            "Invalid transition position."
        )

    if trajectory_length <= 0:
        raise ValueError(
            "trajectory_length must be positive."
        )

    if num_stages <= 0:
        raise ValueError(
            "num_stages must be positive."
        )

    return (
        num_stages
        * transition_position
        - 1
    ) // trajectory_length


def build_joint_pattern(
    trajectory: Trajectory,
    public_domain: PublicDomain,
) -> Tuple[int, int, int]:
    if len(trajectory) < 2:
        raise ValueError(
            "Trajectory must contain at least one transition."
        )

    length = len(trajectory) - 1

    if not (
        1 <= length
        <= public_domain.L_max
    ):
        raise ValueError(
            "Trajectory lies outside Omega_tau."
        )

    source = int(
        trajectory[0]
    )

    destination = int(
        trajectory[-1]
    )

    bin_index = (
        public_domain.length_bin_index(
            length
        )
    )

    q = (
        source,
        destination,
        bin_index,
    )

    if q not in public_domain.q_to_index:
        raise ValueError(
            f"Infeasible joint pattern: {q}"
        )

    return q


def build_stage_edges(
    trajectory: Trajectory,
    public_domain: PublicDomain,
    num_stages: int,
) -> List[List[int]]:
    length = len(trajectory) - 1

    if length <= 0:
        raise ValueError(
            "Trajectory must contain at least one transition."
        )

    stages: List[List[int]] = [
        []
        for _ in range(
            num_stages
        )
    ]

    for offset in range(
        length
    ):
        source = int(
            trajectory[offset]
        )

        target = int(
            trajectory[offset + 1]
        )

        edge: Edge = (
            source,
            target,
        )

        if edge not in (
            public_domain.edge_to_index
        ):
            raise ValueError(
                f"Illegal transition: {edge}"
            )

        r = stage_index(
            transition_position=(
                offset + 1
            ),
            trajectory_length=length,
            num_stages=num_stages,
        )

        stages[r].append(
            public_domain.edge_to_index[
                edge
            ]
        )

    return stages


def sample_representative_category(
    stage_edges: Sequence[
        Sequence[int]
    ],
    selected_stage: int,
    bot_index: int,
    rng: np.random.Generator,
) -> int:
    edges = stage_edges[
        selected_stage
    ]

    if len(edges) == 0:
        return bot_index

    selected = int(
        rng.integers(
            0,
            len(edges),
        )
    )

    return int(
        edges[selected]
    )


def generate_private_reports(
    trajectories: Sequence[
        Trajectory
    ],
    public_domain: PublicDomain,
    epsilon: float,
    rho: float,
    num_stages: int,
    seed: int,
) -> PrivateReports:
    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    if not (
        0.0 < rho < 1.0
    ):
        raise ValueError(
            "rho must lie in (0, 1)."
        )

    if num_stages <= 0:
        raise ValueError(
            "num_stages must be positive."
        )

    num_users = len(
        trajectories
    )

    if num_users == 0:
        raise ValueError(
            "No trajectories provided."
        )

    epsilon_q = (
        rho * epsilon
    )

    epsilon_x = (
        (1.0 - rho)
        * epsilon
    )

    rng = np.random.default_rng(
        seed
    )

    q_indices = np.empty(
        num_users,
        dtype=np.int64,
    )

    x_indices = np.empty(
        num_users,
        dtype=np.int64,
    )

    # Stored stage indices follow the paper convention 1,...,R.
    stages = np.empty(
        num_users,
        dtype=np.int16,
    )

    for user_index, trajectory in enumerate(
        trajectories
    ):
        q = build_joint_pattern(
            trajectory=trajectory,
            public_domain=public_domain,
        )

        q_indices[
            user_index
        ] = public_domain.q_to_index[
            q
        ]

        stage_edges = (
            build_stage_edges(
                trajectory=trajectory,
                public_domain=(
                    public_domain
                ),
                num_stages=(
                    num_stages
                ),
            )
        )

        selected_stage = int(
            rng.integers(
                0,
                num_stages,
            )
        )

        stages[
            user_index
        ] = (
            selected_stage + 1
        )

        x_indices[
            user_index
        ] = (
            sample_representative_category(
                stage_edges=stage_edges,
                selected_stage=(
                    selected_stage
                ),
                bot_index=(
                    public_domain.bot_index
                ),
                rng=rng,
            )
        )

    q_j, q_sigma = (
        batch_encode_indices(
            category_indices=q_indices,
            domain_size=len(
                public_domain.omega_q
            ),
            epsilon=epsilon_q,
            rng=rng,
        )
    )

    x_j, x_sigma = (
        batch_encode_indices(
            category_indices=x_indices,
            domain_size=len(
                public_domain.omega_x
            ),
            epsilon=epsilon_x,
            rng=rng,
        )
    )

    return PrivateReports(
        q_j=q_j,
        q_sigma=q_sigma,
        stage=stages,
        x_j=x_j,
        x_sigma=x_sigma,
        epsilon_q=epsilon_q,
        epsilon_x=epsilon_x,
        num_stages=num_stages,
        omega_q_size=len(
            public_domain.omega_q
        ),
        omega_x_size=len(
            public_domain.omega_x
        ),
    )


def save_reports(
    reports: PrivateReports,
    output_path: Path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        q_j=reports.q_j,
        q_sigma=reports.q_sigma,
        stage=reports.stage,
        x_j=reports.x_j,
        x_sigma=reports.x_sigma,
        epsilon_q=np.asarray(
            reports.epsilon_q,
            dtype=np.float64,
        ),
        epsilon_x=np.asarray(
            reports.epsilon_x,
            dtype=np.float64,
        ),
        num_stages=np.asarray(
            reports.num_stages,
            dtype=np.int32,
        ),
        omega_q_size=np.asarray(
            reports.omega_q_size,
            dtype=np.int64,
        ),
        omega_x_size=np.asarray(
            reports.omega_x_size,
            dtype=np.int64,
        ),
        stage_index_base=np.asarray(
            1,
            dtype=np.int8,
        ),
    )


def load_reports(
    input_path: Path,
) -> PrivateReports:
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Report file does not exist: {input_path}"
        )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:
        reports = PrivateReports(
            q_j=data["q_j"].astype(
                np.int64
            ),
            q_sigma=data[
                "q_sigma"
            ].astype(
                np.int8
            ),
            stage=data[
                "stage"
            ].astype(
                np.int16
            ),
            x_j=data[
                "x_j"
            ].astype(
                np.int64
            ),
            x_sigma=data[
                "x_sigma"
            ].astype(
                np.int8
            ),
            epsilon_q=float(
                data[
                    "epsilon_q"
                ]
            ),
            epsilon_x=float(
                data[
                    "epsilon_x"
                ]
            ),
            num_stages=int(
                data[
                    "num_stages"
                ]
            ),
            omega_q_size=int(
                data[
                    "omega_q_size"
                ]
            ),
            omega_x_size=int(
                data[
                    "omega_x_size"
                ]
            ),
        )

    return reports