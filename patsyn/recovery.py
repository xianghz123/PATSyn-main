import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from patsyn.domain import PublicDomain
from patsyn.hcr import attenuation
from patsyn.reporting import PrivateReports


@dataclass
class RecoveredModel:
    p_q_raw: np.ndarray
    p_q: np.ndarray

    f_raw: np.ndarray
    f: np.ndarray
    f_plus: np.ndarray
    nonempty_mass: np.ndarray

    kernels: np.ndarray
    stabilized_kernels: np.ndarray
    topology_kernel: np.ndarray

    epsilon_q: float
    epsilon_x: float
    num_stages: int
    lambda_value: float


def _parity_signs(
    row: int,
    columns: np.ndarray,
) -> np.ndarray:
    values = np.bitwise_and(
        np.uint64(row),
        columns.astype(
            np.uint64,
            copy=False,
        ),
    )

    values ^= values >> np.uint64(32)
    values ^= values >> np.uint64(16)
    values ^= values >> np.uint64(8)
    values ^= values >> np.uint64(4)

    values &= np.uint64(0xF)

    lookup = np.uint64(0x6996)

    parity = np.bitwise_and(
        np.right_shift(
            lookup,
            values,
        ),
        np.uint64(1),
    ).astype(
        np.int8
    )

    return (
        1 - 2 * parity
    ).astype(
        np.int8
    )


def _decode_category_sums(
    j: np.ndarray,
    sigma: np.ndarray,
    domain_size: int,
    epsilon: float,
    candidate_chunk_size: int = 64,
    report_chunk_size: int = 8192,
) -> np.ndarray:
    if domain_size <= 0:
        raise ValueError(
            "domain_size must be positive."
        )

    if len(j) != len(sigma):
        raise ValueError(
            "j and sigma must have equal length."
        )

    if len(j) == 0:
        return np.zeros(
            domain_size,
            dtype=np.float64,
        )

    alpha = attenuation(
        epsilon
    )

    output = np.zeros(
        domain_size,
        dtype=np.float64,
    )

    num_reports = len(j)

    for candidate_start in range(
        0,
        domain_size,
        candidate_chunk_size,
    ):
        candidate_end = min(
            candidate_start
            + candidate_chunk_size,
            domain_size,
        )

        rows = np.arange(
            candidate_start + 1,
            candidate_end + 1,
            dtype=np.uint64,
        )

        chunk_sum = np.zeros(
            len(rows),
            dtype=np.float64,
        )

        for report_start in range(
            0,
            num_reports,
            report_chunk_size,
        ):
            report_end = min(
                report_start
                + report_chunk_size,
                num_reports,
            )

            j_chunk = j[
                report_start:report_end
            ].astype(
                np.uint64,
                copy=False,
            )

            sigma_chunk = sigma[
                report_start:report_end
            ].astype(
                np.float64,
                copy=False,
            )

            values = np.bitwise_and(
                rows[:, None],
                j_chunk[None, :],
            )

            values ^= (
                values >> np.uint64(32)
            )
            values ^= (
                values >> np.uint64(16)
            )
            values ^= (
                values >> np.uint64(8)
            )
            values ^= (
                values >> np.uint64(4)
            )

            values &= np.uint64(
                0xF
            )

            lookup = np.uint64(
                0x6996
            )

            parity = np.bitwise_and(
                np.right_shift(
                    lookup,
                    values,
                ),
                np.uint64(1),
            ).astype(
                np.int8
            )

            signs = (
                1 - 2 * parity
            ).astype(
                np.float64
            )

            chunk_sum += (
                signs
                @ sigma_chunk
            )

        output[
            candidate_start:
            candidate_end
        ] = (
            chunk_sum / alpha
        )

    return output


def _normalize_probability(
    values: np.ndarray,
) -> np.ndarray:
    clipped = np.maximum(
        values,
        0.0,
    )

    total = float(
        clipped.sum()
    )

    if total > 0:
        return (
            clipped / total
        )

    return np.full(
        len(values),
        1.0 / len(values),
        dtype=np.float64,
    )


def _build_topology_kernel(
    public_domain: PublicDomain,
) -> np.ndarray:
    num_cells = (
        public_domain.grid.num_cells
    )

    kernel = np.zeros(
        (
            num_cells,
            num_cells,
        ),
        dtype=np.float64,
    )

    for source in range(
        num_cells
    ):
        neighbors = (
            public_domain.grid.neighbors(
                source
            )
        )

        if len(neighbors) == 0:
            continue

        probability = (
            1.0 / len(neighbors)
        )

        kernel[
            source,
            list(neighbors),
        ] = probability

    return kernel


def _build_transition_kernels(
    f_plus: np.ndarray,
    nonempty_mass: np.ndarray,
    public_domain: PublicDomain,
    lambda_value: float,
):
    if not (
        0.0 < lambda_value < 1.0
    ):
        raise ValueError(
            "lambda_value must lie in (0, 1)."
        )

    num_stages = (
        f_plus.shape[0]
    )

    num_cells = (
        public_domain.grid.num_cells
    )

    topology_kernel = (
        _build_topology_kernel(
            public_domain
        )
    )

    kernels = np.zeros(
        (
            num_stages,
            num_cells,
            num_cells,
        ),
        dtype=np.float64,
    )

    stabilized = np.zeros_like(
        kernels
    )

    for stage in range(
        num_stages
    ):
        for source in range(
            num_cells
        ):
            neighbors = (
                public_domain.grid.neighbors(
                    source
                )
            )

            if len(neighbors) == 0:
                continue

            use_topology = (
                nonempty_mass[stage]
                <= 0.0
            )

            if not use_topology:
                weights = np.zeros(
                    len(neighbors),
                    dtype=np.float64,
                )

                for index, target in enumerate(
                    neighbors
                ):
                    edge_index = (
                        public_domain.edge_to_index[
                            (
                                source,
                                target,
                            )
                        ]
                    )

                    weights[index] = (
                        f_plus[
                            stage,
                            edge_index,
                        ]
                    )

                total = float(
                    weights.sum()
                )

                if total > 0:
                    kernels[
                        stage,
                        source,
                        list(neighbors),
                    ] = (
                        weights / total
                    )
                else:
                    use_topology = True

            if use_topology:
                kernels[
                    stage,
                    source,
                    :,
                ] = topology_kernel[
                    source,
                    :,
                ]

            stabilized[
                stage,
                source,
                :,
            ] = (
                (1.0 - lambda_value)
                * kernels[
                    stage,
                    source,
                    :,
                ]
                + lambda_value
                * topology_kernel[
                    source,
                    :,
                ]
            )

    return (
        kernels,
        stabilized,
        topology_kernel,
    )


def recover_model(
    reports: PrivateReports,
    public_domain: PublicDomain,
    epsilon: float,
    rho: float,
    num_stages: int,
    lambda_value: float,
) -> RecoveredModel:
    if reports.num_users <= 0:
        raise ValueError(
            "No private reports provided."
        )

    if not (
        0.0 < rho < 1.0
    ):
        raise ValueError(
            "rho must lie in (0, 1)."
        )

    if num_stages != reports.num_stages:
        raise ValueError(
            "Stage-count mismatch."
        )

    expected_epsilon_q = (
        rho * epsilon
    )

    expected_epsilon_x = (
        (1.0 - rho)
        * epsilon
    )

    if not np.isclose(
        reports.epsilon_q,
        expected_epsilon_q,
    ):
        raise ValueError(
            "epsilon_q mismatch."
        )

    if not np.isclose(
        reports.epsilon_x,
        expected_epsilon_x,
    ):
        raise ValueError(
            "epsilon_x mismatch."
        )

    omega_q_size = len(
        public_domain.omega_q
    )

    omega_x_size = len(
        public_domain.omega_x
    )

    if (
        reports.omega_q_size
        != omega_q_size
    ):
        raise ValueError(
            "Omega_q size mismatch."
        )

    if (
        reports.omega_x_size
        != omega_x_size
    ):
        raise ValueError(
            "Omega_x size mismatch."
        )

    num_users = (
        reports.num_users
    )

    q_sum = (
        _decode_category_sums(
            j=reports.q_j,
            sigma=reports.q_sigma,
            domain_size=omega_q_size,
            epsilon=reports.epsilon_q,
        )
    )

    p_q_raw = (
        q_sum / num_users
    )

    f_raw = np.zeros(
        (
            num_stages,
            omega_x_size,
        ),
        dtype=np.float64,
    )

    for stage in range(
        1,
        num_stages + 1,
    ):
        mask = (
            reports.stage == stage
        )

        stage_sum = (
            _decode_category_sums(
                j=reports.x_j[mask],
                sigma=reports.x_sigma[mask],
                domain_size=omega_x_size,
                epsilon=reports.epsilon_x,
            )
        )

        f_raw[
            stage - 1
        ] = (
            num_stages
            * stage_sum
            / num_users
        )

    p_q = (
        _normalize_probability(
            p_q_raw
        )
    )

    f = np.zeros_like(
        f_raw
    )

    for stage in range(
        num_stages
    ):
        f[stage] = (
            _normalize_probability(
                f_raw[stage]
            )
        )

    bot_index = (
        public_domain.bot_index
    )

    nonempty_mass = (
        1.0 - f[:, bot_index]
    )

    f_plus = np.zeros_like(
        f
    )

    for stage in range(
        num_stages
    ):
        if (
            nonempty_mass[stage]
            > 0
        ):
            f_plus[
                stage,
                :,
            ] = (
                f[
                    stage,
                    :,
                ]
                / nonempty_mass[
                    stage
                ]
            )

            f_plus[
                stage,
                bot_index,
            ] = 0.0

    kernels, stabilized, topology = (
        _build_transition_kernels(
            f_plus=f_plus,
            nonempty_mass=(
                nonempty_mass
            ),
            public_domain=(
                public_domain
            ),
            lambda_value=(
                lambda_value
            ),
        )
    )

    return RecoveredModel(
        p_q_raw=p_q_raw,
        p_q=p_q,
        f_raw=f_raw,
        f=f,
        f_plus=f_plus,
        nonempty_mass=(
            nonempty_mass
        ),
        kernels=kernels,
        stabilized_kernels=(
            stabilized
        ),
        topology_kernel=(
            topology
        ),
        epsilon_q=(
            reports.epsilon_q
        ),
        epsilon_x=(
            reports.epsilon_x
        ),
        num_stages=(
            num_stages
        ),
        lambda_value=(
            lambda_value
        ),
    )


def save_model(
    model: RecoveredModel,
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
            model,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_model(
    input_path: Path,
) -> RecoveredModel:
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {input_path}"
        )

    with input_path.open(
        "rb"
    ) as file:
        model = pickle.load(
            file
        )

    if not isinstance(
        model,
        RecoveredModel,
    ):
        raise TypeError(
            "Invalid recovered-model file."
        )

    return model