from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from patsyn.domain import PublicDomain


Trajectory = Sequence[int]


def _normalize_counts(
    counts: np.ndarray,
) -> np.ndarray:
    counts = np.asarray(
        counts,
        dtype=np.float64,
    )

    total = float(
        counts.sum()
    )

    if total <= 0:
        return np.zeros_like(
            counts
        )

    return (
        counts / total
    )


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:
    p = _normalize_counts(
        p
    )

    q = _normalize_counts(
        q
    )

    m = 0.5 * (
        p + q
    )

    def kl_divergence(
        left,
        right,
    ):
        mask = left > 0

        if not np.any(
            mask
        ):
            return 0.0

        return float(
            np.sum(
                left[mask]
                * np.log2(
                    left[mask]
                    / right[mask]
                )
            )
        )

    return (
        0.5
        * kl_divergence(
            p,
            m,
        )
        + 0.5
        * kl_divergence(
            q,
            m,
        )
    )


def _evaluation_cell(
    cell: int,
    public_domain: PublicDomain,
    eval_rows: int,
    eval_cols: int,
) -> int:
    row, col = (
        public_domain.grid.cell_to_row_col(
            int(cell)
        )
    )

    eval_row = min(
        eval_rows - 1,
        row
        * eval_rows
        // public_domain.grid.rows,
    )

    eval_col = min(
        eval_cols - 1,
        col
        * eval_cols
        // public_domain.grid.cols,
    )

    return (
        eval_row * eval_cols
        + eval_col
    )


def _map_trajectory_to_evaluation_grid(
    trajectory: Trajectory,
    public_domain: PublicDomain,
    eval_rows: int,
    eval_cols: int,
) -> List[int]:
    return [
        _evaluation_cell(
            cell=cell,
            public_domain=public_domain,
            eval_rows=eval_rows,
            eval_cols=eval_cols,
        )
        for cell in trajectory
    ]


def _density_counts(
    trajectories: Sequence[
        Trajectory
    ],
    public_domain: PublicDomain,
    eval_rows: int,
    eval_cols: int,
) -> np.ndarray:
    counts = np.zeros(
        eval_rows * eval_cols,
        dtype=np.float64,
    )

    for trajectory in trajectories:
        mapped = (
            _map_trajectory_to_evaluation_grid(
                trajectory,
                public_domain,
                eval_rows,
                eval_cols,
            )
        )

        for cell in mapped:
            counts[cell] += 1

    return counts


def density_error(
    real_trajectories,
    synthetic_trajectories,
    public_domain,
    eval_rows,
    eval_cols,
) -> float:
    real = _normalize_counts(
        _density_counts(
            real_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    )

    synthetic = _normalize_counts(
        _density_counts(
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    )

    return float(
        np.mean(
            np.abs(
                real - synthetic
            )
        )
    )


def _random_rectangles(
    rows: int,
    cols: int,
    num_queries: int,
    seed: int,
):
    rng = np.random.default_rng(
        seed
    )

    rectangles = []

    for _ in range(
        num_queries
    ):
        r1, r2 = sorted(
            rng.integers(
                0,
                rows,
                size=2,
            )
        )

        c1, c2 = sorted(
            rng.integers(
                0,
                cols,
                size=2,
            )
        )

        rectangles.append(
            (
                int(r1),
                int(r2),
                int(c1),
                int(c2),
            )
        )

    return rectangles


def range_query_error(
    real_trajectories,
    synthetic_trajectories,
    public_domain,
    eval_rows,
    eval_cols,
    num_queries,
    seed,
) -> float:
    real = _normalize_counts(
        _density_counts(
            real_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    ).reshape(
        eval_rows,
        eval_cols,
    )

    synthetic = _normalize_counts(
        _density_counts(
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    ).reshape(
        eval_rows,
        eval_cols,
    )

    errors = []

    for (
        r1,
        r2,
        c1,
        c2,
    ) in _random_rectangles(
        eval_rows,
        eval_cols,
        num_queries,
        seed,
    ):
        real_answer = float(
            real[
                r1:r2 + 1,
                c1:c2 + 1,
            ].sum()
        )

        synthetic_answer = float(
            synthetic[
                r1:r2 + 1,
                c1:c2 + 1,
            ].sum()
        )

        errors.append(
            abs(
                real_answer
                - synthetic_answer
            )
        )

    return float(
        np.mean(errors)
    )


def _dcg(
    relevances: Sequence[float],
) -> float:
    value = 0.0

    for rank, relevance in enumerate(
        relevances,
        start=1,
    ):
        value += (
            float(relevance)
            / np.log2(
                rank + 1
            )
        )

    return value


def hotspot_ndcg(
    real_counts: np.ndarray,
    synthetic_counts: np.ndarray,
    k: int,
) -> float:
    k = min(
        k,
        len(real_counts),
    )

    synthetic_rank = np.argsort(
        -synthetic_counts
    )[:k]

    ideal_rank = np.argsort(
        -real_counts
    )[:k]

    observed = [
        real_counts[index]
        for index
        in synthetic_rank
    ]

    ideal = [
        real_counts[index]
        for index
        in ideal_rank
    ]

    ideal_dcg = _dcg(
        ideal
    )

    if ideal_dcg <= 0:
        return 1.0

    return float(
        _dcg(observed)
        / ideal_dcg
    )


def kendall_tau(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    concordant = 0
    discordant = 0

    n = len(first)

    for i in range(n):
        for j in range(
            i + 1,
            n,
        ):
            left = (
                first[i]
                - first[j]
            )

            right = (
                second[i]
                - second[j]
            )

            product = (
                left * right
            )

            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1

    denominator = (
        concordant
        + discordant
    )

    if denominator == 0:
        return 0.0

    return float(
        (
            concordant
            - discordant
        )
        / denominator
    )


def _trip_distribution(
    trajectories,
    public_domain,
    eval_rows,
    eval_cols,
):
    size = (
        eval_rows * eval_cols
    )

    counts = np.zeros(
        size * size,
        dtype=np.float64,
    )

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue

        source = (
            _evaluation_cell(
                trajectory[0],
                public_domain,
                eval_rows,
                eval_cols,
            )
        )

        destination = (
            _evaluation_cell(
                trajectory[-1],
                public_domain,
                eval_rows,
                eval_cols,
            )
        )

        counts[
            source * size
            + destination
        ] += 1

    return _normalize_counts(
        counts
    )


def trip_error(
    real_trajectories,
    synthetic_trajectories,
    public_domain,
    eval_rows,
    eval_cols,
) -> float:
    real = _trip_distribution(
        real_trajectories,
        public_domain,
        eval_rows,
        eval_cols,
    )

    synthetic = (
        _trip_distribution(
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    )

    return float(
        np.mean(
            np.abs(
                real - synthetic
            )
        )
    )


def _length_distribution(
    trajectories,
    l_max,
):
    counts = np.zeros(
        l_max,
        dtype=np.float64,
    )

    for trajectory in trajectories:
        length = (
            len(trajectory) - 1
        )

        if (
            1 <= length <= l_max
        ):
            counts[
                length - 1
            ] += 1

    return _normalize_counts(
        counts
    )


def length_error(
    real_trajectories,
    synthetic_trajectories,
    l_max,
) -> float:
    real = _length_distribution(
        real_trajectories,
        l_max,
    )

    synthetic = (
        _length_distribution(
            synthetic_trajectories,
            l_max,
        )
    )

    return float(
        np.mean(
            np.abs(
                real - synthetic
            )
        )
    )


def _trajectory_diameter(
    trajectory,
    public_domain,
    eval_rows,
    eval_cols,
):
    mapped = (
        _map_trajectory_to_evaluation_grid(
            trajectory,
            public_domain,
            eval_rows,
            eval_cols,
        )
    )

    coordinates = [
        divmod(
            cell,
            eval_cols,
        )
        for cell in mapped
    ]

    rows = [
        value[0]
        for value in coordinates
    ]

    cols = [
        value[1]
        for value in coordinates
    ]

    row_span = (
        max(rows) - min(rows)
    )

    col_span = (
        max(cols) - min(cols)
    )

    return float(
        np.sqrt(
            row_span ** 2
            + col_span ** 2
        )
    )


def diameter_error(
    real_trajectories,
    synthetic_trajectories,
    public_domain,
    eval_rows,
    eval_cols,
) -> float:
    max_diameter = float(
        np.sqrt(
            (eval_rows - 1) ** 2
            + (eval_cols - 1) ** 2
        )
    )

    if max_diameter <= 0:
        return 0.0

    real = np.asarray(
        [
            _trajectory_diameter(
                trajectory,
                public_domain,
                eval_rows,
                eval_cols,
            )
            for trajectory
            in real_trajectories
        ],
        dtype=np.float64,
    )

    synthetic = np.asarray(
        [
            _trajectory_diameter(
                trajectory,
                public_domain,
                eval_rows,
                eval_cols,
            )
            for trajectory
            in synthetic_trajectories
        ],
        dtype=np.float64,
    )

    return float(
        abs(
            real.mean()
            - synthetic.mean()
        )
        / max_diameter
    )


def _top_patterns(
    trajectories,
    lengths,
    top_k,
):
    counts = Counter()

    for trajectory in trajectories:
        values = tuple(
            trajectory
        )

        for pattern_length in lengths:
            if (
                len(values)
                < pattern_length
            ):
                continue

            for start in range(
                len(values)
                - pattern_length
                + 1
            ):
                pattern = values[
                    start:
                    start + pattern_length
                ]

                counts[
                    pattern
                ] += 1

    return {
        pattern
        for pattern, _
        in counts.most_common(
            top_k
        )
    }


def pattern_f1(
    real_trajectories,
    synthetic_trajectories,
    lengths,
    top_k,
) -> float:
    real = _top_patterns(
        real_trajectories,
        lengths,
        top_k,
    )

    synthetic = _top_patterns(
        synthetic_trajectories,
        lengths,
        top_k,
    )

    if not real and not synthetic:
        return 1.0

    if not real or not synthetic:
        return 0.0

    overlap = len(
        real.intersection(
            synthetic
        )
    )

    precision = (
        overlap
        / len(synthetic)
    )

    recall = (
        overlap
        / len(real)
    )

    if (
        precision + recall
        <= 0
    ):
        return 0.0

    return float(
        2.0
        * precision
        * recall
        / (
            precision + recall
        )
    )


def _cell_distribution(
    trajectories,
    position,
    num_cells,
):
    counts = np.zeros(
        num_cells,
        dtype=np.float64,
    )

    for trajectory in trajectories:
        if not trajectory:
            continue

        cell = (
            trajectory[0]
            if position == "start"
            else trajectory[-1]
        )

        counts[
            int(cell)
        ] += 1

    return counts


def _joint_pattern_counts(
    trajectories,
    public_domain,
):
    counts = np.zeros(
        len(
            public_domain.omega_q
        ),
        dtype=np.float64,
    )

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue

        length = (
            len(trajectory) - 1
        )

        try:
            bin_index = (
                public_domain.length_bin_index(
                    length
                )
            )
        except ValueError:
            continue

        q = (
            int(trajectory[0]),
            int(trajectory[-1]),
            int(bin_index),
        )

        index = (
            public_domain.q_to_index.get(
                q
            )
        )

        if index is not None:
            counts[
                index
            ] += 1

    return counts


def _transition_counts(
    trajectories,
    public_domain,
):
    counts = np.zeros(
        len(
            public_domain.grid.edges
        ),
        dtype=np.float64,
    )

    for trajectory in trajectories:
        for source, target in zip(
            trajectory[:-1],
            trajectory[1:],
        ):
            index = (
                public_domain.edge_to_index.get(
                    (
                        int(source),
                        int(target),
                    )
                )
            )

            if index is not None:
                counts[
                    index
                ] += 1

    return counts


def _stage_distribution(
    trajectories,
    public_domain,
    num_stages,
):
    omega_x_size = len(
        public_domain.omega_x
    )

    counts = np.zeros(
        (
            num_stages,
            omega_x_size,
        ),
        dtype=np.float64,
    )

    num_trajectories = len(
        trajectories
    )

    if num_trajectories == 0:
        return counts

    for trajectory in trajectories:
        length = (
            len(trajectory) - 1
        )

        stage_edges = [
            []
            for _ in range(
                num_stages
            )
        ]

        for t, (
            source,
            target,
        ) in enumerate(
            zip(
                trajectory[:-1],
                trajectory[1:],
            ),
            start=1,
        ):
            stage = (
                num_stages * t - 1
            ) // length

            index = (
                public_domain.edge_to_index[
                    (
                        int(source),
                        int(target),
                    )
                ]
            )

            stage_edges[
                stage
            ].append(
                index
            )

        for stage in range(
            num_stages
        ):
            edges = stage_edges[
                stage
            ]

            if not edges:
                counts[
                    stage,
                    public_domain.bot_index,
                ] += 1.0
                continue

            weight = (
                1.0 / len(edges)
            )

            for edge_index in edges:
                counts[
                    stage,
                    edge_index,
                ] += weight

    return (
        counts / num_trajectories
    )


def evaluate_structural(
    real_trajectories,
    synthetic_trajectories,
    public_domain,
    num_stages,
):
    num_cells = (
        public_domain.grid.num_cells
    )

    real_start = (
        _cell_distribution(
            real_trajectories,
            "start",
            num_cells,
        )
    )

    synthetic_start = (
        _cell_distribution(
            synthetic_trajectories,
            "start",
            num_cells,
        )
    )

    real_destination = (
        _cell_distribution(
            real_trajectories,
            "destination",
            num_cells,
        )
    )

    synthetic_destination = (
        _cell_distribution(
            synthetic_trajectories,
            "destination",
            num_cells,
        )
    )

    real_length = (
        _length_distribution(
            real_trajectories,
            public_domain.L_max,
        )
    )

    synthetic_length = (
        _length_distribution(
            synthetic_trajectories,
            public_domain.L_max,
        )
    )

    real_joint = (
        _joint_pattern_counts(
            real_trajectories,
            public_domain,
        )
    )

    synthetic_joint = (
        _joint_pattern_counts(
            synthetic_trajectories,
            public_domain,
        )
    )

    real_transition = (
        _transition_counts(
            real_trajectories,
            public_domain,
        )
    )

    synthetic_transition = (
        _transition_counts(
            synthetic_trajectories,
            public_domain,
        )
    )

    real_stage = (
        _stage_distribution(
            real_trajectories,
            public_domain,
            num_stages,
        )
    )

    synthetic_stage = (
        _stage_distribution(
            synthetic_trajectories,
            public_domain,
            num_stages,
        )
    )

    stage_values = [
        js_divergence(
            real_stage[stage],
            synthetic_stage[stage],
        )
        for stage in range(
            num_stages
        )
    ]

    return {
        "start_js": js_divergence(
            real_start,
            synthetic_start,
        ),
        "destination_js": js_divergence(
            real_destination,
            synthetic_destination,
        ),
        "length_js": js_divergence(
            real_length,
            synthetic_length,
        ),
        "joint_js": js_divergence(
            real_joint,
            synthetic_joint,
        ),
        "transition_js": js_divergence(
            real_transition,
            synthetic_transition,
        ),
        "stage_js": float(
            np.mean(
                stage_values
            )
        ),
    }


def evaluate_constraints(
    synthetic_trajectories,
    sampled_conditions,
    public_domain,
):
    num_trajectories = len(
        synthetic_trajectories
    )

    if num_trajectories == 0:
        raise ValueError(
            "No synthetic trajectories."
        )

    if (
        len(
            sampled_conditions[
                "source"
            ]
        )
        != num_trajectories
    ):
        raise ValueError(
            "Condition count mismatch."
        )

    destination_success = 0
    exact_length_success = 0
    adjacency_success = 0

    for index, trajectory in enumerate(
        synthetic_trajectories
    ):
        expected_destination = int(
            sampled_conditions[
                "destination"
            ][index]
        )

        expected_length = int(
            sampled_conditions[
                "length"
            ][index]
        )

        if (
            int(trajectory[-1])
            == expected_destination
        ):
            destination_success += 1

        if (
            len(trajectory) - 1
            == expected_length
        ):
            exact_length_success += 1

        legal = all(
            public_domain.grid.is_legal_edge(
                int(source),
                int(target),
            )
            for source, target
            in zip(
                trajectory[:-1],
                trajectory[1:],
            )
        )

        if legal:
            adjacency_success += 1

    return {
        "destination_sr": (
            destination_success
            / num_trajectories
        ),
        "exact_length_sr": (
            exact_length_success
            / num_trajectories
        ),
        "adjacency_sr": (
            adjacency_success
            / num_trajectories
        ),
    }


def evaluate_general(
    real_trajectories,
    synthetic_trajectories,
    public_domain,
    config,
    seed,
):
    evaluation = config[
        "evaluation"
    ]

    eval_rows = int(
        evaluation[
            "evaluation_grid"
        ]["rows"]
    )

    eval_cols = int(
        evaluation[
            "evaluation_grid"
        ]["cols"]
    )

    real_density = (
        _density_counts(
            real_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    )

    synthetic_density = (
        _density_counts(
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        )
    )

    return {
        "density_error": density_error(
            real_trajectories,
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        ),
        "query_error": range_query_error(
            real_trajectories,
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
            int(
                evaluation[
                    "num_range_queries"
                ]
            ),
            seed,
        ),
        "hotspot_ndcg": hotspot_ndcg(
            real_density,
            synthetic_density,
            int(
                evaluation[
                    "top_k_hotspots"
                ]
            ),
        ),
        "kendall_tau": kendall_tau(
            real_density,
            synthetic_density,
        ),
        "trip_error": trip_error(
            real_trajectories,
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        ),
        "length_error": length_error(
            real_trajectories,
            synthetic_trajectories,
            public_domain.L_max,
        ),
        "diameter_error": diameter_error(
            real_trajectories,
            synthetic_trajectories,
            public_domain,
            eval_rows,
            eval_cols,
        ),
        "pattern_f1": pattern_f1(
            real_trajectories,
            synthetic_trajectories,
            lengths=tuple(
                int(value)
                for value
                in evaluation[
                    "pattern_lengths"
                ]
            ),
            top_k=int(
                evaluation[
                    "top_k_patterns"
                ]
            ),
        ),
    }


def evaluate_all(
    real_trajectories,
    synthetic_trajectories,
    sampled_conditions,
    public_domain,
    config,
    seed,
):
    return {
        "general": evaluate_general(
            real_trajectories=(
                real_trajectories
            ),
            synthetic_trajectories=(
                synthetic_trajectories
            ),
            public_domain=(
                public_domain
            ),
            config=config,
            seed=seed,
        ),
        "structural": evaluate_structural(
            real_trajectories=(
                real_trajectories
            ),
            synthetic_trajectories=(
                synthetic_trajectories
            ),
            public_domain=(
                public_domain
            ),
            num_stages=int(
                config[
                    "model"
                ]["R"]
            ),
        ),
        "constraints": evaluate_constraints(
            synthetic_trajectories=(
                synthetic_trajectories
            ),
            sampled_conditions=(
                sampled_conditions
            ),
            public_domain=(
                public_domain
            ),
        ),
    }