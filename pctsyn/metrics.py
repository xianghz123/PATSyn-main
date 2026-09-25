from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from pctsyn.domain import PublicDomain

Trajectory = Sequence[int]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values)
    return values / total


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)

    def kl(left, right):
        mask = left > 0
        if not np.any(mask):
            return 0.0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _evaluation_cell(
    cell: int,
    public_domain: PublicDomain,
    eval_rows: int,
    eval_cols: int,
) -> int:
    row, col = public_domain.grid.cell_to_row_col(int(cell))
    eval_row = min(eval_rows - 1, row * eval_rows // public_domain.grid.rows)
    eval_col = min(eval_cols - 1, col * eval_cols // public_domain.grid.cols)
    return eval_row * eval_cols + eval_col


def _map_trajectory(
    trajectory: Trajectory,
    public_domain: PublicDomain,
    eval_rows: int,
    eval_cols: int,
) -> List[int]:
    mapped = [
        _evaluation_cell(cell, public_domain, eval_rows, eval_cols)
        for cell in trajectory
    ]

    if not mapped:
        return []

    output = [mapped[0]]
    for cell in mapped[1:]:
        if cell != output[-1]:
            output.append(cell)
    return output


def _map_database(
    trajectories: Sequence[Trajectory],
    public_domain: PublicDomain,
    eval_rows: int,
    eval_cols: int,
) -> List[List[int]]:
    return [
        _map_trajectory(t, public_domain, eval_rows, eval_cols)
        for t in trajectories
        if len(t) > 0
    ]


def _cell_center(cell: int, rows: int, cols: int) -> Tuple[float, float]:
    row, col = divmod(int(cell), cols)
    return ((col + 0.5) / cols, (row + 0.5) / rows)


def _point_density(trajectories: Sequence[Trajectory], num_cells: int) -> np.ndarray:
    counts = np.zeros(num_cells, dtype=np.float64)
    for trajectory in trajectories:
        for cell in trajectory:
            counts[int(cell)] += 1.0
    return counts


def density_error(real, synthetic, num_cells) -> float:
    return js_divergence(
        _point_density(real, num_cells),
        _point_density(synthetic, num_cells),
    )


def _range_queries(
    rows: int,
    cols: int,
    num_queries: int,
    size_factor: float,
    seed: int,
):
    if size_factor <= 0:
        raise ValueError("size_factor must be positive.")

    rng = np.random.default_rng(seed)
    edge = np.sqrt(1.0 / size_factor)
    queries = []

    for _ in range(num_queries):
        center_x = float(rng.random())
        center_y = float(rng.random())
        half = edge / 2.0
        queries.append(
            (
                center_x - half,
                center_x + half,
                center_y - half,
                center_y + half,
            )
        )

    return queries


def _query_count(trajectories, query, rows, cols) -> int:
    left, right, bottom, top = query
    count = 0

    for trajectory in trajectories:
        for cell in trajectory:
            x, y = _cell_center(cell, rows, cols)
            if left <= x <= right and bottom <= y <= top:
                count += 1

    return count


def range_query_error(
    real,
    synthetic,
    rows,
    cols,
    num_queries,
    size_factor,
    sanity_bound,
    seed,
) -> float:
    queries = _range_queries(rows, cols, num_queries, size_factor, seed)
    total_points = sum(len(t) for t in real)
    errors = []

    for query in queries:
        actual = _query_count(real, query, rows, cols)
        generated = _query_count(synthetic, query, rows, cols)
        denominator = max(actual, total_points * sanity_bound)
        errors.append(abs(actual - generated) / denominator)

    return float(np.mean(errors))


def hotspot_ndcg(real_density, synthetic_density, k=5) -> float:
    k = min(k, len(real_density))
    real_rank = np.argsort(-np.asarray(real_density))[:k].tolist()
    synthetic_rank = np.argsort(-np.asarray(synthetic_density))[:k].tolist()

    relevance = np.zeros(k, dtype=np.float64)
    for index, cell in enumerate(synthetic_rank):
        if cell in real_rank:
            relevance[index] = 1.0 / (real_rank.index(cell) + 1)

    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = (1.0 / np.arange(1, k + 1)) * discount
    idcg = float(np.sum(ideal))
    dcg = float(np.sum(relevance * discount))
    return dcg / idcg if idcg > 0 else 0.0


def coverage_kendall_tau(real, synthetic, num_cells) -> float:
    real_counts = np.zeros(num_cells, dtype=np.float64)
    synthetic_counts = np.zeros(num_cells, dtype=np.float64)

    for trajectory in real:
        for cell in set(trajectory):
            real_counts[int(cell)] += 1.0

    for trajectory in synthetic:
        for cell in set(trajectory):
            synthetic_counts[int(cell)] += 1.0

    concordant = 0
    reversed_pairs = 0

    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            if real_counts[i] > real_counts[j]:
                if synthetic_counts[i] > synthetic_counts[j]:
                    concordant += 1
                else:
                    reversed_pairs += 1
            elif real_counts[i] < real_counts[j]:
                if synthetic_counts[i] < synthetic_counts[j]:
                    concordant += 1
                else:
                    reversed_pairs += 1

    denominator = num_cells * (num_cells - 1) / 2
    if denominator <= 0:
        return 0.0
    return float((concordant - reversed_pairs) / denominator)


def _trip_distribution(trajectories, num_cells) -> np.ndarray:
    counts = np.zeros(num_cells * num_cells, dtype=np.float64)
    for trajectory in trajectories:
        if not trajectory:
            continue
        source = int(trajectory[0])
        destination = int(trajectory[-1])
        counts[source * num_cells + destination] += 1.0
    return counts


def trip_error(real, synthetic, num_cells) -> float:
    return js_divergence(
        _trip_distribution(real, num_cells),
        _trip_distribution(synthetic, num_cells),
    )


def _travel_distance(trajectory, rows, cols) -> float:
    total = 0.0
    for left, right in zip(trajectory[:-1], trajectory[1:]):
        x1, y1 = _cell_center(left, rows, cols)
        x2, y2 = _cell_center(right, rows, cols)
        total += float(np.hypot(x1 - x2, y1 - y2))
    return total


def _diameter(trajectory, rows, cols) -> float:
    if len(trajectory) < 2:
        return 0.0

    points = np.asarray(
        [_cell_center(cell, rows, cols) for cell in trajectory],
        dtype=np.float64,
    )
    max_distance = 0.0

    for index in range(len(points)):
        distances = np.sqrt(np.sum((points[index + 1 :] - points[index]) ** 2, axis=1))
        if len(distances):
            max_distance = max(max_distance, float(np.max(distances)))

    return max_distance


def _histogram_js(real_values, synthetic_values, bucket_num=20) -> float:
    real_values = np.asarray(real_values, dtype=np.float64)
    synthetic_values = np.asarray(synthetic_values, dtype=np.float64)

    if len(real_values) == 0 or len(synthetic_values) == 0:
        return 0.0

    lower = float(np.min(real_values))
    upper = float(np.max(real_values))

    if np.isclose(lower, upper):
        return 0.0 if np.allclose(synthetic_values, lower) else 1.0

    edges = np.linspace(lower, upper, bucket_num + 1)
    real_hist, _ = np.histogram(real_values, bins=edges)
    synthetic_hist, _ = np.histogram(
        np.clip(synthetic_values, lower, upper),
        bins=edges,
    )
    return js_divergence(real_hist, synthetic_hist)


def length_error(real, synthetic, rows, cols, bucket_num=20) -> float:
    real_values = [_travel_distance(t, rows, cols) for t in real]
    synthetic_values = [_travel_distance(t, rows, cols) for t in synthetic]
    return _histogram_js(real_values, synthetic_values, bucket_num)


def diameter_error(real, synthetic, rows, cols, bucket_num=20) -> float:
    real_values = [_diameter(t, rows, cols) for t in real]
    synthetic_values = [_diameter(t, rows, cols) for t in synthetic]
    return _histogram_js(real_values, synthetic_values, bucket_num)


def _mine_patterns(trajectories, pattern_lengths) -> Counter:
    counts = Counter()
    for trajectory in trajectories:
        values = tuple(int(v) for v in trajectory)
        for length in pattern_lengths:
            for start in range(max(0, len(values) - length + 1)):
                counts[values[start : start + length]] += 1
    return counts


def pattern_f1(real, synthetic, pattern_lengths, top_k=100) -> float:
    real_patterns = _mine_patterns(real, pattern_lengths)
    synthetic_patterns = _mine_patterns(synthetic, pattern_lengths)

    real_top = [pattern for pattern, _ in real_patterns.most_common(top_k)]
    synthetic_top = [pattern for pattern, _ in synthetic_patterns.most_common(top_k)]

    if not real_top and not synthetic_top:
        return 1.0
    if not real_top or not synthetic_top:
        return 0.0

    overlap = len(set(real_top).intersection(synthetic_top))
    denominator = float(top_k)
    precision = overlap / denominator
    recall = overlap / denominator

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _length_count_distribution(trajectories) -> np.ndarray:
    max_length = max((len(t) - 1 for t in trajectories), default=0)
    counts = np.zeros(max_length + 1, dtype=np.float64)
    for trajectory in trajectories:
        counts[len(trajectory) - 1] += 1.0
    return counts


def _align_vectors(left: np.ndarray, right: np.ndarray):
    size = max(len(left), len(right))
    left_new = np.zeros(size, dtype=np.float64)
    right_new = np.zeros(size, dtype=np.float64)
    left_new[: len(left)] = left
    right_new[: len(right)] = right
    return left_new, right_new


def _joint_distribution(trajectories, num_cells, length_bins) -> np.ndarray:
    counts = np.zeros(num_cells * num_cells * len(length_bins), dtype=np.float64)

    for trajectory in trajectories:
        if len(trajectory) < 2:
            continue
        length = len(trajectory) - 1
        bin_index = None
        for index, values in enumerate(length_bins):
            if length in values:
                bin_index = index
                break
        if bin_index is None:
            continue

        source = int(trajectory[0])
        destination = int(trajectory[-1])
        index = (source * num_cells + destination) * len(length_bins) + bin_index
        counts[index] += 1.0

    return counts


def _transition_distribution(trajectories, num_cells) -> np.ndarray:
    counts = np.zeros(num_cells * num_cells, dtype=np.float64)
    for trajectory in trajectories:
        for source, destination in zip(trajectory[:-1], trajectory[1:]):
            counts[int(source) * num_cells + int(destination)] += 1.0
    return counts


def _stage_distributions(trajectories, num_cells, num_stages) -> np.ndarray:
    bot_index = num_cells * num_cells
    counts = np.zeros((num_stages, bot_index + 1), dtype=np.float64)

    if not trajectories:
        return counts

    for trajectory in trajectories:
        length = len(trajectory) - 1
        stage_edges = [[] for _ in range(num_stages)]

        if length > 0:
            for t, (source, destination) in enumerate(
                zip(trajectory[:-1], trajectory[1:]), start=1
            ):
                stage = (num_stages * t - 1) // length
                edge_index = int(source) * num_cells + int(destination)
                stage_edges[stage].append(edge_index)

        for stage in range(num_stages):
            edges = stage_edges[stage]
            if not edges:
                counts[stage, bot_index] += 1.0
            else:
                weight = 1.0 / len(edges)
                for edge_index in edges:
                    counts[stage, edge_index] += weight

    counts /= len(trajectories)
    return counts


def evaluate_general(real, synthetic, config, seed) -> Dict[str, float]:
    evaluation = config["evaluation"]
    rows = int(evaluation["evaluation_grid"]["rows"])
    cols = int(evaluation["evaluation_grid"]["cols"])
    num_cells = rows * cols

    real_density = _point_density(real, num_cells)
    synthetic_density = _point_density(synthetic, num_cells)

    query_seed = int(evaluation.get("query_seed", 2022))
    query_size_factor = float(evaluation.get("query_size_factor", 9.0))
    sanity_bound = float(evaluation.get("query_sanity_bound", 0.01))
    bucket_num = int(evaluation.get("distribution_buckets", 20))

    return {
        "density_error": density_error(real, synthetic, num_cells),
        "query_error": range_query_error(
            real,
            synthetic,
            rows,
            cols,
            int(evaluation["num_range_queries"]),
            query_size_factor,
            sanity_bound,
            query_seed,
        ),
        "hotspot_ndcg": hotspot_ndcg(
            real_density,
            synthetic_density,
            int(evaluation["top_k_hotspots"]),
        ),
        "kendall_tau": coverage_kendall_tau(real, synthetic, num_cells),
        "trip_error": trip_error(real, synthetic, num_cells),
        "length_error": length_error(real, synthetic, rows, cols, bucket_num),
        "diameter_error": diameter_error(real, synthetic, rows, cols, bucket_num),
        "pattern_f1": pattern_f1(
            real,
            synthetic,
            [int(value) for value in evaluation["pattern_lengths"]],
            int(evaluation["top_k_patterns"]),
        ),
    }


def evaluate_structural(real, synthetic, public_domain, config) -> Dict[str, float]:
    evaluation = config["evaluation"]
    rows = int(evaluation["evaluation_grid"]["rows"])
    cols = int(evaluation["evaluation_grid"]["cols"])
    num_cells = rows * cols
    num_stages = int(config["model"]["R"])

    start_real = np.zeros(num_cells, dtype=np.float64)
    start_syn = np.zeros(num_cells, dtype=np.float64)
    destination_real = np.zeros(num_cells, dtype=np.float64)
    destination_syn = np.zeros(num_cells, dtype=np.float64)

    for trajectory in real:
        if trajectory:
            start_real[int(trajectory[0])] += 1
            destination_real[int(trajectory[-1])] += 1

    for trajectory in synthetic:
        if trajectory:
            start_syn[int(trajectory[0])] += 1
            destination_syn[int(trajectory[-1])] += 1

    length_real, length_syn = _align_vectors(
        _length_count_distribution(real),
        _length_count_distribution(synthetic),
    )

    real_stage = _stage_distributions(real, num_cells, num_stages)
    synthetic_stage = _stage_distributions(synthetic, num_cells, num_stages)

    return {
        "start_js": js_divergence(start_real, start_syn),
        "destination_js": js_divergence(destination_real, destination_syn),
        "length_js": js_divergence(length_real, length_syn),
        "joint_js": js_divergence(
            _joint_distribution(real, num_cells, public_domain.length_bins),
            _joint_distribution(synthetic, num_cells, public_domain.length_bins),
        ),
        "transition_js": js_divergence(
            _transition_distribution(real, num_cells),
            _transition_distribution(synthetic, num_cells),
        ),
        "stage_js": float(
            np.mean(
                [
                    js_divergence(real_stage[r], synthetic_stage[r])
                    for r in range(num_stages)
                ]
            )
        ),
    }


def evaluate_constraints(synthetic, sampled_conditions, public_domain) -> Dict[str, float]:
    num_trajectories = len(synthetic)
    if num_trajectories == 0:
        raise ValueError("No synthetic trajectories.")

    if len(sampled_conditions["destination"]) != num_trajectories:
        raise ValueError("Condition count mismatch.")

    destination_success = 0
    exact_length_success = 0
    adjacency_success = 0

    for index, trajectory in enumerate(synthetic):
        if int(trajectory[-1]) == int(sampled_conditions["destination"][index]):
            destination_success += 1
        if len(trajectory) - 1 == int(sampled_conditions["length"][index]):
            exact_length_success += 1

        legal = all(
            public_domain.grid.is_legal_edge(int(source), int(destination))
            for source, destination in zip(trajectory[:-1], trajectory[1:])
        )
        adjacency_success += int(legal)

    return {
        "destination_sr": destination_success / num_trajectories,
        "exact_length_sr": exact_length_success / num_trajectories,
        "adjacency_sr": adjacency_success / num_trajectories,
    }


def evaluate_all(
    real_trajectories,
    synthetic_trajectories,
    sampled_conditions,
    public_domain,
    config,
    seed,
):
    evaluation = config["evaluation"]
    rows = int(evaluation["evaluation_grid"]["rows"])
    cols = int(evaluation["evaluation_grid"]["cols"])

    real_eval = _map_database(real_trajectories, public_domain, rows, cols)
    synthetic_eval = _map_database(synthetic_trajectories, public_domain, rows, cols)

    return {
        "general": evaluate_general(real_eval, synthetic_eval, config, seed),
        "structural": evaluate_structural(
            real_eval,
            synthetic_eval,
            public_domain,
            config,
        ),
        "constraints": evaluate_constraints(
            synthetic_trajectories,
            sampled_conditions,
            public_domain,
        ),
    }
