import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from patsyn.domain import GridGraph, build_grid_graph


Point = Tuple[float, float]
Trajectory = List[int]


def _resolve_raw_file(config: dict) -> Path:
    raw_dir = Path(config["paths"]["raw_dir"])

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw-data directory does not exist: {raw_dir}"
        )

    dataset_cfg = config.get("dataset", {})
    raw_file = dataset_cfg.get("raw_file")

    if raw_file:
        path = raw_dir / raw_file
        if not path.exists():
            raise FileNotFoundError(
                f"Raw trajectory file does not exist: {path}"
            )
        return path

    preferred = [
        raw_dir / "train.csv",
        raw_dir / "porto.csv",
        raw_dir / "trajectories.csv",
    ]

    for path in preferred:
        if path.exists():
            return path

    csv_files = sorted(raw_dir.glob("*.csv"))

    if len(csv_files) == 1:
        return csv_files[0]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found under {raw_dir}"
        )

    raise ValueError(
        "Multiple CSV files found. "
        "Set dataset.raw_file in the dataset configuration."
    )


def _parse_polyline(value: str) -> List[Point]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return []

    points: List[Point] = []

    if not isinstance(data, list):
        return points

    for item in data:
        if (
            not isinstance(item, (list, tuple))
            or len(item) < 2
        ):
            continue

        try:
            lon = float(item[0])
            lat = float(item[1])
        except (TypeError, ValueError):
            continue

        points.append((lon, lat))

    return points


def iter_porto_csv(
    path: Path,
    polyline_column: str = "POLYLINE",
) -> Iterator[List[Point]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {path}"
            )

        if polyline_column not in reader.fieldnames:
            raise KeyError(
                f"Column '{polyline_column}' not found in {path}"
            )

        for row in reader:
            yield _parse_polyline(
                row.get(polyline_column, "")
            )


def iter_jsonl(
    path: Path,
    trajectory_key: str = "trajectory",
) -> Iterator[List[Point]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            data = record.get(trajectory_key, [])

            points: List[Point] = []

            for item in data:
                if (
                    not isinstance(item, (list, tuple))
                    or len(item) < 2
                ):
                    continue

                points.append(
                    (float(item[0]), float(item[1]))
                )

            yield points


def iter_raw_trajectories(
    config: dict,
) -> Iterator[List[Point]]:
    path = _resolve_raw_file(config)

    dataset_cfg = config.get("dataset", {})
    data_format = dataset_cfg.get(
        "format",
        "porto_csv",
    )

    if data_format == "porto_csv":
        column = dataset_cfg.get(
            "polyline_column",
            "POLYLINE",
        )
        yield from iter_porto_csv(
            path,
            polyline_column=column,
        )
        return

    if data_format == "jsonl":
        key = dataset_cfg.get(
            "trajectory_key",
            "trajectory",
        )
        yield from iter_jsonl(
            path,
            trajectory_key=key,
        )
        return

    raise ValueError(
        f"Unsupported raw-data format: {data_format}"
    )


def remove_consecutive_duplicates(
    cells: Sequence[int],
) -> List[int]:
    if not cells:
        return []

    output = [int(cells[0])]

    for cell in cells[1:]:
        cell = int(cell)

        if cell != output[-1]:
            output.append(cell)

    return output


def complete_nonadjacent_cells(
    cells: Sequence[int],
    graph: GridGraph,
    path_cache: Optional[
        Dict[Tuple[int, int], Optional[Tuple[int, ...]]]
    ] = None,
) -> Optional[List[int]]:
    if len(cells) < 2:
        return None

    if path_cache is None:
        path_cache = {}

    completed = [int(cells[0])]

    for target in cells[1:]:
        source = completed[-1]
        target = int(target)

        if graph.is_legal_edge(source, target):
            completed.append(target)
            continue

        key = (source, target)

        if key not in path_cache:
            path = graph.shortest_path(
                source,
                target,
            )

            path_cache[key] = (
                tuple(path)
                if path is not None
                else None
            )

        cached = path_cache[key]

        if cached is None:
            return None

        completed.extend(cached[1:])

    return completed


def preprocess_trajectory(
    raw_points: Sequence[Point],
    graph: GridGraph,
    l_max: int,
    path_cache: Optional[
        Dict[Tuple[int, int], Optional[Tuple[int, ...]]]
    ] = None,
) -> Tuple[Optional[Trajectory], str]:
    if len(raw_points) < 2:
        return None, "invalid"

    mapped: List[int] = []

    for lon, lat in raw_points:
        cell = graph.coordinate_to_cell(
            lon,
            lat,
        )

        if cell is None:
            return None, "outside_region"

        mapped.append(cell)

    mapped = remove_consecutive_duplicates(
        mapped
    )

    if len(mapped) < 2:
        return None, "invalid"

    completed = complete_nonadjacent_cells(
        mapped,
        graph=graph,
        path_cache=path_cache,
    )

    if completed is None:
        return None, "completion_failure"

    length = len(completed) - 1

    if length < 1:
        return None, "invalid"

    if length > l_max:
        return None, "length"

    for source, target in zip(
        completed[:-1],
        completed[1:],
    ):
        if not graph.is_legal_edge(
            source,
            target,
        ):
            return None, "invalid"

    return completed, "admitted"


def prepare_dataset(
    config: dict,
    seed: int = 42,
):
    del seed

    graph = build_grid_graph(config)

    l_max = int(
        config["model"]["L_max"]
    )

    target_size = int(
        config["dataset"]["num_trajectories"]
    )

    if target_size <= 0:
        raise ValueError(
            "dataset.num_trajectories must be positive."
        )

    path_cache: Dict[
        Tuple[int, int],
        Optional[Tuple[int, ...]],
    ] = {}

    trajectories: List[Trajectory] = []

    counts = {
        "raw_scanned": 0,
        "admitted": 0,
        "excluded_invalid": 0,
        "excluded_outside_region": 0,
        "excluded_completion_failure": 0,
        "excluded_length": 0,
    }

    for raw_trajectory in iter_raw_trajectories(
        config
    ):
        counts["raw_scanned"] += 1

        trajectory, status = preprocess_trajectory(
            raw_points=raw_trajectory,
            graph=graph,
            l_max=l_max,
            path_cache=path_cache,
        )

        if status == "admitted":
            trajectories.append(
                trajectory
            )
            counts["admitted"] += 1

        elif status == "outside_region":
            counts[
                "excluded_outside_region"
            ] += 1

        elif status == "completion_failure":
            counts[
                "excluded_completion_failure"
            ] += 1

        elif status == "length":
            counts[
                "excluded_length"
            ] += 1

        else:
            counts[
                "excluded_invalid"
            ] += 1

        if len(trajectories) >= target_size:
            break

    if len(trajectories) < target_size:
        raise RuntimeError(
            f"Only {len(trajectories):,} admitted trajectories "
            f"were found, but {target_size:,} are required."
        )

    trajectories = trajectories[:target_size]

    raw_scanned = counts["raw_scanned"]
    admitted = counts["admitted"]

    admission_rate = (
        admitted / raw_scanned
        if raw_scanned > 0
        else 0.0
    )

    lengths = [
        len(trajectory) - 1
        for trajectory in trajectories
    ]

    dataset_stats = {
        "dataset": config["dataset"]["name"],
        "type": config["dataset"].get(
            "type"
        ),
        "area": config["dataset"].get(
            "area"
        ),
        "num_trajectories": len(
            trajectories
        ),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": (
            sum(lengths) / len(lengths)
        ),
    }

    admission_stats = {
        **counts,
        "selected_cohort": len(
            trajectories
        ),
        "admission_rate_among_scanned": (
            admission_rate
        ),
        "L_max": l_max,
        "truncation_used": False,
    }

    return (
        trajectories,
        dataset_stats,
        admission_stats,
    )


def save_processed_dataset(
    trajectories: Sequence[Trajectory],
    output_path: Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        pickle.dump(
            list(trajectories),
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_processed_dataset(
    input_path: Path,
) -> List[Trajectory]:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset does not exist: {input_path}"
        )

    with input_path.open("rb") as file:
        trajectories = pickle.load(file)

    return trajectories