import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Edge = Tuple[int, int]
TripPattern = Tuple[int, int, int]

BOT_EDGE: Edge = (-1, -1)


@dataclass
class GridGraph:
    rows: int
    cols: int
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    connectivity: int
    adjacency: Tuple[Tuple[int, ...], ...]
    edges: Tuple[Edge, ...]

    @property
    def num_cells(self) -> int:
        return self.rows * self.cols

    def coordinate_to_cell(
        self,
        lon: float,
        lat: float,
    ) -> Optional[int]:
        if (
            lon < self.min_lon
            or lon > self.max_lon
            or lat < self.min_lat
            or lat > self.max_lat
        ):
            return None

        lon_width = (
            self.max_lon - self.min_lon
        ) / self.cols

        lat_width = (
            self.max_lat - self.min_lat
        ) / self.rows

        col = int(
            (lon - self.min_lon)
            / lon_width
        )

        row = int(
            (lat - self.min_lat)
            / lat_width
        )

        if col == self.cols:
            col -= 1

        if row == self.rows:
            row -= 1

        if not (
            0 <= row < self.rows
            and 0 <= col < self.cols
        ):
            return None

        return row * self.cols + col

    def cell_to_row_col(
        self,
        cell: int,
    ) -> Tuple[int, int]:
        if not (
            0 <= cell < self.num_cells
        ):
            raise IndexError(
                f"Invalid cell index: {cell}"
            )

        return divmod(
            cell,
            self.cols,
        )

    def neighbors(
        self,
        cell: int,
    ) -> Tuple[int, ...]:
        return self.adjacency[cell]

    def is_legal_edge(
        self,
        source: int,
        target: int,
    ) -> bool:
        if not (
            0 <= source < self.num_cells
            and 0 <= target < self.num_cells
        ):
            return False

        return target in self.adjacency[source]

    def shortest_path(
        self,
        source: int,
        target: int,
    ) -> Optional[List[int]]:
        if source == target:
            return [source]

        queue = deque([source])
        parent = {source: None}

        while queue:
            current = queue.popleft()

            for neighbor in self.adjacency[
                current
            ]:
                if neighbor in parent:
                    continue

                parent[neighbor] = current

                if neighbor == target:
                    path = [target]
                    node = current

                    while node is not None:
                        path.append(node)
                        node = parent[node]

                    path.reverse()
                    return path

                queue.append(neighbor)

        return None


@dataclass
class PublicDomain:
    grid: GridGraph
    L_max: int
    length_bins: Tuple[
        Tuple[int, ...],
        ...
    ]
    reachability: Tuple[
        Tuple[frozenset, ...],
        ...
    ]
    feasible_lengths: Dict[
        TripPattern,
        Tuple[int, ...],
    ]
    omega_q: Tuple[
        TripPattern,
        ...
    ]
    q_to_index: Dict[
        TripPattern,
        int,
    ]
    omega_x: Tuple[
        Edge,
        ...
    ]
    edge_to_index: Dict[
        Edge,
        int,
    ]
    bot_index: int
    D_max: Optional[int]

    @property
    def stats(self) -> dict:
        return {
            "num_cells": (
                self.grid.num_cells
            ),
            "num_edges": len(
                self.grid.edges
            ),
            "L_max": self.L_max,
            "num_length_bins": len(
                self.length_bins
            ),
            "omega_q_size": len(
                self.omega_q
            ),
            "omega_x_size": len(
                self.omega_x
            ),
            "D_max": self.D_max,
        }

    def length_bin_index(
        self,
        length: int,
    ) -> int:
        for index, values in enumerate(
            self.length_bins,
            start=1,
        ):
            if length in values:
                return index

        raise ValueError(
            f"Length {length} is not covered "
            "by the configured length bins."
        )

    def get_feasible_lengths(
        self,
        source: int,
        destination: int,
        bin_index: int,
    ) -> Tuple[int, ...]:
        key = (
            source,
            destination,
            bin_index,
        )

        return self.feasible_lengths.get(
            key,
            (),
        )

    def is_reachable_exact(
        self,
        source: int,
        destination: int,
        length: int,
    ) -> bool:
        if not (
            0 <= length <= self.L_max
        ):
            return False

        return (
            destination
            in self.reachability[
                length
            ][source]
        )


def _require_value(
    mapping: dict,
    key: str,
    scope: str,
):
    value = mapping.get(key)

    if value is None:
        raise ValueError(
            f"Missing configuration: "
            f"{scope}.{key}"
        )

    return value


def _parse_length_bins(
    config: dict,
    l_max: int,
) -> Tuple[Tuple[int, ...], ...]:
    domain_cfg = config.get(
        "public_domain",
        {},
    )

    raw_bins = domain_cfg.get(
        "length_bins"
    )

    if raw_bins is None:
        raise ValueError(
            "Missing public_domain.length_bins. "
            "Use the verified experimental configuration."
        )

    bins: List[Tuple[int, ...]] = []
    used = set()

    for item in raw_bins:
        if isinstance(item, dict):
            lower = int(item["min"])
            upper = int(item["max"])
        elif (
            isinstance(item, (list, tuple))
            and len(item) == 2
        ):
            lower = int(item[0])
            upper = int(item[1])
        else:
            raise ValueError(
                "Each length bin must be "
                "[min, max] or {min: ..., max: ...}."
            )

        if lower < 1 or upper > l_max:
            raise ValueError(
                "Length bins must lie within "
                f"[1, {l_max}]."
            )

        if lower > upper:
            raise ValueError(
                "Invalid length-bin interval."
            )

        values = tuple(
            range(
                lower,
                upper + 1,
            )
        )

        overlap = used.intersection(
            values
        )

        if overlap:
            raise ValueError(
                "Length bins must be disjoint."
            )

        used.update(values)
        bins.append(values)

    expected = set(
        range(
            1,
            l_max + 1,
        )
    )

    if used != expected:
        raise ValueError(
            "Length bins must cover all lengths "
            f"from 1 to {l_max}."
        )

    return tuple(bins)


def build_grid_graph(
    config: dict,
) -> GridGraph:
    domain_cfg = config.get(
        "public_domain",
        {},
    )

    grid_cfg = domain_cfg.get(
        "grid",
        {},
    )

    rows = int(
        _require_value(
            grid_cfg,
            "rows",
            "public_domain.grid",
        )
    )

    cols = int(
        _require_value(
            grid_cfg,
            "cols",
            "public_domain.grid",
        )
    )

    min_lon = float(
        _require_value(
            grid_cfg,
            "min_lon",
            "public_domain.grid",
        )
    )

    max_lon = float(
        _require_value(
            grid_cfg,
            "max_lon",
            "public_domain.grid",
        )
    )

    min_lat = float(
        _require_value(
            grid_cfg,
            "min_lat",
            "public_domain.grid",
        )
    )

    max_lat = float(
        _require_value(
            grid_cfg,
            "max_lat",
            "public_domain.grid",
        )
    )

    connectivity = int(
        grid_cfg.get(
            "connectivity",
            8,
        )
    )

    if rows <= 0 or cols <= 0:
        raise ValueError(
            "Grid dimensions must be positive."
        )

    if not (
        min_lon < max_lon
        and min_lat < max_lat
    ):
        raise ValueError(
            "Invalid grid bounds."
        )

    if connectivity not in (4, 8):
        raise ValueError(
            "Grid connectivity must be 4 or 8."
        )

    if connectivity == 4:
        offsets = [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
        ]
    else:
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    adjacency: List[
        Tuple[int, ...]
    ] = []

    edges: List[Edge] = []

    for row in range(rows):
        for col in range(cols):
            cell = (
                row * cols + col
            )

            neighbors: List[int] = []

            for dr, dc in offsets:
                rr = row + dr
                cc = col + dc

                if (
                    0 <= rr < rows
                    and 0 <= cc < cols
                ):
                    neighbor = (
                        rr * cols + cc
                    )
                    neighbors.append(
                        neighbor
                    )

            neighbors.sort()

            adjacency.append(
                tuple(neighbors)
            )

            for neighbor in neighbors:
                edges.append(
                    (cell, neighbor)
                )

    edges.sort()

    return GridGraph(
        rows=rows,
        cols=cols,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        connectivity=connectivity,
        adjacency=tuple(adjacency),
        edges=tuple(edges),
    )


def build_exact_reachability(
    graph: GridGraph,
    l_max: int,
):
    num_cells = graph.num_cells

    levels: List[
        Tuple[frozenset, ...]
    ] = []

    level_zero = tuple(
        frozenset([source])
        for source in range(
            num_cells
        )
    )

    levels.append(level_zero)

    previous = level_zero

    for _ in range(
        1,
        l_max + 1,
    ):
        current = []

        for source in range(
            num_cells
        ):
            reachable = set()

            for middle in previous[
                source
            ]:
                reachable.update(
                    graph.neighbors(
                        middle
                    )
                )

            current.append(
                frozenset(
                    reachable
                )
            )

        current_tuple = tuple(
            current
        )

        levels.append(
            current_tuple
        )

        previous = current_tuple

    return tuple(levels)


def build_public_domain(
    config: dict,
):
    graph = build_grid_graph(config)

    l_max = int(
        config["model"]["L_max"]
    )

    length_bins = _parse_length_bins(
        config=config,
        l_max=l_max,
    )

    reachability = (
        build_exact_reachability(
            graph=graph,
            l_max=l_max,
        )
    )

    feasible_lengths: Dict[
        TripPattern,
        Tuple[int, ...],
    ] = {}

    omega_q: List[
        TripPattern
    ] = []

    for source in range(
        graph.num_cells
    ):
        for destination in range(
            graph.num_cells
        ):
            for bin_index, values in enumerate(
                length_bins,
                start=1,
            ):
                feasible = tuple(
                    length
                    for length in values
                    if destination
                    in reachability[
                        length
                    ][source]
                )

                if not feasible:
                    continue

                key = (
                    source,
                    destination,
                    bin_index,
                )

                feasible_lengths[
                    key
                ] = feasible

                omega_q.append(
                    key
                )

    omega_q.sort()

    domain_cfg = config.get(
        "public_domain",
        {},
    )

    d_max = domain_cfg.get(
        "D_max"
    )

    if d_max is not None:
        d_max = int(d_max)

        if len(omega_q) > d_max:
            raise ValueError(
                f"|Omega_q|={len(omega_q)} exceeds "
                f"D_max={d_max}."
            )

    omega_x = list(
        graph.edges
    )
    omega_x.append(
        BOT_EDGE
    )

    q_to_index = {
        value: index
        for index, value
        in enumerate(omega_q)
    }

    edge_to_index = {
        value: index
        for index, value
        in enumerate(omega_x)
    }

    bot_index = edge_to_index[
        BOT_EDGE
    ]

    public_domain = PublicDomain(
        grid=graph,
        L_max=l_max,
        length_bins=length_bins,
        reachability=reachability,
        feasible_lengths=(
            feasible_lengths
        ),
        omega_q=tuple(omega_q),
        q_to_index=q_to_index,
        omega_x=tuple(omega_x),
        edge_to_index=edge_to_index,
        bot_index=bot_index,
        D_max=d_max,
    )

    return (
        public_domain,
        public_domain.stats,
    )


def save_public_domain(
    public_domain: PublicDomain,
    output_path: Path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        pickle.dump(
            public_domain,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_public_domain(
    input_path: Path,
) -> PublicDomain:
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Public-domain cache does not exist: {input_path}"
        )

    with input_path.open("rb") as file:
        public_domain = pickle.load(file)

    if not isinstance(
        public_domain,
        PublicDomain,
    ):
        raise TypeError(
            "Invalid public-domain cache."
        )

    return public_domain