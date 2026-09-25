from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class HCRReport:
    j: int
    sigma: int


def hadamard_dimension(
    domain_size: int,
) -> int:
    if domain_size <= 0:
        raise ValueError(
            "domain_size must be positive."
        )

    required = domain_size + 1

    return 1 << (
        required - 1
    ).bit_length()


def attenuation(
    epsilon: float,
) -> float:
    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    exp_epsilon = np.exp(
        epsilon
    )

    return float(
        (exp_epsilon - 1.0)
        / (exp_epsilon + 1.0)
    )


def category_row(
    category_index: int,
    domain_size: int,
) -> int:
    if not (
        0 <= category_index
        < domain_size
    ):
        raise IndexError(
            "Invalid category index."
        )

    # Row 0 is the constant Hadamard row.
    return category_index + 1


def hadamard_sign(
    row: int,
    column: int,
) -> int:
    parity = (
        row & column
    ).bit_count() & 1

    return (
        1
        if parity == 0
        else -1
    )


def hadamard_signs(
    rows: np.ndarray,
    column: int,
) -> np.ndarray:
    values = np.bitwise_and(
        rows.astype(
            np.uint64,
            copy=False,
        ),
        np.uint64(column),
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


def encode_index(
    category_index: int,
    domain_size: int,
    epsilon: float,
    rng: np.random.Generator,
) -> HCRReport:
    k = hadamard_dimension(
        domain_size
    )

    row = category_row(
        category_index,
        domain_size,
    )

    j = int(
        rng.integers(
            0,
            k,
        )
    )

    h = hadamard_sign(
        row,
        j,
    )

    keep_probability = (
        np.exp(epsilon)
        / (
            np.exp(epsilon)
            + 1.0
        )
    )

    sigma = (
        h
        if rng.random()
        < keep_probability
        else -h
    )

    return HCRReport(
        j=j,
        sigma=int(sigma),
    )


def batch_encode_indices(
    category_indices: Sequence[int],
    domain_size: int,
    epsilon: float,
    rng: np.random.Generator,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    indices = np.asarray(
        category_indices,
        dtype=np.int64,
    )

    if np.any(indices < 0) or np.any(
        indices >= domain_size
    ):
        raise IndexError(
            "Category index outside domain."
        )

    k = hadamard_dimension(
        domain_size
    )

    rows = (
        indices + 1
    ).astype(
        np.uint64
    )

    j = rng.integers(
        low=0,
        high=k,
        size=len(indices),
        dtype=np.int64,
    )

    signs = np.empty(
        len(indices),
        dtype=np.int8,
    )

    for index in range(
        len(indices)
    ):
        signs[index] = (
            hadamard_sign(
                int(rows[index]),
                int(j[index]),
            )
        )

    exp_epsilon = np.exp(
        epsilon
    )

    keep_probability = (
        exp_epsilon
        / (
            exp_epsilon + 1.0
        )
    )

    keep = (
        rng.random(
            len(indices)
        )
        < keep_probability
    )

    sigma = np.where(
        keep,
        signs,
        -signs,
    ).astype(
        np.int8
    )

    return j, sigma


def decode_score(
    report: HCRReport,
    candidate_index: int,
    domain_size: int,
    epsilon: float,
) -> float:
    row = category_row(
        candidate_index,
        domain_size,
    )

    sign = hadamard_sign(
        row,
        report.j,
    )

    return float(
        report.sigma
        * sign
        / attenuation(epsilon)
    )


def decode_all_scores(
    j: int,
    sigma: int,
    domain_size: int,
    epsilon: float,
) -> np.ndarray:
    rows = np.arange(
        1,
        domain_size + 1,
        dtype=np.uint64,
    )

    signs = hadamard_signs(
        rows,
        j,
    ).astype(
        np.float64
    )

    return (
        float(sigma)
        * signs
        / attenuation(epsilon)
    )