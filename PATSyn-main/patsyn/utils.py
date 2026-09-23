import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def deep_merge(
    base: dict,
    override: dict,
) -> dict:
    result = deepcopy(
        base
    )

    for key, value in override.items():
        if (
            key in result
            and isinstance(
                result[key],
                dict,
            )
            and isinstance(
                value,
                dict,
            )
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(
                value
            )

    return result


def _load_yaml(
    path: Path,
) -> dict:
    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(
            file
        )

    if data is None:
        return {}

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            f"Configuration must be a mapping: {path}"
        )

    return data


def load_config(
    default_path,
    dataset_path,
) -> dict:
    default_config = (
        _load_yaml(
            Path(default_path)
        )
    )

    dataset_config = (
        _load_yaml(
            Path(dataset_path)
        )
    )

    config = deep_merge(
        default_config,
        dataset_config,
    )

    return config


def set_seed(
    seed: int,
):
    seed = int(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)


def make_rng(
    seed: int,
) -> np.random.Generator:
    return np.random.default_rng(
        int(seed)
    )


def ensure_dir(
    path,
) -> Path:
    path = Path(
        path
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _json_ready(
    value: Any,
):
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_ready(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            _json_ready(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    return value


def save_json(
    data,
    output_path,
    indent: int = 2,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            _json_ready(data),
            file,
            indent=indent,
            ensure_ascii=False,
            sort_keys=False,
        )


def load_json(
    input_path,
):
    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"JSON file does not exist: {input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def peak_rss_mib() -> float:
    try:
        import resource
    except ImportError:
        return float(
            "nan"
        )

    usage = resource.getrusage(
        resource.RUSAGE_SELF
    )

    value = float(
        usage.ru_maxrss
    )

    if sys.platform == "darwin":
        return (
            value
            / 1024.0
            / 1024.0
        )

    return (
        value / 1024.0
    )


def repository_root() -> Path:
    return Path(
        __file__
    ).resolve().parents[1]