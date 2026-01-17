from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Literal, Mapping, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    pass


def load_config(path: str, cls: Type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return decode_dataclass(cls, raw, path="config")


def decode_dataclass(cls: Type[T], data: Any, *, path: str) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must be a mapping, got {type(data).__name__}")

    data_map = dict(data)
    out_kwargs: Dict[str, Any] = {}

    allowed = {f.name for f in fields(cls)}
    extra = sorted(set(data_map.keys()) - allowed)
    if extra:
        raise ConfigError(f"{path} has unknown keys: {extra}")

    # Resolve type hints to concrete types (handles postponed evaluation from 'from __future__ import annotations')
    type_hints = get_type_hints(cls)

    for f in fields(cls):
        key = f.name
        fpath = f"{path}.{key}"
        if key not in data_map:
            raise ConfigError(f"Missing required key: {fpath}")

        tp = type_hints.get(key, f.type)
        out_kwargs[key] = _coerce(tp, data_map[key], path=fpath)

    return cls(**out_kwargs)  # type: ignore[arg-type]


def _coerce(tp: Any, value: Any, *, path: str) -> Any:
    origin = get_origin(tp)

    # Optional[T]
    if origin is Union and type(None) in get_args(tp):
        args = tuple(a for a in get_args(tp) if a is not type(None))
        if value is None:
            return None
        if len(args) != 1:
            return _coerce_union(args, value, path=path)
        return _coerce(args[0], value, path=path)

    # Union[A,B,...]
    if origin is Union:
        return _coerce_union(get_args(tp), value, path=path)

    # Literal[...]
    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            raise ConfigError(f"{path} must be one of {list(allowed)}, got {value!r}")
        return value

    # list[T]
    if origin in (list, List):
        (elem_tp,) = get_args(tp)
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        return [_coerce(elem_tp, v, path=f"{path}[{i}]") for i, v in enumerate(value)]

    # dict[K,V]
    if origin in (dict, Dict):
        key_tp, val_tp = get_args(tp)
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
        out: Dict[Any, Any] = {}
        for k, v in value.items():
            kk = _coerce(key_tp, k, path=f"{path}.<key>")
            out[kk] = _coerce(val_tp, v, path=f"{path}[{kk!r}]")
        return out

    # nested dataclass
    if isinstance(tp, type) and is_dataclass(tp):
        return decode_dataclass(tp, value, path=path)

    # primitives
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be bool, got {type(value).__name__}")
        return value

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be int, got {type(value).__name__}")
        return int(value)

    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be float, got {type(value).__name__}")
        return float(value)

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be str, got {type(value).__name__}")
        return value

    raise ConfigError(f"{path} has unsupported type annotation: {tp!r}")


def _coerce_union(args: Tuple[Any, ...], value: Any, *, path: str) -> Any:
    errors: List[str] = []
    for a in args:
        try:
            return _coerce(a, value, path=path)
        except ConfigError as e:
            errors.append(str(e))
    raise ConfigError(f"{path} did not match any union option. Errors: {errors}")


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    grad_clip: float


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class GMMModelConfig:
    dummy: int


@dataclass(frozen=True)
class MarkovModelConfig:
    emb_dim: int
    num_components: int


@dataclass(frozen=True)
class LatentModelConfig:
    emb_dim: int
    num_components: int
    encoder_hidden: int
    elbo_samples: int


@dataclass(frozen=True)
class ModelConfig:
    gmm: GMMModelConfig
    markov: MarkovModelConfig
    latent: LatentModelConfig


@dataclass(frozen=True)
class AppConfig:
    dataloader: DataLoaderConfig
    train: TrainConfig
    optim: OptimConfig
    model: ModelConfig
