from __future__ import annotations

from typing import Callable, TypeVar

from panorama.core.exceptions import ModelBuildError

T = TypeVar("T")


class Registry:
    """Maps a name -> a class, so config strings can select components."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def _wrap(cls: type[T]) -> type[T]:
            if key in self._entries:
                raise KeyError(f"{key!r} already registered in {self._name!r}")
            self._entries[key] = cls
            return cls
        return _wrap

    def get(self, key: str) -> type:
        if key not in self._entries:
            raise ModelBuildError(
                f"{key!r} not found in registry {self._name!r}. Available: {self.keys()}"
            )
        return self._entries[key]

    def build(self, key: str, **kwargs):
        return self.get(key)(**kwargs)

    def keys(self) -> list[str]:
        return sorted(self._entries)


MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")