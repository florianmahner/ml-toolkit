"""Collection of custom dictionary classes."""

from collections import OrderedDict
from typing import Any


class IndexedDict(dict):
    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            keys = list(self.keys())
            if key < 0:
                key += len(keys)
            if key >= len(keys) or key < 0:
                raise IndexError("Index out of range")
            return self[keys[key]]
        else:
            return super().__getitem__(key)


class FrozenDict(dict):
    def __setitem__(self, key: str | int, value: Any) -> None:
        raise TypeError("FrozenDict does not support item assignment")

    def __delitem__(self, key: str | int) -> None:
        raise TypeError("FrozenDict does not support item deletion")


class OrderedIndexedDict(IndexedDict, OrderedDict):
    """Combines the functionality of IndexedDict and OrderedDict."""

    pass
