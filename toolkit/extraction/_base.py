from abc import ABC, abstractmethod


class BaseModelLoader(ABC):
    @staticmethod
    @abstractmethod
    def load(model_name: str, weights: str | None = "DEFAULT"):
        pass
