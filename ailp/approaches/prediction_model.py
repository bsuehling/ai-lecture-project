import os
from abc import ABC, abstractmethod

from ailp.graph import Graph, Part
from ailp.utils import load_model


class PredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    def __init__(
        self, train_path: str | os.PathLike = None, test_path: str | os.PathLike = None
    ) -> None:
        self._train_path = train_path
        self._model = None if test_path is None else load_model(test_path)

    @abstractmethod
    def predict_graph(self, parts: set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the
        method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, train_graphs: list[Graph], eval_graphs: list[Graph]):
        raise NotImplementedError
