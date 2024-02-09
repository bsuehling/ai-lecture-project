from abc import ABC, abstractmethod

from ailp.graph import Graph, Part


class PredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

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
