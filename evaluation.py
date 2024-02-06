from abc import ABC, abstractmethod
from itertools import permutations
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple

from graph import Graph
from node import Node
from part import Part


class MyPredictionModel(ABC):
    """
    This class is a blueprint for your prediction model(s) serving as base class.
    """

    @abstractmethod
    def predict_graph(self, parts: Set[Part]) -> Graph:
        """
        Returns a graph containing all given parts. This method is called within the method `evaluate`.
        :param parts: set of parts to form up a construction (i.e. graph)
        :return: graph
        """
        # TODO: implement this method
        ...


def load_model(file_path: str) -> MyPredictionModel:
    """
        This method loads the prediction model from a file (needed for evaluating your model on the test set).
        :param file_path: path to file
        :return: the loaded prediction model
    """
    ...


def evaluate(model: MyPredictionModel, data_set: List[Tuple[Set[Part], Graph]]) -> float:
    """
    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    sum_correct_edges = 0
    edges_counter = 0

    for input_parts, target_graph in data_set:
        predicted_graph = model.predict_graph(input_parts)

        edges_counter += len(input_parts) * len(input_parts)
        sum_correct_edges += edge_accuracy(predicted_graph, target_graph)

        # FYI: maybe some more evaluation metrics will be used in final evaluation

    return sum_correct_edges / edges_counter * 100


def edge_accuracy(predicted_graph: Graph, target_graph: Graph) -> int:
    """
    Returns the number of correct predicted edges.
    :param predicted_graph:
    :param target_graph:
    :return:
    """
    assert len(predicted_graph.get_nodes()) == len(target_graph.get_nodes()), 'Mismatch in number of nodes.'
    assert predicted_graph.get_parts() == target_graph.get_parts(), 'Mismatch in expected and given parts.'

    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in evaluation
    perms: List[Tuple[Part]] = __generate_part_list_permutations(predicted_graph.get_parts())

    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)

    return best_score


def __generate_part_list_permutations(parts: Set[Part]) -> List[Tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This method computes all permutations
    of parts while taking this into account. This reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: Dict[Part, Set[Part]] = {}
    for part in parts:
        for seen_part in equal_parts_sets.keys():
            if part.equivalent(seen_part):
                equal_parts_sets[seen_part].add(part)
                break
        else:
            equal_parts_sets[part] = {part}

    multi_occurrence_parts: List[Set[Part]] = [pset for pset in equal_parts_sets.values() if len(pset) > 1]
    single_occurrence_parts: List[Part] = [next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1]

    full_perms: List[Tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = list(perms) if full_perms == [()] else [t1 + t2 for t1 in full_perms for t2 in perms]

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all([len(perm) == len(parts) for perm in full_perms]), 'Mismatching number of elements in permutation(s).'
    return full_perms


# ---------------------------------------------------------------------------------------------------------------------
# Example code for evaluation

if __name__ == '__main__':
    # Load train data
    with open('graphs.dat', 'rb') as file:
        train_graphs: List[Graph] = pickle.load(file)

    # Load the final model

    model_file_path = ''  # ToDo
    prediction_model: MyPredictionModel = load_model(model_file_path)

    # For illustration, compute eval score on train data
    instances = [(graph.get_parts(), graph) for graph in train_graphs[:100]]
    eval_score = evaluate(prediction_model, instances)
