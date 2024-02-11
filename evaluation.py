import os
import pickle
import random
import sys
from itertools import permutations

import numpy as np
from tqdm import tqdm

from ailp.approaches import PredictionModel
from ailp.constants import MODEL_DICT
from ailp.graph import Graph, Part, graph, node, part
from ailp.utils import create_unique_dir, get_latest_path, get_model_path, split_dataset

# Needed to load graphs from ./data/graphs.dat
sys.modules["graph"] = graph
sys.modules["node"] = node
sys.modules["part"] = part


def evaluate(model: PredictionModel, data_set: list[tuple[set[Part], Graph]]) -> float:
    """
    Evaluates a given prediction model on a given data set.
    :param model: prediction model
    :param data_set: data set
    :return: evaluation score (for now, edge accuracy in percent)
    """
    sum_correct_edges = 0
    edges_counter = 0

    for input_parts, target_graph in tqdm((data_set)):
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
    assert len(predicted_graph.nodes) == len(
        target_graph.nodes
    ), "Mismatch in number of nodes."
    assert (
        predicted_graph.parts == target_graph.parts
    ), "Mismatch in expected and given parts."

    best_score = 0

    # Determine all permutations for the predicted graph and choose the best one in
    # evaluation
    perms: list[tuple[Part]] = __generate_part_list_permutations(predicted_graph.parts)

    # Determine one part order for the target graph
    target_parts_order = perms[0]
    target_adj_matrix = target_graph.get_adjacency_matrix(target_parts_order)

    for perm in perms:
        predicted_adj_matrix = predicted_graph.get_adjacency_matrix(perm)
        score = np.sum(predicted_adj_matrix == target_adj_matrix)
        best_score = max(best_score, score)

    return best_score


def __generate_part_list_permutations(parts: set[Part]) -> list[tuple[Part]]:
    """
    Different instances of the same part type may be interchanged in the graph. This
    method computes all permutations of parts while taking this into account. This
    reduced the number of permutations.
    :param parts: Set of parts to compute permutations
    :return: List of part permutations
    """
    # split parts into sets of same part type
    equal_parts_sets: dict[Part, set[Part]] = {}
    for p in parts:
        for seen_part in equal_parts_sets.keys():
            if p.equivalent(seen_part):
                equal_parts_sets[seen_part].add(p)
                break
        else:
            equal_parts_sets[p] = {p}

    multi_occurrence_parts: list[set[Part]] = [
        pset for pset in equal_parts_sets.values() if len(pset) > 1
    ]
    single_occurrence_parts: list[Part] = [
        next(iter(pset)) for pset in equal_parts_sets.values() if len(pset) == 1
    ]

    full_perms: list[tuple[Part]] = [()]
    for mo_parts in multi_occurrence_parts:
        perms = list(permutations(mo_parts))
        full_perms = (
            list(perms)
            if full_perms == [()]
            else [t1 + t2 for t1 in full_perms for t2 in perms]
        )

    # Add single occurrence parts
    full_perms = [fp + tuple(single_occurrence_parts) for fp in full_perms]
    assert all(
        [len(perm) == len(parts) for perm in full_perms]
    ), "Mismatching number of elements in permutation(s)."
    return full_perms


# --------------------------------------------------------------------------------------
# Example code for evaluation


def main():
    with open("./data/graphs.dat", "rb") as file:
        graphs: list[Graph] = pickle.load(file)

    seed = os.environ.get("RANDOM_SEED", 42)
    stage = os.environ.get("STAGE", "train")
    approach = os.environ.get("APPROACH", "no_ml")
    epoch = os.environ.get("EPOCH")

    random.seed(seed)

    train_graphs, eval_graphs, test_graphs = split_dataset(graphs)

    prediction_model: PredictionModel = MODEL_DICT.get(approach)

    if stage == "train":
        model_path = create_unique_dir(approach)
        prediction_model(train_path=model_path).train(train_graphs, eval_graphs)

    elif stage == "test":
        unique_dir = os.environ.get("UNIQUE_DIR", "latest")
        if unique_dir == "latest":
            load_path = get_latest_path()
        else:
            load_path = get_model_path(approach, unique_dir, epoch=epoch)
        instances = [(graph.parts, graph) for graph in test_graphs]
        eval_score = evaluate(prediction_model(test_path=load_path), instances)

        print(eval_score)


if __name__ == "__main__":
    main()
