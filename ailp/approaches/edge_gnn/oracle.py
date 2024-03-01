import torch
import torch.nn.functional as F
from torch import Tensor

from ailp.graph import Edge, Graph, Node


def oracle_loss(
    graph: Graph,
    target: list[Edge],
    prediction: Tensor,  # predicted prob of each candidate (n,)
    candidates: Tensor,  # tensor of possible preds (n,2)
    pools: list[set[Edge]],
    visited: list[set[Edge]],
    start_node: Node | None,
) -> tuple[Tensor, Edge, list[set[Edge]], list[set[Edge]]]:
    pred_idx = prediction.argmax()
    pred = candidates[pred_idx]
    valid_edges = _get_valid_edges(graph, pools, visited, start_node)
    valid = [(n1.part.part_id, n2.part.part_id) for n1, n2 in valid_edges]

    t = torch.tensor([1 if c.tolist() in valid else 0 for c in candidates])

    if tuple(pred.tolist()) in valid:
        t = F.one_hot(torch.tensor([pred_idx]), num_classes=len(prediction))
    else:
        pred_idx = (prediction * t).argmax()
        pred = candidates[pred_idx]

    loss = F.cross_entropy(prediction, t)

    new_edge: Edge = target[pred_idx.item()]
    new_pools, new_visited = _update_pools(graph, pools, visited, tuple(pred.tolist()))

    return loss, new_edge, new_pools, new_visited


def _get_valid_edges(
    graph: Graph,
    pools: list[set[Edge]],
    visited: list[set[Edge]],
    start_node: Node | None,
) -> set[Edge]:
    if not pools:
        return _find_edges_from_node(graph, start_node)
    valid: set[Edge] = set()
    for pool, vis in zip(pools, visited):
        valid |= pool - vis
    return valid


def _update_pools(
    graph: Graph,
    pools: list[set[Edge]],
    visited: list[set[Edge]],
    edge: tuple[int, int],
) -> tuple[list[set[Edge]], list[set[Edge]]]:
    new: set[tuple[tuple[Edge, ...], tuple[Edge, ...]]] = set()
    if not pools:
        return [_find_edges(graph, edge)], [{edge, edge[::-1]}]
    for pool, vis in zip(pools, visited):
        for e in pool - vis:
            if e[0].part.part_id == edge[0] and e[1].part.part_id == edge[1]:
                p = pool | _find_edges(graph, e)
                v = vis | {e, e[::-1]}
                new.add((tuple(p), tuple(v)))
    new_pools = [set(p[0]) for p in new]
    new_visited = [set(p[1]) for p in new]
    return new_pools, new_visited


def _find_edges(graph: Graph, edge: Edge) -> set[Edge]:
    edges: set[Edge] = set()
    for node1, v in graph.edges.items():
        if node1 in edge:
            edges |= {(node1, node2) for node2 in v}
    return edges


def _find_edges_from_node(graph: Graph, node: Node) -> set[Edge]:
    edges: set[Edge] = set()
    for node1, v in graph.edges.items():
        if node1 == node:
            edges |= {(node1, node2) for node2 in v}
    return edges


# def _get_valid_edges(graph: Graph, target: list[Edge], options: set[tuple[Edge, ...]]):
#     if not options:
#         return set(target)

#     valid: set[Edge] = set()
#     for option in options:
#         pool: set[Edge] = set()
#         visited: set[Edge] = set()
#         for edge in option:
#             visited |= {edge, edge[::-1]}
#             pool |= _find_edges(graph, edge)
#         valid |= pool - visited
#     return valid
