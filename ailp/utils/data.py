import json
import os
import random

from ailp.graph import Graph


def split_dataset(graphs: list[Graph], p_train: float = 0.7):
    """Splits the dataset into train, validation, and test set.

    Parameters
    ----------
    graphs : list[Graph]
        The dataset.
    p_train : float = 0.7
        Which proportion to use for the train set. The rest will be equally split into
        validation and test set

    Returns
    -------
    tuple[list[Graph], list[Graph], list[Graph]]
        Training set, validation set, and test set.

    Raises
    ------
    ValueError
        if 0 >= p_train or 1 <= p_train.
    """
    if not 0 < p_train < 1:
        raise ValueError(f"p_train has to be between 0 and 1, but is {p_train}")

    sample = random.sample(graphs, len(graphs))

    end_train = int(p_train * len(graphs))
    end_validate = int((p_train + 1) / 2 * len(graphs))

    return sample[:end_train], sample[end_train:end_validate], sample[end_validate:]


def build_one_hot_mappings(graphs: list[Graph]):
    part_ids: set[int] = set()
    family_ids: set[int] = set()

    for g in graphs:
        for n in g.nodes:
            part_ids.add(int(n.part.part_id))
            family_ids.add(int(n.part.family_id))

    part_map = {part_id: i for i, part_id in enumerate(sorted(part_ids))}
    with open(os.path.join("data", "mappings", "part_map.json"), "w") as f:
        json.dump(part_map, f)

    family_map = {family_id: i for i, family_id in enumerate(sorted(family_ids))}
    with open(os.path.join("data", "mappings", "family_map.json"), "w") as f:
        json.dump(family_map, f)
