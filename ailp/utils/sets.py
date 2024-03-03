from ailp.graph import Part


def remove_by_part_id(parts: set[Part], part_id: int):
    """From a set of parts, remove the one with the spcecified id.

    Parameters
    ----------
    parts : set[Part]
        The set of parts.
    part_id : int
        The id of the part to remove

    Returns
    -------
    Part
        The removed part.
    """
    to_remove = next((p for p in parts if p.part_id == part_id))
    parts.remove(to_remove)
    return to_remove
