from part import Part


class Node:
    """
    A class to represent nodes of a graph.
    A part is described by its ID (id) and its containing part (part).
    """
    def __init__(self, node_id: int, part: Part):
        self.__id: int = node_id
        self.__part: Part = part

    def __eq__(self, other) -> bool:
        """ Defines equality of two nodes. Needed for software tests. """
        if other is None:
            return False
        if not isinstance(other, Node):
            raise TypeError(f'Can not compare different types ({type(self)} and {type(other)})')
        return self.get_id() == other.get_id() and self.get_part() == other.get_part()

    def __repr__(self) -> str:
        """
        Returns a string representation of the current Node object.
        :return: string representation
        """
        return f'{self.__class__.__name__}(NodeID={self.get_id()}, Part={self.get_part()})'

    def __hash__(self) -> int:
        """ Defines hash of a node. Needed for software tests. """
        return hash((self.get_id(), self.get_part()))

    def __lt__(self, other) -> bool:
        """ Defines an order on nodes. """
        if not isinstance(other, Node):
            raise TypeError(f'Can not compare different types ({type(self)} and {type(other)})')
        return self.get_part() < other.get_part()

    def get_id(self) -> int:
        return self.__id

    def get_part(self) -> Part:
        return self.__part
