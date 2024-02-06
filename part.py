class Part:
    """
    A class to represent pseudonymized parts.
    A part is described by its ID (part_id) and the ID of its corresponding family (family_id).
    Multiple parts can belong to the same family (i.e. different value for part_id but same value for family_id).
    """

    def __init__(self, part_id: int, family_id: int):
        assert part_id and family_id, 'Creation of Part failed. Fields `part_id` and `family_id` must not be empty.'
        self.__part_id: int = part_id
        self.__family_id: int = family_id

    def equivalent(self, other) -> bool:
        """
        Defines equivalence of a given object and this Part object. Given both are of type Part, it checks if 
        they are two instances of the same Part Type (as opposed to the same instance determined by __eq__).
        :param other: object to determine equality with this Part
        :raise: TypeError if given object is not a Part
        :return: boolean that indicates if the given object is equivalent to the current Part.

        """
        if not isinstance(other, Part):
            raise TypeError(f'Can not compare different types ({type(self)} and {type(other)})')
        return self.get_part_id() == other.get_part_id() and self.get_family_id() == other.get_family_id()

    def __repr__(self) -> str:
        """
        Returns a string representation of the current Part object.
        :return: string representation
        """
        return f'{self.__class__.__name__}(PartID={self.get_part_id()}, FamilyID={self.get_family_id()})'

    def __hash__(self) -> int:
        """ Defines hash of a component. Needed for software tests. """
        return hash((self.get_part_id(), self.get_family_id()))

    def __lt__(self, other) -> bool:
        """ Defines an order on components. """
        if not isinstance(other, Part):
            raise TypeError(f'Can not define order for different types ({type(self)} and {type(other)})')
        return (self.get_part_id(), self.get_family_id()) < \
               (other.get_part_id(), other.get_family_id())

    def get_part_id(self) -> int:
        return self.__part_id

    def get_family_id(self) -> int:
        return self.__family_id
