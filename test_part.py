import unittest
from part import Part


class TestPart(unittest.TestCase):

    def test_init_empty_partid_raises_assertion_error(self):
        self.assertRaises(AssertionError, Part, 0, 1)

    def test_init_empty_familyid_raises_assertion_error(self):
        self.assertRaises(AssertionError, Part, 1, 0)

    def test_equivalent_returns_true_for_equivalent_parts(self):
        part1 = Part(1, 2)
        part1_equiv = Part(1, 2)

        self.assertTrue(part1.equivalent(part1_equiv), 'Parts with same attribute values should be equivalent.')

    def test_equivalent_returns_false_for_non_equivalent_parts(self):
        part1 = Part(1, 2)
        part2 = Part(3, 2)

        self.assertFalse(part1.equivalent(part2), 'Parts with different attribute values should not be equivalent.')

    def test_equivalent_parts_not_identical(self):
        part1 = Part(1, 2)
        part1_equiv = Part(1, 2)

        self.assertFalse(part1 == part1_equiv,
                         'Different instances with same attribute values should not be identical.')
