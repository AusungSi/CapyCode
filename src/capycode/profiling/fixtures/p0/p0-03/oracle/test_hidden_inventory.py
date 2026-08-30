import unittest

from inventory import Inventory, Product


class HiddenInventoryTests(unittest.TestCase):
    def test_sequential_reservations_reach_zero(self) -> None:
        inventory = Inventory([Product("B-2", 5)])
        self.assertEqual(inventory.reserve("B-2", 2), 3)
        self.assertEqual(inventory.reserve("B-2", 3), 0)

    def test_non_positive_quantity_is_rejected(self) -> None:
        inventory = Inventory([Product("B-2", 5)])
        with self.assertRaises(ValueError):
            inventory.reserve("B-2", 0)


if __name__ == "__main__":
    unittest.main()
