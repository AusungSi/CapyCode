import unittest

from inventory import Inventory, OutOfStockError, Product


class InventoryTests(unittest.TestCase):
    def test_can_reserve_all_remaining_units(self) -> None:
        inventory = Inventory([Product("A-1", 3)])
        self.assertEqual(inventory.reserve("A-1", 3), 0)

    def test_insufficient_stock_is_rejected(self) -> None:
        inventory = Inventory([Product("A-1", 2)])
        with self.assertRaises(OutOfStockError):
            inventory.reserve("A-1", 3)


if __name__ == "__main__":
    unittest.main()
