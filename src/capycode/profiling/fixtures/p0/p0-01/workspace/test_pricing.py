import unittest

from pricing import shipping_fee


class ShippingFeeTests(unittest.TestCase):
    def test_free_shipping_starts_at_fifty(self) -> None:
        self.assertEqual(shipping_fee(50), 0)

    def test_regular_values(self) -> None:
        self.assertEqual(shipping_fee(60), 0)
        self.assertEqual(shipping_fee(30), 5)
        self.assertEqual(shipping_fee(10), 10)


if __name__ == "__main__":
    unittest.main()
