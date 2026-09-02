import unittest

from pricing import shipping_fee


class HiddenShippingFeeTests(unittest.TestCase):
    def test_paid_shipping_boundary_is_inclusive(self) -> None:
        self.assertEqual(shipping_fee(25), 5)

    def test_values_around_boundaries(self) -> None:
        self.assertEqual(shipping_fee(24.99), 10)
        self.assertEqual(shipping_fee(49.99), 5)


if __name__ == "__main__":
    unittest.main()
