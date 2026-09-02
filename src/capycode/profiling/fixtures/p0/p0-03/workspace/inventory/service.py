from .models import Product


class OutOfStockError(ValueError):
    pass


class Inventory:
    def __init__(self, products: list[Product]) -> None:
        self._products = {product.sku: product for product in products}

    def reserve(self, sku: str, quantity: int) -> int:
        """Reserve a positive quantity and return the remaining stock."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        product = self._products[sku]
        if product.stock > quantity:
            product.stock -= quantity
            return product.stock
        raise OutOfStockError(sku)
