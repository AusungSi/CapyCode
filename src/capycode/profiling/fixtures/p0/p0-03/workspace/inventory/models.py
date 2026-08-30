from dataclasses import dataclass


@dataclass
class Product:
    sku: str
    stock: int
