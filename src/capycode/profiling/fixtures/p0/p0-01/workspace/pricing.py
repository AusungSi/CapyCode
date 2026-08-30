def shipping_fee(order_total: float) -> float:
    """Return 0 at 50+, 5 at 25+, and 10 below 25."""
    if order_total > 50:
        return 0
    if order_total > 25:
        return 5
    return 10
