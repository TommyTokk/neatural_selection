import arcade


# Layout utils
def contains(rect: arcade.Rect, x: float, y: float) -> bool:
    return rect.left <= x <= rect.right and rect.bottom <= y <= rect.top


def inset(rect: arcade.Rect, amount: float) -> arcade.Rect:
    return arcade.LBWH(
        rect.left + amount,
        rect.bottom + amount,
        rect.width - 2 * amount,
        rect.height - 2 * amount,
    )
