from collections.abc import Iterator

Tile = tuple[int, int]


def validate_tile(tile: Tile, *, field_name: str = "tile") -> Tile:
    if len(tile) != 2:
        raise ValueError(f"{field_name} must contain exactly two coordinates")
    x, y = tile
    if x < 0 or y < 0:
        raise ValueError(f"{field_name} coordinates must be non-negative")
    return tile


def validate_size(size: Tile, *, field_name: str = "size") -> Tile:
    if len(size) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"{field_name} values must be positive")
    return size


def rectangle_tiles(origin: Tile, size: Tile) -> Iterator[Tile]:
    x0, y0 = origin
    width, height = size
    for y in range(y0, y0 + height):
        for x in range(x0, x0 + width):
            yield (x, y)
