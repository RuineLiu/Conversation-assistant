from collections import deque
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.simulation.coordinates import Tile, validate_tile
from proactive_assistant.simulation.objects import SimulationObject


class MapLayer(StrEnum):
    FLOOR = "floor"
    WALLS = "walls"
    FURNITURE = "furniture"
    OBJECTS = "objects"
    COLLISION = "collision"
    SPAWN_POINTS = "spawn_points"
    INTERACTION_ZONES = "interaction_zones"
    DEBUG = "debug"


class ConferenceRoomMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tile_size: int = Field(default=32, gt=0)
    meters_per_tile: float = Field(default=0.30, gt=0)
    layers: list[MapLayer] = Field(default_factory=list)
    wall_tiles: list[Tile] = Field(default_factory=list)
    spawn_points: dict[str, Tile] = Field(default_factory=dict)

    @model_validator(mode="after")
    def map_coordinates_must_be_in_bounds(self) -> "ConferenceRoomMap":
        for tile in self.wall_tiles:
            validate_tile(tile, field_name="wall_tile")
            if not self.in_bounds(tile):
                raise ValueError(f"wall tile out of bounds: {tile}")
        for name, tile in self.spawn_points.items():
            validate_tile(tile, field_name=f"spawn point {name}")
            if not self.in_bounds(tile):
                raise ValueError(f"spawn point out of bounds: {name}={tile}")
        return self

    def in_bounds(self, tile: Tile) -> bool:
        x, y = tile
        return 0 <= x < self.width and 0 <= y < self.height

    def wall_set(self) -> set[Tile]:
        return set(self.wall_tiles)

    def blocked_tiles(self, objects: Iterable[SimulationObject] = ()) -> set[Tile]:
        blocked = self.wall_set()
        for item in objects:
            if item.blocks_movement:
                blocked.update(item.occupied_tiles())
        return blocked

    def is_walkable(
        self,
        tile: Tile,
        objects: Iterable[SimulationObject] = (),
        extra_blocked: Iterable[Tile] = (),
    ) -> bool:
        if not self.in_bounds(tile):
            return False
        blocked = self.blocked_tiles(objects)
        blocked.update(extra_blocked)
        return tile not in blocked

    def neighbors(
        self,
        tile: Tile,
        objects: Iterable[SimulationObject] = (),
        extra_blocked: Iterable[Tile] = (),
    ) -> list[Tile]:
        x, y = tile
        candidates = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
        return [
            candidate
            for candidate in candidates
            if self.is_walkable(candidate, objects=objects, extra_blocked=extra_blocked)
        ]

    def find_path(
        self,
        start: Tile,
        target: Tile,
        objects: Iterable[SimulationObject] = (),
        extra_blocked: Iterable[Tile] = (),
    ) -> list[Tile] | None:
        validate_tile(start, field_name="start")
        validate_tile(target, field_name="target")
        if not self.in_bounds(start) or not self.in_bounds(target):
            return None

        blocked = set(extra_blocked)
        blocked.discard(start)
        if start == target:
            return [start]
        if not self.is_walkable(target, objects=objects, extra_blocked=blocked):
            return None

        queue: deque[Tile] = deque([start])
        came_from: dict[Tile, Tile | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for neighbor in self.neighbors(current, objects=objects, extra_blocked=blocked):
                if neighbor not in came_from:
                    came_from[neighbor] = current
                    queue.append(neighbor)

        if target not in came_from:
            return None

        path: list[Tile] = []
        current: Tile | None = target
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


def default_conference_room_map() -> ConferenceRoomMap:
    width = 26
    height = 18
    walls = []
    for x in range(width):
        walls.append((x, 0))
        walls.append((x, height - 1))
    for y in range(height):
        walls.append((0, y))
        walls.append((width - 1, y))

    # The glass door sits in the east wall and is interactive from inside.
    walls.remove((25, 8))
    walls.remove((25, 9))

    return ConferenceRoomMap(
        width=width,
        height=height,
        tile_size=32,
        meters_per_tile=0.30,
        layers=[
            MapLayer.FLOOR,
            MapLayer.WALLS,
            MapLayer.FURNITURE,
            MapLayer.OBJECTS,
            MapLayer.COLLISION,
            MapLayer.SPAWN_POINTS,
            MapLayer.INTERACTION_ZONES,
            MapLayer.DEBUG,
        ],
        wall_tiles=walls,
        spawn_points={
            "presenter": (4, 14),
            "participant_a": (21, 14),
            "participant_b": (21, 4),
        },
    )
