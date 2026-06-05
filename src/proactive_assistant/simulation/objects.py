from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from proactive_assistant.simulation.coordinates import (
    Tile,
    rectangle_tiles,
    validate_size,
    validate_tile,
)


class ObjectType(StrEnum):
    DOOR = "door"
    CONFERENCE_TABLE = "conference_table"
    CHAIR = "chair"
    SCREEN = "screen"
    WHITEBOARD = "whiteboard"
    LIGHT_SWITCH = "light_switch"
    LAPTOP = "laptop"
    DOCUMENTS = "documents"
    CUP = "cup"
    REMOTE = "remote"


class SimulationObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    type: ObjectType
    label: str
    tile: Tile
    size: Tile = (1, 1)
    blocks_movement: bool = True
    interaction_tiles: list[Tile] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coordinates_must_be_valid(self) -> "SimulationObject":
        validate_tile(self.tile)
        validate_size(self.size)
        for tile in self.interaction_tiles:
            validate_tile(tile, field_name="interaction_tile")
        return self

    def occupied_tiles(self) -> set[Tile]:
        return set(rectangle_tiles(self.tile, self.size))

    def can_interact_from(self, tile: Tile) -> bool:
        return tile in set(self.interaction_tiles)


def default_conference_room_objects() -> dict[str, SimulationObject]:
    objects = [
        SimulationObject(
            object_id="door_1",
            type=ObjectType.DOOR,
            label="Glass entrance",
            tile=(25, 8),
            size=(1, 2),
            blocks_movement=False,
            interaction_tiles=[(24, 8), (24, 9)],
            state={"open": False},
        ),
        SimulationObject(
            object_id="conference_table_1",
            type=ObjectType.CONFERENCE_TABLE,
            label="Curved conference table",
            tile=(7, 7),
            size=(12, 4),
            blocks_movement=True,
            interaction_tiles=[
                (7, 6),
                (8, 6),
                (9, 6),
                (10, 6),
                (11, 6),
                (12, 6),
                (13, 6),
                (14, 6),
                (15, 6),
                (16, 6),
                (17, 6),
                (18, 6),
                (7, 11),
                (8, 11),
                (9, 11),
                (10, 11),
                (11, 11),
                (12, 11),
                (13, 11),
                (14, 11),
                (15, 11),
                (16, 11),
                (17, 11),
                (18, 11),
                (6, 7),
                (6, 8),
                (6, 9),
                (6, 10),
                (19, 7),
                (19, 8),
                (19, 9),
                (19, 10),
            ],
            state={"surface": "clear"},
        ),
        SimulationObject(
            object_id="screen_1",
            type=ObjectType.SCREEN,
            label="Presentation screen",
            tile=(1, 5),
            size=(1, 3),
            blocks_movement=True,
            interaction_tiles=[(2, 5), (2, 6), (2, 7)],
            state={"power": "off", "display_mode": "blank"},
        ),
        SimulationObject(
            object_id="whiteboard_1",
            type=ObjectType.WHITEBOARD,
            label="Whiteboard",
            tile=(5, 1),
            size=(5, 1),
            blocks_movement=True,
            interaction_tiles=[(5, 2), (6, 2), (7, 2), (8, 2), (9, 2)],
            state={"content": ""},
        ),
        SimulationObject(
            object_id="light_switch_1",
            type=ObjectType.LIGHT_SWITCH,
            label="Light switch",
            tile=(23, 6),
            blocks_movement=False,
            interaction_tiles=[(22, 6), (23, 7)],
            state={"lights": "on"},
        ),
        SimulationObject(
            object_id="laptop_1",
            type=ObjectType.LAPTOP,
            label="Presenter laptop",
            tile=(11, 7),
            blocks_movement=False,
            interaction_tiles=[(11, 6), (10, 6), (12, 6)],
            state={"open": False, "power": "off"},
        ),
        SimulationObject(
            object_id="documents_1",
            type=ObjectType.DOCUMENTS,
            label="Meeting documents",
            tile=(14, 7),
            blocks_movement=False,
            interaction_tiles=[(14, 6), (13, 6), (15, 6)],
            state={"reviewed_by": []},
        ),
        SimulationObject(
            object_id="cup_1",
            type=ObjectType.CUP,
            label="Coffee cup",
            tile=(17, 7),
            blocks_movement=False,
            interaction_tiles=[(17, 6), (16, 6), (18, 6)],
            state={"held_by": None},
        ),
        SimulationObject(
            object_id="remote_1",
            type=ObjectType.REMOTE,
            label="Presentation remote",
            tile=(18, 10),
            blocks_movement=False,
            interaction_tiles=[(18, 11), (17, 11), (19, 10)],
            state={"last_used_by": None},
        ),
    ]

    chair_tiles = [
        (7, 5),
        (10, 5),
        (13, 5),
        (16, 5),
        (19, 7),
        (19, 10),
        (16, 12),
        (13, 12),
        (10, 12),
        (7, 12),
        (5, 7),
        (5, 10),
    ]
    for index, tile in enumerate(chair_tiles, start=1):
        objects.append(
            SimulationObject(
                object_id=f"chair_{index}",
                type=ObjectType.CHAIR,
                label=f"Conference chair {index}",
                tile=tile,
                blocks_movement=True,
                interaction_tiles=[
                    (max(tile[0] - 1, 0), tile[1]),
                    (min(tile[0] + 1, 25), tile[1]),
                    (tile[0], max(tile[1] - 1, 0)),
                    (tile[0], min(tile[1] + 1, 17)),
                ],
                state={"occupied_by": None},
            )
        )

    return {item.object_id: item for item in objects}
