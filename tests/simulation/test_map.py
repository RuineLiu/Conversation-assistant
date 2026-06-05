from proactive_assistant.simulation import (
    ObjectType,
    create_default_world_state,
)


def test_default_conference_room_has_required_objects() -> None:
    state = create_default_world_state()
    object_types = {item.type for item in state.objects.values()}

    assert {
        ObjectType.DOOR,
        ObjectType.CONFERENCE_TABLE,
        ObjectType.CHAIR,
        ObjectType.SCREEN,
        ObjectType.WHITEBOARD,
        ObjectType.LIGHT_SWITCH,
        ObjectType.LAPTOP,
        ObjectType.DOCUMENTS,
        ObjectType.CUP,
        ObjectType.REMOTE,
    }.issubset(object_types)


def test_table_tiles_are_not_walkable() -> None:
    state = create_default_world_state()

    assert not state.map.is_walkable((8, 8), objects=state.objects.values())
    assert state.map.is_walkable((8, 6), objects=state.objects.values())


def test_pathfinding_avoids_blocking_objects() -> None:
    state = create_default_world_state()
    path = state.map.find_path(
        (4, 14),
        (11, 6),
        objects=state.objects.values(),
    )

    assert path is not None
    assert path[0] == (4, 14)
    assert path[-1] == (11, 6)
    blocked = state.map.blocked_tiles(state.objects.values())
    assert not any(tile in blocked for tile in path)


def test_pathfinding_rejects_blocked_target() -> None:
    state = create_default_world_state()

    assert state.map.find_path((4, 14), (8, 8), objects=state.objects.values()) is None
