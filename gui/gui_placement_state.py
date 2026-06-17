"""Placement states for human guidance system."""

from enum import Enum, auto


class PlacementState(Enum):
    IDLE = auto()
    CHOOSING_SETTLEMENT = auto()
    SETTLEMENT_SELECTED = auto()
    CHOOSING_ROAD = auto()
    ROAD_SELECTED = auto()