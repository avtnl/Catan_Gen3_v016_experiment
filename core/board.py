"""
Manages the Catan game board.

This module defines the Board class, handling the initialization of tiles, intersections,
roads, and ports. It ensures correct tile ordering (tiles[0].id == 9, etc.) for land tiles
and initializes sea tiles with type "Sea". The board state is empty initially (no settlements,
cities, or roads).
Key components:
    - Tile: Represents a hexagonal tile with type and value.
    - Intersection: Represents a vertex with building status and port info.
    - Road: Represents an edge with occupancy status.
    - Board: Manages the game board layout and state.
Dependencies:
    - typing: For type hints.
    - datetime: For retrieving a timestamp.
    - copy: For random port locations.
    - random: For random board generation.
    - core.constants: For game configuration constants.
"""
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import random
import copy

from matplotlib.pylab import tile
from sklearn.base import defaultdict
from core.constants import FNFREQ, FILENAME_FREQ, MG, FILENAME_MG

# board.py (near the top, after imports)

def pips_from_tile_value(value: int) -> float:
    """
    Classic Catan pip/dot count (number token strength).

    Returns:
        0    for desert / sea / invalid
        1    for 2 & 12
        2    for 3 & 11
        3    for 4 & 10
        4    for 5 & 9
        5    for 6 & 8
    """
    if not (2 <= value <= 12):
        return 0.0
    return 6 - abs(7 - value)

def true_probability_from_pips(pips: float) -> float:
    """Convert pip count to actual 2d6 roll probability."""
    return pips / 36.0

def true_probability_from_dice_value(value: int) -> float:
    """Direct: dice value → real probability (0–1)."""
    return pips_from_tile_value(value) / 36.0

class Edge:
    """Represents an edge of a tile on the Catan board."""
   
    def __init__(self, location: str, kind: str = "Blank", color: str = "Blank", road: List[int] = [0, 0]) -> None:
        """Initialize an Edge.
     
        Args:
            location: Edge position (e.g., 'NE', 'E', 'SE', 'SW', 'W', 'NW').
            kind: Type of structure ('Blank' or 'Road').
            color: Player color or 'Blank'.
            road: List of two intersection IDs defining the road.
        """
        self.location = location
        self.kind = kind
        self.color = color
        self.road = road


class Corner:
    """Represents a corner of a tile on the Catan board."""
   
    def __init__(self, location: str, kind: str = "Blank", color: str = "Blank", port_type: str = "Blank", intersection: int = 0) -> None:
        """Initialize a Corner.
     
        Args:
            location: Corner position (e.g., 'N', 'EH', 'EL', 'S', 'WL', 'WH').
            kind: Type of structure ('Blank', 'Settlement', 'City').
            color: Player color or 'Blank'.
            port_type: Port type if applicable (e.g., '3:1', '2:1 Brick').
            intersection: Intersection ID associated with the corner.
        """
        self.location = location
        self.kind = kind
        self.color = color
        self.port_type = port_type
        self.intersection = intersection
   

class Tile:
    """Represents a hexagonal tile on the Catan board."""
 
    def __init__(self, id_: int, type_: str = "Blank", value: int = 0, color: str = "Blank") -> None:
        """Initialize a Tile.
     
        Args:
            id_: Unique tile ID.
            type_: Resource type (e.g., 'Field', 'Desert', 'Sea').
            value: Tile number (2-12 for land tiles, 0 for Desert/Sea).
            color: Tile color (e.g., 'Blank', 'Sea', or player color).
        """
        self.id = id_
        self.type = type_
        self.value = value
        self.color = color
        self.occupied_tf = False  # To check placement of Robber
        self.current_settlements: int = 0  # Used by resource_exploration()
        self.edges: List[dict] = []
        self.corners: List[dict] = []


class Intersection:
    """Represents a vertex on the Catan board."""
   
    def __init__(self, id_: int) -> None:
        """Initialize an Intersection.
     
        Args:
            id_: Unique intersection ID.
        """
        self.id = id_
        self.face = "Blank"
        self.occupied_tf = False
        self.color = "Blank"
        self.type: Optional[str] = "Vertex"  # Default to "Vertex" for buildable intersections
        self.can_build_tf: bool = True       # Default True for land intersections
        self.three_tile_ids: List[int] = []
        self.three_tile_pips: List[float] = []
        self.three_tile_types: List[str] = []
        self.three_tile_values: List[int] = []
        self.all_tile_types: List[int] = [0, 0, 0, 0, 0]  # [Grain, Ore, Wood, Brick, Wool]
        self.all_tile_pips: List[float] = [0, 0, 0, 0, 0]
        self.all_tile_values: List[int] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.three_roads: List[Tuple[int, int]] = []
        self.three_intersection_ids: List[int] = []
        self.port_tf = False
        self.port_type = "Blank"
        self.game_round: int = 0
        self.game_turn: int = 0
        self.placement_step: int = -1


class Road:
    """Represents an edge on the Catan board."""
 
    def __init__(self, id_: Tuple[int, int]) -> None:
        """Initialize a Road.
     
        Args:
            id_: Tuple of two intersection IDs defining the road.
        """
        self.id = id_
        self.kind = "Blank"
        self.color = "Blank"
        self.occupied_tf = False
        self.two_tiles: List[List[any]] = []
        self.game_round: int = 0
        self.game_turn: int = 0
        self.placement_step: int = -1


class Board:
    """Represents the Catan game board."""
    NUM_TILES = 46
    NUM_INTERSECTIONS = 67
    LIST_OF_LAND_TILES = [9, 10, 11, 15, 16, 17, 18, 21, 22, 23, 24, 25, 28, 29, 30, 31, 35, 36, 37]
    LIST_OF_SKIPPED_TILE_IDS = [0, 1, 6, 7, 13, 33, 39, 40, 45]
    INTERSECTION_IN_WATER = [0, 1, 2, 10, 11, 12, 22, 45, 55, 56, 57, 65, 66]
    LIST_OF_PORTTYPES = ["3:1", "3:1", "3:1", "3:1", "2:1 Wheat", "2:1 Ore", "2:1 Wood", "2:1 Brick", "2:1 Wool"]
    INTERSECTIONS_ARE_PORT = [[3, 4], [6, 7], [13, 24], [20, 21], [33, 44], [35, 46], [53, 54], [58, 59], [61, 62]]
    BOARD_LAYOUT = [
        [0,1,2,3,4,5,6,0], ##### list of Tile.id's in every row
        [7,8,9,10,11,12,13],
        [0,14,15,16,17,18,19,0],
        [20,21,22,23,24,25,26],
        [0,27,28,29,30,31,32,0],
        [33,34,35,36,37,38,39],
        [0,40,41,42,43,44,45,0]
    ]
    ALL_TILE_IDS = set(range(0, 46))

    def __init__(self, board_name: str = "Base_Random") -> None:
        """Initialize the Board."""
        self.board_name = board_name
        self.round = -2
        self.turn = 1

        # Initialize empty board structures
        self.intersections = [None] * self.NUM_INTERSECTIONS
        for i in range(67):
            if i not in self.INTERSECTION_IN_WATER:
                self.intersections[i] = Intersection(i)

        self.tiles = [None] * self.NUM_TILES
        self.roads = []
        self.list_of_roads_connected_to_intersection = [[] for _ in range(67)]

        # ──────────────────────────────────────────────────────────────
        # LOAD SAVED PLAYBOARD or generate random
        # ──────────────────────────────────────────────────────────────
        from core.constants import LOAD_PLAYBOARD, SAVED_PLAYBOARD
        if LOAD_PLAYBOARD:
            print(f"📂 Loading saved playboard: {SAVED_PLAYBOARD}")
            self._add_tiles()                    # create empty tiles first
            self._add_empty_edges_and_corners()
            self._add_intersections()            # ← important: creates empty intersections
            self.load_board(SAVED_PLAYBOARD)     # overwrite tiles + ports

            # === CRITICAL: refresh ALL intersection data after loading tiles ===
            self._add_intersections()            # ← re-populates three_tile_pips, three_tile_types, etc.

            # Reconstruct roads/edges for GUI
            self._complete_edges()
            self._add_roads()

            self._create_list_of_roads_connected_to_intersection()
            self._update_intersection_types()
            self._add_three_intersection_ids()
            self._add_two_tile_attributes()

            print(f"   → Reconstructed {len(self.roads)} roads/edges and refreshed 54 intersections")

        else:
            print("🎲 Generating random board")
            if board_name == "Base_Random":
                self._get_board()
                self.save_board("")
            else:
                self._initialize_board()

        # Post-load / post-generation steps (always required)
        self._create_list_of_roads_connected_to_intersection()
        self._update_intersection_types()
        self._add_three_intersection_ids()
        self._add_two_tile_attributes()

        # Build reverse mapping for intersections → corners
        from collections import defaultdict
        self.intersection_to_corners = defaultdict(list)
        for tile in self.tiles:
            if tile:
                for corner in tile.corners:
                    iid = corner.intersection
                    if iid > 0:
                        self.intersection_to_corners[iid].append((tile, corner.location))

        # Precompute algorithm=2 raw data (needed for fallback to algo 4)
        from core.algorithms_initial_placement import InitialPlacementStrategies
        InitialPlacementStrategies.precompute_algorithm2_raw(self)

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"board.py | __init__ completed | Loaded {len(self.tiles)} tiles and {sum(1 for i in self.intersections if i and i.port_tf)} ports\n")

    def _initialize_board(self) -> None:
        """Initialize the board based on the board name."""
        self._add_tiles()
        self._add_empty_edges_and_corners()
        self._add_intersections()
        self._complete_edges()
        self._add_roads()
        self._create_list_of_roads_connected_to_intersection()
        self._add_three_tile_values()
        self._update_intersection_types()
        self._add_three_intersection_ids()
        self._add_ports()
        self._add_two_tile_attributes()
        self.load_board(self.board_name)

    def _add_tiles(self) -> None:
        """Add tiles to the board based on BOARD_LAYOUT."""
        tile_id_map = {tid: i for i, tid in enumerate(self.ALL_TILE_IDS)}
        for tile_id in self.ALL_TILE_IDS:
            idx = tile_id_map[tile_id]
            if idx not in self.LIST_OF_SKIPPED_TILE_IDS:
                if tile_id in self.LIST_OF_LAND_TILES:
                    self.tiles[idx] = Tile(tile_id)  # Land tile
                else:
                    self.tiles[idx] = Tile(tile_id, type_="Sea", value=0, color="Blank")  # Sea tile

    def _add_intersections(self) -> None:
        """Add intersections with tile associations based on BOARD_LAYOUT.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | _add_intersections\n")
   
        corner_indices = {"N": 0, "EH": 1, "EL": 2, "S": 3, "WL": 4, "WH": 5}
   
        # Process odd-numbered intersections (1, 3, 5, ...)
        for i in range(1, 67, 2):
            if i in self.INTERSECTION_IN_WATER:
                continue
            intersection = self.intersections[i]
            if intersection is None:
                continue
            intersection.three_tile_ids = []
            intersection.three_tile_pips = []
            intersection.three_tile_types = []
            intersection.three_tile_values = []
            intersection.all_tile_values = [0] * 11 # Counts for tile values 2–12
       
            # Find row and column in BOARD_LAYOUT
            for r in range(6):
                c = i - r * 11
                if i <= (r + 1) * 11:
                    break
       
            # Calculate tile IDs and corresponding corners
            if r % 2 == 0: # Even rows
                col = int((c + 1) / 2)
                tile_ids = [
                    self.BOARD_LAYOUT[r + 1][col - 1] if r + 1 < 7 and col - 1 >= 0 else 0,
                    self.BOARD_LAYOUT[r][col] if col < len(self.BOARD_LAYOUT[r]) else 0,
                    self.BOARD_LAYOUT[r + 1][col] if r + 1 < 7 and col < len(self.BOARD_LAYOUT[r + 1]) else 0
                ]
                corners = [1, 3, 5] # EH, S, WH
            else: # Odd rows
                col = int(c / 2)
                tile_ids = [
                    self.BOARD_LAYOUT[r + 1][col] if r + 1 < 7 and col < len(self.BOARD_LAYOUT[r + 1]) else 0,
                    self.BOARD_LAYOUT[r][col] if col < len(self.BOARD_LAYOUT[r]) else 0,
                    self.BOARD_LAYOUT[r + 1][col + 1] if r + 1 < 7 and col + 1 < len(self.BOARD_LAYOUT[r + 1]) else 0
                ]
                corners = [1, 3, 5] # EH, S, WH
       
            # Assign tile IDs and update tile corners
            for tile_id, corner_idx in zip(tile_ids, corners):
                if tile_id != 0:
                    intersection.three_tile_ids.append(tile_id)
                    tile = self.tiles[tile_id] if tile_id < len(self.tiles) and self.tiles[tile_id] else None
                    if tile:
                        intersection.three_tile_types.append(tile.type)
                        pips = pips_from_tile_value(tile.value)
                        intersection.three_tile_pips.append(pips)
                        if tile.type == "Field":
                            intersection.all_tile_types[0] += 1
                            intersection.all_tile_pips[0] += pips
                        elif tile.type == "Mountain":
                            intersection.all_tile_types[1] += 1
                            intersection.all_tile_pips[1] += pips
                        elif tile.type == "Forest":
                            intersection.all_tile_types[2] += 1
                            intersection.all_tile_pips[2] += pips
                        elif tile.type == "Hill":
                            intersection.all_tile_types[3] += 1
                            intersection.all_tile_pips[3] += pips
                        elif tile.type == "Pasture":
                            intersection.all_tile_types[4] += 1
                            intersection.all_tile_pips[4] += pips

                        # Assign intersection to tile corner, skip if intersection_id in INTERSECTION_IN_WATER
                        for corner in tile.corners:
                            if corner_indices[corner.location] == corner_idx and corner.intersection == 0 and i not in self.INTERSECTION_IN_WATER:
                                corner.intersection = intersection.id
                                break
   
        # Process even-numbered intersections (2, 4, 6, ...)
        for i in range(2, 67, 2):
            if i in self.INTERSECTION_IN_WATER:
                continue
            intersection = self.intersections[i]
            if intersection is None:
                continue
            intersection.three_tile_ids = []
            intersection.three_tile_pips = []
            intersection.three_tile_types = []
            intersection.three_tile_values = []
            intersection.all_tile_values = [0] * 11 # Counts for tile values 2–12
       
            # Find row and column in BOARD_LAYOUT
            for r in range(6):
                c = i - r * 11
                if i <= (r + 1) * 11:
                    break
       
            # Calculate tile IDs and corresponding corners
            if r % 2 == 0: # Even rows
                col = int(c / 2)
                tile_ids = [
                    self.BOARD_LAYOUT[r][col] if col < len(self.BOARD_LAYOUT[r]) else 0,
                    self.BOARD_LAYOUT[r + 1][col] if r + 1 < 7 and col < len(self.BOARD_LAYOUT[r + 1]) else 0,
                    self.BOARD_LAYOUT[r][col + 1] if col + 1 < len(self.BOARD_LAYOUT[r]) else 0
                ]
                corners = [2, 0, 4] # EL, N, WL
            else: # Odd rows
                col = int((c + 1) / 2)
                tile_ids = [
                    self.BOARD_LAYOUT[r][col - 1] if col - 1 >= 0 else 0,
                    self.BOARD_LAYOUT[r + 1][col] if r + 1 < 7 and col < len(self.BOARD_LAYOUT[r + 1]) else 0,
                    self.BOARD_LAYOUT[r][col] if col < len(self.BOARD_LAYOUT[r]) else 0
                ]
                corners = [2, 0, 4] # EL, N, WL
       
            # Assign tile IDs and update tile corners
            for tile_id, corner_idx in zip(tile_ids, corners):
                if tile_id != 0:
                    intersection.three_tile_ids.append(tile_id)
                    tile = self.tiles[tile_id] if tile_id < len(self.tiles) and self.tiles[tile_id] else None
                    if tile:
                        intersection.three_tile_types.append(tile.type)
                        intersection.three_tile_values.append(tile.value)
                        pips = pips_from_tile_value(tile.value)
                        intersection.three_tile_pips.append(pips)
                        if tile.value >= 2 and tile.value <= 12:
                            intersection.all_tile_values[tile.value - 2] += 1
                        if tile.type == "Field":
                            intersection.all_tile_types[0] += 1
                            intersection.all_tile_pips[0] += pips
                        elif tile.type == "Mountain":
                            intersection.all_tile_types[1] += 1
                            intersection.all_tile_pips[1] += pips
                        elif tile.type == "Forest":
                            intersection.all_tile_types[2] += 1
                            intersection.all_tile_pips[2] += pips
                        elif tile.type == "Hill":
                            intersection.all_tile_types[3] += 1
                            intersection.all_tile_pips[3] += pips
                        elif tile.type == "Pasture":
                            intersection.all_tile_types[4] += 1
                            intersection.all_tile_pips[4] += pips
                   
                        # Assign intersection to tile corner, skip if intersection_id in INTERSECTION_IN_WATER
                        for corner in tile.corners:
                            if corner_indices[corner.location] == corner_idx and corner.intersection == 0 and i not in self.INTERSECTION_IN_WATER:
                                corner.intersection = intersection.id
                                break
   
        self._create_list_of_roads_connected_to_intersection()

    def _add_empty_edges_and_corners(self) -> None:
        """Initialize empty edges and corners for each tile.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | add_empty_edges_and_corners\n")
        for tile in self.tiles:
            if tile:
                tile.edges = [
                    Edge("NE"), Edge("E"), Edge("SE"), Edge("SW"), Edge("W"), Edge("NW")
                ]
                tile.corners = [
                    Corner("N"), Corner("EH"), Corner("EL"), Corner("S"), Corner("WL"), Corner("WH")
                ]

    def _create_list_of_roads_connected_to_intersection(self) -> None:
        """Create a list of roads connected to each intersection.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | create_list_of_roads_connected_to_intersection\n")
        for i, intersection in enumerate(self.intersections):
            if intersection is None:
                self.list_of_roads_connected_to_intersection[i] = []
                continue
            help_list: List[Tuple[int, int]] = []
            for road in self.roads:
                if road: # Only check non-None roads
                    road_id = road.id
                    if road_id[0] == intersection.id or road_id[1] == intersection.id:
                        # Skip roads involving intersections in INTERSECTION_IN_WATER
                        if road_id[0] in self.INTERSECTION_IN_WATER or road_id[1] in self.INTERSECTION_IN_WATER:
                            continue
                        help_list.append(road_id)
            self.list_of_roads_connected_to_intersection[intersection.id] = help_list
            intersection.three_roads = help_list

    def _add_three_tile_values(self) -> None:
        """Add tile values for each intersection's neighboring tiles.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | add_three_tile_values\n")
   
        for intersection in self.intersections:
            if intersection is None:
                continue
            # Initialize lists for tile values
            intersection.three_tile_values = []
            # Initialize count of tile values (2 to 12, indices 0 to 10)
            NT_value = [0] * 11 # [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
       
            for tile_id in intersection.three_tile_ids:
                tile = self.tiles[tile_id] if tile_id < len(self.tiles) else None
                if tile and tile.type != "Sea":
                    intersection.three_tile_values.append(tile.value)
                    if tile.value >= 2 and tile.value <= 12:
                        NT_value[tile.value - 2] += 1

            self.intersection_to_corners = defaultdict(list)  # int → list[(tile, corner_location)]       

            intersection.all_tile_values = NT_value

    def _add_three_intersection_ids(self) -> None:
        """Add neighboring intersection IDs for each intersection based on connected roads."""
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | add_three_intersection_ids\n")

        for intersection in self.intersections:
            if intersection is None:
                continue

            # Rebuild from scratch. This method is called multiple times during
            # board generation/loading, so appending without clearing creates
            # duplicate neighbor ids.
            intersection.three_intersection_ids = []

            if self.list_of_roads_connected_to_intersection[intersection.id]:
                for road in self.list_of_roads_connected_to_intersection[intersection.id]:
                    if road[0] == intersection.id:
                        intersection.three_intersection_ids.append(road[1])
                    elif road[1] == intersection.id:
                        intersection.three_intersection_ids.append(road[0])

    def _add_two_tile_attributes(self) -> None:
        """Add tile attributes to roads based on adjacent tiles.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | add_two_tile_attributes\n")
   
        # Assign tiles to roads
        for road in self.roads:
            if road:
                road.two_tiles = []
                for tile in self.tiles:
                    if tile:
                        for edge in tile.edges:
                            if edge.road == road.id:
                                road.two_tiles.append([tile.id, tile.type, tile.value])

    def _update_intersection_types(self) -> None:
        """Update intersection types for intersections in water.
 
        Sets the type attribute to None for intersections that are not adjacent to any land tiles.
 
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | update_intersection_types\n")
        for intersection in self.intersections:
            if intersection is None:
                continue
            if intersection.id in self.INTERSECTION_IN_WATER:
                intersection.type = None # Set type to None for non-buildable intersections

    def _add_ports(self) -> None:
        """Add ports to intersections with randomized port types.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | _add_ports\n")
   
        # Shuffle port types for random assignment
        port_types = copy.copy(self.LIST_OF_PORTTYPES)
        random.shuffle(port_types)
   
        # Assign port types to intersection pairs
        for i, port_pair in enumerate(self.INTERSECTIONS_ARE_PORT):
            if i < len(port_types):
                port_type = port_types[i]
                for intersection_id in port_pair:
                    if 0 <= intersection_id < len(self.intersections) and self.intersections[intersection_id] is not None:
                        self.intersections[intersection_id].port_tf = True
                        self.intersections[intersection_id].port_type = port_type
                    else:
                        if FNFREQ == "Y":
                            with open(FILENAME_FREQ, "a") as f:
                                f.write(f"board.py | _add_ports | Invalid or None intersection ID: {intersection_id}\n")
   
        # Update tile corners for port intersections
        port_intersection_ids = []
        for portpair in self.INTERSECTIONS_ARE_PORT:
            port_intersection_ids.extend(portpair)
        for intersection in self.intersections:
            if intersection is None:
                continue
            if intersection.id in port_intersection_ids:
                for tile in self.tiles:
                    if tile:
                        for c in tile.corners:
                            if c.intersection == intersection.id:
                                c.port_type = intersection.port_type

    def _complete_edges(self) -> None:
        """Assign road IDs to tile edges based on corner intersections.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | complete_edges\n")
   
        for tile in self.tiles:
            if tile:
                list_of_corners = [corner.intersection for corner in tile.corners]
                for edge in tile.edges:
                    if edge.location == "NE":
                        road = tuple(sorted([list_of_corners[0], list_of_corners[1]]))
                        if road[0] in self.INTERSECTION_IN_WATER or road[1] in self.INTERSECTION_IN_WATER:
                            edge.road = (0, 0) # Invalid road
                        else:
                            edge.road = road
                    elif edge.location == "E":
                        road = tuple(sorted([list_of_corners[1], list_of_corners[2]]))
                        if road[0] in self.INTERSECTION_IN_WATER or road[1] in self.INTERSECTION_IN_WATER:
                            edge.road = (0, 0)
                        else:
                            edge.road = road
                    elif edge.location == "SE":
                        road = tuple(sorted([list_of_corners[2], list_of_corners[3]]))
                        if road[0] in self.INTERSECTION_IN_WATER or road[1] in self.INTERSECTION_IN_WATER:
                            edge.road = (0, 0)
                        else:
                            edge.road = road
                    elif edge.location == "SW":
                        road = tuple(sorted([list_of_corners[3], list_of_corners[4]]))
                        if road[0] in self.INTERSECTION_IN_WATER or road[1] in self.INTERSECTION_IN_WATER:
                            edge.road = (0, 0)
                        else:
                            edge.road = road
                    elif edge.location == "W":
                        road = tuple(sorted([list_of_corners[4], list_of_corners[5]]))
                        if road[0] in self.INTERSECTION_IN_WATER or road[1] in self.INTERSECTION_IN_WATER:
                            edge.road = (0, 0)
                        else:
                            edge.road = road
                    elif edge.location == "NW":
                        road = tuple(sorted([list_of_corners[5], list_of_corners[0]]))
                        if road[0] in self.INTERSECTION_IN_WATER or road[1] in self.INTERSECTION_IN_WATER:
                            edge.road = (0, 0)
                        else:
                            edge.road = road

    def _add_roads(self) -> None:
        """Add roads to the board.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | add_roads\n")
   
        added_roads = set()  # Track added road IDs to avoid duplicates
        for tile in self.tiles:
            if tile:
                for edge in tile.edges:
                    road_id = tuple(sorted(edge.road))
                    if (road_id[0] in self.INTERSECTION_IN_WATER or
                        road_id[1] in self.INTERSECTION_IN_WATER or
                        road_id == (0, 0)):  # Skip invalid roads
                        continue
                    if road_id not in added_roads:
                        self.roads.append(Road(road_id))
                        added_roads.add(road_id)
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"board.py | add_roads | Added {len(self.roads)} roads\n")

    def _is_valid_tile_value_placement(self) -> bool:
        """Check if tiles with values 6 or 8 are not adjacent by ensuring each intersection has at most one tile with value 6 or 8.
     
        Args:
            None
        Returns:
            bool: True if no tiles with values 6 or 8 are adjacent, False otherwise.
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | _is_valid_tile_value_placement\n")
   
        for intersection in self.intersections:
            if intersection is None:
                continue
            count_six_or_eight = sum(1 for value in intersection.three_tile_values if value in [6, 8])
            if count_six_or_eight > 1:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"board.py | _is_valid_tile_value_placement | Invalid: Intersection {intersection.id} has {count_six_or_eight} tiles with values 6 or 8: {intersection.three_tile_values}\n")
                return False
        return True

    def _get_board(self) -> None:
        """Generate a random board ensuring tiles with values 6 and 8 are not adjacent.
     
        Args:
            None
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | _get_board\n")
   
        tile_types = ["Field"] * 4 + ["Mountain"] * 3 + ["Forest"] * 4 + ["Hill"] * 3 + ["Pasture"] * 4 + ["Desert"]
        tile_values = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        max_attempts = 100 # Limit retries to prevent infinite loops
        attempt = 0
   
        while attempt < max_attempts:
            # Reset intersections to clear previous attempt's state
            self.intersections = [None] * 67
            for i in range(67):
                if i not in self.INTERSECTION_IN_WATER:
                    self.intersections[i] = Intersection(i)
            self.tiles = [None] * len(self.ALL_TILE_IDS)
            self.roads = [] # Reset roads to empty list
            self.list_of_roads_connected_to_intersection = [[] for _ in range(67)]
       
            # Shuffle tile types
            random.shuffle(tile_types)
            current_tile_values = tile_values.copy()
            random.shuffle(current_tile_values)
       
            # Assign tile types and values
            type_index = 0
            tile_id_map = {tid: i for i, tid in enumerate(self.ALL_TILE_IDS)}
            for tile_id in self.ALL_TILE_IDS:
                idx = tile_id_map[tile_id]
                if tile_id in self.LIST_OF_LAND_TILES:
                    if type_index < len(tile_types):
                        random_tile_type = tile_types[type_index]
                        type_index += 1
                        if random_tile_type == "Desert":
                            random_tile_value = 0
                        else:
                            random_tile_value = current_tile_values.pop(0) if current_tile_values else 0
                        self.tiles[idx] = Tile(tile_id, random_tile_type, random_tile_value, "Blank")
                elif tile_id in self.LIST_OF_SKIPPED_TILE_IDS:
                    continue
                else:
                    self.tiles[idx] = Tile(tile_id, type_="Sea", value=0, color="Blank")
       
            # Initialize intersections for tile value validation
            self._add_empty_edges_and_corners()
            self._add_intersections()
            self._complete_edges()
       
            # Check if tile value placement is valid
            if self._is_valid_tile_value_placement():
                break
       
            attempt += 1
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"board.py | _get_board | Attempt {attempt} failed: Retrying tile value placement\n")
   
        if attempt >= max_attempts:
            raise RuntimeError("Failed to generate a valid board after maximum attempts: 6 and 8 tiles are adjacent")
   
        # Complete board initialization
        self._add_roads()
        self._create_list_of_roads_connected_to_intersection()
        self._update_intersection_types()
        self._add_three_intersection_ids()
        self._add_ports()
        self._add_two_tile_attributes()

    def save_board(self, filename_save: str = "") -> None:
        """Save the board's tile and port data to a file.
     
        Args:
            filename_save: Optional filename for saving; if empty, uses a timestamp-based name.
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("board.py | save_board\n")
        if filename_save == "":
            now = datetime.now()
            today = now.strftime("%d_%b_%Y_%H_%M_%S")
            f = open("PlayBoard " + str(today) + ".txt", "w")
        else:
            f = open("PlayBoard " + str(filename_save) + ".txt", "w")
        for tile in self.tiles:
            if tile:
                f.write(f"{tile.id}\n")
                f.write(f"{tile.type}\n")
                f.write(f"{tile.value}\n")
        for intersection in self.intersections:
            if intersection and intersection.port_tf:
                f.write(f"{intersection.id}\n")
                f.write(f"{intersection.port_type}\n")
        f.close()

    def load_board(self, board_name: str) -> None:
        """Load a saved PlayBoard file.

        Expected save_board() format:
            tile_id
            tile_type
            tile_value
            ...
            port_intersection_id
            port_type
            ...

        Why this parser is deliberately strict:
        - Port rows such as ``3 / 3:1 / 4`` can accidentally look like a
          tile block if we only parse ``int / str / int``.
        - Therefore a block is accepted as a tile only when the middle line is
          a valid terrain type.
        - Once a non-terrain middle line is found, parsing switches to ports.
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write(f"board.py | load_board | Loading {board_name}\n")

        valid_tile_types = {
            "Sea",
            "Desert",
            "Mountain",
            "Hill",
            "Forest",
            "Pasture",
            "Field",
        }
        valid_port_types = set(self.LIST_OF_PORTTYPES)

        try:
            with open(board_name, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            print(f"📄 Loaded {len(lines)} non-empty lines from {board_name}")

            idx = 0
            tiles_loaded = 0
            ports_loaded = 0

            # Reset any previous port state. This makes load_board safe even if
            # it is called after _add_ports() or after loading another board.
            for inter in self.intersections:
                if inter is not None:
                    inter.port_tf = False
                    inter.port_type = "Blank"

            for tile in self.tiles:
                if tile:
                    for corner in tile.corners:
                        corner.port_type = "Blank"

            # ──────────────────────────────────────────────────────────
            # 1. Load tile blocks: id / type / value
            # ──────────────────────────────────────────────────────────
            while idx + 2 < len(lines):
                try:
                    tile_id = int(lines[idx])
                    tile_type = lines[idx + 1]

                    # Critical guard:
                    # If the middle line is not a terrain type, we reached
                    # the port section. Example: 3 / 3:1 / 4.
                    if tile_type not in valid_tile_types:
                        break

                    tile_value = int(lines[idx + 2])

                except ValueError:
                    # Malformed tile block or start of port section.
                    break

                updated = False
                for tile in self.tiles:
                    if tile and tile.id == tile_id:
                        tile.type = tile_type
                        tile.value = tile_value
                        tile.color = "Blank"
                        tile.occupied_tf = False
                        tiles_loaded += 1
                        updated = True
                        break

                if not updated:
                    print(f"⚠️  Tile id {tile_id} not found in board structure")

                idx += 3

            # ──────────────────────────────────────────────────────────
            # 2. Load port pairs: intersection_id / port_type
            # ──────────────────────────────────────────────────────────
            while idx + 1 < len(lines):
                try:
                    inter_id = int(lines[idx])
                except ValueError:
                    print(f"⚠️  Skipping malformed port intersection id: {lines[idx]}")
                    idx += 1
                    continue

                port_type = lines[idx + 1]

                if port_type not in valid_port_types:
                    print(f"⚠️  Invalid port type for intersection {inter_id}: {port_type}")
                    idx += 2
                    continue

                if 0 <= inter_id < len(self.intersections) and self.intersections[inter_id] is not None:
                    inter = self.intersections[inter_id]
                    inter.port_tf = True
                    inter.port_type = port_type
                    ports_loaded += 1
                else:
                    print(f"⚠️  Invalid intersection id {inter_id} for port")

                idx += 2

            # Sync loaded intersection port data back to tile corners.
            for tile in self.tiles:
                if not tile:
                    continue
                for corner in tile.corners:
                    iid = corner.intersection
                    if 0 <= iid < len(self.intersections):
                        inter = self.intersections[iid]
                        if inter and inter.port_tf:
                            corner.port_type = inter.port_type
                        else:
                            corner.port_type = "Blank"

            print(f"✅ Successfully loaded board from {board_name}")
            print(f"   • {tiles_loaded} tiles updated")
            print(f"   • {ports_loaded} port entries processed")

            if ports_loaded != 18:
                print(
                    f"⚠️  Expected 18 port entries, loaded {ports_loaded}. "
                    "Check the PlayBoard file if this is unexpected."
                )

            # Re-run necessary post-load steps. These methods are now safe to
            # call repeatedly because _add_three_intersection_ids clears before
            # rebuilding neighbor ids.
            self._create_list_of_roads_connected_to_intersection()
            self._update_intersection_types()
            self._add_three_intersection_ids()
            self._add_two_tile_attributes()

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"board.py | load_board | Successfully loaded {board_name} "
                        f"({tiles_loaded} tiles + {ports_loaded} ports)\n"
                    )

        except FileNotFoundError:
            print(f"❌ load_board: File not found → {board_name}")
            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(f"board.py | load_board | File not found: {board_name}\n")

        except Exception as e:
            print(f"❌ load_board error: {e}")
            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(f"board.py | load_board | ERROR: {e}\n")

    def occupy_intersection(self, intersection_id: int, kind: str, color: str, 
                           placement_step: int = -1) -> None:
        """Occupy an intersection with a settlement or city.
        
        Board-only operation:
        - Updates intersection state
        - Blocks adjacent intersections
        - Updates tile corners and current_settlement count
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write(f"board.py | occupy_intersection\n")

        if intersection_id < 0 or intersection_id >= len(self.intersections):
            return
        inter = self.intersections[intersection_id]
        if inter is None:
            return

        # Update intersection
        inter.occupied_tf = True
        inter.face = kind
        inter.can_build_tf = False
        inter.color = color
        inter.game_round = self.round
        inter.game_turn = self.turn
        inter.placement_step = placement_step

        # Block adjacent intersections
        self._block_adjacent_intersections(intersection_id)
        if self.round >= 0:  # permanent block after setup phase
            self._block_adjacent_intersections(intersection_id)

        # Update tile corners + increment settlement count per affected tile
        affected_tiles = self.intersection_to_corners.get(intersection_id, [])
        for tile, corner_loc in affected_tiles:
            corner = next((c for c in tile.corners if c.location == corner_loc), None)
            if corner is not None:
                corner.kind = kind
                corner.color = color
                tile.current_settlements += 1   # important for later resource calculations

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"board.py | occupy_intersection | {kind} at {intersection_id} by {color} "
                        f"(step={placement_step})\n")


    def occupy_road(self, road_id: Tuple[int, int], kind: str, color: str, placement_step: int = -1) -> None:
        """Occupy a road."""
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write(f"board.py | occupy_road\n")

        road_id_tuple = tuple(sorted(road_id))

        for road in self.roads:
            if road and road.id == road_id_tuple:
                road.occupied_tf = True
                road.kind = kind
                road.color = color
                road.game_round = self.round
                road.game_turn = self.turn
                road.placement_step = placement_step   # Fixed
                return

        # New road
        new_road = Road(road_id_tuple)
        new_road.occupied_tf = True
        new_road.kind = kind
        new_road.color = color
        new_road.game_round = self.round
        new_road.game_turn = self.turn
        new_road.placement_step = placement_step   # Fixed
        self.roads.append(new_road)

        for tile in self.tiles:
            if tile:
                for edge in tile.edges:
                    if tuple(sorted(edge.road)) == road_id_tuple:
                        edge.kind = kind
                        edge.color = color

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"board.py | occupy_road | {road_id_tuple} {kind} {color} (step={placement_step})\n")

    def can_build_road_for_color_tf(self, road_id: List[int], color: str) -> bool:
        """Check if a road can be built by a player.
     
        Args:
            road_id: List of two intersection IDs.
            color: Player color.
        Returns:
            str: True if buildable, False if occupied or if invalid.
        """
        road_id_tuple = tuple(sorted(road_id))
        # Check if road is valid by looking at tile edges
        valid_road = False
        for tile in self.tiles:
            if tile:
                for edge in tile.edges:
                    if tuple(sorted(edge.road)) == road_id_tuple:
                        valid_road = True
                        break
            if valid_road:
                break
   
        if not valid_road or road_id_tuple[0] in self.INTERSECTION_IN_WATER or road_id_tuple[1] in self.INTERSECTION_IN_WATER:
            return False
   
        # Check if road is already occupied
        for road in self.roads:
            if road and road.id == road_id_tuple and road.occupied_tf:
                return False
   
        return valid_road and not any(
            r and r.id == road_id_tuple and r.occupied_tf for r in self.roads
        )

    def _block_adjacent_intersections(self, intersection_id: int) -> None:
        """Block adjacent intersections from being built on."""
        intersection = self.intersections[intersection_id]
        if intersection:
            for neighbor_id in intersection.three_intersection_ids:
                if (neighbor_id not in self.INTERSECTION_IN_WATER and
                        self.intersections[neighbor_id] is not None and
                        self.intersections[neighbor_id].can_build_tf):
                    self.intersections[neighbor_id].can_build_tf = False
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            f.write(f"board.py | _block_adjacent_intersections | "
                                    f"Blocked neighbor {neighbor_id} for {intersection_id}\n")

    def write_debug_info(self) -> None:
        """Write all board, tile, intersection, and road attributes to FILENAME_MG for debugging."""
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write("board.py | write_debug_info | Board\n")
                board_dict = {k: v for k, v in self.__dict__.items() if k not in ['intersections', 'roads', 'tiles', 'list_of_roads_connected_to_intersection']}
                f.write(f"Board Attributes: {board_dict}\n")
                f.write(f" Intersection IDs: {[i.id for i in self.intersections if i is not None]}\n")
                f.write(f" Road IDs: {[road.id for road in self.roads if road]}\n")
                f.write(f" Tile IDs: {[tile.id for tile in self.tiles if tile]}\n")
           
                f.write("board.py | write_debug_info | Tiles\n")
                for tile in self.tiles:
                    if tile:
                        tile_dict = {k: v for k, v in tile.__dict__.items() if k not in ['edges', 'corners']}
                        f.write(f"Tile Attributes: {tile_dict}\n")
                        f.write(" Edges:\n")
                        for edge in tile.edges:
                            f.write(f" Edge (road={edge.road}): {edge.__dict__}\n")
                        f.write(" Corners:\n")
                        for corner in tile.corners:
                            f.write(f" Corner (intersection={corner.intersection}): {corner.__dict__}\n")
                    else:
                        f.write("Tile: None\n")
           
                f.write("board.py | write_debug_info | Intersections\n")
                for i, intersection in enumerate(self.intersections):
                    if intersection is None:
                        f.write(f"Intersection: None\n")
                    else:
                        intersection_dict = intersection.__dict__.copy()
                        f.write(f"Intersection Attributes: {intersection_dict}\n")
           
                f.write("board.py | write_debug_info | Roads\n")
                for road in self.roads:
                    if road:
                        f.write(f"Road Attributes: {road.__dict__}\n")
                    else:
                        f.write("Road: None\n")

    def _distance_between_intersections(self, id1: int, id2: int) -> int:
        """Return shortest road distance between two intersections (BFS). Used for initial placement distance-2 rule."""
        if id1 == id2:
            return 0
        from collections import deque
        visited = set()
        queue = deque([(id1, 0)])
        while queue:
            curr, dist = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            inter = self.intersections[curr]
            if inter:
                for nid in inter.three_intersection_ids:
                    if nid == id2:
                        return dist + 1
                    if nid not in visited and nid not in self.INTERSECTION_IN_WATER:
                        queue.append((nid, dist + 1))
        return 999  # unreachable

    def _get_settlement_intersections_on_tile(self, tile_id: int) -> list[int]:
        """Return list of intersection IDs that have a settlement or city on this tile."""
        inter_ids = []
        tile = self.tiles[tile_id] if 0 <= tile_id < len(self.tiles) else None
        if not tile:
            return inter_ids
        for corner in tile.corners:
            iid = corner.intersection
            if iid > 0 and self.intersections[iid] is not None:
                inter = self.intersections[iid]
                if inter.occupied_tf and inter.face in ("Settlement", "City"):
                    inter_ids.append(iid)
        return inter_ids

    def get_current_settlement_pips(self) -> dict[str, float]:
        """Sum pips from tiles touched by at least one settlement/city (cities count as 1  NOT as 2)."""
        resource_map = {
            "Field":   "wheat",
            "Mountain": "ore",
            "Forest":  "wood",
            "Hill":    "brick",
            "Pasture": "wool"
        }
        current = {res: 0.0 for res in resource_map.values()}
        visited = set()

        for inter in self.intersections:
            if inter and inter.occupied_tf and inter.face in ("Settlement", "City"):
                for tid in inter.three_tile_ids:
                    if tid in visited:
                        continue
                    tile = self.tiles[tid] if 0 <= tid < len(self.tiles) else None
                    if tile and tile.type in resource_map:
                        p = pips_from_tile_value(tile.value)
                        if p > 0:
                            current[resource_map[tile.type]] += p
                            visited.add(tid)
        return current

    def resource_exploration(self) -> dict[str, dict[str, float]]:
        """
        Approximate remaining pip potential per resource (min / max).
        
        Uses:
        - 2.75 avg settlements per tile when 0–1 real settlement present
        - When 2 real settlements: check distance between them
        → allow 1 more if distance >= 2, else 0 more
        - When 3: 0 remaining
        """
        resource_map = {
            "Field":   "wheat",
            "Mountain": "ore",
            "Forest":  "wood",
            "Hill":    "brick",
            "Pasture": "wool"
        }

        totals = {res: {"min": 0.0, "max": 0.0} for res in resource_map.values()}

        for tile in self.tiles:
            if not tile or tile.type not in resource_map:
                continue

            pips = pips_from_tile_value(tile.value)
            if pips == 0:
                continue

            res = resource_map[tile.type]
            count = tile.current_settlements

            if count >= 3:
                min_mult = max_mult = 0.0

            elif count <= 1:
                min_mult = 2.0
                max_mult = 2.75

            else:  # exactly 2 settlements → check if third is possible
                inter_ids = self._get_settlement_intersections_on_tile(tile.id)
                if len(inter_ids) != 2:
                    # inconsistency → be conservative
                    min_mult = max_mult = 0.0
                else:
                    dist = self._distance_between_intersections(inter_ids[0], inter_ids[1])
                    if dist >= 2:
                        min_mult = max_mult = 1.0   # one more possible
                    else:
                        min_mult = max_mult = 0.0   # blocked

            contrib_min = pips * min_mult
            contrib_max = pips * max_mult

            totals[res]["min"] += contrib_min
            totals[res]["max"] += contrib_max

        # Optional: round for nicer output
        for res in totals:
            totals[res]["min"] = round(totals[res]["min"], 1)
            totals[res]["max"] = round(totals[res]["max"], 1)

        return totals
    
    def get_vertex_to_rolls(self) -> dict[int, list[list[int]]]:
        """Maps each intersection ID to the dice numbers that produce each resource.
        
        Resource order used by MarkovEvaluator (must match internal matrix):
            0 = Brick   (Hill)
            1 = Wood    (Forest)
            2 = Wool    (Pasture)
            3 = Wheat   (Field)
            4 = Ore     (Mountain)
        
        Uses your exact constants.py terrain names. Works with both three_tile_ids and legs.
        """
        terrain_to_idx = {
            "Hill":     0,   # Brick
            "Forest":   1,   # Wood
            "Pasture":  2,   # Wool
            "Field":    3,   # Wheat
            "Mountain": 4    # Ore
        }

        vertex_to_rolls: dict[int, list[list[int]]] = {}
        for inter in self.intersections:
            if inter is None or not hasattr(inter, 'id'):
                continue
            vid = inter.id
            rolls = [[] for _ in range(5)]   # exactly 5 resources

            # Support both storage styles used in your Board
            if hasattr(inter, 'three_tile_ids') and inter.three_tile_ids is not None:
                tile_ids = inter.three_tile_ids
            elif hasattr(inter, 'legs') and inter.legs:
                tile_ids = [getattr(leg, 'tile_id', None) for leg in inter.legs 
                           if getattr(leg, 'tile_id', None) is not None]
            else:
                tile_ids = []

            for tile_id in tile_ids:
                if not (0 <= tile_id < len(self.tiles)):
                    continue
                tile = self.tiles[tile_id]
                if tile is None:
                    continue
                ttype = getattr(tile, 'type', None)
                value = getattr(tile, 'value', 0)
                if ttype in terrain_to_idx and value > 0:
                    idx = terrain_to_idx[ttype]
                    rolls[idx].append(value)

            vertex_to_rolls[vid] = [sorted(r) for r in rolls]

        print(f"✅ get_vertex_to_rolls generated for {len(vertex_to_rolls)} intersections")
        return vertex_to_rolls