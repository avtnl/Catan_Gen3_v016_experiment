from typing import Dict, List, Tuple, Any
import math

from core.constants import (
    PlayerColor,
    ResourceCard,
    RESOURCE_ORDER,
    COSTS,
    FNFREQ,
    FILENAME_FREQ,
    MG,
    FILENAME_MG,
)
from core.board import Board


class Player:
    """Represents a player in the Catan game."""

    def __init__(
        self,
        id_: int,
        color: str,
        sequence: int,
        is_human: bool = False,
        initial_placement_algorithm: int = 1,
        human_like_placement: bool = True,
    ) -> None:
        valid_colors = [pc.color_name for pc in PlayerColor]
        if color not in valid_colors:
            raise ValueError(f"Invalid color: {color}. Must be one of {valid_colors}")

        self.game = None
        self.color = color
        self.color2 = color
        self.id = id_
        self.is_human = is_human
        self.gameover_tf = False
        self.sequence = sequence
        self.initial_placement_algorithm = initial_placement_algorithm
        self.human_like_placement = human_like_placement

        self.points = 0
        self.victory_points: int = 0

        self.longest_route_tf = False
        self.size_longest_route = 0
        self.structure_longest_route: List[Tuple[int, int]] = []

        self.number_of_clusters = 0
        self.structure_of_clusters: List = []

        self.largest_army_tf = False
        self.size_largest_army = 0

        self.number_of_rcards = 0
        self.number_of_dcards = 0

        self.rcards: Dict[ResourceCard, int] = {rc: 0 for rc in ResourceCard}

        self.dcard_summary = [
            ["victory_point", 0, 0, 0],
            ["knight", 0, 0, 0],
            ["two_free_roads", 0, 0, 0],
            ["year_of_plenty", 0, 0, 0],
            ["monopoly", 0, 0, 0],
        ]
        self.development_cards: List[str] = []

        self.settlements: List[int] = []
        self.cities: List[int] = []
        self.roads: List[Tuple[int, int]] = []

        self.turn_details_resource_production = [0, 0, 0, 0, 0, 0]
        self.turn_details_resource_production_robber = [0, 0, 0, 0, 0, 0]
        self.turn_details_buy = [0, 0, 0, 0, 0, 0]
        self.turn_details_steal = [0, 0, 0, 0, 0, 0]
        self.turn_details_discard = [0, 0, 0, 0, 0, 0]
        self.turn_details_TwP = [0, 0, 0, 0, 0, 0]
        self.turn_details_last_TwPdeal = [0, 0, 0, 0, 0, 0]
        self.turn_details_TwB = [0, 0, 0, 0, 0, 0]
        self.turn_details_dcard = [0, 0, 0, 0, 0, 0]

        self.trade_rates: Dict[ResourceCard, int] = {rc: 4 for rc in ResourceCard}
        self.port_access: Dict[str, bool] = {
            "3:1": False,
            "2:1 Wheat": False,
            "2:1 Ore": False,
            "2:1 Wood": False,
            "2:1 Brick": False,
            "2:1 Wool": False,
        }

        self.ff_resource_buffer: Dict[ResourceCard, float] = {
            rc: 0.0 for rc in ResourceCard
        }

        self.last_action: str = "None"

    def _ignore_resource_cards(self) -> bool:
        """
        Return True when fast-forward should ignore actual owned resource cards.
        """
        game = getattr(self, "game", None)
        if game is not None and hasattr(game, "ff_ignore_resource_cards"):
            return bool(game.ff_ignore_resource_cards)
        return False

    def add_rcard(self, resource: ResourceCard, amount: int) -> None:
        """Add resource cards to the player's hand."""
        if amount <= 0:
            return
        self.rcards[resource] = self.rcards.get(resource, 0) + amount
        self.number_of_rcards = sum(self.rcards.get(rc, 0) for rc in ResourceCard)

    def remove_rcard(self, resource: ResourceCard, amount: int) -> bool:
        """Remove resource cards if possible."""
        if amount < 0:
            return False
        current = self.rcards.get(resource, 0)
        if current < amount:
            return False
        self.rcards[resource] = current - amount
        self.number_of_rcards = sum(self.rcards.get(rc, 0) for rc in ResourceCard)
        return True

    def can_afford(self, structure: str) -> bool:
        """
        Check if the player has enough resource cards to build a structure.

        When fast-forward is configured to ignore resource cards, always return True.
        """
        if self._ignore_resource_cards():
            return True

        costs = COSTS.get(structure, {})
        return all(self.rcards.get(res, 0) >= amt for res, amt in costs.items())

    def _spend_cost(self, structure: str) -> None:
        """Deduct the resource cost for a structure."""
        for rc, amt in COSTS.get(structure, {}).items():
            self.rcards[rc] = self.rcards.get(rc, 0) - amt
        self.number_of_rcards = sum(self.rcards.get(rc, 0) for rc in ResourceCard)

    def build_structure(self, structure: str, location: int | Tuple[int, int], board: "Board") -> bool:
        """
        Build a structure on the board and update player state.

        Correct behavior:
        - settlement: build on an empty legal intersection
        - city: upgrade an existing owned settlement
        - road: build on a legal road edge

        When fast-forward is configured to ignore resource cards:
        - do NOT reject due to affordability
        - do NOT deduct resource costs

        Global piece limits enforced here:
        - no 6th occupied settlement/city site
        - no 5th city
        """
        if not self.can_afford(structure):
            return False

        if structure == "settlement":
            if self.game is None:
                return False
            if not isinstance(location, int):
                return False

            # Hard piece-limit guard:
            # in this codebase, settlements + cities together represent occupied building sites.
            if (len(self.settlements) + len(self.cities)) >= 5:
                return False

            if not self.game.can_build_intersection_tf(location, self):
                return False

            board.occupy_intersection(location, "Settlement", self.color)

            if location not in self.settlements:
                self.settlements.append(location)

            self.victory_points += 1
            self.points = self.victory_points

            try:
                self.update_trade_rates(board)
            except Exception:
                pass

        elif structure == "city":
            if not isinstance(location, int):
                return False

            # Hard piece-limit guard: no 5th city
            if len(self.cities) >= 4:
                return False

            if location not in self.settlements:
                return False

            inter = board.intersections[location]
            if inter is None:
                return False
            if getattr(inter, "face", None) != "Settlement":
                return False

            board.occupy_intersection(location, "City", self.color)

            self.settlements.remove(location)
            if location not in self.cities:
                self.cities.append(location)

            # Net +1 VP on upgrade (settlement 1 -> city 2)
            self.victory_points += 1
            self.points = self.victory_points

            try:
                self.update_trade_rates(board)
            except Exception:
                pass

        elif structure == "road":
            if not isinstance(location, tuple) or len(location) != 2:
                return False

            road_id = tuple(sorted(location))
            if not board.can_build_road_for_color_tf(list(road_id), self.color):
                return False

            board.occupy_road(road_id, "Road", self.color)
            if road_id not in self.roads:
                self.roads.append(road_id)

        elif structure == "development_card":
            # Card draw itself happens elsewhere
            pass

        else:
            return False

        # Deduct resources only in realistic mode
        if not self._ignore_resource_cards():
            for rc, amt in COSTS[structure].items():
                self.rcards[rc] -= amt

        self.number_of_rcards = sum(self.rcards.get(rc, 0) for rc in ResourceCard)
        self.last_action = f"Built {structure} at {location}"

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"player.py | build_structure | {structure} at {location} by player {self.id} "
                    f"(settlements: {len(self.settlements)}, cities: {len(self.cities)}, roads: {len(self.roads)})\n"
                )

        return True

    def update_trade_rates(self, board: "Board") -> None:
        """Recompute harbor access and trade rates from current owned structures."""
        self.trade_rates = {rc: 4 for rc in ResourceCard}
        self.port_access = {
            "3:1": False,
            "2:1 Wheat": False,
            "2:1 Ore": False,
            "2:1 Wood": False,
            "2:1 Brick": False,
            "2:1 Wool": False,
        }

        for intersection_id in self.settlements + self.cities:
            if not (0 <= intersection_id < len(board.intersections)):
                continue
            intersection = board.intersections[intersection_id]
            if intersection is None:
                continue
            if getattr(intersection, "port_tf", False):
                port_type = getattr(intersection, "port_type", "Blank")
                if port_type in self.port_access:
                    self.port_access[port_type] = True

        for port, has_access in self.port_access.items():
            if not has_access:
                continue

            if port == "3:1":
                for rc in ResourceCard:
                    self.trade_rates[rc] = min(self.trade_rates[rc], 3)

            elif port.startswith("2:1 "):
                resource_name = port.split(" ", 1)[1].strip()
                resource = next((r for r in ResourceCard if r.value == resource_name), None)
                if resource is not None:
                    self.trade_rates[resource] = min(self.trade_rates[resource], 2)

    def get_resource_production_probability(self, board: "Board") -> Dict[ResourceCard, float]:
        """
        Return current production strength in pips/dots per resource.

        This uses the board's current all_tile_pips layout, which is already aligned with:
        [Wheat, Ore, Wood, Brick, Wool].
        """
        probabilities: Dict[ResourceCard, float] = {rc: 0.0 for rc in ResourceCard}

        for intersection_id in self.settlements + self.cities:
            if not (0 <= intersection_id < len(board.intersections)):
                continue
            inter = board.intersections[intersection_id]
            if inter is None:
                continue

            inter_pips = getattr(inter, "all_tile_pips", [0.0] * 5)
            multiplier = 2 if intersection_id in self.cities else 1

            for idx, rc in enumerate(RESOURCE_ORDER):
                probabilities[rc] += self._safe_float(inter_pips[idx]) * multiplier

        return probabilities

    def rcards_in_hand(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Retrieve resource cards in hand and trade ratios.

        Returns:
            (
                [Wheat, Ore, Wood, Brick, Wool],
                [trade ratios...],
                [floor(count / ratio)...]
            )
        """
        if FNFREQ == "Y" and self.game is not None:
            with open(FILENAME_FREQ, "a") as f:
                f.write(f"{self.game.sequence_number} | {self.game.state} | player.py | rcards_in_hand\n")

        ordered_resources = [
            ResourceCard.WHEAT,
            ResourceCard.ORE,
            ResourceCard.WOOD,
            ResourceCard.BRICK,
            ResourceCard.WOOL,
        ]

        rcards5 = [self.rcards.get(res, 0) for res in ordered_resources]
        trade_ratio = [self.trade_rates.get(res, 4) for res in ordered_resources]
        trade_ratio_in_rcards5 = [
            int(math.floor(rcards5[i] / trade_ratio[i])) if trade_ratio[i] > 0 else 0
            for i in range(5)
        ]

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(
                    f"player.py | rcards_in_hand for Player: {self.id} | "
                    f"rcards5: {rcards5} | TR: {trade_ratio} | TRinR5: {trade_ratio_in_rcards5}\n"
                )

        return rcards5, trade_ratio, trade_ratio_in_rcards5

    def write_debug_info(self) -> None:
        """Write player attributes to the debug log."""
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"player.py | write_debug_info | Player ID: {self.id}\n")
                f.write(f" Color: {self.color}, Sequence: {self.sequence}\n")
                f.write(f" Victory Points: {self.victory_points}, Points: {self.points}\n")
                f.write(
                    f" Longest Route: {self.longest_route_tf}, Size: {self.size_longest_route}, "
                    f"Structure: {self.structure_longest_route}\n"
                )
                f.write(f" Largest Army: {self.largest_army_tf}, Size: {self.size_largest_army}\n")
                f.write(
                    f" Number of Resource Cards: {self.number_of_rcards}, "
                    f"Number of Dev Cards: {self.number_of_dcards}\n"
                )
                f.write(f" Resource Cards: {self.rcards}\n")
                f.write(f" Development Cards: {self.development_cards}, DCard Summary: {self.dcard_summary}\n")
                f.write(f" Settlements: {self.settlements}, Cities: {self.cities}, Roads: {self.roads}\n")
                f.write(f" Port Access: {self.port_access}\n")
                f.write(f" Last Action: {self.last_action}\n")
                rcards, trade_ratio, trade_ratio_in_rcards = self.rcards_in_hand()
                f.write(
                    f" Resource Cards in Hand: {rcards}, Trade Ratio: {trade_ratio}, "
                    f"Trade Ratio in RCards: {trade_ratio_in_rcards}\n"
                )

    @staticmethod
    def _safe_float(val: Any) -> float:
        """Safe conversion to float, returns 0.0 on failure."""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def has_harbor(self) -> bool:
        """Return True if the player owns at least one settlement or city on a harbor."""
        if self.game is None:
            return False

        for inter_id in self.settlements + self.cities:
            if not (0 <= inter_id < len(self.game.board.intersections)):
                continue
            inter = self.game.board.intersections[inter_id]
            if inter and (getattr(inter, "port_tf", False) or getattr(inter, "harborYN", "N") == "Y"):
                return True
        return False

    def get_current_production_pips(self, board: "Board") -> List[float]:
        """
        Return production pips in RESOURCE_ORDER:
        [Wheat, Ore, Wood, Brick, Wool]
        """
        pips = [0.0] * 5

        for inter_id in self.settlements + self.cities:
            if not (0 <= inter_id < len(board.intersections)):
                continue
            inter = board.intersections[inter_id]
            if inter is None:
                continue

            inter_pips = getattr(inter, "all_tile_pips", [0.0] * 5)
            multiplier = 2 if inter_id in self.cities else 1

            for idx in range(5):
                pips[idx] += self._safe_float(inter_pips[idx]) * multiplier

        return pips
    
    def build_structure_initial_placement(
        self,
        structure: str,
        location: int | Tuple[int, int],
        board: "Board",
        placement_step: int = -1,
    ) -> bool:
        """
        Build a structure during Initial Placement.

        Differences vs build_structure():
        - no resource-card affordability checks
        - no resource deduction
        - only supports settlement and road
        - keeps Player state in sync with Board state
        """
        if structure == "settlement":
            if self.game is None:
                return False
            if not isinstance(location, int):
                return False

            if not self.game.can_build_intersection_tf(location, self):
                return False

            board.occupy_intersection(
                location,
                "Settlement",
                self.color,
                placement_step=placement_step,
            )

            if location not in self.settlements:
                self.settlements.append(location)

            self.victory_points += 1
            self.points = self.victory_points

            try:
                self.update_trade_rates(board)
            except Exception:
                pass

        elif structure == "road":
            if not isinstance(location, tuple) or len(location) != 2:
                return False

            road_id = tuple(sorted(location))
            if not board.can_build_road_for_color_tf(list(road_id), self.color):
                return False

            board.occupy_road(
                road_id,
                "Road",
                self.color,
                placement_step=placement_step,
            )

            if road_id not in self.roads:
                self.roads.append(road_id)

        else:
            return False

        self.number_of_rcards = sum(self.rcards.get(rc, 0) for rc in ResourceCard)
        self.last_action = f"Initial placement: built {structure} at {location}"

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"player.py | build_structure_initial_placement | "
                    f"{structure} at {location} by player {self.id} "
                    f"(settlements: {len(self.settlements)}, cities: {len(self.cities)}, roads: {len(self.roads)})\n"
                )

        return True   