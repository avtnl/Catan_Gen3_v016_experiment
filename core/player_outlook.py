"""
core/player_outlook.py

PlayerOutlook - Forward-looking strategy and decision engine for one Catan player.
Fast-forward friendly, with safe fallbacks if the older outlook helper functions
are not currently imported in this module.
"""

from typing import Any, List

from core.constants import MG, FILENAME_MG
from core.algorithms_initial_placement import InitialPlacementStrategies


class PlayerOutlook:
    """Manages all forward-looking calculations and decisions for a single player."""

    def __init__(self, player: "Player", game: "Game"):
        self.player = player
        self.game = game

        # -------------------------------------------------------------------
        # Core outlook fields (kept for compatibility with your older code)
        # -------------------------------------------------------------------
        self.summary = "BBBBBBBBB"
        self.next_road1: List = []
        self.ratio_road = "B"
        self.next_road2: List = []
        self.next_road3: List = []
        self.next_settlement1: List = [0, 0]
        self.next_settlement2: List = [0, 0]
        self.next_settlement_to_city1: List = [0, 0]
        self.next_settlement_to_city2: List = [0, 0]
        self.new_settlement1: List = [0, 0]
        self.distance_new_settlement1: List = [0, []]
        self.new_settlement2: List = [0, 0]
        self.distance_new_settlement2: List = [0, []]
        self.next_settlement_or_new_settlement: int = 0
        self.settlement_to_city_or_new_settlement: int = 0
        self.overall_prio: List = [99]
        self.number_of_unique_roads_to_build: int = 99
        self.risk_for_new_settlement: List[str] = ["None", "None", "None"]
        self.path_to_longest_road: List = []
        self.table_TWcbo: List = []

    def _ignore_resource_cards(self) -> bool:
        """
        Return True when fast-forward should ignore actual owned resource cards.
        """
        return bool(getattr(self.game, "ff_ignore_resource_cards", False))

    # ===================================================================
    # Update logic
    # ===================================================================
    def update(self, position: int, ref_value: str, value_input: Any) -> None:
        """Update a specific outlook field."""
        if ref_value != "X":
            if position == 0:
                self.summary = str(ref_value) + self.summary[1:]
                self.next_road1 = value_input
            elif position == 1:
                self.summary = self.summary[:1] + str(ref_value) + self.summary[2:]
                self.next_road2 = value_input
            elif position == 2:
                self.summary = self.summary[:2] + str(ref_value) + self.summary[3:]
                self.next_road3 = value_input
            elif position == 3:
                self.summary = self.summary[:3] + str(ref_value) + self.summary[4:]
                self.next_settlement1 = value_input
            elif position == 4:
                self.summary = self.summary[:4] + str(ref_value) + self.summary[5:]
                self.next_settlement2 = value_input
            elif position == 5:
                self.summary = self.summary[:5] + str(ref_value) + self.summary[6:]
                self.next_settlement_to_city1 = value_input
            elif position == 6:
                self.summary = self.summary[:6] + str(ref_value) + self.summary[7:]
                self.next_settlement_to_city2 = value_input
            elif position == 7:
                self.summary = self.summary[:7] + str(ref_value) + self.summary[8:]
                self.new_settlement1 = value_input
            elif position == 8:
                self.summary = self.summary[:8] + str(ref_value)
                self.new_settlement2 = value_input
        else:
            if position == 9:
                self.next_settlement_or_new_settlement = value_input
            elif position == 10:
                self.settlement_to_city_or_new_settlement = value_input
            elif position == 12:
                self.overall_prio = value_input
            elif position == 13:
                self.number_of_unique_roads_to_build = value_input

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(f"PlayerOutlook.update | pos={position} ref={ref_value} value={value_input}\n")

    # ===================================================================
    # Refresh all outlook data
    # ===================================================================
    def refresh_all(self) -> None:
        """
        Recompute outlook fields.

        This tries to call your older helper functions if they exist in module scope.
        If not, it falls back to lightweight fast-forward-safe defaults so this module
        does not crash.
        """
        helper_failed = False

        try:
            ns = globals().get("next_settlement")
            nc = globals().get("next_city")
            new_s = globals().get("new_settlement")
            cvs = globals().get("city_vs_settlement")
            aft = globals().get("after_next_or_new_SorC")

            if callable(ns):
                ns(self.game, self.player)
            if callable(nc):
                nc(self.game, self.player)
            if callable(new_s):
                new_s(self.game, self.player, "refresh")
            if callable(cvs):
                cvs(self.game, self.player)
            if callable(aft):
                aft(self.game, self.player, [], "refresh", "refresh_all")

        except Exception as exc:
            helper_failed = True
            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"PlayerOutlook.refresh_all | legacy helper failure for "
                        f"player {self.player.id}: {exc}\n"
                    )

        # Lightweight fallback / refresh summary
        viable_new = self.get_viable_new_settlements(allow_extra_road=False)
        viable_city = self.get_viable_city_upgrades()

        self.new_settlement1 = [viable_new[0], 0] if viable_new else [0, 0]
        self.new_settlement2 = [viable_new[1], 0] if len(viable_new) > 1 else [0, 0]

        self.next_settlement_to_city1 = [viable_city[0], 0] if viable_city else [0, 0]
        self.next_settlement_to_city2 = [viable_city[1], 0] if len(viable_city) > 1 else [0, 0]

        # Small compatibility flags
        self.next_settlement_or_new_settlement = 1 if viable_new else 0
        self.settlement_to_city_or_new_settlement = 1 if viable_city else 0
        self.number_of_unique_roads_to_build = 99  # unchanged placeholder

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                status = "fallback-only" if helper_failed else "ok"
                f.write(
                    f"PlayerOutlook.refresh_all completed for player {self.player.id} "
                    f"| status={status} | viable_new={len(viable_new)} "
                    f"| viable_city={len(viable_city)}\n"
                )

    # ===================================================================
    # Legal move filtering for fast-forward
    # ===================================================================
    def get_viable_new_settlements(self, allow_extra_road: bool = False) -> List[int]:
        """
        Return forecast-viable intersections for a future new settlement.

        IMPORTANT:
        This is intentionally broader than "buildable right now".

        For the fast-forward Markov table we only want to know whether settlement
        is still structurally possible somewhere on the board, not whether the
        player can legally place it this instant.

        Rules used here:
        - not in water
        - not already occupied
        - respects the distance rule against ALL existing settlements/cities
        - road/path ownership is NOT required here
        - allow_extra_road is kept for compatibility but does not narrow the set
        """
        board = self.game.board

        all_existing: List[int] = []
        for p in self.game.players:
            all_existing.extend(getattr(p, "settlements", []) + getattr(p, "cities", []))

        viable: List[int] = []

        for inter in board.intersections:
            if inter is None:
                continue

            inter_id = inter.id

            if inter_id in getattr(board, "INTERSECTION_IN_WATER", []):
                continue

            if getattr(inter, "occupied_tf", False):
                continue

            too_close = False
            for existing_id in all_existing:
                if existing_id == inter_id:
                    too_close = True
                    break

                try:
                    dist = board._distance_between_intersections(inter_id, existing_id)
                except Exception:
                    dist = 999

                if dist <= 1:
                    too_close = True
                    break

            if too_close:
                continue

            viable.append(inter_id)

        return viable

    def _can_reach_with_one_extra_road(self, target_inter_id: int) -> bool:
        """
        Kept for compatibility with older call sites.

        For the unified fast-forward Markov table, settlement viability is no longer
        limited to "reachable with one extra road right now", so this helper should
        not block forecastable settlement strategies.

        We therefore return True whenever the target is structurally a valid
        future-settlement vertex under the distance rule and occupancy constraints.
        """
        board = self.game.board

        if target_inter_id < 0 or target_inter_id >= len(board.intersections):
            return False

        inter = board.intersections[target_inter_id]
        if inter is None:
            return False

        if target_inter_id in getattr(board, "INTERSECTION_IN_WATER", []):
            return False

        if getattr(inter, "occupied_tf", False):
            return False

        all_existing: List[int] = []
        for p in self.game.players:
            all_existing.extend(getattr(p, "settlements", []) + getattr(p, "cities", []))

        for existing_id in all_existing:
            try:
                dist = board._distance_between_intersections(target_inter_id, existing_id)
            except Exception:
                dist = 999

            if dist <= 1:
                return False

        return True

    def get_viable_city_upgrades(self) -> List[int]:
        """Return settlements that can still be upgraded to cities."""
        viable: List[int] = []

        for tw_id in getattr(self.player, "settlements", []):
            if not (0 <= tw_id < len(self.game.board.intersections)):
                continue
            inter = self.game.board.intersections[tw_id]
            if inter and getattr(inter, "face", None) == "Settlement":
                viable.append(tw_id)

        return viable

    # ===================================================================
    # Fast-forward decision methods
    # ===================================================================
    def choose_next_activity(self) -> str:
        """
        Decide the best next activity when a higher-level engine does not override it.

        When ff_ignore_resource_cards is True:
        - do NOT gate activities by current owned cards
        - only use structural/legal viability
        """
        can_upgrade = bool(self.get_viable_city_upgrades())
        can_new = bool(self.get_viable_new_settlements(allow_extra_road=True))

        if self._ignore_resource_cards():
            can_buy_dev = bool(getattr(self.game, "dcards_stack", []))
        else:
            can_buy_dev = bool(getattr(self.game, "dcards_stack", [])) and self.player.can_afford("development_card")

        if self._ignore_resource_cards():
            # No affordability gating: choose by structural possibility only
            if can_upgrade:
                return "upgrade_to_city"
            if can_new:
                return "new_settlement"
            if can_buy_dev:
                return "buy_discovery_card"
            return "buy_discovery_card"

        # Normal realistic path
        if can_upgrade and self.player.can_afford("city"):
            return "upgrade_to_city"

        if can_new and self.player.can_afford("settlement"):
            return "new_settlement"

        if can_buy_dev:
            return "buy_discovery_card"

        # Soft fallbacks if the hand is not ready yet
        if can_upgrade:
            return "upgrade_to_city"
        if can_new:
            return "new_settlement"

        return "buy_discovery_card"

    # ===================================================================
    # Candidate ranking / Markov selection
    # ===================================================================
    def _rank_candidates(self, viable_list: List[int], k: int = 40) -> List[int]:
        """Rank candidates with the fast pips+port heuristic, with a safe fallback."""
        if not viable_list:
            return []

        if hasattr(InitialPlacementStrategies, "get_top_k_by_pips_and_port"):
            try:
                ranked = InitialPlacementStrategies.get_top_k_by_pips_and_port(
                    self.game.board,
                    viable_list,
                    k=k,
                )
                if ranked:
                    return ranked
            except Exception as exc:
                if MG:
                    with open(FILENAME_MG, "a", encoding="utf-8") as f:
                        f.write(f"PlayerOutlook._rank_candidates | heuristic fallback due to: {exc}\n")

        # Fallback ranking: sum all_tile_pips + mild port bonus
        scored = []
        for inter_id in viable_list:
            inter = self.game.board.intersections[inter_id]
            if inter is None:
                continue
            pips = sum(getattr(inter, "all_tile_pips", [0.0] * 5))
            port_bonus = 1.0 if getattr(inter, "port_tf", False) else 0.0
            scored.append((inter_id, pips + port_bonus))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return [iid for iid, _ in scored[:k]]

    def _build_candidate_ports(self, inter_id: int, base_ports: dict, activity: str) -> dict:
        """Build the hypothetical port dictionary for Markov scoring."""
        candidate_ports = dict(base_ports)

        # Upgrading a city does not add new harbor access
        if activity != "new_settlement":
            return candidate_ports

        inter = self.game.board.intersections[inter_id]
        if inter is None or not getattr(inter, "port_tf", False):
            return candidate_ports

        port_type = getattr(inter, "port_type", "").strip()
        if not port_type or port_type in ("", "Blank", "4:1"):
            return candidate_ports

        if port_type == "3:1" or "3:1" in port_type:
            candidate_ports["generic"] = 3
            return candidate_ports

        if port_type.startswith("2:1"):
            res_name = port_type.split()[-1].strip().lower()

            # Match MarkovEvaluator.RES_NAMES
            if res_name == "wood":
                res_name = "lumber"
            elif res_name == "sheep":
                res_name = "wool"

            if hasattr(self.game, "markov") and res_name in getattr(self.game.markov, "RES_NAMES", []):
                candidate_ports[res_name] = 2

        return candidate_ports

    def select_best_location(self, activity: str, viable_list: List[int]) -> int:
        """
        Choose the best legal location for the requested activity.

        Supported activities:
            - "new_settlement"
            - "upgrade_to_city"

        EH-only note:
            In RESOURCE_TIMING_ENGINE='expected_hand', game.markov is None by design.
            Therefore this method must not call Markov. When Markov is unavailable,
            it falls back to the deterministic static ranking produced by
            _rank_candidates(...). This is sufficient for choosing a concrete
            target for exact PLAY execution.
        """
        if not viable_list:
            return -1

        ranked = self._rank_candidates(viable_list, k=40)
        candidates = ranked[:20] if ranked else list(viable_list)

        if not candidates:
            return -1

        markov_obj = getattr(self.game, "markov", None)
        markov_available = bool(
            markov_obj is not None
            and (
                hasattr(markov_obj, "get_expected_turns_fast_initial")
                or hasattr(markov_obj, "get_expected_turns")
            )
        )

        # Expected-hand-only / no-Markov fallback.
        # _rank_candidates already prioritizes static quality; for city upgrades,
        # restrict to existing settlements and pick the best ranked settlement.
        if not markov_available:
            if activity == "upgrade_to_city":
                current_settlements = set(int(x) for x in getattr(self.player, "settlements", []))
                candidates = [int(i) for i in candidates if int(i) in current_settlements]

            if not candidates:
                return -1

            chosen = int(candidates[0])

            if MG:
                try:
                    with open(FILENAME_MG, "a", encoding="utf-8") as f:
                        f.write(
                            "PlayerOutlook.select_best_location | "
                            "Markov unavailable; using static ranked fallback "
                            f"| activity={activity} | chosen={chosen} "
                            f"| candidates={candidates[:10]}\n"
                        )
                except Exception:
                    pass

            return chosen

        current_settlements = list(getattr(self.player, "settlements", []))
        current_cities = list(getattr(self.player, "cities", []))

        if self._ignore_resource_cards():
            hand = [0, 0, 0, 0, 0]
        else:
            hand = self.player.rcards_in_hand()[0] if hasattr(self.player, "rcards_in_hand") else [0, 0, 0, 0, 0]

        base_ports = self.game.get_player_ports_dict(self.player)
        best_score = float("inf")
        best_inter = -1
        tied_inters: List[int] = []

        for inter_id in candidates:
            if activity == "new_settlement":
                vertices = current_settlements + current_cities + [inter_id]
                candidate_ports = self._build_candidate_ports(inter_id, base_ports, activity)
                fast_strategy = "settlement"

            elif activity == "upgrade_to_city":
                if inter_id not in current_settlements:
                    continue

                # Intentional duplicate so evaluator can model doubled production.
                vertices = current_settlements + current_cities + [inter_id]
                candidate_ports = dict(base_ports)
                fast_strategy = "city"

            else:
                raise ValueError(f"Unsupported activity: {activity}")

            # FAST Markov scoring path, only when Markov exists.
            if hasattr(markov_obj, "get_expected_turns_fast_initial"):
                score = markov_obj.get_expected_turns_fast_initial(
                    vertices=vertices,
                    hand=hand,
                    player_ports=candidate_ports,
                    strategy=fast_strategy,
                )
            else:
                try:
                    score = markov_obj.get_expected_turns(
                        vertices=vertices,
                        hand=hand,
                        player_ports=candidate_ports,
                        strategy=fast_strategy,
                    )
                except TypeError:
                    score = markov_obj.get_expected_turns(vertices, hand, candidate_ports)

            if score is None:
                score = 9999.0

            if score < best_score - 0.0001:
                best_score = float(score)
                best_inter = inter_id
                tied_inters = [inter_id]
            elif abs(float(score) - best_score) < 0.0001:
                tied_inters.append(inter_id)

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"PlayerOutlook.select_best_location | activity={activity} "
                        f"| inter={inter_id} | vertices={vertices} "
                        f"| ports={candidate_ports} | hand={hand} | score={float(score):.4f}\n"
                    )

        if tied_inters:
            best_inter = min(tied_inters)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"PlayerOutlook.select_best_location | FINAL "
                    f"| activity={activity} | best_inter={best_inter} "
                    f"| best_score={best_score:.4f} | ties={tied_inters}\n"
                )

        return best_inter
