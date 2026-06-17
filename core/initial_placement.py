"""
v010
Manages the initial placement phase of the Catan game.
Includes human guidance for settlement + road placement and confirmation.
This module defines the InitialPlacement class, handling the placement sequence for
4 AI players (settlement + road in rounds -2 and -1) using specified algorithms.
"""
import pygame
import random
import time
from typing import Dict, List, Tuple, Optional, Set

from core.game import Game, Player
from core.board import Board
from core.constants import (
    NUM_PLAYERS, HUMAN_PLAYER, HP_ID,
    ResourceCard, TERRAIN_TO_RESOURCE, RESOURCE_ORDER,
    FNFREQ, FILENAME_FREQ, MG, FILENAME_MG,
    MARKOV_TIMING_ENABLED, INITIAL_PLACEMENT_MARKOV_FALLBACK_ALGORITHM
)
from gui.gui_human_player import GUIHumanPlayer
from gui.gui_guidance import PlacementState

from core.algorithms_initial_placement import InitialPlacementStrategies


class InitialPlacement:
    """Manages the initial placement phase for the Catan game."""

    def __init__(self, game: Game) -> None:
        """Manages the initial placement phase of the Catan game."""
        self.game = game

        if NUM_PLAYERS not in (2, 3, 4):
            raise ValueError(f"NUM_PLAYERS must be 2, 3 or 4 — got {NUM_PLAYERS}")

        self.num_players = NUM_PLAYERS
        player_ids = list(range(1, NUM_PLAYERS + 1))

        # Default algorithm can still be overridden per player
        self.player_algorithms: Dict[int, int] = {pid: 1 for pid in player_ids}

        self.sequence = player_ids + list(reversed(player_ids))
        self.current_step = 0

    def run(self) -> None:
        """
        Start the Initial Placement phase.

        Human control is allowed only here.
        Execution will later become AI-only.
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a", encoding="utf-8") as f:
                f.write(f"{self.game.id} | {self.game.state} | initial_placement.py | run\n")

        self.game.phase = "InitialPlacement"
        self.game.round = -2
        self.game.turn = 1
        self.current_step = 0

        # Initial-placement uses the button in PLAY mode
        if hasattr(self.game, "ff_button_mode"):
            self.game.ff_button_mode = "PLAY"
        if hasattr(self.game, "ff_pending_event"):
            self.game.ff_pending_event = None
        if hasattr(self.game, "ff_waiting_for_play"):
            self.game.ff_waiting_for_play = False

        self.game.sync_round_turn()
        if hasattr(self.game, "update_current_player_from_turn"):
            self.game.update_current_player_from_turn()

        if self.game.gui:
            try:
                if hasattr(self.game.gui, "human_guidance") and self.game.gui.human_guidance:
                    self.game.gui.human_guidance.clear()
            except Exception:
                pass

            try:
                self.game.gui.update_round_turn(self.game, special=False)
            except Exception:
                pass

    def advance_turn(self) -> None:
        print("InitialPlacement.advance_turn executed")

        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a", encoding="utf-8") as f:
                f.write(
                    f"{self.game.id} | {self.game.state} | "
                    f"initial_placement.py | advance_turn\n"
                )

        # Initial placement complete
        if self.current_step >= len(self.sequence):
            if hasattr(self.game, "enter_execution_phase"):
                self.game.enter_execution_phase()
            else:
                self.game.phase = "Execution"
                self.game.game_over = False
                self.game.round = 1
                self.game.turn = 1
            return

        player_id = self.sequence[self.current_step]
        print(f"player_id: {player_id}")

        player = next(p for p in self.game.players if p.id == player_id)
        self.game.current_player = player

        is_human_initial_turn = (
            HUMAN_PLAYER
            and player_id in HP_ID
            and self.game.round < 0
        )

        # Always disable PLAY while this turn is being processed.
        GUIHumanPlayer.button_next_turn2(self.game.gui, self.game, active=False)
        pygame.display.update()

        # ------------------------------------------------------------
        # HUMAN placement:
        # Start settlement guidance and STOP here.
        #
        # Do NOT increment current_step here.
        # Do NOT call game.advance_turn() here.
        #
        # The human confirmation flow must do that only after both
        # settlement and road are confirmed.
        # ------------------------------------------------------------
        if is_human_initial_turn:
            self.game.gui.human_guidance.start_settlement_phase(player)

            self.game.gui.update_round_turn(self.game, special=False)
            self.game.gui.update_scoreboard(self.game)
            pygame.display.update()
            return

        # ------------------------------------------------------------
        # AI placement:
        # Execute settlement + road immediately.
        # ------------------------------------------------------------
        self.execute_initial_placement_strategy(player)

        print(
            f"Common update | player {player_id} | "
            f"round={self.game.round} turn={self.game.turn} | step={self.current_step}"
        )

        self.game.gui.update_round_turn(self.game, special=False)
        self.game.gui.update_board(self.game.board, "Last")
        self.game.gui.update_scoreboard(self.game)
        pygame.display.update()

        # AI step completes immediately.
        self.current_step += 1

        print(f"AI placement finished – advanced current_step to {self.current_step}")
        print("InitialPlacement calling game.advance_turn")
        self.game.advance_turn()

        # Re-enable PLAY only after AI has fully placed settlement + road.
        if self.game.phase == "InitialPlacement":
            GUIHumanPlayer.button_next_turn2(self.game.gui, self.game, active=True)
            pygame.display.update()

    def execute_initial_placement_strategy(self, player: Player) -> None:
        """
        Execute the full AI initial placement strategy for one turn:
          - choose & place settlement
          - distribute resources (if round == -1)
          - choose & place road (connected to the last settlement, using strategic functions)
        """
        board = self.game.board
        algorithm_id = player.initial_placement_algorithm

        # 1. Choose best settlement location
        intersection_id, valid_intersections = self._choose_settlement_location(player, algorithm_id)

        # Place settlement on board
        board.occupy_intersection(
            intersection_id, "Settlement", player.color,
            placement_step=self.current_step
        )

        # Explicitly update player's settlements list
        player.settlements.append(intersection_id)

        self.game.gui.update_scoreboard(self.game)
        
        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(f"Initial placement | Player {player.id} ({player.color}) "
                        f"placed settlement at {intersection_id} (step={self.current_step})\n")

        # Distribute starting resources if second placement (round -1)
        if self.game.round == -1:
            self.distribute_initial_resources(intersection_id, player)

        # ────────────────────────────────────────────────
        # Road placement — connected to the last settlement + strategic choice
        # ────────────────────────────────────────────────
        last_settlement = player.settlements[-1]

        # Minimal blocked_tws: only already occupied intersections
        selected_tws = [
            i for i in range(len(board.intersections))
            if board.intersections[i] and board.intersections[i].occupied_tf
        ]
        blocked_tws = selected_tws[:]

        top_tws = []

        player_has_port_now = player.has_harbor()

        if player_has_port_now:
            best_road = InitialPlacementStrategies.find_best_road_missing_port(
                board=board,
                settlement_id=last_settlement,
                player=player,
                top_tws=top_tws,
                blocked_tws=blocked_tws,
                selected_tws=selected_tws
            )
        else:
            best_road = InitialPlacementStrategies.find_best_road_missing_port(
                board=board,
                settlement_id=last_settlement,
                player=player,
                top_tws=top_tws,
                blocked_tws=blocked_tws,
                selected_tws=selected_tws
            )

        if best_road:
            board.occupy_road(
                list(best_road), "Road", player.color,
                placement_step=self.current_step
            )
            player.roads.append(tuple(sorted(best_road)))

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(f"Player {player.id} | round {self.game.round} | "
                            f"placed strategic road {best_road} from settlement {last_settlement} "
                            f"(has_port_now={player_has_port_now})\n")
        else:
            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(f"Warning: No valid road found for player {player.id} "
                            f"at settlement {last_settlement} (has_port={player_has_port_now})\n")

    def _get_valid_second_settlement_locations(self, player: Player) -> List[int]:
        """Return all intersections still valid for the second settlement in round -1."""
        valid = []
        for inter in self.game.board.intersections:
            if inter is None or inter.id in self.game.board.INTERSECTION_IN_WATER:
                continue
            if inter.occupied_tf:
                continue
            if not inter.can_build_tf:
                continue
            valid.append(inter.id)
        return valid

    def _choose_settlement_location(self, player: Player, algorithm_id: int) -> Tuple[int, List[int]]:
        """
        Choose an initial-placement settlement location.

        Behavior:
        - algorithm_id != 3: use the configured classic initial-placement strategy.
        - algorithm_id == 3 and Markov is enabled/available: use the existing Markov candidate scorer.
        - algorithm_id == 3 and Markov is disabled/unavailable: redirect to a non-Markov fallback
          algorithm, controlled by INITIAL_PLACEMENT_MARKOV_FALLBACK_ALGORITHM.

        This lets RESOURCE_TIMING_ENGINE='expected_hand' bypass Markov without deleting
        the Markov code path.
        """

        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a", encoding="utf-8") as f:
                f.write(
                    f"initial_placement.py | _choose_settlement_location | "
                    f"player {player.id} | algo={algorithm_id} | round={self.game.round}\n"
                )

        print(
            f"\n>>> _choose_settlement_location called for PLAYER {player.id} | "
            f"algorithm={algorithm_id} | round={self.game.round} | "
            f"time={time.strftime('%H:%M:%S')}"
        )

        if self.game.round == -2:
            all_valid = [
                i.id for i in self.game.board.intersections
                if i is not None and i.can_build_tf and not i.occupied_tf
            ]
            target_num = 8
            min_guaranteed = 2
        else:  # round -1
            all_valid = self._get_valid_second_settlement_locations(player)
            target_num = 12
            min_guaranteed = 3

        if not all_valid:
            print("   → WARNING: no valid initial-placement intersections available")
            return -1, all_valid

        # ------------------------------------------------------------
        # Non-Markov algorithms: unchanged behavior.
        # ------------------------------------------------------------
        if algorithm_id != 3:
            chosen = InitialPlacementStrategies.select_intersection(
                board=self.game.board,
                player=player,
                algorithm_id=algorithm_id,
                valid_intersections=all_valid,
                game_round=self.game.round,
            )
            print(f"   FINAL CHOICE (algo {algorithm_id}): intersection {chosen}")
            return chosen, all_valid

        # ------------------------------------------------------------
        # algorithm_id == 3 normally means Markov.
        # In EH-only mode, Markov may deliberately not exist. Redirect
        # to a safe non-Markov fallback without touching the Markov code.
        # ------------------------------------------------------------
        markov_obj = getattr(self.game, "markov", None)
        markov_available = bool(
            MARKOV_TIMING_ENABLED
            and markov_obj is not None
            and hasattr(markov_obj, "get_expected_turns_fast_initial")
        )

        if not markov_available:
            fallback_algorithm = int(INITIAL_PLACEMENT_MARKOV_FALLBACK_ALGORITHM or 4)
            if fallback_algorithm == 3:
                fallback_algorithm = 4

            chosen = InitialPlacementStrategies.select_intersection(
                board=self.game.board,
                player=player,
                algorithm_id=fallback_algorithm,
                valid_intersections=all_valid,
                game_round=self.game.round,
            )

            print(
                f"   → Markov initial placement disabled/unavailable; "
                f"algorithm 3 redirected to algorithm {fallback_algorithm}. "
                f"FINAL CHOICE: intersection {chosen}"
            )

            if MG:
                try:
                    with open(FILENAME_MG, "a", encoding="utf-8") as f:
                        f.write(
                            f"Initial placement | Player {player.id} | round {self.game.round} | "
                            f"algorithm 3 redirected to algorithm {fallback_algorithm} "
                            f"because Markov is disabled/unavailable | chosen={chosen}\n"
                        )
                except Exception:
                    pass

            return chosen, all_valid

        print("   → Using Markov with 3D tensor (7 trading modes: 4:1 + 3:1 + 5×2:1)")

        # Base ranked list from Algorithm 4.
        ranked = InitialPlacementStrategies.get_top_k_by_pips_and_port(
            board=self.game.board,
            valid_intersections=all_valid,
            k=40,
        )

        valid_ranked = [iid for iid in ranked if iid in all_valid]

        # === FORCED PORT INCLUSION ===
        forced_ports = []
        for iid in all_valid:
            inter = self.game.board.intersections[iid]
            if not getattr(inter, "port_tf", False):
                continue

            port_type = getattr(inter, "port_type", "").strip()
            if not port_type or port_type in ["", "4:1", "Blank"]:
                continue

            total_pips = sum(getattr(inter, "three_tile_pips", [0] * 5))

            if self.game.round == -2:
                if total_pips >= 8:
                    forced_ports.append((iid, total_pips))
            else:  # round -1: match player's strongest resource
                if port_type.startswith("2:1") and player.settlements:
                    first_inter = self.game.board.intersections[player.settlements[0]]
                    if first_inter and hasattr(first_inter, "three_tile_pips"):
                        strongest_idx = first_inter.three_tile_pips.index(
                            max(first_inter.three_tile_pips)
                        )
                        res_names = ["Wheat", "Ore", "Wood", "Brick", "Wool"]
                        if res_names[strongest_idx] in port_type:
                            forced_ports.append((iid, total_pips))

        # Round -2 special rule for ports with pips >= 8.
        if self.game.round == -2 and forced_ports:
            forced_ports.sort(key=lambda x: x[1], reverse=True)

            # Take top 2, or more only if exact tie at 2nd place.
            to_include = forced_ports[:2]
            if len(forced_ports) > 2 and forced_ports[2][1] == forced_ports[1][1]:
                to_include = [p for p in forced_ports if p[1] >= forced_ports[1][1]]

            forced = [p[0] for p in to_include]
        else:
            forced = [p[0] for p in forced_ports]

        # Add forced ports without duplicates.
        for iid in forced:
            if iid not in valid_ranked:
                valid_ranked.append(iid)

        # Extend ranking if still not enough candidates.
        k = 40
        while len(valid_ranked) < target_num and k < 120:
            k += 20
            print(f"   → Only {len(valid_ranked)} valid. Extending to top {k}")
            ranked = InitialPlacementStrategies.get_top_k_by_pips_and_port(
                board=self.game.board,
                valid_intersections=all_valid,
                k=k,
            )
            valid_ranked = [iid for iid in ranked if iid in all_valid]

        # Guarantee minimum candidate count.
        if len(valid_ranked) < min_guaranteed:
            print(
                f"   → WARNING: Only {len(valid_ranked)} valid left. "
                f"Forcing minimum {min_guaranteed}"
            )
            extra = [iid for iid in all_valid if iid not in valid_ranked]
            valid_ranked += extra[:min_guaranteed - len(valid_ranked)]

        candidates = valid_ranked[:target_num]

        if not candidates:
            print("   → WARNING: no Markov candidates available; falling back to first valid intersection")
            return all_valid[0], all_valid

        print(f"   → Running Markov on {len(candidates)} candidates (target={target_num})")

        # Markov evaluation loop.
        best_score = 9999.0
        best_inter = None
        tied_inters = []
        markov_scores = {}

        for idx, inter_id in enumerate(candidates, 1):
            inter = self.game.board.intersections[inter_id]
            vertices = [inter_id] if self.game.round == -2 else (player.settlements + [inter_id])

            player_ports = self.game.get_player_ports_dict(player).copy()
            if getattr(inter, "port_tf", False):
                port_type = getattr(inter, "port_type", "").strip()
                if port_type and port_type not in ["", "4:1", "Blank"]:
                    if port_type == "3:1" or "3:1" in port_type:
                        player_ports["generic"] = 3
                    else:
                        res_name = port_type.split()[-1] if " " in port_type else port_type
                        if res_name.lower() in ("sheep", "wool"):
                            res_name = "Wool"
                        player_ports[res_name] = 2

            try:
                score = markov_obj.get_expected_turns_fast_initial(
                    vertices=vertices,
                    hand=[0, 0, 0, 0, 0],
                    player_ports=player_ports,
                )
            except Exception as exc:
                print(f"     [{idx:2d}/{len(candidates)}] inter {inter_id:2d} → Markov error: {exc}")
                score = 9999.0

            if score is None:
                score = 9999.0

            score = float(score)
            markov_scores[inter_id] = score

            print(f"     [{idx:2d}/{len(candidates)}] inter {inter_id:2d} → {score:6.2f} turns")

            if score < best_score - 0.0001:
                best_score = score
                tied_inters = [inter_id]
                best_inter = inter_id
            elif abs(score - best_score) < 0.0001:
                tied_inters.append(inter_id)

        # Final decision.
        unique_scores = len(
            set(round(s, 4) for s in markov_scores.values() if s is not None)
        )

        if unique_scores <= 1:
            print("   → All Markov scores identical → FALLBACK to Algorithm 4 ranking")
            best_inter = valid_ranked[0] if valid_ranked else all_valid[0]
        elif len(tied_inters) > 1:
            best_inter = random.choice(tied_inters)
            print(f"   → Tie between {len(tied_inters)} intersections → randomized to {best_inter}")
        else:
            print(
                f"   → Markov FINAL CHOICE: intersection {best_inter} "
                f"(score {best_score:.2f} turns)"
            )

        return best_inter, all_valid

    def distribute_initial_resources(self, intersection_id: int, player: Player) -> None:
        """Distribute initial resource cards from the second settlement placement."""
        intersection = self.game.board.intersections[intersection_id]
        if intersection is None:
            return

        resource_counts = [0] * len(RESOURCE_ORDER)

        for tile_id in intersection.three_tile_ids:
            tile = self.game.board.tiles[tile_id]
            if not tile or tile.type in ("Sea", "Desert"):
                continue

            resource = TERRAIN_TO_RESOURCE.get(tile.type)
            if resource:
                try:
                    idx = RESOURCE_ORDER.index(resource)
                    resource_counts[idx] += 1
                    player.add_rcard(resource, 1)
                except ValueError:
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            f.write(f"initial_placement.py | distribute_initial_resources | "
                                    f"Resource {resource} not found in RESOURCE_ORDER "
                                    f"(tile type was '{tile.type}')\n")

        player.number_of_rcards = sum(player.rcards.get(rc, 0) for rc in ResourceCard)

        player_id = player.id
        delta_rc = sum(resource_counts)

        self.game.update_strategy_dashboard(player)

        for dash in self.game.resource_card_dashboard:
            for i, count in enumerate(resource_counts):
                dash.resource_production_game_total[i] += count

            for p in dash.resource_production_game_player:
                if p[0] == player_id:
                    for i, count in enumerate(resource_counts):
                        p[i + 1] += count

            for view in dash.resource_production_game_player_view:
                if view[1] == player_id:
                    for i, count in enumerate(resource_counts):
                        view[i + 2] += count

        current_views = [
            [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]]
            for v in self.game.resource_card_dashboard[0].resource_production_game_player_view
        ]

        self.game.log_event([
            [1, f"RP_InitialPlacement: {intersection_id} for {player.color}"],
            [2, self.game.sequence_number],
            [3, self.game.round],
            [4, self.game.turn],
            [5, player.points],
            [6, 4],
            [7, 5 - len(player.settlements)],
            [8, 15 - len(player.roads)],
            [9, player.size_longest_route],
            [10, player.size_largest_army],
            [12, 0], [13, 0],
            [14, player.number_of_rcards],
            [15, 0],
            [16, intersection_id], [17, 0],
            [18, player_id], [19, 0],
            [20, "RP_InitialPlacement"],
            [21, "99999"], [22, "99999X99999"],
            [23, [resource_counts]],
            [24, delta_rc],
            [25, resource_counts],
            [26, [0, 0, 0, 0, 0]],
            [27, delta_rc],
            [28, current_views],
            [29, 0],
            [30, [0, 0, 0, 0, 0]],
            [31, [0, 0, 0, 0, 0]],
            [32, 0], [33, 0],
        ])

    def handle_click(self, pos: Tuple[int, int]) -> bool:
        """
        Handle board clicks only during InitialPlacement human guidance.

        In Execution, human guidance is disabled entirely.
        """
        if self.game.phase != "InitialPlacement":
            return False

        if not hasattr(self.game, "human_controls_enabled"):
            if not HUMAN_PLAYER:
                return False
        else:
            if not self.game.human_controls_enabled():
                return False

        guidance = self.game.gui.human_guidance
        if guidance.state == PlacementState.IDLE:
            return False

        handled = guidance.on_board_click(pos)
        return bool(handled)