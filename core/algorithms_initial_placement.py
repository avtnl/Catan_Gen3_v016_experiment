"""
core/algorithms_initial_placement.py
FULL classic 5-strategy engine with shared points on ties + pips-based harbor bonus
using existing port_tf / port_type on intersections
"""

from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
from itertools import groupby

from core.board import Board
from core.player import Player
from core.constants import MG, FILENAME_MG, BLOCKED_WEIGHT, TOP_N


class InitialPlacementStrategies:

    @staticmethod
    def _safe_int(val: Any) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def check_harbor(harbor_type: str, frame: List[int], three_types: List[int], three_probs: List[float]) -> float:
        if not three_probs or len(three_probs) != 5:
            return 0.0

        total_pips = sum(InitialPlacementStrategies._safe_float(p) for p in three_probs)

        if harbor_type == "3:1":
            return total_pips / 6.0

        resource_index = -1
        if harbor_type == "2:1 Wheat":
            resource_index = 0
        elif harbor_type == "2:1 Ore":
            resource_index = 1
        elif harbor_type == "2:1 Wood":
            resource_index = 2
        elif harbor_type == "2:1 Brick":
            resource_index = 3
        elif harbor_type == "2:1 Wool":
            resource_index = 4

        if resource_index != -1 and frame[resource_index] == 1:
            matching_pips = InitialPlacementStrategies._safe_float(three_probs[resource_index])
            return matching_pips / 2.0

        return 0.0

    @staticmethod
    def select_intersection(
        board: Board,
        player: Player,
        algorithm_id: int = 2,
        valid_intersections: List[int] = None,
        game_round: int = -2,
    ) -> int:
        if valid_intersections is None:
            valid_intersections = [
                i for i in range(len(board.intersections))
                if getattr(board.intersections[i], "canbuildYNX", "N") != "X"
            ]

        if algorithm_id == 1:
            return InitialPlacementStrategies._max_pips(board, valid_intersections)
        if algorithm_id == 2:
            return InitialPlacementStrategies._five_strategy_engine(
                board, player, valid_intersections, game_round=game_round
            )
        if algorithm_id == 3:
            # Markov is handled in initial_placement.py
            raise ValueError("algorithm_id 3 (Markov) must be called via game.markov")
        if algorithm_id == 4:
            return InitialPlacementStrategies._max_of_pips_and_port(board, valid_intersections)

        raise ValueError(f"algorithm_id {algorithm_id} not supported")

    @staticmethod
    def _max_pips(board: Board, valid_intersections: List[int]) -> int:
        best_inter = -1
        best_score = -1.0

        for inter_id in valid_intersections:
            if inter_id < 0 or inter_id >= len(board.intersections):
                continue
            inter = board.intersections[inter_id]
            if inter is None:
                continue

            probs_raw = getattr(inter, "all_tile_probabilities",
                                getattr(inter, "three_tile_probabilities_v2",
                                        getattr(inter, "three_tile_probabilities", [0] * 5)))
            probs = [InitialPlacementStrategies._safe_float(v) for v in probs_raw]
            pips = sum(probs)

            port_bonus = 5.0 if getattr(inter, "port_tf", False) or getattr(inter, "harborYN", "N") == "Y" else 0.0

            score = pips + port_bonus

            if score > best_score or (score == best_score and inter_id < best_inter):
                best_score = score
                best_inter = inter_id

        return best_inter if best_inter != -1 else (valid_intersections[0] if valid_intersections else -1)

    # ===================================================================
    # NEW: ALGORITHM 4 — pips + floor(pips / port_ratio)
    # ===================================================================
    @staticmethod
    def _max_of_pips_and_port(
        board: "Board",
        valid_intersections: List[int]
    ) -> int:
        """Return the intersection with the highest score = Σ(pips_r + floor(pips_r / ratio_r)) over all resources.
        
        Exactly as you specified:
        - 3:1 port → ratio 3 for EVERY resource
        - 2:1 specific port → ratio 2 only for the matching resource, 4 for others
        - No port → ratio 4 for all resources
        - Uses floor division (as in your examples)
        """
        if not valid_intersections:
            return min(i.id for i in board.intersections if i is not None)

        # Resource mapping (already in constants.py)
        terrain_to_res = {
            "Hill":     "brick",
            "Forest":   "wood",
            "Pasture":  "wool",
            "Field":    "wheat",
            "Mountain": "ore"
        }

        scores = []
        debug_lines = ["\n=== _max_of_pips_and_port FULL RANKING (per your spec) ==="]

        for inter_id in valid_intersections:
            inter = board.intersections[inter_id]
            if inter is None:
                continue

            # Group pips by resource
            resource_pips: dict[str, float] = {"brick": 0.0, "wood": 0.0, "wool": 0.0, "wheat": 0.0, "ore": 0.0}

            for i, tile_id in enumerate(inter.three_tile_ids):
                tile = board.tiles[tile_id] if 0 <= tile_id < len(board.tiles) else None
                if not tile or tile.type not in terrain_to_res:
                    continue
                res = terrain_to_res[tile.type]
                pips = inter.three_tile_pips[i] if hasattr(inter, 'three_tile_pips') else pips_from_tile_value(tile.value)
                resource_pips[res] += pips

            # Determine port ratio per resource
            ratio_dict = {"brick": 4, "wood": 4, "wool": 4, "wheat": 4, "ore": 4}
            port_str = ""

            if inter.port_tf and inter.port_type != "Blank":
                if inter.port_type == "3:1":
                    ratio_dict = {r: 3 for r in ratio_dict}
                    port_str = "3:1"
                elif inter.port_type.startswith("2:1"):
                    specific_res = inter.port_type.split()[-1].lower()  # "Ore" → "ore"
                    if specific_res in ratio_dict:
                        ratio_dict[specific_res] = 2
                        port_str = inter.port_type

            # Calculate score per resource and total
            total_score = 0.0
            breakdown = []
            for res, pips in resource_pips.items():
                if pips > 0:
                    bonus = pips // ratio_dict[res]          # floor division as in your examples
                    total_score += pips + bonus
                    breakdown.append(f"{res[:4]}:{pips:.0f}+{bonus:.0f}")

            scores.append((inter_id, total_score, breakdown, port_str))

        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Debug output (console + MG log)
        for rank, (iid, score, breakdown, port_str) in enumerate(scores[:12], 1):
            port_info = f" + {port_str}" if port_str else ""
            line = f"   #{rank:2d} inter {iid:2d} | score={score:4.1f}{port_info} | {' '.join(breakdown)}"
            debug_lines.append(line)
            print(line)

        if len(scores) > 12:
            debug_lines.append(f"   ... {len(scores)-12} more intersections")

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write("\n".join(debug_lines) + "\n")

        # Return best intersection
        best_id = scores[0][0]
        print(f"   → Algorithm 4 FINAL CHOICE: intersection {best_id} (score {scores[0][1]:.1f})")
        return best_id

    @staticmethod
    def get_top_k_by_pips_and_port(board: "Board", valid_intersections: List[int], k: int = 40) -> List[int]:
        """Return the top K intersections ranked exactly like _max_of_pips_and_port.
        This is the deterministic ranking source for Markov candidate selection."""
        if not valid_intersections:
            return []

        scores = []
        terrain_to_res = {
            "Hill": "brick", "Forest": "wood", "Pasture": "wool",
            "Field": "wheat", "Mountain": "ore"
        }

        for inter_id in valid_intersections:
            inter = board.intersections[inter_id]
            if inter is None:
                continue

            # --- Exact same scoring as _max_of_pips_and_port ---
            resource_pips = {"brick": 0.0, "wood": 0.0, "wool": 0.0, "wheat": 0.0, "ore": 0.0}

            for i, tile_id in enumerate(inter.three_tile_ids):
                tile = board.tiles[tile_id] if 0 <= tile_id < len(board.tiles) else None
                if not tile or tile.type not in terrain_to_res:
                    continue
                res = terrain_to_res[tile.type]
                pips = inter.three_tile_pips[i] if hasattr(inter, 'three_tile_pips') else 0.0
                resource_pips[res] += pips

            # Determine port ratios
            ratio_dict = {"brick": 4, "wood": 4, "wool": 4, "wheat": 4, "ore": 4}

            if getattr(inter, "port_tf", False):
                port_type = getattr(inter, "port_type", "")
                if port_type == "3:1":
                    ratio_dict = {r: 3 for r in ratio_dict}
                elif port_type.startswith("2:1"):
                    specific_res = port_type.split()[-1].lower()
                    if specific_res in ratio_dict:
                        ratio_dict[specific_res] = 2

            # Calculate total score
            total_score = 0.0
            for res, pips in resource_pips.items():
                if pips > 0:
                    bonus = pips // ratio_dict[res]   # floor division
                    total_score += pips + bonus

            scores.append((inter_id, total_score))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in scores[:k]]

    @staticmethod
    def combined_harbor_bonus_round_minus1(
        board: Board,
        existing_settlements: List[int],   # usually just [first settlement id]
        candidate_id: int,
        candidate_harbor_type: str
    ) -> float:
        """
        Calculate combined harbor value for round -1 when adding second settlement.
        - 3:1 → total pips / 6 (once only)
        - 2:1 X → (X pips s1+s2)/2 + bonuses from existing ports (non-overlapping)
        """
        if not candidate_harbor_type or candidate_harbor_type == "Blank":
            return 0.0

        # Total pips from both settlements (existing + candidate)
        total_pips = [0.0] * 5  # [wheat, ore, wood, brick, wool]
        for sid in existing_settlements + [candidate_id]:
            inter = board.intersections[sid]
            if inter:
                probs = getattr(inter, "all_tile_probabilities",
                                getattr(inter, "three_tile_probabilities_v2", [0.0]*5))
                types = getattr(inter, "all_tile_types", [0]*5)
                for idx in range(5):
                    if types[idx] > 0:
                        total_pips[idx] += probs[idx]

        total_prod = sum(total_pips)

        # ─── Existing ports ─────────────────────────────────────────────────────
        existing_ports = []
        for sid in existing_settlements:
            inter = board.intersections[sid]
            if inter and getattr(inter, "port_tf", False):
                t = getattr(inter, "port_type", "")
                if t and t != "Blank":
                    existing_ports.append(t)

        # ─── Candidate is 3:1 ──────────────────────────────────────────────────
        if "3:1" in candidate_harbor_type:
            return total_prod / 6.0

        # ─── Candidate is 2:1 specific ─────────────────────────────────────────
        res_map = {
            "2:1 Wheat": 0,
            "2:1 Ore":   1,
            "2:1 Wood":  2,
            "2:1 Brick": 3,
            "2:1 Wool": 4,   # or Wool
        }

        if candidate_harbor_type not in res_map:
            return 0.0

        spec_idx = res_map[candidate_harbor_type]
        spec_pips = total_pips[spec_idx]
        bonus_new = spec_pips / 2.0

        # Add value from existing ports
        bonus_existing = 0.0
        for ep in existing_ports:
            if "3:1" in ep:
                # exclude the new 2:1 resource from 3:1 discount
                adjusted_total = total_prod - spec_pips
                bonus_existing += adjusted_total / 6.0

            elif ep in res_map:
                ep_idx = res_map[ep]
                if ep_idx != spec_idx:  # different resource → full add
                    ep_pips = total_pips[ep_idx]
                    bonus_existing += ep_pips / 2.0
                # same resource → skip (no extra stacking)

        return bonus_new + bonus_existing

    @staticmethod
    def _five_strategy_engine(
        board: Board,
        player: Player,
        valid_intersections: List[int],
        game_round: int = -2
    ) -> int:
        if not valid_intersections:
            return -1

        # ────────────────────────────────────────────────
        # Prepare raw TW probabilities (still needed for tiebreaker / fallback)
        # ────────────────────────────────────────────────
        list_of_TW_prob = []
        for inter_id in range(len(board.intersections)):
            inter = board.intersections[inter_id]
            if getattr(inter, "canbuildYNX", "N") == "X":
                continue
            probs_raw = getattr(inter, "all_tile_probabilities",
                                getattr(inter, "three_tile_probabilities_v2",
                                        getattr(inter, "three_tile_probabilities", [0] * 5)))
            probs = [InitialPlacementStrategies._safe_float(v) for v in probs_raw]
            tw_prob = sum(probs)
            list_of_TW_prob.append([inter_id, tw_prob])

        frames = [
            [1, 1, 1, 1, 1],      # 0: Balanced
            [0, 0, 1, 1, 0],      # 1: WB (Wood+Brick)
            [1, 1, 0, 0, 0],      # 2: WO (Wheat+Ore)
            [1, 1, 0, 0, 1],      # 3: WOS (Wheat+Ore+Wool)
            [0, 0, 0, 0, 0],      # 4: Monopoly
        ]

        # ────────────────────────────────────────────────
        # Use precomputed raw data (or compute on first use)
        # ────────────────────────────────────────────────
        if not hasattr(board, 'precomputed_pp_raw'):
            board.precompute_algorithm2_raw()

        player.PP_balanced = []
        player.PP_WB = []
        player.PP_WO = []
        player.PP_WOS = []
        player.PP_monopoly = []

        for inter_id in valid_intersections:
            if inter_id not in board.precomputed_pp_raw:
                continue

            raw = board.precomputed_pp_raw[inter_id]
            three_types = raw['three_types']
            three_probs = raw['three_probs']

            for fr in range(5):
                frame = frames[fr]
                diversity, sum_prob, tiles_having_RP, min_prob, harbor_prob = raw['frames'][fr]

                entry = [inter_id, 0.0, diversity, sum_prob, tiles_having_RP, min_prob, harbor_prob]

                if fr == 0:
                    player.PP_balanced.append(entry)
                elif fr == 1:
                    player.PP_WB.append(entry)
                elif fr == 2:
                    player.PP_WO.append(entry)
                elif fr == 3:
                    player.PP_WOS.append(entry)
                elif fr == 4:
                    player.PP_monopoly.append(entry)

        # ────────────────────────────────────────────────
        # Round -1: Replace standalone harbor with combined/context-aware value
        # ────────────────────────────────────────────────
        if game_round == -1:
            existing_settlements = player.settlements[:]
            if existing_settlements:
                for pp_list in [
                    player.PP_balanced,
                    player.PP_WB,
                    player.PP_WO,
                    player.PP_WOS,
                    player.PP_monopoly
                ]:
                    for entry in pp_list:
                        inter_id = entry[0]
                        inter = board.intersections[inter_id]
                        if not inter:
                            continue

                        harbor_type = ""
                        if getattr(inter, "port_tf", False):
                            harbor_type = getattr(inter, "port_type", "")

                        combined_bonus = InitialPlacementStrategies.combined_harbor_bonus_round_minus1(
                            board,
                            existing_settlements,
                            inter_id,
                            harbor_type
                        )

                        entry[3] += combined_bonus   # add to pure production sum_prob

        # ────────────────────────────────────────────────
        # Add blocked bonus (positive — blocking opponents) — applies in both rounds
        # ────────────────────────────────────────────────
        for pp_list in [
            player.PP_balanced,
            player.PP_WB,
            player.PP_WO,
            player.PP_WOS,
            player.PP_monopoly
        ]:
            for entry in pp_list:
                inter_id = entry[0]
                inter = board.intersections[inter_id]
                if not inter:
                    continue

                neighbors = getattr(inter, "three_intersection_ids", [])
                blocked_pips = 0.0
                for nid in neighbors:
                    raw = board.precomputed_pp_raw.get(nid)
                    if raw:
                        blocked_pips += raw['tw_prob']  # raw neighbor production

                blocked_bonus = blocked_pips * BLOCKED_WEIGHT  # positive!
                entry[3] += blocked_bonus

        # ────────────────────────────────────────────────
        # Apply strict diversity requirement only in round -1
        # ────────────────────────────────────────────────
        def apply_diversity_filter(pp_list, fr):
            if game_round != -1:
                return
            for entry in pp_list:
                diversity = entry[2]
                if (fr == 0 and diversity != 5) or \
                (fr == 1 and diversity != 2) or \
                (fr == 2 and diversity != 2) or \
                (fr == 3 and diversity != 3):
                    entry[1] = 0.0          # zero the strategy points

        apply_diversity_filter(player.PP_balanced, 0)
        apply_diversity_filter(player.PP_WB,       1)
        apply_diversity_filter(player.PP_WO,       2)
        apply_diversity_filter(player.PP_WOS,      3)
        # Monopoly (fr=4) intentionally not filtered

        # ────────────────────────────────────────────────
        # Sorting (priorities preserved)
        # ────────────────────────────────────────────────
        s1 = sorted(player.PP_balanced, key=lambda x: x[4], reverse=True)   # tiles_having_RP
        s2 = sorted(s1,         key=lambda x: x[2], reverse=True)           # diversity
        player.PP_balanced = sorted(s2, key=lambda x: x[3], reverse=True)   # sum_prob

        for pp_list in [player.PP_WB, player.PP_WO, player.PP_WOS]:
            s1 = sorted(pp_list, key=lambda x: x[2], reverse=True)          # diversity
            s2 = sorted(s1,      key=lambda x: x[5], reverse=True)          # min_prob
            pp_list[:] = sorted(s2, key=lambda x: x[3], reverse=True)       # sum_prob

        s1 = sorted(player.PP_monopoly, key=lambda x: x[2], reverse=True)   # diversity (max count)
        player.PP_monopoly = sorted(s1, key=lambda x: x[3], reverse=True)   # sum_prob

        # ────────────────────────────────────────────────
        # Assign shared points on ties
        # ────────────────────────────────────────────────
        def assign_shared_points(pp_list, sort_keys_indices):
            def tie_key(entry):
                return tuple(entry[i] for i in sort_keys_indices)

            current_rank = 1
            for key, group_iter in groupby(pp_list, key=tie_key):
                group = list(group_iter)
                num_tied = len(group)
                start_rank = current_rank
                end_rank = min(current_rank + num_tied - 1, TOP_N)
                if start_rank > TOP_N:
                    points = 0.0
                else:
                    total_pts = sum(float(TOP_N - r + 1) for r in range(start_rank, end_rank + 1))
                    points = total_pts / num_tied
                for entry in group:
                    entry[1] = points
                current_rank += num_tied

        assign_shared_points(player.PP_balanced, [4, 2, 3])   # tiles_having_RP, diversity, sum_prob
        assign_shared_points(player.PP_WB,       [2, 5, 3])   # diversity, min_prob, sum_prob
        assign_shared_points(player.PP_WO,       [2, 5, 3])
        assign_shared_points(player.PP_WOS,      [2, 5, 3])
        assign_shared_points(player.PP_monopoly, [2, 3])      # diversity, sum_prob

        # ────────────────────────────────────────────────
        # Rank-sum across strategies + pure TW pips tiebreaker
        # ────────────────────────────────────────────────
        list_of_TWs = []
        TW_in_WO = []

        for entry in player.PP_balanced:
            if entry[1] > 0:
                list_of_TWs.append([entry[0], entry[1], "Balanced"])

        for entry in player.PP_WB:
            if entry[1] > 0:
                list_of_TWs.append([entry[0], entry[1], "WB"])

        for entry in player.PP_WO:
            if entry[1] > 0:
                iid = entry[0]
                list_of_TWs.append([iid, entry[1], "WO"])
                TW_in_WO.append(iid)

        for entry in player.PP_WOS:
            if entry[1] > 0:
                iid = entry[0]
                if iid not in TW_in_WO:
                    list_of_TWs.append([iid, entry[1], "WOS"])

        for entry in player.PP_monopoly:
            if entry[1] > 0:
                list_of_TWs.append([entry[0], entry[1], "Monopoly"])

        # Add pure TW pips as fallback for any position not ranked high enough
        sort_TW_prob = sorted(list_of_TW_prob, key=lambda x: x[1], reverse=True)
        for i in range(min(TOP_N, len(sort_TW_prob))):
            list_of_TWs.append([sort_TW_prob[i][0], float(TOP_N - i), "TW_prob"])

        # Aggregate by intersection (sum strategy points)
        sort_list_of_TWs = sorted(list_of_TWs, key=lambda x: x[0])
        list_of_unique_TWs = []
        if sort_list_of_TWs:
            current_id = sort_list_of_TWs[0][0]
            current_val = 0.0
            for entry in sort_list_of_TWs:
                iid = entry[0]
                if iid == current_id:
                    current_val += entry[1]
                else:
                    tw_prob = next((p[1] for p in list_of_TW_prob if p[0] == current_id), 0.0)
                    list_of_unique_TWs.append([current_id, current_val, tw_prob])
                    current_id = iid
                    current_val = entry[1]
            tw_prob = next((p[1] for p in list_of_TW_prob if p[0] == current_id), 0.0)
            list_of_unique_TWs.append([current_id, current_val, tw_prob])

        sort_list_of_unique_TWs = sorted(list_of_unique_TWs, key=lambda x: x[1], reverse=True)

        if not sort_list_of_unique_TWs:
            return valid_intersections[0] if valid_intersections else -1

        best_inter = sort_list_of_unique_TWs[0][0]

        # ────────────────────────────────────────────────
        # Fill player.PP_strategies (per-strategy scores of the chosen spot)
        # ────────────────────────────────────────────────
        inter = board.intersections[best_inter]
        three_probs = getattr(inter, "all_tile_probabilities",
                            getattr(inter, "three_tile_probabilities_v2",
                                    getattr(inter, "three_tile_probabilities", [0]*5)))
        three_types = getattr(inter, "all_tile_types", [0]*5)

        strategy_scores = [0.0] * 5
        for fr in range(5):
            frame = frames[fr]
            if fr == 4:
                strategy_scores[fr] = max(three_probs) if three_probs else 0.0
            else:
                s = 0.0
                for idx, t in enumerate(three_types):
                    if t > 0 and idx < len(frame) and frame[idx] == 1:
                        s += three_probs[idx]
                harbor_prob = 0.0
                if getattr(inter, "port_tf", False):
                    harbor_type = getattr(inter, "port_type", "")
                    if harbor_type and harbor_type != "Blank":
                        harbor_prob = InitialPlacementStrategies.check_harbor(
                            harbor_type, frame, three_types, three_probs
                        )
                s += harbor_prob
                strategy_scores[fr] = s
        player.PP_strategies = strategy_scores[:]

        # ────────────────────────────────────────────────
        # Debug logging (enhanced with blocked bonus info)
        # ────────────────────────────────────────────────
        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(f"\n=== FIVE STRATEGY ENGINE FULL DEBUG - Player {player.id} "
                        f"(algorithm=2, round={game_round}) ===\n")
                f.write("Columns: id | strategy_points | diversity | sum_prob | tiles_having_RP | "
                        f"min_prob | harbor_bonus | blocked_bonus\n")
                f.write("-" * 130 + "\n")

                for name, lst in [
                    ("Balanced", player.PP_balanced),
                    ("WB", player.PP_WB),
                    ("WO", player.PP_WO),
                    ("WOS", player.PP_WOS),
                    ("Monopoly", player.PP_monopoly),
                ]:
                    f.write(f"{name} (FULL list):\n")
                    for e in lst:
                        inter_id = e[0]
                        inter = board.intersections[inter_id]
                        blocked_bonus = 0.0
                        if inter:
                            neighbors = getattr(inter, "three_intersection_ids", [])
                            blocked_pips = 0.0
                            for nid in neighbors:
                                raw = board.precomputed_pp_raw.get(nid)
                                if raw:
                                    blocked_pips += raw['tw_prob']
                            blocked_bonus = blocked_pips * BLOCKED_WEIGHT

                        f.write(f"{inter_id:3d} | {e[1]:13.1f} | {e[2]:2d} | {e[3]:6.1f} | "
                                f"{e[4]:2d} | {e[5]:6.1f} | {e[6]:6.1f} | {blocked_bonus:6.1f}\n")
                    f.write("-" * 130 + "\n")

                f.write("\nRank-sum selection (top 10):\n")
                f.write("id | total_strategy_points | tw_prob\n")
                for e in sort_list_of_unique_TWs[:10]:
                    f.write(f"{e[0]:3d} | {e[1]:20.1f} | {e[2]:6.1f}\n")

                # Final chosen intersection (with blocked_bonus)
                blocked_bonus_final = 0.0
                blocked_sum = 0.0
                if inter:
                    neighbors = getattr(inter, "three_intersection_ids", [])
                    blocked_pips = 0.0
                    for nid in neighbors:
                        raw = board.precomputed_pp_raw.get(nid)
                        if raw:
                            blocked_pips += raw['tw_prob']
                    blocked_bonus_final = blocked_pips * BLOCKED_WEIGHT
                    blocked_sum = blocked_pips

                f.write(f"\nChosen: {best_inter}\n")
                f.write(f"  Blocked TW sum neighbours: {blocked_sum:.1f} "
                        f"(bonus added: +{blocked_bonus_final:.2f})\n")
                f.write(f"  Final PP_strategies: {[round(x,1) for x in strategy_scores]}\n")
                f.write("=== END DEBUG ===\n\n")

        return best_inter

    @staticmethod
    def find_best_road_having_port(
        board: Board,
        settlement_id: int,
        top_tws: List[int],
        blocked_tws: List[int],
        selected_tws: List[int]
    ) -> Tuple[int, int] | None:
        conn = board.list_of_roads_connected_to_intersection
        legs = conn[settlement_id] if isinstance(conn, list) and 0 <= settlement_id < len(conn) else conn.get(settlement_id, []) if isinstance(conn, dict) else []
        if not legs:
            return None
        best_score = -1
        best_road = None
        for leg in legs:
            a, b = leg
            next_tw = a if b == settlement_id else b
            if next_tw == settlement_id:
                continue
            prob_raw = getattr(board.intersections[next_tw], "three_tile_probabilities_v2", [0]*5)
            prob = sum(InitialPlacementStrategies._safe_float(v) for v in prob_raw)
            if prob > best_score:
                best_score = prob
                best_road = (settlement_id, next_tw)
        return best_road

    @staticmethod
    def find_best_road_missing_port(
        board: Board,
        settlement_id: int,
        player: Player,
        top_tws: List[int],
        blocked_tws: List[int],
        selected_tws: List[int]
    ) -> Tuple[int, int] | None:
        conn = board.list_of_roads_connected_to_intersection
        legs = conn[settlement_id] if isinstance(conn, list) and 0 <= settlement_id < len(conn) else conn.get(settlement_id, []) if isinstance(conn, dict) else []
        if not legs:
            return None
        tree = []
        for leg in legs:
            a, b = leg
            current = a if b == settlement_id else b
            if current == settlement_id:
                continue
            next_legs = conn.get(current, []) if isinstance(conn, dict) else (conn[current] if 0 <= current < len(conn) else [])
            for next_leg in next_legs:
                n1, n2 = next_leg
                next_tw = n1 if n2 == current else n2
                if next_tw in (settlement_id, current):
                    continue
                tree.append([current, next_tw, settlement_id])
        if not tree:
            for leg in legs:
                a, b = leg
                candidate = a if b == settlement_id else b
                if candidate != settlement_id:
                    return (settlement_id, candidate)
            return None
        tw_probs = {}
        for i, inter in enumerate(board.intersections):
            if inter:
                probs_raw = getattr(inter, "three_tile_probabilities_v2", [0]*5)
                tw_probs[i] = sum(InitialPlacementStrategies._safe_float(v) for v in probs_raw)
        direction_scores = defaultdict(float)
        for entry in tree:
            _, future_tw, _ = entry
            if future_tw in selected_tws or future_tw in blocked_tws:
                continue
            score = tw_probs.get(future_tw, 0)
            if future_tw in top_tws:
                score *= 1.5
            direction_scores[entry[0]] += score
        if not direction_scores:
            for leg in legs:
                a, b = leg
                candidate = a if b == settlement_id else b
                if candidate != settlement_id:
                    return (settlement_id, candidate)
        best_first = max(direction_scores, key=direction_scores.get)
        return (settlement_id, best_first)
    
    @staticmethod
    def precompute_algorithm2_raw(board: Board) -> None:
        """Moved from Board class — computes once all static raw metrics for algorithm=2."""
        if hasattr(board, 'precomputed_pp_raw'):
            return

        board.precomputed_pp_raw = {}

        frames = [
            [1,1,1,1,1], [0,0,1,1,0], [1,1,0,0,0],
            [1,1,0,0,1], [0,0,0,0,0]
        ]

        for inter_id in range(len(board.intersections)):
            inter = board.intersections[inter_id]
            if inter is None or inter.id in board.INTERSECTION_IN_WATER:
                continue

            three_types = getattr(inter, "all_tile_types", [0]*5)
            three_probs  = getattr(inter, "all_tile_probabilities",
                                getattr(inter, "three_tile_probabilities_v2",
                                        getattr(inter, "three_tile_probabilities", [0.0]*5)))

            raw_frames = []
            tw_prob = sum(InitialPlacementStrategies._safe_float(p) for p in three_probs)

            for fr, frame in enumerate(frames):
                if fr == 4:  # monopoly
                    diversity = max(three_types) if three_types else 0
                    tiles_having_RP = diversity
                    sum_prob = max(three_probs) if three_probs else 0.0
                    min_prob = sum_prob
                else:
                    contributing = set()
                    tiles_having_RP = 0
                    sum_prob = 0.0
                    min_prob = 99.0
                    for idx, t in enumerate(three_types):
                        if t > 0 and idx < len(frame) and frame[idx] == 1:
                            contributing.add(idx)
                            tiles_having_RP += 1
                            p = InitialPlacementStrategies._safe_float(three_probs[idx])
                            sum_prob += p
                            if p < min_prob:
                                min_prob = p
                    diversity = len(contributing)
                    if min_prob == 99.0:
                        min_prob = 0.0

                harbor_prob = 0.0
                if getattr(inter, "port_tf", False):
                    harbor_type = getattr(inter, "port_type", "")
                    if harbor_type and harbor_type != "Blank":
                        harbor_prob = InitialPlacementStrategies.check_harbor(
                            harbor_type, frame, three_types, three_probs
                        )
                # Note: do NOT add harbor_prob to sum_prob here (pure pips only)

                raw_frames.append([diversity, sum_prob, tiles_having_RP, min_prob, harbor_prob])

            board.precomputed_pp_raw[inter_id] = {
                'tw_prob': tw_prob,
                'frames': raw_frames,
                'three_types': three_types[:],
                'three_probs': [InitialPlacementStrategies._safe_float(p) for p in three_probs]
            }

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"precompute_algorithm2_raw | Stored raw data for "
                        f"{len(board.precomputed_pp_raw)} intersections\n")    