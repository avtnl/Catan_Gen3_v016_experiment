"""
core/fast_forward.py

Event-driven fast-forward engine for Catan.
Uses the configured timing backend (expected-hand or Markov), applies skipped expected income
to all players, then lets PlayerOutlook choose the best legal target for the chosen action.
"""

import pygame
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.game import Game
from core.player import Player
from core.player_outlook import PlayerOutlook
from core.constants import (
    MG,
    FILENAME_MG,
    ResourceCard,
    RESOURCE_ORDER,
    SETTLEMENT_AT_FAILURE,
    MARKOV_ENABLE_HEAVY_REFINEMENT,
)

try:
    from core.action_evaluator import (
        rank_expected_viable_actions,
        expected_action_evaluations_to_rows,
    )
    ACTION_EVALUATOR_IMPORT_ERROR = None
except Exception as exc:
    rank_expected_viable_actions = None
    expected_action_evaluations_to_rows = None
    ACTION_EVALUATOR_IMPORT_ERROR = exc

try:
    from core.victory_path_evaluator import load_142_ways
    VICTORY_PATH_IMPORT_ERROR = None
except Exception as exc:
    load_142_ways = None
    VICTORY_PATH_IMPORT_ERROR = exc


class FastForwardEngine:
    """Orchestrates event-driven fast-forward simulation."""

    def __init__(self, game: Game):
        self.game = game
        self.action_table: List[Dict[str, Any]] = []

        if not hasattr(self.game, "ff_pending_event"):
            self.game.ff_pending_event = None
        if not hasattr(self.game, "ff_waiting_for_play"):
            self.game.ff_waiting_for_play = False
        if not hasattr(self.game, "ff_button_mode"):
            self.game.ff_button_mode = "JUMP" if self.game.phase == "Execution" else "PLAY"

        if not hasattr(self.game, "ff_ignore_resource_cards"):
            self.game.ff_ignore_resource_cards = False

        if not hasattr(self.game, "ff_contract_rows"):
            self.game.ff_contract_rows = []

        if ACTION_EVALUATOR_IMPORT_ERROR is not None:
            print("⚠️ action_evaluator import failed:", repr(ACTION_EVALUATOR_IMPORT_ERROR))

        if VICTORY_PATH_IMPORT_ERROR is not None:
            print("⚠️ victory_path_evaluator import failed:", repr(VICTORY_PATH_IMPORT_ERROR))            

    # ============================================================
    # Helpers
    # ============================================================
    def _resource_timing_engine(self) -> str:
        """Return the configured timing backend in a safe, dependency-light way."""
        try:
            from core.constants import RESOURCE_TIMING_ENGINE
            return str(RESOURCE_TIMING_ENGINE or "hybrid").strip().lower()
        except Exception:
            return "hybrid"

    def _expected_hand_only_runtime(self) -> bool:
        """True when this run must not rely on Markov timing at all."""
        return self._resource_timing_engine() == "expected_hand"

    def _activity_to_requested_activity_ff(self, activity: Any) -> str:
        """Normalize event/action keys to executable requested_activity names."""
        raw = str(activity or "").strip().lower()

        if raw in ("dev_card", "development_card", "buy_dev_card", "buy_discovery_card"):
            return "buy_discovery_card"

        if raw in ("city", "upgrade_city", "upgrade_to_city"):
            return "upgrade_to_city"

        if raw in (
            "settlement",
            "new_settlement",
            "build_settlement",
            "next_settlement",
            "settlement_0r",
            "settlement_1r",
            "settlement_2r",
        ):
            return "new_settlement"

        return raw

    def _sanitize_zero_time_event_times(
        self,
        player: Player,
        event_times: Dict[str, Any],
        *,
        timing_source: str = "expected_hand",
    ) -> Dict[str, Any]:
        """
        Defensive EH-only guard.

        The expected-hand estimator can sometimes return 0.0 because its
        continuous trade-feasibility check is optimistic. Before we allow a
        zero-time prediction into the fast-forward table, verify the exact
        staged plan can really execute now. If not, suppress only that
        zero-time event for this rebuild so the table can select the next
        viable future strategy.
        """
        sanitized = dict(event_times or {})

        for key, raw_score in list(sanitized.items()):
            key_text = str(key or "")
            if key_text.startswith("__"):
                continue

            try:
                score = float(raw_score)
            except Exception:
                continue

            if score > 1e-9:
                continue

            requested = self._activity_to_requested_activity_ff(key_text)
            if requested not in ("new_settlement", "upgrade_to_city", "buy_discovery_card"):
                continue

            try:
                staged_plan = self._build_staged_plan(player, requested)
                guard_info = self._is_staged_plan_executable_now(
                    player=player,
                    staged_plan=staged_plan,
                    requested_activity=requested,
                )
            except Exception as exc:
                guard_info = {
                    "executable": False,
                    "reason": f"zero-time exact guard failed: {exc!r}",
                    "guard_mode": "zero_time_sanitize_error",
                }

            debug = sanitized.get("__debug__")
            item_debug = None
            if isinstance(debug, dict):
                item_debug = debug.get(key_text)

            if not bool(guard_info.get("executable", False)):
                sanitized[key] = 9999.0

                if isinstance(item_debug, dict):
                    item_debug["zero_time_suppressed_by_exact_guard"] = True
                    item_debug["zero_time_suppression_reason"] = guard_info.get("reason")
                    item_debug["zero_time_suppression_timing_source"] = timing_source
                    item_debug["found"] = False
                    item_debug["confidence_label"] = "suppressed_exact_guard"

                print("EH ZERO-TIME SUPPRESSED:", {
                    "player": getattr(player, "id", "?"),
                    "activity": key_text,
                    "requested": requested,
                    "timing_source": timing_source,
                    "reason": guard_info.get("reason"),
                })

            elif isinstance(item_debug, dict):
                item_debug["zero_time_verified_by_exact_guard"] = True
                item_debug["zero_time_verification_reason"] = guard_info.get("reason", "ok")
                item_debug["zero_time_verification_timing_source"] = timing_source

        return sanitized

    def _reestimate_contract_after_play_failure(
        self,
        player: Player,
        *,
        failure_result: Optional[Dict[str, Any]] = None,
        pending: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        After PLAY fails exact guard, rebuild this player's future contract
        from the current hand/board state and put it back into the FF table.

        This is the important EH-only behavior: a failed pursued strategy does
        not remove the player from the schedule. We reassess all expected
        viable actions, let the 142-way ranking/PLAY logic evaluate them later,
        and schedule the next expected round/turn for the newly selected best
        strategy.
        """
        player_id = int(getattr(player, "id", -1))

        rows = [
            dict(r)
            for r in (getattr(self.game, "ff_contract_rows", []) or [])
            if int(r.get("player_id", -1)) != player_id
        ]

        fresh_contract = None
        try:
            fresh = self._fresh_contract_for_player(player)
            if fresh is not None:
                fresh_contract = self._stamp_contract_row(fresh)
                fresh_contract["reestimated_after_play_failure"] = True
                fresh_contract["play_failure_reason"] = (failure_result or {}).get("reason")
                fresh_contract["play_failure_requested_activity"] = (
                    (failure_result or {}).get("requested_activity")
                    or (pending or {}).get("requested_activity")
                )
                rows.append(fresh_contract)
        except Exception as exc:
            print("PLAY FAILURE RE-ESTIMATE FAILED:", {
                "player": player_id,
                "error": repr(exc),
            })
            fresh_contract = None

        rows = self._sort_ff_contract_rows(rows)
        self.game.ff_contract_rows = rows

        print("PLAY FAILURE RE-ESTIMATED CONTRACT:", {
            "player": player_id,
            "created": fresh_contract is not None,
            "strategy": None if fresh_contract is None else fresh_contract.get("strategy"),
            "requested": None if fresh_contract is None else fresh_contract.get("requested_activity"),
            "pred_round": None if fresh_contract is None else fresh_contract.get("pred_round"),
            "pred_turn": None if fresh_contract is None else fresh_contract.get("pred_turn"),
            "delta": None if fresh_contract is None else fresh_contract.get("delta_rolls"),
            "source": None if fresh_contract is None else (
                fresh_contract.get("scheduling_source")
                or fresh_contract.get("display_source")
                or fresh_contract.get("source_mode")
            ),
        })

        return fresh_contract

    def _ff_now_player_turns(self) -> float:
        return float(getattr(self.game, "ff_elapsed_rounds", 0.0) or 0.0)

    def _stamp_contract_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)

        # Raw Markov prediction before enforcing that the player can only act
        # on their own turn.
        raw_delta = float(row.get("delta_rolls", 9999.0))
        player_sequence = int(row.get("player_sequence", row.get("pred_turn", 1)))

        scheduled_abs = self._scheduled_abs_round_time_for_player(
            delta_rolls=raw_delta,
            player_sequence=player_sequence,
        )

        current_round = max(1, int(getattr(self.game, "round", 1)))
        current_turn = max(1, int(getattr(self.game, "turn", 1)))
        num_players = max(len(getattr(self.game, "players", [])), 1)

        now_abs = float(current_round - 1) + float(current_turn - 1) / float(num_players)
        scheduled_delta = max(0.0, scheduled_abs - now_abs)

        pred_round, pred_turn = self._estimate_predicted_round_turn(
            raw_delta,
            player_sequence,
        )

        row["raw_markov_delta"] = raw_delta
        row["contract_delta_player_turns"] = scheduled_delta
        row["delta_rolls"] = scheduled_delta
        row["abs_player_turns"] = scheduled_abs
        row["pred_round"] = pred_round
        row["pred_turn"] = pred_turn
        row["contract_active"] = True

        return row

    def _fresh_contract_for_player(self, player: Player) -> Optional[Dict[str, Any]]:
        fresh_rows = self._build_prediction_rows()
        for row in fresh_rows:
            if int(row.get("player_id", -1)) == int(player.id):
                return self._stamp_contract_row(row)
        return None

    def _sort_ff_contract_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = list(rows or [])
        rows.sort(key=lambda r: (
            float(r.get("abs_player_turns", 9999.0)),
            int(r.get("pred_turn", 999999)) if r.get("pred_turn", "?") != "?" else 999999,
            int(r.get("player_id", 999999)),
        ))
        return rows

    def _ensure_ff_contract_rows(self) -> List[Dict[str, Any]]:
        rows = list(getattr(self.game, "ff_contract_rows", []) or [])

        # First table: build all contracts from one consistent prediction table.
        if not rows:
            rows = [self._stamp_contract_row(r) for r in self._build_prediction_rows()]
            rows = self._sort_ff_contract_rows(rows)
            self.game.ff_contract_rows = rows
            return rows

        # Later: only fill genuinely missing players.
        existing_ids = {int(r.get("player_id", -1)) for r in rows}

        for player in self.game.players:
            if int(player.id) not in existing_ids:
                fresh = self._fresh_contract_for_player(player)
                if fresh is not None:
                    rows.append(fresh)

        rows = self._sort_ff_contract_rows(rows)
        self.game.ff_contract_rows = rows
        return rows

    def _contract_has_board_conflict(self, row: Dict[str, Any]) -> bool:
        player_id = int(row.get("player_id", -1))
        player = next((p for p in self.game.players if int(p.id) == player_id), None)
        if player is None:
            return True

        requested = str(row.get("requested_activity", row.get("strategy", ""))).lower()
        strategy = str(row.get("strategy", "")).lower()

        # Settlement strategy conflict:
        # Keep settlement_1r if there is still a viable settlement reachable with <= 1 new road.
        if requested == "new_settlement":
            predicted_roads = row.get("predicted_extra_roads", None)

            if predicted_roads is None and strategy.startswith("settlement_"):
                try:
                    predicted_roads = int(strategy.split("_")[1].replace("r", ""))
                except Exception:
                    predicted_roads = None

            if predicted_roads is None:
                return False

            min_roads, _target = self._estimate_min_extra_roads_to_any_settlement(player)

            # This is the agreed rule:
            # settlement_1r survives if a viable target exists with <= 1 road.
            return int(min_roads) > int(predicted_roads)

        # City contract only conflicts if no city upgrade is structurally available anymore.
        if requested == "upgrade_to_city":
            if len(getattr(player, "cities", [])) >= 4:
                return True
            outlook = self._ensure_outlook(player)
            return not bool(outlook.get_viable_city_upgrades())

        # Dev-card contract only conflicts structurally if the deck is empty.
        if requested == "buy_discovery_card":
            return not bool(getattr(self.game, "dcards_stack", None))

        return False

    def _repair_ff_contract_rows(self) -> List[Dict[str, Any]]:
        rows = self._ensure_ff_contract_rows()
        repaired = []

        for row in rows:
            player_id = int(row.get("player_id", -1))
            player = next((p for p in self.game.players if int(p.id) == player_id), None)
            if player is None:
                continue

            if self._contract_has_board_conflict(row):
                fresh = self._fresh_contract_for_player(player)
                if fresh is not None:
                    repaired.append(fresh)
            else:
                repaired.append(row)

        repaired = self._sort_ff_contract_rows(repaired)
        self.game.ff_contract_rows = repaired
        return repaired

    def _complete_ff_contract_for_player(self, player_id: int) -> None:
        rows = [
            r for r in getattr(self.game, "ff_contract_rows", []) or []
            if int(r.get("player_id", -1)) != int(player_id)
        ]

        player = next(
            (p for p in self.game.players if int(p.id) == int(player_id)),
            None
        )

        if player is not None:
            fresh = self._fresh_contract_for_player(player)
            if fresh is not None:
                rows.append(fresh)

        rows = self._sort_ff_contract_rows(rows)
        self.game.ff_contract_rows = rows

    def _contract_rows_for_display(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return copied rows with delta_rolls converted from absolute contract time
        to remaining player-turns for display only.

        Do not mutate the real contract rows.
        """
        now = self._ff_now_player_turns()
        display_rows = []

        for row in rows:
            r = dict(row)

            if "abs_player_turns" in r:
                r["delta_rolls"] = max(
                    0.0,
                    float(r.get("abs_player_turns", 9999.0)) - now
                )

            display_rows.append(r)

        return display_rows

    def _remove_ff_contract_for_player(self, player_id: int) -> None:
        rows = [
            r for r in getattr(self.game, "ff_contract_rows", []) or []
            if int(r.get("player_id", -1)) != int(player_id)
        ]
        self.game.ff_contract_rows = self._sort_ff_contract_rows(rows)

    def _ignore_resource_cards(self) -> bool:
        """
        Return True when fast-forward should ignore actual owned resource cards.
        """
        return bool(getattr(self.game, "ff_ignore_resource_cards", False))

    def _scheduled_abs_round_time_for_player(self, delta_rolls: float, player_sequence: int) -> float:
        """
        Return the absolute round-time of the first legal future turn for this player
        at or after the Markov-predicted ready time.
        """
        try:
            delta = float(delta_rolls)
        except Exception:
            return 9999.0

        if delta >= 9999.0:
            return 9999.0

        num_players = max(len(getattr(self.game, "players", [])), 1)

        player_seq = max(1, min(num_players, int(player_sequence)))
        player_slot = player_seq - 1

        current_round = max(1, int(getattr(self.game, "round", 1)))
        current_turn = max(1, int(getattr(self.game, "turn", 1)))
        current_turn = max(1, min(num_players, current_turn))

        now_abs = float(current_round - 1) + float(current_turn - 1) / float(num_players)
        ready_abs = now_abs + max(0.0, delta)

        eps = 1e-9
        min_slot_index = int(math.ceil((ready_abs * num_players) - eps))

        while (min_slot_index % num_players) != player_slot:
            min_slot_index += 1

        return float(min_slot_index) / float(num_players)

    def _get_victory_ways_for_ff_ranking(self) -> List[Any]:
        """
        Return cached 142-way data for PLAY-time and same-turn action ranking.

        Diagnostic version:
        - prints why ranking is unavailable
        - caches successful load on game.ff_victory_ways
        """
        for attr_name in (
            "ff_victory_ways",
            "victory_ways",
            "ways_142",
            "ways",
            "victory_path_ways",
        ):
            ways = getattr(self.game, attr_name, None)
            if ways:
                print("142-WAY RANKING: using cached ways", {
                    "attr": attr_name,
                    "count": len(ways),
                })
                return list(ways)

        if load_142_ways is None:
            print("142-WAY RANKING UNAVAILABLE:", {
                "reason": "load_142_ways_is_None",
                "victory_path_import_error": repr(VICTORY_PATH_IMPORT_ERROR),
                "action_evaluator_import_error": repr(ACTION_EVALUATOR_IMPORT_ERROR),
            })
            return []

        try:
            root = Path(__file__).resolve().parents[1]

            candidate_paths = [
                root / "data" / "142_ways.csv",
                root / "data" / "victory_ways_142.csv",
                root / "data" / "valid_victory_ways_142.csv",
                root / "data" / "catan_142_ways_resource_requirements.csv",

                root / "142_ways.csv",
                root / "victory_ways_142.csv",
                root / "valid_victory_ways_142.csv",
                root / "catan_142_ways_resource_requirements.csv",
            ]

            print("142-WAY RANKING: checking paths", {
                "root": str(root),
                "paths": [str(p) for p in candidate_paths],
            })

            for path in candidate_paths:
                if path.exists():
                    ways = load_142_ways(path)
                    ways = list(ways or [])

                    print("142-WAY RANKING: loaded", {
                        "path": str(path),
                        "count": len(ways),
                    })

                    if ways:
                        self.game.ff_victory_ways = list(ways)
                        return list(ways)

            print("142-WAY RANKING UNAVAILABLE:", {
                "reason": "no_candidate_csv_found",
                "checked_paths": [str(p) for p in candidate_paths],
            })

        except Exception as exc:
            print("⚠️ 142-WAY RANKING LOAD FAILED:", repr(exc))

        return []

    def _activity_from_action_evaluation(self, evaluation: Any) -> str:
        """
        Convert an ActionEvaluation into a fast_forward requested_activity.
        """
        try:
            action_type = str(evaluation.action.action_type.value)
        except Exception:
            action_type = str(getattr(evaluation, "action_type", ""))

        action_type = action_type.upper()

        if action_type == "BUY_DEV_CARD":
            return "buy_discovery_card"

        if action_type == "UPGRADE_CITY":
            return "upgrade_to_city"

        if action_type in ("BUILD_SETTLEMENT", "BUILD_ROAD_TO_SETTLEMENT"):
            return "new_settlement"

        return ""

    def _evaluation_primary_target(self, evaluation: Any) -> Optional[int]:
        """
        Extract primary target from an ActionEvaluation if it is an integer target.
        """
        try:
            target = evaluation.action.primary_target
        except Exception:
            return None

        if target is None:
            return None

        try:
            return int(target)
        except Exception:
            return None

    def _build_staged_plan_for_ranked_evaluation(
        self,
        player: Player,
        requested_activity: str,
        evaluation: Any,
    ) -> Dict[str, Any]:
        """
        Build an exact staged plan for the strategic ranking result.

        Important:
        - city target can be forced safely, because city execution only needs chosen_target
        - dev card has no target
        - settlement target forcing is deliberately conservative for now because
        it also needs a road_path; existing _build_staged_plan already computes
        a safe exact settlement plan.
        """
        staged_plan = self._build_staged_plan(player, requested_activity)

        target = self._evaluation_primary_target(evaluation)

        if target is None:
            staged_plan["strategic_target_override"] = None
            return staged_plan

        if requested_activity == "upgrade_to_city":
            candidate_upgrades = staged_plan.get("candidate_upgrades", [])

            # Only force the target if it is still a legal city upgrade.
            if int(target) in [int(x) for x in candidate_upgrades]:
                staged_plan["chosen_upgrade"] = int(target)
                staged_plan["chosen_target"] = int(target)
                staged_plan["strategic_target_override"] = int(target)
            else:
                staged_plan["strategic_target_override_skipped"] = int(target)
                staged_plan["strategic_target_override_reason"] = (
                    "ranked city target is no longer in candidate_upgrades"
                )

        elif requested_activity == "new_settlement":
            # For now, do not overwrite settlement target unless it already matches.
            # Settlement target requires matching road_path; that needs a separate
            # target-specific settlement-plan builder in a later module.
            existing_target = staged_plan.get("chosen_target")

            if existing_target is not None and int(existing_target) == int(target):
                staged_plan["strategic_target_override"] = int(target)
            else:
                staged_plan["strategic_target_override_skipped"] = int(target)
                staged_plan["strategic_target_override_reason"] = (
                    "settlement target override deferred; exact road_path required"
                )

        return staged_plan

    def _requested_activity_from_expected_action(self, action: Dict[str, Any]) -> str:
        """
        Convert a fast_forward expected_viable_actions row to requested_activity.
        """
        activity = str(
            action.get("activity", action.get("raw_activity_key", ""))
        ).strip().lower()

        if activity in ("dev_card", "development_card", "buy_dev_card", "buy_discovery_card"):
            return "buy_discovery_card"

        if activity in ("city", "upgrade_city", "upgrade_to_city"):
            return "upgrade_to_city"

        if activity in ("settlement", "new_settlement", "settlement_0r", "settlement_1r", "settlement_2r"):
            return "new_settlement"

        return activity


    def _target_as_optional_int(self, value: Any) -> Optional[int]:
        """
        Convert target values to int when possible.
        """
        if value is None:
            return None

        text = str(value).strip()
        if text in ("", "-", "?", "None", "development_card"):
            return None

        try:
            return int(float(text))
        except Exception:
            return None


    def _build_staged_plan_with_optional_target(
        self,
        player: Player,
        requested_activity: str,
        target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build an exact staged plan for a requested activity.

        City target can be safely forced if still legal.
        Settlement target forcing is deferred for now because settlement also
        needs a target-specific road_path.
        """
        staged_plan = self._build_staged_plan(player, requested_activity)

        if target is None:
            staged_plan["strategic_target_override"] = None
            return staged_plan

        if requested_activity == "upgrade_to_city":
            candidate_upgrades = staged_plan.get("candidate_upgrades", [])

            try:
                candidate_set = {int(x) for x in candidate_upgrades}
            except Exception:
                candidate_set = set()

            if int(target) in candidate_set:
                staged_plan["chosen_upgrade"] = int(target)
                staged_plan["chosen_target"] = int(target)
                staged_plan["strategic_target_override"] = int(target)
            else:
                staged_plan["strategic_target_override_skipped"] = int(target)
                staged_plan["strategic_target_override_reason"] = (
                    "ranked city target is no longer in candidate_upgrades"
                )

        elif requested_activity == "new_settlement":
            existing_target = staged_plan.get("chosen_target")
            if existing_target is not None and int(existing_target) == int(target):
                staged_plan["strategic_target_override"] = int(target)
            else:
                staged_plan["strategic_target_override_skipped"] = int(target)
                staged_plan["strategic_target_override_reason"] = (
                    "settlement target override deferred; exact road_path required"
                )

        return staged_plan


    def _choose_best_expected_viable_plan_at_play(
        self,
        player: Player,
        pending: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        PLAY-time selector.

        It first tries the 142-way strategic ranking if available.
        If unavailable, it still falls back to Markov-score order from
        expected_viable_actions, so C/D/S alternatives are tried before giving up.

        v015/v018 policy:
        - Markov/light remains primary.
        - Expected-hand metadata is surfaced in every PLAY attempt.
        - If exact guard says an expected action is executable now, force that
          action's EH timing to 0.0/conf=1.0 before ranking. This lets the
          EH=0 hybrid timing override affect the delay penalty.
        - If no ranked expected action passes exact guard, return a clean
          no-executable decision instead of falling back to an impossible
          original staged plan.
        """
        original_requested = str(pending.get("requested_activity", "buy_discovery_card"))
        original_staged_plan = dict(pending.get("staged_plan", {}) or {})
        expected_actions = list(pending.get("expected_viable_actions", []) or [])

        decision: Dict[str, Any] = {
            "used_expected_action_ranking": False,
            "ranking_available": False,
            "ranking_source": "none",
            "selected_requested_activity": original_requested,
            "selected_staged_plan": dict(original_staged_plan),
            "selected_guard_info": {},
            "ranking_rows": [],
            "attempts": [],
            "fallback_reason": None,
        }

        def _guard(requested: str, plan: Dict[str, Any]) -> Dict[str, Any]:
            return self._is_staged_plan_executable_now(
                player=player,
                staged_plan=plan,
                requested_activity=requested,
            )

        def _metadata_from_evaluation(evaluation: Any) -> Dict[str, Any]:
            try:
                return dict(evaluation.action.metadata or {})
            except Exception:
                return {}

        def _spec_from_metadata(
            *,
            requested: str,
            target: Optional[int],
            rank_score: Optional[float],
            source: str,
            metadata: Dict[str, Any],
            fallback_markov_score: Any = None,
        ) -> Dict[str, Any]:
            """Build one PLAY attempt spec with explicit timing metadata."""
            markov_score = metadata.get("markov_score", fallback_markov_score)

            timing_score_for_ranking = metadata.get("timing_score_for_ranking")
            if timing_score_for_ranking is None:
                timing_score_for_ranking = metadata.get("timing_primary_score")
            if timing_score_for_ranking is None:
                timing_score_for_ranking = markov_score

            timing_source_for_ranking = metadata.get("timing_source_for_ranking")
            if timing_source_for_ranking is None:
                timing_source_for_ranking = metadata.get("timing_primary_source")
            if timing_source_for_ranking is None:
                timing_source_for_ranking = source

            return {
                "requested_activity": requested,
                "target": target,
                "rank_score": rank_score,

                "markov_score": markov_score,
                "expected_hand_score": metadata.get("expected_hand_score"),
                "expected_hand_confidence": metadata.get("expected_hand_confidence"),
                "expected_hand_confidence_label": metadata.get("expected_hand_confidence_label"),
                "expected_hand_found": metadata.get("expected_hand_found"),

                "timing_score_for_ranking": timing_score_for_ranking,
                "timing_source_for_ranking": timing_source_for_ranking,

                "source": source,
            }

        def _requested_from_expected_action(action: Dict[str, Any]) -> str:
            code = str(action.get("code", "") or "").upper()
            activity = str(
                action.get("requested_activity")
                or action.get("activity")
                or action.get("raw_activity_key")
                or ""
            ).strip().lower()

            if code == "S" or activity in (
                "new_settlement",
                "settlement",
                "settlement_0r",
                "settlement_1r",
                "settlement_2r",
            ):
                return "new_settlement"

            if code == "C" or activity in (
                "upgrade_to_city",
                "city",
                "upgrade_city",
            ):
                return "upgrade_to_city"

            if code == "D" or activity in (
                "buy_discovery_card",
                "dev_card",
                "development_card",
                "buy_dev_card",
            ):
                return "buy_discovery_card"

            return activity

        def _guard_pay_mode(guard_info: Dict[str, Any]) -> Any:
            pay_info = guard_info.get("pay_info")
            if isinstance(pay_info, dict):
                return pay_info.get("mode")
            return guard_info.get("pay_mode")

        def _force_zero_eh_for_exact_payable_actions(
            expected_actions_to_patch: List[Dict[str, Any]],
        ) -> None:
            """
            v015/v018 defensive EH=0 correction before 142-way ranking.

            If exact guard says an expected action is executable now, force EH
            timing to zero so the hybrid delay override can activate.

            This fixes cases where resource_time_estimator returns a future EH
            score even though exact guard proves the action is payable now
            directly or after legal bank/port trades.
            """
            for action in expected_actions_to_patch or []:
                try:
                    requested = _requested_from_expected_action(action)

                    if requested not in (
                        "new_settlement",
                        "upgrade_to_city",
                        "buy_discovery_card",
                    ):
                        continue

                    # Only zero-correct actions that the EH table already says
                    # are ready now. Future actions do not need exact-guard probing
                    # here, and probing them can invoke target-selection code that
                    # is Markov-specific in older helpers.
                    current_eh_score = self._safe_float_ff(
                        action.get("expected_hand_score", action.get("score", 9999.0)),
                        9999.0,
                    )
                    current_primary_score = self._safe_float_ff(
                        action.get("score", 9999.0),
                        9999.0,
                    )
                    if min(current_eh_score, current_primary_score) > 1e-9:
                        continue

                    target = self._target_as_optional_int(action.get("target"))

                    staged_plan = self._build_staged_plan_with_optional_target(
                        player=player,
                        requested_activity=requested,
                        target=target,
                    )

                    guard_info = _guard(requested, staged_plan)

                    if not bool(guard_info.get("executable", False)):
                        continue

                    old_eh_score = action.get("expected_hand_score")
                    old_eh_conf = action.get("expected_hand_confidence")

                    markov_score = action.get("markov_score", action.get("score"))
                    try:
                        expected_hand_delta = 0.0 - float(markov_score)
                    except Exception:
                        expected_hand_delta = None

                    action["expected_hand_score"] = 0.0
                    action["expected_hand_delta"] = expected_hand_delta
                    action["expected_hand_confidence"] = 1.0
                    action["expected_hand_confidence_label"] = "exact"
                    action["expected_hand_found"] = True
                    action["expected_hand_key"] = action.get(
                        "expected_hand_key",
                        action.get("activity", requested),
                    )

                    # Pre-fill for fallback/debug paths. The action evaluator
                    # should also recompute the same source from EH=0/conf=1.
                    action["timing_score_for_ranking"] = 0.0
                    action["timing_source_for_ranking"] = "expected_hand_zero_override"

                    action["expected_hand_zero_corrected_by_guard"] = True
                    action["expected_hand_zero_correction_reason"] = (
                        "exact_guard_payable_now"
                    )
                    action["expected_hand_zero_correction_old_score"] = old_eh_score
                    action["expected_hand_zero_correction_old_confidence"] = old_eh_conf
                    action["expected_hand_zero_correction_guard_info"] = dict(guard_info)

                    try:
                        chosen_target = (
                            staged_plan.get("chosen_target")
                            or staged_plan.get("chosen_upgrade")
                            or staged_plan.get("chosen_tw")
                        )
                        if chosen_target is not None:
                            action["exact_zero_turn_chosen_target"] = chosen_target
                    except Exception:
                        pass

                    print("EH ZERO CORRECTION:", {
                        "player": getattr(player, "id", "?"),
                        "requested": requested,
                        "code": action.get("code"),
                        "activity": action.get("activity"),
                        "target": target,
                        "old_eh_score": old_eh_score,
                        "old_eh_conf": old_eh_conf,
                        "markov_score": markov_score,
                        "guard_mode": guard_info.get("guard_mode"),
                        "pay_mode": _guard_pay_mode(guard_info),
                    })

                except Exception as exc:
                    print("EH ZERO CORRECTION FAILED:", {
                        "player": getattr(player, "id", "?"),
                        "action": dict(action or {}),
                        "error": repr(exc),
                    })

        if not expected_actions:
            guard_info = _guard(original_requested, original_staged_plan)
            decision["selected_guard_info"] = dict(guard_info)
            decision["fallback_reason"] = "no_expected_viable_actions"
            return decision

        # v015/v018: exact guard is the source of truth for "payable now".
        # Patch EH=0/conf=1 before 142-way ranking applies delay penalties.
        _force_zero_eh_for_exact_payable_actions(expected_actions)

        ranked_attempt_specs: List[Dict[str, Any]] = []

        ways = self._get_victory_ways_for_ff_ranking()

        if ways and rank_expected_viable_actions is not None:
            try:
                evaluations = rank_expected_viable_actions(
                    game=self.game,
                    player=player,
                    ways=ways,
                    expected_viable_actions=expected_actions,
                    top_n=None,
                    require_affordable=False,
                    markov_delay_weight=0.0 if self._expected_hand_only_runtime() else 0.25,
                )

                decision["ranking_available"] = True
                decision["ranking_source"] = "142_way_action_evaluator"

                if expected_action_evaluations_to_rows is not None:
                    try:
                        decision["ranking_rows"] = expected_action_evaluations_to_rows(evaluations)
                    except Exception:
                        decision["ranking_rows"] = []

                for ev in evaluations:
                    requested = self._activity_from_action_evaluation(ev)
                    if not requested:
                        continue

                    target = self._evaluation_primary_target(ev)
                    metadata = _metadata_from_evaluation(ev)

                    ranked_attempt_specs.append(
                        _spec_from_metadata(
                            requested=requested,
                            target=target,
                            rank_score=float(getattr(ev, "final_score", 0.0)),
                            source="142_way_action_evaluator",
                            metadata=metadata,
                        )
                    )

            except Exception as exc:
                decision["fallback_reason"] = f"ranking_failed: {exc}"

        # Critical fallback:
        # even if 142-way ranking is unavailable, try expected actions in Markov order.
        if not ranked_attempt_specs:
            decision["ranking_source"] = "timing_score_order_fallback"

            def _fallback_timing_score(action: Dict[str, Any]) -> float:
                value = action.get("timing_score_for_ranking")
                if value is None:
                    value = action.get("expected_hand_score")
                if value is None:
                    value = action.get("score", 9999.0)
                return self._safe_float_ff(value, 9999.0)

            sorted_actions = sorted(
                expected_actions,
                key=lambda a: (
                    _fallback_timing_score(a),
                    str(a.get("activity", "")),
                    str(a.get("target", "")),
                )
            )

            for action in sorted_actions:
                requested = self._requested_activity_from_expected_action(action)
                if not requested:
                    continue

                metadata = dict(action or {})
                fallback_markov_score = action.get("markov_score")

                ranked_attempt_specs.append(
                    _spec_from_metadata(
                        requested=requested,
                        target=self._target_as_optional_int(action.get("target")),
                        rank_score=None,
                        source="markov_score_order_fallback",
                        metadata=metadata,
                        fallback_markov_score=fallback_markov_score,
                    )
                )

        # v015/v018: Do not append original_staged_plan_fallback here.
        # If all ranked expected actions fail exact guard, return a clean
        # no-executable decision instead of falling back to an impossible original plan.
        tried_keys = set()

        for spec in ranked_attempt_specs:
            requested = str(spec.get("requested_activity", ""))
            target = spec.get("target", None)
            key = (requested, str(target), spec.get("source"))

            if key in tried_keys:
                continue
            tried_keys.add(key)

            staged_plan = self._build_staged_plan_with_optional_target(
                player=player,
                requested_activity=requested,
                target=target,
            )

            guard_info = _guard(requested, staged_plan)

            attempt = {
                "requested_activity": requested,
                "target": target,
                "source": spec.get("source"),
                "rank_score": spec.get("rank_score"),

                "markov_score": spec.get("markov_score"),
                "expected_hand_score": spec.get("expected_hand_score"),
                "expected_hand_confidence": spec.get("expected_hand_confidence"),
                "expected_hand_confidence_label": spec.get("expected_hand_confidence_label"),
                "expected_hand_found": spec.get("expected_hand_found"),
                "timing_score_for_ranking": spec.get("timing_score_for_ranking"),
                "timing_source_for_ranking": spec.get("timing_source_for_ranking"),

                "guard_executable": bool(guard_info.get("executable", False)),
                "guard_reason": guard_info.get("reason"),
                "staged_plan": dict(staged_plan),
            }

            decision["attempts"].append(attempt)

            if bool(guard_info.get("executable", False)):
                decision["used_expected_action_ranking"] = True
                decision["selected_requested_activity"] = requested
                decision["selected_staged_plan"] = dict(staged_plan)
                decision["selected_guard_info"] = dict(guard_info)
                return decision

        decision["selected_requested_activity"] = None
        decision["selected_staged_plan"] = {}
        decision["selected_guard_info"] = {
            "executable": False,
            "reason": "No expected viable action passed exact guard",
            "plan_type": None,
            "guard_mode": "no_exact_executable_action",
        }
        decision["fallback_reason"] = "no_exact_executable_action"

        return decision

    def _build_immediate_executable_action_candidates(
        self,
        player: Player,
    ) -> List[Dict[str, Any]]:
        """
        Build all activities that are exactly executable now.

        This is same-turn logic:
            - no Markov time is added
            - no future prediction is assumed
            - exact staged plan + exact guard decide executability
        """
        candidates: List[Dict[str, Any]] = []

        def _add_candidate(
            activity: str,
            target: Optional[int],
            staged_plan: Dict[str, Any],
            guard_info: Dict[str, Any],
            code: str,
        ) -> None:
            candidates.append({
                "requested_activity": activity,
                "target": target,
                "code": code,
                "staged_plan": dict(staged_plan),
                "guard_info": dict(guard_info),
                "expected_action": {
                    "activity": activity,
                    "target": target if target is not None else (
                        "development_card" if activity == "buy_discovery_card" else None
                    ),
                    "score": 0.0,
                    "code": code,
                    "source": "same_turn_exact_executable",
                },
            })

        # ------------------------------------------------------------
        # 1. City upgrades: test each city target separately.
        # ------------------------------------------------------------
        city_probe = self._build_staged_plan(player, "upgrade_to_city")
        city_targets = list(city_probe.get("candidate_upgrades", []) or [])

        for target in city_targets:
            try:
                target_int = int(target)
            except Exception:
                continue

            staged_plan = self._build_staged_plan_with_optional_target(
                player=player,
                requested_activity="upgrade_to_city",
                target=target_int,
            )

            guard_info = self._is_staged_plan_executable_now(
                player=player,
                staged_plan=staged_plan,
                requested_activity="upgrade_to_city",
            )

            if bool(guard_info.get("executable", False)):
                _add_candidate(
                    activity="upgrade_to_city",
                    target=target_int,
                    staged_plan=staged_plan,
                    guard_info=guard_info,
                    code="C",
                )

        # ------------------------------------------------------------
        # 2. New settlement: for now use the exact staged plan builder.
        #    Target-specific settlement override comes later, because
        #    settlement needs a matching road_path.
        # ------------------------------------------------------------
        settlement_plan = self._build_staged_plan(player, "new_settlement")

        settlement_guard = self._is_staged_plan_executable_now(
            player=player,
            staged_plan=settlement_plan,
            requested_activity="new_settlement",
        )

        if bool(settlement_guard.get("executable", False)):
            target = (
                settlement_plan.get("chosen_target")
                or settlement_guard.get("chosen_target")
            )

            try:
                target = int(target) if target is not None else None
            except Exception:
                target = None

            _add_candidate(
                activity="new_settlement",
                target=target,
                staged_plan=settlement_plan,
                guard_info=settlement_guard,
                code="S",
            )

        # ------------------------------------------------------------
        # 3. Dev card: one candidate.
        #    If still executable after buying one, the chain loop will
        #    rebuild candidates and can buy another.
        # ------------------------------------------------------------
        dev_plan = self._build_staged_plan(player, "buy_discovery_card")

        dev_guard = self._is_staged_plan_executable_now(
            player=player,
            staged_plan=dev_plan,
            requested_activity="buy_discovery_card",
        )

        try:
            hand_debug = player.rcards_in_hand()[0]
        except Exception as exc:
            hand_debug = f"hand_read_error: {exc}"

        print("SAME-TURN DEV PROBE:", {
            "player": getattr(player, "id", "?"),
            "hand": hand_debug,
            "plan_available": dev_plan.get("plan_available"),
            "plan_reason": dev_plan.get("reason"),
            "guard_executable": bool(dev_guard.get("executable", False)),
            "guard_reason": dev_guard.get("reason"),
            "pay_mode": dev_guard.get("pay_mode"),
            "payable_direct": dev_guard.get("payable_direct"),
            "payable_after_trades": dev_guard.get("payable_after_trades"),
            "pay_info": dev_guard.get("pay_info"),
        })

        if bool(dev_guard.get("executable", False)):
            _add_candidate(
                activity="buy_discovery_card",
                target=None,
                staged_plan=dev_plan,
                guard_info=dev_guard,
                code="D",
            )

        return candidates

    def _rank_immediate_executable_action_candidates(
        self,
        player: Player,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rank exact executable same-turn candidates.

        Preferred:
            142-way action evaluator

        Fallback:
            simple deterministic priority:
                city > settlement > dev card
        """
        if not candidates:
            return []

        print("SAME-TURN RANK INPUT:", {
            "player": getattr(player, "id", "?"),
            "count": len(candidates),
            "items": [
                {
                    "activity": c.get("requested_activity"),
                    "target": c.get("target"),
                    "guard_reason": (c.get("guard_info", {}) or {}).get("reason"),
                }
                for c in candidates
            ],
        })

        ways = self._get_victory_ways_for_ff_ranking()

        if ways and rank_expected_viable_actions is not None:
            try:
                expected_actions = [
                    dict(candidate.get("expected_action", {}))
                    for candidate in candidates
                ]

                evaluations = rank_expected_viable_actions(
                    game=self.game,
                    player=player,
                    ways=ways,
                    expected_viable_actions=expected_actions,
                    top_n=None,
                    require_affordable=False,
                    markov_delay_weight=0.0,
                )

                ranked: List[Dict[str, Any]] = []
                used_ids = set()

                for ev in evaluations:
                    requested = self._activity_from_action_evaluation(ev)
                    target = self._evaluation_primary_target(ev)

                    matching_candidate = None

                    for idx, candidate in enumerate(candidates):
                        if idx in used_ids:
                            continue

                        cand_activity = candidate.get("requested_activity")
                        cand_target = candidate.get("target")

                        if cand_activity != requested:
                            continue

                        # Dev cards have no target.
                        if requested == "buy_discovery_card":
                            matching_candidate = (idx, candidate)
                            break

                        # City/settlement should match target where possible.
                        if target is None or cand_target is None or int(cand_target) == int(target):
                            matching_candidate = (idx, candidate)
                            break

                    if matching_candidate is None:
                        continue

                    idx, candidate = matching_candidate
                    used_ids.add(idx)

                    enriched = dict(candidate)
                    enriched["rank_source"] = "142_way_action_evaluator"
                    enriched["rank_score"] = float(getattr(ev, "final_score", 0.0))

                    try:
                        enriched["rank_metadata"] = dict(ev.action.metadata)
                    except Exception:
                        enriched["rank_metadata"] = {}

                    ranked.append(enriched)

                # Append anything not matched by the evaluator.
                for idx, candidate in enumerate(candidates):
                    if idx not in used_ids:
                        enriched = dict(candidate)
                        enriched["rank_source"] = "fallback_after_142"
                        enriched["rank_score"] = 0.0
                        ranked.append(enriched)

                if ranked:
                    return ranked

            except Exception as exc:
                print(f"⚠️ Same-turn ranking failed; using fallback priority: {exc}")

        priority = {
            "upgrade_to_city": 0,
            "new_settlement": 1,
            "buy_discovery_card": 2,
        }

        ranked = sorted(
            candidates,
            key=lambda c: (
                priority.get(str(c.get("requested_activity")), 99),
                str(c.get("target")),
            )
        )

        for candidate in ranked:
            candidate["rank_source"] = "fallback_priority"
            candidate["rank_score"] = None

        return ranked

    def _execute_immediate_action_chain(
        self,
        player: Player,
        *,
        max_steps: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Same-turn action loop.

        Repeatedly:
            1. find all exactly executable activities
            2. rank them
            3. execute the best one
            4. refresh state
            5. repeat

        Stops when no exact executable activity remains.
        """
        chain_results: List[Dict[str, Any]] = []

        def _hand_debug() -> Any:
            try:
                return player.rcards_in_hand()[0]
            except Exception as exc:
                return f"hand_read_error: {exc}"

        print("SAME-TURN CHAIN ENTERED:", {
            "player": getattr(player, "id", "?"),
            "hand": _hand_debug(),
            "max_steps": max_steps,
        })

        for step_idx in range(int(max_steps)):
            candidates = self._build_immediate_executable_action_candidates(player)

            print("SAME-TURN CHAIN CANDIDATES:", {
                "step": step_idx + 1,
                "count": len(candidates),
                "items": [
                    {
                        "activity": c.get("requested_activity"),
                        "target": c.get("target"),
                        "guard_reason": (c.get("guard_info", {}) or {}).get("reason"),
                    }
                    for c in candidates
                ],
                "hand": _hand_debug(),
            })

            if not candidates:
                print("SAME-TURN CHAIN STOP:", {
                    "reason": "no_exact_executable_candidates",
                    "step": step_idx + 1,
                    "hand": _hand_debug(),
                })
                break

            ranked = self._rank_immediate_executable_action_candidates(
                player=player,
                candidates=candidates,
            )

            print("SAME-TURN CHAIN RANKED:", {
                "step": step_idx + 1,
                "count": len(ranked),
                "items": [
                    {
                        "activity": c.get("requested_activity"),
                        "target": c.get("target"),
                        "rank_source": c.get("rank_source"),
                        "rank_score": c.get("rank_score"),
                    }
                    for c in ranked
                ],
            })

            if not ranked:
                print("SAME-TURN CHAIN STOP:", {
                    "reason": "ranking_returned_empty",
                    "step": step_idx + 1,
                    "hand": _hand_debug(),
                })
                break

            chosen = ranked[0]
            requested_activity = str(chosen.get("requested_activity"))
            staged_plan = dict(chosen.get("staged_plan", {}) or {})
            guard_info = dict(chosen.get("guard_info", {}) or {})

            if not bool(guard_info.get("executable", False)):
                print("SAME-TURN CHAIN STOP:", {
                    "reason": "top_ranked_candidate_not_executable",
                    "step": step_idx + 1,
                    "activity": requested_activity,
                    "target": chosen.get("target"),
                    "guard_reason": guard_info.get("reason"),
                    "hand": _hand_debug(),
                })
                break

            print("SAME-TURN CHAIN EXECUTING:", {
                "step": step_idx + 1,
                "requested_activity": requested_activity,
                "target": chosen.get("target"),
                "rank_source": chosen.get("rank_source"),
                "rank_score": chosen.get("rank_score"),
                "hand_before": _hand_debug(),
            })

            result = self._execute_staged_plan(
                player,
                staged_plan,
                requested_activity=requested_activity,
            )

            print("SAME-TURN CHAIN RESULT:", {
                "step": step_idx + 1,
                "requested_activity": requested_activity,
                "success": result.get("success"),
                "actual_activity": result.get("actual_activity"),
                "reason": result.get("reason"),
                "hand_after": _hand_debug(),
            })

            chain_row = {
                "chain_step": step_idx + 1,
                "requested_activity": requested_activity,
                "target": chosen.get("target"),
                "rank_source": chosen.get("rank_source"),
                "rank_score": chosen.get("rank_score"),
                "guard_info": dict(guard_info),
                "staged_plan": dict(staged_plan),
                "result": dict(result),
            }

            chain_results.append(chain_row)

            if not bool(result.get("success", False)):
                print("SAME-TURN CHAIN STOP:", {
                    "reason": "execution_failed",
                    "step": step_idx + 1,
                    "result_reason": result.get("reason"),
                    "hand": _hand_debug(),
                })
                break

            self._refresh_all_outlooks()

            if hasattr(self.game, "update_strategy_dashboard"):
                try:
                    self.game.update_strategy_dashboard(player)
                except Exception:
                    pass

            if getattr(player, "victory_points", 0) >= 10 or getattr(player, "points", 0) >= 10:
                self.game.game_over = True
                print("SAME-TURN CHAIN STOP:", {
                    "reason": "player_reached_10_points",
                    "player": getattr(player, "id", "?"),
                })
                break

        return chain_results

    # ============================================================
    # Public API
    # ============================================================
    def run_one_step(self) -> None:
        """
        Compatibility helper:
        runs one complete fast-forward event by doing JUMP then PLAY.
        """
        if self.game.game_over:
            return

        self.jump_to_next_event()
        self.play_staged_event()

    def run_steps(self, num_steps: int = 1) -> None:
        """
        Compatibility helper:
        runs multiple complete JUMP+PLAY cycles.
        """
        for _ in range(num_steps):
            if self.game.game_over:
                break
            self.run_one_step()

    # ============================================================
    # Event timing
    # ============================================================
    def _get_event_times_for_player(
        self,
        player: Player,
        light_rows_so_far: Optional[List[Dict[str, Any]]] = None,
        precomputed_light_event_times: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Hybrid event-time estimator for one player.

        Policy:
        1) use precomputed LIGHT event times when provided
        2) otherwise compute LIGHT event times
        3) decide whether heavier reasoning is justified
        4) if yes, recompute only the focused activities with the HEAVY path
        5) merge heavy results back into the light result

        Returned keys:
            - new_settlement
            - upgrade_to_city
            - buy_discovery_card
        """
        def _safe_float(x: Any, default: float = 9999.0) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)

        def _rank_events(events: Dict[str, Any]) -> Tuple[Optional[str], float, Optional[str], float, float]:
            valid_items: List[Tuple[str, float]] = []
            for k, v in (events or {}).items():
                fv = _safe_float(v, 9999.0)
                if fv < 9999.0:
                    valid_items.append((k, fv))

            if not valid_items:
                return None, 9999.0, None, 9999.0, 9999.0

            valid_items.sort(key=lambda x: x[1])
            best_activity, best_time = valid_items[0]

            if len(valid_items) >= 2:
                second_best_activity, second_best_time = valid_items[1]
                gap = float(second_best_time) - float(best_time)
            else:
                second_best_activity, second_best_time, gap = None, 9999.0, 9999.0

            return (
                best_activity,
                float(best_time),
                second_best_activity,
                float(second_best_time),
                float(gap),
            )

        def _extract_breakdown_summary(breakdown_dict: Dict[str, Any], activity_key: str) -> Dict[str, Any]:
            info = dict((breakdown_dict or {}).get(activity_key, {}) or {})
            expl = dict(info.get("explanation", {}) or {})

            summary = {
                "activity": activity_key,
                "available": bool(info),
                "score": _safe_float(info.get("score", 9999.0), 9999.0),
                "heavy_mode": bool(info.get("heavy_mode", False)),
                "overflow_triggered": bool(expl.get("overflow_triggered", False)),
                "overflow_can_fund_within_horizon": bool(
                    expl.get("overflow_can_fund_within_horizon", False)
                ),
                "overflow_cap_bind_risk": bool(
                    expl.get("overflow_cap_bind_risk", False)
                ),
                "overflow_weak_off_resource_exists": bool(
                    expl.get("overflow_weak_off_resource_exists", False)
                ),
                "overflow_needed_trades": int(expl.get("overflow_needed_trades", 0) or 0),
                "overflow_off_resource_cards_to_buy": int(
                    expl.get("overflow_off_resource_cards_to_buy", 0) or 0
                ),
                "overflow_expected_dominant_cards_by_horizon": _safe_float(
                    expl.get("overflow_expected_dominant_cards_by_horizon", 0.0), 0.0
                ),
                "overflow_dominant_resource": expl.get("overflow_dominant_resource"),
                "extra_roads_needed": int(
                    expl.get("extra_roads_needed", info.get("extra_roads_needed", 0)) or 0
                ),
                "chosen_target": info.get("chosen_target", expl.get("chosen_target")),
                "settlement_target_type": info.get(
                    "settlement_target_type",
                    expl.get("settlement_target_type"),
                ),
                "chosen_upgrade": info.get("chosen_upgrade", expl.get("chosen_upgrade")),
                "light_prefilter_score": _safe_float(
                    info.get("light_prefilter_score", 9999.0), 9999.0
                ),
            }
            return summary

        # ------------------------------------------------------------
        # 1) Start with LIGHT
        # ------------------------------------------------------------
        if precomputed_light_event_times is not None:
            light_event_times = dict(precomputed_light_event_times)
        else:
            light_event_times = self._get_event_times_for_player_light(player)

        if not hasattr(self, "_last_strategy_breakdown"):
            self._last_strategy_breakdown = {}

        if player.id not in self._last_strategy_breakdown:
            self._last_strategy_breakdown[player.id] = {}

        light_breakdown = dict(self._last_strategy_breakdown.get(player.id, {}) or {})

        (
            best_activity_light,
            best_time_light,
            second_best_activity_light,
            second_best_time_light,
            light_gap_to_second,
        ) = _rank_events(light_event_times)

        best_light_summary = (
            _extract_breakdown_summary(light_breakdown, best_activity_light)
            if best_activity_light is not None
            else {}
        )
        settlement_light_summary = _extract_breakdown_summary(light_breakdown, "new_settlement")

        # ------------------------------------------------------------
        # 2) If heavy helpers are unavailable, LIGHT is the final answer
        # ------------------------------------------------------------
        if not hasattr(self, "_should_refine_with_heavy") or not hasattr(self, "_get_event_times_for_player_heavy"):
            self._last_strategy_breakdown[player.id]["__refine__"] = {
                "used_heavy": False,
                "reasons": [],
                "focus_activities": [],
                "light_event_times": dict(light_event_times),
                "heavy_event_times": {},
                "merged_event_times": dict(light_event_times),
                "light_reused": precomputed_light_event_times is not None,
                "best_activity_light": best_activity_light,
                "best_time_light": float(best_time_light),
                "second_best_activity_light": second_best_activity_light,
                "second_best_time_light": float(second_best_time_light),
                "light_gap_to_second": float(light_gap_to_second),
                "final_best_activity": best_activity_light,
                "final_best_time": float(best_time_light),
                "final_second_best_activity": second_best_activity_light,
                "final_second_best_time": float(second_best_time_light),
                "final_gap_to_second": float(light_gap_to_second),
                "best_activity_changed": False,
                "best_time_improvement": 0.0,
                "changed_activities": [],
                "activity_improvements": {},
                "best_activity_light_summary": best_light_summary,
                "settlement_light_summary": settlement_light_summary,
                "final_best_activity_summary": best_light_summary,
                "final_settlement_summary": settlement_light_summary,
            }
            return light_event_times

        refine_info = self._should_refine_with_heavy(
            player=player,
            light_event_times=light_event_times,
            light_rows_so_far=light_rows_so_far,
        )

        # v014 heavy-refinement policy:
        #
        # MARKOV_ENABLE_HEAVY_REFINEMENT = False
        #     -> never run heavy
        #
        # MARKOV_ENABLE_HEAVY_REFINEMENT = True
        # MARKOV_USE_ADAPTIVE_HEAVY = True
        #     -> use _should_refine_with_heavy(...) decision
        #
        # MARKOV_ENABLE_HEAVY_REFINEMENT = True
        # MARKOV_USE_ADAPTIVE_HEAVY = False
        #     -> force heavy refinement for the top N finite light activities
        #
        try:
            from core.constants import MARKOV_USE_ADAPTIVE_HEAVY
            use_adaptive_heavy = bool(MARKOV_USE_ADAPTIVE_HEAVY)
        except Exception:
            use_adaptive_heavy = True

        suppress_heavy_once = bool(getattr(self.game, "ff_suppress_heavy_once", False))

        if suppress_heavy_once:
            refine_info["refine"] = False
            refine_info["reasons"] = ["heavy_suppressed_during_enter_execution_preview"]
            refine_info["focus_activities"] = []

        elif not MARKOV_ENABLE_HEAVY_REFINEMENT:
            refine_info["refine"] = False
            refine_info["reasons"] = []
            refine_info["focus_activities"] = []

        elif not use_adaptive_heavy:
            # Force-heavy test mode, but keep it bounded.
            #
            # Instead of refining every finite activity, refine only the top N
            # light candidates. This lets HEAVY challenge the LIGHT winner without
            # making the first JUMP preparation unacceptably slow.
            FORCE_HEAVY_TOP_N = 1

            sorted_light_activities = sorted(
                [
                    (activity, _safe_float(value, 9999.0))
                    for activity, value in dict(light_event_times).items()
                    if _safe_float(value, 9999.0) < 9999.0
                ],
                key=lambda item: item[1],
            )

            forced_focus_activities = [
                activity
                for activity, value in sorted_light_activities[:FORCE_HEAVY_TOP_N]
            ]

            # Fallback: if all activities are 9999, still refine the light best.
            if not forced_focus_activities and best_activity_light is not None:
                forced_focus_activities = [best_activity_light]

            refine_info["refine"] = bool(forced_focus_activities)
            refine_info["reasons"] = [
                f"forced_heavy_top_{FORCE_HEAVY_TOP_N}_because_adaptive_heavy_is_false"
            ]
            refine_info["focus_activities"] = list(forced_focus_activities)
            refine_info["best_activity"] = best_activity_light
            refine_info["best_time"] = float(best_time_light)
            refine_info["second_best_activity"] = second_best_activity_light
            refine_info["second_best_time"] = float(second_best_time_light)
            refine_info["light_gap_to_second"] = float(light_gap_to_second)

        # ------------------------------------------------------------
        # 3) LIGHT only branch
        # ------------------------------------------------------------
        if not bool(refine_info.get("refine", False)):
            self._last_strategy_breakdown[player.id]["__refine__"] = {
                "used_heavy": False,
                "reasons": list(refine_info.get("reasons", [])),
                "focus_activities": list(refine_info.get("focus_activities", [])),
                "light_event_times": dict(light_event_times),
                "heavy_event_times": {},
                "merged_event_times": dict(light_event_times),
                "light_reused": precomputed_light_event_times is not None,
                "best_activity_light": refine_info.get("best_activity"),
                "best_time_light": float(refine_info.get("best_time", 9999.0)),
                "second_best_activity_light": refine_info.get("second_best_activity"),
                "second_best_time_light": float(refine_info.get("second_best_time", 9999.0)),
                "light_gap_to_second": float(light_gap_to_second),
                "final_best_activity": best_activity_light,
                "final_best_time": float(best_time_light),
                "final_second_best_activity": second_best_activity_light,
                "final_second_best_time": float(second_best_time_light),
                "final_gap_to_second": float(light_gap_to_second),
                "best_activity_changed": False,
                "best_time_improvement": 0.0,
                "changed_activities": [],
                "activity_improvements": {},
                "best_activity_light_summary": best_light_summary,
                "settlement_light_summary": settlement_light_summary,
                "final_best_activity_summary": best_light_summary,
                "final_settlement_summary": settlement_light_summary,
            }
            return light_event_times

        # ------------------------------------------------------------
        # 4) Run HEAVY only for the focused activities
        # ------------------------------------------------------------
        focus_activities = list(refine_info.get("focus_activities", [])) or [
            refine_info.get("best_activity")
        ]

        heavy_event_times = self._get_event_times_for_player_heavy(
            player=player,
            focus_activities=focus_activities,
        )

        # ------------------------------------------------------------
        # 5) Merge HEAVY results back into LIGHT
        # ------------------------------------------------------------
        merged_event_times = dict(light_event_times)

        for activity in focus_activities:
            if activity in heavy_event_times:
                heavy_val = _safe_float(heavy_event_times.get(activity), 9999.0)
                if heavy_val < 9999.0:
                    merged_event_times[activity] = heavy_val

        # Read updated breakdown AFTER heavy call, because heavy may have replaced
        # activity-level entries with heavy_mode=True and additional metadata.
        final_breakdown = dict(self._last_strategy_breakdown.get(player.id, {}) or {})

        (
            final_best_activity,
            final_best_time,
            final_second_best_activity,
            final_second_best_time,
            final_gap_to_second,
        ) = _rank_events(merged_event_times)

        changed_activities: List[str] = []
        activity_improvements: Dict[str, float] = {}

        all_keys = sorted(set(light_event_times.keys()) | set(merged_event_times.keys()))
        for activity in all_keys:
            light_val = _safe_float(light_event_times.get(activity), 9999.0)
            merged_val = _safe_float(merged_event_times.get(activity), 9999.0)

            if abs(light_val - merged_val) > 1e-9:
                changed_activities.append(activity)

            if light_val < 9999.0 and merged_val < 9999.0:
                activity_improvements[activity] = float(light_val - merged_val)

        best_activity_changed = bool(final_best_activity != best_activity_light)
        best_time_improvement = float(best_time_light - final_best_time)

        final_best_summary = (
            _extract_breakdown_summary(final_breakdown, final_best_activity)
            if final_best_activity is not None
            else {}
        )
        final_settlement_summary = _extract_breakdown_summary(final_breakdown, "new_settlement")

        # ------------------------------------------------------------
        # 6) Store refinement metadata for diagnostics
        # ------------------------------------------------------------
        self._last_strategy_breakdown[player.id]["__refine__"] = {
            "used_heavy": True,
            "reasons": list(refine_info.get("reasons", [])),
            "focus_activities": list(focus_activities),
            "best_activity_light": refine_info.get("best_activity"),
            "best_time_light": float(refine_info.get("best_time", 9999.0)),
            "second_best_activity_light": refine_info.get("second_best_activity"),
            "second_best_time_light": float(refine_info.get("second_best_time", 9999.0)),
            "light_gap_to_second": float(light_gap_to_second),
            "light_event_times": dict(light_event_times),
            "heavy_event_times": {k: heavy_event_times.get(k, 9999.0) for k in focus_activities},
            "merged_event_times": dict(merged_event_times),
            "light_reused": precomputed_light_event_times is not None,

            # new metadata
            "final_best_activity": final_best_activity,
            "final_best_time": float(final_best_time),
            "final_second_best_activity": final_second_best_activity,
            "final_second_best_time": float(final_second_best_time),
            "final_gap_to_second": float(final_gap_to_second),
            "best_activity_changed": bool(best_activity_changed),
            "best_time_improvement": float(best_time_improvement),
            "changed_activities": list(changed_activities),
            "activity_improvements": dict(activity_improvements),

            # compact summaries for debugging / prediction transparency
            "best_activity_light_summary": best_light_summary,
            "settlement_light_summary": settlement_light_summary,
            "final_best_activity_summary": final_best_summary,
            "final_settlement_summary": final_settlement_summary,
        }

        return merged_event_times

    # ============================================================
    # Time progression + resource consistency
    # ============================================================
    def _advance_markov_time(self, delta_rolls: float) -> None:
        """Advance the global fast-forward clock."""
        if not hasattr(self.game, "ff_elapsed_rolls"):
            self.game.ff_elapsed_rolls = 0.0
        if not hasattr(self.game, "ff_step_index"):
            self.game.ff_step_index = 0
        if not hasattr(self.game, "ff_last_delta"):
            self.game.ff_last_delta = 0.0
        if not hasattr(self.game, "ff_elapsed_rounds"):
            self.game.ff_elapsed_rounds = 0.0
        if not hasattr(self.game, "ff_player_time"):
            self.game.ff_player_time = {p.id: 0.0 for p in self.game.players}

        self.game.ff_elapsed_rolls += delta_rolls
        self.game.ff_last_delta = delta_rolls
        self.game.ff_step_index += 1

        num_players = max(len(self.game.players), 1)
        self.game.ff_elapsed_rounds = self.game.ff_elapsed_rolls / float(num_players)

        for player in self.game.players:
            self.game.ff_player_time[player.id] = (
                self.game.ff_player_time.get(player.id, 0.0) + delta_rolls
            )

    def _apply_elapsed_income_to_all_players(self, delta_rolls: float) -> None:
        """
        Convert skipped expected time into deterministic expected resource gain.

        Uses:
            expected_cards = (pips / 36) * delta_rolls
        with fractional carry-over stored in player.ff_resource_buffer.
        """
        if delta_rolls <= 0:
            return

        for player in self.game.players:
            pips = player.get_current_production_pips(self.game.board)

            for idx, rc in enumerate(RESOURCE_ORDER):
                if idx >= len(pips):
                    continue

                expected_gain = (float(pips[idx]) / 36.0) * delta_rolls
                player.ff_resource_buffer[rc] = player.ff_resource_buffer.get(rc, 0.0) + expected_gain

                whole_cards = int(player.ff_resource_buffer[rc])
                if whole_cards > 0:
                    player.add_rcard(rc, whole_cards)
                    player.ff_resource_buffer[rc] -= whole_cards

            player.update_trade_rates(self.game.board)
            if hasattr(self.game, "update_strategy_dashboard"):
                self.game.update_strategy_dashboard(player)

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"FastForwardEngine._apply_elapsed_income_to_all_players | "
                        f"player={player.id} delta_rolls={delta_rolls:.3f} "
                        f"pips={pips} hand={player.rcards_in_hand()[0]}\n"
                    )

    # ============================================================
    # Execution
    # ============================================================
    def _execute_activity(self, player: Player, requested_activity: str) -> Dict[str, Any]:
        """
        Try the requested activity, then apply settlement-failure policy.

        Settlement-failure policy:
        - if SETTLEMENT_AT_FAILURE is True:
            * do NOT fall back to city/dev-card
            * force a fresh re-estimate later with strategy=new_settlement
        - if SETTLEMENT_AT_FAILURE is False:
            * recompute the best alternative strategy from expected-hand probabilities
            * try the best non-settlement alternative first

        Returns a details dict for logging.
        """
        # ------------------------------------------------------------
        # Special handling for requested settlement
        # ------------------------------------------------------------
        if requested_activity == "new_settlement":
            settlement_details = self._try_execute_single_activity(player, "new_settlement")

            if settlement_details.get("success", False):
                settlement_details["requested_activity"] = requested_activity
                return settlement_details

            # --------------------------------------------------------
            # Policy A: keep pursuing settlement after failure
            # --------------------------------------------------------
            if SETTLEMENT_AT_FAILURE:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": None,
                    "reason": "Settlement attempt failed; policy forces fresh settlement re-estimate",
                    "retry_strategy": "new_settlement",
                    "reestimate_required": True,
                    "failed_details": dict(settlement_details),
                }

            # --------------------------------------------------------
            # Policy B: switch to the best alternative strategy
            # based on the current expected-hand / Markov estimate
            # --------------------------------------------------------
            recomputed_times = self._get_event_times_for_player(player)

            alternative_event_times = {
                k: v for k, v in recomputed_times.items()
                if k in ("upgrade_to_city", "buy_discovery_card")
            }

            ordered_alternatives = sorted(
                alternative_event_times.keys(),
                key=lambda k: float(alternative_event_times.get(k, 9999.0))
            )

            for activity in ordered_alternatives:
                details = self._try_execute_single_activity(player, activity)
                if details.get("success", False):
                    details["requested_activity"] = requested_activity
                    details["redirected_from_settlement"] = True
                    details["recomputed_event_times"] = dict(recomputed_times)
                    details["failed_settlement_details"] = dict(settlement_details)
                    return details

            return {
                "success": False,
                "requested_activity": requested_activity,
                "actual_activity": None,
                "reason": "Settlement failed and no alternative executable strategy was found",
                "recomputed_event_times": dict(recomputed_times),
                "failed_details": dict(settlement_details),
            }

        # ------------------------------------------------------------
        # Default fallback order for non-settlement requests
        # ------------------------------------------------------------
        fallback_order = {
            "upgrade_to_city": ["upgrade_to_city", "buy_discovery_card", "new_settlement"],
            "buy_discovery_card": ["buy_discovery_card", "upgrade_to_city", "new_settlement"],
            "buy_4_discovery_cards": ["buy_4_discovery_cards", "buy_discovery_card", "upgrade_to_city", "new_settlement"],
        }

        attempts = fallback_order.get(requested_activity, [requested_activity])

        for activity in attempts:
            details = self._try_execute_single_activity(player, activity)
            if details.get("success", False):
                details["requested_activity"] = requested_activity
                return details

        return {
            "success": False,
            "requested_activity": requested_activity,
            "actual_activity": None,
            "reason": "No legal executable activity found",
        }

    def _try_execute_single_activity(self, player: Player, activity: str) -> Dict[str, Any]:
        """Attempt exactly one activity."""
        outlook = self._ensure_outlook(player)

        if activity == "new_settlement":
            from collections import deque

            # Hard piece-limit guard:
            # in this codebase, settlements + cities together represent occupied building sites.
            if (len(getattr(player, "settlements", [])) + len(getattr(player, "cities", []))) >= 5:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "Player already has 5 occupied settlement/city sites",
                }

            board = self.game.board
            hand = player.rcards_in_hand()[0] if hasattr(player, "rcards_in_hand") else [0, 0, 0, 0, 0]
            current_settlements = list(getattr(player, "settlements", []))
            current_cities = list(getattr(player, "cities", []))
            base_ports = self.game.get_player_ports_dict(player)

            def _can_afford_settlement_bundle(extra_roads: int) -> bool:
                """
                Check whether the player can directly afford:
                    settlement + extra_roads * road

                Hand order:
                    [Wheat, Ore, Wood, Brick, Wool]
                """
                if self._ignore_resource_cards():
                    return True

                need_wheat = 1
                need_ore = 0
                need_wood = 1 + int(extra_roads)
                need_brick = 1 + int(extra_roads)
                need_wool = 1

                return (
                    hand[0] >= need_wheat and
                    hand[1] >= need_ore and
                    hand[2] >= need_wood and
                    hand[3] >= need_brick and
                    hand[4] >= need_wool
                )

            def _post_hand_after_settlement_bundle(extra_roads: int) -> List[int]:
                """
                Return the hypothetical hand AFTER paying for:
                    settlement + extra_roads * road

                If ff_ignore_resource_cards is enabled, keep using a zero hand.
                """
                if self._ignore_resource_cards():
                    return [0, 0, 0, 0, 0]

                post = list(hand)
                post[0] -= 1                       # wheat
                post[2] -= (1 + int(extra_roads)) # wood
                post[3] -= (1 + int(extra_roads)) # brick
                post[4] -= 1                       # wool
                return [max(0, int(x)) for x in post]

            def _road_obj(road_id: Tuple[int, int]):
                road_id = tuple(sorted(road_id))
                for r in board.roads:
                    if r and tuple(sorted(r.id)) == road_id:
                        return r
                return None


            def _is_valid_future_settlement_target(inter_id: int) -> bool:
                if inter_id in getattr(board, "INTERSECTION_IN_WATER", []):
                    return False

                inter = board.intersections[inter_id]
                if inter is None:
                    return False
                if getattr(inter, "occupied_tf", False):
                    return False

                occupied_vertices: List[int] = []
                for p in self.game.players:
                    occupied_vertices.extend(list(getattr(p, "settlements", [])))
                    occupied_vertices.extend(list(getattr(p, "cities", [])))

                # for existing_id in occupied_vertices:
                #     try:
                #         dist = board._distance_between_intersections(inter_id, existing_id)
                #     except Exception:
                #         dist = 999
                #     if dist <= 1:
                #         return False

                blocked = {int(inter_id)}

                inter_obj = board.intersections[int(inter_id)]
                for road_tuple in getattr(inter_obj, "three_roads", []):
                    if road_tuple and len(road_tuple) == 2:
                        a, b = int(road_tuple[0]), int(road_tuple[1])
                        blocked.add(a if b == int(inter_id) else b)

                for existing_id in occupied_vertices:
                    if int(existing_id) in blocked:
                        return False

                return True


            def _shortest_path_empty_roads_to_target(target_intersection: int):
                """
                Return:
                    (extra_roads_needed, ordered_empty_road_ids)

                Uses 0-1 BFS:
                - own occupied road   -> cost 0
                - empty road          -> cost 1
                - opponent road       -> blocked
                """
                start_vertices = set(player.settlements + player.cities)
                for road in getattr(player, "roads", []):
                    if isinstance(road, tuple) and len(road) == 2:
                        start_vertices.add(road[0])
                        start_vertices.add(road[1])

                if not start_vertices:
                    return (9999, [])

                INF = 10**9
                dist_map = {v.id: INF for v in board.intersections if v is not None}
                prev = {}
                dq = deque()

                for s in start_vertices:
                    if s in dist_map:
                        dist_map[s] = 0
                        dq.appendleft(s)

                while dq:
                    cur = dq.popleft()

                    if cur == target_intersection:
                        break

                    inter = board.intersections[cur]
                    if inter is None:
                        continue

                    cur_dist = dist_map[cur]

                    for road_tuple in getattr(inter, "three_roads", []):
                        road_id = tuple(sorted(road_tuple))
                        other = road_id[0] if road_id[1] == cur else road_id[1]

                        if other not in dist_map:
                            continue

                        road = _road_obj(road_id)
                        if road is not None and getattr(road, "occupied_tf", False):
                            if getattr(road, "color", None) != player.color:
                                continue
                            weight = 0
                        else:
                            weight = 1

                        nd = cur_dist + weight
                        if nd < dist_map[other]:
                            dist_map[other] = nd
                            prev[other] = (cur, road_id, weight)
                            if weight == 0:
                                dq.appendleft(other)
                            else:
                                dq.append(other)

                if dist_map.get(target_intersection, INF) >= INF:
                    return (9999, [])

                cur = target_intersection
                road_path_reversed = []
                while cur in prev:
                    parent, road_id, weight = prev[cur]
                    road_path_reversed.append((road_id, weight))
                    cur = parent

                road_path = list(reversed(road_path_reversed))
                empty_road_path = [road_id for road_id, weight in road_path if weight == 1]

                return (len(empty_road_path), empty_road_path)


            def _build_candidate_ports_for_target(inter_id: int) -> dict:
                if hasattr(outlook, "_build_candidate_ports"):
                    try:
                        return outlook._build_candidate_ports(inter_id, base_ports, "new_settlement")
                    except Exception:
                        pass
                return dict(base_ports)


            def _light_score_settlement_plan(plan: Dict[str, Any]) -> Tuple[float, float]:
                """
                LIGHT score of the position AFTER executing this settlement bundle.

                Returns:
                    (raw_score, tie_broken_score)
                """
                target = int(plan["target"])
                extra_roads = int(plan["extra_roads_needed"])

                vertices_after = current_settlements + current_cities + [target]
                ports_after = _build_candidate_ports_for_target(target)
                hand_after = _post_hand_after_settlement_bundle(extra_roads)

                raw_score = 9999.0

                if hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                    for strategy_name, road_need in [
                        ("settlement_0r", 0),
                        ("settlement_1r", 1),
                        ("settlement_2r", 2),
                        ("city", 0),
                        ("dev_card", 0),
                    ]:
                        cand = self.game.markov.get_expected_turns_fast_initial(
                            vertices=vertices_after,
                            hand=hand_after,
                            player_ports=ports_after,
                            strategy=strategy_name,
                            extra_roads_needed=road_need,
                        )
                        if cand is not None:
                            raw_score = min(raw_score, float(cand))

                elif hasattr(self.game.markov, "get_expected_turns_fast_initial_with_explanation"):
                    for strategy_name, road_need in [
                        ("settlement_0r", 0),
                        ("settlement_1r", 1),
                        ("settlement_2r", 2),
                        ("city", 0),
                        ("dev_card", 0),
                    ]:
                        out = self.game.markov.get_expected_turns_fast_initial_with_explanation(
                            vertices=vertices_after,
                            hand=hand_after,
                            player_ports=ports_after,
                            strategy=strategy_name,
                            extra_roads_needed=road_need,
                        )
                        raw_score = min(raw_score, float(out.get("score", 9999.0)))

                tie_broken_score = float(raw_score) + (0.001 * extra_roads) + (0.000001 * target)
                return float(raw_score), float(tie_broken_score)


            heavy_cache: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, int], ...]], Dict[str, float]] = {}


            def _canonical_ports(port_dict: Dict[str, Any]) -> Tuple[Tuple[str, int], ...]:
                items: List[Tuple[str, int]] = []
                for k, v in (port_dict or {}).items():
                    try:
                        items.append((str(k), int(v)))
                    except Exception:
                        continue
                items.sort(key=lambda x: x[0])
                return tuple(items)


            def _cached_heavy_full_times(
                vertices_in: List[int],
                hand_in: List[int],
                ports_in: Dict[str, Any],
            ) -> Dict[str, float]:
                key = (
                    tuple(int(x) for x in vertices_in),
                    tuple(int(x) for x in hand_in),
                    _canonical_ports(ports_in),
                )
                if key not in heavy_cache:
                    heavy_cache[key] = self.game.markov.get_expected_time_to_event(
                        vertices=list(vertices_in),
                        hand=list(hand_in),
                        player_ports=dict(ports_in),
                    )
                return heavy_cache[key]


            def _heavy_score_settlement_plan(plan: Dict[str, Any]) -> Tuple[float, float]:
                """
                HEAVY score of the position AFTER executing this settlement bundle.

                Returns:
                    (raw_score, tie_broken_score)
                """
                target = int(plan["target"])
                extra_roads = int(plan["extra_roads_needed"])

                vertices_after = current_settlements + current_cities + [target]
                ports_after = _build_candidate_ports_for_target(target)
                hand_after = _post_hand_after_settlement_bundle(extra_roads)

                if hasattr(self.game.markov, "get_expected_time_to_event"):
                    future_times = _cached_heavy_full_times(vertices_after, hand_after, ports_after)
                    raw_score = min(
                        float(future_times.get("settlement_0r", 9999.0)),
                        float(future_times.get("settlement_1r", 9999.0)),
                        float(future_times.get("settlement_2r", 9999.0)),
                        float(future_times.get("city", 9999.0)),
                        float(future_times.get("dev_card", 9999.0)),
                    )
                else:
                    raw_score, _ = _light_score_settlement_plan(plan)

                tie_broken_score = float(raw_score) + (0.001 * extra_roads) + (0.000001 * target)
                return float(raw_score), float(tie_broken_score)

            # --------------------------------------------------------
            # Build candidate settlement plans
            # --------------------------------------------------------
            candidate_plans: Dict[int, Dict[str, Any]] = {}
            structurally_possible_but_unaffordable: List[Dict[str, Any]] = []

            for inter in board.intersections:
                if inter is None:
                    continue

                inter_id = inter.id
                if not _is_valid_future_settlement_target(inter_id):
                    continue

                extra_roads_needed, road_path = _shortest_path_empty_roads_to_target(inter_id)

                if extra_roads_needed > 2:
                    continue

                plan = {
                    "target": inter_id,
                    "extra_roads_needed": extra_roads_needed,
                    "road_path": road_path,
                }

                if _can_afford_settlement_bundle(extra_roads_needed):
                    candidate_plans[inter_id] = plan
                else:
                    structurally_possible_but_unaffordable.append(plan)

            if not candidate_plans:
                if structurally_possible_but_unaffordable:
                    best_missing = min(
                        structurally_possible_but_unaffordable,
                        key=lambda x: (x["extra_roads_needed"], x["target"])
                    )
                    return {
                        "success": False,
                        "actual_activity": activity,
                        "reason": "Settlement exists structurally, but full settlement+road bundle is not executable now",
                        "needed_extra_roads": best_missing["extra_roads_needed"],
                        "chosen_tw": best_missing["target"],
                        "road_path": best_missing["road_path"],
                    }

                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "No viable new settlements",
                }

            # --------------------------------------------------------
            # LIGHT first: score all plans cheaply
            # --------------------------------------------------------
            scored_plans: List[Dict[str, Any]] = []

            for plan in candidate_plans.values():
                raw_light_score, light_score = _light_score_settlement_plan(plan)
                scored_plans.append({
                    "plan": plan,
                    "raw_light_score": raw_light_score,
                    "final_score": light_score,
                    "used_heavy": False,
                    "raw_heavy_score": None,
                })

            scored_plans.sort(key=lambda x: float(x["final_score"]))

            # --------------------------------------------------------
            # Focused HEAVY refinement:
            # refine only the top few light plans that are near-tied
            # --------------------------------------------------------
            # if scored_plans and hasattr(self.game.markov, "get_expected_time_to_event"):
            if False and scored_plans and hasattr(self.game.markov, "get_expected_time_to_event"):
                best_raw_light = float(scored_plans[0]["raw_light_score"])
                plans_to_refine: List[Dict[str, Any]] = []

                for entry in scored_plans:
                    if len(plans_to_refine) >= 3:
                        break
                    if float(entry["raw_light_score"]) <= best_raw_light + 0.50:
                        plans_to_refine.append(entry)

                for entry in plans_to_refine:
                    raw_heavy_score, heavy_score = _heavy_score_settlement_plan(entry["plan"])
                    if raw_heavy_score < 9999.0:
                        entry["raw_heavy_score"] = raw_heavy_score
                        entry["final_score"] = heavy_score
                        entry["used_heavy"] = True

            # --------------------------------------------------------
            # Choose the best FULL settlement plan directly
            # --------------------------------------------------------
            best_entry = None
            best_plan_score = float("inf")

            for entry in scored_plans:
                plan = entry["plan"]
                plan_score = float(entry["final_score"])

                if MG:
                    with open(FILENAME_MG, "a", encoding="utf-8") as f:
                        f.write(
                            f"FastForwardEngine._try_execute_single_activity | "
                            f"settlement_plan target={plan['target']} "
                            f"roads={plan['extra_roads_needed']} "
                            f"path={plan['road_path']} "
                            f"light_raw={float(entry['raw_light_score']):.6f} "
                            f"heavy_raw={entry['raw_heavy_score'] if entry['raw_heavy_score'] is not None else 'n/a'} "
                            f"used_heavy={entry['used_heavy']} "
                            f"final={plan_score:.6f}\n"
                        )

                if plan_score < best_plan_score:
                    best_plan_score = plan_score
                    best_entry = entry

            if best_entry is None:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "No candidate plan selected",
                }

            best_plan = best_entry["plan"]
            chosen_tw = int(best_plan["target"])
            road_path = list(best_plan["road_path"])
            extra_roads_needed = int(best_plan["extra_roads_needed"])

            built_roads: List[Tuple[int, int]] = []

            # --------------------------------------------------------
            # Execute required roads first
            # --------------------------------------------------------
            for road_id in road_path:
                success = player.build_structure("road", road_id, self.game.board)
                if not success:
                    return {
                        "success": False,
                        "actual_activity": activity,
                        "chosen_tw": chosen_tw,
                        "road_path": road_path,
                        "built_roads": built_roads,
                        "reason": "build_structure(road) failed before settlement",
                    }
                built_roads.append(road_id)

            # --------------------------------------------------------
            # Then execute the settlement itself
            # --------------------------------------------------------
            success = player.build_structure("settlement", chosen_tw, self.game.board)
            if not success:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "chosen_tw": chosen_tw,
                    "road_path": road_path,
                    "built_roads": built_roads,
                    "reason": "build_structure(settlement) failed after road build",
                }
            return {
                "success": True,
                "actual_activity": activity,
                "chosen_tw": chosen_tw,
                "extra_roads_needed": extra_roads_needed,
                "road_path": road_path,
                "built_roads": built_roads,
                "plan_score": best_plan_score,
                "plan_used_heavy": bool(best_entry.get("used_heavy", False)),
            }

        if activity == "upgrade_to_city":
            # Hard piece-limit guard: no 5th city
            if len(getattr(player, "cities", [])) >= 4:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "Player already has 4 cities",
                }

            viable = outlook.get_viable_city_upgrades()
            if not viable:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "No viable city upgrades",
                }

            chosen_tw = outlook.select_best_location("upgrade_to_city", viable)
            if chosen_tw == -1:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "No candidate selected",
                }

            success = player.build_structure("city", chosen_tw, self.game.board)
            if not success:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "chosen_tw": chosen_tw,
                    "reason": "build_structure(city) failed",
                }

            return {
                "success": True,
                "actual_activity": activity,
                "chosen_tw": chosen_tw,
            }

        if activity == "buy_discovery_card":
            if not self._can_buy_dev_card(player):
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "Cannot afford dev card or empty deck",
                }

            success = player.build_structure("development_card", -1, self.game.board)
            if not success:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "build_structure(development_card) failed",
                }

            picked = self._draw_dev_card_from_stack()
            if picked is None:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "Deck empty after cost spend",
                }

            self._grant_dev_card(player, picked)

            return {
                "success": True,
                "actual_activity": activity,
                "drawn_card": picked,
            }

        if activity == "buy_4_discovery_cards":
            drawn: List[str] = []

            for _ in range(4):
                if not self._can_buy_dev_card(player):
                    break

                success = player.build_structure("development_card", -1, self.game.board)
                if not success:
                    break

                picked = self._draw_dev_card_from_stack()
                if picked is None:
                    break

                self._grant_dev_card(player, picked)
                drawn.append(picked)

            if not drawn:
                return {
                    "success": False,
                    "actual_activity": activity,
                    "reason": "Could not buy any dev cards",
                }

            return {
                "success": True,
                "actual_activity": activity,
                "drawn_cards": drawn,
            }

        return {
            "success": False,
            "actual_activity": activity,
            "reason": "Unknown activity",
        }

    # ============================================================
    # Dev cards
    # ============================================================
    def _can_buy_dev_card(self, player: Player) -> bool:
        """
        For the fast-forward table, dev-card strategy should be considered
        whenever the deck is non-empty.

        Markov predicts the time to reach the dev-card purchase state, so
        we should NOT require current affordability here.
        """
        return bool(getattr(self.game, "dcards_stack", None))

    def _draw_dev_card_from_stack(self) -> Optional[str]:
        """Draw the top development card from the real stack."""
        if not getattr(self.game, "dcards_stack", None):
            return None
        if len(self.game.dcards_stack) == 0:
            return None
        return self.game.dcards_stack.pop(0)

    def _grant_dev_card(self, player: Player, card_name: str) -> None:
        """
        Grant a development card to the player without relying on Game.add_dcard().

        This keeps fast-forward self-contained.
        """
        player.development_cards.append(card_name)
        player.number_of_dcards += 1

        card_aliases = {
            "victory_points": "victory_point",
            "victory_point": "victory_point",
            "knight": "knight",
            "two_free_roads": "two_free_roads",
            "road_building": "two_free_roads",
            "year_of_plenty": "year_of_plenty",
            "monopoly": "monopoly",
        }

        normalized_card_name = card_aliases.get(str(card_name), str(card_name))

        updated_summary = False
        for row in getattr(player, "dcard_summary", []):
            if row and len(row) > 1 and row[0] == normalized_card_name:
                row[1] += 1  # bought_this_turn
                updated_summary = True
                break

        if not updated_summary and hasattr(player, "dcard_summary"):
            player.dcard_summary.append([normalized_card_name, 1, 0, 0])

        if normalized_card_name == "victory_point":
            player.victory_points += 1
            player.points = player.victory_points

        if hasattr(self.game, "update_strategy_dashboard"):
            self.game.update_strategy_dashboard(player)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"FastForwardEngine._grant_dev_card | "
                    f"player={player.id} card={card_name} vp={player.victory_points}\n"
                )

    # ============================================================
    # Helpers
    # ============================================================
    def _ensure_outlook(self, player: Player) -> PlayerOutlook:
        """Ensure player.outlook exists."""
        if not hasattr(player, "outlook") or player.outlook is None:
            player.outlook = PlayerOutlook(player, self.game)
        return player.outlook

    def _refresh_all_outlooks(self) -> None:
        """Refresh outlooks after state changes, but tolerate missing dependencies."""
        for player in self.game.players:
            outlook = self._ensure_outlook(player)
            try:
                outlook.refresh_all()
            except Exception as exc:
                if MG:
                    with open(FILENAME_MG, "a", encoding="utf-8") as f:
                        f.write(
                            f"FastForwardEngine._refresh_all_outlooks | "
                            f"player={player.id} refresh failed: {exc}\n"
                        )

    def _has_wood_brick_surplus(self, player: Player) -> bool:
        """
        Check if player currently has enough expected Wood + Brick for the
        'one extra road' settlement reachability shortcut.
        """
        hand = player.rcards_in_hand()[0] if hasattr(player, "rcards_in_hand") else [0] * 5
        return hand[2] >= 2 and hand[3] >= 2  # [Wheat, Ore, Wood, Brick, Wool]

    def _log_and_finalize(
        self,
        player: Player,
        requested_activity: str,
        delta_rolls: float,
        details: Dict[str, Any],
        all_event_times: Dict[str, float],
    ) -> None:
        """
        Log executed PLAY result.

        Important:
        - does NOT call game.advance_turn()
        - does NOT redraw the GUI
        - the next JUMP will decide the next displayed round/turn
        """
        actual_activity = details.get("actual_activity")
        success = details.get("success", False)

        pending = getattr(self.game, "ff_pending_event", {}) or {}

        log_row = {
            "round": self.game.round,
            "turn": self.game.turn,
            "ff_step_index": getattr(self.game, "ff_step_index", 0),
            "ff_last_delta": delta_rolls,
            "ff_elapsed_rolls": getattr(self.game, "ff_elapsed_rolls", 0.0),
            "ff_elapsed_rounds": getattr(self.game, "ff_elapsed_rounds", 0.0),

            "player_id": player.id,
            "player_color": player.color,
            "requested_activity": requested_activity,
            "actual_activity": actual_activity,
            "success": success,

            "event_times": dict(all_event_times),
            "details": dict(details),
            "hand_after": player.rcards_in_hand()[0],
            "vp_after": player.victory_points,

            # Prediction / visibility metadata
            "prediction_source_mode": pending.get("prediction_source_mode"),
            "prediction_used_heavy": bool(pending.get("prediction_used_heavy", False)),
            "prediction_refine_reasons": list(pending.get("prediction_refine_reasons", [])),
            "prediction_focus_activities": list(pending.get("prediction_focus_activities", [])),
            "prediction_changed_activities": list(pending.get("prediction_changed_activities", [])),
            "prediction_activity_improvements": dict(
                pending.get("prediction_activity_improvements", {})
            ),
            "prediction_overflow_summary": dict(
                pending.get("prediction_overflow_summary", {})
            ),
            "prediction_plan_summary": dict(
                pending.get("prediction_plan_summary", {})
            ),
            "prediction_chosen_explanation": dict(
                pending.get("prediction_chosen_explanation", {})
            ),

            # v014 Module 4: expected viable action ranking at PLAY time
            "expected_viable_actions": list(
                pending.get("expected_viable_actions", [])
            ),
            "expected_viable_codes": pending.get("expected_viable_codes", "-"),
            "expected_action_decision": dict(
                pending.get("expected_action_decision", {})
            ),
            "requested_activity_after_expected_ranking": pending.get(
                "requested_activity_after_expected_ranking"
            ),

            # Original staged plan from JUMP, plus final staged plan after ranking
            "staged_plan": dict(pending.get("staged_plan", {})),
            "staged_plan_after_expected_ranking": dict(
                pending.get("staged_plan_after_expected_ranking", {})
            ),
        }

        self.action_table.append(log_row)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"FastForwardEngine._log_and_finalize | "
                    f"player={player.id} requested={requested_activity} actual={actual_activity} "
                    f"success={success} delta={delta_rolls:.3f} "
                    f"source={log_row['prediction_source_mode']} "
                    f"used_heavy={log_row['prediction_used_heavy']} "
                    f"reasons={log_row['prediction_refine_reasons']} "
                    f"overflow={log_row['prediction_overflow_summary']} "
                    f"plan={log_row['prediction_plan_summary']} "
                    f"ff_rolls={getattr(self.game, 'ff_elapsed_rolls', 0.0):.3f} "
                    f"details={details}\n"
                )

    def _is_activity_executable_now(self, player: Player, requested_activity: str) -> Dict[str, Any]:
        """
        Dry-run executability guard for a staged PLAY event.

        Purpose:
        - prevent PLAY from trying to execute a prediction that is not actually
        executable from the current deterministic hand / board state
        - return structured diagnostics for logging and UI snapshotting

        Notes:
        - this is intentionally conservative
        - it does NOT mutate board state
        - it only answers: "is this staged activity executable now?"
        """
        normalized = str(requested_activity or "").strip().lower()
        if normalized == "settlement":
            normalized = "new_settlement"
        elif normalized == "city":
            normalized = "upgrade_to_city"
        elif normalized == "dev_card":
            normalized = "buy_discovery_card"

        hand = player.rcards_in_hand()[0] if hasattr(player, "rcards_in_hand") else [0, 0, 0, 0, 0]
        hand = list(hand) if hand is not None else [0, 0, 0, 0, 0]

        hand_dict = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        def _hand_can_pay(cost: Dict[str, int]) -> bool:
            if self._ignore_resource_cards():
                return True
            for res, need in cost.items():
                if int(hand_dict.get(res, 0)) < int(need):
                    return False
            return True

        if normalized == "new_settlement":
            occupied_sites = len(getattr(player, "settlements", [])) + len(getattr(player, "cities", []))
            if occupied_sites >= 5:
                return {
                    "executable": False,
                    "activity": normalized,
                    "reason": "Player already has 5 occupied settlement/city sites",
                    "current_hand": hand_dict,
                }

            extra_roads_needed, settlement_target = self._estimate_min_extra_roads_to_any_settlement(player)
            if settlement_target is None or extra_roads_needed > 2:
                return {
                    "executable": False,
                    "activity": normalized,
                    "reason": "No reachable settlement target within 2 extra roads",
                    "current_hand": hand_dict,
                    "extra_roads_needed": extra_roads_needed,
                    "chosen_target": settlement_target,
                }

            if hasattr(self.game, "markov") and hasattr(self.game.markov, "_game_order_target_cost"):
                required_cost = self.game.markov._game_order_target_cost(
                    "settlement",
                    extra_roads_needed=extra_roads_needed,
                )
            else:
                required_cost = {
                    "wheat": 1,
                    "wood": 1 + int(extra_roads_needed),
                    "brick": 1 + int(extra_roads_needed),
                    "wool": 1,
                }

            affordable = _hand_can_pay(required_cost)

            return {
                "executable": bool(affordable),
                "activity": normalized,
                "reason": "ok" if affordable else "Cannot afford settlement bundle at PLAY time",
                "current_hand": hand_dict,
                "required_cost": required_cost,
                "extra_roads_needed": int(extra_roads_needed),
                "chosen_target": settlement_target,
            }

        if normalized == "upgrade_to_city":
            if len(getattr(player, "cities", [])) >= 4:
                return {
                    "executable": False,
                    "activity": normalized,
                    "reason": "Player already has 4 cities",
                    "current_hand": hand_dict,
                }

            outlook = self._ensure_outlook(player)
            viable = outlook.get_viable_city_upgrades()
            if not viable:
                return {
                    "executable": False,
                    "activity": normalized,
                    "reason": "No viable city upgrades at PLAY time",
                    "current_hand": hand_dict,
                }

            required_cost = {"wheat": 2, "ore": 3}
            affordable = _hand_can_pay(required_cost)

            return {
                "executable": bool(affordable),
                "activity": normalized,
                "reason": "ok" if affordable else "Cannot afford city at PLAY time",
                "current_hand": hand_dict,
                "required_cost": required_cost,
                "candidate_upgrades": list(viable),
            }

        if normalized == "buy_discovery_card":
            deck_nonempty = bool(getattr(self.game, "dcards_stack", None))
            required_cost = {"wheat": 1, "ore": 1, "wool": 1}
            affordable = _hand_can_pay(required_cost)

            executable = bool(deck_nonempty and affordable)
            reason = "ok"
            if not deck_nonempty:
                reason = "Development card deck is empty"
            elif not affordable:
                reason = "Cannot afford development card at PLAY time"

            return {
                "executable": executable,
                "activity": normalized,
                "reason": reason,
                "current_hand": hand_dict,
                "required_cost": required_cost,
                "deck_nonempty": deck_nonempty,
            }

        if normalized == "buy_4_discovery_cards":
            deck_size = len(getattr(self.game, "dcards_stack", []) or [])
            required_cost_single = {"wheat": 1, "ore": 1, "wool": 1}

            if self._ignore_resource_cards():
                affordable_count = 4
            else:
                affordable_count = min(
                    hand_dict.get("wheat", 0),
                    hand_dict.get("ore", 0),
                    hand_dict.get("wool", 0),
                )

            buyable_now = min(int(deck_size), int(affordable_count))
            executable = buyable_now > 0

            return {
                "executable": executable,
                "activity": normalized,
                "reason": "ok" if executable else "Cannot buy any development cards at PLAY time",
                "current_hand": hand_dict,
                "required_cost_per_card": required_cost_single,
                "deck_size": int(deck_size),
                "buyable_now": int(buyable_now),
            }

        return {
            "executable": False,
            "activity": normalized,
            "reason": "Unknown staged activity",
            "current_hand": hand_dict,
        }

    def _build_staged_plan(self, player: Player, requested_activity: str) -> Dict[str, Any]:
        """
        Build an EXACT staged plan from the player's CURRENT post-JUMP state.

        Purpose:
        - Convert a predicted activity into a concrete executable plan
        - Keep PLAY aligned with the exact target/path chosen at JUMP time
        - Allow plans that are payable after bank/port trades, not only direct hand payment

        Returned plan types:
        - new_settlement
        - upgrade_to_city
        - buy_discovery_card
        - buy_4_discovery_cards
        """
        normalized = str(requested_activity or "").strip().lower()

        if normalized == "settlement":
            normalized = "new_settlement"
        elif normalized == "city":
            normalized = "upgrade_to_city"
        elif normalized == "dev_card":
            normalized = "buy_discovery_card"

        hand = player.rcards_in_hand()[0] if hasattr(player, "rcards_in_hand") else [0, 0, 0, 0, 0]
        hand = list(hand) if hand is not None else [0, 0, 0, 0, 0]

        hand_dict = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        def _normalize_ports(port_dict: Dict[str, Any]) -> Dict[str, int]:
            out: Dict[str, int] = {}

            for k, v in (port_dict or {}).items():
                key = str(k or "").strip().lower()
                try:
                    rate = int(v)
                except Exception:
                    continue

                if key in ("generic", "3:1"):
                    out["generic"] = min(out.get("generic", 4), rate)
                elif "wheat" in key:
                    out["wheat"] = min(out.get("wheat", 4), rate)
                elif "ore" in key:
                    out["ore"] = min(out.get("ore", 4), rate)
                elif "wood" in key or "lumber" in key:
                    out["wood"] = min(out.get("wood", 4), rate)
                elif "brick" in key:
                    out["brick"] = min(out.get("brick", 4), rate)
                elif "wool" in key or "sheep" in key:
                    out["wool"] = min(out.get("wool", 4), rate)

            return out

        def _trade_rate_for_resource(res_name: str, ports: Dict[str, int]) -> int:
            res_name = str(res_name).lower()
            if res_name in ports:
                return int(ports[res_name])
            if "generic" in ports:
                return int(ports["generic"])
            return 4

        def _normalize_cost(cost: Dict[str, Any]) -> Dict[str, int]:
            out = {"wheat": 0, "ore": 0, "wood": 0, "brick": 0, "wool": 0}
            aliases = {
                "lumber": "wood",
                "wood": "wood",
                "brick": "brick",
                "sheep": "wool",
                "wool": "wool",
                "wheat": "wheat",
                "ore": "ore",
            }

            for k, v in (cost or {}).items():
                key = aliases.get(str(k).strip().lower(), str(k).strip().lower())
                if key not in out:
                    continue
                try:
                    out[key] += int(v)
                except Exception:
                    pass

            return {k: v for k, v in out.items() if v > 0}

        def _hand_can_pay_direct(cost: Dict[str, int]) -> bool:
            if self._ignore_resource_cards():
                return True

            required = _normalize_cost(cost)
            return all(int(hand_dict.get(res, 0)) >= int(amt) for res, amt in required.items())

        def _can_pay_after_bank_trades(cost: Dict[str, int]) -> Dict[str, Any]:
            """
            Non-mutating affordability simulation.

            Mirrors the intent of _execute_staged_plan(...):
            allow bank/port trades before the actual build cost is paid.
            """
            required = _normalize_cost(cost)

            if self._ignore_resource_cards():
                return {
                    "payable": True,
                    "mode": "ignored_resources",
                    "required_cost": dict(required),
                    "current_hand": dict(hand_dict),
                    "simulated_trades": [],
                    "simulated_hand_after_trades": dict(hand_dict),
                    "ports": {},
                }

            hand_sim = dict(hand_dict)
            ports = _normalize_ports(self.game.get_player_ports_dict(player))
            simulated_trades: List[Dict[str, Any]] = []

            if all(hand_sim.get(res, 0) >= amt for res, amt in required.items()):
                return {
                    "payable": True,
                    "mode": "direct",
                    "required_cost": dict(required),
                    "current_hand": dict(hand_dict),
                    "simulated_trades": [],
                    "simulated_hand_after_trades": dict(hand_sim),
                    "ports": dict(ports),
                }

            for _ in range(20):
                deficits = {
                    res: max(0, int(required.get(res, 0)) - int(hand_sim.get(res, 0)))
                    for res in required
                }

                if all(v <= 0 for v in deficits.values()):
                    return {
                        "payable": True,
                        "mode": "after_trades",
                        "required_cost": dict(required),
                        "current_hand": dict(hand_dict),
                        "simulated_trades": list(simulated_trades),
                        "simulated_hand_after_trades": dict(hand_sim),
                        "ports": dict(ports),
                    }

                target_res = max(deficits.keys(), key=lambda r: deficits[r])
                if deficits[target_res] <= 0:
                    break

                best_give_res = None
                best_rate = None
                best_score = -999999

                # Prefer true surplus.
                for give_res in ("wheat", "ore", "wood", "brick", "wool"):
                    if give_res == target_res:
                        continue

                    rate = _trade_rate_for_resource(give_res, ports)
                    available = int(hand_sim.get(give_res, 0))
                    required_keep = int(required.get(give_res, 0))
                    surplus = available - required_keep

                    if surplus >= rate:
                        score = surplus - rate
                        if best_give_res is None or rate < best_rate or (
                            rate == best_rate and score > best_score
                        ):
                            best_give_res = give_res
                            best_rate = rate
                            best_score = score

                # Fallback: allow trading from any sufficiently large pile.
                if best_give_res is None:
                    candidates = []
                    for give_res in ("wheat", "ore", "wood", "brick", "wool"):
                        if give_res == target_res:
                            continue

                        rate = _trade_rate_for_resource(give_res, ports)
                        available = int(hand_sim.get(give_res, 0))

                        if available >= rate:
                            candidates.append((rate, -available, give_res))

                    if not candidates:
                        return {
                            "payable": False,
                            "mode": "not_enough_tradable_cards",
                            "required_cost": dict(required),
                            "current_hand": dict(hand_dict),
                            "simulated_trades": list(simulated_trades),
                            "simulated_hand_after_trades": dict(hand_sim),
                            "ports": dict(ports),
                        }

                    candidates.sort()
                    best_rate, _, best_give_res = candidates[0]

                hand_sim[best_give_res] = int(hand_sim.get(best_give_res, 0)) - int(best_rate)
                hand_sim[target_res] = int(hand_sim.get(target_res, 0)) + 1

                simulated_trades.append({
                    "give": best_give_res,
                    "give_amount": int(best_rate),
                    "receive": target_res,
                    "receive_amount": 1,
                })

            payable = all(hand_sim.get(res, 0) >= amt for res, amt in required.items())

            return {
                "payable": bool(payable),
                "mode": "after_trades" if payable else "max_trade_iterations_reached",
                "required_cost": dict(required),
                "current_hand": dict(hand_dict),
                "simulated_trades": list(simulated_trades),
                "simulated_hand_after_trades": dict(hand_sim),
                "ports": dict(ports),
            }

        # ========================
        # NEW SETTLEMENT PLAN
        # ========================
        if normalized == "new_settlement":
            from collections import deque

            if (len(getattr(player, "settlements", [])) + len(getattr(player, "cities", []))) >= 5:
                return {
                    "plan_type": "new_settlement",
                    "plan_available": False,
                    "reason": "Player already has 5 occupied settlement/city sites",
                    "current_hand": dict(hand_dict),
                }

            board = self.game.board
            outlook = self._ensure_outlook(player)
            current_settlements = list(getattr(player, "settlements", []))
            current_cities = list(getattr(player, "cities", []))
            base_ports = self.game.get_player_ports_dict(player)

            def _road_obj(road_id: Tuple[int, int]):
                rid = tuple(sorted(road_id))
                for r in board.roads:
                    if r and tuple(sorted(r.id)) == rid:
                        return r
                return None

            def _is_valid_future_settlement_target(inter_id: int) -> bool:
                inter_id = int(inter_id)

                if inter_id in getattr(board, "INTERSECTION_IN_WATER", []):
                    return False

                if not (0 <= inter_id < len(board.intersections)):
                    return False

                inter = board.intersections[inter_id]
                if inter is None:
                    return False

                if getattr(inter, "occupied_tf", False):
                    return False

                occupied_vertices: List[int] = []
                for p in self.game.players:
                    occupied_vertices.extend(list(getattr(p, "settlements", [])))
                    occupied_vertices.extend(list(getattr(p, "cities", [])))

                # Local Catan distance rule:
                # a candidate is blocked if it or any directly adjacent intersection
                # is already occupied.
                blocked = {inter_id}
                for road_tuple in getattr(inter, "three_roads", []):
                    if road_tuple and len(road_tuple) == 2:
                        a, b = int(road_tuple[0]), int(road_tuple[1])
                        blocked.add(a if b == inter_id else b)

                for existing_id in occupied_vertices:
                    if int(existing_id) in blocked:
                        return False

                return True

            def _shortest_path_empty_roads_to_target(target_intersection: int):
                """
                Return:
                    (extra_roads_needed, ordered_empty_road_ids)

                Uses 0-1 BFS:
                - own occupied road   -> cost 0
                - empty road          -> cost 1
                - opponent road       -> blocked
                """
                target_intersection = int(target_intersection)

                start_vertices = set(player.settlements + player.cities)
                for road in getattr(player, "roads", []):
                    if isinstance(road, tuple) and len(road) == 2:
                        start_vertices.add(int(road[0]))
                        start_vertices.add(int(road[1]))

                if not start_vertices:
                    return (9999, [])

                INF = 10**9
                dist_map = {v.id: INF for v in board.intersections if v is not None}
                prev = {}
                dq = deque()

                for s in start_vertices:
                    if s in dist_map:
                        dist_map[s] = 0
                        dq.appendleft(s)

                while dq:
                    cur = dq.popleft()

                    if cur == target_intersection:
                        break

                    inter = board.intersections[cur]
                    if inter is None:
                        continue

                    cur_dist = dist_map[cur]

                    for road_tuple in getattr(inter, "three_roads", []):
                        road_id = tuple(sorted(road_tuple))
                        other = road_id[0] if road_id[1] == cur else road_id[1]

                        if other not in dist_map:
                            continue

                        road = _road_obj(road_id)
                        if road is not None and getattr(road, "occupied_tf", False):
                            if getattr(road, "color", None) != player.color:
                                continue
                            weight = 0
                        else:
                            weight = 1

                        nd = cur_dist + weight
                        if nd < dist_map[other]:
                            dist_map[other] = nd
                            prev[other] = (cur, road_id, weight)
                            if weight == 0:
                                dq.appendleft(other)
                            else:
                                dq.append(other)

                if dist_map.get(target_intersection, INF) >= INF:
                    return (9999, [])

                cur = target_intersection
                road_path_reversed = []

                while cur in prev:
                    parent, road_id, weight = prev[cur]
                    road_path_reversed.append((road_id, weight))
                    cur = parent

                road_path = list(reversed(road_path_reversed))
                empty_road_path = [road_id for road_id, weight in road_path if weight == 1]

                return (len(empty_road_path), empty_road_path)

            def _build_candidate_ports_for_target(inter_id: int) -> dict:
                if hasattr(outlook, "_build_candidate_ports"):
                    try:
                        return outlook._build_candidate_ports(inter_id, base_ports, "new_settlement")
                    except Exception:
                        pass
                return dict(base_ports)

            def _required_settlement_cost(extra_roads: int) -> Dict[str, int]:
                if hasattr(self.game, "markov") and hasattr(self.game.markov, "_game_order_target_cost"):
                    return self.game.markov._game_order_target_cost(
                        "settlement",
                        extra_roads_needed=extra_roads,
                    )

                return {
                    "wheat": 1,
                    "wood": 1 + int(extra_roads),
                    "brick": 1 + int(extra_roads),
                    "wool": 1,
                }

            def _post_hand_after_settlement_bundle(extra_roads: int) -> List[int]:
                if self._ignore_resource_cards():
                    return [0, 0, 0, 0, 0]

                post = list(hand)
                post[0] -= 1
                post[2] -= (1 + int(extra_roads))
                post[3] -= (1 + int(extra_roads))
                post[4] -= 1

                return [max(0, int(x)) for x in post]

            def _score_plan(plan: Dict[str, Any]) -> Tuple[float, float]:
                target = int(plan["chosen_target"])
                extra_roads = int(plan["extra_roads_needed"])

                vertices_after = current_settlements + current_cities + [target]
                ports_after = _build_candidate_ports_for_target(target)
                hand_after = _post_hand_after_settlement_bundle(extra_roads)

                raw_score = 9999.0

                if hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                    for strategy_name, road_need in [
                        ("settlement_0r", 0),
                        ("settlement_1r", 1),
                        ("settlement_2r", 2),
                        ("city", 0),
                        ("dev_card", 0),
                    ]:
                        cand = self.game.markov.get_expected_turns_fast_initial(
                            vertices=vertices_after,
                            hand=hand_after,
                            player_ports=ports_after,
                            strategy=strategy_name,
                            extra_roads_needed=road_need,
                        )
                        if cand is not None:
                            raw_score = min(raw_score, float(cand))

                tie_broken_score = float(raw_score) + (0.001 * extra_roads) + (0.000001 * target)
                return float(raw_score), float(tie_broken_score)

            candidate_plans: List[Dict[str, Any]] = []
            blocked_candidates: List[Dict[str, Any]] = []

            for inter in board.intersections:
                if inter is None:
                    continue

                inter_id = int(inter.id)

                if not _is_valid_future_settlement_target(inter_id):
                    continue

                extra_roads_needed, road_path = _shortest_path_empty_roads_to_target(inter_id)

                if extra_roads_needed > 2:
                    continue

                required_cost = _required_settlement_cost(extra_roads_needed)
                pay_info = _can_pay_after_bank_trades(required_cost)

                plan = {
                    "plan_type": "new_settlement",
                    "plan_available": True,
                    "chosen_target": inter_id,
                    "extra_roads_needed": int(extra_roads_needed),
                    "road_path": list(road_path),
                    "required_cost": dict(_normalize_cost(required_cost)),
                    "current_hand": dict(hand_dict),
                    "player_ports": dict(base_ports),
                    "pay_info": dict(pay_info),
                    "pay_mode": str(pay_info.get("mode", "unknown")),
                    "payable_direct": bool(_hand_can_pay_direct(required_cost)),
                    "payable_after_trades": bool(pay_info.get("payable", False)),
                }

                if bool(pay_info.get("payable", False)):
                    candidate_plans.append(plan)
                else:
                    blocked_candidates.append(plan)

            if not candidate_plans:
                best_blocked = None
                if blocked_candidates:
                    best_blocked = min(
                        blocked_candidates,
                        key=lambda x: (
                            int(x["extra_roads_needed"]),
                            int(x["chosen_target"]),
                        )
                    )

                return {
                    "plan_type": "new_settlement",
                    "plan_available": False,
                    "reason": (
                        "Settlement exists structurally, but full settlement bundle is not payable now even after bank/port trades"
                        if best_blocked is not None
                        else "No viable new settlement plan"
                    ),
                    "current_hand": dict(hand_dict),
                    "best_blocked_plan": best_blocked,
                }

            scored = []
            for plan in candidate_plans:
                raw_score, final_score = _score_plan(plan)
                plan["future_raw_score_after_build"] = float(raw_score)
                plan["future_score_after_build"] = float(final_score)
                scored.append((float(final_score), float(raw_score), int(plan["extra_roads_needed"]), int(plan["chosen_target"]), plan))

            scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            best_plan = scored[0][-1]

            return best_plan

        # ========================
        # UPGRADE TO CITY
        # ========================
        if normalized == "upgrade_to_city":
            if len(getattr(player, "cities", [])) >= 4:
                return {
                    "plan_type": "upgrade_to_city",
                    "plan_available": False,
                    "reason": "Player already has 4 cities",
                    "current_hand": dict(hand_dict),
                }

            outlook = self._ensure_outlook(player)
            viable = outlook.get_viable_city_upgrades()

            if not viable:
                return {
                    "plan_type": "upgrade_to_city",
                    "plan_available": False,
                    "reason": "No viable city upgrades",
                    "current_hand": dict(hand_dict),
                }

            required_cost = {"wheat": 2, "ore": 3}
            pay_info = _can_pay_after_bank_trades(required_cost)

            if not bool(pay_info.get("payable", False)):
                return {
                    "plan_type": "upgrade_to_city",
                    "plan_available": False,
                    "reason": "City upgrade is not payable now even after bank/port trades",
                    "required_cost": dict(_normalize_cost(required_cost)),
                    "current_hand": dict(hand_dict),
                    "candidate_upgrades": list(viable),
                    "pay_info": dict(pay_info),
                }

            chosen_upgrade = outlook.select_best_location("upgrade_to_city", viable)

            if chosen_upgrade == -1:
                return {
                    "plan_type": "upgrade_to_city",
                    "plan_available": False,
                    "reason": "No city upgrade candidate selected",
                    "required_cost": dict(_normalize_cost(required_cost)),
                    "current_hand": dict(hand_dict),
                    "candidate_upgrades": list(viable),
                    "pay_info": dict(pay_info),
                }

            return {
                "plan_type": "upgrade_to_city",
                "plan_available": True,
                "chosen_upgrade": int(chosen_upgrade),
                "chosen_target": int(chosen_upgrade),
                "required_cost": dict(_normalize_cost(required_cost)),
                "current_hand": dict(hand_dict),
                "candidate_upgrades": list(viable),
                "pay_info": dict(pay_info),
                "pay_mode": str(pay_info.get("mode", "unknown")),
                "payable_direct": bool(_hand_can_pay_direct(required_cost)),
                "payable_after_trades": bool(pay_info.get("payable", False)),
            }

        # ========================
        # BUY DISCOVERY CARD
        # ========================
        if normalized == "buy_discovery_card":
            deck_nonempty = bool(getattr(self.game, "dcards_stack", None))
            required_cost = {"wheat": 1, "ore": 1, "wool": 1}
            pay_info = _can_pay_after_bank_trades(required_cost)

            return {
                "plan_type": "buy_discovery_card",
                "plan_available": bool(deck_nonempty and pay_info.get("payable", False)),
                "reason": (
                    "ok"
                    if bool(deck_nonempty and pay_info.get("payable", False))
                    else (
                        "Development card deck is empty"
                        if not deck_nonempty
                        else "Development card is not payable now even after bank/port trades"
                    )
                ),
                "required_cost": dict(_normalize_cost(required_cost)),
                "current_hand": dict(hand_dict),
                "deck_nonempty": deck_nonempty,
                "pay_info": dict(pay_info),
                "pay_mode": str(pay_info.get("mode", "unknown")),
                "payable_direct": bool(_hand_can_pay_direct(required_cost)),
                "payable_after_trades": bool(pay_info.get("payable", False)),
            }

        # ========================
        # BUY 4 DISCOVERY CARDS
        # ========================
        if normalized == "buy_4_discovery_cards":
            deck_size = len(getattr(self.game, "dcards_stack", []) or [])
            required_cost_per_card = {"wheat": 1, "ore": 1, "wool": 1}
            pay_info = _can_pay_after_bank_trades(required_cost_per_card)

            return {
                "plan_type": "buy_4_discovery_cards",
                "plan_available": bool(deck_size > 0 and pay_info.get("payable", False)),
                "reason": (
                    "ok"
                    if bool(deck_size > 0 and pay_info.get("payable", False))
                    else (
                        "Development card deck is empty"
                        if deck_size <= 0
                        else "No development card is payable now even after bank/port trades"
                    )
                ),
                "required_cost_per_card": dict(_normalize_cost(required_cost_per_card)),
                "current_hand": dict(hand_dict),
                "deck_size": int(deck_size),
                "pay_info": dict(pay_info),
                "pay_mode": str(pay_info.get("mode", "unknown")),
                "payable_direct": bool(_hand_can_pay_direct(required_cost_per_card)),
                "payable_after_trades": bool(pay_info.get("payable", False)),
            }

        return {
            "plan_type": normalized,
            "plan_available": False,
            "reason": "Unknown staged activity",
            "current_hand": dict(hand_dict),
        }

    def _is_staged_plan_executable_now(
        self,
        player: Player,
        staged_plan: Dict[str, Any],
        requested_activity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check whether the EXACT staged plan is still executable now.

        Important:
        - This guard should validate structure/legal feasibility.
        - It should NOT reject plans merely because the player cannot pay directly
        before bank/port trades.
        - Actual card mutation / trading is done later in _execute_staged_plan(...).
        """
        staged_plan = dict(staged_plan or {})

        if requested_activity is None:
            requested_activity = str(staged_plan.get("plan_type", "") or "").strip().lower()

        plan_type = str(staged_plan.get("plan_type", requested_activity) or requested_activity).strip().lower()

        if plan_type == "settlement":
            plan_type = "new_settlement"
        elif plan_type == "city":
            plan_type = "upgrade_to_city"
        elif plan_type == "dev_card":
            plan_type = "buy_discovery_card"

        if not staged_plan:
            return {
                "executable": False,
                "reason": "No staged plan available",
                "plan_type": plan_type,
                "guard_mode": "exact_plan",
                "staged_plan": {},
            }

        if not bool(staged_plan.get("plan_available", True)):
            return {
                "executable": False,
                "reason": str(staged_plan.get("reason", "Staged plan is unavailable")),
                "plan_type": plan_type,
                "guard_mode": "exact_plan",
                "staged_plan": dict(staged_plan),
            }

        # ------------------------------------------------------------
        # Resource helpers
        # ------------------------------------------------------------
        def _resource_name(rc) -> str:
            raw = str(getattr(rc, "value", getattr(rc, "name", rc))).lower()
            if raw == "lumber":
                return "wood"
            if raw == "sheep":
                return "wool"
            return raw

        def _current_hand_dict() -> Dict[str, int]:
            out = {"wheat": 0, "ore": 0, "wood": 0, "brick": 0, "wool": 0}

            for rc in ResourceCard:
                name = _resource_name(rc)
                if name in out:
                    try:
                        out[name] = int(player.rcards.get(rc, 0))
                    except Exception:
                        out[name] = 0

            return out

        def _normalize_cost(cost: Dict[str, Any]) -> Dict[str, int]:
            out = {"wheat": 0, "ore": 0, "wood": 0, "brick": 0, "wool": 0}

            aliases = {
                "lumber": "wood",
                "wood": "wood",
                "brick": "brick",
                "sheep": "wool",
                "wool": "wool",
                "wheat": "wheat",
                "ore": "ore",
            }

            for k, v in (cost or {}).items():
                key = aliases.get(str(k).strip().lower(), str(k).strip().lower())
                if key not in out:
                    continue
                try:
                    out[key] += int(v)
                except Exception:
                    pass

            return {k: v for k, v in out.items() if v > 0}

        def _normalize_ports(port_dict: Dict[str, Any]) -> Dict[str, int]:
            out: Dict[str, int] = {}

            for k, v in (port_dict or {}).items():
                key = str(k or "").strip().lower()
                try:
                    rate = int(v)
                except Exception:
                    continue

                if key in ("generic", "3:1"):
                    out["generic"] = min(out.get("generic", 4), rate)
                elif "wheat" in key:
                    out["wheat"] = min(out.get("wheat", 4), rate)
                elif "ore" in key:
                    out["ore"] = min(out.get("ore", 4), rate)
                elif "wood" in key or "lumber" in key:
                    out["wood"] = min(out.get("wood", 4), rate)
                elif "brick" in key:
                    out["brick"] = min(out.get("brick", 4), rate)
                elif "wool" in key or "sheep" in key:
                    out["wool"] = min(out.get("wool", 4), rate)

            return out

        def _trade_rate_for_resource(res_name: str, ports: Dict[str, int]) -> int:
            res_name = str(res_name).lower()
            if res_name in ports:
                return int(ports[res_name])
            if "generic" in ports:
                return int(ports["generic"])
            return 4

        def _hand_can_pay_direct(required_cost: Dict[str, int]) -> bool:
            if self._ignore_resource_cards():
                return True

            hand = _current_hand_dict()
            required_cost = _normalize_cost(required_cost)

            return all(
                int(hand.get(res, 0)) >= int(amt)
                for res, amt in required_cost.items()
            )

        def _can_pay_after_bank_trades(required_cost: Dict[str, int]) -> Dict[str, Any]:
            """
            Non-mutating affordability simulation.

            Returns whether the player can satisfy required_cost after legal bank/port
            trades using current hand and current ports.
            """
            required = _normalize_cost(required_cost)

            if self._ignore_resource_cards():
                return {
                    "payable": True,
                    "mode": "ignored_resources",
                    "required_cost": dict(required),
                    "current_hand": _current_hand_dict(),
                    "simulated_trades": [],
                }

            hand = _current_hand_dict()
            ports = _normalize_ports(self.game.get_player_ports_dict(player))
            simulated_trades: List[Dict[str, Any]] = []

            if all(hand.get(res, 0) >= amt for res, amt in required.items()):
                return {
                    "payable": True,
                    "mode": "direct",
                    "required_cost": dict(required),
                    "current_hand": _current_hand_dict(),
                    "simulated_hand_after_trades": dict(hand),
                    "simulated_trades": [],
                    "ports": dict(ports),
                }

            # Greedy simulation matching _execute_staged_plan's auto-trade intent.
            for _ in range(20):
                deficits = {
                    res: max(0, int(required.get(res, 0)) - int(hand.get(res, 0)))
                    for res in required
                }

                if all(v <= 0 for v in deficits.values()):
                    return {
                        "payable": True,
                        "mode": "after_trades",
                        "required_cost": dict(required),
                        "current_hand": _current_hand_dict(),
                        "simulated_hand_after_trades": dict(hand),
                        "simulated_trades": list(simulated_trades),
                        "ports": dict(ports),
                    }

                target_res = max(deficits.keys(), key=lambda r: deficits[r])
                if deficits[target_res] <= 0:
                    break

                best_give_res = None
                best_rate = None
                best_score = -999999

                # Prefer trading true surplus.
                for give_res in ("wheat", "ore", "wood", "brick", "wool"):
                    if give_res == target_res:
                        continue

                    rate = _trade_rate_for_resource(give_res, ports)
                    available = int(hand.get(give_res, 0))
                    required_keep = int(required.get(give_res, 0))
                    surplus = available - required_keep

                    if surplus >= rate:
                        score = surplus - rate
                        if best_give_res is None or rate < best_rate or (
                            rate == best_rate and score > best_score
                        ):
                            best_give_res = give_res
                            best_rate = rate
                            best_score = score

                # Fallback: allow any pile large enough, but avoid the target resource.
                if best_give_res is None:
                    candidates = []
                    for give_res in ("wheat", "ore", "wood", "brick", "wool"):
                        if give_res == target_res:
                            continue

                        rate = _trade_rate_for_resource(give_res, ports)
                        available = int(hand.get(give_res, 0))

                        if available >= rate:
                            candidates.append((rate, -available, give_res))

                    if not candidates:
                        return {
                            "payable": False,
                            "mode": "not_enough_tradable_cards",
                            "required_cost": dict(required),
                            "current_hand": _current_hand_dict(),
                            "simulated_hand_after_trades": dict(hand),
                            "simulated_trades": list(simulated_trades),
                            "ports": dict(ports),
                        }

                    candidates.sort()
                    best_rate, _, best_give_res = candidates[0]

                hand[best_give_res] = int(hand.get(best_give_res, 0)) - int(best_rate)
                hand[target_res] = int(hand.get(target_res, 0)) + 1

                simulated_trades.append({
                    "give": best_give_res,
                    "give_amount": int(best_rate),
                    "receive": target_res,
                    "receive_amount": 1,
                })

            payable = all(hand.get(res, 0) >= amt for res, amt in required.items())

            return {
                "payable": bool(payable),
                "mode": "after_trades" if payable else "max_trade_iterations_reached",
                "required_cost": dict(required),
                "current_hand": _current_hand_dict(),
                "simulated_hand_after_trades": dict(hand),
                "simulated_trades": list(simulated_trades),
                "ports": dict(ports),
            }

        def _cost_check(required_cost: Dict[str, int]) -> Dict[str, Any]:
            required_cost = _normalize_cost(required_cost)

            pay_info = _can_pay_after_bank_trades(required_cost)

            return {
                "ok": bool(pay_info.get("payable", False)),
                "required_cost": dict(required_cost),
                "current_hand": _current_hand_dict(),
                "pay_info": dict(pay_info),
            }

        # ------------------------------------------------------------
        # Board helpers
        # ------------------------------------------------------------
        def _road_obj(road_id: Tuple[int, int]):
            rid = tuple(sorted(road_id))
            for r in self.game.board.roads:
                if r and tuple(sorted(r.id)) == rid:
                    return r
            return None

        def _is_valid_future_settlement_target(inter_id: int) -> bool:
            board = self.game.board

            if inter_id in getattr(board, "INTERSECTION_IN_WATER", []):
                return False

            if not (0 <= int(inter_id) < len(board.intersections)):
                return False

            inter = board.intersections[int(inter_id)]
            if inter is None:
                return False

            if getattr(inter, "occupied_tf", False):
                return False

            occupied_vertices: List[int] = []
            for p in self.game.players:
                occupied_vertices.extend(list(getattr(p, "settlements", [])))
                occupied_vertices.extend(list(getattr(p, "cities", [])))

            for existing_id in occupied_vertices:
                try:
                    dist = board._distance_between_intersections(int(inter_id), int(existing_id))
                except Exception:
                    dist = 999

                if dist <= 1:
                    return False

            return True

        # ========================
        # NEW SETTLEMENT CHECK
        # ========================
        if plan_type == "new_settlement":
            occupied_sites = len(getattr(player, "settlements", [])) + len(getattr(player, "cities", []))
            if occupied_sites >= 5:
                return {
                    "executable": False,
                    "reason": "Player already has 5 occupied settlement/city sites",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            chosen_target = staged_plan.get("chosen_target")
            extra_roads_needed = int(staged_plan.get("extra_roads_needed", 9999))
            road_path = [tuple(r) for r in staged_plan.get("road_path", []) or []]

            if chosen_target is None:
                return {
                    "executable": False,
                    "reason": "Staged settlement plan has no target",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            chosen_target = int(chosen_target)

            if extra_roads_needed > 2:
                return {
                    "executable": False,
                    "reason": "Staged settlement plan requires more than 2 extra roads",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            if not _is_valid_future_settlement_target(chosen_target):
                return {
                    "executable": False,
                    "reason": "Exact staged settlement target is no longer legally buildable",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            for road_id in road_path:
                road = _road_obj(tuple(road_id))
                if road is not None and getattr(road, "occupied_tf", False):
                    if getattr(road, "color", None) != player.color:
                        return {
                            "executable": False,
                            "reason": "Exact staged road path is now blocked by another player",
                            "plan_type": plan_type,
                            "guard_mode": "exact_plan",
                            "staged_plan": dict(staged_plan),
                        }

            required_cost = dict(staged_plan.get("required_cost", {}) or {})
            if not required_cost:
                required_cost = {
                    "wheat": 1,
                    "wood": 1 + int(extra_roads_needed),
                    "brick": 1 + int(extra_roads_needed),
                    "wool": 1,
                }

            cost_info = _cost_check(required_cost)
            if not bool(cost_info.get("ok", False)):
                return {
                    "executable": False,
                    "reason": "Cannot afford exact staged settlement bundle at PLAY time, even after bank/port trades",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "required_cost": dict(cost_info.get("required_cost", {})),
                    "current_hand": dict(cost_info.get("current_hand", {})),
                    "pay_info": dict(cost_info.get("pay_info", {})),
                    "staged_plan": dict(staged_plan),
                }

            return {
                "executable": True,
                "reason": "ok",
                "plan_type": plan_type,
                "guard_mode": "exact_plan",
                "required_cost": dict(cost_info.get("required_cost", {})),
                "current_hand": dict(cost_info.get("current_hand", {})),
                "pay_info": dict(cost_info.get("pay_info", {})),
                "staged_plan": dict(staged_plan),
            }

        # ========================
        # UPGRADE TO CITY CHECK
        # ========================
        if plan_type == "upgrade_to_city":
            if len(getattr(player, "cities", [])) >= 4:
                return {
                    "executable": False,
                    "reason": "Player already has 4 cities",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            chosen_upgrade = staged_plan.get("chosen_upgrade", staged_plan.get("chosen_target"))
            if chosen_upgrade is None:
                return {
                    "executable": False,
                    "reason": "Staged city plan has no chosen upgrade",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            chosen_upgrade = int(chosen_upgrade)
            outlook = self._ensure_outlook(player)
            viable = outlook.get_viable_city_upgrades()

            if chosen_upgrade not in viable:
                return {
                    "executable": False,
                    "reason": "Exact staged city upgrade is no longer viable",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "candidate_upgrades": list(viable),
                    "staged_plan": dict(staged_plan),
                }

            required_cost = dict(staged_plan.get("required_cost", {"wheat": 2, "ore": 3}) or {})
            cost_info = _cost_check(required_cost)

            if not bool(cost_info.get("ok", False)):
                return {
                    "executable": False,
                    "reason": "Cannot afford exact staged city at PLAY time, even after bank/port trades",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "required_cost": dict(cost_info.get("required_cost", {})),
                    "current_hand": dict(cost_info.get("current_hand", {})),
                    "pay_info": dict(cost_info.get("pay_info", {})),
                    "staged_plan": dict(staged_plan),
                }

            return {
                "executable": True,
                "reason": "ok",
                "plan_type": plan_type,
                "guard_mode": "exact_plan",
                "required_cost": dict(cost_info.get("required_cost", {})),
                "current_hand": dict(cost_info.get("current_hand", {})),
                "pay_info": dict(cost_info.get("pay_info", {})),
                "staged_plan": dict(staged_plan),
            }

        # ========================
        # BUY DISCOVERY CARD
        # ========================
        if plan_type == "buy_discovery_card":
            deck_nonempty = bool(getattr(self.game, "dcards_stack", None))
            if not deck_nonempty:
                return {
                    "executable": False,
                    "reason": "Development card deck is empty",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "staged_plan": dict(staged_plan),
                }

            required_cost = dict(staged_plan.get("required_cost", {"wheat": 1, "ore": 1, "wool": 1}) or {})
            cost_info = _cost_check(required_cost)

            if not bool(cost_info.get("ok", False)):
                return {
                    "executable": False,
                    "reason": "Cannot afford exact staged development card at PLAY time, even after bank/port trades",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "required_cost": dict(cost_info.get("required_cost", {})),
                    "current_hand": dict(cost_info.get("current_hand", {})),
                    "pay_info": dict(cost_info.get("pay_info", {})),
                    "staged_plan": dict(staged_plan),
                }

            return {
                "executable": True,
                "reason": "ok",
                "plan_type": plan_type,
                "guard_mode": "exact_plan",
                "required_cost": dict(cost_info.get("required_cost", {})),
                "current_hand": dict(cost_info.get("current_hand", {})),
                "pay_info": dict(cost_info.get("pay_info", {})),
                "staged_plan": dict(staged_plan),
            }

        # ========================
        # BUY 4 DISCOVERY CARDS
        # ========================
        if plan_type == "buy_4_discovery_cards":
            deck_size = len(getattr(self.game, "dcards_stack", []) or [])
            if deck_size <= 0:
                return {
                    "executable": False,
                    "reason": "Development card deck is empty",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "deck_size": int(deck_size),
                    "staged_plan": dict(staged_plan),
                }

            required_cost_per_card = dict(
                staged_plan.get("required_cost_per_card", {"wheat": 1, "ore": 1, "wool": 1}) or {}
            )

            # Guard allows PLAY if at least one card is payable after trades.
            # The exact executor will buy up to 4.
            cost_info = _cost_check(required_cost_per_card)

            if not bool(cost_info.get("ok", False)):
                return {
                    "executable": False,
                    "reason": "Cannot afford any staged development card at PLAY time, even after bank/port trades",
                    "plan_type": plan_type,
                    "guard_mode": "exact_plan",
                    "required_cost_per_card": dict(_normalize_cost(required_cost_per_card)),
                    "current_hand": dict(cost_info.get("current_hand", {})),
                    "pay_info": dict(cost_info.get("pay_info", {})),
                    "deck_size": int(deck_size),
                    "staged_plan": dict(staged_plan),
                }

            return {
                "executable": True,
                "reason": "ok",
                "plan_type": plan_type,
                "guard_mode": "exact_plan",
                "required_cost_per_card": dict(_normalize_cost(required_cost_per_card)),
                "current_hand": dict(cost_info.get("current_hand", {})),
                "pay_info": dict(cost_info.get("pay_info", {})),
                "deck_size": int(deck_size),
                "staged_plan": dict(staged_plan),
            }

        return {
            "executable": False,
            "reason": f"Unknown staged plan type: {plan_type}",
            "plan_type": plan_type,
            "guard_mode": "exact_plan",
            "staged_plan": dict(staged_plan),
        }

    def _execute_staged_plan(
        self,
        player: Player,
        staged_plan: Dict[str, Any],
        requested_activity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the EXACT staged plan chosen at JUMP time.

        This version adds automatic bank/port trading before executing the staged
        build, so Markov-predicted trade-reachable actions can actually be paid.
        """
        if requested_activity is None:
            requested_activity = str(staged_plan.get("plan_type", "") or "").strip().lower()

        if not staged_plan:
            return self._execute_activity(player, requested_activity)

        plan_type = str(staged_plan.get("plan_type", requested_activity) or requested_activity).strip().lower()
        if plan_type == "settlement":
            plan_type = "new_settlement"
        elif plan_type == "city":
            plan_type = "upgrade_to_city"
        elif plan_type == "dev_card":
            plan_type = "buy_discovery_card"

        # ------------------------------------------------------------
        # Resource helpers
        # ------------------------------------------------------------
        def _rc_for_name(name: str):
            n = str(name or "").strip().lower()
            aliases = {
                "wood": "wood",
                "lumber": "wood",
                "brick": "brick",
                "wool": "wool",
                "sheep": "wool",
                "wheat": "wheat",
                "ore": "ore",
            }

            wanted = aliases.get(n, n)

            for rc in ResourceCard:
                rc_name = str(getattr(rc, "name", "")).lower()
                rc_value = str(getattr(rc, "value", "")).lower()
                if rc_name == wanted or rc_value == wanted:
                    return rc

            return None

        def _resource_name(rc) -> str:
            raw = str(getattr(rc, "value", getattr(rc, "name", rc))).lower()
            if raw == "lumber":
                return "wood"
            if raw == "sheep":
                return "wool"
            return raw

        def _current_hand_dict() -> Dict[str, int]:
            out = {"wheat": 0, "ore": 0, "wood": 0, "brick": 0, "wool": 0}
            for rc in ResourceCard:
                name = _resource_name(rc)
                if name in out:
                    out[name] = int(player.rcards.get(rc, 0))
            return out

        def _normalize_ports(port_dict: Dict[str, Any]) -> Dict[str, int]:
            out: Dict[str, int] = {}

            for k, v in (port_dict or {}).items():
                key = str(k or "").strip().lower()
                try:
                    rate = int(v)
                except Exception:
                    continue

                if key in ("generic", "3:1"):
                    out["generic"] = min(out.get("generic", 4), rate)
                elif "wheat" in key:
                    out["wheat"] = min(out.get("wheat", 4), rate)
                elif "ore" in key:
                    out["ore"] = min(out.get("ore", 4), rate)
                elif "wood" in key or "lumber" in key:
                    out["wood"] = min(out.get("wood", 4), rate)
                elif "brick" in key:
                    out["brick"] = min(out.get("brick", 4), rate)
                elif "wool" in key or "sheep" in key:
                    out["wool"] = min(out.get("wool", 4), rate)

            return out

        def _trade_rate_for_resource(res_name: str, ports: Dict[str, int]) -> int:
            res_name = str(res_name).lower()
            if res_name in ports:
                return int(ports[res_name])
            if "generic" in ports:
                return int(ports["generic"])
            return 4

        def _required_cost_for_plan(ptype: str) -> Dict[str, int]:
            staged_cost = staged_plan.get("required_cost")
            if isinstance(staged_cost, dict) and staged_cost:
                return dict(staged_cost)

            if ptype == "new_settlement":
                road_path = list(staged_plan.get("road_path", []) or [])
                extra_roads = int(staged_plan.get("extra_roads_needed", len(road_path)) or 0)
                return {
                    "wheat": 1,
                    "ore": 0,
                    "wood": 1 + extra_roads,
                    "brick": 1 + extra_roads,
                    "wool": 1,
                }

            if ptype == "upgrade_to_city":
                return {
                    "wheat": 2,
                    "ore": 3,
                    "wood": 0,
                    "brick": 0,
                    "wool": 0,
                }

            if ptype == "buy_discovery_card":
                return {
                    "wheat": 1,
                    "ore": 1,
                    "wood": 0,
                    "brick": 0,
                    "wool": 1,
                }

            if ptype == "buy_4_discovery_cards":
                return {
                    "wheat": 4,
                    "ore": 4,
                    "wood": 0,
                    "brick": 0,
                    "wool": 4,
                }

            return {}

        def _can_pay_direct(required: Dict[str, int]) -> bool:
            if self._ignore_resource_cards():
                return True
            hand = _current_hand_dict()
            return all(hand.get(res, 0) >= int(amt) for res, amt in required.items())

        def _auto_trade_to_pay(required: Dict[str, int]) -> Dict[str, Any]:
            """
            Greedily perform bank/port trades so the player can pay required.

            This mutates player.rcards by removing traded-away cards and adding
            received cards. The actual build cost is still paid later by
            player.build_structure(...).
            """
            if self._ignore_resource_cards():
                return {
                    "success": True,
                    "trades": [],
                    "reason": "ff_ignore_resource_cards=True",
                    "hand_after": _current_hand_dict(),
                }

            required = {str(k).lower(): int(v) for k, v in (required or {}).items() if int(v) > 0}
            if not required:
                return {
                    "success": True,
                    "trades": [],
                    "reason": "no required cost",
                    "hand_after": _current_hand_dict(),
                }

            if _can_pay_direct(required):
                return {
                    "success": True,
                    "trades": [],
                    "reason": "already affordable",
                    "hand_after": _current_hand_dict(),
                }

            ports = _normalize_ports(self.game.get_player_ports_dict(player))
            trades: List[Dict[str, Any]] = []

            # Conservative bound: enough for all normal Catan costs, avoids loops.
            for _ in range(20):
                hand = _current_hand_dict()
                deficits = {
                    res: max(0, int(required.get(res, 0)) - int(hand.get(res, 0)))
                    for res in required
                }

                if all(v <= 0 for v in deficits.values()):
                    return {
                        "success": True,
                        "trades": trades,
                        "reason": "affordable after trades",
                        "hand_after": _current_hand_dict(),
                    }

                # Pick the currently missing resource with largest deficit.
                target_res = max(deficits.keys(), key=lambda r: deficits[r])
                if deficits[target_res] <= 0:
                    break

                best_give_res = None
                best_rate = None
                best_surplus_after_trade = -999999

                for give_res in ("wheat", "ore", "wood", "brick", "wool"):
                    if give_res == target_res:
                        continue

                    rate = _trade_rate_for_resource(give_res, ports)
                    available = int(hand.get(give_res, 0))
                    required_keep = int(required.get(give_res, 0))

                    # Only trade true surplus whenever possible.
                    surplus = available - required_keep
                    if surplus >= rate:
                        score = surplus - rate
                        if best_give_res is None or rate < best_rate or (
                            rate == best_rate and score > best_surplus_after_trade
                        ):
                            best_give_res = give_res
                            best_rate = rate
                            best_surplus_after_trade = score

                # Fallback: if no true surplus exists, allow trading from the largest pile
                # only if it still does not make direct affordability worse for that resource.
                if best_give_res is None:
                    candidates = []
                    for give_res in ("wheat", "ore", "wood", "brick", "wool"):
                        if give_res == target_res:
                            continue
                        rate = _trade_rate_for_resource(give_res, ports)
                        available = int(hand.get(give_res, 0))
                        if available >= rate and available - rate >= 0:
                            candidates.append((rate, -available, give_res))

                    if not candidates:
                        return {
                            "success": False,
                            "trades": trades,
                            "reason": "not enough tradable surplus",
                            "required": dict(required),
                            "hand_after": _current_hand_dict(),
                            "ports": dict(ports),
                        }

                    candidates.sort()
                    best_rate, _, best_give_res = candidates[0]

                give_rc = _rc_for_name(best_give_res)
                target_rc = _rc_for_name(target_res)

                if give_rc is None or target_rc is None:
                    return {
                        "success": False,
                        "trades": trades,
                        "reason": "resource enum lookup failed",
                        "give_res": best_give_res,
                        "target_res": target_res,
                        "required": dict(required),
                        "hand_after": _current_hand_dict(),
                    }

                if not player.remove_rcard(give_rc, int(best_rate)):
                    return {
                        "success": False,
                        "trades": trades,
                        "reason": "remove_rcard failed during auto trade",
                        "give_res": best_give_res,
                        "rate": int(best_rate),
                        "required": dict(required),
                        "hand_after": _current_hand_dict(),
                    }

                player.add_rcard(target_rc, 1)

                trade = {
                    "give": best_give_res,
                    "give_amount": int(best_rate),
                    "receive": target_res,
                    "receive_amount": 1,
                }
                trades.append(trade)

            return {
                "success": _can_pay_direct(required),
                "trades": trades,
                "reason": "max trade iterations reached",
                "required": dict(required),
                "hand_after": _current_hand_dict(),
                "ports": dict(ports),
            }

        def _road_obj(road_id: Tuple[int, int]):
            rid = tuple(sorted(road_id))
            for r in self.game.board.roads:
                if r and tuple(sorted(r.id)) == rid:
                    return r
            return None

        # ------------------------------------------------------------
        # Auto-trade before exact execution
        # ------------------------------------------------------------
        required_cost = _required_cost_for_plan(plan_type)
        trade_info = _auto_trade_to_pay(required_cost)

        if not bool(trade_info.get("success", False)):
            return {
                "success": False,
                "requested_activity": requested_activity,
                "actual_activity": plan_type,
                "reason": "Could not auto-trade to pay staged plan",
                "trade_info": dict(trade_info),
                "required_cost": dict(required_cost),
                "current_hand": _current_hand_dict(),
                "staged_plan": dict(staged_plan),
                "reestimate_required": bool(plan_type == "new_settlement" and SETTLEMENT_AT_FAILURE),
                "retry_strategy": "new_settlement" if plan_type == "new_settlement" and SETTLEMENT_AT_FAILURE else None,
            }

        # ========================
        # EXECUTE NEW SETTLEMENT
        # ========================
        if plan_type == "new_settlement":
            chosen_target = staged_plan.get("chosen_target")
            if chosen_target is None:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "reason": "Exact staged settlement plan has no target",
                    "staged_plan": dict(staged_plan),
                    "trade_info": dict(trade_info),
                    "reestimate_required": bool(SETTLEMENT_AT_FAILURE),
                    "retry_strategy": "new_settlement" if SETTLEMENT_AT_FAILURE else None,
                }

            chosen_target = int(chosen_target)
            road_path = [tuple(r) for r in staged_plan.get("road_path", [])]
            extra_roads_needed = int(staged_plan.get("extra_roads_needed", len(road_path)))

            built_roads: List[Tuple[int, int]] = []

            for road_id in road_path:
                road_id = tuple(sorted(road_id))
                road = _road_obj(road_id)

                if road is not None and getattr(road, "occupied_tf", False):
                    if getattr(road, "color", None) == player.color:
                        built_roads.append(road_id)
                        continue

                    return {
                        "success": False,
                        "requested_activity": requested_activity,
                        "actual_activity": plan_type,
                        "chosen_tw": chosen_target,
                        "road_path": road_path,
                        "built_roads": built_roads,
                        "reason": "Staged road is now occupied by another player",
                        "trade_info": dict(trade_info),
                        "staged_plan": dict(staged_plan),
                        "reestimate_required": bool(SETTLEMENT_AT_FAILURE),
                        "retry_strategy": "new_settlement" if SETTLEMENT_AT_FAILURE else None,
                    }

                success = player.build_structure("road", road_id, self.game.board)
                if not success:
                    return {
                        "success": False,
                        "requested_activity": requested_activity,
                        "actual_activity": plan_type,
                        "chosen_tw": chosen_target,
                        "road_path": road_path,
                        "built_roads": built_roads,
                        "reason": "build_structure(road) failed during staged settlement plan",
                        "trade_info": dict(trade_info),
                        "staged_plan": dict(staged_plan),
                        "reestimate_required": bool(SETTLEMENT_AT_FAILURE),
                        "retry_strategy": "new_settlement" if SETTLEMENT_AT_FAILURE else None,
                    }

                built_roads.append(road_id)

            success = player.build_structure("settlement", chosen_target, self.game.board)
            if not success:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "chosen_tw": chosen_target,
                    "road_path": road_path,
                    "built_roads": built_roads,
                    "reason": "build_structure(settlement) failed during exact staged plan",
                    "trade_info": dict(trade_info),
                    "staged_plan": dict(staged_plan),
                    "reestimate_required": bool(SETTLEMENT_AT_FAILURE),
                    "retry_strategy": "new_settlement" if SETTLEMENT_AT_FAILURE else None,
                }

            result = {
                "success": True,
                "requested_activity": requested_activity,
                "actual_activity": "new_settlement",
                "chosen_tw": chosen_target,
                "extra_roads_needed": len(road_path),
                "road_path": list(road_path),
                "built_roads": list(built_roads),
                "trade_info": dict(trade_info),
                "staged_plan": dict(staged_plan),
                "hand_after": _current_hand_dict(),
            }

            print("PLAY EXECUTION RESULT:", {
                "player_id": getattr(player, "id", "?"),
                "player_color": getattr(player, "color", "?"),                
                "requested_activity": requested_activity,
                "plan_type": plan_type,
                "success": result.get("success"),
                "actual_activity": result.get("actual_activity"),
                "chosen_tw": result.get("chosen_tw"),
                "extra_roads_needed": result.get("extra_roads_needed"),
                "built_roads": result.get("built_roads"),
                "player_settlements_after": list(getattr(player, "settlements", [])),
                "player_cities_after": list(getattr(player, "cities", [])),
                "hand_after": result.get("hand_after"),
            })

            return result       

        # ========================
        # EXECUTE CITY
        # ========================
        if plan_type == "upgrade_to_city":
            chosen_target = staged_plan.get("chosen_target", staged_plan.get("chosen_upgrade"))
            if chosen_target is None:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "reason": "Exact staged city plan has no target",
                    "trade_info": dict(trade_info),
                    "staged_plan": dict(staged_plan),
                }

            chosen_target = int(chosen_target)
            success = player.build_structure("city", chosen_target, self.game.board)

            if not success:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "chosen_tw": chosen_target,
                    "reason": "build_structure(city) failed during exact staged plan",
                    "trade_info": dict(trade_info),
                    "staged_plan": dict(staged_plan),
                    "hand_after": _current_hand_dict(),
                }

            result = {
                "success": True,
                "requested_activity": requested_activity,
                "actual_activity": plan_type,
                "chosen_tw": chosen_target,
                "trade_info": dict(trade_info),
                "staged_plan": dict(staged_plan),
                "hand_after": _current_hand_dict(),
            }

            print("PLAY EXECUTION RESULT:", {
                "player_id": getattr(player, "id", "?"),
                "player_color": getattr(player, "color", "?"),                  
                "requested_activity": requested_activity,
                "plan_type": plan_type,
                "success": result.get("success"),
                "actual_activity": result.get("actual_activity"),
                "chosen_tw": result.get("chosen_tw"),
                "chosen_target": staged_plan.get("chosen_target"),
                "chosen_upgrade": staged_plan.get("chosen_upgrade"),
                "player_settlements_after": list(getattr(player, "settlements", [])),
                "player_cities_after": list(getattr(player, "cities", [])),
                "hand_after": result.get("hand_after"),
            })

            return result

        # ========================
        # EXECUTE DEV CARD
        # ========================
        if plan_type == "buy_discovery_card":
            success = player.build_structure("development_card", -1, self.game.board)
            if not success:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "reason": "build_structure(development_card) failed during exact staged plan",
                    "trade_info": dict(trade_info),
                    "staged_plan": dict(staged_plan),
                    "hand_after": _current_hand_dict(),
                }

            picked = self._draw_dev_card_from_stack()
            if picked is None:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "reason": "Deck empty after staged dev-card cost spend",
                    "trade_info": dict(trade_info),
                    "staged_plan": dict(staged_plan),
                    "hand_after": _current_hand_dict(),
                }

            self._grant_dev_card(player, picked)

            result = {
                "success": True,
                "actual_activity": "buy_discovery_card",
                "drawn_card": picked,
            }

            print("PLAY EXECUTION RESULT:", {
                "player_id": getattr(player, "id", "?"),
                "player_color": getattr(player, "color", "?"),                  
                "requested_activity": requested_activity,
                "plan_type": plan_type,
                "success": result.get("success"),
                "actual_activity": result.get("actual_activity"),
                "drawn_card": result.get("drawn_card"),
                "hand_after": player.rcards_in_hand()[0],
            })

            return result

        # ========================
        # EXECUTE 4 DEV CARDS
        # ========================
        if plan_type == "buy_4_discovery_cards":
            drawn_cards: List[str] = []

            for _ in range(4):
                required_one = {"wheat": 1, "ore": 1, "wool": 1}
                trade_one = _auto_trade_to_pay(required_one)

                if not bool(trade_one.get("success", False)):
                    break

                success = player.build_structure("development_card", -1, self.game.board)
                if not success:
                    break

                picked = self._draw_dev_card_from_stack()
                if picked is None:
                    break

                self._grant_dev_card(player, picked)
                drawn_cards.append(picked)

            if not drawn_cards:
                return {
                    "success": False,
                    "requested_activity": requested_activity,
                    "actual_activity": plan_type,
                    "reason": "Could not buy any staged development cards",
                    "trade_info": dict(trade_info),
                    "staged_plan": dict(staged_plan),
                    "hand_after": _current_hand_dict(),
                }

            return {
                "success": True,
                "requested_activity": requested_activity,
                "actual_activity": plan_type,
                "drawn_cards": drawn_cards,
                "trade_info": dict(trade_info),
                "staged_plan": dict(staged_plan),
                "hand_after": _current_hand_dict(),
            }

        return self._execute_activity(player, requested_activity)

    def jump_to_next_event(self) -> None:
        """
        JUMP phase:
        - Build and print the unified fast-forward table
        - Choose the earliest chronological event
        - Advance deterministic expected-hand time
        - Apply skipped expected income
        - Build and stage an EXACT plan when possible
        - Store expected viable actions for PLAY-time reranking
        - Move displayed round/turn/current player to that staged event

        No board action is executed here.
        """
        if self.game.game_over:
            return
        if self.game.phase != "Execution":
            return
        if getattr(self.game, "ff_waiting_for_play", False) and getattr(self.game, "ff_pending_event", None):
            return

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"FastForwardEngine.jump_to_next_event | "
                    f"round={self.game.round} turn={self.game.turn} "
                    f"ff_rolls={getattr(self.game, 'ff_elapsed_rolls', 0.0):.3f}\n"
                )

        # ------------------------------------------------------------
        # 1. Repair / refresh contract rows and apply one-time skip if needed
        # ------------------------------------------------------------
        rows = self._repair_ff_contract_rows()

        skip = getattr(self.game, "ff_skip_once", None)
        if skip:
            rows = [
                r for r in rows
                if not (
                    int(r.get("player_id", -1)) == int(skip.get("player_id", -2))
                    and (
                        str(r.get("requested_activity", r.get("strategy", ""))) == str(skip.get("requested_activity", ""))
                        or str(r.get("strategy", "")) == str(skip.get("strategy", ""))
                    )
                    and int(r.get("pred_round", -1)) == int(skip.get("round", -2))
                    and int(r.get("pred_turn", -1)) == int(skip.get("turn", -2))
                )
            ]

            self.game.ff_skip_once = None
            rows = self._sort_ff_contract_rows(rows)
            self.game.ff_contract_rows = rows

        print("FF ROW EH CHECK:")
        for row in getattr(self.game, "ff_contract_rows", []) or []:
            print({
                "player": row.get("player_id"),
                "strategy": row.get("strategy"),
                "requested": row.get("requested_activity"),
                "chosen_eh_score": row.get("chosen_expected_hand_score"),
                "chosen_eh_conf": row.get("chosen_expected_hand_confidence"),
                "actions": [
                    {
                        "code": a.get("code"),
                        "activity": a.get("activity"),
                        "score": a.get("score"),
                        "markov_score": a.get("markov_score"),
                        "eh_score": a.get("expected_hand_score"),
                        "eh_conf": a.get("expected_hand_confidence"),
                        "timing_score": a.get("timing_score_for_ranking"),
                        "timing_source": a.get("timing_source_for_ranking"),
                    }
                    for a in (row.get("expected_viable_actions", []) or [])
                ],
            })

        # ------------------------------------------------------------
        # 2. Print/display rows
        # ------------------------------------------------------------
        display_rows = self._contract_rows_for_display(rows)

        # TEMP v014 debug overlay:
        # keep the exact rows shown in the terminal table available to the GUI.
        self.game.ff_debug_prediction_rows = list(display_rows)

        self._print_prediction_rows(
            display_rows,
            "Fast-forward table after JUMP (nearest event at top)"
        )

        if not rows:
            return

        chosen = rows[0]

        # ------------------------------------------------------------
        # 3. Convert chosen contract time to elapsed time
        # ------------------------------------------------------------
        now_player_turns = self._ff_now_player_turns()
        chosen_abs_player_turns = float(
            chosen.get(
                "abs_player_turns",
                now_player_turns + float(chosen.get("delta_rolls", 0.0))
            )
        )

        delta_player_turns = max(0.0, chosen_abs_player_turns - now_player_turns)

        num_players = max(len(getattr(self.game, "players", [])), 1)
        delta_rolls = delta_player_turns * float(num_players)

        if delta_player_turns < 0:
            delta_player_turns = 0.0
        if delta_player_turns > 9999:
            delta_player_turns = 9999.0

        if delta_rolls < 0:
            delta_rolls = 0.0
        if delta_rolls > 9999:
            delta_rolls = 9999.0

        # ------------------------------------------------------------
        # 4. Read chosen row metadata
        # ------------------------------------------------------------
        chosen_id = int(chosen["player_id"])
        requested_activity = chosen.get("requested_activity", chosen["strategy"])
        display_strategy = chosen.get("strategy", requested_activity)
        predicted_extra_roads = chosen.get("predicted_extra_roads")
        chosen_events = dict(chosen.get("event_times", {}))

        chosen_player = next(p for p in self.game.players if int(p.id) == int(chosen_id))

        # ------------------------------------------------------------
        # 5. Advance expected time and income
        # ------------------------------------------------------------
        self._advance_markov_time(delta_rolls)
        self._apply_elapsed_income_to_all_players(delta_rolls)

        # ------------------------------------------------------------
        # 6. Build exact staged plan for the headline chosen action
        # ------------------------------------------------------------
        staged_plan = self._build_staged_plan(chosen_player, requested_activity)

        if requested_activity == "new_settlement":
            actual_extra_roads = staged_plan.get("extra_roads_needed")

            if actual_extra_roads is None and isinstance(staged_plan.get("best_blocked_plan"), dict):
                actual_extra_roads = staged_plan["best_blocked_plan"].get("extra_roads_needed")

            if predicted_extra_roads is not None and actual_extra_roads is not None:
                if int(actual_extra_roads) != int(predicted_extra_roads):
                    staged_plan = {
                        "plan_type": "new_settlement",
                        "plan_available": False,
                        "reason": (
                            f"Settlement road-count mismatch: "
                            f"predicted={int(predicted_extra_roads)}, "
                            f"actual={int(actual_extra_roads)}. Recompute best strategy."
                        ),
                        "predicted_extra_roads": int(predicted_extra_roads),
                        "actual_extra_roads": int(actual_extra_roads),
                        "best_blocked_plan": dict(staged_plan),
                        "reestimate_required": True,
                        "retry_strategy": "best_strategy_from_scratch",
                    }

        # ------------------------------------------------------------
        # 7. Stage the event but do not execute yet
        # ------------------------------------------------------------
        self.game.ff_pending_event = {
            "player_id": chosen_player.id,
            "player_sequence": chosen_player.sequence,
            "player_color": chosen_player.color,

            "requested_activity": requested_activity,
            "display_strategy": display_strategy,
            "predicted_extra_roads": predicted_extra_roads,

            "delta_player_turns": float(delta_player_turns),
            "delta_rolls": float(delta_rolls),

            "event_times": dict(chosen_events),

            # v014 Module 1/4:
            # expected viable action menu from the JUMP table.
            # PLAY uses this to rank/try C/D/S alternatives before giving up.
            "expected_viable_actions": list(chosen.get("expected_viable_actions", [])),
            "expected_viable_codes": chosen.get("expected_viable_codes", "-"),

            # Original exact staged plan for the headline chosen action.
            "staged_plan": dict(staged_plan),

            # Prediction / diagnostic metadata
            "prediction_source_mode": chosen.get("source_mode"),
            "prediction_used_heavy": bool(chosen.get("used_heavy", False)),
            "prediction_refine_reasons": list(chosen.get("refine_reasons", [])),
            "prediction_focus_activities": list(chosen.get("focus_activities", [])),
            "prediction_changed_activities": list(chosen.get("changed_activities", [])),
            "prediction_activity_improvements": dict(chosen.get("activity_improvements", {})),
            "prediction_overflow_summary": dict(chosen.get("overflow_summary", {})),
            "prediction_chosen_explanation": dict(chosen.get("chosen_explanation", {})),
            "prediction_plan_summary": dict(chosen.get("prediction_plan_summary", {})),
        }

        self.game.ff_waiting_for_play = True
        self.game.ff_button_mode = "PLAY"

        # ------------------------------------------------------------
        # 8. Move displayed round/turn/current player to chosen row
        # ------------------------------------------------------------
        chosen_pred_round = chosen.get("pred_round", "?")
        chosen_pred_turn = chosen.get("pred_turn", "?")

        if chosen_pred_round != "?":
            self.game.round = int(chosen_pred_round)

        if chosen_pred_turn != "?":
            self.game.turn = int(chosen_pred_turn)
        else:
            self.game.turn = chosen_player.sequence

        self.game.current_player = chosen_player
        self.game.sync_round_turn()

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"FastForwardEngine.jump_to_next_event | "
                    f"staged player={chosen_player.id} color={chosen_player.color} "
                    f"activity={requested_activity} delta={delta_rolls:.3f} "
                    f"round={self.game.round} turn={self.game.turn} "
                    f"expected_viable_codes={self.game.ff_pending_event.get('expected_viable_codes')} "
                    f"staged_plan={staged_plan}\n"
                )

    def play_staged_event(self) -> None:
        """
        PLAY phase:
        - Verify the EXACT staged plan is still executable now
        - Only then execute that exact plan
        - Refresh outlooks if the board/state actually changed
        - Store last-event snapshot on game
        - Log + finalize
        - If the staged plan is not executable anymore, clear it and return to JUMP

        No new prediction is staged here.
        """
        print(
        "PLAY ENTERED | "
        f"phase={getattr(self.game, 'phase', None)} "
        f"pending={bool(getattr(self.game, 'ff_pending_event', None))} "
        f"waiting={getattr(self.game, 'ff_waiting_for_play', None)} "
        f"button={getattr(self.game, 'ff_button_mode', None)}"
        )

        if self.game.game_over:
            return
        if self.game.phase != "Execution":
            return

        pending = getattr(self.game, "ff_pending_event", None)
        if not pending:
            return

        player_id = pending.get("player_id")
        requested_activity = pending.get("requested_activity", "buy_discovery_card")
        delta_rolls = float(pending.get("delta_rolls", 0.0))
        all_event_times = dict(pending.get("event_times", {}))
        staged_plan = dict(pending.get("staged_plan", {}) or {})

        player = next((p for p in self.game.players if p.id == player_id), None)
        if player is None:
            self.game.clear_pending_fast_forward_event()
            return

        # Keep display aligned with the staged event
        self.game.turn = player.sequence
        self.game.current_player = player
        self.game.sync_round_turn()

        # ------------------------------------------------------------
        # 1. Choose best expected viable action, then verify exact guard.
        # ------------------------------------------------------------
        print("PLAY BEFORE EXPECTED-ACTION RANKING / GUARD")

        expected_decision = self._choose_best_expected_viable_plan_at_play(
            player=player,
            pending=pending,
        )

        selected_requested_activity = expected_decision.get(
            "selected_requested_activity",
            requested_activity,
        )

        if selected_requested_activity is None:
            requested_activity = None
        else:
            requested_activity = str(selected_requested_activity)

        staged_plan = dict(
            expected_decision.get("selected_staged_plan", staged_plan) or {}
        )
        guard_info = dict(
            expected_decision.get("selected_guard_info", {}) or {}
        )

        print("PLAY EXPECTED-ACTION DECISION:", {
            "used_expected_action_ranking": expected_decision.get("used_expected_action_ranking"),
            "ranking_available": expected_decision.get("ranking_available"),
            "ranking_source": expected_decision.get("ranking_source"),
            "selected_requested_activity": requested_activity,
            "fallback_reason": expected_decision.get("fallback_reason"),
            "attempts": expected_decision.get("attempts"),
        })
        print("PLAY GUARD:", guard_info)

        # Store for logging/debugging.
        pending["expected_action_decision"] = dict(expected_decision)
        pending["requested_activity_after_expected_ranking"] = requested_activity
        pending["staged_plan_after_expected_ranking"] = dict(staged_plan)

        if not bool(guard_info.get("executable", False)):
            no_exact_action = requested_activity is None

            result = {
                "success": False,
                "requested_activity": requested_activity,
                "actual_activity": None if no_exact_action else guard_info.get("plan_type", requested_activity),
                "reason": guard_info.get("reason", "Staged plan not executable"),
                "guard_info": dict(guard_info),
                "reestimate_required": True,
                "retry_strategy": None if no_exact_action else requested_activity,
                "no_exact_executable_action": bool(no_exact_action),
            }

            pending = getattr(self.game, "ff_pending_event", {}) or {}

            fresh_contract = self._reestimate_contract_after_play_failure(
                player,
                failure_result=result,
                pending=pending,
            )
            result["reestimated_contract"] = dict(fresh_contract or {})
            result["reestimated_contract_created"] = fresh_contract is not None

            self.game.ff_pending_event = None
            self.game.ff_waiting_for_play = False
            self.game.ff_button_mode = "JUMP"

            self.game.ff_last_actor_id = player.id
            self.game.ff_last_requested_activity = requested_activity
            self.game.ff_last_actual_activity = None
            self.game.ff_last_details = dict(result)

            self._log_and_finalize(
                player=player,
                requested_activity=requested_activity or "no_exact_executable_action",
                delta_rolls=delta_rolls,
                details=result,
                all_event_times=all_event_times,
            )

            return

        # ------------------------------------------------------------
        # 2. Execute the EXACT staged plan
        # ------------------------------------------------------------
        result = self._execute_staged_plan(
            player,
            staged_plan,
            requested_activity=requested_activity,
        )

        def _result_changed_state(details: Dict[str, Any]) -> bool:
            """
            Detect whether the board/game state actually changed,
            even if the top-level result is a failure wrapper.

            Important edge case: settlement may fail AFTER one or more roads were built.
            """
            if not isinstance(details, dict):
                return False

            if bool(details.get("success", False)) and bool(details.get("actual_activity")):
                return True

            # Check for partial progress
            built_roads = details.get("built_roads", [])
            if isinstance(built_roads, list) and len(built_roads) > 0:
                return True

            drawn_cards = details.get("drawn_cards", [])
            if isinstance(drawn_cards, list) and len(drawn_cards) > 0:
                return True

            if details.get("drawn_card") is not None:
                return True

            # Recurse into nested failure details
            failed_details = details.get("failed_details")
            if isinstance(failed_details, dict):
                return _result_changed_state(failed_details)

            return False

        state_changed = _result_changed_state(result)

        if state_changed:
            self._refresh_all_outlooks()

        # ------------------------------------------------------------
        # 2b. Same-turn executable action chain.
        # ------------------------------------------------------------
        same_turn_chain_results: List[Dict[str, Any]] = []

        if bool(result.get("success", False)) and not getattr(self.game, "game_over", False):
            same_turn_chain_results = self._execute_immediate_action_chain(
                player=player,
                max_steps=8,
            )

            if same_turn_chain_results:
                result["same_turn_chain_results"] = list(same_turn_chain_results)
                result["same_turn_chain_count"] = len(same_turn_chain_results)

                last_chain_result = same_turn_chain_results[-1].get("result", {})
                if isinstance(last_chain_result, dict):
                    result["last_same_turn_activity"] = last_chain_result.get("actual_activity")

                # The same-turn chain may have changed the board/resources too.
                state_changed = True

        # Determine what activity actually happened for logging/snapshot
        failed_details = result.get("failed_details", {})
        actual_activity_for_snapshot = result.get("actual_activity")

        if actual_activity_for_snapshot is None and isinstance(failed_details, dict):
            if _result_changed_state(failed_details):
                actual_activity_for_snapshot = failed_details.get(
                    "actual_activity", requested_activity
                )

        # Store last event snapshot
        self.game.ff_last_actor_id = player.id
        self.game.ff_last_requested_activity = requested_activity

        if result.get("last_same_turn_activity"):
            actual_activity_for_snapshot = result.get("last_same_turn_activity")

        self.game.ff_last_actual_activity = actual_activity_for_snapshot
        self.game.ff_last_details = dict(result)

        self._log_and_finalize(
            player=player,
            requested_activity=requested_activity,
            delta_rolls=delta_rolls,
            details=result,
            all_event_times=all_event_times,
        )

        self._remove_ff_contract_for_player(player.id)

        # If re-estimation is needed (e.g. settlement failed after building roads),
        # rebuild this player's future contract immediately so the next JUMP
        # table contains the newly selected expected strategy/round/turn.
        if result.get("reestimate_required", False):
            fresh_contract = self._reestimate_contract_after_play_failure(
                player,
                failure_result=result,
                pending=pending,
            )
            result["reestimated_contract"] = dict(fresh_contract or {})
            result["reestimated_contract_created"] = fresh_contract is not None
            self.game.clear_pending_fast_forward_event()
            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"FastForwardEngine.play_staged_event | "
                        f"player={player.id} requested={requested_activity} "
                        f"reestimate_required=True state_changed={state_changed} "
                        f"reason={result.get('reason')} staged_plan={staged_plan}\n"
                    )
            return

        # Normal completion - consume actor's contract and create a fresh one for that actor only.
        self._complete_ff_contract_for_player(player.id)

        # Normal completion - clear pending event
        self.game.clear_pending_fast_forward_event()

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"FastForwardEngine.play_staged_event | "
                    f"player={player.id} requested={requested_activity} "
                    f"actual={actual_activity_for_snapshot} "
                    f"success={result.get('success', False)} "
                    f"state_changed={state_changed} staged_plan={staged_plan}\n"
                )

    def _activity_code(self, activity: str) -> str:
        """Short display code for expected viable actions."""
        activity = str(activity or "")

        if activity == "buy_discovery_card":
            return "D"
        if activity == "buy_4_discovery_cards":
            return "D4"
        if activity == "upgrade_to_city":
            return "C"
        if activity == "new_settlement":
            return "S"
        if activity.startswith("settlement_"):
            return "S"
        if activity.startswith("road"):
            return "R"

        return "?"

    def _activity_display_name(self, activity: str) -> str:
        """Normalize activity names for the expected viable action list."""
        activity = str(activity or "")

        if activity in ("settlement_0r", "settlement_1r", "settlement_2r"):
            return "new_settlement"

        return activity

    def _safe_float_ff(self, value: Any, default: float = 9999.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _extract_action_details_from_breakdown(
        self,
        player_id: int,
        activity: str,
    ) -> Dict[str, Any]:
        """
        Extract target/explanation metadata already stored in _last_strategy_breakdown.

        This is intentionally best-effort. Missing values are allowed.
        """
        breakdown = getattr(self, "_last_strategy_breakdown", {}) or {}
        player_breakdown = dict(breakdown.get(player_id, {}) or {})
        info = dict(player_breakdown.get(activity, {}) or {})
        explanation = dict(info.get("explanation", {}) or {})

        details: Dict[str, Any] = {
            "activity": activity,
            "target": None,
            "extra_roads_needed": None,
            "settlement_target_type": None,
            "explanation": explanation,
        }

        if activity == "new_settlement":
            details["target"] = (
                info.get("chosen_target")
                or explanation.get("chosen_target")
                or info.get("predicted_chosen_target")
                or explanation.get("predicted_chosen_target")
            )
            details["extra_roads_needed"] = (
                info.get("extra_roads_needed")
                or explanation.get("extra_roads_needed")
                or info.get("predicted_extra_roads_needed")
                or explanation.get("predicted_extra_roads_needed")
            )
            details["settlement_target_type"] = (
                info.get("settlement_target_type")
                or explanation.get("settlement_target_type")
                or info.get("predicted_settlement_target_type")
                or explanation.get("predicted_settlement_target_type")
            )

        elif activity == "upgrade_to_city":
            details["target"] = (
                info.get("chosen_upgrade")
                or explanation.get("chosen_upgrade")
                or info.get("chosen_target")
                or explanation.get("chosen_target")
            )

        elif activity == "buy_discovery_card":
            details["target"] = "development_card"

        return details

    def _build_expected_viable_actions(
        self,
        player: Player,
        event_times: Dict[str, Any],
        markov_event_times: Optional[Dict[str, Any]] = None,
        expected_hand_event_times: Optional[Dict[str, Any]] = None,
        timing_source: str = "markov_light",
    ) -> List[Dict[str, Any]]:
        """
        Build the expected viable action list for one fast-forward prediction row.

        These are expected/probabilistic viable actions, not exact PAY-now actions.
        The exact guard still decides at PLAY time.

        v015/v023:
        - event_times is the ACTIVE timing table used for scheduling/ranking.
          In EH-only mode, this is expected-hand.
          In Markov/hybrid mode, this may be Markov/light.
        - markov_event_times is comparison metadata only.
        - expected_hand_event_times is EH metadata/debug only.
        """
        player_id = int(getattr(player, "id", -1))

        event_times = dict(event_times or {})
        markov_event_times = dict(markov_event_times or {})
        expected_hand_event_times = dict(expected_hand_event_times or {})

        expected_hand_debug = expected_hand_event_times.get("__debug__", {})
        if not isinstance(expected_hand_debug, dict):
            expected_hand_debug = {}

        actions: List[Dict[str, Any]] = []

        def _first_present_value(
            source: Dict[str, Any],
            aliases: Tuple[str, ...],
        ) -> Tuple[Optional[str], Any]:
            for key in aliases:
                if key in source:
                    return key, source.get(key)
            return None, None

        def _activity_aliases(activity: Any, normalized_activity: str) -> Tuple[str, ...]:
            """
            Return likely equivalent keys across:
            - Markov/light event tables
            - expected-hand event tables
            - display-normalized activities
            """
            raw = str(activity or "").strip().lower()
            norm = str(normalized_activity or "").strip().lower()

            if raw in ("buy_discovery_card", "dev_card", "development_card", "buy_dev_card") or norm == "buy_discovery_card":
                return (
                    "buy_discovery_card",
                    "dev_card",
                    "development_card",
                    "buy_dev_card",
                )

            if raw in ("upgrade_to_city", "city", "upgrade_city") or norm == "upgrade_to_city":
                return (
                    "upgrade_to_city",
                    "city",
                    "upgrade_city",
                )

            if raw in ("settlement_0r", "settlement", "next_settlement"):
                return (
                    "settlement_0r",
                    "settlement",
                    "next_settlement",
                    "new_settlement",
                )

            if raw in ("settlement_1r", "new_settlement", "build_settlement"):
                return (
                    "settlement_1r",
                    "new_settlement",
                    "build_settlement",
                    "settlement",
                )

            if raw == "settlement_2r":
                return (
                    "settlement_2r",
                    "new_settlement",
                    "settlement",
                )

            if norm == "new_settlement":
                return (
                    raw,
                    "new_settlement",
                    "settlement_1r",
                    "settlement_0r",
                    "settlement_2r",
                    "settlement",
                )

            return (raw, norm)

        def _debug_for_key(key: Optional[str], aliases: Tuple[str, ...]) -> Dict[str, Any]:
            if not expected_hand_debug:
                return {}

            if key and isinstance(expected_hand_debug.get(key), dict):
                return dict(expected_hand_debug.get(key) or {})

            for alias in aliases:
                if isinstance(expected_hand_debug.get(alias), dict):
                    return dict(expected_hand_debug.get(alias) or {})

            return {}

        for activity, raw_score in event_times.items():
            activity_text = str(activity or "")

            # Ignore debug/internal payloads.
            if activity_text.startswith("__"):
                continue

            # This is the selected primary scheduling score.
            score = self._safe_float_ff(raw_score, 9999.0)

            if score >= 9999.0:
                continue

            normalized_activity = self._activity_display_name(activity)
            aliases = _activity_aliases(activity, normalized_activity)

            details = self._extract_action_details_from_breakdown(
                player_id=player_id,
                activity=normalized_activity,
            )

            markov_key, markov_raw_score = _first_present_value(
                markov_event_times,
                aliases,
            )
            expected_hand_key, expected_hand_raw_score = _first_present_value(
                expected_hand_event_times,
                aliases,
            )

            markov_score = self._safe_float_ff(markov_raw_score, 9999.0)
            expected_hand_score = self._safe_float_ff(expected_hand_raw_score, 9999.0)

            # Display/ranking honesty:
            # In expected-hand-only mode there is no real Markov comparison value.
            # Keep markov_score as None so debug output prints '-' instead of
            # accidentally mirroring the EH timing as M=EH.
            has_real_markov_score = bool(
                markov_event_times
                and markov_key is not None
                and markov_score < 9999.0
            )
            markov_score_for_row = float(markov_score) if has_real_markov_score else None

            eh_debug = _debug_for_key(expected_hand_key, aliases)

            expected_hand_confidence = self._safe_float_ff(
                eh_debug.get("confidence", 0.0),
                0.0,
            )
            expected_hand_confidence_target = self._safe_float_ff(
                eh_debug.get("confidence_target", None),
                None,
            )
            expected_hand_confidence_label = eh_debug.get("confidence_label")
            expected_hand_found = bool(
                eh_debug.get(
                    "found",
                    expected_hand_score < 9999.0,
                )
            )

            if has_real_markov_score and expected_hand_score < 9999.0:
                expected_hand_delta = float(expected_hand_score) - float(markov_score)
            else:
                expected_hand_delta = None

            action_row = {
                "activity": normalized_activity,
                "raw_activity_key": activity,
                "code": self._activity_code(normalized_activity),

                # Active scheduling/ranking timing.
                # In EH-only mode this is EH.
                # In Markov mode this is Markov/light.
                "score": float(score),
                "timing_source": timing_source,
                "timing_score_for_ranking": float(score),
                "timing_source_for_ranking": timing_source,

                # Explicit timing metadata.
                # In EH-only mode markov_score will usually be 9999.0 because
                # markov_event_times is intentionally empty.
                "markov_score": markov_score_for_row,
                "markov_key": markov_key if has_real_markov_score else None,
                "expected_hand_score": float(expected_hand_score),
                "expected_hand_turns": float(expected_hand_score),
                "expected_hand_delta": expected_hand_delta,
                "expected_hand_confidence": float(expected_hand_confidence),
                "expected_hand_confidence_target": expected_hand_confidence_target,
                "expected_hand_confidence_label": expected_hand_confidence_label,
                "expected_hand_found": expected_hand_found,
                "expected_hand_key": expected_hand_key,
                "expected_hand_debug": eh_debug,

                # Existing action details.
                "target": details.get("target"),
                "extra_roads_needed": details.get("extra_roads_needed"),
                "settlement_target_type": details.get("settlement_target_type"),
                "expected_viable": True,
                "exact_guard_required": True,
                "explanation": details.get("explanation", {}),
            }

            actions.append(action_row)

        # Deduplicate aliases that normalize to the same practical action.
        # Example: new_settlement, settlement_0r, settlement_1r and settlement_2r
        # can all display as S/new_settlement with the same target. Keep the
        # quickest row and merge useful metadata from duplicates.
        deduped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for row in actions:
            key = (
                str(row.get("code", "")),
                str(row.get("activity", "")),
                str(row.get("target", "")),
            )
            if key not in deduped:
                deduped[key] = row
                continue

            old_row = deduped[key]
            old_score = self._safe_float_ff(old_row.get("score"), 9999.0)
            new_score = self._safe_float_ff(row.get("score"), 9999.0)

            # Keep the quickest timing row. If scores tie, prefer the one with
            # more specific road-count metadata.
            replace = new_score < old_score - 1e-9
            if abs(new_score - old_score) <= 1e-9:
                replace = (
                    old_row.get("extra_roads_needed") is None
                    and row.get("extra_roads_needed") is not None
                )

            if replace:
                # Preserve a compact alias breadcrumb from the old row.
                aliases_seen = list(old_row.get("deduped_aliases", []) or [])
                aliases_seen.append(old_row.get("raw_activity_key"))
                row["deduped_aliases"] = [a for a in aliases_seen if a is not None]
                deduped[key] = row
            else:
                aliases_seen = list(old_row.get("deduped_aliases", []) or [])
                aliases_seen.append(row.get("raw_activity_key"))
                old_row["deduped_aliases"] = [a for a in aliases_seen if a is not None]

        actions = list(deduped.values())

        actions.sort(
            key=lambda row: (
                self._safe_float_ff(row.get("score"), 9999.0),
                str(row.get("activity", "")),
                str(row.get("target", "")),
            )
        )

        return actions

    def _format_expected_viable_codes(
        self,
        expected_viable_actions: List[Dict[str, Any]],
    ) -> str:
        """
        Compact display form:
            D,C,S

        D  = dev card
        C  = city
        S  = settlement
        R  = road
        """
        codes: List[str] = []

        for action in expected_viable_actions or []:
            code = str(action.get("code", "?"))

            if code not in codes:
                codes.append(code)

        return ",".join(codes) if codes else "-"

    def _build_prediction_rows(self) -> List[Dict[str, Any]]:
        """
        Build fast-forward prediction rows for all players.

        Hybrid policy:
        1) first build a LIGHT-only provisional table for all players
        2) sort that provisional table
        3) rebuild each player's row with the hybrid event estimator, using:
        - the provisional table EXCLUDING that player's own row
        - the already computed LIGHT event_times from pass 1
        4) sort the final rows chronologically

        Transparency additions:
        - store refinement reasons in each final row
        - store whether final winning value came from LIGHT or HEAVY
        - store chosen strategy explanation
        - store compact overflow summary
        - store a prediction/staged-plan summary placeholder
        """
        light_rows: List[Dict[str, Any]] = []

        def _safe_float(value: Any, default: float = 9999.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _sort_key(r: Dict[str, Any]):
            pr = r.get("pred_round", "?")
            pt = r.get("pred_turn", "?")

            if pr == "?" or pt == "?":
                return (999999, 999999, _safe_float(r.get("delta_rolls", 9999.0)))

            return (int(pr), int(pt), _safe_float(r.get("delta_rolls", 9999.0)))

        def _rank_event_times(event_times: Dict[str, Any]):
            valid = []
            for activity, val in (event_times or {}).items():
                fv = _safe_float(val, 9999.0)
                if fv < 9999.0:
                    valid.append((activity, fv))

            if not valid:
                return None, 9999.0, None, 9999.0

            valid.sort(key=lambda x: x[1])
            best_activity, best_time = valid[0]

            if len(valid) >= 2:
                second_activity, second_time = valid[1]
            else:
                second_activity, second_time = None, 9999.0

            return best_activity, float(best_time), second_activity, float(second_time)

        def _split_display_and_requested_activity(activity: Optional[str]) -> Tuple[str, str, Optional[int]]:
            """
            Return:
                display_strategy, requested_activity, predicted_extra_roads
            """
            raw = str(activity or "").strip().lower()

            if raw in ("settlement_0r", "settlement_1r", "settlement_2r"):
                try:
                    predicted_extra_roads = int(raw.split("_")[1].replace("r", ""))
                except Exception:
                    predicted_extra_roads = None

                return raw, "new_settlement", predicted_extra_roads

            if raw == "new_settlement":
                return raw, "new_settlement", None

            return raw, raw, None

        def _extract_overflow_summary(expl: Dict[str, Any]) -> Dict[str, Any]:
            expl = expl or {}
            return {
                "triggered": bool(expl.get("overflow_triggered", False)),
                "resource": expl.get("overflow_dominant_resource"),
                "pips": _safe_float(expl.get("overflow_dominant_resource_pips", 0.0), 0.0),
                "rate": int(expl.get("overflow_effective_trade_rate", 4) or 4),
                "trades": int(expl.get("overflow_needed_trades", 0) or 0),
                "off_resource_cards_to_buy": int(expl.get("overflow_off_resource_cards_to_buy", 0) or 0),
                "expected_dominant_cards_by_horizon": _safe_float(
                    expl.get("overflow_expected_dominant_cards_by_horizon", 0.0),
                    0.0,
                ),
                "can_fund": bool(expl.get("overflow_can_fund_within_horizon", False)),
                "cap_bind": bool(expl.get("overflow_cap_bind_risk", False)),
                "weak_off": bool(expl.get("overflow_weak_off_resource_exists", False)),
                "score": _safe_float(expl.get("overflow_score", 9999.0), 9999.0),
            }

        def _make_plan_summary_from_breakdown(activity: str, breakdown: Dict[str, Any]) -> Dict[str, Any]:
            breakdown = breakdown or {}
            expl = breakdown.get("explanation", {}) or {}

            summary = {
                "activity": activity,
                "score": _safe_float(breakdown.get("score", 9999.0), 9999.0),
                "heavy_mode": bool(breakdown.get("heavy_mode", False)),
            }

            if activity == "new_settlement":
                summary.update({
                    "chosen_target": breakdown.get("chosen_target", expl.get("chosen_target")),
                    "extra_roads_needed": int(
                        expl.get("extra_roads_needed", breakdown.get("extra_roads_needed", 0)) or 0
                    ),
                    "settlement_target_type": breakdown.get(
                        "settlement_target_type",
                        expl.get("settlement_target_type"),
                    ),
                })

            elif activity == "upgrade_to_city":
                summary.update({
                    "chosen_upgrade": breakdown.get("chosen_upgrade", expl.get("chosen_upgrade")),
                    "light_prefilter_score": _safe_float(
                        breakdown.get("light_prefilter_score", expl.get("light_prefilter_score", 9999.0)),
                        9999.0,
                    ),
                })

            elif activity == "buy_discovery_card":
                summary.update({
                    "target": "development_card",
                })

            return summary

        # ------------------------------------------------------------
        # v015 expected-hand side-by-side helpers
        # ------------------------------------------------------------
        def _expected_hand_debug_prints_enabled() -> bool:
            try:
                from core.constants import EXPECTED_HAND_DEBUG_PRINTS
                return bool(EXPECTED_HAND_DEBUG_PRINTS)
            except Exception:
                return True

        def _expected_hand_primary_enabled() -> bool:
            try:
                from core.constants import EXPECTED_HAND_PRIMARY_ENGINE
                return bool(EXPECTED_HAND_PRIMARY_ENGINE)
            except Exception:
                return False

        def _expected_hand_primary_for_jump_enabled() -> bool:
            """
            v015/v023 staged rollout:
            - EXPECTED_HAND_PRIMARY_ENGINE enables expected-hand generally.
            - EXPECTED_HAND_PRIMARY_FOR_JUMP decides whether JUMP scheduling
              may choose best_activity/best_time from expected-hand instead
              of Markov/light.

            If the new constant is missing, stay conservative and keep
            Markov/light for JUMP scheduling.
            """
            try:
                from core.constants import (
                    EXPECTED_HAND_PRIMARY_ENGINE,
                    EXPECTED_HAND_PRIMARY_FOR_JUMP,
                )
                return (
                    bool(EXPECTED_HAND_PRIMARY_ENGINE)
                    and bool(EXPECTED_HAND_PRIMARY_FOR_JUMP)
                )
            except Exception:
                return False

        def _expected_hand_only_enabled() -> bool:
            return self._expected_hand_only_runtime()

        def _expected_hand_config() -> Dict[str, Any]:
            cfg = {
                "confidence_target": 0.90,
                "max_turns": 60.0,
                "step": 0.25,
                "continuous_trading": True,
            }

            try:
                from core.constants import (
                    EXPECTED_HAND_CONFIDENCE_TARGET,
                    EXPECTED_HAND_MAX_TURNS,
                    EXPECTED_HAND_STEP,
                    EXPECTED_HAND_CONTINUOUS_TRADING,
                )

                cfg["confidence_target"] = float(EXPECTED_HAND_CONFIDENCE_TARGET)
                cfg["max_turns"] = float(EXPECTED_HAND_MAX_TURNS)
                cfg["step"] = float(EXPECTED_HAND_STEP)
                cfg["continuous_trading"] = bool(EXPECTED_HAND_CONTINUOUS_TRADING)
            except Exception:
                pass

            return cfg

        def _get_expected_hand_event_times_for_player(
            player_obj: Player,
            *,
            require_confidence: bool = True,
        ) -> Dict[str, Any]:
            """
            Local v015 expected-hand bridge.

            This keeps _build_prediction_rows() usable even when this
            fast_forward.py version does not yet define a separate
            _get_event_times_for_player_expected_hand(...) method.
            """
            try:
                from core.resource_time_estimator import estimate_event_times_for_player

                cfg = _expected_hand_config()

                eh_times = estimate_event_times_for_player(
                    self.game.board,
                    player_obj,
                    confidence_target=float(cfg["confidence_target"]),
                    step=float(cfg["step"]),
                    max_turns=float(cfg["max_turns"]),
                    continuous_trading=bool(cfg["continuous_trading"]),
                    require_confidence=bool(require_confidence),
                    include_debug=True,
                )

                eh_times = dict(eh_times or {})
                eh_times = self._sanitize_zero_time_event_times(
                    player_obj,
                    eh_times,
                    timing_source="expected_hand",
                )

                # Alias the generic settlement estimate to exact road-count keys
                # when the estimator only returned new_settlement.
                if "new_settlement" in eh_times:
                    eh_times.setdefault("settlement_0r", eh_times.get("new_settlement"))
                    eh_times.setdefault("settlement_1r", eh_times.get("new_settlement"))
                    eh_times.setdefault("settlement_2r", eh_times.get("new_settlement"))

                debug = dict(eh_times.get("__debug__", {}) or {})
                if "new_settlement" in debug:
                    debug.setdefault("settlement_0r", debug.get("new_settlement"))
                    debug.setdefault("settlement_1r", debug.get("new_settlement"))
                    debug.setdefault("settlement_2r", debug.get("new_settlement"))
                    eh_times["__debug__"] = debug

                return eh_times

            except Exception as exc:
                print("EXPECTED-HAND EVENT TIMES FAILED:", {
                    "player": getattr(player_obj, "id", "?"),
                    "error": repr(exc),
                })
                return {"__debug__": {"error": repr(exc)}}

        def _first_present_value(source: Dict[str, Any], aliases: Tuple[str, ...]) -> Tuple[Optional[str], Any]:
            for key in aliases:
                if key in source:
                    return key, source.get(key)
            return None, None

        def _build_expected_hand_comparison_rows(
            markov_event_times: Dict[str, Any],
            expected_hand_event_times: Dict[str, Any],
            *,
            display_strategy: Optional[str] = None,
            requested_activity: Optional[str] = None,
            best_activity: Optional[str] = None,
            predicted_extra_roads: Optional[int] = None,
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
            """
            Build compact Markov-vs-expected-hand rows for storage and printing.

            These rows are comparison metadata only. They do not change the
            primary timing source unless EXPECTED_HAND_PRIMARY_ENGINE is enabled
            elsewhere.

            v015 fix:
            - Defines _as_timing_float locally so this nested helper never depends
              on an outer helper that may not exist in all fast_forward.py versions.
            - Normalizes EH=0/conf=1 rows to label="exact".
            - Preserves exact settlement aliases so settlement_0r/1r/2r do not get
              accidentally compared against the wrong generic settlement row.
            """

            def _as_timing_float(value: Any, default: Any = None) -> Any:
                try:
                    if value is None:
                        return default
                    value_f = float(value)
                    if value_f != value_f:  # NaN
                        return default
                    return value_f
                except Exception:
                    return default

            def _first_present(mapping: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[str], Any]:
                for key in keys:
                    if key in mapping:
                        return key, mapping.get(key)
                return None, None

            def _debug_for_key(debug_payload: Dict[str, Any], key: Optional[str]) -> Dict[str, Any]:
                if not key:
                    return {}
                value = debug_payload.get(key, {})
                return dict(value or {}) if isinstance(value, dict) else {}

            markov_times = dict(markov_event_times or {})
            eh_times_all = dict(expected_hand_event_times or {})
            eh_debug = dict(eh_times_all.get("__debug__", {}) or {})

            # Public EH times only. Never store the full __debug__ payload here.
            eh_times = {
                str(k): v
                for k, v in eh_times_all.items()
                if k != "__debug__" and not str(k).startswith("__")
            }

            def _settlement_aliases_for_row() -> Tuple[str, ...]:
                """
                Prefer the exact settlement variant for this row when known.
                This avoids comparing settlement_1r/settlement_2r against a
                generic new_settlement value when the current chosen row is exact.
                """
                exact = None

                for candidate in (
                    display_strategy,
                    requested_activity,
                    best_activity,
                ):
                    candidate_s = str(candidate or "")
                    if candidate_s in ("settlement_0r", "settlement_1r", "settlement_2r"):
                        exact = candidate_s
                        break

                if exact is None:
                    try:
                        roads_needed = int(predicted_extra_roads if predicted_extra_roads is not None else 1)
                        roads_needed = max(0, min(2, roads_needed))
                        exact = f"settlement_{roads_needed}r"
                    except Exception:
                        exact = None

                base = [
                    "settlement_0r",
                    "settlement_1r",
                    "settlement_2r",
                    "new_settlement",
                    "settlement",
                    "next_settlement",
                ]

                if exact and exact in base:
                    return tuple([exact] + [k for k in base if k != exact])

                return tuple(base)

            settlement_aliases = _settlement_aliases_for_row()

            comparison_specs = [
                (
                    "S",
                    "new_settlement",
                    settlement_aliases,
                    settlement_aliases,
                ),
                (
                    "C",
                    "upgrade_to_city",
                    ("upgrade_to_city", "city"),
                    ("upgrade_to_city", "city"),
                ),
                (
                    "D",
                    "buy_discovery_card",
                    ("buy_discovery_card", "dev_card", "development_card"),
                    ("buy_discovery_card", "dev_card", "development_card"),
                ),
            ]

            rows: List[Dict[str, Any]] = []

            for code, activity, markov_aliases, eh_aliases in comparison_specs:
                markov_key, markov_raw = _first_present(markov_times, tuple(str(k) for k in markov_aliases))
                eh_key, eh_raw = _first_present(eh_times, tuple(str(k) for k in eh_aliases))

                markov_value = _as_timing_float(markov_raw, None)
                eh_value = _as_timing_float(eh_raw, None)

                debug_info = _debug_for_key(eh_debug, eh_key)

                # Some estimator versions store the useful fields under nested
                # result/payability/confidence_info dicts. Prefer top-level values,
                # then fall back carefully.
                confidence = debug_info.get("confidence", None)
                if confidence is None and isinstance(debug_info.get("confidence_info"), dict):
                    confidence = debug_info.get("confidence_info", {}).get("confidence", None)

                confidence_label = (
                    debug_info.get("confidence_label", None)
                    or debug_info.get("label", None)
                )
                if confidence_label is None and isinstance(debug_info.get("confidence_info"), dict):
                    confidence_label = (
                        debug_info.get("confidence_info", {}).get("label", None)
                        or debug_info.get("confidence_info", {}).get("confidence_label", None)
                    )

                found = debug_info.get("found", None)
                if found is None:
                    found = debug_info.get("payable_after_trades", None)
                if found is None and isinstance(debug_info.get("payability"), dict):
                    payability = debug_info.get("payability", {})
                    found = bool(
                        payability.get("payable", False)
                        or payability.get("payable_direct", False)
                        or payability.get("payable_after_trades", False)
                    )

                confidence_value = _as_timing_float(confidence, None)

                zero_time_suppressed = bool(
                    isinstance(debug_info, dict)
                    and debug_info.get("zero_time_suppressed_by_exact_guard")
                )

                if zero_time_suppressed:
                    found = False
                    confidence_label = "suppressed_exact_guard"
                    # Keep confidence out of the optimistic/exact display when exact guard rejected it.
                    confidence_value = None

                try:
                    eh_is_zero = eh_value is not None and float(eh_value) <= 1e-9
                except Exception:
                    eh_is_zero = False

                try:
                    confidence_is_exact = confidence_value is not None and float(confidence_value) >= 0.999999
                except Exception:
                    confidence_is_exact = False

                # Normalize exact/current-hand rows. This fixes display cases like:
                # EH=0.00 conf=1.00 label=high  ->  label=exact
                if eh_is_zero and confidence_is_exact:
                    confidence_label = "exact"
                    found = True

                if markov_value is not None and eh_value is not None:
                    delta = eh_value - markov_value
                else:
                    delta = None

                row: Dict[str, Any] = {
                    "code": code,
                    "activity": activity,
                    "markov_key": markov_key,
                    "expected_hand_key": eh_key,
                    "markov_score": markov_value,
                    "expected_hand_score": eh_value,
                    "expected_hand_delta": delta,
                    "expected_hand_confidence": confidence_value,
                    "expected_hand_confidence_label": confidence_label,
                    "expected_hand_found": found,
                    "expected_hand_zero_time_suppressed_by_exact_guard": zero_time_suppressed,
                    "expected_hand_zero_time_verified_by_exact_guard": bool(
                        isinstance(debug_info, dict)
                        and debug_info.get("zero_time_verified_by_exact_guard")
                    ),
                    "expected_hand_zero_time_suppression_reason": (
                        debug_info.get("zero_time_suppression_reason")
                        if isinstance(debug_info, dict) else None
                    ),
                }

                # Settlement rows need to carry the exact key for downstream
                # action matching. This prevents settlement_1r and settlement_2r
                # from being treated as interchangeable generic settlements.
                if code == "S":
                    row["expected_hand_exact_settlement_key"] = eh_key
                    row["markov_exact_settlement_key"] = markov_key
                    row["predicted_extra_roads"] = predicted_extra_roads

                rows.append(row)

            expected_hand_event_times_public = dict(eh_times)
            expected_hand_debug_keys = list(eh_debug.keys())

            return rows, expected_hand_event_times_public, expected_hand_debug_keys

        def _chosen_expected_hand_row(
            comparison_rows: List[Dict[str, Any]],
            requested_activity: Any,
        ) -> Dict[str, Any]:
            activity_to_code = {
                "new_settlement": "S",
                "settlement": "S",
                "settlement_0r": "S",
                "settlement_1r": "S",
                "settlement_2r": "S",
                "upgrade_to_city": "C",
                "city": "C",
                "buy_discovery_card": "D",
                "dev_card": "D",
                "development_card": "D",
            }

            requested = str(requested_activity or "")
            chosen_code = activity_to_code.get(requested, None)

            if chosen_code:
                for row in comparison_rows or []:
                    if str(row.get("code", "")) == str(chosen_code):
                        return dict(row)

            return {}

        def _attach_expected_hand_metadata_to_actions(
            expected_viable_actions: List[Dict[str, Any]],
            comparison_rows: List[Dict[str, Any]],
        ) -> None:
            """
            Attach compact expected-hand comparison metadata to each expected viable
            action row in place.

            v015 notes:
            - Markov/light remains the primary timing source unless another layer
              explicitly chooses otherwise.
            - Settlement rows prefer exact road-count matching
              (settlement_0r / settlement_1r / settlement_2r) before falling back
              to generic new_settlement / S.
            - EH rows with score 0.0 and confidence 1.0 are normalized to
              confidence_label='exact'. This fixes display/log rows like
              EH=0.00 conf=1.00 label=high.
            """

            def _key(value: Any) -> str:
                return str(value or "").strip().lower()

            def _code(value: Any) -> str:
                return str(value or "").strip().upper()

            def _as_float_or_none(value: Any) -> Optional[float]:
                try:
                    if value is None:
                        return None
                    return float(value)
                except Exception:
                    return None

            def _is_exact_zero_eh(score: Any, confidence: Any) -> bool:
                score_f = _as_float_or_none(score)
                conf_f = _as_float_or_none(confidence)
                return (
                    score_f is not None
                    and score_f <= 1e-9
                    and conf_f is not None
                    and conf_f >= 0.999999
                )

            def _normalize_label(score: Any, confidence: Any, label: Any) -> Any:
                if _is_exact_zero_eh(score, confidence):
                    return "exact"
                return label

            def _normalize_found(score: Any, confidence: Any, found: Any) -> Any:
                if _is_exact_zero_eh(score, confidence):
                    return True
                return found

            def _action_exact_settlement_key(action: Dict[str, Any]) -> Optional[str]:
                """
                Infer the exact settlement variant for an expected action row.

                Prefer explicit extra_roads_needed. Fall back to a raw activity key
                if the action already carries settlement_0r/1r/2r.
                """
                try:
                    for raw_key in (
                        action.get("extra_roads_needed"),
                        action.get("predicted_extra_roads"),
                        action.get("roads"),
                    ):
                        if raw_key is None:
                            continue
                        text = str(raw_key).strip().lower()
                        if text in ("", "-", "?", "none", "null"):
                            continue
                        roads_needed = int(float(text))
                        roads_needed = max(0, min(2, roads_needed))
                        return f"settlement_{roads_needed}r"
                except Exception:
                    pass

                for raw_activity in (
                    action.get("raw_activity_key"),
                    action.get("activity"),
                    action.get("requested_activity"),
                    action.get("settlement_target_type"),
                ):
                    text = _key(raw_activity)
                    if text in ("settlement_0r", "settlement_1r", "settlement_2r"):
                        return text

                return None

            def _copy_comparison_to_action(
                action: Dict[str, Any],
                comparison: Dict[str, Any],
            ) -> None:
                markov_score = comparison.get("markov_score")
                eh_score = comparison.get("expected_hand_score")
                eh_confidence = comparison.get("expected_hand_confidence")
                eh_label = _normalize_label(
                    eh_score,
                    eh_confidence,
                    comparison.get("expected_hand_confidence_label"),
                )
                eh_found = _normalize_found(
                    eh_score,
                    eh_confidence,
                    comparison.get("expected_hand_found"),
                )

                action["markov_score"] = markov_score
                action["expected_hand_score"] = eh_score
                action["expected_hand_delta"] = comparison.get("expected_hand_delta")
                action["expected_hand_confidence"] = eh_confidence
                action["expected_hand_confidence_label"] = eh_label
                action["expected_hand_found"] = eh_found
                action["expected_hand_key"] = comparison.get("expected_hand_key")
                action["expected_hand_exact_settlement_key"] = comparison.get(
                    "exact_settlement_key"
                )

                # Optional debug breadcrumb. Safe for logs and useful when tracing
                # whether PLAY received the same comparison row that JUMP printed.
                action["expected_hand_comparison_row"] = dict(comparison)

            try:
                comparison_by_code = {
                    _code(r.get("code")): dict(r)
                    for r in comparison_rows or []
                    if r.get("code") is not None
                }

                comparison_by_activity = {
                    _key(r.get("activity")): dict(r)
                    for r in comparison_rows or []
                    if r.get("activity") is not None
                }

                comparison_by_expected_hand_key = {
                    _key(r.get("expected_hand_key")): dict(r)
                    for r in comparison_rows or []
                    if r.get("expected_hand_key") is not None
                }

                comparison_by_markov_key = {
                    _key(r.get("markov_key")): dict(r)
                    for r in comparison_rows or []
                    if r.get("markov_key") is not None
                }

                comparison_by_exact_settlement_key = {
                    _key(r.get("exact_settlement_key")): dict(r)
                    for r in comparison_rows or []
                    if r.get("exact_settlement_key") is not None
                }

                for action in expected_viable_actions or []:
                    action_code = _code(action.get("code"))
                    action_activity = _key(action.get("activity"))
                    raw_activity_key = _key(action.get("raw_activity_key"))
                    requested_activity = _key(action.get("requested_activity"))

                    comparison: Dict[str, Any] = {}

                    is_settlement_action = (
                        action_code == "S"
                        or action_activity in (
                            "new_settlement",
                            "settlement",
                            "settlement_0r",
                            "settlement_1r",
                            "settlement_2r",
                        )
                        or raw_activity_key in (
                            "new_settlement",
                            "settlement",
                            "settlement_0r",
                            "settlement_1r",
                            "settlement_2r",
                        )
                        or requested_activity in (
                            "new_settlement",
                            "settlement",
                            "settlement_0r",
                            "settlement_1r",
                            "settlement_2r",
                        )
                    )

                    if is_settlement_action:
                        exact_settlement_key = _action_exact_settlement_key(action)

                        # Prefer exact settlement variant first.
                        if exact_settlement_key:
                            comparison = (
                                comparison_by_exact_settlement_key.get(exact_settlement_key)
                                or comparison_by_activity.get(exact_settlement_key)
                                or comparison_by_expected_hand_key.get(exact_settlement_key)
                                or comparison_by_markov_key.get(exact_settlement_key)
                                or {}
                            )

                        # Then fall back to generic settlement aliases.
                        if not comparison:
                            comparison = (
                                comparison_by_activity.get("new_settlement")
                                or comparison_by_activity.get("settlement")
                                or comparison_by_expected_hand_key.get("new_settlement")
                                or comparison_by_expected_hand_key.get("settlement")
                                or comparison_by_markov_key.get("new_settlement")
                                or comparison_by_markov_key.get("settlement")
                                or comparison_by_code.get("S")
                                or {}
                            )

                    elif action_code == "C" or action_activity in ("upgrade_to_city", "city"):
                        comparison = (
                            comparison_by_code.get("C")
                            or comparison_by_activity.get("upgrade_to_city")
                            or comparison_by_activity.get("city")
                            or comparison_by_expected_hand_key.get("upgrade_to_city")
                            or comparison_by_expected_hand_key.get("city")
                            or comparison_by_markov_key.get("upgrade_to_city")
                            or comparison_by_markov_key.get("city")
                            or {}
                        )

                    elif action_code == "D" or action_activity in (
                        "buy_discovery_card",
                        "dev_card",
                        "development_card",
                        "buy_development_card",
                    ):
                        comparison = (
                            comparison_by_code.get("D")
                            or comparison_by_activity.get("buy_discovery_card")
                            or comparison_by_activity.get("dev_card")
                            or comparison_by_activity.get("development_card")
                            or comparison_by_expected_hand_key.get("buy_discovery_card")
                            or comparison_by_expected_hand_key.get("dev_card")
                            or comparison_by_expected_hand_key.get("development_card")
                            or comparison_by_markov_key.get("buy_discovery_card")
                            or comparison_by_markov_key.get("dev_card")
                            or comparison_by_markov_key.get("development_card")
                            or {}
                        )

                    else:
                        comparison = (
                            comparison_by_code.get(action_code)
                            or comparison_by_activity.get(action_activity)
                            or comparison_by_activity.get(raw_activity_key)
                            or comparison_by_activity.get(requested_activity)
                            or comparison_by_expected_hand_key.get(action_activity)
                            or comparison_by_expected_hand_key.get(raw_activity_key)
                            or comparison_by_expected_hand_key.get(requested_activity)
                            or comparison_by_markov_key.get(action_activity)
                            or comparison_by_markov_key.get(raw_activity_key)
                            or comparison_by_markov_key.get(requested_activity)
                            or {}
                        )

                    if comparison:
                        _copy_comparison_to_action(action, comparison)

                    # Final safety normalization even if no comparison matched or if
                    # the action already carried EH metadata from another path.
                    try:
                        eh_score = action.get("expected_hand_score")
                        eh_conf = action.get("expected_hand_confidence")
                        if _is_exact_zero_eh(eh_score, eh_conf):
                            action["expected_hand_confidence_label"] = "exact"
                            action["expected_hand_found"] = True
                    except Exception:
                        pass

            except Exception as exc:
                print("EH ACTION METADATA ATTACH FAILED:", repr(exc))

        def _fmt_eh_value(value: Any, *, signed: bool = False) -> str:
            if value is None:
                return "-"
            try:
                fv = float(value)
                if fv >= 9999.0:
                    return "+9999" if signed else "9999"
                if signed:
                    return f"{fv:+.2f}"
                return f"{fv:.2f}"
            except Exception:
                return "-"

        def _print_expected_hand_comparison_rows(
            *,
            player: Player,
            rows: List[Dict[str, Any]],
            expected_hand_event_times_public: Dict[str, Any],
            expected_hand_debug_keys: List[str],
            estimator_primary: str,
        ) -> None:
            if not _expected_hand_debug_prints_enabled():
                return

            try:
                comparison_label = "EH SUMMARY" if str(estimator_primary) == "expected_hand_only" else "EH/MARKOV SUMMARY"
                print(comparison_label, {
                    "player": getattr(player, "id", "?"),
                    "color": getattr(player, "color", "?"),
                    "primary": estimator_primary,
                    "eh_keys": sorted(list((expected_hand_event_times_public or {}).keys())),
                    "eh_debug_keys": list(expected_hand_debug_keys or []),
                })

                for row in rows or []:
                    print(
                        ("EH | " if str(estimator_primary) == "expected_hand_only" else "EH/MARKOV | ")
                        + f"P{getattr(player, 'id', '?')}:{getattr(player, 'color', '?')} | "
                        f"{str(row.get('code', '?'))} "
                        f"{str(row.get('activity', '?')):<18} | "
                        f"M={_fmt_eh_value(row.get('markov_score')):>6} "
                        f"EH={_fmt_eh_value(row.get('expected_hand_score')):>6} "
                        f"Δ={_fmt_eh_value(row.get('expected_hand_delta'), signed=True):>7} "
                        f"conf={_fmt_eh_value(row.get('expected_hand_confidence')):>6} "
                        f"label={row.get('expected_hand_confidence_label')} "
                        f"found={row.get('expected_hand_found')} "
                        f"m_key={row.get('markov_key')} "
                        f"eh_key={row.get('expected_hand_key')}"
                    )

            except Exception as exc:
                print("EH/MARKOV DEBUG PRINT FAILED:", repr(exc))

        # ------------------------------------------------------------
        # PASS 1: LIGHT-ONLY provisional table
        # ------------------------------------------------------------
        for player in self.game.players:
            self._ensure_outlook(player)

            if _expected_hand_only_enabled():
                expected_hand_event_times = _get_expected_hand_event_times_for_player(
                    player,
                    require_confidence=True,
                )

                event_times = {
                    str(k): v
                    for k, v in dict(expected_hand_event_times or {}).items()
                    if k != "__debug__" and not str(k).startswith("__")
                }

                markov_event_times = {}
                source_mode = "expected_hand_only"

            else:
                if hasattr(self, "_get_event_times_for_player_light"):
                    event_times = self._get_event_times_for_player_light(player)
                else:
                    event_times = {
                        "new_settlement": 9999.0,
                        "upgrade_to_city": 9999.0,
                        "buy_discovery_card": 9999.0,
                    }

                markov_event_times = dict(event_times)
                source_mode = "light"

            if not event_times:
                event_times = {
                    "new_settlement": 9999.0,
                    "upgrade_to_city": 9999.0,
                    "buy_discovery_card": 9999.0,
                }

            best_activity, best_time, second_activity, second_time = _rank_event_times(event_times)

            if best_activity is None:
                best_activity = "buy_discovery_card"
                best_time = 9999.0
            display_strategy, requested_activity, predicted_extra_roads = (
                _split_display_and_requested_activity(best_activity)
            )    

            pred_round, pred_turn = self._estimate_predicted_round_turn(best_time, player.sequence)

            expected_viable_actions = self._build_expected_viable_actions(
                player=player,
                event_times=event_times,
                markov_event_times=markov_event_times,
                expected_hand_event_times=expected_hand_event_times if _expected_hand_only_enabled() else None,
                timing_source=source_mode,
            )

            expected_viable_codes = self._format_expected_viable_codes(
                expected_viable_actions
            )

            light_rows.append({
                "player_id": player.id,
                "player_color": player.color,
                "player_sequence": player.sequence,
                "pred_round": pred_round,
                "pred_turn": pred_turn,
                "delta_rolls": float(best_time),
                # "strategy": best_activity,
                "strategy": display_strategy,
                "requested_activity": requested_activity,
                "predicted_extra_roads": predicted_extra_roads,
                "event_times": dict(event_times),
                "markov_event_times": dict(markov_event_times),
                "expected_hand_event_times": {
                    str(k): v
                    for k, v in dict((expected_hand_event_times if _expected_hand_only_enabled() else {}) or {}).items()
                    if k != "__debug__" and not str(k).startswith("__")
                },
                "expected_hand_debug": dict(
                    dict((expected_hand_event_times if _expected_hand_only_enabled() else {}) or {}).get("__debug__", {}) or {}
                ),
                "estimator_primary": source_mode,

                # v014 expected viable actions
                "expected_viable_actions": expected_viable_actions,
                "expected_viable_codes": expected_viable_codes,

                "mode": source_mode,
                "source_mode": source_mode,
                "used_heavy": False,
                "refine_reasons": [],
                "focus_activities": [],
                "light_reused": False,
                "second_best_activity": second_activity,
                "second_best_time": float(second_time),
            })

        light_rows.sort(key=_sort_key)

        # ------------------------------------------------------------
        # PASS 2: HYBRID refinement using provisional rows EXCLUDING self
        # and REUSING pass-1 LIGHT event_times
        # ------------------------------------------------------------
        final_rows: List[Dict[str, Any]] = []

        for light_row in light_rows:
            player_id = int(light_row["player_id"])
            player = next((p for p in self.game.players if int(p.id) == player_id), None)

            if player is None:
                continue

            self._ensure_outlook(player)

            comparison_light_rows = [
                r for r in light_rows
                if int(r.get("player_id", -1)) != int(player_id)
            ]

            if _expected_hand_only_enabled():
                # In EH-only mode, pass-1 already produced sanitized expected-hand timings.
                # Do not route through the Markov/light/heavy estimator, because that
                # produces misleading source_mode='light' diagnostics and may touch
                # Markov-only paths in older helpers.
                event_times = dict(light_row.get("event_times", {}))
            elif hasattr(self, "_get_event_times_for_player"):
                event_times = self._get_event_times_for_player(
                    player,
                    light_rows_so_far=comparison_light_rows,
                    precomputed_light_event_times=dict(light_row.get("event_times", {})),
                )
            else:
                event_times = dict(light_row.get("event_times", {}))

            if not event_times:
                event_times = {
                    "new_settlement": 9999.0,
                    "upgrade_to_city": 9999.0,
                    "buy_discovery_card": 9999.0,
                }

            # ------------------------------------------------------------
            # v015: PASS 2 expected-hand defaults.
            # These defaults prevent runtime NameError crashes if any EH
            # comparison path is skipped/fails.
            # ------------------------------------------------------------
            estimator_primary = "expected_hand_only" if _expected_hand_only_enabled() else "markov_light"
            expected_hand_event_times = {"__debug__": {}}
            expected_hand_event_times_public = {}
            expected_hand_debug_keys = []
            expected_hand_comparison = {}
            expected_hand_comparison_rows = []
            chosen_expected_hand_comparison = {}

            # Build/reuse expected-hand event times.
            try:
                # Prefer pass-1 EH values if they exist.
                expected_hand_event_times = dict(
                    light_row.get("expected_hand_event_times", {}) or {}
                )

                expected_hand_debug = dict(
                    light_row.get("expected_hand_debug", {}) or {}
                )

                if expected_hand_debug:
                    expected_hand_event_times["__debug__"] = expected_hand_debug

                # If pass-1 did not store useful EH data, calculate fresh.
                if not expected_hand_event_times or not expected_hand_event_times.get("__debug__"):
                    expected_hand_event_times = _get_expected_hand_event_times_for_player(
                        player,
                        require_confidence=True,
                    )

            except TypeError:
                # Older helper signature fallback.
                try:
                    expected_hand_event_times = _get_expected_hand_event_times_for_player(player)
                except Exception as exc:
                    print("EXPECTED-HAND EVENT TIMES FAILED:", {
                        "player": getattr(player, "id", "?"),
                        "error": repr(exc),
                    })
                    expected_hand_event_times = {"__debug__": {"error": repr(exc)}}

            except Exception as exc:
                print("EXPECTED-HAND EVENT TIMES FAILED:", {
                    "player": getattr(player, "id", "?"),
                    "error": repr(exc),
                })
                expected_hand_event_times = {"__debug__": {"error": repr(exc)}}

            # ------------------------------------------------------------
            # v015/v023: choose scheduling source for JUMP.
            #
            # Markov/light remains available as fallback/comparison.
            # If EXPECTED_HAND_PRIMARY_FOR_JUMP=True and EH has finite values,
            # JUMP scheduling chooses best_activity/best_time from EH.
            # ------------------------------------------------------------
            scheduling_event_times = dict(event_times or {})
            scheduling_source = "expected_hand_only" if _expected_hand_only_enabled() else "markov_light"

            if _expected_hand_primary_for_jump_enabled():
                eh_scheduling_times = {
                    str(k): v
                    for k, v in dict(expected_hand_event_times or {}).items()
                    if k != "__debug__" and not str(k).startswith("__")
                }

                (
                    eh_best_activity,
                    eh_best_time,
                    eh_second_activity,
                    eh_second_time,
                ) = _rank_event_times(eh_scheduling_times)

                if eh_best_activity is not None and float(eh_best_time) < 9999.0:
                    scheduling_event_times = dict(eh_scheduling_times)
                    scheduling_source = "expected_hand_only" if _expected_hand_only_enabled() else "expected_hand_primary"
                    best_activity = eh_best_activity
                    best_time = float(eh_best_time)
                    second_activity = eh_second_activity
                    second_time = float(eh_second_time)
                else:
                    best_activity, best_time, second_activity, second_time = (
                        _rank_event_times(event_times)
                    )
                    scheduling_source = (
                        "expected_hand_only_no_finite_eh"
                        if _expected_hand_only_enabled()
                        else "markov_light_fallback_no_finite_eh"
                    )
            else:
                best_activity, best_time, second_activity, second_time = (
                    _rank_event_times(event_times)
                )

            if best_activity is None:
                best_activity = "buy_discovery_card"
                best_time = 9999.0

            display_strategy, requested_activity, predicted_extra_roads = (
                _split_display_and_requested_activity(best_activity)
            )

            estimator_primary = scheduling_source

            pred_round, pred_turn = self._estimate_predicted_round_turn(
                best_time,
                player.sequence,
            )

            markov_event_times_for_actions = (
                {}
                if _expected_hand_only_enabled()
                else dict(light_row.get("markov_event_times", event_times) or {})
            )

            # v015/v023:
            # Build expected viable actions using the selected scheduling source.
            # Keep Markov/light timings separately only when they exist.
            try:
                expected_viable_actions = self._build_expected_viable_actions(
                    player=player,
                    event_times=scheduling_event_times,
                    markov_event_times=markov_event_times_for_actions,
                    expected_hand_event_times=expected_hand_event_times,
                    timing_source=scheduling_source,
                )
            except TypeError:
                expected_viable_actions = self._build_expected_viable_actions(
                    player=player,
                    event_times=scheduling_event_times,
                )

            # Build compact side-by-side EH/Markov comparison rows.
            try:
                expected_hand_comparison_rows, expected_hand_event_times_public, expected_hand_debug_keys = (
                    _build_expected_hand_comparison_rows(
                        markov_event_times_for_actions,
                        expected_hand_event_times,
                        display_strategy=display_strategy,
                        requested_activity=requested_activity,
                        best_activity=best_activity,
                        predicted_extra_roads=predicted_extra_roads,
                    )
                )
            except Exception as exc:
                print("EH COMPARISON ROW BUILD FAILED:", {
                    "player": getattr(player, "id", "?"),
                    "error": repr(exc),
                })
                expected_hand_comparison_rows = []
                expected_hand_event_times_public = {}
                expected_hand_debug_keys = []

            # Backward-compatible compact comparison dict, if older row storage
            # still expects expected_hand_comparison.
            try:
                expected_hand_comparison = {
                    str(r.get("activity", r.get("code", "?"))): dict(r)
                    for r in expected_hand_comparison_rows or []
                }
            except Exception:
                expected_hand_comparison = {}

            try:
                chosen_expected_hand_comparison = _chosen_expected_hand_row(
                    expected_hand_comparison_rows,
                    requested_activity,
                )
            except Exception:
                chosen_expected_hand_comparison = {}

            # Critical: reconnect EH data to expected_viable_actions so PLAY can
            # receive expected_hand_score/confidence/timing metadata.
            try:
                _attach_expected_hand_metadata_to_actions(
                    expected_viable_actions,
                    expected_hand_comparison_rows,
                )
            except Exception as exc:
                print("EH ACTION METADATA ATTACH FAILED:", repr(exc))

            try:
                _print_expected_hand_comparison_rows(
                    player=player,
                    rows=expected_hand_comparison_rows,
                    expected_hand_event_times_public=expected_hand_event_times_public,
                    expected_hand_debug_keys=expected_hand_debug_keys,
                    estimator_primary=estimator_primary,
                )
            except Exception as exc:
                print("EH COMPARISON PRINT FAILED:", repr(exc))

            expected_viable_codes = self._format_expected_viable_codes(
                expected_viable_actions
            )

            breakdown = (
                getattr(self, "_last_strategy_breakdown", {})
                .get(player.id, {})
            )

            refine_meta = dict(breakdown.get("__refine__", {}) or {})
            # chosen_breakdown = dict(breakdown.get(best_activity, {}) or {})
            chosen_breakdown = dict(
                breakdown.get(best_activity)
                or breakdown.get(requested_activity)
                or {}
            )
            chosen_expl = dict(chosen_breakdown.get("explanation", {}) or {})

            used_heavy = bool(refine_meta.get("used_heavy", False))
            changed_activities = list(refine_meta.get("changed_activities", []))
            focus_activities = list(refine_meta.get("focus_activities", []))
            refine_reasons = list(refine_meta.get("reasons", []))
            light_reused = bool(refine_meta.get("light_reused", False))

            # Did the winning strategy/value come from a HEAVY-refined activity?
            chosen_breakdown_heavy = bool(chosen_breakdown.get("heavy_mode", False))

            if _expected_hand_only_enabled():
                used_heavy = False
                light_reused = False
                source_mode = scheduling_source
                mode = scheduling_source
                refine_reasons = []
                focus_activities = []
                changed_activities = []
            else:
                source_mode = "heavy" if (used_heavy and (best_activity in changed_activities or chosen_breakdown_heavy)) else "light"

                if used_heavy:
                    mode = "hybrid_heavy"
                elif light_reused:
                    mode = "light_reused"
                else:
                    mode = "light"

            # TEMP v014: heavy-refinement diagnostic.
            # Purpose:
            #   Explain why the FF table says Src=light / Heavy=False even when
            #   MARKOV_ENABLE_HEAVY_REFINEMENT=True.
            try:
                from core.constants import MARKOV_USE_ADAPTIVE_HEAVY
                adaptive_flag_value = bool(MARKOV_USE_ADAPTIVE_HEAVY)
            except Exception as exc:
                adaptive_flag_value = f"import_error: {exc!r}"

            heavy_helpers_available = (
                hasattr(self, "_should_refine_with_heavy")
                and hasattr(self, "_get_event_times_for_player_heavy")
            )

            heavy_diag = {
                "player_id": getattr(player, "id", "?"),
                "player_color": getattr(player, "color", "?"),

                # Constants / availability
                "MARKOV_ENABLE_HEAVY_REFINEMENT": bool(MARKOV_ENABLE_HEAVY_REFINEMENT),
                "MARKOV_USE_ADAPTIVE_HEAVY": adaptive_flag_value,
                "heavy_helpers_available": bool(heavy_helpers_available),

                # Row outcome
                "best_activity": best_activity,
                "display_strategy": display_strategy,
                "requested_activity": requested_activity,
                "best_time": _safe_float(best_time, 9999.0),
                "second_activity": second_activity,
                "second_time": _safe_float(second_time, 9999.0),

                # IMPORTANT:
                # light_gap_to_second is not local in _build_prediction_rows().
                # It must be read from refine_meta.
                "light_gap_to_second": _safe_float(
                    refine_meta.get("light_gap_to_second", 9999.0),
                    9999.0,
                ),
                "final_gap_to_second": _safe_float(
                    refine_meta.get("final_gap_to_second", 9999.0),
                    9999.0,
                ),

                # Refinement decision metadata
                "used_heavy": bool(used_heavy),
                "source_mode": source_mode,
                "mode": mode,
                "refine_reasons": list(refine_reasons),
                "focus_activities": list(focus_activities),
                "changed_activities": list(changed_activities),
                "best_activity_changed": bool(refine_meta.get("best_activity_changed", False)),
                "activity_improvements": dict(refine_meta.get("activity_improvements", {}) or {}),

                # Light/heavy comparison metadata
                "best_activity_light": refine_meta.get("best_activity_light"),
                "best_time_light": _safe_float(
                    refine_meta.get("best_time_light", 9999.0),
                    9999.0,
                ),
                "second_best_activity_light": refine_meta.get("second_best_activity_light"),
                "second_best_time_light": _safe_float(
                    refine_meta.get("second_best_time_light", 9999.0),
                    9999.0,
                ),
                "final_best_activity": refine_meta.get("final_best_activity"),
                "final_best_time": _safe_float(
                    refine_meta.get("final_best_time", 9999.0),
                    9999.0,
                ),
                "final_second_best_activity": refine_meta.get("final_second_best_activity"),
                "final_second_best_time": _safe_float(
                    refine_meta.get("final_second_best_time", 9999.0),
                    9999.0,
                ),

                # Chosen activity details
                "chosen_breakdown_heavy_mode": bool(chosen_breakdown.get("heavy_mode", False)),
                "chosen_breakdown_keys": sorted(list(chosen_breakdown.keys())),
            }

            diag_label = "FF EH TIMING DIAGNOSTIC" if _expected_hand_only_enabled() else "FF HEAVY DIAGNOSTIC"
            print(diag_label + ":", heavy_diag)

            if MG:
                try:
                    with open(FILENAME_MG, "a", encoding="utf-8") as f:
                        f.write(f"{diag_label}: {heavy_diag}\n")
                except Exception:
                    pass

            overflow_summary = _extract_overflow_summary(chosen_expl)
            chosen_plan_summary = _make_plan_summary_from_breakdown(display_strategy, chosen_breakdown)
            chosen_plan_summary["requested_activity"] = requested_activity

            # ------------------------------------------------------------
            # v015/v025 display honesty for plan_summary
            # ------------------------------------------------------------
            # chosen_plan_summary is built from the Markov/light breakdown. When JUMP
            # scheduling is EH-primary, that old breakdown score may be stale or 9999.
            # Keep the raw score for diagnostics, but make "score" reflect the score that
            # actually scheduled this row.
            try:
                raw_breakdown_score = chosen_plan_summary.get("score")
            except Exception:
                raw_breakdown_score = None

            chosen_markov_score_for_display = (
                None
                if _expected_hand_only_enabled()
                else _safe_float(
                    chosen_expected_hand_comparison.get(
                        "markov_score",
                        event_times.get(requested_activity, event_times.get(best_activity, best_time)),
                    ),
                    best_time,
                )
            )

            chosen_expected_hand_score_for_display = chosen_expected_hand_comparison.get(
                "expected_hand_score"
            )

            # This should already exist in your version, but keep it here if needed.
            display_source = scheduling_source

            chosen_plan_summary["raw_breakdown_score"] = raw_breakdown_score
            chosen_plan_summary["scheduling_source"] = scheduling_source
            chosen_plan_summary["scheduling_score"] = float(best_time)
            chosen_plan_summary["markov_score"] = chosen_markov_score_for_display
            chosen_plan_summary["expected_hand_score"] = chosen_expected_hand_score_for_display

            # Important: headline score now means "score used for JUMP scheduling".
            # The old Markov/light breakdown value is preserved as raw_breakdown_score.
            chosen_plan_summary["score"] = float(best_time)

            if predicted_extra_roads is not None:
                try:
                    chosen_plan_summary["predicted_extra_roads"] = int(predicted_extra_roads)
                except Exception:
                    chosen_plan_summary["predicted_extra_roads"] = predicted_extra_roads

            if predicted_extra_roads is not None:
                chosen_plan_summary["predicted_extra_roads"] = int(predicted_extra_roads)

            chosen_markov_score_for_display = (
                None
                if _expected_hand_only_enabled()
                else _safe_float(
                    chosen_expected_hand_comparison.get(
                        "markov_score",
                        event_times.get(requested_activity, event_times.get(best_activity, best_time)),
                    ),
                    best_time,
                )
            )

            chosen_expected_hand_score_for_display = chosen_expected_hand_comparison.get(
                "expected_hand_score"
            )

            # v015/v024 display honesty:
            # source_mode/mode still describe the Markov light/heavy pipeline.
            # scheduling_source describes what actually selected best_activity/best_time.
            display_source = scheduling_source

            # EH-only has no real Markov score. Do not mirror EH into markov_score.
            if not markov_event_times_for_actions:
                chosen_markov_score_for_display = None

            chosen_plan_summary["scheduling_source"] = scheduling_source
            chosen_plan_summary["scheduling_score"] = float(best_time)
            chosen_plan_summary["markov_score"] = chosen_markov_score_for_display
            chosen_plan_summary["expected_hand_score"] = chosen_expected_hand_score_for_display

            final_rows.append({
                "player_id": player.id,
                "player_color": player.color,
                "player_sequence": player.sequence,
                "pred_round": pred_round,
                "pred_turn": pred_turn,
                "delta_rolls": float(best_time),
                "scheduling_score": float(best_time),
                "display_source": display_source,
                "raw_breakdown_score": raw_breakdown_score,

                # "strategy": best_activity,
                "strategy": display_strategy,
                "requested_activity": requested_activity,
                "predicted_extra_roads": predicted_extra_roads,
                "event_times": dict(event_times),

                # v015/v023 scheduling source used for JUMP ordering.
                # event_times / markov_event_times remain Markov/light for
                # comparison and backward compatibility.
                "scheduling_source": scheduling_source,
                "scheduling_event_times": dict(scheduling_event_times),
                "chosen_scheduling_score": float(best_time),

                # v015 expected-hand comparison
                "estimator_primary": estimator_primary,
                "markov_event_times": dict(markov_event_times_for_actions),
                "expected_hand_event_times": dict(expected_hand_event_times_public),
                "expected_hand_debug_keys": list(expected_hand_debug_keys),
                "expected_hand_comparison_rows": list(expected_hand_comparison_rows),
                "chosen_expected_hand_comparison": dict(chosen_expected_hand_comparison),
                "chosen_markov_score": chosen_markov_score_for_display,
                "chosen_expected_hand_score": chosen_expected_hand_score_for_display,
                "chosen_expected_hand_delta": chosen_expected_hand_comparison.get("expected_hand_delta"),
                "chosen_expected_hand_confidence": chosen_expected_hand_comparison.get("expected_hand_confidence"),
                "chosen_expected_hand_confidence_label": chosen_expected_hand_comparison.get("expected_hand_confidence_label"),
                "chosen_expected_hand_found": chosen_expected_hand_comparison.get("expected_hand_found"),

                # v014 expected viable actions, now enriched above with v015 EH metadata
                "expected_viable_actions": expected_viable_actions,
                "expected_viable_codes": expected_viable_codes,

                # Existing / backward-compatible fields
                "mode": mode,
                "refine_reasons": refine_reasons,
                "light_reused": light_reused,

                # New transparency fields
                "used_heavy": used_heavy,
                "source_mode": source_mode,
                "focus_activities": focus_activities,
                "best_activity_changed": bool(refine_meta.get("best_activity_changed", False)),
                "changed_activities": changed_activities,
                "activity_improvements": dict(refine_meta.get("activity_improvements", {}) or {}),
                "best_activity_light": refine_meta.get("best_activity_light"),
                "best_time_light": _safe_float(refine_meta.get("best_time_light", 9999.0), 9999.0),
                "second_best_activity": second_activity,
                "second_best_time": float(second_time),
                "light_gap_to_second": _safe_float(refine_meta.get("light_gap_to_second", 9999.0), 9999.0),
                "final_gap_to_second": _safe_float(refine_meta.get("final_gap_to_second", 9999.0), 9999.0),

                # Explanation / exact-plan visibility
                "chosen_breakdown": chosen_breakdown,
                "chosen_explanation": chosen_expl,
                "overflow_summary": overflow_summary,
                "prediction_plan_summary": chosen_plan_summary,
                "final_best_activity_summary": dict(refine_meta.get("final_best_activity_summary", {}) or {}),
                "final_settlement_summary": dict(refine_meta.get("final_settlement_summary", {}) or {}),
            })

        final_rows.sort(key=_sort_key)
        return final_rows

    def _print_prediction_rows(self, rows: List[Dict[str, Any]], title: str = "Fast-forward prediction table") -> None:
        """
        Print prediction rows with the actual JUMP/PLAY decision contract visible.

        Shows:
        - predicted round / turn
        - displayed strategy, e.g. settlement_1r
        - requested execution activity, e.g. new_settlement
        - predicted road count
        - predicted target / upgrade / dev-card target
        - source mode and heavy flag
        - compact overflow summary
        """
        if not rows:
            print("FastForward prediction table: no rows")
            return

        def _short(value: Any, width: int) -> str:
            txt = str(value)
            if len(txt) <= width:
                return txt
            return txt[: max(0, width - 3)] + "..."

        def _target_text(row: Dict[str, Any]) -> str:
            plan = dict(row.get("prediction_plan_summary", {}) or {})
            strategy = str(row.get("strategy", "") or "")

            if "chosen_target" in plan:
                return str(plan.get("chosen_target"))
            if "chosen_upgrade" in plan:
                return str(plan.get("chosen_upgrade"))
            if plan.get("target") is not None:
                return str(plan.get("target"))

            if strategy.startswith("settlement_"):
                return str(plan.get("predicted_chosen_target", "?"))
            if row.get("requested_activity") == "upgrade_to_city":
                return str(plan.get("chosen_upgrade", "?"))
            if row.get("requested_activity") == "buy_discovery_card":
                return "dev"

            return "-"

        def _roads_text(row: Dict[str, Any]) -> str:
            plan = dict(row.get("prediction_plan_summary", {}) or {})
            val = row.get("predicted_extra_roads", None)

            if val is None:
                val = plan.get("predicted_extra_roads", None)
            if val is None:
                val = plan.get("extra_roads_needed", None)

            if val is None:
                return "-"

            try:
                return str(int(val))
            except Exception:
                return str(val)

        def _pay_text(row: Dict[str, Any]) -> str:
            plan = dict(row.get("prediction_plan_summary", {}) or {})
            pay = plan.get("pay_mode") or plan.get("pay") or plan.get("payable_mode")
            if pay:
                return str(pay)

            # Prediction rows often do not know exact payment yet; JUMP builds that.
            if row.get("requested_activity") in ("new_settlement", "upgrade_to_city", "buy_discovery_card"):
                return "at_jump"

            return "-"

        def _overflow_text(row: Dict[str, Any]) -> str:
            overflow = dict(row.get("overflow_summary", {}) or {})
            if not overflow.get("triggered", False):
                return "-"

            res = overflow.get("resource", "?")
            try:
                pips = float(overflow.get("pips", 0.0) or 0.0)
            except Exception:
                pips = 0.0

            rate = overflow.get("rate", "?")
            trades = overflow.get("trades", "?")
            can_fund = overflow.get("can_fund", False)
            cap_bind = overflow.get("cap_bind", False)

            return f"{res}@{pips:.1f} {rate}:1 T{trades} fund={can_fund} cap={cap_bind}"


        def _fmt_time(value: Any) -> str:
            try:
                if value is None:
                    return "-"
                value = float(value)
                if value >= 9999.0:
                    return "9999"
                return f"{value:.2f}"
            except Exception:
                return "-"

        def _fmt_delta(value: Any) -> str:
            try:
                if value is None:
                    return "-"
                value = float(value)
                if value >= 9999.0:
                    return "+9999"
                return f"{value:+.2f}"
            except Exception:
                return "-"

        def _fmt_conf(value: Any) -> str:
            try:
                if value is None:
                    return "-"
                return f"{float(value):.2f}"
            except Exception:
                return "-"

        def _source_text(row: Dict[str, Any]) -> str:
            """
            Display the actual JUMP scheduling source first.

            source_mode/mode still describe the old Markov light/heavy pipeline.
            scheduling_source/display_source describe what selected the current
            best_activity/best_time.
            """
            src = (
                row.get("display_source")
                or row.get("scheduling_source")
                or row.get("estimator_primary")
                or row.get("source_mode")
                or row.get("mode")
                or "light"
            )

            # Keep naming aligned with PLAY logs.
            if str(src) == "expected_hand":
                src = "expected_hand_primary"

            return str(src)

        print(f"\n================ {title.upper()} ================")
        print(
            f"{'Player':<10} "
            f"{'Round':>5} "
            f"{'Turn':>4} "
            f"{'Δ':>7} "
            f"{'Strategy':<16} "
            f"{'Opts':<7} "
            f"{'Exec':<16} "
            f"{'Roads':>5} "
            f"{'Target':>6} "
            f"{'Pay':<10} "
            f"{'Src':<20} "
            f"{'Heavy':<5} "
            f"{'Overflow':<34}"
        )
        print("-" * 167)

        for row in rows:
            player_id = row.get("player_id", "?")
            player_color = row.get("player_color", "?")
            pred_round = row.get("pred_round", "?")
            pred_turn = row.get("pred_turn", "?")
            delta = row.get("delta_rolls", 9999.0)

            strategy = row.get("strategy", "?")
            requested_activity = row.get("requested_activity", strategy)

            source_mode = _source_text(row)
            used_heavy = bool(row.get("used_heavy", False))

            try:
                delta_txt = f"{float(delta):.2f}"
            except Exception:
                delta_txt = "?"

            options_txt = str(row.get("expected_viable_codes", "-") or "-")

            print(
                f"{_short(str(player_id) + ':' + str(player_color), 10):<10} "
                f"{str(pred_round):>5} "
                f"{str(pred_turn):>4} "
                f"{delta_txt:>7} "
                f"{_short(strategy, 16):<16} "
                f"{_short(options_txt, 7):<7} "
                f"{_short(requested_activity, 16):<16} "
                f"{_roads_text(row):>5} "
                f"{_target_text(row):>6} "
                f"{_short(_pay_text(row), 10):<10} "
                f"{_short(source_mode, 20):<20} "
                f"{str(used_heavy):<5} "
                f"{_short(_overflow_text(row), 34):<34}"
            )

        print("-" * 167)

        # Detailed row diagnostics
        for row in rows:
            player_id = row.get("player_id", "?")
            strategy = row.get("strategy", "?")
            requested_activity = row.get("requested_activity", strategy)
            plan_summary = dict(row.get("prediction_plan_summary", {}) or {})
            chosen_expl = dict(row.get("chosen_explanation", {}) or {})
            overflow = dict(row.get("overflow_summary", {}) or {})

            print(f"\nPlayer {player_id} | strategy={strategy} | exec={requested_activity}")

            print(
                f"   prediction_contract="
                f"round={row.get('pred_round', '?')} "
                f"turn={row.get('pred_turn', '?')} "
                f"delta={row.get('delta_rolls', '?')} "
                f"roads={_roads_text(row)} "
                f"target={_target_text(row)} "
                f"source={_source_text(row)} "
                f"heavy={bool(row.get('used_heavy', False))}"
            )

            if plan_summary:
                compact_plan = {
                    k: plan_summary.get(k)
                    for k in [
                        "activity",
                        "requested_activity",
                        "strategy",
                        "settlement_strategy",
                        "score",
                        "raw_breakdown_score",
                        "scheduling_source",
                        "scheduling_score",
                        "markov_score",
                        "expected_hand_score",                        
                        "chosen_target",
                        "chosen_upgrade",
                        "extra_roads_needed",
                        "predicted_extra_roads",
                        "settlement_target_type",
                        "target",
                        "pay_mode",
                    ]
                    if k in plan_summary
                }
                print(f"   plan_summary={compact_plan}")

            # v014: show all expected viable actions for this player.
            # These are Markov/expected viable actions, not final exact PLAY viability.
            expected_viable_actions = list(row.get("expected_viable_actions", []) or [])

            if expected_viable_actions:
                print("   expected_viable_actions:")

                for action in expected_viable_actions:
                    code = action.get("code", "?")
                    activity = action.get("activity", action.get("requested_activity", "?"))

                    markov_score = action.get("markov_score", action.get("score"))
                    eh_score = action.get("expected_hand_score")
                    eh_delta = action.get("expected_hand_delta")
                    eh_conf = action.get("expected_hand_confidence")
                    eh_label = action.get("expected_hand_confidence_label")

                    # This is not necessarily the PLAY ranking source yet; before PLAY it
                    # usually reflects the current row's primary estimator. The PLAY
                    # decision log remains the source of truth for actual ranking override.
                    timing_source = action.get(
                        "timing_source_for_ranking",
                        action.get(
                            "timing_primary_source",
                            row.get("estimator_primary", row.get("source_mode", row.get("mode", "-"))),
                        ),
                    )

                    # v015: display-name normalization.
                    # Older/stale action metadata may still say "expected_hand".
                    # PLAY ranking uses "expected_hand_primary", so print the same label.
                    if str(timing_source) == "expected_hand":
                        timing_source = "expected_hand_primary"

                    target = action.get("target", action.get("chosen_target", "-"))
                    if target is None or str(target).lower() == "none" or str(target) == "":
                        target = "-"

                    roads_raw = action.get("extra_roads_needed", action.get("roads", None))
                    if roads_raw is None or str(roads_raw).lower() == "none" or str(roads_raw) == "":
                        roads = "-"
                    else:
                        roads = roads_raw

                    print(
                        f"      {code} {str(activity):<18} "
                        f"M={_fmt_time(markov_score):>6} "
                        f"EH={_fmt_time(eh_score):>6} "
                        f"Δ={_fmt_delta(eh_delta):>7} "
                        f"conf={_fmt_conf(eh_conf):>5} "
                        f"{str(eh_label or '-'):>6} "
                        f"timing={str(timing_source or '-'):<18} "
                        f"target={target} roads={roads}"
                    )
            if overflow.get("triggered", False):
                print(f"   overflow_summary={overflow}")

            if chosen_expl:
                compact_keys = [
                    "normalized_target_type",
                    "extra_roads_needed",
                    "settlement_target_type",
                    "overflow_triggered",
                    "overflow_dominant_resource",
                    "overflow_dominant_resource_pips",
                    "overflow_effective_trade_rate",
                    "overflow_needed_trades",
                    "overflow_expected_dominant_cards_by_horizon",
                    "overflow_can_fund_within_horizon",
                    "overflow_cap_bind_risk",
                ]

                compact_expl = {
                    k: chosen_expl.get(k)
                    for k in compact_keys
                    if k in chosen_expl
                }

                if compact_expl:
                    print(f"   chosen_explanation={compact_expl}")

        print("===============================================================\n")

    def _estimate_predicted_round_turn(self, delta_rolls: float, player_sequence: int) -> Tuple[object, object]:
        """
        Convert a Markov delta into the first legal future turn for this player.

        Markov delta is interpreted as continuous rounds from the current
        fast-forward time. The actual action can only happen on the player's
        own turn, so we schedule to the first matching player slot at or after
        the predicted ready time.
        """
        if delta_rolls is None:
            return "?", "?"

        try:
            delta = float(delta_rolls)
        except Exception:
            return "?", "?"

        if delta >= 9999.0:
            return "?", "?"

        num_players = max(len(getattr(self.game, "players", [])), 1)

        try:
            player_seq = int(player_sequence)
        except Exception:
            return "?", "?"

        player_seq = max(1, min(num_players, player_seq))
        player_slot = player_seq - 1

        # Current calendar position as continuous round time.
        # round 1 turn 1 => 0.00
        # round 1 turn 2 => 0.25 in a 4-player game
        # round 1 turn 3 => 0.50
        # round 1 turn 4 => 0.75
        current_round = max(1, int(getattr(self.game, "round", 1)))
        current_turn = max(1, int(getattr(self.game, "turn", 1)))
        current_turn = max(1, min(num_players, current_turn))

        now_abs = float(current_round - 1) + float(current_turn - 1) / float(num_players)
        ready_abs = now_abs + max(0.0, delta)

        # Convert ready_abs to the first player-slot index at or after ready_abs.
        eps = 1e-9
        min_slot_index = int(math.ceil((ready_abs * num_players) - eps))

        # Advance to this player's next legal turn slot.
        while (min_slot_index % num_players) != player_slot:
            min_slot_index += 1

        predicted_round = (min_slot_index // num_players) + 1
        predicted_turn = player_seq

        return predicted_round, predicted_turn
    
    def print_fast_forward_table(
        self,
        header: str = "Fast-forward table after JUMP (nearest event at top)"
    ) -> List[Dict[str, Any]]:
        """
        Build and print the unified fast-forward prediction table.
        """
        rows = self._build_prediction_rows()

        # TEMP v014 debug overlay:
        # keep the latest prediction rows available to the GUI.
        self.game.ff_debug_prediction_rows = list(rows)

        self._print_prediction_rows(rows, header)
        return rows

    def print_fast_forward_table_settlement_only(
        self,
        header: str = "Fast-forward table after Initial Placement (settlement-only, nearest at top)"
    ) -> List[Dict[str, Any]]:
        """
        Print a settlement-only snapshot table from the final setup position.

        IMPORTANT:
        Sort chronologically:
            (predicted_round, predicted_turn, delta_turns)
        """
        rows: List[Dict[str, Any]] = []

        for player in self.game.players:
            hand = player.rcards_in_hand()[0]
            ports = self.game.get_player_ports_dict(player)
            vertices = list(player.settlements) + list(player.cities)

            if not vertices:
                delta_rolls = 9999.0
            elif hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                delta_rolls = float(
                    self.game.markov.get_expected_turns_fast_initial(
                        vertices=vertices,
                        hand=hand,
                        player_ports=ports,
                        strategy="settlement",
                    )
                )
            else:
                delta_rolls = 9999.0

            pred_round, pred_turn = self._estimate_predicted_round_turn(delta_rolls, player.sequence)

            rows.append({
                "player_id": player.id,
                "player_color": player.color,
                "player_sequence": player.sequence,
                "pred_round": pred_round,
                "pred_turn": pred_turn,
                "delta_rolls": delta_rolls,
                "strategy": "new_settlement",
                "event_times": {"new_settlement": delta_rolls},
            })

        def _sort_key(r: Dict[str, Any]):
            pr = r["pred_round"]
            pt = r["pred_turn"]

            if pr == "?" or pt == "?":
                return (999999, 999999, float(r["delta_rolls"]))

            return (int(pr), int(pt), float(r["delta_rolls"]))

        rows.sort(key=_sort_key)
        self._print_prediction_rows(rows, header)
        return rows
    
    def _estimate_min_extra_roads_to_any_settlement(self, player: Player) -> tuple[int, int | None]:
        """
        Estimate how many additional roads are needed to reach the nearest
        structurally valid future settlement target.

        Returns:
            (min_extra_roads_needed, example_target_intersection)

        Logic:
        - Candidate target must satisfy:
            * not in water
            * not occupied
            * distance rule against all existing settlements/cities
        - Path cost:
            * 0 for already-owned road
            * 1 for empty road that could eventually be built
            * blocked for opponent-occupied road
        """
        from collections import deque

        board = self.game.board

        occupied_vertices: list[int] = []
        for p in self.game.players:
            occupied_vertices.extend(list(getattr(p, "settlements", [])))
            occupied_vertices.extend(list(getattr(p, "cities", [])))

        candidates: list[int] = []
        for inter in board.intersections:
            if inter is None:
                continue

            iid = inter.id

            if iid in getattr(board, "INTERSECTION_IN_WATER", []):
                continue
            if getattr(inter, "occupied_tf", False):
                continue

            too_close = False
            for existing_id in occupied_vertices:
                try:
                    dist = board._distance_between_intersections(iid, existing_id)
                except Exception:
                    dist = 999
                if dist <= 1:
                    too_close = True
                    break

            if not too_close:
                candidates.append(iid)

        if not candidates:
            return (9999, None)

        start_vertices = set(player.settlements + player.cities)
        for road in getattr(player, "roads", []):
            if isinstance(road, tuple) and len(road) == 2:
                start_vertices.add(road[0])
                start_vertices.add(road[1])

        if not start_vertices:
            return (9999, None)

        INF = 10**9
        dist_map = {v.id: INF for v in board.intersections if v is not None}
        dq = deque()

        for s in start_vertices:
            if s in dist_map:
                dist_map[s] = 0
                dq.appendleft(s)

        def _road_obj(road_id: tuple[int, int]):
            for r in board.roads:
                if r and tuple(sorted(r.id)) == tuple(sorted(road_id)):
                    return r
            return None

        while dq:
            cur = dq.popleft()
            cur_dist = dist_map[cur]
            inter = board.intersections[cur]
            if inter is None:
                continue

            for road_tuple in getattr(inter, "three_roads", []):
                road_id = tuple(sorted(road_tuple))
                other = road_id[0] if road_id[1] == cur else road_id[1]

                if other not in dist_map:
                    continue

                road = _road_obj(road_id)

                if road is not None and getattr(road, "occupied_tf", False):
                    if getattr(road, "color", None) != player.color:
                        continue
                    weight = 0
                else:
                    weight = 1

                nd = cur_dist + weight
                if nd < dist_map[other]:
                    dist_map[other] = nd
                    if weight == 0:
                        dq.appendleft(other)
                    else:
                        dq.append(other)

        best_target = None
        best_cost = 9999

        for cand in candidates:
            cand_cost = dist_map.get(cand, INF)
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_target = cand

        if best_cost >= INF:
            return (9999, None)

        return (int(best_cost), best_target)

    def _get_event_times_for_player_light(self, player: Player) -> Dict[str, float]:
        """
        Return predicted LIGHT Markov time-to-event for the player.

        Light policy:
        - do NOT call markov.get_expected_time_to_event(...)
        - use the fast scorer / fast explanation path only
        - keep the same breakdown structure used by prediction printing
        - preserve richer explanation data so _should_refine_with_heavy(...)
        can detect overflow-risk, settlement ambiguity, and near-tie context

        Strategy set exposed here:
            * new_settlement        -> mapped to settlement_0r / settlement_1r / settlement_2r
            * upgrade_to_city
            * buy_discovery_card
        """
        hand = player.rcards_in_hand()[0]
        ports = self.game.get_player_ports_dict(player)
        base_vertices = list(player.settlements) + list(player.cities)

        if not hasattr(self, "_last_strategy_breakdown"):
            self._last_strategy_breakdown = {}

        self._last_strategy_breakdown[player.id] = {}

        if not base_vertices:
            return {
                "new_settlement": 9999.0,
                "upgrade_to_city": 9999.0,
                "buy_discovery_card": 9999.0,
            }

        outlook = self._ensure_outlook(player)

        # ------------------------------------------------------------
        # Structural caps
        # ------------------------------------------------------------
        current_total_settlement_sites = len(player.settlements) + len(player.cities)
        can_have_more_settlements = current_total_settlement_sites < 5
        can_have_more_cities = len(player.cities) < 4

        # ------------------------------------------------------------
        # Settlement planning:
        # estimate nearest road burden and map it to one of the Markov
        # settlement targets:
        #   settlement_0r / settlement_1r / settlement_2r
        # ------------------------------------------------------------
        extra_roads_needed, settlement_target = self._estimate_min_extra_roads_to_any_settlement(player)

        settlement_target_type = None
        if can_have_more_settlements and extra_roads_needed == 0:
            settlement_target_type = "settlement_0r"
        elif can_have_more_settlements and extra_roads_needed == 1:
            settlement_target_type = "settlement_1r"
        elif can_have_more_settlements and extra_roads_needed == 2:
            settlement_target_type = "settlement_2r"

        can_new = settlement_target_type is not None

        # ------------------------------------------------------------
        # City candidates
        # ------------------------------------------------------------
        city_candidates = outlook.get_viable_city_upgrades() if can_have_more_cities else []
        can_city = bool(city_candidates)

        # ------------------------------------------------------------
        # Dev card
        # ------------------------------------------------------------
        can_dev = self._can_buy_dev_card(player)

        event_times: Dict[str, float] = {
            "new_settlement": 9999.0,
            "upgrade_to_city": 9999.0,
            "buy_discovery_card": 9999.0,
        }

        # ------------------------------------------------------------
        # 1) Settlement score = LIGHT fast scorer
        # ------------------------------------------------------------
        if can_new:
            if hasattr(self.game.markov, "get_expected_turns_fast_initial_with_explanation"):
                explain = self.game.markov.get_expected_turns_fast_initial_with_explanation(
                    vertices=base_vertices,
                    hand=hand,
                    player_ports=ports,
                    strategy="settlement",
                    extra_roads_needed=extra_roads_needed,
                )
                explain = dict(explain)
                # settlement_score = float(explain.get("score", settlement_score))
                settlement_score = float(explain.get("score", 9999.0))
                explain["score"] = settlement_score

                # These 3 values are the important new prediction contract.
                explain["chosen_target"] = settlement_target
                explain["extra_roads_needed"] = int(extra_roads_needed)
                explain["settlement_target_type"] = settlement_target_type

                explain["predicted_chosen_target"] = settlement_target
                explain["predicted_extra_roads_needed"] = int(extra_roads_needed)
                explain["predicted_settlement_target_type"] = settlement_target_type

                explain["heavy_mode"] = False

                if "explanation" not in explain or not isinstance(explain.get("explanation"), dict):
                    explain["explanation"] = {}

                explain["explanation"]["chosen_target"] = settlement_target
                explain["explanation"]["extra_roads_needed"] = int(extra_roads_needed)
                explain["explanation"]["settlement_target_type"] = settlement_target_type

                explain["explanation"]["predicted_chosen_target"] = settlement_target
                explain["explanation"]["predicted_extra_roads_needed"] = int(extra_roads_needed)
                explain["explanation"]["predicted_settlement_target_type"] = settlement_target_type

            elif hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                settlement_score = float(
                    self.game.markov.get_expected_turns_fast_initial(
                        vertices=base_vertices,
                        hand=hand,
                        player_ports=ports,
                        strategy="settlement",
                        extra_roads_needed=extra_roads_needed,
                    )
                )
                explain = {
                    "score": settlement_score,
                    "chosen_target": settlement_target,
                    "settlement_target_type": settlement_target_type,
                    "heavy_mode": False,
                    "explanation": {
                        "extra_roads_needed": int(extra_roads_needed),
                        "chosen_target": settlement_target,
                        "settlement_target_type": settlement_target_type,
                    },
                }

            else:
                settlement_score = 9999.0
                explain = {
                    "score": settlement_score,
                    "chosen_target": settlement_target,
                    "settlement_target_type": settlement_target_type,
                    "heavy_mode": False,
                    "explanation": {
                        "extra_roads_needed": int(extra_roads_needed),
                        "chosen_target": settlement_target,
                        "settlement_target_type": settlement_target_type,
                    },
                }

            # event_times["new_settlement"] = settlement_score
            # self._last_strategy_breakdown[player.id]["new_settlement"] = explain

            # ------------------------------------------------------------
            # Store precise settlement subtype:
            # settlement_0r / settlement_1r / settlement_2r
            # ------------------------------------------------------------
            extra_roads_needed = int(extra_roads_needed)

            if extra_roads_needed < 0:
                extra_roads_needed = 0
            if extra_roads_needed > 2:
                extra_roads_needed = 2

            settlement_strategy = f"settlement_{extra_roads_needed}r"

            settlement_score = float(explain.get("score", 9999.0))
            explain["score"] = settlement_score

            # Canonical activity used by execution
            explain["activity"] = "new_settlement"

            # Precise Markov strategy used by prediction
            explain["settlement_strategy"] = settlement_strategy
            explain["strategy"] = settlement_strategy

            # Road-count prediction contract
            explain["chosen_target"] = settlement_target
            explain["extra_roads_needed"] = extra_roads_needed
            explain["settlement_target_type"] = settlement_strategy

            explain["predicted_chosen_target"] = settlement_target
            explain["predicted_extra_roads_needed"] = extra_roads_needed
            explain["predicted_settlement_target_type"] = settlement_strategy

            explain["heavy_mode"] = False

            if "explanation" not in explain or not isinstance(explain.get("explanation"), dict):
                explain["explanation"] = {}

            explain["explanation"]["activity"] = "new_settlement"
            explain["explanation"]["settlement_strategy"] = settlement_strategy
            explain["explanation"]["strategy"] = settlement_strategy

            explain["explanation"]["chosen_target"] = settlement_target
            explain["explanation"]["extra_roads_needed"] = extra_roads_needed
            explain["explanation"]["settlement_target_type"] = settlement_strategy

            explain["explanation"]["predicted_chosen_target"] = settlement_target
            explain["explanation"]["predicted_extra_roads_needed"] = extra_roads_needed
            explain["explanation"]["predicted_settlement_target_type"] = settlement_strategy

            # Keep old canonical event key for execution/table compatibility.
            # event_times["new_settlement"] = settlement_score
            event_times["new_settlement"] = settlement_score + 0.000001

            # Add precise subtype key for strategy selection / validation.
            event_times[settlement_strategy] = settlement_score

            # Store both lookup keys, pointing to the same explanation.
            self._last_strategy_breakdown[player.id]["new_settlement"] = dict(explain)
            self._last_strategy_breakdown[player.id][settlement_strategy] = dict(explain)

        # ------------------------------------------------------------
        # 2) City score = LIGHT fast scorer per upgrade target
        # ------------------------------------------------------------
        if can_city:
            best_city_score = 9999.0
            best_city_out = None

            for inter_id in city_candidates:
                city_vertices = list(player.settlements) + list(player.cities) + [inter_id]

                if hasattr(self.game.markov, "get_expected_turns_fast_initial_with_explanation"):
                    out = self.game.markov.get_expected_turns_fast_initial_with_explanation(
                        vertices=city_vertices,
                        hand=hand,
                        player_ports=ports,
                        strategy="city",
                        extra_roads_needed=0,
                    )
                    out = dict(out)
                    score = float(out.get("score", 9999.0))
                    out["score"] = score
                    out["chosen_upgrade"] = inter_id
                    out["heavy_mode"] = False

                    if "explanation" not in out or not isinstance(out.get("explanation"), dict):
                        out["explanation"] = {}
                    out["explanation"]["chosen_upgrade"] = inter_id

                elif hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                    score = float(
                        self.game.markov.get_expected_turns_fast_initial(
                            vertices=city_vertices,
                            hand=hand,
                            player_ports=ports,
                            strategy="city",
                            extra_roads_needed=0,
                        )
                    )
                    out = {
                        "score": score,
                        "chosen_upgrade": inter_id,
                        "heavy_mode": False,
                        "explanation": {
                            "chosen_upgrade": inter_id,
                        },
                    }

                else:
                    score = 9999.0
                    out = {
                        "score": score,
                        "chosen_upgrade": inter_id,
                        "heavy_mode": False,
                        "explanation": {
                            "chosen_upgrade": inter_id,
                        },
                    }

                if score < best_city_score:
                    best_city_score = score
                    best_city_out = out

            event_times["upgrade_to_city"] = best_city_score
            self._last_strategy_breakdown[player.id]["upgrade_to_city"] = best_city_out

        # ------------------------------------------------------------
        # 3) Dev-card score = LIGHT fast scorer
        # ------------------------------------------------------------
        if can_dev:
            if hasattr(self.game.markov, "get_expected_turns_fast_initial_with_explanation"):
                out = self.game.markov.get_expected_turns_fast_initial_with_explanation(
                    vertices=base_vertices,
                    hand=hand,
                    player_ports=ports,
                    strategy="dev_card",
                    extra_roads_needed=0,
                )
                out = dict(out)
                dev_score = float(out.get("score", 9999.0))
                out["score"] = dev_score
                out["heavy_mode"] = False

                if "explanation" not in out or not isinstance(out.get("explanation"), dict):
                    out["explanation"] = {}

            elif hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                dev_score = float(
                    self.game.markov.get_expected_turns_fast_initial(
                        vertices=base_vertices,
                        hand=hand,
                        player_ports=ports,
                        strategy="dev_card",
                        extra_roads_needed=0,
                    )
                )
                out = {
                    "score": dev_score,
                    "heavy_mode": False,
                    "explanation": {},
                }

            else:
                dev_score = 9999.0
                out = {
                    "score": dev_score,
                    "heavy_mode": False,
                    "explanation": {},
                }

            event_times["buy_discovery_card"] = dev_score
            self._last_strategy_breakdown[player.id]["buy_discovery_card"] = out

        return event_times
    
    def _should_refine_with_heavy(
        self,
        player: Player,
        light_event_times: Dict[str, float],
        light_rows_so_far: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Decide whether this player's LIGHT prediction should be refined with heavier reasoning.

        Triggers:
        - post-failure settlement re-estimate
        - near tie between best and second-best activities
        - overflow-risk in LIGHT explanation
        - settlement ambiguity / road-burden ambiguity
        """
        reasons: List[str] = []
        focus_activities: List[str] = []

        valid_items: List[Tuple[str, float]] = []
        for k, v in (light_event_times or {}).items():
            try:
                fv = float(v)
            except Exception:
                continue
            if fv < 9999.0:
                valid_items.append((k, fv))

        if not valid_items:
            # return {
            #     "refine": False,
            #     "reasons": [],
            #     "best_activity": None,
            #     "best_time": 9999.0,
            #     "second_best_activity": None,
            #     "second_best_time": 9999.0,
            #     "focus_activities": [],
            # }
            return {
                "refine": False,
                "reasons": [],
                "best_activity": best_activity,
                "best_time": float(best_time),
                "second_best_activity": second_best_activity,
                "second_best_time": float(second_best_time),
                "focus_activities": [],
            }

        valid_items.sort(key=lambda x: x[1])
        best_activity, best_time = valid_items[0]
        focus_activities.append(best_activity)

        if len(valid_items) >= 2:
            second_best_activity, second_best_time = valid_items[1]
        else:
            second_best_activity, second_best_time = None, 9999.0

        # ------------------------------------------------------------
        # MUST-refine: post-failure settlement re-estimate
        # ------------------------------------------------------------
        last_actor_id = getattr(self.game, "ff_last_actor_id", None)
        last_requested = getattr(self.game, "ff_last_requested_activity", None)
        last_details = getattr(self.game, "ff_last_details", {}) or {}

        post_failure_settlement_reestimate = (
            last_actor_id == player.id
            and last_requested == "new_settlement"
            and (
                bool(last_details.get("reestimate_required", False))
                or str(last_details.get("retry_strategy", "")) == "new_settlement"
            )
        )

        if post_failure_settlement_reestimate:
            return {
                "refine": True,
                "reasons": ["post_failure_settlement_reestimate"],
                "best_activity": best_activity,
                "best_time": float(best_time),
                "second_best_activity": second_best_activity,
                "second_best_time": float(second_best_time),
                "focus_activities": ["new_settlement"],
            }

        # ------------------------------------------------------------
        # Read LIGHT strategy breakdown / explanation data
        # ------------------------------------------------------------
        breakdown = getattr(self, "_last_strategy_breakdown", {}).get(player.id, {}) or {}

        best_info = breakdown.get(best_activity, {}) or {}
        best_expl = best_info.get("explanation", {}) or {}

        settlement_info = breakdown.get("new_settlement", {}) or {}
        settlement_expl = settlement_info.get("explanation", {}) or {}

        # ------------------------------------------------------------
        # Trigger A: near tie
        # ------------------------------------------------------------
        if second_best_activity is not None and second_best_time < 9999.0:
            gap = float(second_best_time) - float(best_time)

            if gap <= 0.50:
                reasons.append("near_tie<=0.50")
                if second_best_activity not in focus_activities:
                    focus_activities.append(second_best_activity)

            elif best_activity == "new_settlement" and gap <= 0.75:
                reasons.append("settlement_near_tie<=0.75")
                if second_best_activity not in focus_activities:
                    focus_activities.append(second_best_activity)

        # ------------------------------------------------------------
        # Trigger B: overflow-risk on best activity
        # ------------------------------------------------------------
        overflow_triggered = bool(best_expl.get("overflow_triggered", False))
        overflow_cap_bind = bool(best_expl.get("overflow_cap_bind_risk", False))
        overflow_can_fund = bool(best_expl.get("overflow_can_fund_within_horizon", False))
        overflow_weak_off = bool(best_expl.get("overflow_weak_off_resource_exists", False))
        overflow_needed_trades = int(best_expl.get("overflow_needed_trades", 0) or 0)

        if (
            overflow_triggered
            and overflow_cap_bind
            and overflow_can_fund
            and (overflow_weak_off or overflow_needed_trades >= 2)
        ):
            reasons.append("overflow_risk_best_activity")

        # ------------------------------------------------------------
        # Trigger C: settlement ambiguity / settlement overflow
        # ------------------------------------------------------------
        try:
            settlement_score = float(
                light_event_times.get(
                    "new_settlement",
                    settlement_info.get("score", 9999.0),
                )
            )
        except Exception:
            settlement_score = 9999.0

        settlement_extra_roads = int(settlement_expl.get("extra_roads_needed", 0) or 0)
        settlement_overflow_triggered = bool(settlement_expl.get("overflow_triggered", False))
        settlement_overflow_cap_bind = bool(settlement_expl.get("overflow_cap_bind_risk", False))
        settlement_overflow_can_fund = bool(settlement_expl.get("overflow_can_fund_within_horizon", False))
        settlement_needed_trades = int(settlement_expl.get("overflow_needed_trades", 0) or 0)

        if settlement_score < 9999.0:
            if best_activity == "new_settlement" and settlement_extra_roads >= 1:
                reasons.append("settlement_requires_extra_roads")

            if (
                best_activity != "new_settlement"
                and second_best_activity == "new_settlement"
                and (float(second_best_time) - float(best_time) <= 0.75)
            ):
                reasons.append("settlement_competitive")
                if "new_settlement" not in focus_activities:
                    focus_activities.append("new_settlement")

            if (
                settlement_overflow_triggered
                and settlement_overflow_cap_bind
                and settlement_overflow_can_fund
                and settlement_needed_trades >= 2
            ):
                reasons.append("settlement_overflow_risk")
                if "new_settlement" not in focus_activities:
                    focus_activities.append("new_settlement")

        # ------------------------------------------------------------
        # Trigger D: table race near tie
        # ------------------------------------------------------------
        if light_rows_so_far:
            other_deltas = []
            for r in light_rows_so_far:
                try:
                    dv = float(r.get("delta_rolls", 9999.0))
                except Exception:
                    continue
                if dv < 9999.0:
                    other_deltas.append(dv)

            if other_deltas:
                nearest_other = min(other_deltas)
                if abs(nearest_other - float(best_time)) <= 0.50:
                    reasons.append("table_race_near_tie")

        # ------------------------------------------------------------
        # Final
        # ------------------------------------------------------------
        dedup_focus: List[str] = []
        for act in focus_activities:
            if act and act not in dedup_focus:
                dedup_focus.append(act)

        return {
            "refine": False,
            "reasons": [],
            "best_activity": None,
            "best_time": 9999.0,
            "second_best_activity": None,
            "second_best_time": 9999.0,
            "focus_activities": [],
        }
    
    def _get_event_times_for_player_heavy(
        self,
        player: Player,
        focus_activities: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Return predicted HEAVY Markov time-to-event for the player.

        Heavy policy:
        - use markov.get_expected_time_to_event(...)
        - compute only the requested focus activities when provided
        - preserve the same breakdown structure as LIGHT
        - mark updated activity rows with heavy_mode=True
        """
        hand = player.rcards_in_hand()[0]
        ports = self.game.get_player_ports_dict(player)
        base_vertices = list(player.settlements) + list(player.cities)

        if focus_activities is None:
            focus_activities = [
                "new_settlement",
                "upgrade_to_city",
                "buy_discovery_card",
            ]

        focus_set = set(focus_activities)

        if not hasattr(self, "_last_strategy_breakdown"):
            self._last_strategy_breakdown = {}
        if player.id not in self._last_strategy_breakdown:
            self._last_strategy_breakdown[player.id] = {}

        event_times = {
            "new_settlement": 9999.0,
            "upgrade_to_city": 9999.0,
            "buy_discovery_card": 9999.0,
        }

        if not base_vertices:
            return event_times

        if not hasattr(self.game.markov, "get_expected_time_to_event"):
            return event_times

        outlook = self._ensure_outlook(player)

        current_total_settlement_sites = len(player.settlements) + len(player.cities)
        can_have_more_settlements = current_total_settlement_sites < 5
        can_have_more_cities = len(player.cities) < 4

        extra_roads_needed, settlement_target = self._estimate_min_extra_roads_to_any_settlement(player)

        settlement_target_type = None
        if can_have_more_settlements and extra_roads_needed == 0:
            settlement_target_type = "settlement_0r"
        elif can_have_more_settlements and extra_roads_needed == 1:
            settlement_target_type = "settlement_1r"
        elif can_have_more_settlements and extra_roads_needed == 2:
            settlement_target_type = "settlement_2r"

        can_new = settlement_target_type is not None
        city_candidates = outlook.get_viable_city_upgrades() if can_have_more_cities else []
        can_city = bool(city_candidates)
        can_dev = self._can_buy_dev_card(player)

        heavy_cache: Dict[
            Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[Tuple[str, int], ...]],
            Dict[str, float],
        ] = {}

        def _canonical_ports(port_dict: Dict[str, Any]) -> Tuple[Tuple[str, int], ...]:
            items: List[Tuple[str, int]] = []
            for k, v in (port_dict or {}).items():
                try:
                    items.append((str(k), int(v)))
                except Exception:
                    continue
            items.sort(key=lambda x: x[0])
            return tuple(items)

        def _cached_full_times(
            vertices_in: List[int],
            hand_in: List[int],
            ports_in: Dict[str, Any],
        ) -> Dict[str, float]:
            key = (
                tuple(int(x) for x in vertices_in),
                tuple(int(x) for x in hand_in),
                _canonical_ports(ports_in),
            )
            if key not in heavy_cache:
                heavy_cache[key] = self.game.markov.get_expected_time_to_event(
                    vertices=list(vertices_in),
                    hand=list(hand_in),
                    player_ports=dict(ports_in),
                )
            return heavy_cache[key]

        def _make_markov_explanation(
            vertices_in: List[int],
            strategy: str,
            score: float,
            extra_roads: int = 0,
            extra_fields: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Build a standard strategy breakdown row.

            Uses the Markov explanation helper when available, then overwrites score
            with the true HEAVY score from get_expected_time_to_event(...).
            """
            extra_fields = extra_fields or {}

            if hasattr(self.game.markov, "get_expected_turns_fast_initial_with_explanation"):
                out = self.game.markov.get_expected_turns_fast_initial_with_explanation(
                    vertices=list(vertices_in),
                    hand=hand,
                    player_ports=ports,
                    strategy=strategy,
                    extra_roads_needed=extra_roads,
                )
                out = dict(out)
                if "explanation" not in out or not isinstance(out.get("explanation"), dict):
                    out["explanation"] = {}
            else:
                out = {
                    "score": float(score),
                    "explanation": {},
                }

            out["score"] = float(score)
            out["heavy_mode"] = True

            for k, v in extra_fields.items():
                out[k] = v
                if k not in out["explanation"]:
                    out["explanation"][k] = v

            return out

        # ------------------------------------------------------------
        # 1) Settlement score = HEAVY traded Markov score
        # ------------------------------------------------------------
        if "new_settlement" in focus_set and can_new:
            full_times = _cached_full_times(base_vertices, hand, ports)

            settlement_score = float(
                full_times.get(
                    settlement_target_type,
                    full_times.get("settlement", 9999.0),
                )
            )

            settlement_heavy_debug = dict(
                full_times.get("__debug__", {}).get(settlement_target_type, {}) or {}
            )

            event_times["new_settlement"] = settlement_score

            out = _make_markov_explanation(
                vertices_in=base_vertices,
                strategy="settlement",
                score=settlement_score,
                extra_roads=extra_roads_needed,
                extra_fields={
                    "chosen_target": settlement_target,
                    "settlement_target_type": settlement_target_type,
                    "extra_roads_needed": int(extra_roads_needed),
                    **settlement_heavy_debug,
                },
            )

            self._last_strategy_breakdown[player.id]["new_settlement"] = out

        # ------------------------------------------------------------
        # 2) City score = LIGHT prefilter, then HEAVY on top near-tied candidates
        # ------------------------------------------------------------
        if "upgrade_to_city" in focus_set and can_city:
            light_city_entries: List[Dict[str, Any]] = []

            for inter_id in city_candidates:
                city_vertices = list(player.settlements) + list(player.cities) + [inter_id]

                if hasattr(self.game.markov, "get_expected_turns_fast_initial_with_explanation"):
                    light_out = self.game.markov.get_expected_turns_fast_initial_with_explanation(
                        vertices=city_vertices,
                        hand=hand,
                        player_ports=ports,
                        strategy="city",
                        extra_roads_needed=0,
                    )
                    light_out = dict(light_out)
                    light_score = float(light_out.get("score", 9999.0))
                elif hasattr(self.game.markov, "get_expected_turns_fast_initial"):
                    light_score = float(
                        self.game.markov.get_expected_turns_fast_initial(
                            vertices=city_vertices,
                            hand=hand,
                            player_ports=ports,
                            strategy="city",
                            extra_roads_needed=0,
                        )
                    )
                    light_out = {"score": light_score, "explanation": {}}
                else:
                    light_score = 9999.0
                    light_out = {"score": light_score, "explanation": {}}

                light_city_entries.append({
                    "inter_id": inter_id,
                    "light_score": float(light_score),
                    "light_out": light_out,
                })

            light_city_entries.sort(key=lambda x: float(x["light_score"]))

            if light_city_entries:
                best_light_score = float(light_city_entries[0]["light_score"])
                heavy_city_candidates: List[Dict[str, Any]] = []

                for entry in light_city_entries:
                    if len(heavy_city_candidates) >= 2:
                        break
                    if float(entry["light_score"]) <= best_light_score + 0.35:
                        heavy_city_candidates.append(entry)

                best_city_score = 9999.0
                best_city_out = None

                for entry in heavy_city_candidates:
                    inter_id = int(entry["inter_id"])
                    city_vertices = list(player.settlements) + list(player.cities) + [inter_id]

                    city_times = _cached_full_times(city_vertices, hand, ports)
                    score = float(city_times.get("city", 9999.0))

                    city_heavy_debug = dict(
                        city_times.get("__debug__", {}).get("city", {}) or {}
                    )

                    if score < best_city_score:
                        best_city_score = score

                        best_city_out = _make_markov_explanation(
                            vertices_in=city_vertices,
                            strategy="city",
                            score=score,
                            extra_roads=0,
                            extra_fields={
                                "chosen_upgrade": inter_id,
                                "light_prefilter_score": float(entry["light_score"]),
                                **city_heavy_debug,
                            },
                        )

                if best_city_score < 9999.0 and best_city_out is not None:
                    event_times["upgrade_to_city"] = best_city_score
                    self._last_strategy_breakdown[player.id]["upgrade_to_city"] = best_city_out

        # ------------------------------------------------------------
        # 3) Dev-card score = HEAVY traded Markov score
        # ------------------------------------------------------------
        if "buy_discovery_card" in focus_set and can_dev:
            full_times = _cached_full_times(base_vertices, hand, ports)
            dev_score = float(full_times.get("dev_card", 9999.0))

            dev_heavy_debug = dict(
                full_times.get("__debug__", {}).get("dev_card", {}) or {}
            )            

            event_times["buy_discovery_card"] = dev_score

            out = _make_markov_explanation(
                vertices_in=base_vertices,
                strategy="dev_card",
                score=dev_score,
                extra_roads=0,
                extra_fields=dev_heavy_debug,
            )

            self._last_strategy_breakdown[player.id]["buy_discovery_card"] = out

        return event_times