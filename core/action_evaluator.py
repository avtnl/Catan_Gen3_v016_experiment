"""
core/action_evaluator.py

One-ply action recommender built on top of victory_path_evaluator.py.

Purpose
-------
`victory_path_evaluator.py` answers:
    "Which of the 142 victory paths currently fit this player/board?"

This module answers:
    "Which legal or candidate action improves my best paths most, while denying
     opponents valuable settlement/port/resource options?"

Version 1 deliberately stays heuristic and explainable. It does not mutate the real
board or player objects. Instead it evaluates action deltas using:
    - expected victory-path quality before/after the action
    - production deltas from new settlements or city upgrades
    - demand-aware port value via the 142-way evaluator
    - opponent denial from vertices blocked by the distance rule
    - tactical bonuses for Longest Road, Largest Army, ports, and future settlements

Recommended use order
---------------------
1. Initial placement:
       rank_initial_settlement_actions(game, player, ways, top_n=15)

2. Execution-phase decision support:
       rank_actions(game, player, ways, top_n=10)

The output is a list of ActionEvaluation dataclasses. Convert them with
`action_evaluations_to_rows(...)` for logging or GUI display.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from math import inf, isfinite
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from core.constants import (
        NUM_PLAYERS,
        RCARDS_FOR_CITY,
        RCARDS_FOR_DCARD,
        RCARDS_FOR_ROAD,
        RCARDS_FOR_SETTLEMENT,
        RESOURCE_ORDER,
        ResourceCard,
    )
except Exception:  # Allows syntax/import smoke tests outside the project package.
    NUM_PLAYERS = 4

    class ResourceCard(Enum):  # type: ignore[no-redef]
        WHEAT = "Wheat"
        ORE = "Ore"
        WOOD = "Wood"
        BRICK = "Brick"
        WOOL = "Wool"

    RESOURCE_ORDER = [
        ResourceCard.WHEAT,
        ResourceCard.ORE,
        ResourceCard.WOOD,
        ResourceCard.BRICK,
        ResourceCard.WOOL,
    ]
    RCARDS_FOR_CITY = [2, 3, 0, 0, 0]
    RCARDS_FOR_SETTLEMENT = [1, 0, 1, 1, 1]
    RCARDS_FOR_ROAD = [0, 0, 1, 1, 0]
    RCARDS_FOR_DCARD = [1, 1, 0, 0, 1]

# v015/v016 expected-hand ranking policy constants.
#
# Important: import these defensively one-by-one from constants.py. Some older
# constants.py versions do not define every EH tuning knob. A tuple-style import
# would fail the whole block and accidentally disable EH mode.
try:
    import core.constants as _catan_constants

    RESOURCE_TIMING_ENGINE = str(
        getattr(_catan_constants, "RESOURCE_TIMING_ENGINE", "hybrid")
    ).strip().lower()

    MARKOV_TIMING_ENABLED = bool(
        getattr(
            _catan_constants,
            "MARKOV_TIMING_ENABLED",
            RESOURCE_TIMING_ENGINE in ("markov", "hybrid"),
        )
    )

    EXPECTED_HAND_PRIMARY_ENGINE = bool(
        getattr(
            _catan_constants,
            "EXPECTED_HAND_PRIMARY_ENGINE",
            RESOURCE_TIMING_ENGINE in ("hybrid", "expected_hand"),
        )
    )

    EXPECTED_HAND_PRIMARY_MIN_CONFIDENCE = float(
        getattr(_catan_constants, "EXPECTED_HAND_PRIMARY_MIN_CONFIDENCE", 0.85)
    )

    EXPECTED_HAND_ZERO_OVERRIDE_MIN_CONFIDENCE = float(
        getattr(_catan_constants, "EXPECTED_HAND_ZERO_OVERRIDE_MIN_CONFIDENCE", 0.99)
    )

    EXPECTED_HAND_PRIMARY_FALLBACK_TO_MARKOV = bool(
        getattr(
            _catan_constants,
            "EXPECTED_HAND_PRIMARY_FALLBACK_TO_MARKOV",
            RESOURCE_TIMING_ENGINE == "hybrid",
        )
    )
except Exception:  # pragma: no cover
    RESOURCE_TIMING_ENGINE = "hybrid"
    MARKOV_TIMING_ENABLED = True
    EXPECTED_HAND_PRIMARY_ENGINE = False
    EXPECTED_HAND_PRIMARY_MIN_CONFIDENCE = 0.85
    EXPECTED_HAND_ZERO_OVERRIDE_MIN_CONFIDENCE = 0.99
    EXPECTED_HAND_PRIMARY_FALLBACK_TO_MARKOV = True

try:
    from core.victory_path_evaluator import (
        OpeningPairEvaluation,
        PlacementEvaluation,
        VictoryWay,
        WayEvaluation,
        blocked_neighbor_pips,
        evaluate_way_with_vectors,
        get_intersection_port,
        get_intersection_resource_pips,
        get_player_ports,
        get_player_production_vector,
        get_player_resource_cards_vector,
        get_trade_rates,
        is_initial_intersection_buildable,
        main_bottleneck_resource,
        normalize_port_type,
        opening_pair_evaluations_to_rows,
        port_value_for_way,
        rank_initial_settlement_locations,
        rank_opening_pairs,
        resource_fit_components,
        score_initial_settlement_candidate,
        turns_to_afford_with_trading,
        valid_initial_intersections,
        vector_add,
        vector_sum,
        vector_to_named_dict,
    )
except Exception:  # pragma: no cover - only for isolated py_compile contexts.
    # Import errors are intentionally not swallowed at runtime inside the project.
    raise


RESOURCE_NAMES = [rc.value for rc in RESOURCE_ORDER]

ZERO_VECTOR = [0.0, 0.0, 0.0, 0.0, 0.0]
COST_VECTOR_BY_KEY: Dict[str, List[float]] = {
    "settlement": [float(x) for x in RCARDS_FOR_SETTLEMENT],
    "city": [float(x) for x in RCARDS_FOR_CITY],
    "road": [float(x) for x in RCARDS_FOR_ROAD],
    "dev_card": [float(x) for x in RCARDS_FOR_DCARD],
}


class ActionType(str, Enum):
    """Action categories used by the recommender."""

    INITIAL_SETTLEMENT = "INITIAL_SETTLEMENT"
    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    UPGRADE_CITY = "UPGRADE_CITY"
    BUY_DEV_CARD = "BUY_DEV_CARD"
    BUILD_ROAD_TO_SETTLEMENT = "BUILD_ROAD_TO_SETTLEMENT"
    BUILD_ROAD_TO_PORT = "BUILD_ROAD_TO_PORT"
    BUILD_ROAD_TO_BLOCK = "BUILD_ROAD_TO_BLOCK"
    BUILD_ROAD_GENERIC = "BUILD_ROAD_GENERIC"


@dataclass(frozen=True)
class ActionCandidate:
    """A candidate action before scoring."""

    action_type: ActionType
    description: str
    primary_target: Optional[int | Tuple[int, int]] = None
    secondary_target: Optional[int | Tuple[int, int]] = None
    resource_cost: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    production_delta: Tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    ports_delta: Tuple[str, ...] = ()
    denied_intersections: Tuple[int, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        data["resource_cost"] = list(self.resource_cost)
        data["production_delta"] = list(self.production_delta)
        data["ports_delta"] = list(self.ports_delta)
        data["denied_intersections"] = list(self.denied_intersections)
        return data


@dataclass
class PlayerStateEvaluation:
    """Compact summary of one player's ranked victory-path outlook."""

    player_id: int
    best_way_id: int
    best_score: float
    best_expected_turns: float
    top_k_average_score: float
    best_quality: float
    top_k_quality: float
    production_vector: Tuple[float, float, float, float, float]
    ports: Tuple[str, ...]
    trade_rates: Tuple[int, int, int, int, int]
    top_ways: Tuple[WayEvaluation, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "best_way_id": self.best_way_id,
            "best_score": self.best_score,
            "best_expected_turns": self.best_expected_turns,
            "top_k_average_score": self.top_k_average_score,
            "best_quality": self.best_quality,
            "top_k_quality": self.top_k_quality,
            "production_vector": list(self.production_vector),
            "production_named": vector_to_named_dict(self.production_vector),
            "ports": list(self.ports),
            "trade_rates": list(self.trade_rates),
            "top_ways": [ev.as_dict() for ev in self.top_ways],
        }


@dataclass
class ActionEvaluation:
    """Scored action with explainable components."""

    action: ActionCandidate
    final_score: float
    my_best_improvement: float
    my_top_k_improvement: float
    damage_to_best_opponent: float
    damage_to_all_opponents: float
    tactical_block_bonus: float
    opponent_damage: float
    tactical_score: float
    cost_penalty: float
    before_best_way_id: int
    after_best_way_id: int
    after_expected_turns: float
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action.action_type.value,
            "description": self.action.description,
            "primary_target": self.action.primary_target,
            "secondary_target": self.action.secondary_target,
            "final_score": self.final_score,
            "my_best_improvement": self.my_best_improvement,
            "my_top_k_improvement": self.my_top_k_improvement,
            "my_top5_way_improvement": self.my_top_k_improvement,
            "damage_to_best_opponent": self.damage_to_best_opponent,
            "damage_to_all_opponents": self.damage_to_all_opponents,
            "tactical_block_bonus": self.tactical_block_bonus,
            "opponent_damage": self.opponent_damage,
            "tactical_score": self.tactical_score,
            "cost_penalty": self.cost_penalty,
            "before_best_way_id": self.before_best_way_id,
            "after_best_way_id": self.after_best_way_id,
            "after_expected_turns": self.after_expected_turns,
            "resource_cost": list(self.action.resource_cost),
            "production_delta": list(self.action.production_delta),
            "production_delta_named": vector_to_named_dict(self.action.production_delta),
            "ports_delta": list(self.action.ports_delta),
            "denied_intersections": list(self.action.denied_intersections),
            "metadata": self.action.metadata,
            "notes": self.notes,
        }


DEFAULT_ACTION_WEIGHTS: Dict[str, float] = {
    # Version-1 action formula. Higher final score is better.
    # action_score = 1.00 * my_best_way_improvement
    #              + 0.50 * my_top5_way_improvement
    #              + 0.60 * damage_to_best_opponent
    #              + 0.30 * damage_to_all_opponents
    #              + 0.25 * tactical_block_bonus
    "my_best_way_improvement": 1.00,
    "my_best_improvement": 1.00,  # backward-compatible alias
    "my_top5_way_improvement": 0.50,
    "my_top_k_improvement": 0.50,  # backward-compatible alias
    "damage_to_best_opponent": 0.60,
    "damage_to_all_opponents": 0.30,
    "tactical_block_bonus": 0.25,

    # Optional affordability damping. Not part of the default formula.
    "cost_penalty": 0.0,
}


DEFAULT_TACTICAL_WEIGHTS: Dict[str, float] = {
    "raw_pips": 0.06,
    "port_value": 0.28,
    "blocked_neighbor_pips": 0.05,
    "future_settlement_value": 0.12,
    "road_longest_road": 0.55,
    "road_to_port": 0.35,
    "dev_card_path": 0.90,
    "city_focus": 0.30,
}


# ──────────────────────────────────────────────────────────────────────────────
# Small numeric helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_vector(values: Sequence[Any], length: int = 5) -> List[float]:
    cleaned = [_safe_float(v) for v in list(values)[:length]]
    if len(cleaned) < length:
        cleaned.extend([0.0] * (length - len(cleaned)))
    return cleaned


def _vector_subtract_nonnegative(a: Sequence[float], b: Sequence[float]) -> List[float]:
    a_v = _clean_vector(a)
    b_v = _clean_vector(b)
    return [max(0.0, a_v[i] - b_v[i]) for i in range(5)]


def _quality_from_score(score: float) -> float:
    """Convert lower-is-better path score into higher-is-better quality."""
    if not isfinite(score):
        return 0.0
    return 100.0 / (1.0 + max(0.0, score))


def _average(values: Sequence[float], default: float = inf) -> float:
    vals = list(values)
    if not vals:
        return default
    return sum(vals) / len(vals)


def _player_id(player: Any) -> int:
    return int(getattr(player, "id", -1))


def _player_color(player: Any) -> str:
    return str(getattr(player, "color", ""))


def _player_can_afford(player: Any, cost: Sequence[float]) -> bool:
    hand = get_player_resource_cards_vector(player)
    return all(hand[i] + 1e-9 >= _clean_vector(cost)[i] for i in range(5))


def _normalize_road(road: Sequence[int]) -> Tuple[int, int]:
    if len(road) != 2:
        raise ValueError(f"Road must contain two intersection ids, got {road}")
    a, b = int(road[0]), int(road[1])
    return tuple(sorted((a, b)))  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────────
# Player-state evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_player_state(
    board: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    top_k: int = 5,
    production_delta: Optional[Sequence[float]] = None,
    ports_delta: Optional[Iterable[str]] = None,
    hand_delta_cost: Optional[Sequence[float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
) -> PlayerStateEvaluation:
    """
    Evaluate a player's victory-path outlook, optionally after a hypothetical action.

    `production_delta` and `ports_delta` model future production/trading changes.
    `hand_delta_cost` subtracts immediate resource cost from the current hand.
    """
    base_production = get_player_production_vector(board, player)
    production = vector_add(base_production, production_delta or ZERO_VECTOR)

    ports = get_player_ports(board, player)
    for port in ports_delta or []:
        norm = normalize_port_type(port)
        if norm and norm not in ports:
            ports.append(norm)

    current_hand = get_player_resource_cards_vector(player)
    if hand_delta_cost is not None:
        current_hand = _vector_subtract_nonnegative(current_hand, hand_delta_cost)

    evaluations = [
        evaluate_way_with_vectors(
            way,
            production_vector=production,
            ports=ports,
            current_hand=current_hand,
            player=player,
            weights=eval_weights,
            rolls_per_player_turn=NUM_PLAYERS,
        )
        for way in ways
    ]
    evaluations.sort(key=lambda ev: (not isfinite(ev.final_score), ev.final_score, ev.expected_turns, ev.way_id))

    top = evaluations[: max(1, top_k)]
    finite_top = [ev for ev in top if isfinite(ev.final_score)]
    best = top[0]
    top_avg = _average([ev.final_score for ev in finite_top], default=inf)
    top_quality = _average([_quality_from_score(ev.final_score) for ev in top], default=0.0)

    return PlayerStateEvaluation(
        player_id=_player_id(player),
        best_way_id=best.way_id,
        best_score=best.final_score,
        best_expected_turns=best.expected_turns,
        top_k_average_score=top_avg,
        best_quality=_quality_from_score(best.final_score),
        top_k_quality=top_quality,
        production_vector=tuple(_clean_vector(production)),
        ports=tuple(ports),
        trade_rates=tuple(get_trade_rates(ports)),
        top_ways=tuple(top),
    )


def evaluate_game_state(
    board: Any,
    players: Sequence[Any],
    ways: Sequence[VictoryWay],
    *,
    top_k: int = 5,
    eval_weights: Optional[Dict[str, float]] = None,
) -> Dict[int, PlayerStateEvaluation]:
    """Evaluate every player's current victory-path outlook."""
    return {
        _player_id(player): evaluate_player_state(board, player, ways, top_k=top_k, eval_weights=eval_weights)
        for player in players
    }


# ──────────────────────────────────────────────────────────────────────────────
# Buildability and graph helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_board(game_or_board: Any) -> Any:
    return getattr(game_or_board, "board", game_or_board)


def _get_players(game_or_players: Any) -> List[Any]:
    if isinstance(game_or_players, Sequence) and not hasattr(game_or_players, "players"):
        return list(game_or_players)
    return list(getattr(game_or_players, "players", []))


def _can_build_intersection(game: Any, inter_id: int, player: Optional[Any] = None) -> bool:
    if game is not None and hasattr(game, "can_build_intersection_tf"):
        try:
            return bool(game.can_build_intersection_tf(inter_id, player))
        except TypeError:
            try:
                return bool(game.can_build_intersection_tf(inter_id))
            except Exception:
                pass
        except Exception:
            pass
    board = _get_board(game)
    return is_initial_intersection_buildable(board, inter_id)


def _roads_connected_to_intersection(board: Any, inter_id: int) -> List[Tuple[int, int]]:
    conn = getattr(board, "list_of_roads_connected_to_intersection", [])
    roads: Iterable[Any]
    if isinstance(conn, dict):
        roads = conn.get(inter_id, [])
    elif isinstance(conn, list) and 0 <= inter_id < len(conn):
        roads = conn[inter_id]
    else:
        roads = []

    normalized: List[Tuple[int, int]] = []
    for road in roads:
        try:
            normalized.append(_normalize_road(road))
        except Exception:
            continue
    return list(dict.fromkeys(normalized))


def _road_is_occupied(board: Any, road: Tuple[int, int]) -> bool:
    road = _normalize_road(road)
    for r in getattr(board, "roads", []):
        if r is not None and getattr(r, "id", None) == road and getattr(r, "occupied_tf", False):
            return True
    return False


def _can_build_road(board: Any, road: Tuple[int, int], player: Any) -> bool:
    road = _normalize_road(road)
    if hasattr(board, "can_build_road_for_color_tf"):
        try:
            return bool(board.can_build_road_for_color_tf(list(road), _player_color(player)))
        except Exception:
            pass
    if _road_is_occupied(board, road):
        return False
    if road[0] in getattr(board, "INTERSECTION_IN_WATER", []) or road[1] in getattr(board, "INTERSECTION_IN_WATER", []):
        return False
    valid = False
    for tile in getattr(board, "tiles", []):
        if tile is None:
            continue
        for edge in getattr(tile, "edges", []):
            if _normalize_road(getattr(edge, "road", [0, 0])) == road:
                valid = True
                break
        if valid:
            break
    return valid


def _network_endpoints(player: Any) -> List[int]:
    ids: List[int] = []
    ids.extend(int(x) for x in getattr(player, "settlements", []))
    ids.extend(int(x) for x in getattr(player, "cities", []))
    for road in getattr(player, "roads", []):
        try:
            a, b = _normalize_road(road)
        except Exception:
            continue
        ids.extend([a, b])
    return list(dict.fromkeys(ids))


def legal_road_candidates(
    board: Any,
    player: Any,
    *,
    restrict_to_intersection: Optional[int] = None,
    max_candidates: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """
    Return unoccupied road tuples connected to the player's current network.

    For initial placement road selection, pass restrict_to_intersection=last_settlement.
    """
    sources = [restrict_to_intersection] if restrict_to_intersection is not None else _network_endpoints(player)
    candidates: List[Tuple[int, int]] = []
    for source in sources:
        if source is None:
            continue
        for road in _roads_connected_to_intersection(board, int(source)):
            if _can_build_road(board, road, player):
                candidates.append(road)
    candidates = list(dict.fromkeys(candidates))

    # Prefer roads pointing toward higher future value when trimming.
    candidates.sort(key=lambda r: _road_static_priority(board, r), reverse=True)
    if max_candidates is not None:
        return candidates[:max_candidates]
    return candidates


def _road_static_priority(board: Any, road: Tuple[int, int]) -> float:
    a, b = road
    score = 0.0
    for endpoint in (a, b):
        score += vector_sum(get_intersection_resource_pips(board, endpoint))
        if get_intersection_port(board, endpoint):
            score += 6.0
        for next_road in _roads_connected_to_intersection(board, endpoint):
            n1, n2 = next_road
            other = n1 if n2 == endpoint else n2
            score += 0.20 * vector_sum(get_intersection_resource_pips(board, other))
            if get_intersection_port(board, other):
                score += 1.5
    return score


def denied_intersections_from_settlement(board: Any, inter_id: int) -> Tuple[int, ...]:
    """Intersections denied to opponents by placing a settlement at inter_id."""
    denied = [inter_id]
    if 0 <= inter_id < len(board.intersections):
        inter = board.intersections[inter_id]
        if inter is not None:
            for neighbor_id in getattr(inter, "three_intersection_ids", []):
                if is_initial_intersection_buildable(board, neighbor_id):
                    denied.append(int(neighbor_id))
    return tuple(dict.fromkeys(denied))


# ──────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ──────────────────────────────────────────────────────────────────────────────


def initial_settlement_candidate(
    board: Any,
    inter_id: int,
    *,
    description_prefix: str = "Initial settlement",
) -> ActionCandidate:
    prod = tuple(_clean_vector(get_intersection_resource_pips(board, inter_id)))
    port = get_intersection_port(board, inter_id)
    denied = denied_intersections_from_settlement(board, inter_id)
    return ActionCandidate(
        action_type=ActionType.INITIAL_SETTLEMENT,
        description=f"{description_prefix} at intersection {inter_id}",
        primary_target=inter_id,
        resource_cost=(0.0, 0.0, 0.0, 0.0, 0.0),
        production_delta=prod,
        ports_delta=(port,) if port else (),
        denied_intersections=denied,
        metadata={
            "raw_pips": vector_sum(prod),
            "port": port,
            "blocked_neighbor_pips": blocked_neighbor_pips(board, inter_id),
        },
    )


def build_settlement_candidate(board: Any, inter_id: int) -> ActionCandidate:
    prod = tuple(_clean_vector(get_intersection_resource_pips(board, inter_id)))
    port = get_intersection_port(board, inter_id)
    denied = denied_intersections_from_settlement(board, inter_id)
    return ActionCandidate(
        action_type=ActionType.BUILD_SETTLEMENT,
        description=f"Build settlement at intersection {inter_id}",
        primary_target=inter_id,
        resource_cost=tuple(COST_VECTOR_BY_KEY["settlement"]),
        production_delta=prod,
        ports_delta=(port,) if port else (),
        denied_intersections=denied,
        metadata={
            "raw_pips": vector_sum(prod),
            "port": port,
            "blocked_neighbor_pips": blocked_neighbor_pips(board, inter_id),
        },
    )


def upgrade_city_candidate(board: Any, inter_id: int) -> ActionCandidate:
    # Upgrading from settlement to city adds exactly one more settlement-worth of production.
    prod = tuple(_clean_vector(get_intersection_resource_pips(board, inter_id)))
    return ActionCandidate(
        action_type=ActionType.UPGRADE_CITY,
        description=f"Upgrade settlement {inter_id} to city",
        primary_target=inter_id,
        resource_cost=tuple(COST_VECTOR_BY_KEY["city"]),
        production_delta=prod,
        ports_delta=(),
        denied_intersections=(),
        metadata={"raw_pips": vector_sum(prod)},
    )


def buy_dev_card_candidate() -> ActionCandidate:
    return ActionCandidate(
        action_type=ActionType.BUY_DEV_CARD,
        description="Buy development card",
        resource_cost=tuple(COST_VECTOR_BY_KEY["dev_card"]),
        metadata={},
    )


def build_road_candidate(board: Any, player: Any, road: Tuple[int, int]) -> ActionCandidate:
    road = _normalize_road(road)
    action_type = classify_road_action(board, player, road)
    description = road_action_description(board, road, action_type)
    metadata = road_metadata(board, player, road)
    return ActionCandidate(
        action_type=action_type,
        description=description,
        primary_target=road,
        resource_cost=tuple(COST_VECTOR_BY_KEY["road"]),
        metadata=metadata,
    )


def generate_initial_settlement_actions(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
) -> List[ActionCandidate]:
    """Generate initial-placement settlement candidates."""
    board = _get_board(game)
    if valid_intersections is None:
        valid_intersections = [i for i in range(len(board.intersections)) if _can_build_intersection(game, i, player)]
    return [initial_settlement_candidate(board, int(i)) for i in valid_intersections]


def generate_candidate_actions(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    require_affordable: bool = False,
    include_initial_settlements: bool = False,
    valid_intersections: Optional[Sequence[int]] = None,
    max_road_candidates: int = 24,
) -> List[ActionCandidate]:
    """
    Generate a broad set of candidate actions for the current player.

    This is intentionally permissive. Set require_affordable=True when you only want
    immediately legal actions given the player's current hand.
    """
    board = _get_board(game)
    candidates: List[ActionCandidate] = []

    if include_initial_settlements:
        candidates.extend(generate_initial_settlement_actions(game, player, ways, valid_intersections=valid_intersections))
    else:
        if valid_intersections is None:
            valid_intersections = [i for i in range(len(board.intersections)) if _can_build_intersection(game, i, player)]
        for inter_id in valid_intersections:
            candidates.append(build_settlement_candidate(board, int(inter_id)))

    for inter_id in getattr(player, "settlements", []):
        if inter_id not in getattr(player, "cities", []):
            candidates.append(upgrade_city_candidate(board, int(inter_id)))

    candidates.append(buy_dev_card_candidate())

    for road in legal_road_candidates(board, player, max_candidates=max_road_candidates):
        candidates.append(build_road_candidate(board, player, road))

    if require_affordable:
        candidates = [c for c in candidates if _player_can_afford(player, c.resource_cost)]

    # Remove duplicates by action type + target.
    unique: Dict[Tuple[str, str], ActionCandidate] = {}
    for c in candidates:
        key = (c.action_type.value, str(c.primary_target))
        unique[key] = c
    return list(unique.values())


# ──────────────────────────────────────────────────────────────────────────────
# Road classification and tactical scoring
# ──────────────────────────────────────────────────────────────────────────────


def classify_road_action(board: Any, player: Any, road: Tuple[int, int]) -> ActionType:
    a, b = _normalize_road(road)

    if get_intersection_port(board, a) or get_intersection_port(board, b):
        return ActionType.BUILD_ROAD_TO_PORT

    # One-step lookahead for ports.
    for endpoint in (a, b):
        for next_road in _roads_connected_to_intersection(board, endpoint):
            n1, n2 = next_road
            other = n1 if n2 == endpoint else n2
            if get_intersection_port(board, other):
                return ActionType.BUILD_ROAD_TO_PORT

    # If it points toward a buildable high-pip intersection, classify as expansion.
    for endpoint in (a, b):
        if is_initial_intersection_buildable(board, endpoint) and vector_sum(get_intersection_resource_pips(board, endpoint)) >= 7.0:
            return ActionType.BUILD_ROAD_TO_SETTLEMENT
        for next_road in _roads_connected_to_intersection(board, endpoint):
            n1, n2 = next_road
            other = n1 if n2 == endpoint else n2
            if is_initial_intersection_buildable(board, other) and vector_sum(get_intersection_resource_pips(board, other)) >= 7.0:
                return ActionType.BUILD_ROAD_TO_SETTLEMENT

    # Near opponent structures can be a block/contest road.
    for endpoint in (a, b):
        inter = board.intersections[endpoint] if 0 <= endpoint < len(board.intersections) else None
        if inter is not None and getattr(inter, "occupied_tf", False) and getattr(inter, "color", "") != _player_color(player):
            return ActionType.BUILD_ROAD_TO_BLOCK

    return ActionType.BUILD_ROAD_GENERIC


def road_action_description(board: Any, road: Tuple[int, int], action_type: ActionType) -> str:
    base = f"Build road {road}"
    if action_type == ActionType.BUILD_ROAD_TO_PORT:
        return f"{base} toward port access"
    if action_type == ActionType.BUILD_ROAD_TO_SETTLEMENT:
        return f"{base} toward future settlement"
    if action_type == ActionType.BUILD_ROAD_TO_BLOCK:
        return f"{base} to contest/block opponent path"
    return base


def road_metadata(board: Any, player: Any, road: Tuple[int, int]) -> Dict[str, Any]:
    a, b = _normalize_road(road)
    future_value = 0.0
    nearest_port_bonus = 0.0
    endpoint_pips = 0.0
    best_future_intersection: Optional[int] = None

    checked: set[int] = set()
    for endpoint in (a, b):
        endpoint_pips += vector_sum(get_intersection_resource_pips(board, endpoint))
        if get_intersection_port(board, endpoint):
            nearest_port_bonus = max(nearest_port_bonus, 10.0)
        for next_road in _roads_connected_to_intersection(board, endpoint):
            n1, n2 = next_road
            other = n1 if n2 == endpoint else n2
            if other in checked:
                continue
            checked.add(other)
            pips = vector_sum(get_intersection_resource_pips(board, other))
            port = get_intersection_port(board, other)
            if port:
                nearest_port_bonus = max(nearest_port_bonus, 5.0 + 0.3 * pips)
            if is_initial_intersection_buildable(board, other):
                val = pips + (4.0 if port else 0.0)
                if val > future_value:
                    future_value = val
                    best_future_intersection = other

    return {
        "endpoint_pips": endpoint_pips,
        "future_settlement_value": future_value,
        "best_future_intersection": best_future_intersection,
        "nearest_port_bonus": nearest_port_bonus,
        "road_length_after_proxy": float(getattr(player, "size_longest_route", 0) or 0) + 1.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Opponent denial and tactical action score
# ──────────────────────────────────────────────────────────────────────────────


def opponent_denial_damage_components(
    board: Any,
    players: Sequence[Any],
    current_player: Any,
    denied_intersections: Sequence[int],
    ways: Sequence[VictoryWay],
    *,
    eval_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """
    Return (damage_to_best_opponent, damage_to_all_opponents).

    Each opponent's damage is estimated by scoring the denied intersections from
    that opponent's perspective using the same 142-way placement evaluator.
    """
    if not denied_intersections:
        return 0.0, 0.0

    per_opponent_damage: List[float] = []
    for opponent in players:
        if _player_id(opponent) == _player_id(current_player):
            continue
        best_for_opponent = 0.0
        total_for_opponent = 0.0
        for inter_id in denied_intersections:
            if inter_id < 0 or inter_id >= len(board.intersections):
                continue
            if not is_initial_intersection_buildable(board, int(inter_id)):
                continue
            try:
                placement = score_initial_settlement_candidate(
                    board,
                    opponent,
                    int(inter_id),
                    ways,
                    eval_weights=eval_weights,
                    include_current_hand=True,
                )
            except Exception:
                continue
            if isfinite(placement.placement_score):
                best_for_opponent = max(best_for_opponent, placement.placement_score)
                total_for_opponent += max(0.0, placement.placement_score)
        # Best denied spot matters most; total captures multi-vertex distance-rule blocks.
        per_opponent_damage.append(0.55 * best_for_opponent + 0.10 * total_for_opponent)

    if not per_opponent_damage:
        return 0.0, 0.0
    return max(per_opponent_damage), sum(per_opponent_damage)


def opponent_denial_damage(
    board: Any,
    players: Sequence[Any],
    current_player: Any,
    denied_intersections: Sequence[int],
    ways: Sequence[VictoryWay],
    *,
    eval_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Backward-compatible aggregate opponent-denial score."""
    best_damage, all_damage = opponent_denial_damage_components(
        board,
        players,
        current_player,
        denied_intersections,
        ways,
        eval_weights=eval_weights,
    )
    return 0.60 * best_damage + 0.30 * all_damage


def _top_way_lookup(ways: Sequence[VictoryWay]) -> Dict[int, VictoryWay]:
    return {w.way_id: w for w in ways}


def tactical_score_for_action(
    board: Any,
    player: Any,
    action: ActionCandidate,
    ways: Sequence[VictoryWay],
    before_state: PlayerStateEvaluation,
    after_state: PlayerStateEvaluation,
    *,
    tactical_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, List[str]]:
    """Return tactical_score and notes."""
    weights = {**DEFAULT_TACTICAL_WEIGHTS, **(tactical_weights or {})}
    notes: List[str] = []
    score = 0.0

    raw_pips = _safe_float(action.metadata.get("raw_pips", vector_sum(action.production_delta)))
    if raw_pips:
        score += weights["raw_pips"] * raw_pips
        notes.append(f"adds {raw_pips:.1f} production pips")

    if action.ports_delta:
        best_way = next((w for w in ways if w.way_id == after_state.best_way_id), ways[0])
        production_after = after_state.production_vector
        existing_ports = get_player_ports(board, player)
        hand_after = _vector_subtract_nonnegative(get_player_resource_cards_vector(player), action.resource_cost)
        p_values = [
            port_value_for_way(
                best_way,
                production_after,
                port,
                current_hand=hand_after,
                existing_ports=existing_ports,
            )
            for port in action.ports_delta
        ]
        p_value = max(p_values) if p_values else 0.0
        score += weights["port_value"] * p_value
        notes.append(f"adds port {', '.join(action.ports_delta)} with estimated value {p_value:.2f}")

    blocked = _safe_float(action.metadata.get("blocked_neighbor_pips", 0.0))
    if blocked:
        score += weights["blocked_neighbor_pips"] * blocked
        notes.append(f"blocks/denies neighboring pip potential {blocked:.1f}")

    if action.action_type == ActionType.UPGRADE_CITY:
        way_map = _top_way_lookup(ways)
        top_city_need = 0.0
        for ev in before_state.top_ways:
            way = way_map.get(ev.way_id)
            if way is not None:
                top_city_need += way.cities
        top_city_need /= max(1, len(before_state.top_ways))
        bonus = weights["city_focus"] * top_city_need
        score += bonus
        notes.append(f"city upgrade supports top-way city demand ({top_city_need:.1f})")

    if action.action_type == ActionType.BUY_DEV_CARD:
        way_map = _top_way_lookup(ways)
        dev_focus = 0.0
        for ev in before_state.top_ways:
            way = way_map.get(ev.way_id)
            if way is None:
                continue
            dev_focus += way.victory_point_cards + (2.0 if way.biggest_army else 0.0)
        dev_focus /= max(1, len(before_state.top_ways))
        # Being near Largest Army makes a dev card more valuable.
        army_size = float(getattr(player, "size_largest_army", 0) or 0)
        army_bonus = max(0.0, army_size - 1.0) * 0.35
        bonus = weights["dev_card_path"] * (dev_focus + army_bonus)
        score += bonus
        notes.append(f"development-card focus among top ways is {dev_focus:.2f}")

    if action.action_type in {
        ActionType.BUILD_ROAD_TO_SETTLEMENT,
        ActionType.BUILD_ROAD_TO_PORT,
        ActionType.BUILD_ROAD_TO_BLOCK,
        ActionType.BUILD_ROAD_GENERIC,
    }:
        future = _safe_float(action.metadata.get("future_settlement_value", 0.0))
        if future:
            score += weights["future_settlement_value"] * future
            notes.append(f"road points toward future settlement value {future:.1f}")

        if action.action_type == ActionType.BUILD_ROAD_TO_PORT:
            port_bonus = _safe_float(action.metadata.get("nearest_port_bonus", 0.0))
            score += weights["road_to_port"] * port_bonus
            notes.append(f"road improves port access proxy {port_bonus:.1f}")

        # Longest-road need from current top ways.
        lr_need = sum(1.0 for ev in before_state.top_ways for w in ways if w.way_id == ev.way_id and w.longest_road)
        lr_need /= max(1, len(before_state.top_ways))
        if lr_need:
            score += weights["road_longest_road"] * lr_need
            notes.append(f"road supports Longest Road in {lr_need:.0%} of top ways")

    return score, notes


# ──────────────────────────────────────────────────────────────────────────────
# Action scoring and ranking
# ──────────────────────────────────────────────────────────────────────────────


def score_action_candidate(
    game: Any,
    players: Sequence[Any],
    current_player: Any,
    ways: Sequence[VictoryWay],
    action: ActionCandidate,
    *,
    top_k: int = 5,
    action_weights: Optional[Dict[str, float]] = None,
    tactical_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
) -> ActionEvaluation:
    """Score one candidate action without mutating game state."""
    board = _get_board(game)
    weights = {**DEFAULT_ACTION_WEIGHTS, **(action_weights or {})}

    before = evaluate_player_state(board, current_player, ways, top_k=top_k, eval_weights=eval_weights)
    after = evaluate_player_state(
        board,
        current_player,
        ways,
        top_k=top_k,
        production_delta=action.production_delta,
        ports_delta=action.ports_delta,
        hand_delta_cost=action.resource_cost,
        eval_weights=eval_weights,
    )

    my_best_improvement = after.best_quality - before.best_quality
    my_top_k_improvement = after.top_k_quality - before.top_k_quality
    damage_to_best_opponent, damage_to_all_opponents = opponent_denial_damage_components(
        board,
        players,
        current_player,
        action.denied_intersections,
        ways,
        eval_weights=eval_weights,
    )
    opp_damage = 0.60 * damage_to_best_opponent + 0.30 * damage_to_all_opponents
    tactical, notes = tactical_score_for_action(
        board,
        current_player,
        action,
        ways,
        before,
        after,
        tactical_weights=tactical_weights,
    )

    # Cost penalty is small because actions are usually only considered when affordable.
    # It prevents expensive actions from winning solely due to small heuristic bonuses.
    cost_total = vector_sum(action.resource_cost)
    hand_total = vector_sum(get_player_resource_cards_vector(current_player))
    cost_penalty = cost_total / max(1.0, hand_total + 1.0)

    final_score = 0.0
    # Version-1 formula, kept explicit for readability and tuning.
    final_score += weights.get("my_best_way_improvement", weights.get("my_best_improvement", 1.0)) * my_best_improvement
    final_score += weights.get("my_top5_way_improvement", weights.get("my_top_k_improvement", 0.5)) * my_top_k_improvement
    final_score += weights["damage_to_best_opponent"] * damage_to_best_opponent
    final_score += weights["damage_to_all_opponents"] * damage_to_all_opponents
    final_score += weights["tactical_block_bonus"] * tactical
    # Optional affordability damping; default is 0.0 and therefore not part of v1.
    final_score -= weights.get("cost_penalty", 0.0) * cost_penalty

    if cost_total and not _player_can_afford(current_player, action.resource_cost):
        notes.append("not currently affordable; useful as a planning target, not immediate action")

    return ActionEvaluation(
        action=action,
        final_score=final_score,
        my_best_improvement=my_best_improvement,
        my_top_k_improvement=my_top_k_improvement,
        damage_to_best_opponent=damage_to_best_opponent,
        damage_to_all_opponents=damage_to_all_opponents,
        tactical_block_bonus=tactical,
        opponent_damage=opp_damage,
        tactical_score=tactical,
        cost_penalty=cost_penalty,
        before_best_way_id=before.best_way_id,
        after_best_way_id=after.best_way_id,
        after_expected_turns=after.best_expected_turns,
        notes=notes,
    )


def rank_initial_settlement_actions(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
    top_n: Optional[int] = 15,
    top_k: int = 5,
    action_weights: Optional[Dict[str, float]] = None,
    tactical_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
) -> List[ActionEvaluation]:
    """
    Rank initial settlement placements as actions.

    This extends rank_initial_settlement_locations(...) by adding opponent denial.
    """
    board = _get_board(game)
    players = _get_players(game)
    if valid_intersections is None:
        valid_intersections = [i for i in range(len(board.intersections)) if _can_build_intersection(game, i, player)]

    candidates = generate_initial_settlement_actions(game, player, ways, valid_intersections=valid_intersections)
    evaluations = [
        score_action_candidate(
            game,
            players,
            player,
            ways,
            candidate,
            top_k=top_k,
            action_weights=action_weights,
            tactical_weights=tactical_weights,
            eval_weights=eval_weights,
        )
        for candidate in candidates
    ]
    evaluations.sort(key=lambda ev: (-ev.final_score, ev.action.primary_target if ev.action.primary_target is not None else 999))
    if top_n is not None:
        return evaluations[:top_n]
    return evaluations


def rank_initial_settlement_locations_with_opponents(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
    top_n: Optional[int] = 15,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper returning row dictionaries for GUI/log display.

    Use this when you want the opponent-aware action score but a settlement-location table.
    """
    ranked = rank_initial_settlement_actions(
        game,
        player,
        ways,
        valid_intersections=valid_intersections,
        top_n=top_n,
    )
    return action_evaluations_to_rows(ranked)


def rank_actions(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    top_n: Optional[int] = 10,
    top_k: int = 5,
    require_affordable: bool = False,
    include_initial_settlements: bool = False,
    valid_intersections: Optional[Sequence[int]] = None,
    max_road_candidates: int = 24,
    action_weights: Optional[Dict[str, float]] = None,
    tactical_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
) -> List[ActionEvaluation]:
    """Generate and rank candidate actions for the current player."""
    players = _get_players(game)
    candidates = generate_candidate_actions(
        game,
        player,
        ways,
        require_affordable=require_affordable,
        include_initial_settlements=include_initial_settlements,
        valid_intersections=valid_intersections,
        max_road_candidates=max_road_candidates,
    )
    evaluations = [
        score_action_candidate(
            game,
            players,
            player,
            ways,
            candidate,
            top_k=top_k,
            action_weights=action_weights,
            tactical_weights=tactical_weights,
            eval_weights=eval_weights,
        )
        for candidate in candidates
    ]
    evaluations.sort(key=lambda ev: (-ev.final_score, ev.action.action_type.value, str(ev.action.primary_target)))
    if top_n is not None:
        return evaluations[:top_n]
    return evaluations



# ──────────────────────────────────────────────────────────────────────────────
# Fast-forward expected-action bridge
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_expected_activity(activity: Any) -> str:
    """
    Normalize fast_forward expected action names to action_evaluator activity names.
    """
    text = str(activity or "").strip().lower()

    aliases = {
        "dev_card": "buy_discovery_card",
        "development_card": "buy_discovery_card",
        "buy_dev_card": "buy_discovery_card",
        "buy_discovery_card": "buy_discovery_card",

        "city": "upgrade_to_city",
        "upgrade_city": "upgrade_to_city",
        "upgrade_to_city": "upgrade_to_city",

        "settlement": "new_settlement",
        "build_settlement": "new_settlement",
        "new_settlement": "new_settlement",
        "settlement_0r": "new_settlement",
        "settlement_1r": "new_settlement",
        "settlement_2r": "new_settlement",
    }

    return aliases.get(text, text)


def _target_as_int(value: Any) -> Optional[int]:
    """
    Convert fast_forward target values to int when possible.

    Returns None for values like:
        None, "-", "?", "development_card"
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text in {"-", "?", "None", "development_card"}:
        return None

    try:
        return int(float(text))
    except Exception:
        return None


def _is_expected_hand_only_mode() -> bool:
    """Return True when the active timing backend is EH-only."""
    return str(RESOURCE_TIMING_ENGINE).strip().lower() == "expected_hand"


def _markov_timing_available(game: Any = None) -> bool:
    """Return True only when Markov timing is enabled and a Markov object exists."""
    if not bool(MARKOV_TIMING_ENABLED):
        return False

    if game is None:
        return bool(MARKOV_TIMING_ENABLED)

    return getattr(game, "markov", None) is not None


def _none_like(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "-", "none", "null", "nan"}


def _optional_float(value: Any) -> Optional[float]:
    """Return float(value), or None for absent/non-finite-like values."""
    if _none_like(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _expected_action_timing_source(action: Dict[str, Any]) -> str:
    """Best-effort source label from a fast_forward expected action row."""
    for key in (
        "timing_source_for_ranking",
        "timing_primary_source",
        "timing_source",
        "scheduling_source",
        "source_mode",
        "mode",
    ):
        value = action.get(key)
        if not _none_like(value):
            return str(value).strip().lower()
    return ""


def _expected_action_markov_score(action: Dict[str, Any]) -> Optional[float]:
    """
    Read the real light-Markov timing score from a fast_forward expected action.

    In EH-only mode, ``score`` is the active EH score, not Markov. Therefore this
    function must *not* fall back from missing/None ``markov_score`` to ``score``
    when the row source is expected-hand-only.
    """
    if "markov_score" in action:
        return _optional_float(action.get("markov_score"))

    source = _expected_action_timing_source(action)
    if _is_expected_hand_only_mode() or source == "expected_hand_only":
        return None

    for key in ("markov_turns", "light_markov_score", "expected_turns", "delta_rolls"):
        if key in action:
            value = _optional_float(action.get(key))
            if value is not None:
                return value

    # Compatibility only for old Markov/hybrid rows that did not yet carry an
    # explicit markov_score field. Do not use this in EH-only mode.
    if not _is_expected_hand_only_mode() and "score" in action:
        return _optional_float(action.get("score"))

    return None


def _expected_action_expected_hand_score(action: Dict[str, Any]) -> float:
    """Read the expected-hand timing score from a fast_forward action row."""
    for key in ("expected_hand_score", "eh_score", "expected_hand_turns"):
        if key in action:
            value = _optional_float(action.get(key))
            if value is not None:
                return float(value)

    # In EH-only rows, score/timing_score is the primary EH timing.
    source = _expected_action_timing_source(action)
    if _is_expected_hand_only_mode() or source == "expected_hand_only":
        for key in ("timing_score_for_ranking", "timing_primary_score", "score"):
            if key in action:
                value = _optional_float(action.get(key))
                if value is not None:
                    return float(value)

    return 9999.0


def _with_expected_metadata(
    candidate: ActionCandidate,
    base_metadata: Dict[str, Any],
) -> ActionCandidate:
    """Return a copy of candidate with fast-forward metadata merged in."""
    return ActionCandidate(
        action_type=candidate.action_type,
        description=candidate.description,
        primary_target=candidate.primary_target,
        secondary_target=candidate.secondary_target,
        resource_cost=candidate.resource_cost,
        production_delta=candidate.production_delta,
        ports_delta=candidate.ports_delta,
        denied_intersections=candidate.denied_intersections,
        metadata={**dict(candidate.metadata), **base_metadata},
    )


def expected_viable_action_to_candidates(
    game: Any,
    player: Any,
    expected_action: Dict[str, Any],
) -> List[ActionCandidate]:
    """
    Convert one fast_forward expected action dict into one or more ActionCandidate objects.

    This does not mutate the board or player.

    Important:
    - If fast_forward provides a concrete target, we use it.
    - If the target is missing, we fall back to the current legal/candidate options
      for that action type.
    """
    board = _get_board(game)
    activity = _normalize_expected_activity(
        expected_action.get("activity", expected_action.get("raw_activity_key"))
    )
    target = _target_as_int(expected_action.get("target"))
    markov_score = _expected_action_markov_score(expected_action)
    expected_hand_score = _expected_action_expected_hand_score(expected_action)
    expected_timing_source = _expected_action_timing_source(expected_action)

    base_metadata = {
        "source": "fast_forward_expected_viable_actions",
        "expected_activity": activity,
        "raw_expected_action": dict(expected_action),

        # Real Markov/light timing only. In EH-only mode this intentionally stays
        # None, even though the row's backward-compatible `score` field contains
        # an EH timing.
        "markov_score": markov_score,

        # v015/v016 expected-hand timing metadata.
        "expected_hand_score": expected_action.get("expected_hand_score", expected_hand_score),
        "expected_timing_source": expected_timing_source,
        "resource_timing_engine": RESOURCE_TIMING_ENGINE,
        "markov_timing_enabled": bool(MARKOV_TIMING_ENABLED),
        "markov_timing_available": bool(_markov_timing_available(game)),
        "expected_hand_delta": expected_action.get("expected_hand_delta"),
        "expected_hand_confidence": expected_action.get("expected_hand_confidence"),
        "expected_hand_confidence_target": expected_action.get("expected_hand_confidence_target"),
        "expected_hand_confidence_label": expected_action.get("expected_hand_confidence_label"),
        "expected_hand_found": expected_action.get("expected_hand_found"),
        "expected_hand_key": expected_action.get("expected_hand_key"),
        "expected_hand_exact_settlement_key": expected_action.get("expected_hand_exact_settlement_key"),
        "expected_hand_zero_corrected_by_guard": expected_action.get("expected_hand_zero_corrected_by_guard"),
        "expected_hand_zero_correction_reason": expected_action.get("expected_hand_zero_correction_reason"),
        "expected_hand_zero_correction_old_score": expected_action.get("expected_hand_zero_correction_old_score"),
        "expected_hand_zero_correction_old_confidence": expected_action.get("expected_hand_zero_correction_old_confidence"),

        "extra_roads_needed": expected_action.get("extra_roads_needed"),
        "settlement_target_type": expected_action.get("settlement_target_type"),
        "fast_forward_code": expected_action.get("code"),
    }

    candidates: List[ActionCandidate] = []

    # ------------------------------------------------------------
    # Dev card
    # ------------------------------------------------------------
    if activity == "buy_discovery_card":
        candidates.append(_with_expected_metadata(buy_dev_card_candidate(), base_metadata))
        return candidates

    # ------------------------------------------------------------
    # City upgrade
    # ------------------------------------------------------------
    if activity == "upgrade_to_city":
        if target is not None:
            if target in list(getattr(player, "settlements", [])):
                candidates.append(
                    _with_expected_metadata(
                        upgrade_city_candidate(board, int(target)),
                        base_metadata,
                    )
                )
            return candidates

        # Fallback if fast_forward did not provide a concrete city target.
        for inter_id in getattr(player, "settlements", []):
            if inter_id in getattr(player, "cities", []):
                continue

            candidates.append(
                _with_expected_metadata(
                    upgrade_city_candidate(board, int(inter_id)),
                    base_metadata,
                )
            )

        return candidates

    # ------------------------------------------------------------
    # New settlement
    # ------------------------------------------------------------
    if activity == "new_settlement":
        if target is not None:
            candidates.append(
                _with_expected_metadata(
                    build_settlement_candidate(board, int(target)),
                    base_metadata,
                )
            )
            return candidates

        # Fallback if fast_forward did not provide a concrete settlement target.
        # This is broader, but still restricted to buildable intersections.
        for inter_id in range(len(getattr(board, "intersections", []))):
            if not _can_build_intersection(game, int(inter_id), player):
                continue

            candidates.append(
                _with_expected_metadata(
                    build_settlement_candidate(board, int(inter_id)),
                    base_metadata,
                )
            )

        return candidates

    return candidates


def expected_viable_actions_to_candidates(
    game: Any,
    player: Any,
    expected_viable_actions: Sequence[Dict[str, Any]],
) -> List[ActionCandidate]:
    """
    Convert the whole fast_forward expected_viable_actions list into ActionCandidates.
    """
    candidates: List[ActionCandidate] = []

    for expected_action in expected_viable_actions or []:
        try:
            candidates.extend(
                expected_viable_action_to_candidates(
                    game,
                    player,
                    dict(expected_action),
                )
            )
        except Exception:
            continue

    # Remove duplicate action_type + target combinations.
    unique: Dict[Tuple[str, str], ActionCandidate] = {}
    for cand in candidates:
        key = (cand.action_type.value, str(cand.primary_target))
        unique[key] = cand

    return list(unique.values())


def rank_expected_viable_actions(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    expected_viable_actions: Sequence[Dict[str, Any]],
    *,
    top_n: Optional[int] = None,
    top_k: int = 5,
    markov_delay_weight: float = 0.25,
    require_affordable: bool = False,
    action_weights: Optional[Dict[str, float]] = None,
    tactical_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
) -> List[ActionEvaluation]:
    """
    Rank only the actions that fast_forward considered expected viable.

    This is the bridge for v015:

        fast_forward expected_viable_actions
            -> ActionCandidate list
            -> 142-way action scoring
            -> timing delay penalty selected by policy
            -> ranked ActionEvaluation list

    Timing policy:
        1. EH=0/conf>=threshold wins as exact-payable-now.
        2. If EXPECTED_HAND_PRIMARY_ENGINE is enabled and EH is valid/confident,
           use expected-hand timing for the delay penalty.
        3. Otherwise fall back to light Markov timing when allowed.

    Higher final_score is better.

    This function does NOT execute anything and does NOT mutate the board/player.
    The exact PLAY guard must still be used before execution.
    """
    players = _get_players(game)

    candidates = expected_viable_actions_to_candidates(
        game,
        player,
        expected_viable_actions,
    )

    if require_affordable:
        candidates = [
            cand for cand in candidates
            if _player_can_afford(player, cand.resource_cost)
        ]

    evaluations: List[ActionEvaluation] = []

    for cand in candidates:
        ev = score_action_candidate(
            game,
            players,
            player,
            ways,
            cand,
            top_k=top_k,
            action_weights=action_weights,
            tactical_weights=tactical_weights,
            eval_weights=eval_weights,
        )

        markov_score_optional = _optional_float(cand.metadata.get("markov_score"))
        markov_score_for_sort = (
            float(markov_score_optional)
            if markov_score_optional is not None
            else 9999.0
        )
        has_real_markov_score = (
            markov_score_optional is not None
            and float(markov_score_optional) < 9999.0
            and bool(MARKOV_TIMING_ENABLED)
            and not _is_expected_hand_only_mode()
        )

        expected_hand_score = _safe_float(
            cand.metadata.get("expected_hand_score", 9999.0),
            9999.0,
        )
        expected_hand_confidence = _safe_float(
            cand.metadata.get("expected_hand_confidence", 0.0),
            0.0,
        )
        expected_hand_label = str(
            cand.metadata.get("expected_hand_confidence_label", "") or ""
        ).strip().lower()
        expected_hand_found = bool(cand.metadata.get("expected_hand_found", False))

        eh_zero_exact = (
            expected_hand_score == 0.0
            and expected_hand_confidence >= float(EXPECTED_HAND_ZERO_OVERRIDE_MIN_CONFIDENCE)
        )

        eh_score_valid = (
            expected_hand_found
            and expected_hand_score < 9999.0
        )

        eh_confident_enough = (
            expected_hand_confidence >= float(EXPECTED_HAND_PRIMARY_MIN_CONFIDENCE)
        )

        timing_score_for_ranking = 9999.0
        timing_source_for_ranking = "no_valid_timing"

        if eh_zero_exact:
            timing_score_for_ranking = 0.0
            timing_source_for_ranking = "expected_hand_zero_override"

        elif (
            bool(EXPECTED_HAND_PRIMARY_ENGINE)
            and eh_score_valid
            and eh_confident_enough
        ):
            timing_score_for_ranking = float(expected_hand_score)
            timing_source_for_ranking = (
                "expected_hand_only"
                if _is_expected_hand_only_mode()
                else "expected_hand_primary"
            )

        elif (
            bool(EXPECTED_HAND_PRIMARY_FALLBACK_TO_MARKOV)
            and has_real_markov_score
        ):
            timing_score_for_ranking = float(markov_score_optional)
            timing_source_for_ranking = "markov_light"

        elif not bool(EXPECTED_HAND_PRIMARY_FALLBACK_TO_MARKOV):
            timing_score_for_ranking = 9999.0
            timing_source_for_ranking = "expected_hand_missing_no_markov_fallback"

        # Penalize actions that are strategically good but expected much later.
        # This keeps "very good but far away" from automatically beating
        # "good and soon".
        if timing_score_for_ranking < 9999.0 and markov_delay_weight:
            ev.final_score -= float(markov_delay_weight) * float(timing_score_for_ranking)
            ev.notes.append(
                f"Timing delay penalty ({timing_source_for_ranking}): "
                f"-{markov_delay_weight:.2f} * {timing_score_for_ranking:.2f}"
            )

        # Add useful metadata for fast_forward debug printing.
        # v015: timing source is selected by policy:
        # - EH=0/conf>=threshold overrides as exact-payable-now
        # - EH primary is used when enabled and confident
        # - Markov remains fallback when EH is missing/invalid
        ev.action.metadata["strategic_final_score_after_delay"] = float(ev.final_score)
        ev.action.metadata["markov_delay_weight"] = float(markov_delay_weight)

        ev.action.metadata["markov_score"] = (
            float(markov_score_optional)
            if markov_score_optional is not None
            else None
        )
        ev.action.metadata["markov_timing_enabled"] = bool(MARKOV_TIMING_ENABLED)
        ev.action.metadata["markov_timing_available"] = bool(_markov_timing_available(game))
        ev.action.metadata["resource_timing_engine"] = RESOURCE_TIMING_ENGINE
        ev.action.metadata["expected_hand_only_mode"] = bool(_is_expected_hand_only_mode())
        ev.action.metadata["expected_hand_score"] = float(expected_hand_score)
        ev.action.metadata["expected_hand_confidence"] = float(expected_hand_confidence)
        ev.action.metadata["expected_hand_confidence_label"] = cand.metadata.get(
            "expected_hand_confidence_label"
        )
        ev.action.metadata["expected_hand_found"] = cand.metadata.get("expected_hand_found")
        ev.action.metadata["expected_hand_delta"] = cand.metadata.get("expected_hand_delta")
        ev.action.metadata["expected_hand_key"] = cand.metadata.get("expected_hand_key")
        ev.action.metadata["expected_hand_exact_settlement_key"] = cand.metadata.get(
            "expected_hand_exact_settlement_key"
        )

        ev.action.metadata["timing_score_for_ranking"] = float(timing_score_for_ranking)
        ev.action.metadata["timing_source_for_ranking"] = timing_source_for_ranking

        # Keep old names too, so existing debug/table code does not break.
        ev.action.metadata["timing_primary_score"] = float(timing_score_for_ranking)
        ev.action.metadata["timing_primary_source"] = timing_source_for_ranking

        ev.action.metadata["expected_hand_primary_engine"] = bool(EXPECTED_HAND_PRIMARY_ENGINE)
        ev.action.metadata["expected_hand_primary_min_confidence"] = float(
            EXPECTED_HAND_PRIMARY_MIN_CONFIDENCE
        )
        ev.action.metadata["expected_hand_zero_override_min_confidence"] = float(
            EXPECTED_HAND_ZERO_OVERRIDE_MIN_CONFIDENCE
        )
        ev.action.metadata["expected_hand_primary_fallback_to_markov"] = bool(
            EXPECTED_HAND_PRIMARY_FALLBACK_TO_MARKOV
        )
        ev.action.metadata["expected_hand_label_used_for_ranking"] = expected_hand_label

        evaluations.append(ev)

    evaluations.sort(
        key=lambda ev: (
            -float(ev.final_score),
            _safe_float(ev.action.metadata.get("timing_score_for_ranking", 9999.0), 9999.0),
            _safe_float(ev.action.metadata.get("markov_score", 9999.0), 9999.0),
            ev.action.action_type.value,
            str(ev.action.primary_target),
        )
    )

    if top_n is not None:
        return evaluations[:top_n]

    return evaluations

def expected_action_evaluations_to_rows(
    evaluations: Sequence[ActionEvaluation],
) -> List[Dict[str, Any]]:
    """
    Formatting helper for expected viable action rankings.

    This keeps the standard action_evaluations_to_rows(...) output but adds
    fast-forward-specific fields at top level for easier terminal logging.
    """
    rows: List[Dict[str, Any]] = []

    for ev in evaluations:
        row = ev.as_dict()
        metadata = dict(row.get("metadata", {}) or {})

        row["expected_activity"] = metadata.get("expected_activity")
        row["fast_forward_code"] = metadata.get("fast_forward_code")
        row["markov_score"] = metadata.get("markov_score")
        row["markov_delay_weight"] = metadata.get("markov_delay_weight")
        row["markov_timing_enabled"] = metadata.get("markov_timing_enabled")
        row["markov_timing_available"] = metadata.get("markov_timing_available")
        row["resource_timing_engine"] = metadata.get("resource_timing_engine")
        row["expected_hand_only_mode"] = metadata.get("expected_hand_only_mode")

        # v015 expected-hand comparison/ranking metadata.
        row["expected_hand_score"] = metadata.get("expected_hand_score")
        row["expected_hand_confidence"] = metadata.get("expected_hand_confidence")
        row["expected_hand_confidence_target"] = metadata.get("expected_hand_confidence_target")
        row["expected_hand_confidence_label"] = metadata.get("expected_hand_confidence_label")
        row["expected_hand_found"] = metadata.get("expected_hand_found")
        row["expected_hand_delta"] = metadata.get("expected_hand_delta")
        row["expected_hand_key"] = metadata.get("expected_hand_key")
        row["expected_hand_exact_settlement_key"] = metadata.get("expected_hand_exact_settlement_key")

        row["expected_hand_primary_engine"] = metadata.get("expected_hand_primary_engine")
        row["expected_hand_primary_min_confidence"] = metadata.get(
            "expected_hand_primary_min_confidence"
        )
        row["expected_hand_zero_override_min_confidence"] = metadata.get(
            "expected_hand_zero_override_min_confidence"
        )
        row["expected_hand_primary_fallback_to_markov"] = metadata.get(
            "expected_hand_primary_fallback_to_markov"
        )

        # Explicit ranking timing fields.
        row["timing_score_for_ranking"] = metadata.get("timing_score_for_ranking")
        row["timing_source_for_ranking"] = metadata.get("timing_source_for_ranking")

        # Backward-compatible names used by existing debug/table code.
        row["timing_primary_score"] = metadata.get("timing_primary_score")
        row["timing_primary_source"] = metadata.get("timing_primary_source")

        row["strategic_final_score_after_delay"] = metadata.get(
            "strategic_final_score_after_delay",
            row.get("final_score"),
        )

        rows.append(row)

    return rows

# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────


def action_evaluations_to_rows(evaluations: Sequence[ActionEvaluation]) -> List[Dict[str, Any]]:
    return [ev.as_dict() for ev in evaluations]


def player_state_to_row(state: PlayerStateEvaluation) -> Dict[str, Any]:
    return state.as_dict()


def game_state_to_rows(state: Dict[int, PlayerStateEvaluation]) -> List[Dict[str, Any]]:
    return [state[pid].as_dict() for pid in sorted(state)]


# ──────────────────────────────────────────────────────────────────────────────
# Integration helper for InitialPlacement
# ──────────────────────────────────────────────────────────────────────────────


def choose_initial_settlement(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
) -> int:
    """Return the best opponent-aware initial settlement id, or -1 if none exists."""
    ranked = rank_initial_settlement_actions(
        game,
        player,
        ways,
        valid_intersections=valid_intersections,
        top_n=1,
    )
    if not ranked:
        return -1
    target = ranked[0].action.primary_target
    return int(target) if isinstance(target, int) else -1


def choose_initial_road(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    settlement_id: int,
) -> Optional[Tuple[int, int]]:
    """
    Choose a road from a just-placed initial settlement.

    This reuses the action road scoring but restricts candidates to roads connected to
    the new settlement. It returns a sorted road tuple or None.
    """
    board = _get_board(game)
    players = _get_players(game)
    roads = legal_road_candidates(board, player, restrict_to_intersection=settlement_id)
    if not roads:
        return None

    candidates = [build_road_candidate(board, player, road) for road in roads]
    scored = [score_action_candidate(game, players, player, ways, c) for c in candidates]
    scored.sort(key=lambda ev: (-ev.final_score, str(ev.action.primary_target)))
    target = scored[0].action.primary_target
    if isinstance(target, tuple):
        return target
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Explainability helpers
# ──────────────────────────────────────────────────────────────────────────────


def _resource_gap_for_way(player: Any, way_eval: WayEvaluation) -> List[float]:
    hand = get_player_resource_cards_vector(player)
    return _vector_subtract_nonnegative(way_eval.need_vector, hand)


def _port_dependency_for_way(state: PlayerStateEvaluation, way_eval: WayEvaluation) -> str:
    bottleneck = getattr(way_eval, "main_bottleneck_resource", None) or main_bottleneck_resource(
        way_eval.need_vector,
        state.production_vector,
    )
    owned = set(state.ports)
    specific = f"2:1 {bottleneck}"
    if specific in owned or "3:1" in owned:
        return "covered"
    return f"would benefit from {specific} or 3:1"


def explain_player_state(
    board: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    state: Optional[PlayerStateEvaluation] = None,
    top_k: int = 5,
    eval_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return an explainable summary of a player's top victory-path outlook."""
    state = state or evaluate_player_state(board, player, ways, top_k=top_k, eval_weights=eval_weights)
    way_by_id = {w.way_id: w for w in ways}
    best_eval = state.top_ways[0] if state.top_ways else None
    best_way = way_by_id.get(best_eval.way_id) if best_eval else None

    top_way_rows = []
    for ev in state.top_ways[:top_k]:
        way = way_by_id.get(ev.way_id)
        top_way_rows.append({
            "way_id": ev.way_id,
            "score": ev.final_score,
            "expected_turns": ev.expected_turns,
            "need_vector": list(ev.need_vector),
            "main_bottleneck": getattr(ev, "main_bottleneck_resource", main_bottleneck_resource(ev.need_vector, ev.production_vector)),
            "requires_longest_road": bool(way.longest_road) if way else False,
            "requires_largest_army": bool(way.biggest_army) if way else False,
            "vp_cards": int(way.victory_point_cards) if way else 0,
        })

    lr_count = sum(1 for row in top_way_rows if row["requires_longest_road"])
    la_count = sum(1 for row in top_way_rows if row["requires_largest_army"])

    return {
        "player_id": _player_id(player),
        "best_path_now": best_eval.as_dict() if best_eval else None,
        "top_5_paths": top_way_rows,
        "resources_missing_for_best_path": vector_to_named_dict(_resource_gap_for_way(player, best_eval)) if best_eval else {},
        "best_port_dependency": _port_dependency_for_way(state, best_eval) if best_eval else "None",
        "lr_dependency": {
            "best_path_requires_lr": bool(best_way.longest_road) if best_way else False,
            "top5_lr_count": lr_count,
            "top5_lr_share": lr_count / max(1, len(top_way_rows)),
        },
        "la_dependency": {
            "best_path_requires_la": bool(best_way.biggest_army) if best_way else False,
            "top5_la_count": la_count,
            "top5_la_share": la_count / max(1, len(top_way_rows)),
        },
        "main_bottleneck": getattr(best_eval, "main_bottleneck_resource", "None") if best_eval else "None",
        "production_vector": list(state.production_vector),
        "production_named": vector_to_named_dict(state.production_vector),
        "ports": list(state.ports),
        "trade_rates": list(state.trade_rates),
    }


def find_most_damaging_opponent_block(
    game: Any,
    players: Sequence[Any],
    current_player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Find the settlement location that most damages opponents through vertex denial."""
    board = _get_board(game)
    candidates = list(valid_intersections) if valid_intersections is not None else valid_initial_intersections(board)
    best = {"intersection_id": None, "damage_to_best_opponent": 0.0, "damage_to_all_opponents": 0.0}
    for inter_id in candidates:
        denied = denied_intersections_from_settlement(board, int(inter_id))
        best_damage, all_damage = opponent_denial_damage_components(
            board,
            players,
            current_player,
            denied,
            ways,
            eval_weights=eval_weights,
        )
        weighted = 0.60 * best_damage + 0.30 * all_damage
        if weighted > best.get("weighted_damage", -1.0):
            best = {
                "intersection_id": int(inter_id),
                "denied_intersections": list(denied),
                "damage_to_best_opponent": best_damage,
                "damage_to_all_opponents": all_damage,
                "weighted_damage": weighted,
            }
    return best


def explain_all_player_states(
    game: Any,
    players: Sequence[Any],
    ways: Sequence[VictoryWay],
    *,
    current_player: Any = None,
    top_k: int = 5,
    eval_weights: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Explain best path, top paths, bottlenecks, dependencies and ports for every player."""
    board = _get_board(game)
    output: Dict[int, Dict[str, Any]] = {}
    for player in players:
        output[_player_id(player)] = explain_player_state(
            board,
            player,
            ways,
            top_k=top_k,
            eval_weights=eval_weights,
        )
    if current_player is not None:
        output[_player_id(current_player)]["most_damaging_opponent_block"] = find_most_damaging_opponent_block(
            game,
            players,
            current_player,
            ways,
            eval_weights=eval_weights,
        )
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Opening-pair planning wrappers
# ──────────────────────────────────────────────────────────────────────────────


def rank_opening_pair_plans(
    game: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
    top_n: Optional[int] = 15,
    top_k_ways: int = 5,
    eval_weights: Optional[Dict[str, float]] = None,
) -> List[OpeningPairEvaluation]:
    """
    Rank two-settlement opening plans for the current board.

    This is a planning view rather than a single immediate action. It helps decide an
    opening goal before the first settlement is placed, while rank_initial_settlement_actions
    remains the one-ply action view.
    """
    board = _get_board(game)
    if valid_intersections is None:
        valid_intersections = [i for i in range(len(board.intersections)) if _can_build_intersection(game, i, player)]
    return rank_opening_pairs(
        board,
        player,
        ways,
        valid_intersections=valid_intersections,
        top_n=top_n,
        top_k_ways=top_k_ways,
        eval_weights=eval_weights,
    )


def opening_pair_plans_to_rows(evaluations: Sequence[OpeningPairEvaluation]) -> List[Dict[str, Any]]:
    """Return row dictionaries for logging/terminal/GUI display of opening-pair plans."""
    return opening_pair_evaluations_to_rows(evaluations)
