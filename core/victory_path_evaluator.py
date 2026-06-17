"""
core/victory_path_evaluator.py

First-pass evaluator for mapping the 142 Catan victory paths to an actual board state.

Design goals
------------
1. Keep the 142 victory rows as data: target LR/LA/cities/settlements/VP cards +
   production-only resource needs.
2. Convert a player position into a resource-specific production vector:
      [Wheat, Ore, Wood, Brick, Wool]
   using pip/dot strength as the common unit.
3. Evaluate each victory path against current production, owned ports, and bank trade.
4. Provide starting helpers for initial-placement candidate ranking.

This module intentionally does not mutate the board. It should be safe to call from
initial-placement guidance, human hinting, or later action-evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from math import inf, isfinite
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import csv

try:
    from core.constants import NUM_PLAYERS, RESOURCE_ORDER, ResourceCard, TERRAIN_TO_RESOURCE
except Exception:  # Allows this file to be imported in isolation for lightweight tests.
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
    TERRAIN_TO_RESOURCE = {
        "Field": ResourceCard.WHEAT,
        "Mountain": ResourceCard.ORE,
        "Forest": ResourceCard.WOOD,
        "Hill": ResourceCard.BRICK,
        "Pasture": ResourceCard.WOOL,
    }


RESOURCE_NAMES: List[str] = [rc.value for rc in RESOURCE_ORDER]
RESOURCE_INDEX_BY_NAME: Dict[str, int] = {name.lower(): idx for idx, name in enumerate(RESOURCE_NAMES)}
# Your older code sometimes says Sheep while constants.py says Wool.
RESOURCE_INDEX_BY_NAME["sheep"] = RESOURCE_INDEX_BY_NAME["wool"]

DEFAULT_EVALUATION_WEIGHTS: Dict[str, float] = {
    # Version-1 way scoring formula. Added to expected turns. Tune after simulation.
    # way_score = expected_turns
    #           + 0.40 * dev_card_risk
    #           + 0.75 * lr_risk
    #           + 0.75 * la_risk
    #           + 0.50 * port_distance_penalty
    #           + 0.30 * resource_bottleneck_penalty
    "dev_card_risk": 0.40,
    "longest_road_risk": 0.75,
    "largest_army_risk": 0.75,
    "port_distance_penalty": 0.50,
    "resource_bottleneck_penalty": 0.30,

    # Kept for backward-compatible experimentation. Not used by the default formula.
    "resource_mismatch": 0.0,
    "bottleneck_pressure": 0.30,
}


DEFAULT_PLACEMENT_WEIGHTS: Dict[str, float] = {
    # Higher placement score is better.
    # Revised after comparing full-board baseline turns vs one-settlement turns:
    # do not let a small port/block bonus outrank strong production and flexibility.
    "raw_pips": 1.00,
    "resource_diversity": 1.25,
    "best_way_viability": 2.00,
    "top5_way_viability": 4.00,
    "port_value": 0.10,
    "blocked_neighbor_pips": 0.03,

    # Backward-compatible aliases accepted by callers that override weights.
    "own_best_way": 0.0,
    "own_top5_average": 0.0,
}

DEFAULT_OPENING_PAIR_WEIGHTS: Dict[str, float] = {
    # Higher pair score is better. This is the preferred initial-placement baseline
    # because Catan openings are two settlements, not one isolated vertex.
    "combined_pips": 1.00,
    "resource_diversity": 1.50,
    "best_way_viability": 2.00,
    "top5_way_viability": 5.00,
    "port_value": 0.10,
    "blocked_neighbor_pips": 0.02,
}


@dataclass(frozen=True)
class VictoryWay:
    """One row from the 142-way table."""

    way_id: int
    source_row: int
    longest_road: bool
    biggest_army: bool
    cities: int
    settlements: int
    victory_point_cards: int
    total_victory_points: int
    twelve_point_edge_case: bool
    article_min_cost: float
    buildings: int
    new_settlements_to_build: int
    city_upgrades: int
    roads_to_build: int
    development_cards_to_buy: int
    need_vector: Tuple[float, float, float, float, float]
    production_only_total: float

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "VictoryWay":
        def as_bool(value: Any) -> bool:
            return str(value).strip().lower() in {"yes", "y", "true", "1"}

        def as_int(name: str, default: int = 0) -> int:
            value = row.get(name, default)
            try:
                return int(float(str(value).strip()))
            except (TypeError, ValueError):
                return default

        def as_float(name: str, default: float = 0.0) -> float:
            value = row.get(name, default)
            try:
                return float(str(value).strip())
            except (TypeError, ValueError):
                return default

        need = (
            as_float("Wheat_Needed"),
            as_float("Ore_Needed"),
            as_float("Wood_Needed"),
            as_float("Brick_Needed"),
            as_float("Wool_Needed"),
        )

        return cls(
            way_id=as_int("Way_ID"),
            source_row=as_int("Source_Row"),
            longest_road=as_bool(row.get("Longest_Road", "no")),
            biggest_army=as_bool(row.get("Biggest_Army", "no")),
            cities=as_int("Cities"),
            settlements=as_int("Settlements"),
            victory_point_cards=as_int("Victory_Point_Cards"),
            total_victory_points=as_int("Total_Victory_Points"),
            twelve_point_edge_case=as_bool(row.get("Twelve_Point_Edge_Case", "no")),
            article_min_cost=as_float("Article_Min_Cost"),
            buildings=as_int("Buildings"),
            new_settlements_to_build=as_int("New_Settlements_To_Build"),
            city_upgrades=as_int("City_Upgrades"),
            roads_to_build=as_int("Roads_To_Build"),
            development_cards_to_buy=as_int("Development_Cards_To_Buy"),
            need_vector=need,
            production_only_total=as_float("Production_Only_Total", sum(need)),
        )

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["need_vector"] = list(self.need_vector)
        return data


@dataclass
class WayEvaluation:
    """Scored evaluation of one victory way for one player/position."""

    way_id: int
    final_score: float
    expected_turns: float
    resource_mismatch: float
    bottleneck_pressure: float
    resource_bottleneck_penalty: float
    port_distance_penalty: float
    main_bottleneck_resource: str
    dev_card_risk: float
    longest_road_risk: float
    largest_army_risk: float
    need_vector: Tuple[float, float, float, float, float]
    production_vector: Tuple[float, float, float, float, float]
    trade_rates: Tuple[int, int, int, int, int]
    owned_ports: Tuple[str, ...]
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["need_vector"] = list(self.need_vector)
        data["production_vector"] = list(self.production_vector)
        data["trade_rates"] = list(self.trade_rates)
        data["owned_ports"] = list(self.owned_ports)
        return data


@dataclass
class PlacementEvaluation:
    """Scored evaluation of a possible settlement placement."""

    intersection_id: int
    placement_score: float
    best_way_id: int
    best_way_score_after: float
    best_way_expected_turns_after: float
    top5_average_score_after: float
    production_added: Tuple[float, float, float, float, float]
    port_added: str
    raw_pips_added: float
    port_value: float
    blocked_neighbor_pips: float
    resource_diversity: float = 0.0
    best_way_viability: float = 0.0
    top5_way_viability: float = 0.0
    best_way_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["production_added"] = list(self.production_added)
        return data


@dataclass
class OpeningPairEvaluation:
    """Scored evaluation of a two-settlement opening pair."""

    first_intersection_id: int
    second_intersection_id: int
    pair_score: float
    best_way_id: int
    best_way_score_after: float
    best_way_expected_turns_after: float
    top5_average_score_after: float
    top5_way_ids: Tuple[int, ...]
    production_first: Tuple[float, float, float, float, float]
    production_second: Tuple[float, float, float, float, float]
    production_combined: Tuple[float, float, float, float, float]
    ports_added: Tuple[str, ...]
    raw_pips_combined: float
    port_value: float
    blocked_neighbor_pips: float
    resource_diversity: float
    best_way_viability: float
    top5_way_viability: float
    best_way_reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["production_first"] = list(self.production_first)
        data["production_second"] = list(self.production_second)
        data["production_combined"] = list(self.production_combined)
        data["ports_added"] = list(self.ports_added)
        data["top5_way_ids"] = list(self.top5_way_ids)
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Loading victory ways
# ──────────────────────────────────────────────────────────────────────────────


def load_142_ways(path: str | Path) -> List[VictoryWay]:
    """Load the CSV table created earlier for the 142 valid Catan victory ways."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Victory-way CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        ways = [VictoryWay.from_csv_row(row) for row in reader]

    if len(ways) != 142:
        # Do not fail hard; this allows testing smaller files, but make the issue visible.
        print(f"Warning: expected 142 victory ways, loaded {len(ways)} from {path}")

    return ways


# ──────────────────────────────────────────────────────────────────────────────
# Vector helpers
# ──────────────────────────────────────────────────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_vector(values: Sequence[Any], length: int = 5) -> List[float]:
    cleaned = [_safe_float(v) for v in values[:length]]
    if len(cleaned) < length:
        cleaned.extend([0.0] * (length - len(cleaned)))
    return cleaned


def vector_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_safe_float(x) + _safe_float(y) for x, y in zip(_clean_vector(a), _clean_vector(b))]


def vector_scale(a: Sequence[float], factor: float) -> List[float]:
    return [_safe_float(x) * factor for x in _clean_vector(a)]


def vector_sum(a: Sequence[float]) -> float:
    return float(sum(_clean_vector(a)))


def resource_diversity(vector: Sequence[float], *, threshold: float = 0.0) -> int:
    """Number of resource types with meaningful production in the vector."""
    return sum(1 for value in _clean_vector(vector) if value > threshold)


def viability_from_score(score: float) -> float:
    """Convert a lower-is-better path score into a bounded higher-is-better value."""
    if not isfinite(score):
        return 0.0
    return 100.0 / (1.0 + max(0.0, score))


def normalize_port_type(port_type: Optional[str]) -> str:
    """Normalize Blank/Sheep/Wool variants so trading logic is consistent."""
    if not port_type:
        return ""
    text = str(port_type).strip()
    if not text or text.lower() == "blank":
        return ""
    return text.replace("Sheep", "Wool")


def resource_index_from_port(port_type: str) -> Optional[int]:
    port_type = normalize_port_type(port_type)
    if not port_type.startswith("2:1"):
        return None
    parts = port_type.split(maxsplit=1)
    if len(parts) != 2:
        return None
    return RESOURCE_INDEX_BY_NAME.get(parts[1].lower())


# ──────────────────────────────────────────────────────────────────────────────
# Production and ports
# ──────────────────────────────────────────────────────────────────────────────


def _pips_from_value(value: Any) -> float:
    """Classic Catan pip count. Kept local to avoid import cycles."""
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return 0.0
    if not 2 <= value_int <= 12 or value_int == 7:
        return 0.0
    return float(6 - abs(7 - value_int))


def get_intersection_resource_pips(board: Any, inter_id: int, multiplier: float = 1.0) -> List[float]:
    """
    Return [Wheat, Ore, Wood, Brick, Wool] pip production for one intersection.

    The code first uses board/intersection precomputed vectors when available. If those
    are absent or all zero, it computes from adjacent tile IDs and tile values.
    """
    if inter_id < 0 or inter_id >= len(board.intersections):
        return [0.0] * 5

    inter = board.intersections[inter_id]
    if inter is None:
        return [0.0] * 5

    # Best source in the current board.py: all_tile_pips is explicitly in resource order.
    for attr in ("all_tile_pips", "all_tile_probabilities", "three_tile_probabilities_v2", "three_tile_probabilities"):
        raw = getattr(inter, attr, None)
        if raw is not None:
            vec = _clean_vector(raw)
            # all_tile_pips/all_tile_probabilities are already resource-specific.
            # three_tile_* may be resource-specific in your current code; if not, the fallback below covers it.
            if vector_sum(vec) > 0:
                return vector_scale(vec, multiplier)

    # Fallback: compute directly from adjacent tile IDs and terrain.
    result = [0.0] * 5
    for tile_id in getattr(inter, "three_tile_ids", []):
        if tile_id is None:
            continue
        try:
            tile = board.tiles[int(tile_id)]
        except (TypeError, ValueError, IndexError):
            continue
        if tile is None or getattr(tile, "type", None) in ("Sea", "Desert", "Blank", None):
            continue
        resource = TERRAIN_TO_RESOURCE.get(tile.type)
        if resource is None:
            continue
        try:
            idx = RESOURCE_ORDER.index(resource)
        except ValueError:
            continue
        result[idx] += _pips_from_value(getattr(tile, "value", 0))

    return vector_scale(result, multiplier)


def get_player_production_vector(board: Any, player: Any) -> List[float]:
    """
    Return current resource-specific pip production for a player.

    Settlements count 1x; cities count 2x. If a city is still present in player.settlements,
    it is not double-counted as a settlement.
    """
    production = [0.0] * 5
    city_ids = set(getattr(player, "cities", []))

    for inter_id in getattr(player, "settlements", []):
        if inter_id in city_ids:
            continue
        production = vector_add(production, get_intersection_resource_pips(board, inter_id, multiplier=1.0))

    for inter_id in city_ids:
        production = vector_add(production, get_intersection_resource_pips(board, inter_id, multiplier=2.0))

    return production


def get_player_ports(board: Any, player: Any) -> List[str]:
    """Return normalized unique ports owned by settlements/cities."""
    ports: List[str] = []
    for inter_id in list(getattr(player, "settlements", [])) + list(getattr(player, "cities", [])):
        if inter_id < 0 or inter_id >= len(board.intersections):
            continue
        inter = board.intersections[inter_id]
        if inter is None:
            continue
        if getattr(inter, "port_tf", False) or getattr(inter, "harborYN", "N") == "Y":
            port_type = normalize_port_type(getattr(inter, "port_type", ""))
            if port_type and port_type not in ports:
                ports.append(port_type)
    return ports


def get_intersection_port(board: Any, inter_id: int) -> str:
    """Return normalized port type for an intersection, or ''."""
    if inter_id < 0 or inter_id >= len(board.intersections):
        return ""
    inter = board.intersections[inter_id]
    if inter is None:
        return ""
    if getattr(inter, "port_tf", False) or getattr(inter, "harborYN", "N") == "Y":
        return normalize_port_type(getattr(inter, "port_type", ""))
    return ""


def get_trade_rates(ports: Iterable[str] = ()) -> List[int]:
    """
    Convert owned ports to bank trade rates in RESOURCE_ORDER.

    No port -> [4,4,4,4,4]
    3:1     -> [3,3,3,3,3]
    2:1 X   -> X becomes 2, other resources remain their best existing rate.
    """
    rates = [4, 4, 4, 4, 4]
    for port in ports:
        port = normalize_port_type(port)
        if not port:
            continue
        if port == "3:1":
            rates = [min(rate, 3) for rate in rates]
            continue
        idx = resource_index_from_port(port)
        if idx is not None:
            rates[idx] = min(rates[idx], 2)
    return rates


def get_player_resource_cards_vector(player: Any) -> List[float]:
    """Return current hand as [Wheat, Ore, Wood, Brick, Wool]."""
    rcards = getattr(player, "rcards", {})
    return [float(rcards.get(rc, 0)) for rc in RESOURCE_ORDER]


# ──────────────────────────────────────────────────────────────────────────────
# Expected turns and trading
# ──────────────────────────────────────────────────────────────────────────────


def can_cover_need_with_trade(
    have: Sequence[float],
    need: Sequence[float],
    trade_rates: Sequence[int],
    *,
    continuous: bool = True,
) -> bool:
    """
    True if current resources can cover need after bank/port trades.

    This is an expected-value check. With continuous=True, surplus 1.5 at 3:1 contributes
    0.5 imported resource. With continuous=False, it floors export trades by resource.
    """
    have_v = _clean_vector(have)
    need_v = _clean_vector(need)
    rates = [max(1, int(r)) for r in _clean_vector(trade_rates)]

    deficit = 0.0
    import_capacity = 0.0
    for h, n, rate in zip(have_v, need_v, rates):
        if h < n:
            deficit += n - h
        else:
            surplus = h - n
            import_capacity += surplus / rate if continuous else int(surplus // rate)

    return import_capacity + 1e-9 >= deficit


def turns_to_afford_with_trading(
    need: Sequence[float],
    production_pips: Sequence[float],
    trade_rates: Sequence[int],
    *,
    current_hand: Optional[Sequence[float]] = None,
    rolls_per_player_turn: int = NUM_PLAYERS,
    max_turns: float = 250.0,
    continuous_trading: bool = True,
) -> float:
    """
    Estimate player turns until need is affordable from production + bank/port trades.

    production_pips are classic Catan pips per 36 dice rolls. In a 4-player game, a player
    receives production from about 4 dice rolls per own turn, so:
        expected_cards_per_own_turn = pips * 4 / 36

    This function returns expected own turns. It is meant for ranking, not exact rules play.
    """
    need_v = _clean_vector(need)
    production_v = _clean_vector(production_pips)
    hand_v = _clean_vector(current_hand or [0.0] * 5)
    rates_v = [max(1, int(r)) for r in _clean_vector(trade_rates)]

    if can_cover_need_with_trade(hand_v, need_v, rates_v, continuous=continuous_trading):
        return 0.0

    if vector_sum(production_v) <= 0:
        return inf

    def have_after(turns: float) -> List[float]:
        factor = float(rolls_per_player_turn) * turns / 36.0
        return [hand_v[i] + production_v[i] * factor for i in range(5)]

    # Expand upper bound until affordable or max_turns reached.
    lo = 0.0
    hi = 1.0
    while hi < max_turns and not can_cover_need_with_trade(have_after(hi), need_v, rates_v, continuous=continuous_trading):
        hi *= 2.0

    if hi >= max_turns and not can_cover_need_with_trade(have_after(max_turns), need_v, rates_v, continuous=continuous_trading):
        return inf

    hi = min(hi, max_turns)
    for _ in range(48):  # plenty for stable continuous ranking
        mid = (lo + hi) / 2.0
        if can_cover_need_with_trade(have_after(mid), need_v, rates_v, continuous=continuous_trading):
            hi = mid
        else:
            lo = mid
    return hi


# ──────────────────────────────────────────────────────────────────────────────
# Way scoring
# ──────────────────────────────────────────────────────────────────────────────


def resource_fit_components(need: Sequence[float], production: Sequence[float]) -> Tuple[float, float]:
    """Return (resource_mismatch, bottleneck_pressure). Lower is better."""
    need_v = _clean_vector(need)
    prod_v = _clean_vector(production)
    need_total = vector_sum(need_v)
    prod_total = vector_sum(prod_v)

    if need_total <= 0:
        return 0.0, 0.0
    if prod_total <= 0:
        return 1.0, inf

    need_share = [x / need_total for x in need_v]
    prod_share = [x / prod_total for x in prod_v]
    mismatch = 0.5 * sum(abs(n - p) for n, p in zip(need_share, prod_share))

    bottleneck = 0.0
    for n, p in zip(need_v, prod_v):
        if n <= 0:
            continue
        bottleneck = max(bottleneck, n / max(p, 0.25))

    return mismatch, bottleneck


def main_bottleneck_resource(need: Sequence[float], production: Sequence[float]) -> str:
    """Return the resource with the highest need/production pressure."""
    need_v = _clean_vector(need)
    prod_v = _clean_vector(production)
    best_idx = 0
    best_pressure = -1.0
    for idx, (n, p) in enumerate(zip(need_v, prod_v)):
        if n <= 0:
            continue
        pressure = n / max(p, 0.25)
        if pressure > best_pressure:
            best_pressure = pressure
            best_idx = idx
    return RESOURCE_NAMES[best_idx] if best_pressure >= 0 else "None"


def _player_network_intersections(player: Any) -> List[int]:
    ids: List[int] = []
    ids.extend(int(x) for x in getattr(player, "settlements", []) or [])
    ids.extend(int(x) for x in getattr(player, "cities", []) or [])
    for road in getattr(player, "roads", []) or []:
        try:
            a, b = road
            ids.extend([int(a), int(b)])
        except Exception:
            continue
    return list(dict.fromkeys(ids))


def _distance_between_intersections(board: Any, a: int, b: int) -> float:
    if board is None:
        return inf
    if hasattr(board, "_distance_between_intersections"):
        try:
            dist = board._distance_between_intersections(int(a), int(b))
            return float(dist) if dist is not None else inf
        except Exception:
            pass
    return 0.0 if a == b else inf


def _port_matches_resource(port_type: str, resource_name: str) -> bool:
    port = normalize_port_type(port_type)
    if not port:
        return False
    return port == "3:1" or port == f"2:1 {resource_name}"


def estimate_port_distance_penalty(
    board: Any,
    player: Any,
    way: VictoryWay,
    production_vector: Sequence[float],
    ports: Iterable[str],
) -> float:
    """
    Estimate how far the player is from a useful port for this way.

    A player with an already useful port receives 0. Without a current network
    or without board context the penalty is 0 so the evaluator remains safe in
    isolated tests. When a network exists, the target port is the 2:1 port for
    the current bottleneck resource, with 3:1 as an acceptable fallback.
    """
    bottleneck = main_bottleneck_resource(way.need_vector, production_vector)
    owned = [normalize_port_type(p) for p in ports if normalize_port_type(p)]
    if any(_port_matches_resource(p, bottleneck) for p in owned):
        return 0.0

    if board is None or player is None:
        return 0.0

    starts = _player_network_intersections(player)
    if not starts:
        return 0.0

    best_distance = inf
    for idx, inter in enumerate(getattr(board, "intersections", []) or []):
        if inter is None:
            continue
        port = normalize_port_type(getattr(inter, "port_type", ""))
        if not _port_matches_resource(port, bottleneck):
            continue
        for start in starts:
            best_distance = min(best_distance, _distance_between_intersections(board, start, idx))

    if not isfinite(best_distance):
        return 2.0

    # Scale road-distance into a small additive penalty. Distance 2-3 is realistic;
    # long paths become increasingly unlikely to be worth planning around.
    return min(3.0, max(0.0, best_distance - 1.0) / 2.0)


def dev_card_risk(way: VictoryWay) -> float:
    """
    Simple penalty for paths depending on development-card draws.

    This is not the full expected-cost model from the article. It is a tunable heuristic
    so VP-card-heavy/Biggest-Army-heavy paths do not look too attractive from raw cost alone.
    """
    risk = 0.0
    risk += 0.85 * way.victory_point_cards
    if way.biggest_army:
        risk += 2.25  # needs three knight cards and competition for LA
    risk += 0.10 * max(0, way.development_cards_to_buy - way.victory_point_cards)
    return risk


def longest_road_risk(player: Any, way: VictoryWay) -> float:
    if not way.longest_road:
        return 0.0
    if getattr(player, "longest_route_tf", False):
        return 0.0
    route_size = float(getattr(player, "size_longest_route", 0) or 0)
    return max(0.0, 5.0 - route_size) / 2.0


def largest_army_risk(player: Any, way: VictoryWay) -> float:
    if not way.biggest_army:
        return 0.0
    if getattr(player, "largest_army_tf", False):
        return 0.0
    army_size = float(getattr(player, "size_largest_army", 0) or 0)
    return max(0.0, 3.0 - army_size) / 1.5


def evaluate_way_with_vectors(
    way: VictoryWay,
    *,
    production_vector: Sequence[float],
    ports: Iterable[str] = (),
    current_hand: Optional[Sequence[float]] = None,
    player: Any = None,
    weights: Optional[Dict[str, float]] = None,
    rolls_per_player_turn: int = NUM_PLAYERS,
) -> WayEvaluation:
    """Evaluate one way using explicit production/port vectors."""
    weights = {**DEFAULT_EVALUATION_WEIGHTS, **(weights or {})}
    ports_tuple = tuple(dict.fromkeys(normalize_port_type(p) for p in ports if normalize_port_type(p)))
    trade_rates = tuple(get_trade_rates(ports_tuple))
    production = tuple(_clean_vector(production_vector))
    need = tuple(_clean_vector(way.need_vector))
    current_hand_v = _clean_vector(current_hand or [0.0] * 5)

    expected_turns = turns_to_afford_with_trading(
        need,
        production,
        trade_rates,
        current_hand=current_hand_v,
        rolls_per_player_turn=rolls_per_player_turn,
    )
    mismatch, bottleneck = resource_fit_components(need, production)
    resource_bottleneck = bottleneck
    bottleneck_name = main_bottleneck_resource(need, production)
    board_for_player = getattr(getattr(player, "game", None), "board", None) if player is not None else None
    port_distance = estimate_port_distance_penalty(board_for_player, player, way, production, ports_tuple)
    dc_risk = dev_card_risk(way)
    lr_risk = longest_road_risk(player, way) if player is not None else (2.5 if way.longest_road else 0.0)
    la_risk = largest_army_risk(player, way) if player is not None else (2.0 if way.biggest_army else 0.0)

    if isfinite(expected_turns):
        final_score = expected_turns
        # Version-1 formula, kept explicit for readability and tuning.
        final_score += weights["dev_card_risk"] * dc_risk
        final_score += weights["longest_road_risk"] * lr_risk
        final_score += weights["largest_army_risk"] * la_risk
        final_score += weights["port_distance_penalty"] * port_distance
        final_score += weights["resource_bottleneck_penalty"] * resource_bottleneck
        # Optional experimental mismatch term; default weight is 0.0.
        final_score += weights.get("resource_mismatch", 0.0) * mismatch
        reason = "finite"
    else:
        final_score = inf
        reason = "insufficient production to cover need"

    return WayEvaluation(
        way_id=way.way_id,
        final_score=final_score,
        expected_turns=expected_turns,
        resource_mismatch=mismatch,
        bottleneck_pressure=bottleneck,
        resource_bottleneck_penalty=resource_bottleneck,
        port_distance_penalty=port_distance,
        main_bottleneck_resource=bottleneck_name,
        dev_card_risk=dc_risk,
        longest_road_risk=lr_risk,
        largest_army_risk=la_risk,
        need_vector=need,
        production_vector=production,
        trade_rates=trade_rates,
        owned_ports=ports_tuple,
        reason=reason,
    )


def evaluate_way_for_player(
    board: Any,
    player: Any,
    way: VictoryWay,
    *,
    extra_production: Optional[Sequence[float]] = None,
    extra_ports: Optional[Iterable[str]] = None,
    include_current_hand: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> WayEvaluation:
    """Evaluate one victory way for a player's current board position."""
    production = get_player_production_vector(board, player)
    if extra_production is not None:
        production = vector_add(production, extra_production)

    ports = get_player_ports(board, player)
    if extra_ports:
        for port in extra_ports:
            port = normalize_port_type(port)
            if port and port not in ports:
                ports.append(port)

    hand = get_player_resource_cards_vector(player) if include_current_hand else [0.0] * 5

    return evaluate_way_with_vectors(
        way,
        production_vector=production,
        ports=ports,
        current_hand=hand,
        player=player,
        weights=weights,
    )


def rank_ways_for_player(
    board: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    top_n: Optional[int] = None,
    extra_production: Optional[Sequence[float]] = None,
    extra_ports: Optional[Iterable[str]] = None,
    include_current_hand: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> List[WayEvaluation]:
    """Return victory ways sorted by final_score ascending."""
    evaluations = [
        evaluate_way_for_player(
            board,
            player,
            way,
            extra_production=extra_production,
            extra_ports=extra_ports,
            include_current_hand=include_current_hand,
            weights=weights,
        )
        for way in ways
    ]
    evaluations.sort(key=lambda ev: (not isfinite(ev.final_score), ev.final_score, ev.expected_turns, ev.way_id))
    if top_n is not None:
        return evaluations[:top_n]
    return evaluations


# ──────────────────────────────────────────────────────────────────────────────
# Port value and settlement-placement helpers
# ──────────────────────────────────────────────────────────────────────────────


def port_value_for_way(
    way: VictoryWay,
    production_vector: Sequence[float],
    port_type: str,
    *,
    current_hand: Optional[Sequence[float]] = None,
    existing_ports: Iterable[str] = (),
) -> float:
    """
    Estimate how many expected turns a port saves for a way.

    Positive means the port improves the path. Zero means no measurable improvement.
    """
    port_type = normalize_port_type(port_type)
    if not port_type:
        return 0.0

    base_rates = get_trade_rates(existing_ports)
    with_port_rates = get_trade_rates(list(existing_ports) + [port_type])

    turns_without = turns_to_afford_with_trading(
        way.need_vector,
        production_vector,
        base_rates,
        current_hand=current_hand,
    )
    turns_with = turns_to_afford_with_trading(
        way.need_vector,
        production_vector,
        with_port_rates,
        current_hand=current_hand,
    )

    if not isfinite(turns_without) and isfinite(turns_with):
        return 25.0  # useful but bounded for placement scoring
    if not isfinite(turns_without) or not isfinite(turns_with):
        return 0.0
    return max(0.0, turns_without - turns_with)


def is_initial_intersection_buildable(board: Any, inter_id: int) -> bool:
    """
    Lightweight buildability check for initial placement.

    Prefer Game.can_build_intersection_tf when available. This helper is intentionally
    conservative and relies on board.occupy_intersection having blocked neighbors.
    """
    if inter_id < 0 or inter_id >= len(board.intersections):
        return False
    inter = board.intersections[inter_id]
    if inter is None:
        return False
    if inter_id in getattr(board, "INTERSECTION_IN_WATER", []):
        return False
    if getattr(inter, "occupied_tf", False):
        return False
    if hasattr(inter, "can_build_tf") and not getattr(inter, "can_build_tf"):
        return False

    # Extra safety: reject distance-1 from occupied intersections if neighbor data exists.
    for neighbor_id in getattr(inter, "three_intersection_ids", []):
        if 0 <= neighbor_id < len(board.intersections):
            neighbor = board.intersections[neighbor_id]
            if neighbor is not None and getattr(neighbor, "occupied_tf", False):
                return False
    return True


def valid_initial_intersections(board: Any) -> List[int]:
    return [i for i in range(len(board.intersections)) if is_initial_intersection_buildable(board, i)]


def blocked_neighbor_pips(board: Any, inter_id: int) -> float:
    """Pip value of immediately blocked neighboring intersections."""
    if inter_id < 0 or inter_id >= len(board.intersections):
        return 0.0
    inter = board.intersections[inter_id]
    if inter is None:
        return 0.0
    total = 0.0
    for neighbor_id in getattr(inter, "three_intersection_ids", []):
        if is_initial_intersection_buildable(board, neighbor_id):
            total += vector_sum(get_intersection_resource_pips(board, neighbor_id))
    return total


def score_initial_settlement_candidate(
    board: Any,
    player: Any,
    inter_id: int,
    ways: Sequence[VictoryWay],
    *,
    top_k: int = 5,
    placement_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
    include_current_hand: bool = True,
) -> PlacementEvaluation:
    """
    Score one possible settlement placement for the current player.

    This does not mutate board. It evaluates the player's best/top-k ways *after* adding
    the candidate settlement production and port.
    """
    placement_weights = {**DEFAULT_PLACEMENT_WEIGHTS, **(placement_weights or {})}

    prod_added = get_intersection_resource_pips(board, inter_id)
    port_added = get_intersection_port(board, inter_id)
    raw_pips = vector_sum(prod_added)

    evals_after = rank_ways_for_player(
        board,
        player,
        ways,
        top_n=None,
        extra_production=prod_added,
        extra_ports=[port_added] if port_added else [],
        include_current_hand=include_current_hand,
        weights=eval_weights,
    )
    finite_evals = [ev for ev in evals_after if isfinite(ev.final_score)]

    if finite_evals:
        best = finite_evals[0]
        top = finite_evals[: max(1, top_k)]
        top_avg = sum(ev.final_score for ev in top) / len(top)
    else:
        best = evals_after[0]
        top_avg = inf

    # Demand-aware port value based on best path after adding this settlement.
    existing_ports = get_player_ports(board, player)
    current_hand = get_player_resource_cards_vector(player) if include_current_hand else [0.0] * 5
    prod_total_after = vector_add(get_player_production_vector(board, player), prod_added)
    p_value = port_value_for_way(
        next((w for w in ways if w.way_id == best.way_id), ways[0]),
        prod_total_after,
        port_added,
        current_hand=current_hand,
        existing_ports=existing_ports,
    )

    neighbor_block = blocked_neighbor_pips(board, inter_id)

    best_viability = viability_from_score(best.final_score)
    top5_viability = viability_from_score(top_avg)
    diversity = float(resource_diversity(prod_added))

    if isfinite(best.final_score):
        # Revised single-vertex score: production and flexibility dominate;
        # port/block value remains useful but cannot swamp poor own production.
        placement_score = 0.0
        placement_score += placement_weights.get("raw_pips", 1.0) * raw_pips
        placement_score += placement_weights.get("resource_diversity", 1.25) * diversity
        placement_score += placement_weights.get("best_way_viability", 2.0) * best_viability
        placement_score += placement_weights.get("top5_way_viability", 4.0) * top5_viability
        placement_score += placement_weights.get("port_value", 0.10) * p_value
        placement_score += placement_weights.get("blocked_neighbor_pips", 0.03) * neighbor_block
        # Backward-compatible optional aliases, defaulting to zero in v2.
        placement_score += placement_weights.get("own_best_way", 0.0) * best_viability
        placement_score += placement_weights.get("own_top5_average", 0.0) * top5_viability
    else:
        placement_score = -inf

    return PlacementEvaluation(
        intersection_id=inter_id,
        placement_score=placement_score,
        best_way_id=best.way_id,
        best_way_score_after=best.final_score,
        best_way_expected_turns_after=best.expected_turns,
        top5_average_score_after=top_avg,
        production_added=tuple(_clean_vector(prod_added)),
        port_added=port_added,
        raw_pips_added=raw_pips,
        port_value=p_value,
        blocked_neighbor_pips=neighbor_block,
        resource_diversity=diversity,
        best_way_viability=best_viability,
        top5_way_viability=top5_viability,
        best_way_reason=best.reason,
    )


def rank_initial_settlement_locations(
    board: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
    top_n: Optional[int] = 15,
    top_k_ways: int = 5,
    placement_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
    include_current_hand: bool = True,
) -> List[PlacementEvaluation]:
    """Rank buildable initial settlement intersections for one player."""
    candidate_ids = list(valid_intersections) if valid_intersections is not None else valid_initial_intersections(board)
    results = [
        score_initial_settlement_candidate(
            board,
            player,
            inter_id,
            ways,
            top_k=top_k_ways,
            placement_weights=placement_weights,
            eval_weights=eval_weights,
            include_current_hand=include_current_hand,
        )
        for inter_id in candidate_ids
    ]
    results.sort(key=lambda ev: (not isfinite(ev.placement_score), -ev.placement_score, ev.intersection_id))
    if top_n is not None:
        return results[:top_n]
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Two-settlement opening-pair helpers
# ──────────────────────────────────────────────────────────────────────────────


def are_initial_intersections_compatible(board: Any, first_id: int, second_id: int) -> bool:
    """True if two initial settlements can coexist under the distance rule."""
    if first_id == second_id:
        return False
    if not is_initial_intersection_buildable(board, first_id):
        return False
    if not is_initial_intersection_buildable(board, second_id):
        return False
    first = board.intersections[first_id]
    if first is None:
        return False
    if second_id in getattr(first, "three_intersection_ids", []):
        return False
    if hasattr(board, "_distance_between_intersections"):
        try:
            dist = board._distance_between_intersections(int(first_id), int(second_id))
            if dist is not None and dist <= 1:
                return False
        except Exception:
            pass
    return True


def blocked_intersections_by_settlements(board: Any, settlement_ids: Sequence[int]) -> List[int]:
    """Unique buildable intersections denied by placing the given settlements."""
    blocked: set[int] = set()
    settlement_set = {int(i) for i in settlement_ids}
    for inter_id in settlement_set:
        if inter_id < 0 or inter_id >= len(board.intersections):
            continue
        inter = board.intersections[inter_id]
        if inter is None:
            continue
        for neighbor_id in getattr(inter, "three_intersection_ids", []):
            if neighbor_id in settlement_set:
                continue
            if is_initial_intersection_buildable(board, int(neighbor_id)):
                blocked.add(int(neighbor_id))
    return sorted(blocked)


def blocked_neighbor_pips_for_settlements(board: Any, settlement_ids: Sequence[int]) -> float:
    return sum(vector_sum(get_intersection_resource_pips(board, inter_id)) for inter_id in blocked_intersections_by_settlements(board, settlement_ids))


def _unique_ports_from_intersections(board: Any, intersection_ids: Sequence[int]) -> List[str]:
    ports: List[str] = []
    for inter_id in intersection_ids:
        port = get_intersection_port(board, int(inter_id))
        if port and port not in ports:
            ports.append(port)
    return ports


def score_opening_pair_candidate(
    board: Any,
    player: Any,
    first_id: int,
    second_id: int,
    ways: Sequence[VictoryWay],
    *,
    top_k: int = 5,
    pair_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
    include_current_hand: bool = True,
) -> OpeningPairEvaluation:
    """Score a two-settlement initial-placement pair without mutating the board."""
    pair_weights = {**DEFAULT_OPENING_PAIR_WEIGHTS, **(pair_weights or {})}
    if not are_initial_intersections_compatible(board, first_id, second_id):
        zero = (0.0, 0.0, 0.0, 0.0, 0.0)
        return OpeningPairEvaluation(
            first_intersection_id=first_id,
            second_intersection_id=second_id,
            pair_score=-inf,
            best_way_id=-1,
            best_way_score_after=inf,
            best_way_expected_turns_after=inf,
            top5_average_score_after=inf,
            top5_way_ids=(),
            production_first=zero,
            production_second=zero,
            production_combined=zero,
            ports_added=(),
            raw_pips_combined=0.0,
            port_value=0.0,
            blocked_neighbor_pips=0.0,
            resource_diversity=0.0,
            best_way_viability=0.0,
            top5_way_viability=0.0,
            best_way_reason="incompatible opening pair",
        )

    prod_first = get_intersection_resource_pips(board, first_id)
    prod_second = get_intersection_resource_pips(board, second_id)
    prod_combined = vector_add(prod_first, prod_second)
    ports_added = _unique_ports_from_intersections(board, [first_id, second_id])

    evals_after = rank_ways_for_player(
        board,
        player,
        ways,
        top_n=None,
        extra_production=prod_combined,
        extra_ports=ports_added,
        include_current_hand=include_current_hand,
        weights=eval_weights,
    )
    finite_evals = [ev for ev in evals_after if isfinite(ev.final_score)]
    if finite_evals:
        best = finite_evals[0]
        top = finite_evals[: max(1, top_k)]
        top_avg = sum(ev.final_score for ev in top) / len(top)
        top_ids = tuple(ev.way_id for ev in top)
    else:
        best = evals_after[0]
        top_avg = inf
        top_ids = ()

    existing_ports = get_player_ports(board, player)
    current_hand = get_player_resource_cards_vector(player) if include_current_hand else [0.0] * 5
    prod_total_after = vector_add(get_player_production_vector(board, player), prod_combined)
    best_way = next((w for w in ways if w.way_id == best.way_id), ways[0])
    p_value = 0.0
    for port in ports_added:
        p_value = max(
            p_value,
            port_value_for_way(
                best_way,
                prod_total_after,
                port,
                current_hand=current_hand,
                existing_ports=existing_ports,
            ),
        )

    blocked = blocked_neighbor_pips_for_settlements(board, [first_id, second_id])
    raw_pips = vector_sum(prod_combined)
    diversity = float(resource_diversity(prod_combined))
    best_viability = viability_from_score(best.final_score)
    top5_viability = viability_from_score(top_avg)

    if isfinite(best.final_score):
        pair_score = 0.0
        pair_score += pair_weights.get("combined_pips", 1.0) * raw_pips
        pair_score += pair_weights.get("resource_diversity", 1.5) * diversity
        pair_score += pair_weights.get("best_way_viability", 2.0) * best_viability
        pair_score += pair_weights.get("top5_way_viability", 5.0) * top5_viability
        pair_score += pair_weights.get("port_value", 0.10) * p_value
        pair_score += pair_weights.get("blocked_neighbor_pips", 0.02) * blocked
    else:
        pair_score = -inf

    return OpeningPairEvaluation(
        first_intersection_id=first_id,
        second_intersection_id=second_id,
        pair_score=pair_score,
        best_way_id=best.way_id,
        best_way_score_after=best.final_score,
        best_way_expected_turns_after=best.expected_turns,
        top5_average_score_after=top_avg,
        top5_way_ids=top_ids,
        production_first=tuple(_clean_vector(prod_first)),
        production_second=tuple(_clean_vector(prod_second)),
        production_combined=tuple(_clean_vector(prod_combined)),
        ports_added=tuple(ports_added),
        raw_pips_combined=raw_pips,
        port_value=p_value,
        blocked_neighbor_pips=blocked,
        resource_diversity=diversity,
        best_way_viability=best_viability,
        top5_way_viability=top5_viability,
        best_way_reason=best.reason,
    )


def rank_opening_pairs(
    board: Any,
    player: Any,
    ways: Sequence[VictoryWay],
    *,
    valid_intersections: Optional[Sequence[int]] = None,
    top_n: Optional[int] = 15,
    top_k_ways: int = 5,
    pair_weights: Optional[Dict[str, float]] = None,
    eval_weights: Optional[Dict[str, float]] = None,
    include_current_hand: bool = True,
    ordered: bool = False,
) -> List[OpeningPairEvaluation]:
    """
    Rank two-settlement opening pairs.

    With ordered=False, (A,B) and (B,A) are considered the same pair and only one is kept.
    With ordered=True, both orders are returned; useful later when second-placement resources matter.
    """
    ids = list(valid_intersections) if valid_intersections is not None else valid_initial_intersections(board)
    results: List[OpeningPairEvaluation] = []
    for i, first_id in enumerate(ids):
        second_iter = ids if ordered else ids[i + 1 :]
        for second_id in second_iter:
            if first_id == second_id:
                continue
            if not are_initial_intersections_compatible(board, int(first_id), int(second_id)):
                continue
            results.append(
                score_opening_pair_candidate(
                    board,
                    player,
                    int(first_id),
                    int(second_id),
                    ways,
                    top_k=top_k_ways,
                    pair_weights=pair_weights,
                    eval_weights=eval_weights,
                    include_current_hand=include_current_hand,
                )
            )

    results.sort(
        key=lambda ev: (
            not isfinite(ev.pair_score),
            -ev.pair_score,
            ev.first_intersection_id,
            ev.second_intersection_id,
        )
    )
    if top_n is not None:
        return results[:top_n]
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Convenience formatting for logs/debug prints
# ──────────────────────────────────────────────────────────────────────────────


def vector_to_named_dict(vector: Sequence[float]) -> Dict[str, float]:
    return {RESOURCE_NAMES[i]: round(_clean_vector(vector)[i], 3) for i in range(5)}


def evaluations_to_rows(evaluations: Sequence[WayEvaluation]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in evaluations:
        row = ev.as_dict()
        row["production_named"] = vector_to_named_dict(ev.production_vector)
        row["need_named"] = vector_to_named_dict(ev.need_vector)
        rows.append(row)
    return rows


def placement_evaluations_to_rows(evaluations: Sequence[PlacementEvaluation]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in evaluations:
        row = ev.as_dict()
        row["production_added_named"] = vector_to_named_dict(ev.production_added)
        rows.append(row)
    return rows


def opening_pair_evaluations_to_rows(evaluations: Sequence[OpeningPairEvaluation]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ev in evaluations:
        row = ev.as_dict()
        row["production_first_named"] = vector_to_named_dict(ev.production_first)
        row["production_second_named"] = vector_to_named_dict(ev.production_second)
        row["production_combined_named"] = vector_to_named_dict(ev.production_combined)
        rows.append(row)
    return rows
