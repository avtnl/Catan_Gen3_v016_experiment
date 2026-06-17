import torch
from collections import defaultdict
import time
import math
from typing import Optional

try:
    from core.resource_time_estimator import (
        EXPECTED_HAND_CONFIDENCE_TARGET,
        EXPECTED_HAND_CONTINUOUS_TRADING,
        compute_payability_with_trades,
        estimate_expected_hand_after_turns,
        estimate_first_payable_turn,
        estimate_payability_confidence,
        target_cost_vector,
    )
except Exception:  # Allows isolated smoke tests before core is on sys.path.
    try:
        from resource_time_estimator import (
            EXPECTED_HAND_CONFIDENCE_TARGET,
            EXPECTED_HAND_CONTINUOUS_TRADING,
            compute_payability_with_trades,
            estimate_expected_hand_after_turns,
            estimate_first_payable_turn,
            estimate_payability_confidence,
            target_cost_vector,
        )
    except Exception:  # pragma: no cover
        EXPECTED_HAND_CONFIDENCE_TARGET = 0.90
        EXPECTED_HAND_CONTINUOUS_TRADING = True
        compute_payability_with_trades = None
        estimate_expected_hand_after_turns = None
        estimate_first_payable_turn = None
        estimate_payability_confidence = None
        target_cost_vector = None

class MarkovEvaluator:
    """
    Markov evaluator for Catan.

    Internal resource order used by the matrix model:
        0 = brick
        1 = lumber/wood
        2 = wool
        3 = wheat
        4 = ore

    Important:
    - Single-vertex production matrices stay in self.precomp_cache, keyed by int vertex id.
    - Full-position matrices are built from the FULL vertices multiset and cached separately
      using tuple(sorted(vertices)), which preserves duplicates.
    - A repeated vertex id therefore represents doubled production (city modeling).
    """

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ MarkovEvaluator using {self.device.upper()}")

        # Optional runtime links set later by the game/bootstrap code
        self.game = None
        self.board = None

        # Adaptive expanded-evaluator caches
        self.adaptive_matrix_cache = {}      # (position_key, cap_key) -> matrix
        self.adaptive_traded_ev_cache = {}   # (position_key, ports_key, target_type, cap_key) -> ev tuple

        # Fixed-cap 0..4 state space
        self.resource_poss = self._generate_states()
        self.state_to_index = {tuple(state): idx for idx, state in enumerate(self.resource_poss)}

        self.M_template = torch.zeros((3125, 3125), dtype=torch.float32, device="cpu")

        # Single-vertex caches
        self.vertex_rolls = {}          # vid -> list[list[int]]
        self.precomp_cache = {}         # vid -> production matrix

        # Full-position caches
        self.position_rolls_cache = {}  # tuple(sorted(vertices)) -> combined rolls
        self.position_matrix_cache = {} # tuple(sorted(vertices)) -> production matrix
        self.base_ev_cache = {}         # tuple(sorted(vertices)) -> expected_vectors(matrix)
        self.traded_ev_cache = {}       # (position_key, ports_key, target_type) -> expected_vectors(matrix)

        self.RES_NAMES = ["brick", "lumber", "wool", "wheat", "ore"]

        self.num_players = 4

    # ============================================================
    # Core state helpers
    # ============================================================
    def _generate_states(self):
        states = []
        for b in range(5):
            for l_ in range(5):
                for s in range(5):
                    for w in range(5):
                        for o in range(5):
                            states.append([b, l_, s, w, o])
        return states

    def _get_num_players(self) -> int:
        game = getattr(self, "game", None)
        players = getattr(game, "players", None) if game is not None else None
        try:
            return max(1, len(players))
        except Exception:
            return int(getattr(self, "num_players", 4) or 4)

    def _empty_rolls(self):
        return [[], [], [], [], []]

    def _normalize_rolls(self, rolls):
        """Ensure exactly 5 resource lists, preserving duplicates."""
        out = self._empty_rolls()
        if not isinstance(rolls, (list, tuple)):
            return out

        for i in range(min(5, len(rolls))):
            rlist = rolls[i]
            if isinstance(rlist, (list, tuple)):
                out[i] = [int(x) for x in rlist if isinstance(x, (int, float))]
        return out

    def _position_key(self, vertices):
        """
        Order-insensitive but duplicate-preserving key.
        Example:
            [12, 8, 12] -> (8, 12, 12)
        """
        return tuple(sorted(int(v) for v in vertices))

    def _ports_key(self, player_ports):
        """Hashable normalized key for port dictionaries."""
        if not player_ports:
            return ()
        return tuple(sorted((str(k).lower(), int(v)) for k, v in player_ports.items()))

    def _state_to_vec(self, state_idx: int) -> list[int]:
        """
        Convert linear index -> [brick, lumber, wool, wheat, ore].

        Index order follows _generate_states():
            idx = b*625 + l*125 + s*25 + w*5 + o
        """
        vec = [0] * 5
        for i in range(4, -1, -1):
            vec[i] = state_idx % 5
            state_idx //= 5
        return vec

    def _vec_to_state(self, vec: list[int]) -> int:
        """
        Convert [brick, lumber, wool, wheat, ore] -> linear index.

        Must match _generate_states():
            idx = b*625 + l*125 + s*25 + w*5 + o
        """
        b, l_, s, w, o = [min(max(0, int(x)), 4) for x in vec]
        return b * 625 + l_ * 125 + s * 25 + w * 5 + o

    def _game_hand_to_markov_vec(self, hand: list[int]) -> list[int]:
        """
        Convert game hand order:
            [Wheat, Ore, Wood, Brick, Wool]
        to Markov internal order:
            [Brick, Lumber/Wood, Wool, Wheat, Ore]
        """
        if hand is None:
            hand = [0, 0, 0, 0, 0]

        h = [0, 0, 0, 0, 0]
        for i in range(min(5, len(hand))):
            try:
                h[i] = min(max(0, int(hand[i])), 4)
            except Exception:
                h[i] = 0

        wheat, ore, wood, brick, wool = h
        return [brick, wood, wool, wheat, ore]

    def _hand_to_state_index(self, hand: list[int]) -> int:
        markov_vec = self._game_hand_to_markov_vec(hand)
        return self._vec_to_state(markov_vec)

    def _get_target_requirements(self, target_type: str) -> list[int]:
        """
        Resource order here is Markov internal order:
        [brick, wood/lumber, wool, wheat, ore]
        """
        t = str(target_type or "").strip().lower()

        if t in ("settlement", "next_settlement", "settlement_0r"):
            return [1, 1, 1, 1, 0]

        elif t in ("new_settlement", "settlement_1r"):
            # settlement + 1 road
            return [2, 2, 1, 1, 0]

        elif t == "settlement_2r":
            # settlement + 2 roads
            return [3, 3, 1, 1, 0]

        elif t in ("city", "upgrade_to_city"):
            return [0, 0, 0, 2, 3]

        elif t in ("dev_card", "buy_discovery_card"):
            return [0, 0, 1, 1, 1]

        elif t in ("dev_card_4", "4x_dev_card", "buy_4_discovery_cards"):
            return [0, 0, 4, 4, 4]

        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    def _normalize_target_type(self, strategy: str) -> str:
        if not strategy:
            return "settlement"

        s = str(strategy).strip().lower()

        if s == "best":
            return "settlement"
        if s in ("settlement", "next_settlement"):
            return "settlement"
        if s == "new_settlement":
            return "new_settlement"
        if s == "city":
            return "city"
        if s == "dev_card":
            return "dev_card"
        if s in ("4x_dev_card", "dev_card_4"):
            return "dev_card_4"

        return "settlement"

    def _hand_to_state_index_with_caps(self, hand, cap_vec):
        markov_vec = self._game_hand_to_markov_vec(hand)
        capped_vec = [
            min(max(0, int(markov_vec[i])), int(cap_vec[i]))
            for i in range(5)
        ]
        return self._vec_to_state_with_caps(capped_vec, cap_vec)
    

    # ============================================================
    # Trading layer
    # ============================================================
    def apply_trading_layer(
        self,
        M_original: torch.Tensor,
        player_ports: dict = None,
        target_type: str = "settlement",
        use_bank_4to1: bool = True,
        max_trades_per_roll: int = 4,
    ) -> torch.Tensor:
        if player_ports is None:
            player_ports = {}

        n_states = 3125
        M_new = torch.zeros((n_states, n_states), dtype=torch.float32, device=self.device)
        req = self._get_target_requirements(target_type)
        M_orig = M_original.to(self.device)

        for i in range(n_states):
            for j in range(n_states):
                prob = M_orig[i, j]
                if prob <= 0.0:
                    continue

                vec = self._state_to_vec(j)
                trades_made = 0

                while trades_made < max_trades_per_roll:
                    best_gain = -1
                    best_new_vec = None

                    for r in range(5):
                        if self.RES_NAMES[r] in player_ports:
                            ratio = player_ports[self.RES_NAMES[r]]
                        elif "generic" in player_ports:
                            ratio = player_ports["generic"]
                        elif use_bank_4to1:
                            ratio = 4
                        else:
                            continue

                        if vec[r] < ratio:
                            continue

                        deficit = [max(0, req[k] - vec[k]) for k in range(5)]

                        for target_r in range(5):
                            if target_r == r or deficit[target_r] == 0:
                                continue

                            new_vec = vec[:]
                            new_vec[r] -= ratio
                            new_vec[target_r] += 1
                            new_vec = [min(4, x) for x in new_vec]

                            new_deficit = [max(0, req[k] - new_vec[k]) for k in range(5)]
                            gain = sum(d - nd for d, nd in zip(deficit, new_deficit))

                            if gain > best_gain:
                                best_gain = gain
                                best_new_vec = new_vec

                    if best_gain <= 0 or best_new_vec is None:
                        break

                    vec = best_new_vec
                    trades_made += 1

                k = self._vec_to_state(vec)
                M_new[i, k] += prob

            row_sum = M_new[i].sum()
            if abs(float(row_sum) - 1.0) > 1e-9:
                M_new[i, i] += (1.0 - row_sum)

        return M_new

    # ============================================================
    # Matrix construction
    # ============================================================
    def build_matrix(self, die_num):
        """
        Build one production transition matrix from resource roll lists
        in internal order:
            0 brick, 1 lumber, 2 wool, 3 wheat, 4 ore
        """
        die_num = self._normalize_rolls(die_num)
        die_prob = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

        M = self.M_template.clone().to(self.device)
        roll_list = defaultdict(lambda: [0] * 5)

        for res_idx, rolls in enumerate(die_num):
            for roll in rolls:
                roll_list[int(roll)][res_idx] += 1

        len_list = [[] for _ in range(6)]
        for roll, gains in roll_list.items():
            count = sum(1 for g in gains if g > 0)
            len_list[count].append((roll, gains))

        for idx, current in enumerate(self.resource_poss):
            count = 0
            for n_res in range(1, 6):
                for roll, gains in len_list[n_res]:
                    next_state = current[:]
                    for r in range(5):
                        if gains[r]:
                            next_state[r] = min(4, next_state[r] + gains[r])

                    next_idx = self.state_to_index[tuple(next_state)]
                    p = die_prob.get(int(roll), 0)
                    M[idx, next_idx] += p
                    count += p

            M[idx, idx] += 36 - count

        return (M / 36.0).to(self.device)

    # ============================================================
    # Expected values
    # ============================================================
    def expected_vectors(self, matrix: torch.Tensor):
        """
        Compute expected hitting-time vectors for all supported target events.

        Returns, in this order:
            1) settlement_0r
            2) settlement_1r
            3) settlement_2r
            4) city
            5) dev_card
            6) dev_card_4

        Important:
        - This solves true expected time-to-hit-target.
        - Target-satisfying states have expected time 0.
        - Non-target states solve:
            t = 1 + Q_non_target @ t
        """
        matrix = matrix.to(self.device).float()

        row_sums = matrix.sum(dim=1, keepdim=True)
        mask = row_sums > 1e-12
        matrix = torch.where(mask, matrix / row_sums, matrix)

        n_states = matrix.shape[0]
        I_full = torch.eye(n_states, device=self.device, dtype=torch.float32)

        def _expected_hitting_time_for_target(target_type: str) -> torch.Tensor:
            target_vec = self._build_target_vector(target_type).to(self.device).bool()

            # Already-satisfied states take 0 turns.
            out = torch.zeros(n_states, device=self.device, dtype=torch.float32)

            non_target_idx = torch.where(~target_vec)[0]

            if non_target_idx.numel() == 0:
                return out

            # Q restricted to non-target states only.
            Q = matrix.index_select(0, non_target_idx).index_select(1, non_target_idx)
            n = Q.shape[0]
            I = torch.eye(n, device=self.device, dtype=torch.float32)
            ones = torch.ones(n, device=self.device, dtype=torch.float32)

            try:
                t = torch.linalg.solve(I - Q, ones)
            except Exception:
                t = torch.linalg.pinv(I - Q) @ ones

            t = torch.nan_to_num(
                t,
                nan=9999.0,
                posinf=9999.0,
                neginf=9999.0,
            )

            # Keep runaway / unreachable fixed-cap results bounded but obvious.
            t = torch.clamp(t, min=0.0, max=9999.0)

            out[non_target_idx] = t
            return out

        e_settlement_0r = _expected_hitting_time_for_target("settlement_0r")
        e_settlement_1r = _expected_hitting_time_for_target("settlement_1r")
        e_settlement_2r = _expected_hitting_time_for_target("settlement_2r")
        e_city = _expected_hitting_time_for_target("city")
        e_dev = _expected_hitting_time_for_target("dev_card")
        e_dev4 = _expected_hitting_time_for_target("dev_card_4")

        return (
            e_settlement_0r,
            e_settlement_1r,
            e_settlement_2r,
            e_city,
            e_dev,
            e_dev4,
        )

    # ============================================================
    # Precompute single-vertex data
    # ============================================================
    def precompute_game(self, vertex_to_rolls):
        """
        Precompute only single-vertex production matrices.
        Full-position models are built later from the full vertex multiset.
        """
        start_time = time.time()
        print(
            f"🚀 Precomputing Markov production matrices for {len(vertex_to_rolls)} vertices... "
            f"started at {time.strftime('%H:%M:%S')}"
        )

        # Reset single-vertex caches
        self.vertex_rolls = {}
        self.precomp_cache = {}

        # Reset full-position fixed-cap caches
        self.position_rolls_cache = {}
        self.position_matrix_cache = {}
        self.base_ev_cache = {}
        self.traded_ev_cache = {}

        # Reset adaptive expanded-evaluator caches too
        self.adaptive_matrix_cache = {}
        self.adaptive_traded_ev_cache = {}

        for vid, rolls in vertex_to_rolls.items():
            vid = int(vid)
            norm_rolls = self._normalize_rolls(rolls)
            self.vertex_rolls[vid] = norm_rolls
            self.precomp_cache[vid] = self.build_matrix(norm_rolls)

        duration = time.time() - start_time
        print(
            f"✅ Precomputation finished at {time.strftime('%H:%M:%S')} — "
            f"Duration: {duration:.1f} seconds"
        )
        print(f"   {len(self.precomp_cache)} vertices ready (single-vertex production cached)")

    # ============================================================
    # Full-position aggregation
    # ============================================================
    def _combine_vertex_rolls(self, vertices):
        """
        Combine roll production from the FULL vertex multiset.

        Duplicates are preserved on purpose:
        - [12, 12] means vertex 12 counts twice (city-like doubled production)
        """
        position_key = self._position_key(vertices)
        if position_key in self.position_rolls_cache:
            return self.position_rolls_cache[position_key]

        combined = self._empty_rolls()

        for vid in position_key:
            vrolls = self.vertex_rolls.get(int(vid), self._empty_rolls())
            for r in range(5):
                combined[r].extend(vrolls[r])

        self.position_rolls_cache[position_key] = combined
        return combined

    def _get_position_matrix(self, vertices):
        """Build or fetch the full-position production matrix."""
        position_key = self._position_key(vertices)
        if position_key in self.position_matrix_cache:
            return self.position_matrix_cache[position_key]

        # Fast path: single vertex can reuse the already precomputed matrix directly
        if len(position_key) == 1:
            vid = int(position_key[0])
            if vid in self.precomp_cache:
                M_prod = self.precomp_cache[vid]
                self.position_matrix_cache[position_key] = M_prod
                return M_prod

        rolls = self._combine_vertex_rolls(position_key)
        M_prod = self.build_matrix(rolls)
        self.position_matrix_cache[position_key] = M_prod
        return M_prod

    def _get_base_expected_vectors(self, vertices):
        """Expected vectors for the raw full-position production matrix."""
        position_key = self._position_key(vertices)
        if position_key in self.base_ev_cache:
            return self.base_ev_cache[position_key]

        M_prod = self._get_position_matrix(position_key)
        ev = self.expected_vectors(M_prod)
        self.base_ev_cache[position_key] = ev
        return ev

    def _get_traded_expected_vectors(self, vertices, player_ports, target_type):
        """
        Expected vectors for a traded full-position production matrix.
        Cached by (position_key, ports_key, target_type).

        Supported target_type values include:
            - settlement / settlement_0r
            - settlement_1r
            - settlement_2r
            - city
            - dev_card
            - dev_card_4
        """
        position_key = self._position_key(vertices)
        ports_key = self._ports_key(player_ports)
        cache_key = (position_key, ports_key, str(target_type))

        if cache_key in self.traded_ev_cache:
            return self.traded_ev_cache[cache_key]

        M_prod = self._get_position_matrix(position_key)
        M_trade = self.apply_trading_layer(
            M_prod,
            player_ports=player_ports or {},
            target_type=target_type,
        )

        ev = self.expected_vectors(M_trade)
        self.traded_ev_cache[cache_key] = ev
        return ev

    # ============================================================
    # Public scoring API
    # ============================================================
    def get_expected_turns_fast_initial_with_explanation(
        self,
        vertices,
        hand=None,
        player_ports=None,
        strategy="settlement",
        extra_roads_needed: int = 0,
    ):
        """
        Return:
            {
                "score": <float>,
                "explanation": <dict>
            }

        Uses the fast score plus a transparent approximate trade explanation.

        Enhancements:
        - normalizes fast-forward aliases consistently with get_expected_turns_fast_initial()
        - exposes richer overflow diagnostics for terminal validation
        - includes bounded-horizon trade-funding diagnostics from the stricter
        dominant-resource overflow trigger
        """
        hand = hand or [0, 0, 0, 0, 0]
        player_ports = player_ports or {}

        normalized_strategy = str(strategy or "").strip().lower()

        if normalized_strategy == "new_settlement":
            normalized_strategy = "settlement"
        elif normalized_strategy == "upgrade_to_city":
            normalized_strategy = "city"
        elif normalized_strategy == "buy_discovery_card":
            normalized_strategy = "dev_card"
        elif normalized_strategy == "buy_4_discovery_cards":
            normalized_strategy = "dev_card_4"

        score = self.get_expected_turns_fast_initial(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            strategy=normalized_strategy,
            extra_roads_needed=extra_roads_needed,
        )

        explanation = self._estimate_trade_plan(
            vertices=list(vertices),
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_strategy,
            extra_roads_needed=extra_roads_needed,
        )

        overflow_info = self._overflow_port_monopoly_score(
            vertices=list(vertices),
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_strategy,
            extra_roads_needed=extra_roads_needed,
        )

        overflow_triggered = bool(overflow_info.get("trigger", False))

        try:
            overflow_score = float(overflow_info.get("score", 9999.0))
        except Exception:
            overflow_score = 9999.0

        overflow_guard_used = (
            overflow_triggered
            and 0.0 <= overflow_score < 9999.0
            and abs(float(score) - overflow_score) < 1e-6
        )

        explanation["normalized_target_type"] = normalized_strategy

        # ------------------------------------------------------------
        # Existing overflow explanation fields
        # ------------------------------------------------------------
        explanation["overflow_triggered"] = overflow_triggered
        explanation["overflow_dominant_resource"] = overflow_info.get("dominant_resource")
        explanation["overflow_dominant_resource_pips"] = float(
            overflow_info.get("dominant_resource_pips", 0.0)
        )
        explanation["overflow_effective_trade_rate"] = int(
            overflow_info.get("effective_trade_rate", 4)
        )

        # Keep backward-compatible fields already used in your terminal/debug flow
        explanation["overflow_equivalent_needed"] = float(
            overflow_info.get("dominant_equivalent_needed", 0.0)
        )
        explanation["overflow_equivalent_raw"] = float(
            overflow_info.get("dominant_equivalent_raw", 0.0)
        )
        explanation["overflow_hand_surplus"] = float(
            overflow_info.get("dominant_surplus_in_hand", 0.0)
        )
        explanation["overflow_equivalent_turns"] = float(
            overflow_info.get("equivalent_turns", 9999.0)
        )
        explanation["overflow_blended_turns"] = float(
            overflow_info.get("blended_turns", 9999.0)
        )
        explanation["overflow_score"] = overflow_score
        explanation["overflow_guard_used"] = overflow_guard_used

        # ------------------------------------------------------------
        # New stricter bounded-horizon diagnostics
        # ------------------------------------------------------------
        explanation["overflow_needed_trades"] = int(
            overflow_info.get("needed_trades", 0)
        )
        explanation["overflow_off_resource_cards_to_buy"] = int(
            overflow_info.get("off_resource_cards_to_buy", 0)
        )
        explanation["overflow_dominant_direct_deficit"] = float(
            overflow_info.get("dominant_direct_deficit", 0.0)
        )
        explanation["overflow_dominant_trade_cards_needed_raw"] = float(
            overflow_info.get("dominant_trade_cards_needed_raw", 0.0)
        )
        explanation["overflow_dominant_trade_cards_needed_adjusted"] = float(
            overflow_info.get("dominant_trade_cards_needed_adjusted", 0.0)
        )
        explanation["overflow_expected_future_dominant_cards"] = float(
            overflow_info.get("expected_future_dominant_cards", 0.0)
        )
        explanation["overflow_expected_dominant_cards_by_horizon"] = float(
            overflow_info.get("expected_dominant_cards_by_horizon", 0.0)
        )
        explanation["overflow_horizon_turns"] = int(
            overflow_info.get("horizon_turns", 9)
        )
        explanation["overflow_horizon_rolls"] = float(
            overflow_info.get("horizon_rolls", 36.0)
        )
        explanation["overflow_can_fund_within_horizon"] = bool(
            overflow_info.get("can_fund_within_horizon", False)
        )
        explanation["overflow_cap_bind_risk"] = bool(
            overflow_info.get("cap_bind_risk", False)
        )
        explanation["overflow_weak_off_resource_exists"] = bool(
            overflow_info.get("weak_off_resource_exists", False)
        )

        # Helpful redundancy for downstream consumers / logs
        explanation["overflow_normalized_target_type"] = overflow_info.get(
            "normalized_target_type",
            normalized_strategy,
        )
        explanation["extra_roads_needed"] = int(extra_roads_needed)

        return {
            "score": float(score),
            "explanation": explanation,
        }

    def get_expected_time_to_event(self, vertices: list[int], hand: list[int], player_ports: dict) -> dict:
        """
        Return expected player-turns until each event from the CURRENT full position.

        Hybrid HEAVY policy:
        - immediate-affordability shortcut still applies
        - default path uses the existing traded 0..4 Markov evaluator
        - overflow-triggered targets are rerouted through the adaptive expanded
        dominant-resource evaluator with per-target caps

        Supported targets:
            - settlement / settlement_0r
            - settlement_1r
            - settlement_2r
            - city
            - dev_card
            - dev_card_4

        vertices:
            full owned vertex multiset; duplicates are preserved intentionally
        hand:
            game order [Wheat, Ore, Wood, Brick, Wool]
        player_ports:
            expected lowercase/internal trading dict, e.g.
                {"generic": 3}
                {"lumber": 2}
                {"brick": 2, "generic": 3}

        When ff_ignore_resource_cards is True, the hand is forced to zero and the
        immediate-affordability shortcuts are disabled.
        """
        if not vertices:
            return {
                "settlement": 9999.0,
                "settlement_0r": 9999.0,
                "settlement_1r": 9999.0,
                "settlement_2r": 9999.0,
                "city": 9999.0,
                "dev_card": 9999.0,
                "dev_card_4": 9999.0,
            }

        if self._ignore_resource_cards():
            hand = [0, 0, 0, 0, 0]
        else:
            hand = hand or [0, 0, 0, 0, 0]

        player_ports = player_ports or {}

        current_hand = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        def _can_afford_target_now(target_name: str) -> bool:
            if self._ignore_resource_cards():
                return False

            if target_name in ("settlement", "settlement_0r"):
                need = {"wheat": 1, "ore": 0, "wood": 1, "brick": 1, "wool": 1}
            elif target_name == "settlement_1r":
                need = {"wheat": 1, "ore": 0, "wood": 2, "brick": 2, "wool": 1}
            elif target_name == "settlement_2r":
                need = {"wheat": 1, "ore": 0, "wood": 3, "brick": 3, "wool": 1}
            elif target_name == "city":
                need = {"wheat": 2, "ore": 3, "wood": 0, "brick": 0, "wool": 0}
            elif target_name == "dev_card":
                need = {"wheat": 1, "ore": 1, "wood": 0, "brick": 0, "wool": 1}
            elif target_name == "dev_card_4":
                need = {"wheat": 4, "ore": 4, "wood": 4, "brick": 4, "wool": 4}
            else:
                return False

            return all(current_hand.get(res, 0) >= amt for res, amt in need.items())

        fixed_idx = self._hand_to_state_index(hand)

        # v014 adaptive-heavy diagnostics.
        # Filled by _score_target(...) and attached to the return dict as "__debug__".
        debug_by_target = {}

        def _score_target(target_name: str, vec_idx: int, extra_roads_needed: int = 0) -> float:
            # ------------------------------------------------------------
            # 0) Immediate affordability shortcut
            # ------------------------------------------------------------
            if _can_afford_target_now(target_name):
                debug_by_target[target_name] = {
                    "target": target_name,
                    "immediate_affordable": True,
                    "overflow_triggered": False,
                    "adaptive_attempted": False,
                    "adaptive_used": False,
                    "adaptive_score": 0.0,
                    "fixed_cap_score": 0.0,
                    "overflow_fallback_score": 9999.0,
                    "adaptive_cap_vec": None,
                    "adaptive_state_count": 3125,
                    "adaptive_error": None,
                }
                return 0.0

            # ------------------------------------------------------------
            # 1) Detect whether this target deserves adaptive overflow handling
            # ------------------------------------------------------------
            overflow_info = self._dominant_port_overflow_trigger(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                target_type=target_name,
                extra_roads_needed=extra_roads_needed,
            )

            overflow_triggered = bool(overflow_info.get("trigger", False))

            # Keep this OFF unless Update 11 / constants flag has been added.
            try:
                from core.constants import MARKOV_USE_ADAPTIVE_HEAVY
                use_adaptive = bool(MARKOV_USE_ADAPTIVE_HEAVY and overflow_triggered)
            except Exception:
                use_adaptive = False

            try:
                overflow_fallback_score = float(overflow_info.get("score", 9999.0))
            except Exception:
                overflow_fallback_score = 9999.0

            target_debug = {
                "target": target_name,
                "immediate_affordable": False,
                "overflow_triggered": overflow_triggered,
                "overflow_dominant_resource": overflow_info.get("dominant_resource"),
                "overflow_dominant_resource_pips": float(
                    overflow_info.get("dominant_resource_pips", 0.0)
                ),
                "overflow_effective_trade_rate": int(
                    overflow_info.get("effective_trade_rate", 4)
                ),
                "overflow_fallback_score": overflow_fallback_score,
                "adaptive_attempted": bool(use_adaptive),
                "adaptive_used": False,
                "adaptive_score": 9999.0,
                "adaptive_cap_vec": None,
                "adaptive_state_count": 3125,
                "adaptive_error": None,
                "fixed_cap_score": 9999.0,
            }

            # ------------------------------------------------------------
            # 2) Always compute fixed-cap HEAVY score first
            #    This preserves existing behavior and gives us a comparison value.
            # ------------------------------------------------------------
            ev_fixed = self._get_traded_expected_vectors(vertices, player_ports, target_name)
            idx_clamped = min(fixed_idx, len(ev_fixed[vec_idx]) - 1)
            fixed_score = float(ev_fixed[vec_idx][idx_clamped])

            target_debug["fixed_cap_score"] = fixed_score

            # ------------------------------------------------------------
            # 3) Adaptive expanded evaluator for overflow-triggered cases
            # ------------------------------------------------------------
            if use_adaptive:
                cap_vec = self._choose_adaptive_cap_vec(overflow_info, target_name)
                cap_vec = [int(x) for x in cap_vec]

                target_debug["adaptive_cap_vec"] = list(cap_vec)

                try:
                    target_debug["adaptive_state_count"] = int(
                        self._num_states_from_caps(cap_vec)
                    )
                except Exception:
                    target_debug["adaptive_state_count"] = -1

                try:
                    ev_adaptive = self._get_traded_expected_vectors_adaptive(
                        vertices=vertices,
                        player_ports=player_ports,
                        target_type=target_name,
                        cap_vec=cap_vec,
                    )

                    adaptive_idx = self._hand_to_state_index_with_caps(hand, cap_vec)
                    adaptive_idx = min(adaptive_idx, len(ev_adaptive[vec_idx]) - 1)

                    adaptive_score = float(ev_adaptive[vec_idx][adaptive_idx])

                    target_debug["adaptive_score"] = adaptive_score
                    target_debug["adaptive_used"] = True
                    debug_by_target[target_name] = target_debug

                    print(
                        f"   Markov HEAVY adaptive {self._position_key(vertices)} "
                        f"| target={target_name} | caps={tuple(cap_vec)} "
                        f"| dominant={overflow_info.get('dominant_resource')} "
                        f"@{float(overflow_info.get('dominant_resource_pips', 0.0)):.1f}pips "
                        f"| rate={int(overflow_info.get('effective_trade_rate', 4))}:1 "
                        f"| fixed_score={fixed_score:.2f} "
                        f"| adaptive_score={adaptive_score:.2f}"
                    )

                    return adaptive_score

                except Exception as exc:
                    target_debug["adaptive_error"] = str(exc)
                    target_debug["adaptive_used"] = False

                    print(
                        f"   Markov HEAVY adaptive FAILED {self._position_key(vertices)} "
                        f"| target={target_name} | fallback_to_fixed_cap | error={exc}"
                    )

            # ------------------------------------------------------------
            # 4) Default / fallback behavior
            # ------------------------------------------------------------
            fallback_score = fixed_score
            fallback_source = "fixed_cap"

            # If adaptive was attempted but failed, prefer the existing overflow fallback
            # only when it produced a sane finite score.
            if use_adaptive and 0.0 <= overflow_fallback_score < 9999.0:
                fallback_score = overflow_fallback_score
                fallback_source = "overflow_fallback"

            target_debug["fallback_score"] = float(fallback_score)
            target_debug["fallback_source"] = fallback_source

            debug_by_target[target_name] = target_debug

            if use_adaptive:
                print(
                    f"   Markov HEAVY fallback {self._position_key(vertices)} "
                    f"| target={target_name} "
                    f"| source={fallback_source} "
                    f"| fixed_score={fixed_score:.2f} "
                    f"| overflow_fallback_score={overflow_fallback_score:.2f} "
                    f"| final={fallback_score:.2f}"
                )

            return fallback_score

        settlement_0r_time = _score_target("settlement_0r", 0, extra_roads_needed=0)
        settlement_1r_time = _score_target("settlement_1r", 1, extra_roads_needed=1)
        settlement_2r_time = _score_target("settlement_2r", 2, extra_roads_needed=2)
        city_time = _score_target("city", 3, extra_roads_needed=0)
        dev_time = _score_target("dev_card", 4, extra_roads_needed=0)
        dev4_time = _score_target("dev_card_4", 5, extra_roads_needed=0)

        result = {
            "settlement": settlement_0r_time,
            "settlement_0r": settlement_0r_time,
            "settlement_1r": settlement_1r_time,
            "settlement_2r": settlement_2r_time,
            "city": city_time,
            "dev_card": dev_time,
            "dev_card_4": dev4_time,
        }

        # v014: attach diagnostics without changing the existing numeric keys.
        result["__debug__"] = debug_by_target

        # Also store the latest diagnostics on the evaluator for manual inspection.
        self.last_heavy_markov_debug = debug_by_target

        return result
    
    def _simple_port_bonus(self, vertices, player_ports) -> float:
        """
        Lightweight initial-placement harbor bonus.

        Uses the most recently added candidate vertex and a simple pip-based bonus.
        This is intentionally cheap and is only meant for setup scoring.
        """
        if not player_ports or not vertices or self.board is None:
            return 0.0

        inter_id = int(vertices[-1])
        if inter_id < 0 or inter_id >= len(self.board.intersections):
            return 0.0

        inter = self.board.intersections[inter_id]
        if inter is None:
            return 0.0

        # Board pip order is game order:
        # [Wheat, Ore, Wood, Brick, Wool]
        pips_game_order = getattr(inter, "all_tile_pips", None)
        if not pips_game_order:
            pips_game_order = getattr(inter, "three_tile_pips", [0.0] * 5)

        try:
            pips_game_order = [float(x) for x in pips_game_order]
        except Exception:
            pips_game_order = [0.0] * 5

        total_pips = sum(pips_game_order)
        bonus = 0.0

        # Generic 3:1 port
        if "generic" in player_ports:
            # modest bonus only
            extra_trades = total_pips / 3.0
            bonus = extra_trades * 0.1125

        else:
            # Specific 2:1 port using INTERNAL resource names from get_player_ports_dict()
            internal_to_game_idx = {
                "wheat": 0,
                "ore": 1,
                "lumber": 2,
                "wood": 2,
                "brick": 3,
                "wool": 4,
            }

            for res_name, idx in internal_to_game_idx.items():
                if res_name in player_ports:
                    resource_pips = pips_game_order[idx] if idx < len(pips_game_order) else 0.0
                    extra_trades = resource_pips / 2.0
                    bonus = extra_trades * 0.1625
                    break

        # Keep the bonus bounded so setup scoring stays stable
        return min(float(bonus), 6.0)

    def get_expected_turns_fast_initial(
        self,
        vertices,
        hand=None,
        player_ports=None,
        strategy="settlement_0r",
        extra_roads_needed: int = 0,
    ):
        """
        Fast scoring path for Initial Placement / fast-forward candidate ranking.

        Adds expected-hand/trade feasibility diagnostics and clamps impossible
        early FAST scores upward to the first feasible whole round.
        """
        if not vertices:
            return 9999.0

        if self._ignore_resource_cards():
            hand = [0, 0, 0, 0, 0]
        else:
            hand = hand or [0, 0, 0, 0, 0]

        player_ports = player_ports or {}

        normalized_strategy = str(strategy or "").strip().lower()

        if normalized_strategy == "new_settlement":
            normalized_strategy = "settlement"
        elif normalized_strategy == "upgrade_to_city":
            normalized_strategy = "city"
        elif normalized_strategy == "buy_discovery_card":
            normalized_strategy = "dev_card"
        elif normalized_strategy == "buy_4_discovery_cards":
            normalized_strategy = "dev_card_4"

        if normalized_strategy == "settlement":
            if extra_roads_needed <= 0:
                normalized_strategy = "settlement_0r"
            elif extra_roads_needed == 1:
                normalized_strategy = "settlement_1r"
            elif extra_roads_needed == 2:
                normalized_strategy = "settlement_2r"
            else:
                return 9999.0

        (
            ev_settlement_0r,
            ev_settlement_1r,
            ev_settlement_2r,
            ev_city,
            ev_dev,
            ev_dev4,
        ) = self._get_base_expected_vectors(vertices)

        state_idx = min(self._hand_to_state_index(hand), len(ev_settlement_0r) - 1)

        num_players = self._get_num_players()

        if normalized_strategy == "settlement_0r":
            raw_base_score = float(ev_settlement_0r[state_idx])
        elif normalized_strategy == "settlement_1r":
            raw_base_score = float(ev_settlement_1r[state_idx])
        elif normalized_strategy == "settlement_2r":
            raw_base_score = float(ev_settlement_2r[state_idx])
        elif normalized_strategy == "city":
            raw_base_score = float(ev_city[state_idx])
        elif normalized_strategy == "dev_card":
            raw_base_score = float(ev_dev[state_idx])
        elif normalized_strategy == "dev_card_4":
            raw_base_score = float(ev_dev4[state_idx])
        else:
            raw_base_score = min(
                float(ev_settlement_0r[state_idx]),
                float(ev_settlement_1r[state_idx]),
                float(ev_settlement_2r[state_idx]),
                float(ev_city[state_idx]),
                float(ev_dev[state_idx]),
                float(ev_dev4[state_idx]),
            )

        base_score = raw_base_score / float(max(num_players, 1))

        # Safety fallback if the raw EV is clearly broken
        if base_score < 1.0 or base_score > 200.0:
            inter = None
            last_vid = int(vertices[-1])
            if self.board is not None and 0 <= last_vid < len(self.board.intersections):
                inter = self.board.intersections[last_vid]

            if inter is not None:
                total_pips = sum(
                    getattr(inter, "all_tile_pips", getattr(inter, "three_tile_pips", [0.0] * 5))
                )
                base_score = max(3.0, 36.0 / max(float(total_pips), 0.1))

        bonus = self._simple_port_bonus(vertices, player_ports)
        base_after_bonus = max(0.0, float(base_score - bonus))
        final_score = float(base_after_bonus)

        overflow_info = self._overflow_port_monopoly_score(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_strategy,
            extra_roads_needed=extra_roads_needed,
        )

        used_overflow = False
        overflow_triggered = bool(overflow_info.get("trigger", False))
        overflow_score = 9999.0

        if overflow_triggered:
            try:
                overflow_score = float(overflow_info.get("score", 9999.0))
            except Exception:
                overflow_score = 9999.0

            if 0.0 <= overflow_score < 9999.0:
                guarded_score = max(base_after_bonus, overflow_score)
                used_overflow = guarded_score > base_after_bonus + 1e-9
                final_score = float(guarded_score)

        # ------------------------------------------------------------
        # Expected-hand feasibility sanity clamp
        # ------------------------------------------------------------
        feasibility_at_score = self._expected_hand_trade_feasibility_check(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_strategy,
            extra_roads_needed=extra_roads_needed,
            candidate_rounds=final_score,
        )

        first_feasible = self._first_feasible_round_by_expected_trade_math(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_strategy,
            extra_roads_needed=extra_roads_needed,
            start_round=final_score,
            horizon_rounds=9,
            step=1.0,
        )

        feasibility_used = False
        feasibility_score = float(final_score)

        if not bool(feasibility_at_score.get("viable_after_trades", False)):
            if bool(first_feasible.get("found", False)):
                feasibility_score = float(first_feasible.get("round", final_score))
                if feasibility_score > final_score + 1e-9:
                    final_score = feasibility_score
                    feasibility_used = True
            else:
                final_score = 9999.0
                feasibility_used = True

        port_str = str(player_ports)[:60] if player_ports else "4:1 bank"

        def _fmt_vec(values, digits=2):
            out = []
            for v in values:
                try:
                    fv = float(v)
                    if abs(fv - round(fv)) < 1e-9:
                        out.append(str(int(round(fv))))
                    else:
                        out.append(f"{fv:.{digits}f}")
                except Exception:
                    out.append(str(v))
            return "[" + ", ".join(out) + "]"

        feas_info = feasibility_at_score
        first_info = first_feasible.get("info", {}) or {}

        feasibility_text = (
            f"| feasible@score={feas_info.get('conclusion')} "
            f"trades={float(feas_info.get('trades_available', 0.0)):.2f}/"
            f"{float(feas_info.get('trades_needed', 0.0)):.2f} "
            f"| first_feasible={first_feasible.get('round')} "
            f"| feasible_used={feasibility_used} "
        )

        detail_text = (
            f"\n      feasibility detail | "
            f"rounds={float(feas_info.get('candidate_rounds', 0.0)):.2f} "
            f"hand={_fmt_vec(feas_info.get('current_hand', []))} "
            f"pips={_fmt_vec(feas_info.get('pips', []))} "
            f"add={_fmt_vec(feas_info.get('additional', []))} "
            f"exp={_fmt_vec(feas_info.get('expected_hand', []))} "
            f"need={_fmt_vec(feas_info.get('needed', []))} "
            f"short={_fmt_vec(feas_info.get('short', []))} "
            f"surplus={_fmt_vec(feas_info.get('surplus', []))} "
            f"trade_avail_detail={_fmt_vec(feas_info.get('trades_available_detail', []))}"
        )

        if first_info:
            detail_text += (
                f"\n      first feasible detail | "
                f"rounds={float(first_info.get('candidate_rounds', 0.0)):.2f} "
                f"add={_fmt_vec(first_info.get('additional', []))} "
                f"exp={_fmt_vec(first_info.get('expected_hand', []))} "
                f"short={_fmt_vec(first_info.get('short', []))} "
                f"surplus={_fmt_vec(first_info.get('surplus', []))} "
                f"trade_avail_detail={_fmt_vec(first_info.get('trades_available_detail', []))}"
            )

        if overflow_triggered:
            dom = overflow_info.get("dominant_resource")
            dom_pips = float(overflow_info.get("dominant_resource_pips", 0.0))
            trade_rate = int(overflow_info.get("effective_trade_rate", 4))
            eq_need = float(overflow_info.get("dominant_equivalent_needed", 0.0))
            eq_raw = float(overflow_info.get("dominant_equivalent_raw", eq_need))
            hand_surplus = float(overflow_info.get("dominant_surplus_in_hand", 0.0))

            needed_trades = int(overflow_info.get("needed_trades", 0))
            off_buy = int(overflow_info.get("off_resource_cards_to_buy", 0))
            expected_dom = float(overflow_info.get("expected_dominant_cards_by_horizon", 0.0))
            horizon_turns = int(overflow_info.get("horizon_turns", 9))
            can_fund = bool(overflow_info.get("can_fund_within_horizon", False))
            cap_bind = bool(overflow_info.get("cap_bind_risk", False))
            weak_off = bool(overflow_info.get("weak_off_resource_exists", False))

            print(
                f"   Markov FAST init {self._position_key(vertices)} | port={port_str} "
                # f"| strategy={normalized_strategy} | base={base_score:.2f} | bonus={bonus:.2f} "
                f"| strategy={normalized_strategy} | raw_rolls={raw_base_score:.2f} | base={base_score:.2f} | bonus={bonus:.2f} "
                f"| overflow={dom}@{dom_pips:.1f}pips rate={trade_rate}:1 "
                f"raw_eq={eq_raw:.1f} hand_surplus={hand_surplus:.1f} eq={eq_need:.1f} "
                f"| trades={needed_trades} off_buy={off_buy} "
                f"| exp_dom@{horizon_turns}t={expected_dom:.1f} fund={can_fund} "
                f"| cap_bind={cap_bind} weak_off={weak_off} "
                f"| overflow_score={overflow_score:.2f} | used_overflow={used_overflow} "
                f"{feasibility_text}"
                f"| final={final_score:.2f}"
                f"{detail_text}"
            )
        else:
            print(
                f"   Markov FAST init {self._position_key(vertices)} | port={port_str} "
                f"| strategy={normalized_strategy} | base={base_score:.2f} | bonus={bonus:.2f} "
                f"{feasibility_text}"
                f"| final={final_score:.2f}"
                f"{detail_text}"
            )

        return final_score

    def get_expected_time_to_event_fast(self, vertices: list[int], hand: list[int], player_ports: dict) -> dict:
        """
        Fast approximate event timing.

        Uses the fast scorer path for all supported targets:
            - settlement / settlement_0r
            - settlement_1r
            - settlement_2r
            - city
            - dev_card
            - dev_card_4

        Notes:
        - does NOT call apply_trading_layer()
        - uses cached raw production EVs + lightweight port bonus
        - may apply the dominant-resource overflow fallback through
        get_expected_turns_fast_initial(...)
        """
        if not vertices:
            return {
                "settlement": 9999.0,
                "settlement_0r": 9999.0,
                "settlement_1r": 9999.0,
                "settlement_2r": 9999.0,
                "city": 9999.0,
                "dev_card": 9999.0,
                "dev_card_4": 9999.0,
            }

        if self._ignore_resource_cards():
            hand = [0, 0, 0, 0, 0]
        else:
            hand = hand or [0, 0, 0, 0, 0]

        player_ports = player_ports or {}

        settlement_0r_time = float(
            self.get_expected_turns_fast_initial(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                strategy="settlement_0r",
                extra_roads_needed=0,
            )
        )

        settlement_1r_time = float(
            self.get_expected_turns_fast_initial(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                strategy="settlement_1r",
                extra_roads_needed=1,
            )
        )

        settlement_2r_time = float(
            self.get_expected_turns_fast_initial(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                strategy="settlement_2r",
                extra_roads_needed=2,
            )
        )

        city_time = float(
            self.get_expected_turns_fast_initial(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                strategy="city",
                extra_roads_needed=0,
            )
        )

        dev_time = float(
            self.get_expected_turns_fast_initial(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                strategy="dev_card",
                extra_roads_needed=0,
            )
        )

        dev4_time = float(
            self.get_expected_turns_fast_initial(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                strategy="dev_card_4",
                extra_roads_needed=0,
            )
        )

        return {
            "settlement": settlement_0r_time,
            "settlement_0r": settlement_0r_time,
            "settlement_1r": settlement_1r_time,
            "settlement_2r": settlement_2r_time,
            "city": city_time,
            "dev_card": dev_time,
            "dev_card_4": dev4_time,
        }

    def _ignore_resource_cards(self) -> bool:
        """
        Return True when fast-forward should ignore actual owned resource cards.

        Preferred source:
            self.game.ff_ignore_resource_cards

        Fallback:
            self.ignore_resource_cards
        """
        game = getattr(self, "game", None)
        if game is not None and hasattr(game, "ff_ignore_resource_cards"):
            return bool(game.ff_ignore_resource_cards)

        return bool(getattr(self, "ignore_resource_cards", False))
    
    def _estimate_trade_plan(
        self,
        vertices: list[int],
        hand: list[int],
        player_ports: dict,
        target_type: str,
        extra_roads_needed: int = 0,
    ) -> dict:
        """
        Build a transparent, approximate trade explanation for diagnostics.

        IMPORTANT:
        - This is a heuristic explanation layer for validation / transparency.
        - It does NOT claim to be the exact hidden Markov trade path.
        - For settlement, extra_roads_needed is included in the displayed cost.
        - Dominance is based on production, not on port ownership.
        - Reported trade strings are now split between:
        * immediate trades that are actually possible from current surplus
        * future conversion plans based on production strength
        """

        hand = hand or [0, 0, 0, 0, 0]
        player_ports = player_ports or {}

        normalized_target_type = str(target_type or "").strip().lower()

        if normalized_target_type == "new_settlement":
            normalized_target_type = "settlement"
        elif normalized_target_type == "upgrade_to_city":
            normalized_target_type = "city"
        elif normalized_target_type == "buy_discovery_card":
            normalized_target_type = "dev_card"
        elif normalized_target_type == "buy_4_discovery_cards":
            normalized_target_type = "dev_card_4"

        res_names = ["wheat", "ore", "wood", "brick", "wool"]

        target_cost = self._game_order_target_cost(
            normalized_target_type, extra_roads_needed=extra_roads_needed
        )

        current_hand = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        keep_for_target = {
            res: int(target_cost.get(res, 0)) for res in res_names
        }

        tradable_surplus = {
            res: max(0, int(current_hand.get(res, 0)) - int(keep_for_target.get(res, 0)))
            for res in res_names
        }

        deficits = {}
        for res, need in target_cost.items():
            have = current_hand.get(res, 0)
            if have < need:
                deficits[res] = int(need - have)

        dominant_info = self._get_dominant_resource_from_vertices(vertices)
        dominant_resource = dominant_info.get("resource")
        dominant_resource_pips = float(dominant_info.get("pips", 0.0))
        pips = dominant_info.get("all_pips", {}) or {}

        trade_rates = {
            r: self._get_effective_trade_rate_for_resource(r, player_ports)
            for r in res_names
        }

        available_surplus = dict(tradable_surplus)

        trades = []
        immediate_trades = []
        future_trade_plans = []

        def _rank_future_sources(deficit_res: str):
            ranked = []
            for src in res_names:
                if src == deficit_res:
                    continue
                surplus_cards = float(available_surplus.get(src, 0))
                pip_score = float(pips.get(src, 0.0))
                demand_penalty = float(target_cost.get(src, 0)) * 1.25
                dominant_bonus = 2.0 if src == dominant_resource else 0.0
                cheaper_trade_bonus = (4.0 - float(trade_rates.get(src, 4))) * 1.5

                score = (
                    (3.0 * surplus_cards)
                    + pip_score
                    + dominant_bonus
                    + cheaper_trade_bonus
                    - demand_penalty
                )

                if surplus_cards > 0.0 or pip_score > 0.0:
                    ranked.append((score, surplus_cards, pip_score, src))

            ranked.sort(reverse=True)
            return ranked

        # Process immediate trades and future plans for each deficit
        for deficit_res, deficit_amt in deficits.items():
            remaining = int(deficit_amt)

            # --- Immediate trades ---
            immediate_candidates = []
            for src in res_names:
                if src == deficit_res:
                    continue
                rate = int(trade_rates.get(src, 4))
                surplus_cards = int(available_surplus.get(src, 0))
                possible_now = surplus_cards // max(rate, 1)

                if possible_now > 0:
                    pip_score = float(pips.get(src, 0.0))
                    dominant_bonus = 2.0 if src == dominant_resource else 0.0
                    immediate_score = (4.0 * possible_now) + pip_score + dominant_bonus
                    immediate_candidates.append(
                        (immediate_score, src, rate, possible_now)
                    )

            immediate_candidates.sort(reverse=True)

            for _, src, rate, possible_now in immediate_candidates:
                if remaining <= 0:
                    break
                used_now = min(remaining, possible_now)
                if used_now <= 0:
                    continue

                available_surplus[src] -= used_now * rate
                remaining -= used_now

                msg = f"{used_now}x immediate {rate} {src} -> 1 {deficit_res}"
                immediate_trades.append(msg)
                trades.append(msg)

            if remaining <= 0:
                continue

            # --- Future trade plans ---
            ranked_sources = _rank_future_sources(deficit_res)

            if not ranked_sources:
                msg = f"no clear future source found for {remaining} {deficit_res}"
                future_trade_plans.append(msg)
                trades.append(msg)
                continue

            best_sources = [src for _, _, _, src in ranked_sources[:2]]
            counts = {src: 0 for src in best_sources}

            idx = 0
            for _ in range(remaining):
                src = best_sources[idx % len(best_sources)]
                counts[src] += 1
                idx += 1

            for src in best_sources:
                n = counts[src]
                if n <= 0:
                    continue

                rate = int(trade_rates.get(src, 4))
                pip_score = float(pips.get(src, 0.0))
                src_is_dominant = src == dominant_resource

                if src_is_dominant:
                    msg = f"{n}x future dominant {src} surplus via {rate}:1 -> {deficit_res}"
                elif pip_score >= float(pips.get(deficit_res, 0.0)) + 1.0:
                    msg = f"{n}x future {src} surplus via {rate}:1 -> {deficit_res}"
                else:
                    msg = f"{n}x mixed direct {deficit_res} + occasional {src} trades"

                future_trade_plans.append(msg)
                trades.append(msg)

        # Get overflow / monopoly information
        overflow_info = self._overflow_port_monopoly_score(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_target_type,
            extra_roads_needed=extra_roads_needed,
        )

        target_needs_trade_conversion = self._target_needs_trade_conversion(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_target_type,
            extra_roads_needed=extra_roads_needed,
        )

        return {
            "normalized_target_type": normalized_target_type,
            "target_cost": target_cost,
            "current_hand": current_hand,
            "keep_for_target": keep_for_target,
            "tradable_surplus": tradable_surplus,
            "deficits": deficits,
            "production_pips": pips,
            "dominant_resource": dominant_resource,
            "dominant_resource_pips": dominant_resource_pips,
            "trade_rates": trade_rates,
            "extra_roads_needed": int(extra_roads_needed),
            "target_needs_trade_conversion": bool(target_needs_trade_conversion),
            "trades": trades,
            "immediate_trades": immediate_trades,
            "future_trade_plans": future_trade_plans,
            "overflow_triggered": bool(overflow_info.get("trigger", False)),
            "overflow_dominant_resource": overflow_info.get("dominant_resource"),
            "overflow_dominant_resource_pips": float(
                overflow_info.get("dominant_resource_pips", 0.0)
            ),
            "overflow_effective_trade_rate": int(
                overflow_info.get("effective_trade_rate", 4)
            ),
            "overflow_equivalent_needed": float(
                overflow_info.get("dominant_equivalent_needed", 0.0)
            ),
            "overflow_equivalent_raw": float(
                overflow_info.get("dominant_equivalent_raw", 0.0)
            ),
            "overflow_hand_surplus": float(
                overflow_info.get("dominant_surplus_in_hand", 0.0)
            ),
            "overflow_equivalent_turns": float(
                overflow_info.get("equivalent_turns", 9999.0)
            ),
            "overflow_blended_turns": float(
                overflow_info.get("blended_turns", 9999.0)
            ),
            "overflow_score": float(overflow_info.get("score", 9999.0)),
        }
    
    def _game_order_pips_from_vertices(self, vertices) -> dict:
        """
        Return production pips in GAME order names:
            wheat, ore, wood, brick, wool

        Duplicates in vertices are preserved, so [49, 49, 51] naturally
        models doubled production on vertex 49.
        """
        pips = {
            "wheat": 0.0,
            "ore": 0.0,
            "wood": 0.0,
            "brick": 0.0,
            "wool": 0.0,
        }

        if self.board is None:
            return pips

        for vid in vertices:
            try:
                inter_id = int(vid)
            except Exception:
                continue

            if inter_id < 0 or inter_id >= len(self.board.intersections):
                continue

            inter = self.board.intersections[inter_id]
            if inter is None:
                continue

            all_pips = getattr(inter, "all_tile_pips", None)
            if not all_pips:
                all_pips = getattr(inter, "three_tile_pips", [0.0] * 5)

            try:
                pips["wheat"] += float(all_pips[0])
                pips["ore"] += float(all_pips[1])
                pips["wood"] += float(all_pips[2])
                pips["brick"] += float(all_pips[3])
                pips["wool"] += float(all_pips[4])
            except Exception:
                pass

        return pips
    
    def _game_order_target_cost(self, target_type: str, extra_roads_needed: int = 0) -> dict:
        """
        Return target cost in GAME-order resource names.

        Game order:
            [Wheat, Ore, Wood, Brick, Wool]

        Normalized supported targets:
            - settlement / settlement_0r / new_settlement
            - settlement_1r
            - settlement_2r
            - city / upgrade_to_city
            - dev_card / buy_discovery_card
            - dev_card_4 / buy_4_discovery_cards

        Important:
        - For plain settlement aliases, extra_roads_needed is honored.
        """
        tt = str(target_type or "").strip().lower()

        # ------------------------------------------------------------
        # Normalize external / fast-forward aliases
        # ------------------------------------------------------------
        if tt == "new_settlement":
            tt = "settlement"
        elif tt == "upgrade_to_city":
            tt = "city"
        elif tt == "buy_discovery_card":
            tt = "dev_card"
        elif tt == "buy_4_discovery_cards":
            tt = "dev_card_4"

        # ------------------------------------------------------------
        # Settlement family
        # ------------------------------------------------------------
        if tt in ("settlement", "settlement_0r"):
            roads = max(0, int(extra_roads_needed))

            if roads <= 0:
                return {"wheat": 1, "wood": 1, "brick": 1, "wool": 1}
            if roads == 1:
                return {"wheat": 1, "wood": 2, "brick": 2, "wool": 1}
            if roads == 2:
                return {"wheat": 1, "wood": 3, "brick": 3, "wool": 1}

            # Unsupported beyond 2 roads in current fast-forward model
            return {}

        if tt == "settlement_1r":
            return {"wheat": 1, "wood": 2, "brick": 2, "wool": 1}

        if tt == "settlement_2r":
            return {"wheat": 1, "wood": 3, "brick": 3, "wool": 1}

        # ------------------------------------------------------------
        # City / dev-card families
        # ------------------------------------------------------------
        if tt == "city":
            return {"wheat": 2, "ore": 3}

        if tt == "dev_card":
            return {"wheat": 1, "ore": 1, "wool": 1}

        if tt == "dev_card_4":
            return {"wheat": 4, "ore": 4, "wood": 4, "brick": 4, "wool": 4}

        return {}

    def _expected_hand_trade_feasibility_check(
        self,
        vertices,
        hand,
        player_ports,
        target_type: str,
        extra_roads_needed: int = 0,
        candidate_rounds: float = 0.0,
    ) -> dict:
        """
        Sanity-check Markov FAST timing using the standalone v015 expected-hand
        estimator.

        This method intentionally keeps the old return keys because
        get_expected_turns_fast_initial(...) logs/reads them downstream.

        Game order:
            [wheat, ore, wood, brick, wool]
        """
        if (
            target_cost_vector is None
            or estimate_expected_hand_after_turns is None
            or compute_payability_with_trades is None
            or estimate_payability_confidence is None
        ):
            raise RuntimeError(
                "core.resource_time_estimator is required for "
                "_expected_hand_trade_feasibility_check in Catan v015"
            )

        resources = ["wheat", "ore", "wood", "brick", "wool"]
        hand = hand or [0, 0, 0, 0, 0]
        player_ports = player_ports or {}

        target = str(target_type or "").strip().lower()

        if target == "new_settlement":
            target = "settlement"
        elif target == "upgrade_to_city":
            target = "city"
        elif target == "buy_discovery_card":
            target = "dev_card"
        elif target == "buy_4_discovery_cards":
            target = "dev_card_4"

        if target == "settlement":
            if int(extra_roads_needed) <= 0:
                target = "settlement_0r"
            elif int(extra_roads_needed) == 1:
                target = "settlement_1r"
            elif int(extra_roads_needed) == 2:
                target = "settlement_2r"

        try:
            needed = [
                float(x)
                for x in target_cost_vector(
                    target,
                    extra_roads_needed=int(extra_roads_needed),
                )
            ]
        except Exception:
            # Compatibility fallback: keep the old local target-cost helper usable.
            target_cost = self._game_order_target_cost(
                target,
                extra_roads_needed=extra_roads_needed,
            )
            needed = [float(target_cost.get(res, 0.0)) for res in resources]

        current_hand = [0.0] * 5
        for i in range(min(5, len(hand))):
            try:
                current_hand[i] = float(hand[i])
            except Exception:
                current_hand[i] = 0.0

        pips_dict = self._game_order_pips_from_vertices(vertices)
        pips = [
            float(pips_dict.get("wheat", 0.0)),
            float(pips_dict.get("ore", 0.0)),
            float(pips_dict.get("wood", 0.0)),
            float(pips_dict.get("brick", 0.0)),
            float(pips_dict.get("wool", 0.0)),
        ]

        rounds = max(0.0, float(candidate_rounds))
        num_players = self._get_num_players()

        expected_hand = estimate_expected_hand_after_turns(
            current_hand=current_hand,
            production_pips=pips,
            turns=rounds,
            rolls_per_player_turn=num_players,
        )
        additional = [
            float(expected_hand[i]) - float(current_hand[i])
            for i in range(5)
        ]

        trade_rates = [
            int(self._get_effective_trade_rate_for_resource(resources[i], player_ports))
            for i in range(5)
        ]

        payability = compute_payability_with_trades(
            expected_hand=expected_hand,
            need=needed,
            trade_rates=trade_rates,
            continuous=bool(EXPECTED_HAND_CONTINUOUS_TRADING),
        )

        confidence_info = estimate_payability_confidence(
            current_hand=current_hand,
            need=needed,
            production_pips=pips,
            turns=rounds,
            num_players=num_players,
            payability=payability,
            confidence_target=float(EXPECTED_HAND_CONFIDENCE_TARGET),
        )

        short = [float(x) for x in payability.get("short", [0.0] * 5)]
        surplus = [float(x) for x in payability.get("surplus", [0.0] * 5)]

        trades_needed_detail = list(short)
        trades_needed = float(payability.get("trades_needed", sum(trades_needed_detail)))

        # Preserve the old diagnostic field, even though the new estimator also
        # returns exports_used/imports_received in the nested payability dict.
        trades_available_detail = [
            surplus[i] / max(float(trade_rates[i]), 1.0)
            for i in range(5)
        ]
        trades_available = float(
            payability.get("trades_available", sum(trades_available_detail))
        )

        viable_direct = bool(payability.get("payable_direct", False))
        viable_after_trades = bool(payability.get("payable_after_trades", False))

        confidence = float(confidence_info.get("confidence", 0.0))
        confidence_label = str(confidence_info.get("label", "very_low"))

        return {
            "strategy": target,
            "candidate_rounds": float(rounds),

            "resources": list(resources),
            "current_hand": current_hand,
            "pips": pips,
            "additional": additional,
            "expected_hand": [float(x) for x in expected_hand],
            "needed": needed,

            "short": short,
            "surplus": surplus,

            "trade_rates": trade_rates,
            "trades_needed": trades_needed,
            "trades_needed_detail": trades_needed_detail,
            "trades_available": trades_available,
            "trades_available_detail": trades_available_detail,

            "viable_direct": bool(viable_direct),
            "viable_after_trades": bool(viable_after_trades),
            "conclusion": "viable" if viable_after_trades else "not viable",

            # New v015 expected-hand confidence fields.
            "confidence": confidence,
            "confidence_target": float(EXPECTED_HAND_CONFIDENCE_TARGET),
            "confidence_label": confidence_label,
            "confidence_info": dict(confidence_info),
            "payability": dict(payability),
            "estimator": "expected_hand_v015",
            "estimator_primary": "markov_light_comparison",
        }

    def _first_feasible_round_by_expected_trade_math(
        self,
        vertices,
        hand,
        player_ports,
        target_type: str,
        extra_roads_needed: int = 0,
        start_round: float = 0.0,
        horizon_rounds: int = 9,
        step: float = 1.0,
    ) -> dict:
        """
        Find the first round where expected-hand + expected-trades is feasible.

        Delegates the turns-to-afford search to core.resource_time_estimator while
        keeping this method's original return shape:
            {"found": bool, "round": float, "info": <feasibility dict>}

        Important compatibility choice:
        - require_confidence=False here, so this still mimics the old Markov
          validator's expected-value crossing.
        - confidence fields are still returned in info for Step-1 comparison.
        """
        if estimate_first_payable_turn is None or target_cost_vector is None:
            raise RuntimeError(
                "core.resource_time_estimator is required for "
                "_first_feasible_round_by_expected_trade_math in Catan v015"
            )

        resources = ["wheat", "ore", "wood", "brick", "wool"]
        hand = hand or [0, 0, 0, 0, 0]
        player_ports = player_ports or {}

        try:
            start = int(max(0, math.ceil(float(start_round))))
        except Exception:
            start = 0

        try:
            horizon = int(max(start, math.ceil(float(horizon_rounds))))
        except Exception:
            horizon = max(start, 9)

        try:
            step_value = max(float(step), 0.25)
        except Exception:
            step_value = 1.0

        target = str(target_type or "").strip().lower()
        if target == "new_settlement":
            target = "settlement"
        elif target == "upgrade_to_city":
            target = "city"
        elif target == "buy_discovery_card":
            target = "dev_card"
        elif target == "buy_4_discovery_cards":
            target = "dev_card_4"

        if target == "settlement":
            if int(extra_roads_needed) <= 0:
                target = "settlement_0r"
            elif int(extra_roads_needed) == 1:
                target = "settlement_1r"
            elif int(extra_roads_needed) == 2:
                target = "settlement_2r"

        current_hand = [0.0] * 5
        for i in range(min(5, len(hand))):
            try:
                current_hand[i] = float(hand[i])
            except Exception:
                current_hand[i] = 0.0

        pips_dict = self._game_order_pips_from_vertices(vertices)
        pips = [
            float(pips_dict.get("wheat", 0.0)),
            float(pips_dict.get("ore", 0.0)),
            float(pips_dict.get("wood", 0.0)),
            float(pips_dict.get("brick", 0.0)),
            float(pips_dict.get("wool", 0.0)),
        ]

        try:
            needed = [
                float(x)
                for x in target_cost_vector(
                    target,
                    extra_roads_needed=int(extra_roads_needed),
                )
            ]
        except Exception:
            target_cost = self._game_order_target_cost(
                target,
                extra_roads_needed=extra_roads_needed,
            )
            needed = [float(target_cost.get(res, 0.0)) for res in resources]

        trade_rates = [
            int(self._get_effective_trade_rate_for_resource(resources[i], player_ports))
            for i in range(5)
        ]

        estimator_result = estimate_first_payable_turn(
            current_hand=current_hand,
            production_pips=pips,
            need=needed,
            trade_rates=trade_rates,
            confidence_target=float(EXPECTED_HAND_CONFIDENCE_TARGET),
            num_players=self._get_num_players(),
            step=step_value,
            max_turns=float(horizon),
            continuous_trading=bool(EXPECTED_HAND_CONTINUOUS_TRADING),
            require_confidence=False,
        )

        if bool(estimator_result.get("found", False)):
            found_round = float(estimator_result.get("turns", 9999.0))

            # Keep the old "first feasible at or after ceil(start_round)" behavior.
            if found_round < float(start):
                found_round = float(start)

            info = self._expected_hand_trade_feasibility_check(
                vertices=vertices,
                hand=hand,
                player_ports=player_ports,
                target_type=target,
                extra_roads_needed=extra_roads_needed,
                candidate_rounds=found_round,
            )
            info["first_payable_estimator_result"] = dict(estimator_result)

            return {
                "found": True,
                "round": float(found_round),
                "info": info,
                "estimator": "expected_hand_v015",
            }

        best_info = self._expected_hand_trade_feasibility_check(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=target,
            extra_roads_needed=extra_roads_needed,
            candidate_rounds=float(horizon),
        )
        best_info["first_payable_estimator_result"] = dict(estimator_result)

        return {
            "found": False,
            "round": 9999.0,
            "info": best_info or {},
            "estimator": "expected_hand_v015",
        }

    def _target_needs_trade_conversion(
        self,
        vertices,
        hand,
        player_ports,
        target_type: str,
        extra_roads_needed: int = 0,
    ) -> bool:
        """
        Heuristic detector for whether the target materially depends on converting
        a dominant produced resource into other missing resources.

        Refined intent:
        - detect classic overflow situations where one resource engine is much stronger
        than the direct production of one or more missing off-resources
        - be broad enough to catch city cases like strong wheat -> weak ore conversion
        - give some credit for dominant surplus already in hand, but still trigger when
        the remaining conversion burden is clearly meaningful
        """
        normalized_target_type = str(target_type or "").strip().lower()
        if normalized_target_type == "new_settlement":
            normalized_target_type = "settlement"
        elif normalized_target_type == "upgrade_to_city":
            normalized_target_type = "city"
        elif normalized_target_type == "buy_discovery_card":
            normalized_target_type = "dev_card"
        elif normalized_target_type == "buy_4_discovery_cards":
            normalized_target_type = "dev_card_4"

        dominant_info = self._get_dominant_resource_from_vertices(vertices)
        dominant_resource = dominant_info.get("resource")
        dominant_pips = float(dominant_info.get("pips", 0.0))
        pips = dominant_info.get("all_pips", {}) or {}

        if dominant_resource is None:
            return False

        if dominant_pips < 5.0:
            return False

        target_cost = self._game_order_target_cost(
            normalized_target_type,
            extra_roads_needed=extra_roads_needed,
        )

        current_hand = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        deficits = {}
        for res, need in target_cost.items():
            have = current_hand.get(res, 0)
            if have < need:
                deficits[res] = int(need - have)

        if not deficits:
            return False

        trade_rate = self._get_effective_trade_rate_for_resource(
            dominant_resource,
            player_ports,
        )

        dominant_keep_need = float(target_cost.get(dominant_resource, 0))
        dominant_in_hand = float(current_hand.get(dominant_resource, 0))
        dominant_surplus_in_hand = max(0.0, dominant_in_hand - dominant_keep_need)

        off_resource_trade_units_raw = 0.0
        off_resource_trade_units_adjusted = 0.0
        weak_off_resource_exists = False

        for res, deficit_amt in deficits.items():
            if deficit_amt <= 0:
                continue

            if res == dominant_resource:
                continue

            direct_pips = float(pips.get(res, 0.0))
            converted_amt = float(deficit_amt * trade_rate)
            off_resource_trade_units_raw += converted_amt

            traded_dominant_pips = dominant_pips / float(trade_rate)

            if direct_pips < 2.0:
                weak_off_resource_exists = True

            if direct_pips <= traded_dominant_pips * 1.10:
                weak_off_resource_exists = True

            if normalized_target_type == "city" and res == "ore" and dominant_resource != "ore":
                if deficit_amt >= 2 and direct_pips < max(4.0, 0.80 * dominant_pips):
                    weak_off_resource_exists = True

        off_resource_trade_units_adjusted = max(
            0.0,
            off_resource_trade_units_raw - dominant_surplus_in_hand,
        )

        if not weak_off_resource_exists:
            return False

        if off_resource_trade_units_raw <= 0.0:
            return False

        if off_resource_trade_units_adjusted > 4.0:
            return True

        if normalized_target_type == "city" and off_resource_trade_units_adjusted >= float(trade_rate):
            return True

        if off_resource_trade_units_raw > 4.0 and dominant_pips >= 8.0:
            return True

        return False

    def _can_dominant_resource_fund_trades_within_horizon(
        self,
        vertices,
        hand,
        player_ports,
        target_type: str,
        extra_roads_needed: int = 0,
        horizon_turns: int = 9,
        num_players: Optional[int] = None,
    ) -> dict:
        """
        Conservative overflow gate for the FAST scorer.

        Core idea:
        overflow should only be considered when the dominant produced resource can
        realistically fund the required trade burden for this specific target within
        a bounded horizon.

        Horizon interpretation:
        - horizon_turns = player-turns / rounds
        - each round has one roll event per player
        - expected future cards from pips:
            expected_cards = pips * (horizon_turns * num_players) / 36

        Returns a diagnostic dict with:
            - trigger_candidate
            - dominant_resource
            - dominant_resource_pips
            - effective_trade_rate
            - needed_trades
            - off_resource_cards_to_buy
            - dominant_direct_deficit
            - dominant_trade_cards_needed_raw
            - dominant_trade_cards_needed_adjusted
            - dominant_surplus_in_hand
            - expected_future_dominant_cards
            - expected_dominant_cards_by_horizon
            - horizon_turns
            - horizon_rolls
            - can_fund_within_horizon
            - cap_bind_risk
            - weak_off_resource_exists
            - normalized_target_type
        """
        hand = hand or [0, 0, 0, 0, 0]
        player_ports = player_ports or {}

        normalized_target_type = str(target_type or "").strip().lower()
        if normalized_target_type == "new_settlement":
            normalized_target_type = "settlement"
        elif normalized_target_type == "upgrade_to_city":
            normalized_target_type = "city"
        elif normalized_target_type == "buy_discovery_card":
            normalized_target_type = "dev_card"
        elif normalized_target_type == "buy_4_discovery_cards":
            normalized_target_type = "dev_card_4"

        dominant_info = self._get_dominant_resource_from_vertices(vertices)
        dominant_resource = dominant_info.get("resource")
        dominant_pips = float(dominant_info.get("pips", 0.0))
        pips = dominant_info.get("all_pips", {}) or {}

        if dominant_resource is None:
            return {
                "trigger_candidate": False,
                "dominant_resource": None,
                "dominant_resource_pips": 0.0,
                "effective_trade_rate": 4,
                "needed_trades": 0,
                "off_resource_cards_to_buy": 0,
                "dominant_direct_deficit": 0.0,
                "dominant_trade_cards_needed_raw": 0.0,
                "dominant_trade_cards_needed_adjusted": 0.0,
                "dominant_surplus_in_hand": 0.0,
                "expected_future_dominant_cards": 0.0,
                "expected_dominant_cards_by_horizon": 0.0,
                "horizon_turns": int(max(1, horizon_turns)),
                "horizon_rolls": float(max(1, horizon_turns) * max(1, num_players)),
                "can_fund_within_horizon": False,
                "cap_bind_risk": False,
                "weak_off_resource_exists": False,
                "normalized_target_type": normalized_target_type,
            }

        effective_trade_rate = self._get_effective_trade_rate_for_resource(
            dominant_resource,
            player_ports,
        )

        target_cost = self._game_order_target_cost(
            normalized_target_type,
            extra_roads_needed=extra_roads_needed,
        )

        current_hand = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        deficits = {}
        for res, need in target_cost.items():
            have = current_hand.get(res, 0)
            if have < need:
                deficits[res] = int(need - have)

        if not deficits:
            return {
                "trigger_candidate": False,
                "dominant_resource": dominant_resource,
                "dominant_resource_pips": dominant_pips,
                "effective_trade_rate": effective_trade_rate,
                "needed_trades": 0,
                "off_resource_cards_to_buy": 0,
                "dominant_direct_deficit": 0.0,
                "dominant_trade_cards_needed_raw": 0.0,
                "dominant_trade_cards_needed_adjusted": 0.0,
                "dominant_surplus_in_hand": 0.0,
                "expected_future_dominant_cards": 0.0,
                "expected_dominant_cards_by_horizon": float(current_hand.get(dominant_resource, 0)),
                "horizon_turns": int(max(1, horizon_turns)),
                "horizon_rolls": float(max(1, horizon_turns) * max(1, num_players)),
                "can_fund_within_horizon": False,
                "cap_bind_risk": False,
                "weak_off_resource_exists": False,
                "normalized_target_type": normalized_target_type,
            }

        # Keep the existing conceptual gate:
        # only bother if the target really appears to depend on trade conversion.
        needs_trade_conversion = self._target_needs_trade_conversion(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=normalized_target_type,
            extra_roads_needed=extra_roads_needed,
        )

        dominant_keep_need = float(target_cost.get(dominant_resource, 0))
        dominant_in_hand = float(current_hand.get(dominant_resource, 0))
        dominant_surplus_in_hand = max(0.0, dominant_in_hand - dominant_keep_need)

        off_resource_cards_to_buy = 0
        weak_off_resource_exists = False

        for res, deficit_amt in deficits.items():
            if deficit_amt <= 0 or res == dominant_resource:
                continue

            off_resource_cards_to_buy += int(deficit_amt)

            direct_pips = float(pips.get(res, 0.0))
            traded_dominant_pips = dominant_pips / float(max(effective_trade_rate, 1))

            if direct_pips < 2.0:
                weak_off_resource_exists = True

            if direct_pips <= traded_dominant_pips * 1.10:
                weak_off_resource_exists = True

            if normalized_target_type == "city" and res == "ore" and dominant_resource != "ore":
                if deficit_amt >= 2 and direct_pips < max(4.0, 0.80 * dominant_pips):
                    weak_off_resource_exists = True

        needed_trades = int(off_resource_cards_to_buy)

        dominant_direct_deficit = float(deficits.get(dominant_resource, 0))
        dominant_trade_cards_needed_raw = float(needed_trades * effective_trade_rate)
        dominant_trade_cards_needed_adjusted = max(
            0.0,
            dominant_trade_cards_needed_raw - dominant_surplus_in_hand,
        )

        total_dominant_cards_needed_adjusted = (
            dominant_direct_deficit + dominant_trade_cards_needed_adjusted
        )

        if num_players is None:
            resolved_num_players = self._get_num_players()
        else:
            try:
                resolved_num_players = int(num_players)
            except Exception:
                resolved_num_players = self._get_num_players()

        resolved_num_players = max(1, int(resolved_num_players))

        try:
            resolved_horizon_turns = int(horizon_turns)
        except Exception:
            resolved_horizon_turns = 9
        resolved_horizon_turns = max(1, resolved_horizon_turns)

        horizon_rolls = float(resolved_horizon_turns * resolved_num_players)
        expected_future_dominant_cards = float(dominant_pips * horizon_rolls / 36.0)
        expected_dominant_cards_by_horizon = float(dominant_in_hand + expected_future_dominant_cards)

        # Only care about overflow if the 4-cap is plausibly binding.
        cap_bind_risk = bool(
            expected_future_dominant_cards > 4.0
            or expected_dominant_cards_by_horizon > 4.5
            or dominant_trade_cards_needed_adjusted > 4.0
        )

        can_fund_within_horizon = bool(
            expected_dominant_cards_by_horizon + 1e-9 >= total_dominant_cards_needed_adjusted
        )

        trigger_candidate = bool(
            dominant_pips >= 5.0
            and needs_trade_conversion
            and weak_off_resource_exists
            and needed_trades >= 1
            and can_fund_within_horizon
            and cap_bind_risk
        )

        return {
            "trigger_candidate": trigger_candidate,
            "dominant_resource": dominant_resource,
            "dominant_resource_pips": float(dominant_pips),
            "effective_trade_rate": int(effective_trade_rate),
            "needed_trades": int(needed_trades),
            "off_resource_cards_to_buy": int(off_resource_cards_to_buy),
            "dominant_direct_deficit": float(dominant_direct_deficit),
            "dominant_trade_cards_needed_raw": float(dominant_trade_cards_needed_raw),
            "dominant_trade_cards_needed_adjusted": float(dominant_trade_cards_needed_adjusted),
            "dominant_surplus_in_hand": float(dominant_surplus_in_hand),
            "expected_future_dominant_cards": float(expected_future_dominant_cards),
            "expected_dominant_cards_by_horizon": float(expected_dominant_cards_by_horizon),
            "horizon_turns": int(resolved_horizon_turns),
            "horizon_rolls": float(horizon_rolls),
            "can_fund_within_horizon": bool(can_fund_within_horizon),
            "cap_bind_risk": bool(cap_bind_risk),
            "weak_off_resource_exists": bool(weak_off_resource_exists),
            "normalized_target_type": normalized_target_type,
        }

    def _dominant_port_overflow_trigger(
        self,
        vertices,
        hand,
        player_ports,
        target_type: str,
        extra_roads_needed: int = 0,
    ) -> dict:
        """
        Detect overflow cases where the 0..4 cap can make the evaluator unreliable
        because one resource is produced exceptionally strongly and is likely to be
        traded repeatedly into missing resources.

        New trigger rule:
        - dominant produced resource exists
        - dominant resource pips are meaningfully strong
        - the target really depends on trade conversion
        - the dominant resource can realistically fund the required trades
        within the bounded horizon
        - the 4-cap is plausibly binding
        """
        funding_info = self._can_dominant_resource_fund_trades_within_horizon(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=target_type,
            extra_roads_needed=extra_roads_needed,
            horizon_turns=9,   # agreed bounded lookahead
            num_players=4,     # current project setup
        )

        return {
            "trigger": bool(funding_info.get("trigger_candidate", False)),
            "dominant_resource": funding_info.get("dominant_resource"),
            "dominant_resource_pips": float(funding_info.get("dominant_resource_pips", 0.0)),
            "effective_trade_rate": int(funding_info.get("effective_trade_rate", 4)),

            # extra diagnostics are harmless to existing callers and useful for debugging
            "needed_trades": int(funding_info.get("needed_trades", 0)),
            "off_resource_cards_to_buy": int(funding_info.get("off_resource_cards_to_buy", 0)),
            "dominant_direct_deficit": float(funding_info.get("dominant_direct_deficit", 0.0)),
            "dominant_trade_cards_needed_raw": float(
                funding_info.get("dominant_trade_cards_needed_raw", 0.0)
            ),
            "dominant_trade_cards_needed_adjusted": float(
                funding_info.get("dominant_trade_cards_needed_adjusted", 0.0)
            ),
            "dominant_surplus_in_hand": float(
                funding_info.get("dominant_surplus_in_hand", 0.0)
            ),
            "expected_future_dominant_cards": float(
                funding_info.get("expected_future_dominant_cards", 0.0)
            ),
            "expected_dominant_cards_by_horizon": float(
                funding_info.get("expected_dominant_cards_by_horizon", 0.0)
            ),
            "horizon_turns": int(funding_info.get("horizon_turns", 9)),
            "horizon_rolls": float(funding_info.get("horizon_rolls", 36.0)),
            "can_fund_within_horizon": bool(
                funding_info.get("can_fund_within_horizon", False)
            ),
            "cap_bind_risk": bool(funding_info.get("cap_bind_risk", False)),
            "weak_off_resource_exists": bool(
                funding_info.get("weak_off_resource_exists", False)
            ),
            "normalized_target_type": funding_info.get("normalized_target_type"),
        }
    
    def _overflow_port_monopoly_score(
        self,
        vertices,
        hand,
        player_ports,
        target_type: str,
        extra_roads_needed: int = 0,
    ) -> dict:
        """
        Special fallback scorer for overflow cases where one produced resource is so strong
        that the 0..4 cap can understate its trading value.

        Core idea:
        - detect the dominant produced resource from pips
        - determine its effective trade rate from ports (2:1 / 3:1 / 4:1)
        - compute the missing burden in dominant-resource equivalents
        - subtract already-held dominant-resource surplus that can be traded immediately
        - estimate rounds from:
            * dominant production alone
            * blended direct + traded production for off-resources

        Notes:
        - This is a fallback/patch for overflow cases, not a replacement for the full model.
        - Returned score is intended to be used as the overflow fallback score.
        - Score is in player-turns / rounds (not raw roll count).
        """
        hand = hand or [0, 0, 0, 0, 0]
        player_ports = player_ports or {}

        trigger_info = self._dominant_port_overflow_trigger(
            vertices=vertices,
            hand=hand,
            player_ports=player_ports,
            target_type=target_type,
            extra_roads_needed=extra_roads_needed,
        )

        if not trigger_info.get("trigger", False):
            return {
                "trigger": False,
                "score": 9999.0,
                "dominant_resource": trigger_info.get("dominant_resource"),
                "dominant_resource_pips": float(trigger_info.get("dominant_resource_pips", 0.0)),
                "effective_trade_rate": int(trigger_info.get("effective_trade_rate", 4)),
                "dominant_equivalent_needed": 0.0,
                "dominant_equivalent_raw": 0.0,
                "dominant_surplus_in_hand": float(trigger_info.get("dominant_surplus_in_hand", 0.0)),
                "target_cost": {},
                "deficits": {},
                "production_pips": {},
                "equivalent_turns": 9999.0,
                "blended_turns": 9999.0,
                "needed_trades": int(trigger_info.get("needed_trades", 0)),
                "off_resource_cards_to_buy": int(trigger_info.get("off_resource_cards_to_buy", 0)),
                "dominant_direct_deficit": float(trigger_info.get("dominant_direct_deficit", 0.0)),
                "dominant_trade_cards_needed_raw": float(
                    trigger_info.get("dominant_trade_cards_needed_raw", 0.0)
                ),
                "dominant_trade_cards_needed_adjusted": float(
                    trigger_info.get("dominant_trade_cards_needed_adjusted", 0.0)
                ),
                "expected_future_dominant_cards": float(
                    trigger_info.get("expected_future_dominant_cards", 0.0)
                ),
                "expected_dominant_cards_by_horizon": float(
                    trigger_info.get("expected_dominant_cards_by_horizon", 0.0)
                ),
                "horizon_turns": int(trigger_info.get("horizon_turns", 9)),
                "horizon_rolls": float(trigger_info.get("horizon_rolls", 36.0)),
                "can_fund_within_horizon": bool(
                    trigger_info.get("can_fund_within_horizon", False)
                ),
                "cap_bind_risk": bool(trigger_info.get("cap_bind_risk", False)),
                "weak_off_resource_exists": bool(
                    trigger_info.get("weak_off_resource_exists", False)
                ),
                "normalized_target_type": trigger_info.get("normalized_target_type"),
            }

        dominant_resource = trigger_info["dominant_resource"]
        dominant_resource_pips = float(trigger_info["dominant_resource_pips"])
        trade_rate = int(trigger_info.get("effective_trade_rate", 4))

        target_cost = self._game_order_target_cost(
            target_type,
            extra_roads_needed=extra_roads_needed,
        )

        dominant_info = self._get_dominant_resource_from_vertices(vertices)
        pips = dominant_info.get("all_pips", {}) or {}

        current_hand = {
            "wheat": int(hand[0]) if len(hand) > 0 else 0,
            "ore": int(hand[1]) if len(hand) > 1 else 0,
            "wood": int(hand[2]) if len(hand) > 2 else 0,
            "brick": int(hand[3]) if len(hand) > 3 else 0,
            "wool": int(hand[4]) if len(hand) > 4 else 0,
        }

        deficits = {}
        for res, need in target_cost.items():
            have = current_hand.get(res, 0)
            if have < need:
                deficits[res] = int(need - have)

        dominant_keep_need = float(target_cost.get(dominant_resource, 0))
        dominant_in_hand = float(current_hand.get(dominant_resource, 0))
        dominant_surplus_in_hand = float(
            trigger_info.get(
                "dominant_surplus_in_hand",
                max(0.0, dominant_in_hand - dominant_keep_need),
            )
        )

        dominant_direct_deficit = float(
            trigger_info.get("dominant_direct_deficit", float(deficits.get(dominant_resource, 0)))
        )

        needed_trades = int(trigger_info.get("needed_trades", 0))
        off_resource_cards_to_buy = int(trigger_info.get("off_resource_cards_to_buy", 0))

        dominant_equivalent_raw = float(
            trigger_info.get(
                "dominant_trade_cards_needed_raw",
                float(dominant_direct_deficit + (needed_trades * trade_rate)),
            )
        )

        dominant_trade_cards_needed_adjusted = float(
            trigger_info.get(
                "dominant_trade_cards_needed_adjusted",
                max(0.0, float(needed_trades * trade_rate) - dominant_surplus_in_hand),
            )
        )

        dominant_equivalent_needed = float(
            dominant_direct_deficit + dominant_trade_cards_needed_adjusted
        )

        # ------------------------------------------------------------
        # 1) Dominant-equivalent time:
        #    "If I only trust the dominant-resource engine, how long until the
        #     adjusted dominant-equivalent burden is generated?"
        #
        # In a 4-player game:
        #   1 player-turn = 1 round = 4 roll events
        #   expected cards per player-turn = pips * 4 / 36 = pips / 9
        # ------------------------------------------------------------
        num_players = self._get_num_players()
        dominant_cards_per_turn = dominant_resource_pips * float(num_players) / 36.0
        if dominant_cards_per_turn <= 1e-9:
            equivalent_turns = 9999.0
        else:
            equivalent_turns = dominant_equivalent_needed / dominant_cards_per_turn

        # ------------------------------------------------------------
        # 2) Blended time:
        #    "How long until each missing resource can plausibly be filled by its own
        #     direct production plus conversion from the dominant engine?"
        #
        # We give immediate credit for dominant surplus already in hand by converting
        # that surplus into whole off-resource cards before timing the future flow.
        # ------------------------------------------------------------
        immediate_trade_cards_available = int(max(0.0, dominant_surplus_in_hand) // max(trade_rate, 1))

        off_deficits_sorted = sorted(
            [(res, int(amt)) for res, amt in deficits.items() if res != dominant_resource and int(amt) > 0],
            key=lambda x: float(pips.get(x[0], 0.0))
        )

        remaining_off_deficits = {res: amt for res, amt in off_deficits_sorted}

        # Spend immediate dominant surplus first on the weakest off-resources.
        for res, amt in off_deficits_sorted:
            if immediate_trade_cards_available <= 0:
                break
            used_now = min(int(amt), int(immediate_trade_cards_available))
            if used_now <= 0:
                continue
            remaining_off_deficits[res] = max(0, int(amt) - used_now)
            immediate_trade_cards_available -= used_now

        blended_component_turns = []

        # Dominant resource itself must still be produced directly.
        if dominant_direct_deficit > 0.0:
            if dominant_cards_per_turn <= 1e-9:
                blended_component_turns.append(9999.0)
            else:
                blended_component_turns.append(dominant_direct_deficit / dominant_cards_per_turn)

        # Off-resources can arrive directly and/or via traded dominant production.
        for res, original_amt in off_deficits_sorted:
            remaining_deficit_amt = int(remaining_off_deficits.get(res, original_amt))
            if remaining_deficit_amt <= 0:
                continue

            direct_pips = float(pips.get(res, 0.0))
            direct_cards_per_turn = direct_pips * float(num_players) / 36.0
            traded_cards_per_turn = dominant_cards_per_turn / float(max(trade_rate, 1))
            effective_cards_per_turn = direct_cards_per_turn + traded_cards_per_turn

            if effective_cards_per_turn <= 1e-9:
                blended_component_turns.append(9999.0)
            else:
                blended_component_turns.append(float(remaining_deficit_amt) / effective_cards_per_turn)

        if not blended_component_turns:
            blended_turns = 0.0
        else:
            blended_turns = max(float(x) for x in blended_component_turns)

        # Conservative correction:
        # use the slower of the two heuristic estimates so overflow fallback
        # does not remain unrealistically optimistic.
        score = max(float(equivalent_turns), float(blended_turns))
        score = max(0.0, score)

        return {
            "trigger": True,
            "score": score,
            "dominant_resource": dominant_resource,
            "dominant_resource_pips": dominant_resource_pips,
            "effective_trade_rate": trade_rate,
            "dominant_equivalent_needed": float(dominant_equivalent_needed),
            "dominant_equivalent_raw": float(dominant_equivalent_raw),
            "dominant_surplus_in_hand": float(dominant_surplus_in_hand),
            "target_cost": dict(target_cost),
            "deficits": dict(deficits),
            "production_pips": dict(pips),
            "equivalent_turns": float(equivalent_turns),
            "blended_turns": float(blended_turns),

            # richer diagnostics from the stricter trigger
            "needed_trades": int(needed_trades),
            "off_resource_cards_to_buy": int(off_resource_cards_to_buy),
            "dominant_direct_deficit": float(dominant_direct_deficit),
            "dominant_trade_cards_needed_raw": float(
                trigger_info.get("dominant_trade_cards_needed_raw", dominant_equivalent_raw)
            ),
            "dominant_trade_cards_needed_adjusted": float(dominant_trade_cards_needed_adjusted),
            "expected_future_dominant_cards": float(
                trigger_info.get("expected_future_dominant_cards", 0.0)
            ),
            "expected_dominant_cards_by_horizon": float(
                trigger_info.get("expected_dominant_cards_by_horizon", 0.0)
            ),
            "horizon_turns": int(trigger_info.get("horizon_turns", 9)),
            "horizon_rolls": float(trigger_info.get("horizon_rolls", 36.0)),
            "can_fund_within_horizon": bool(
                trigger_info.get("can_fund_within_horizon", False)
            ),
            "cap_bind_risk": bool(trigger_info.get("cap_bind_risk", False)),
            "weak_off_resource_exists": bool(
                trigger_info.get("weak_off_resource_exists", False)
            ),
            "normalized_target_type": trigger_info.get("normalized_target_type"),
        }
    
    def _build_target_vector(self, target_type: str) -> torch.Tensor:
        """
        Build a 3125-length indicator vector for a target event.

        Internal Markov resource order:
            [brick, lumber, wool, wheat, ore]

        A state is marked 1.0 if it satisfies the requirements for target_type,
        otherwise 0.0.

        Supported targets:
            - settlement / settlement_0r
            - settlement_1r
            - settlement_2r
            - city
            - dev_card
            - dev_card_4
        """
        req = self._get_target_requirements(target_type)
        v = torch.zeros(3125, device=self.device, dtype=torch.float32)

        for b in range(5):
            for l_ in range(5):
                for s in range(5):
                    for w in range(5):
                        for o in range(5):
                            idx = b * 625 + l_ * 125 + s * 25 + w * 5 + o

                            state_vec = [b, l_, s, w, o]
                            if all(state_vec[i] >= req[i] for i in range(5)):
                                v[idx] = 1.0

        return v
    
    def _get_dominant_resource_from_vertices(self, vertices):
        """
        Determine the dominant produced resource from the current position.

        Returns:
            {
                "resource": <str or None>,   # one of: wheat, ore, wood, brick, wool
                "pips": <float>,             # production pips for that resource
                "all_pips": <dict>           # full per-resource pip dict in game order names
            }

        Dominance is based purely on production, not on ports.
        """
        pips = self._game_order_pips_from_vertices(vertices)

        if not pips:
            return {
                "resource": None,
                "pips": 0.0,
                "all_pips": {
                    "wheat": 0.0,
                    "ore": 0.0,
                    "wood": 0.0,
                    "brick": 0.0,
                    "wool": 0.0,
                },
            }

        dominant_resource = None
        dominant_pips = -1.0

        # Fixed order keeps ties deterministic
        for res in ["wheat", "ore", "wood", "brick", "wool"]:
            val = float(pips.get(res, 0.0))
            if val > dominant_pips:
                dominant_pips = val
                dominant_resource = res

        if dominant_pips < 0.0:
            dominant_resource = None
            dominant_pips = 0.0

        return {
            "resource": dominant_resource,
            "pips": float(dominant_pips),
            "all_pips": dict(pips),
        }

    def _get_effective_trade_rate_for_resource(self, resource_name: str, player_ports: dict) -> int:
        """
        Return the effective trade rate for a given GAME-order resource name.

        resource_name:
            one of: wheat, ore, wood, brick, wool

        Priority:
            1) matching specific 2:1 port
            2) generic 3:1 port
            3) bank 4:1
        """
        if not resource_name:
            return 4

        player_ports = player_ports or {}
        rn = str(resource_name).lower()

        aliases = {
            "wheat": {"wheat"},
            "ore": {"ore"},
            "wood": {"wood", "lumber"},
            "brick": {"brick"},
            "wool": {"wool", "sheep"},
        }

        wanted = aliases.get(rn, {rn})

        # 1) matching specific 2:1 port
        for raw_name, rate in player_ports.items():
            key = str(raw_name).lower()
            if key == "generic":
                continue
            try:
                if int(rate) == 2 and key in wanted:
                    return 2
            except Exception:
                continue

        # 2) generic 3:1 port
        try:
            if "generic" in player_ports and int(player_ports["generic"]) == 3:
                return 3
        except Exception:
            pass

        # 3) bank
        return 4
    
    def _choose_adaptive_cap_vec(self, overflow_info: dict, target_type: str) -> list[int]:
        """
        Choose per-resource caps for the adaptive expanded evaluator.

        Internal Markov resource order:
            [brick, lumber/wood, wool, wheat, ore]

        Strategy:
        - expand only the dominant resource dimension
        - keep all other dimensions at 4
        - use a conservative ladder so state space stays manageable
        """
        base_caps = [4, 4, 4, 4, 4]

        dominant_resource = str(overflow_info.get("dominant_resource") or "").strip().lower()
        trade_rate = int(overflow_info.get("effective_trade_rate", 4))
        dominant_pips = float(overflow_info.get("dominant_resource_pips", 0.0))
        needed_trades = int(overflow_info.get("needed_trades", 0))
        expected_dom = float(overflow_info.get("expected_dominant_cards_by_horizon", 0.0))

        res_to_idx = {
            "brick": 0,
            "lumber": 1,
            "wood": 1,
            "wool": 2,
            "sheep": 2,
            "wheat": 3,
            "ore": 4,
        }

        dom_idx = res_to_idx.get(dominant_resource)
        if dom_idx is None:
            return base_caps

        # Conservative default ladder
        dom_cap = 6

        if trade_rate <= 2:
            dom_cap = 10
        elif trade_rate == 3:
            dom_cap = 8

        # Mildly lift if the burden / horizon expectation is bigger than the ladder base
        if expected_dom >= 9.5 or needed_trades >= 4 or dominant_pips >= 10.0:
            dom_cap = max(dom_cap, 10)
        elif expected_dom >= 7.5 or needed_trades >= 3 or dominant_pips >= 8.0:
            dom_cap = max(dom_cap, 8)
        else:
            dom_cap = max(dom_cap, 6)

        # Keep state space bounded for the first rollout
        dom_cap = max(5, min(int(dom_cap), 12))

        base_caps[dom_idx] = dom_cap
        return base_caps


    def _num_states_from_caps(self, cap_vec: list[int]) -> int:
        """
        Number of states in a mixed-radix capped space.

        Example:
            [4,4,4,4,4] -> 5^5 = 3125
            [4,4,4,8,4] -> 5*5*5*9*5
        """
        if not isinstance(cap_vec, (list, tuple)) or len(cap_vec) != 5:
            raise ValueError(f"cap_vec must have length 5, got {cap_vec}")

        n = 1
        for c in cap_vec:
            cc = max(0, int(c))
            n *= (cc + 1)
        return int(n)


    def _vec_to_state_with_caps(self, vec: list[int], cap_vec: list[int]) -> int:
        """
        Mixed-radix encoding of:
            [brick, lumber, wool, wheat, ore]
        using per-dimension caps.

        Matches dimension order exactly as stored in vec.
        """
        if len(vec) != 5 or len(cap_vec) != 5:
            raise ValueError(f"vec and cap_vec must both have length 5, got {vec}, {cap_vec}")

        idx = 0
        mul = 1
        for v, c in zip(vec, cap_vec):
            cc = max(0, int(c))
            vv = min(max(0, int(v)), cc)
            idx += vv * mul
            mul *= (cc + 1)
        return int(idx)


    def _state_to_vec_with_caps(self, state_idx: int, cap_vec: list[int]) -> list[int]:
        """
        Inverse of _vec_to_state_with_caps(...).

        Returns:
            [brick, lumber, wool, wheat, ore]
        """
        if len(cap_vec) != 5:
            raise ValueError(f"cap_vec must have length 5, got {cap_vec}")

        idx = max(0, int(state_idx))
        vec = [0] * 5

        for i in range(5):
            base = max(0, int(cap_vec[i])) + 1
            vec[i] = idx % base
            idx //= base

        return vec


    def _hand_to_state_index_with_caps(self, hand: list[int], cap_vec: list[int]) -> int:
        """
        Convert game hand order:
            [Wheat, Ore, Wood, Brick, Wool]
        into adaptive mixed-radix Markov state index.

        Internal Markov order:
            [Brick, Lumber/Wood, Wool, Wheat, Ore]
        """
        if hand is None:
            hand = [0, 0, 0, 0, 0]

        h = [0, 0, 0, 0, 0]
        for i in range(min(5, len(hand))):
            try:
                h[i] = max(0, int(hand[i]))
            except Exception:
                h[i] = 0

        wheat, ore, wood, brick, wool = h
        markov_vec = [brick, wood, wool, wheat, ore]

        clipped = [
            min(markov_vec[i], max(0, int(cap_vec[i])))
            for i in range(5)
        ]
        return self._vec_to_state_with_caps(clipped, cap_vec)

    def build_matrix_with_caps(self, die_num, cap_vec: list[int]) -> torch.Tensor:
        """
        Adaptive production matrix builder using per-resource caps.

        Internal resource order:
            0 brick, 1 lumber, 2 wool, 3 wheat, 4 ore
        """
        die_num = self._normalize_rolls(die_num)

        if not isinstance(cap_vec, (list, tuple)) or len(cap_vec) != 5:
            raise ValueError(f"cap_vec must have length 5, got {cap_vec}")

        cap_vec = [max(0, int(x)) for x in cap_vec]
        n_states = self._num_states_from_caps(cap_vec)

        die_prob = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

        M = torch.zeros((n_states, n_states), dtype=torch.float32, device=self.device)
        roll_list = defaultdict(lambda: [0] * 5)

        for res_idx, rolls in enumerate(die_num):
            for roll in rolls:
                roll_list[int(roll)][res_idx] += 1

        len_list = [[] for _ in range(6)]
        for roll, gains in roll_list.items():
            count = sum(1 for g in gains if g > 0)
            len_list[count].append((roll, gains))

        for idx in range(n_states):
            current = self._state_to_vec_with_caps(idx, cap_vec)
            count = 0

            for n_res in range(1, 6):
                for roll, gains in len_list[n_res]:
                    next_state = current[:]
                    for r in range(5):
                        if gains[r]:
                            next_state[r] = min(cap_vec[r], next_state[r] + gains[r])

                    next_idx = self._vec_to_state_with_caps(next_state, cap_vec)
                    p = die_prob.get(int(roll), 0)
                    M[idx, next_idx] += p
                    count += p

            M[idx, idx] += (36 - count)

        return (M / 36.0).to(self.device)


    def _build_target_vector_with_caps(self, target_type: str, cap_vec: list[int]) -> torch.Tensor:
        """
        Adaptive indicator vector for one target in a capped mixed-radix state space.

        Returns a vector of length _num_states_from_caps(cap_vec) with:
            1.0 for states satisfying the target
            0.0 otherwise
        """
        if not isinstance(cap_vec, (list, tuple)) or len(cap_vec) != 5:
            raise ValueError(f"cap_vec must have length 5, got {cap_vec}")

        tt = str(target_type or "").strip().lower()
        if tt == "new_settlement":
            tt = "settlement"
        elif tt == "upgrade_to_city":
            tt = "city"
        elif tt == "buy_discovery_card":
            tt = "dev_card"
        elif tt == "buy_4_discovery_cards":
            tt = "dev_card_4"

        if tt == "settlement":
            req = self._get_target_requirements("new_settlement")
        elif tt == "settlement_0r":
            req = self._get_target_requirements("settlement")
        elif tt == "settlement_1r":
            req = [2, 2, 1, 1, 0]
        elif tt == "settlement_2r":
            req = [3, 3, 1, 1, 0]
        elif tt == "city":
            req = self._get_target_requirements("city")
        elif tt == "dev_card":
            req = self._get_target_requirements("dev_card")
        elif tt == "dev_card_4":
            req = self._get_target_requirements("dev_card_4")
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        n_states = self._num_states_from_caps(cap_vec)
        v = torch.zeros(n_states, device=self.device, dtype=torch.float32)

        for idx in range(n_states):
            vec = self._state_to_vec_with_caps(idx, cap_vec)
            if (
                vec[0] >= req[0]
                and vec[1] >= req[1]
                and vec[2] >= req[2]
                and vec[3] >= req[3]
                and vec[4] >= req[4]
            ):
                v[idx] = 1.0

        return v


    def expected_vectors_with_caps(self, matrix: torch.Tensor, cap_vec: list[int]):
        """
        Adaptive equivalent of expected_vectors(...).

        Returns, in this order:
            1) settlement_0r
            2) settlement_1r
            3) settlement_2r
            4) city
            5) dev_card
            6) dev_card_4

        Important:
        - This solves true expected time-to-hit-target.
        - Target-satisfying states have expected time 0.
        - Non-target states solve:
            t = 1 + Q_non_target @ t
        """
        if not isinstance(cap_vec, (list, tuple)) or len(cap_vec) != 5:
            raise ValueError(f"cap_vec must have length 5, got {cap_vec}")

        cap_vec = [max(0, int(x)) for x in cap_vec]

        matrix = matrix.to(self.device).float()

        row_sums = matrix.sum(dim=1, keepdim=True)
        mask = row_sums > 1e-12
        matrix = torch.where(mask, matrix / row_sums, matrix)

        n_states = matrix.shape[0]
        expected_n_states = self._num_states_from_caps(cap_vec)

        if n_states != expected_n_states:
            raise ValueError(
                f"matrix has {n_states} states, but cap_vec {cap_vec} implies "
                f"{expected_n_states} states"
            )

        def _expected_hitting_time_for_target(target_type: str) -> torch.Tensor:
            target_vec = self._build_target_vector_with_caps(
                target_type,
                cap_vec,
            ).to(self.device).bool()

            # Already-satisfied states take 0 turns.
            out = torch.zeros(n_states, device=self.device, dtype=torch.float32)

            non_target_idx = torch.where(~target_vec)[0]

            if non_target_idx.numel() == 0:
                return out

            # Q restricted to non-target states only.
            Q = matrix.index_select(0, non_target_idx).index_select(1, non_target_idx)
            I = torch.eye(Q.shape[0], device=self.device, dtype=torch.float32)
            ones = torch.ones(Q.shape[0], device=self.device, dtype=torch.float32)

            try:
                t = torch.linalg.solve(I - Q, ones)
            except Exception:
                try:
                    t = torch.linalg.pinv(I - Q) @ ones
                except Exception:
                    t = torch.full_like(ones, 9999.0)

            t = torch.nan_to_num(
                t,
                nan=9999.0,
                posinf=9999.0,
                neginf=9999.0,
            )

            t = torch.clamp(t, min=0.0, max=9999.0)

            out[non_target_idx] = t
            return out

        e_settlement_0r = _expected_hitting_time_for_target("settlement_0r")
        e_settlement_1r = _expected_hitting_time_for_target("settlement_1r")
        e_settlement_2r = _expected_hitting_time_for_target("settlement_2r")
        e_city = _expected_hitting_time_for_target("city")
        e_dev = _expected_hitting_time_for_target("dev_card")
        e_dev4 = _expected_hitting_time_for_target("dev_card_4")

        return (
            e_settlement_0r,
            e_settlement_1r,
            e_settlement_2r,
            e_city,
            e_dev,
            e_dev4,
        )


    def apply_trading_layer_with_caps(
        self,
        M_original: torch.Tensor,
        cap_vec: list[int],
        player_ports: dict = None,
        target_type: str = "settlement",
        use_bank_4to1: bool = True,
        max_trades_per_roll: int = 4,
    ) -> torch.Tensor:
        """
        Adaptive equivalent of apply_trading_layer(...), using mixed-radix states and cap_vec.
        """
        if player_ports is None:
            player_ports = {}

        if not isinstance(cap_vec, (list, tuple)) or len(cap_vec) != 5:
            raise ValueError(f"cap_vec must have length 5, got {cap_vec}")

        cap_vec = [max(0, int(x)) for x in cap_vec]
        n_states = self._num_states_from_caps(cap_vec)

        tt = str(target_type or "").strip().lower()
        if tt == "settlement":
            req = [1, 1, 1, 1, 0]
        elif tt == "settlement_0r":
            req = [1, 1, 1, 1, 0]
        elif tt == "settlement_1r":
            req = [2, 2, 1, 1, 0]
        elif tt == "settlement_2r":
            req = [3, 3, 1, 1, 0]
        elif tt == "new_settlement":
            req = [2, 2, 1, 1, 0]
        elif tt == "city":
            req = [0, 0, 0, 2, 3]
        elif tt == "dev_card":
            req = [0, 0, 1, 1, 1]
        elif tt == "dev_card_4":
            req = [0, 0, 4, 4, 4]
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

        M_new = torch.zeros((n_states, n_states), dtype=torch.float32, device=self.device)
        M_orig = M_original.to(self.device)

        for i in range(n_states):
            for j in range(n_states):
                prob = M_orig[i, j]
                if prob <= 0.0:
                    continue

                vec = self._state_to_vec_with_caps(j, cap_vec)
                trades_made = 0

                while trades_made < max_trades_per_roll:
                    best_gain = -1
                    best_new_vec = None

                    for r in range(5):
                        if self.RES_NAMES[r] in player_ports:
                            ratio = int(player_ports[self.RES_NAMES[r]])
                        elif "generic" in player_ports:
                            ratio = int(player_ports["generic"])
                        elif use_bank_4to1:
                            ratio = 4
                        else:
                            continue

                        if vec[r] < ratio:
                            continue

                        deficit = [max(0, req[k] - vec[k]) for k in range(5)]

                        for target_r in range(5):
                            if target_r == r or deficit[target_r] == 0:
                                continue

                            new_vec = vec[:]
                            new_vec[r] -= ratio
                            new_vec[target_r] += 1
                            new_vec[target_r] = min(cap_vec[target_r], new_vec[target_r])

                            new_deficit = [max(0, req[k] - new_vec[k]) for k in range(5)]
                            gain = sum(d - nd for d, nd in zip(deficit, new_deficit))

                            if gain > best_gain:
                                best_gain = gain
                                best_new_vec = new_vec

                    if best_gain <= 0 or best_new_vec is None:
                        break

                    vec = best_new_vec
                    trades_made += 1

                k = self._vec_to_state_with_caps(vec, cap_vec)
                M_new[i, k] += prob

            row_sum = M_new[i].sum()
            if abs(float(row_sum) - 1.0) > 1e-9:
                M_new[i, i] += (1.0 - row_sum)

        return M_new


    def _get_traded_expected_vectors_adaptive(self, vertices, player_ports, target_type, cap_vec):
        """
        Adaptive traded EV cache for overflow-triggered heavy evaluation.

        Cache key:
            (position_key, ports_key, target_type, cap_key)
        """
        position_key = self._position_key(vertices)
        ports_key = self._ports_key(player_ports)
        cap_key = tuple(int(x) for x in cap_vec)

        cache_key = (position_key, ports_key, str(target_type), cap_key)
        if cache_key in self.adaptive_traded_ev_cache:
            return self.adaptive_traded_ev_cache[cache_key]

        matrix_cache_key = (position_key, cap_key)
        if matrix_cache_key in self.adaptive_matrix_cache:
            M_prod = self.adaptive_matrix_cache[matrix_cache_key]
        else:
            rolls = self._combine_vertex_rolls(position_key)
            M_prod = self.build_matrix_with_caps(rolls, cap_vec)
            self.adaptive_matrix_cache[matrix_cache_key] = M_prod

        M_trade = self.apply_trading_layer_with_caps(
            M_prod,
            cap_vec=cap_vec,
            player_ports=player_ports or {},
            target_type=target_type,
        )

        ev = self.expected_vectors_with_caps(M_trade, cap_vec)
        self.adaptive_traded_ev_cache[cache_key] = ev
        return ev    