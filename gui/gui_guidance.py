"""
Handles all on-screen guidance / instructions for human players.
Clean state machine using the central animate_queue_elements.
"""

import pygame
import math
from enum import Enum, auto
from typing import Optional, Tuple, List
from core.constants import MG
from gui.gui_constants import WIN, COLORS, IMAGES, SOUNDS, POSITIONS, Font
from gui.gui_human_player import GUIHumanPlayer
from core.game import Player


class PlacementState(Enum):
    IDLE = auto()
    CHOOSING_SETTLEMENT = auto()
    SETTLEMENT_SELECTED = auto()
    CHOOSING_ROAD = auto()
    ROAD_SELECTED = auto()


class HumanGuidance:
    def __init__(self, gui):
        self.gui = gui
        self.state: PlacementState = PlacementState.IDLE
        self.player: Optional[Player] = None
        self.placement_step: Optional[int] = None
        self.selected_inter_id: Optional[int] = None
        self.selected_road: Optional[Tuple[int, int]] = None
        self.available_inters: List[int] = []
        self.available_roads: List[Tuple[int, int]] = []
        self.confirm_center: Optional[Tuple[int, int]] = None
        self.last_pulse = 0

    def is_placing(self) -> bool:
        """Used by gui_human_player.py to decide button states."""
        return self.state != PlacementState.IDLE

    def start_settlement_phase(self, player: Player):

        print("Starting human placement phase - queue before clear:", len(self.gui.animate_queue_elements))
        self.gui.animate_queue_elements.clear()
        print("Queue after forced clear:", len(self.gui.animate_queue_elements))

        self.player = player
        self.placement_step = self.gui.game.ip.current_step
        self.state = PlacementState.CHOOSING_SETTLEMENT
        self.selected_inter_id = None
        self.selected_road = None
        self.confirm_center = None

        # ─── KEY FIX ───
        # Force clean board redraw BEFORE adding new highlights
        self.gui.draw_board_base(self.gui.game.board)
        self.gui.draw_all_permanent_buildings(self.gui.game.board)
        # Optionally also clear any leftover animation queue explicitly
        self.gui.animate_queue_elements.clear()
        # ─── end fix ───

        self._compute_available_settlements()
        self._highlight_available("settlement")
        self.gui.set_button("next_turn2", False)
        self.gui.draw_guidance_text("Click a highlighted intersection for your settlement", y_offset=25)
        pygame.display.update()

    def start_road_phase(self):

        print("Starting human placement phase - queue before clear:", len(self.gui.animate_queue_elements))
        self.gui.animate_queue_elements.clear()
        print("Queue after forced clear:", len(self.gui.animate_queue_elements))
        
        self.state = PlacementState.CHOOSING_ROAD
        self.selected_road = None
        self.confirm_center = None

        # Same clean board redraw
        self.gui.draw_board_base(self.gui.game.board)
        self.gui.draw_all_permanent_buildings(self.gui.game.board)
        self.gui.animate_queue_elements.clear()   # extra safety

        self.gui.draw_board_base(self.gui.game.board)
        self.gui.draw_all_permanent_buildings(self.gui.game.board)
        self._compute_available_roads()
        self._highlight_available("road")
        self.gui.draw_guidance_text("Click a highlighted road connected to your settlement", y_offset=25)
        pygame.display.update()

    def _highlight_available(self, kind: str):
        """Highlight available spots using the central queue (new system)."""
        self.gui.animate_queue_elements.clear()
        color = COLORS[self.player.color.upper()]
        items = self.available_inters if kind == "settlement" else self.available_roads

        for item in items:
            if kind == "settlement":
                pos = POSITIONS["intersections"][item]
                self.gui.animate_queue_elements.append((pos, color, 20, "settlement"))
            else:
                x1, y1 = POSITIONS["intersections"][item[0]]
                x2, y2 = POSITIONS["intersections"][item[1]]
                mid = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.gui.animate_queue_elements.append((mid, color, 18, "road"))

        ### print(f"Highlighting {len(self.gui.animate_queue_elements)} {kind} positions (pulsing active)")

        pygame.display.update()

    # def clear(self):
    #     self.state = PlacementState.IDLE
    #     self.player = None
    #     self.selected_inter_id = None
    #     self.selected_road = None
    #     self.confirm_center = None
    #     self.gui.animate_queue_elements.clear()
    #     self.gui.set_button("next_turn2", True)
    #     pygame.display.update()

    def clear(self):
        self.state = PlacementState.IDLE
        self.player = None
        self.selected_inter_id = None
        self.selected_road = None
        self.confirm_center = None
        
        # DO NOT clear the animation queue here any more!
        # We want the "last placement" to stay for continuous pulsing.
        # _highlight_available() will clear it when the next placement phase starts.
        # self.gui.animate_queue_elements.clear()   # ← DELETE THIS LINE

        self.gui.set_button("next_turn2", True)
        pygame.display.update()

    def draw(self):
        ### if self.state == PlacementState.CHOOSING_SETTLEMENT:
        ###     print(f"draw() called - CHOOSING_SETTLEMENT active - queue size = {len(self.gui.animate_queue_elements)}")

        if not self.player:
            return

        now = pygame.time.get_ticks()

        # Pulsing of available options only while choosing
        if self.state in (PlacementState.CHOOSING_SETTLEMENT, PlacementState.CHOOSING_ROAD):
            if now - self.last_pulse > 300:
                kind = "settlement" if self.state == PlacementState.CHOOSING_SETTLEMENT else "road"
                self._highlight_available(kind)
                self.last_pulse = now

        # Show selected item
        if self.state == PlacementState.SETTLEMENT_SELECTED:
            self._show_selected("settlement")
        elif self.state == PlacementState.ROAD_SELECTED:
            self._show_selected("road")

        # Confirmation icons ONLY when we are waiting for OK/NOK
        if self.state in (PlacementState.SETTLEMENT_SELECTED, PlacementState.ROAD_SELECTED) and self.confirm_center:
            x, y = self.confirm_center
            WIN.blit(IMAGES["OKY"]["default"], (x + 35, y - 45))
            WIN.blit(IMAGES["NOK"]["default"], (x + 35, y + 10))

    def _compute_available_settlements(self):
        self.available_inters = [
            i for i in range(len(self.gui.game.board.intersections))
            if self.gui.game.can_build_intersection_tf(i, self.player)
        ]

        print(f"Round {self.gui.game.round} - Available settlement spots for {self.player.color}: {len(self.available_inters)} locations")
        if self.available_inters:
            print("   First few:", self.available_inters[:5])
        else:
            print("   → NO VALID SPOTS!")

    def _compute_available_roads(self):
        inter = self.gui.game.board.intersections[self.selected_inter_id]
        self.available_roads = [
            road for road in inter.three_roads
            if self.gui.game.board.can_build_road_for_color_tf(list(road), self.player.color)
        ]

    def _show_selected(self, kind: str):
        self.gui.animate_queue_elements.clear()
        self.gui.draw_board_base(self.gui.game.board)
        self.gui.draw_all_permanent_buildings(self.gui.game.board)
        color = COLORS[self.player.color.upper()]

        if kind == "settlement":
            pos = POSITIONS["intersections"][self.selected_inter_id]
            self.gui.animate_queue_elements.append((pos, color, 20, "settlement"))
        
        else:
            x1, y1 = POSITIONS["intersections"][self.selected_road[0]]
            x2, y2 = POSITIONS["intersections"][self.selected_road[1]]
            mid = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.gui.animate_queue_elements.append((mid, color, 18, "road"))

        # No need for pygame.display.update() here – usually done in main loop or after

    def _find_clicked_inter(self, pos):
        for iid, coord in POSITIONS["intersections"].items():
            if (pos[0] - coord[0])**2 + (pos[1] - coord[1])**2 < 225:
                return iid
        return None

    def _find_clicked_road(self, pos):
        for road in self.available_roads:
            x1, y1 = POSITIONS["intersections"][road[0]]
            x2, y2 = POSITIONS["intersections"][road[1]]
            mid = ((x1 + x2) // 2, (y1 + y2) // 2)
            if (pos[0] - mid[0])**2 + (pos[1] - mid[1])**2 < 225:
                return road
        return None

    def _place_settlement(self):
        """
        Place the human player's settlement during Initial Placement
        and keep the Player object in sync with the Board.
        """
        step = self.placement_step

        success = self.player.build_structure_initial_placement(
            "settlement",
            self.selected_inter_id,
            self.gui.game.board,
            placement_step=step,
        )
        if not success:
            print(f"Failed to place settlement at {self.selected_inter_id} for player {self.player.id}")
            return

        # Initial resources only for the 2nd placement round
        if self.gui.game.round == -1:
            self.gui.game.ip.distribute_initial_resources(self.selected_inter_id, self.player)

        print(
            f"Settlement placed at {self.selected_inter_id} by player {self.player.id} "
            f"during initial placement step {step}"
        )

        self.gui.update_scoreboard(self.gui.game)

    def _place_road(self):
        """
        Place the human player's road during Initial Placement
        and keep the Player object in sync with the Board.
        """
        step = self.placement_step
        road_id = tuple(sorted(self.selected_road))

        success = self.player.build_structure_initial_placement(
            "road",
            road_id,
            self.gui.game.board,
            placement_step=step,
        )
        if not success:
            print(f"Failed to place road at {road_id} for player {self.player.id}")
            return

        print(
            f"Road placed at {road_id} by player {self.player.id} "
            f"during initial placement step {step}"
        )

    def _finish_human_turn(self):
        """
        Human completed settlement + road.
        Now advance exactly one initial-placement step and hand control back to PLAY.
        """
        self.gui.draw_board_base(self.gui.game.board)
        self.gui.draw_all_permanent_buildings(self.gui.game.board)
        pygame.draw.rect(WIN, COLORS["LGRAY"], [10, 45, 400, 40])
        self.gui.update_round_turn(self.gui.game, special=False)

        # Show animation for the placement that was just completed
        self.gui.update_board(self.gui.game.board, "Last")
        self.gui.update_scoreboard(self.gui.game)

        # IMPORTANT: human path advances step HERE, not earlier
        self.gui.game.ip.current_step += 1
        print(f"Human placement finished – advanced current_step to {self.gui.game.ip.current_step}")

        # Rotate round/turn to the next setup actor
        self.gui.game.advance_turn()

        # Re-enable PLAY button for the next setup step
        GUIHumanPlayer.button_next_turn2(self.gui, self.gui.game, active=True)

        self.clear()
        pygame.display.update()

        print("Finished human turn — current_step =", self.gui.game.ip.current_step)

    def on_board_click(self, pos: Tuple[int, int]) -> bool:
        """
        Handle mouse click on the board during placement phases.
        Returns True if the click was processed (valid action taken).
        """
        handled = False

        if self.state == PlacementState.CHOOSING_SETTLEMENT:
            inter_id = self._find_clicked_inter(pos)
            if inter_id is not None and inter_id in self.available_inters:
                # valid click -> proceed
                pygame.mixer.Sound.play(SOUNDS["BUTTON"])
                self.selected_inter_id = inter_id
                self.state = PlacementState.SETTLEMENT_SELECTED
                self.confirm_center = POSITIONS["intersections"][inter_id]

                self.gui.animate_queue_elements.clear()  # remove pulsing highlights
                self._show_selected("settlement")
                self.gui.draw_guidance_text("Confirm settlement placement?", y_offset=25)
                pygame.display.update()
                handled = True
            else:
                # invalid settlement click -> error
                pygame.mixer.Sound.play(SOUNDS["ERROR"])

        elif self.state == PlacementState.CHOOSING_ROAD:
            road = self._find_clicked_road(pos)
            if road is not None and road in self.available_roads:
                # valid road click -> proceed
                pygame.mixer.Sound.play(SOUNDS["BUTTON"])
                self.selected_road = road
                self.state = PlacementState.ROAD_SELECTED
                x1, y1 = POSITIONS["intersections"][road[0]]
                x2, y2 = POSITIONS["intersections"][road[1]]
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                self.confirm_center = (mid_x, mid_y)

                self.gui.animate_queue_elements.clear()
                self._show_selected("road")
                self.gui.draw_guidance_text("Confirm road placement?", y_offset=25)
                pygame.display.update()
                handled = True
            else:
                # invalid road click -> error
                pygame.mixer.Sound.play(SOUNDS["ERROR"])

        return handled

    def on_confirmation(self, choice: str) -> None:
        """Handle OKY (confirm) or OKN (cancel) click."""
        if choice == "OKY":
            pygame.mixer.Sound.play(SOUNDS["BUTTON"])
            if self.state == PlacementState.SETTLEMENT_SELECTED:
                self._place_settlement()
                self.start_road_phase()           # move to road selection
            elif self.state == PlacementState.ROAD_SELECTED:
                self._place_road()
                self._finish_human_turn()         # placement complete → enable PLAY button
                print("After _finish_human_turn in confirmation OKY — phase =", self.gui.game.phase)

        elif choice == "OKN":
            pygame.mixer.Sound.play(SOUNDS["BUTTON"])
            if self.state == PlacementState.SETTLEMENT_SELECTED:
                print("OKN clicked -> going back to settlement selection")
                self.state = PlacementState.CHOOSING_SETTLEMENT
                self.selected_inter_id = None
                self.confirm_center = None
                self.gui.draw_board_base(self.gui.game.board)
                self.gui.draw_all_permanent_buildings(self.gui.game.board)
                self._highlight_available("settlement")
                self.gui.draw_guidance_text("Click a highlighted intersection for your settlement", y_offset=25)

            elif self.state == PlacementState.ROAD_SELECTED:
                self.state = PlacementState.CHOOSING_ROAD
                self.selected_road = None
                self.confirm_center = None
                self.gui.draw_board_base(self.gui.game.board)
                self.gui.draw_all_permanent_buildings(self.gui.game.board)
                self._highlight_available("road")
                self.gui.draw_guidance_text("Click a highlighted road connected to your settlement",y_offset=25)                

        pygame.display.update()

          