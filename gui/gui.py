"""
Handles general GUI functionality for the Catan game.
Now includes HumanGuidance for settlement/road placement and confirmation.
"""
import pygame
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from core import board
from gui.gui_constants import WIN, COLORS, Font, IMAGES, POSITIONS, BOARD_OFFSET
from gui.gui_guidance import HumanGuidance
from core.board import Board
from core.game import Game, Player
from core.constants import FNFREQ, FILENAME_FREQ, MG, FILENAME_MG, SAVE_PATH, ResourceCard

def convert_tile(tile_id: int) -> Optional[Tuple[int, int]]:
    """Convert a tile ID to its GUI midpoint coordinates."""
    coords = POSITIONS["tiles"].get(tile_id)
    if coords is None and MG:
        with open(FILENAME_MG, "a") as f:
            f.write(f"gui.py | convert_tile | Missing coordinates for tile ID: {tile_id}\n")
    return tuple(coords) if coords else None

class Button:
    """Represents a button with name, display state, and switch status."""
    def __init__(self, name: str, display_tf: bool):
        self.name = name
        self.display_tf = display_tf
        self.switched_tf = False

class GUI:
    """Manages button states, modes, board rendering, and human guidance."""
    def __init__(self, round_number: int, turn: int, game: 'Game'):  # Add game parameter with forward reference
        """Initialize the GUI with game round and turn."""
        if not pygame.font.get_init():
            pygame.font.init()
        Font.initialize_fonts()
        self.game = game  # Set the attribute here
        self.round = round_number
        self.turn = turn
        self.buttons: List[Button] = []
        self.modes: List[any] = []
        # Persistent queue for continuous subtle highlight of last placement
        self.animate_queue_elements: List[Tuple[Tuple[int,int], Tuple[int,int,int], int, str]] = []

        # Human guidance system
        self.human_guidance = HumanGuidance(self)

        # Pre-register all buttons so they exist from the start
        for name in ["next_turn2", "roll_dices", "end_turn", "buy_city", "buy_settlement",
                     "buy_road", "buy_dcard", "twp", "twb", "text_buy", "text_trade", "cancel"]:
            self.set_button(name, False)

    def print_queues(self) -> None:
        """Log the contents of animation queues for debugging."""
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui.py | print_queues | Elements: {self.animate_queue_elements}\n")

    def _animate_elements(self, board: Board) -> None:
        """Unified animation for settlements, cities, roads and tiles using quarter-circle reveal."""
        if not self.animate_queue_elements:
            return

        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write(f"gui.py | _animate_elements | Queue size: {len(self.animate_queue_elements)}\n")

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui.py | _animate_elements | Animating {len(self.animate_queue_elements)} elements\n")

        for step in range(4):
            quadrants = [
                (True,  True,  True,  False),   # 0: top-right
                (True,  False, True,  True),    # 1: top-left
                (False, True,  True,  True),    # 2: bottom-right
                (True,  True,  False, True),    # 3: bottom-left
            ]

            draw_tr, draw_tl, draw_br, draw_bl = quadrants[step]

            for center, color, diameter, kind in self.animate_queue_elements:
                # For tiles we usually use width=3, others width=2 → you can parameterize later if needed
                width = 3 if kind == "tile" else 2

                # Clear previous animation frame (use background-appropriate color)
                clear_color = COLORS["WHITE"] if color == COLORS["BLUE"] else COLORS["BLUE"]
                pygame.draw.circle(WIN, clear_color, center, diameter, width,
                                draw_top_right=True, draw_top_left=True,
                                draw_bottom_right=True, draw_bottom_left=True)

                # Draw current quadrant
                pygame.draw.circle(WIN, color, center, diameter, width,
                                draw_top_right=draw_tr, draw_top_left=draw_tl,
                                draw_bottom_right=draw_br, draw_bottom_left=draw_bl)

    def animate_continuous(self):
        """Very conservative continuous animation.
        - Disabled completely during InitialPlacement
        - Only animates newest-looking items
        - Auto-clears queue when suspicious
        """
        if not self.animate_queue_elements:
            return

        # Disable pulsing entirely during setup phase (cleanest look)
        # if self.game.phase == "InitialPlacement":
        #     return

        # Quick check: does queue look like it contains current game objects?
        has_valid = any(
            len(item) >= 4 and item[3] in ("settlement", "city", "road", "tile")
            for item in self.animate_queue_elements
        )

        if not has_valid:
            self.animate_queue_elements.clear()
            print("Cleared animate_queue_elements due to invalid contents")
            return

        quadrants = [
            (True,  True,  True,  False),
            (True,  False, True,  True),
            (False, True,  True,  True),
            (True,  True,  False, True),
        ]

        # Normal smooth quadrant animation
        # step = (pygame.time.get_ticks() // 80) % 4
        for step in range(4):

            draw_tr, draw_tl, draw_br, draw_bl = quadrants[step]

            for center, color, diameter, kind in self.animate_queue_elements:
                # For tiles we usually use width=3, others width=2 → you can parameterize later if needed
                width = 3 if kind == "tile" else 2

                # Clear previous animation frame (use background-appropriate color)
                clear_color = COLORS["WHITE"] if color == COLORS["BLUE"] else COLORS["BLUE"]
                pygame.draw.circle(WIN, clear_color, center, diameter, width,
                                draw_top_right=True, draw_top_left=True,
                                draw_bottom_right=True, draw_bottom_left=True)

                # Draw current quadrant
                pygame.draw.circle(WIN, color, center, diameter, width,
                                draw_top_right=draw_tr, draw_top_left=draw_tl,
                                draw_bottom_right=draw_br, draw_bottom_left=draw_bl)

            pygame.display.flip() 
            pygame.time.delay(100)

    def set_button(self, name: str, display_tf: bool) -> None:
        """Set button state. Creates the button if it does not exist yet."""

        if FNFREQ=="Y":
            f= open(FILENAME_Freq,"a")
            f.write("gui.py | set_button"+"\n")
            f.close()

        found=False
        for button in self.buttons:
            if button.name == name:
                if button.display_tf==display_tf:
                    button.switched_tf=False
                else:
                    button.switched_tf=True
                button.display_tf=display_tf
                found=True
        if found == False:
            self.buttons.append(Button(name, display_tf))

    def check_button(self, name: str) -> bool:
        """Check if a button is set to display."""
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("gui.py | check_button\n")
        for button in self.buttons:
            if button.name == name:
                return button.display_tf
        return False

    def check_mode(self, name: str) -> bool:
        """Check if a mode is active (stub implementation)."""
        return False

    def set_mode_duo(self, mode1: str, mode2: str, source: str) -> None:
        """Set two modes simultaneously (stub implementation)."""
        pass

    def display_fresh_board(self, board: Board, scoreboard_tf: bool = False) -> None:
        """Render the initial empty board and optionally the scoreboard."""
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write("gui.py | display_fresh_board\n")
        WIN.fill(COLORS["LGRAY"], (180 + BOARD_OFFSET[0], 25, 480, 475))
        self._draw_hexagon_lines(board)
        self._draw_tiles(board)
        self._draw_tile_values(board)
        self._draw_ports(board)
        self._draw_intersections(board)
        if scoreboard_tf:
            self.display_scoreboard()
        pygame.display.update()
        self.draw_guidance()

    def _draw_hexagon_lines(self, board: Board) -> None:
        """Draw lines connecting intersections to form hexagons."""
        if not pygame.display.get_init():
            return
        for road in board.roads:
            if road and road.id:
                start_id, end_id = road.id
                start_pos = POSITIONS["intersections"].get(start_id, None)
                end_pos = POSITIONS["intersections"].get(end_id, None)
                if start_pos is None or end_pos is None:
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            if start_pos is None:
                                f.write(f"gui.py | _draw_hexagon_lines | No coordinates for intersection ID: {start_id}\n")
                            if end_pos is None:
                                f.write(f"gui.py | _draw_hexagon_lines | No coordinates for intersection ID: {end_id}\n")
                    continue
                pygame.draw.line(WIN, COLORS["BLACK"], start_pos, end_pos, 2)

    def _draw_tiles(self, board: Board) -> None:
        """Draw hexagonal tiles on the board."""
        rendered_tiles = []
        for tile in board.tiles:
            if tile and tile.id in POSITIONS["tiles"]:
                pos = convert_tile(tile.id)
                if pos is None:
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            f.write(f"gui.py | _draw_tiles | No coordinates for tile ID: {tile.id}\n")
                    continue
                image_key = {
                    "Field": "FIELD",
                    "Mountain": "MOUNTAIN",
                    "Forest": "FOREST",
                    "Hill": "HILL",
                    "Pasture": "PASTURE",
                    "Desert": "DESERT",
                    "Sea": "SEA"
                }.get(tile.type, "SEA")
                image = IMAGES[image_key]["40x40"] if image_key in ["FIELD", "MOUNTAIN", "FOREST", "HILL", "PASTURE"] else IMAGES[image_key]["default"]
                if image is not None:
                    WIN.blit(image, (pos[0] - 20, pos[1] - 20))
                    rendered_tiles.append(tile.id)
                else:
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            f.write(f"gui.py | _draw_tiles | Failed to render tile ID: {tile.id}, Type: {tile.type}, Pos: {pos}\n")
            else:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_tiles | Skipped tile ID: {tile.id if tile else None}, Pos: {convert_tile(tile.id) if tile else None}\n")
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui.py | _draw_tiles | Rendered tile IDs: {rendered_tiles}\n")

    def _draw_tile_values(self, board: Board) -> None:
        """Draw number chits on tiles."""
        font = Font.LARGE.value["regular"]
        for tile in board.tiles:
            if tile and tile.id in POSITIONS["tiles"] and tile.value != 0:
                pos = convert_tile(tile.id)
                if pos is None:
                    continue
                color = COLORS["RED"] if tile.value in [6, 8] else COLORS["BLACK"]
                text = font.render(str(tile.value), True, color)
                WIN.blit(text, (pos[0] - 8, pos[1] + 15))

    def _draw_ports(self, board: Board) -> None:
        """Draw port icons, circles, and lines on the board."""
        font = Font.SMALL.value["regular"]
        port_intersection_ids = set()
        for port_pair in board.INTERSECTIONS_ARE_PORT:
            port_intersection_ids.update(port_pair)
        for intersection_id in port_intersection_ids:
            if intersection_id in board.INTERSECTION_IN_WATER or board.intersections[intersection_id] is None:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_ports | Skipping None or water intersection ID: {intersection_id}\n")
                continue
            pos = POSITIONS["intersections"].get(intersection_id, None)
            if pos is None:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_ports | No coordinates for intersection ID: {intersection_id}\n")
                continue
            pygame.draw.circle(WIN, COLORS["BLACK"], pos, 5, 0)
        for port_pair in board.INTERSECTIONS_ARE_PORT:
            first_intersection_id = port_pair[0]
            second_intersection_id = port_pair[1]
            if (first_intersection_id in board.INTERSECTION_IN_WATER or 
                second_intersection_id in board.INTERSECTION_IN_WATER or
                board.intersections[first_intersection_id] is None or
                board.intersections[second_intersection_id] is None):
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_ports | Skipping port pair {port_pair} due to None or water intersections\n")
                continue
            first_intersection = next((i for i in board.intersections if i is not None and i.id == first_intersection_id), None)
            if not first_intersection:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_ports | No valid intersection found for ID: {first_intersection_id}\n")
                continue
            sea_tile_id = None
            for tile in board.tiles:
                if tile and tile.type == "Sea":
                    corner_intersections = [corner.intersection for corner in tile.corners]
                    if first_intersection_id in corner_intersections and second_intersection_id in corner_intersections:
                        sea_tile_id = tile.id
                        break
            if sea_tile_id is None:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_ports | No sea tile found for port pair: {port_pair}\n")
                continue
            pos = convert_tile(sea_tile_id)
            if pos is None:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui.py | _draw_ports | No coordinates for sea tile id: {sea_tile_id}\n")
                continue
            first_intersection_pos = POSITIONS["intersections"].get(first_intersection_id, None)
            second_intersection_pos = POSITIONS["intersections"].get(second_intersection_id, None)
            if first_intersection_pos:
                pygame.draw.line(WIN, COLORS["BLACK"], first_intersection_pos, pos, 2)
            if second_intersection_pos:
                pygame.draw.line(WIN, COLORS["BLACK"], second_intersection_pos, pos, 2)
            if first_intersection.port_type == "Blank":
                pygame.draw.rect(WIN, COLORS["WHITE"], [pos[0] - 10, pos[1] - 10, 20, 20])
                text = font.render(" ?", True, COLORS["BLACK"])
                WIN.blit(text, (pos[0] - 7, pos[1] - 8))
            elif first_intersection.port_type == "3:1":
                pygame.draw.rect(WIN, COLORS["WHITE"], [pos[0] - 10, pos[1] - 10, 20, 20])
                text = font.render("3:1", True, COLORS["BLACK"])
                WIN.blit(text, (pos[0] - 7, pos[1] - 8))
            else:
                image_key = {
                    "2:1 Wheat": "FIELD",
                    "2:1 Ore": "MOUNTAIN",
                    "2:1 Wood": "FOREST",
                    "2:1 Brick": "HILL",
                    "2:1 Wool": "PASTURE"
                }.get(first_intersection.port_type)
                if image_key:
                    image = IMAGES[image_key]["20x20"]
                    if image is not None:
                        WIN.blit(image, (pos[0] - 10, pos[1] - 10))
                    else:
                        if MG:
                            with open(FILENAME_MG, "a") as f:
                                f.write(f"gui.py | _draw_ports | Missing image for port type: {first_intersection.port_type}, Tile ID: {sea_tile_id}\n")

    def _draw_intersections(self, board: Board) -> None:
        """Draw intersections (vertices) on the board with bold IDs."""
        font = Font.SMALL.value["bold"]
        offset_minus_3 = {4, 6, 8}
        offset_minus_6 = {59, 61, 63}
        for intersection in board.intersections:
            if intersection is None or intersection.id in board.INTERSECTION_IN_WATER:
                continue
            if intersection.id in POSITIONS["intersections"]:
                pos = POSITIONS["intersections"][intersection.id]
                text = font.render(str(intersection.id), True, COLORS["DGRAY"])
                y_offset = -16 if intersection.id in {14, 16, 18, 20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 58, 60, 62, 64} else 2
                x_offset = -3 if intersection.id in offset_minus_3 else -6 if intersection.id in offset_minus_6 else 4
                WIN.blit(text, (pos[0] + x_offset, pos[1] + y_offset))

    def display_scoreboard(self) -> None:
        """Display the empty scoreboard (placeholder)."""
        pass

    def _occupy_settlement_in_gui(self, board: Board, intersection_id: int, color: str) -> None:
        pos = POSITIONS["intersections"].get(intersection_id)
        if not pos: return
        image = IMAGES.get(f"SETTLEMENT_{color.upper()}", {}).get("30x30")
        if image:
            WIN.blit(image, (pos[0] - 15, pos[1] - 15))

    def _occupy_road_in_gui(self, board: Board, road_id: Tuple[int, int], color: str) -> None:
        pos1 = POSITIONS["intersections"].get(road_id[0])
        pos2 = POSITIONS["intersections"].get(road_id[1])
        if pos1 and pos2:
            pygame.draw.line(WIN, COLORS[color.upper()], pos1, pos2, 5)

    def _occupy_city_in_gui(self, board: Board, intersection_id: int, color: str) -> None:
        pos = POSITIONS["intersections"].get(intersection_id)
        if not pos: return
        image = IMAGES.get(f"CITY_{color.upper()}", {}).get("30x30")
        if image:
            WIN.blit(image, (pos[0] - 15, pos[1] - 15))

    def draw_board_base(self, board: Board) -> None:
        """Static empty board only (tiles, lines, numbers, ports, intersection IDs)."""
        WIN.fill(COLORS["LGRAY"], (180 + BOARD_OFFSET[0], 25, 480, 475))
        self._draw_hexagon_lines(board)
        self._draw_tiles(board)
        self._draw_tile_values(board)
        self._draw_ports(board)
        self._draw_intersections(board)

    def draw_all_permanent_buildings(self, board: Board, block_visual: bool = False):
        """Draw EVERY currently placed road/settlement/city + blocked dots."""
        # Roads
        for road in board.roads:
            if road and road.occupied_tf:
                self._occupy_road_in_gui(board, road.id, road.color)  # simplified version below

        # Settlements & Cities
        for inter in board.intersections:
            if inter and inter.occupied_tf:
                if inter.face == "Settlement":
                    self._occupy_settlement_in_gui(board, inter.id, inter.color)
                elif inter.face == "City":
                    self._occupy_city_in_gui(board, inter.id, inter.color)

        # Blocked dots (adjacent to any settlement)
        for inter in board.intersections:
                if inter and inter.occupied_tf:
                    self._block_adjacent_in_gui(board, inter.id, block_visual=block_visual)

    def save_screenshot(self) -> None:
        """Save a screenshot of the game window to SAVE_PATH with a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_PATH, f"Catan_Screenshot_{timestamp}.png")
        Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)
        try:
            pygame.image.save(WIN, filename)
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui.py | save_screenshot | Saved to {filename}\n")
        except pygame.error as e:
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui.py | save_screenshot | Error saving screenshot to {filename}: {e}\n")

    def update_round_turn(self, game: Game, special: bool) -> None:
        """
        Update the round / turn display.

        During Execution, also show a compact fast-forward status block.
        """
        self.round = game.round
        self.turn = game.turn

        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a", encoding="utf-8") as f:
                f.write("gui.py | update_round_turn\n")

        pygame.draw.rect(WIN, COLORS["LGRAY"], [2, 2, 420, 95])

        help_round = game.round
        help_turn = game.turn

        color_name = {1: "BLUE", 2: "RED", 3: "WHITE", 4: "ORANGE"}.get(help_turn, "BLACK")
        font = Font.LARGE.value["regular"]

        turn_text = font.render(f"Turn: {help_turn}", True, COLORS[color_name])
        round_text = font.render(f"Round: {help_round}", True, COLORS[color_name])

        WIN.blit(turn_text, (165, 5))
        WIN.blit(round_text, (15, 5))

        if game.phase == "Execution":
            self.draw_fast_forward_status(game)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"gui.py | update_round_turn | Actual: {self.round}, {self.turn} | "
                    f"Display: {help_round}, {help_turn}, Color: {color_name}, "
                    f"Phase: {game.phase}, Special: {special}\n"
                )

    def _block_adjacent_in_gui(self, board: Board, intersection_id: int, block_visual: bool = False) -> None:
        """
            Optionally render visual indication of blocked (forbidden) adjacent intersections.

            This method does nothing unless `block_visual=True` is explicitly passed.
            It is meant to highlight intersections that cannot be built on due to adjacency rules.

            Args:
                board: The current game board instance.
                intersection_id: ID of the occupied intersection whose neighbors should be checked.
                block_visual: If True, draw blocking indicators on adjacent valid intersections.
                            Defaults to False (no visual change).
            """
        if not block_visual:
            return  # do nothing by default
        
        # existing blocking/highlighting logic
        intersection = board.intersections[intersection_id]
        if intersection:
            for neighbor_id in intersection.three_intersection_ids:
                if (neighbor_id not in board.INTERSECTION_IN_WATER and
                    board.intersections[neighbor_id] is not None and
                    board.intersections[neighbor_id].can_build_tf):
                    pos = POSITIONS["intersections"].get(neighbor_id)
                    if pos:
                        pygame.draw.circle(WIN, COLORS["BLACK"], pos, 10)

    def queue_latest_placement(self) -> None:
        """
        Populate self.animate_queue_elements with the most recent placement
        using placement_step (works for both AI and human, no more round/turn bugs).
        """
        temp_queue = []
        max_step = -1

        # Find the highest placement_step that was used
        for inter in self.game.board.intersections:
            if inter and inter.occupied_tf and inter.face in ("Settlement", "City"):
                max_step = max(max_step, inter.placement_step)
        for road in self.game.board.roads:
            if road and road.occupied_tf:
                max_step = max(max_step, road.placement_step)

        if max_step == -1:
            print("No placement found to add to animation queue")
            self.animate_queue_elements = []
            return

        # ── Latest settlement/city with this step ─────────────────────────────
        latest_inter = None
        for inter in self.game.board.intersections:
            if (inter and inter.occupied_tf and inter.face in ("Settlement", "City") and
                inter.placement_step == max_step):
                latest_inter = inter
                break

        if latest_inter:
            pos = POSITIONS["intersections"].get(latest_inter.id)
            if pos:
                kind = "settlement" if latest_inter.face == "Settlement" else "city"
                color = COLORS[latest_inter.color.upper()]
                temp_queue.append((pos, color, 20, kind))

                # Second settlement -> highlight resource tiles (setup phase)
                if self.game.round == -1:
                    for tile_id in latest_inter.three_tile_ids:
                        tile = self.game.board.tiles[tile_id]
                        if tile and tile.type not in ("Sea", "Desert"):
                            tile_pos = convert_tile(tile_id)
                            if tile_pos:
                                temp_queue.append((tile_pos, (255, 255, 0), 26, "tile"))

        # ── Latest road with this step ───────────────────────────────────────
        latest_road = None
        for road in self.game.board.roads:
            if road and road.occupied_tf and road.placement_step == max_step:
                latest_road = road
                break

        if latest_road:
            pos1 = POSITIONS["intersections"].get(latest_road.id[0])
            pos2 = POSITIONS["intersections"].get(latest_road.id[1])
            if pos1 and pos2:
                mid = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                color = COLORS[latest_road.color.upper()]
                temp_queue.append((mid, color, 20, "road"))

        print(f"Queueing {len(temp_queue)} items for animation queue (step {max_step})")
        for item in temp_queue:
            print(f"  -> {item}")

        self.animate_queue_elements = temp_queue

    def update_board(self, board: Board, update_type: str) -> None:
        """
        Update board rendering.

        Supported update types:
            - "All"              : full redraw
            - "Last"             : initial-placement style latest placement animation
            - "FastForwardJump"  : redraw board + scoreboard + FF status, no action reveal
            - "FastForwardLast"  : redraw + animate last executed fast-forward action
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a", encoding="utf-8") as f:
                f.write(f"gui.py | update_board | type={update_type}\n")

        # ============================================================
        # Full redraw
        # ============================================================
        if update_type == "All":
            self.display_fresh_board(board, scoreboard_tf=True)

            for intersection in board.intersections:
                if intersection and intersection.occupied_tf:
                    self._block_adjacent_in_gui(board, intersection.id)

            for road in board.roads:
                if road and road.occupied_tf:
                    self._occupy_road_in_gui(board, road.id, road.color)

            for intersection in board.intersections:
                if intersection and intersection.occupied_tf:
                    if intersection.face == "Settlement":
                        self._occupy_settlement_in_gui(board, intersection.id, intersection.color)
                    elif intersection.face == "City":
                        self._occupy_city_in_gui(board, intersection.id, intersection.color)

            if self.game.phase == "Execution":
                self.draw_fast_forward_status(self.game)

            pygame.display.update()
            return

        # ============================================================
        # Execution: JUMP only updates board view + status, no reveal yet
        # ============================================================
        if update_type == "FastForwardJump":
            self.animate_queue_elements.clear()
            self._redraw_execution_view(board)
            pygame.display.update()

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write("gui.py | update_board | FastForwardJump\n")
            return

        # ============================================================
        # Execution: PLAY reveals last fast-forward action
        # ============================================================
        if update_type == "FastForwardLast":
            # Redraw permanent board state first.
            # This makes sure cities/settlements/roads that were already executed
            # are visible even before the pulse animation starts.
            self._redraw_execution_view(board)

            # Then queue every action from the latest PLAY, including same-turn chain.
            self.queue_fast_forward_action()

            if self.animate_queue_elements:
                self._animate_elements(board)

                # Redraw once more after the pulse so the final permanent state is crisp.
                self._redraw_execution_view(board)

            pygame.display.update()

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"gui.py | update_board | FastForwardLast | "
                        f"queue={self.animate_queue_elements}\n"
                    )
            return

        # ============================================================
        # Default "Last" branch = initial placement latest-placement animation
        # ============================================================
        self.draw_board_base(board)
        self.draw_all_permanent_buildings(board)

        max_step = -1
        for inter in board.intersections:
            if inter and inter.occupied_tf and inter.face in ("Settlement", "City"):
                max_step = max(max_step, inter.placement_step)
        for road in board.roads:
            if road and road.occupied_tf:
                max_step = max(max_step, road.placement_step)

        temp_queue = []

        if max_step == -1:
            print("No placement found to animate this turn (update_board 'Last')")
        else:
            print(f"Animating items for latest placement (step {max_step})")

            latest_inter = None
            for inter in board.intersections:
                if (
                    inter is not None
                    and inter.occupied_tf
                    and inter.face in ("Settlement", "City")
                    and inter.placement_step == max_step
                ):
                    latest_inter = inter
                    break

            if latest_inter:
                pos = POSITIONS["intersections"].get(latest_inter.id)
                if pos:
                    kind = "settlement" if latest_inter.face == "Settlement" else "city"
                    color = COLORS[latest_inter.color.upper()]
                    temp_queue.append((pos, color, 20, kind))

                    if latest_inter.face == "Settlement":
                        self._occupy_settlement_in_gui(board, latest_inter.id, latest_inter.color)
                    else:
                        self._occupy_city_in_gui(board, latest_inter.id, latest_inter.color)

                    if self.round == -1:
                        for tile_id in latest_inter.three_tile_ids:
                            tile = board.tiles[tile_id]
                            if tile and tile.type not in ("Sea", "Desert"):
                                tile_pos = convert_tile(tile_id)
                                if tile_pos:
                                    temp_queue.append((tile_pos, (255, 255, 0), 26, "tile"))

            latest_road = None
            for road in board.roads:
                if road is not None and road.occupied_tf and road.placement_step == max_step:
                    latest_road = road
                    break

            if latest_road:
                pos1 = POSITIONS["intersections"].get(latest_road.id[0])
                pos2 = POSITIONS["intersections"].get(latest_road.id[1])
                if pos1 and pos2:
                    mid = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                    color = COLORS[latest_road.color.upper()]
                    temp_queue.append((mid, color, 20, "road"))
                    self._occupy_road_in_gui(board, latest_road.id, latest_road.color)

        self.animate_queue_elements = temp_queue

        if temp_queue:
            self._animate_elements(board)

        if self.game.phase == "Execution":
            self.draw_fast_forward_status(self.game)

        pygame.display.update()

    def draw_fast_forward_player_debug(self, game: Game) -> None:
        """
        TEMP v014 debug overlay.

        Shows the latest fast-forward prediction contract per player:
            R/T + strategy

        Drawn just to the right of the scoreboard.

        Note:
        This older overlay is currently not the one called by update_scoreboard()
        if you are using draw_fast_forward_expected_actions_debug(...).
        Keeping it correct anyway.
        """
        if getattr(game, "phase", None) != "Execution":
            return

        rows = list(getattr(game, "ff_debug_prediction_rows", []) or [])

        # Area just right of the dev-card columns.
        panel_x = 820
        panel_y = 570
        panel_w = 360
        panel_h = 170

        # Move only the header 30 px up; player rows stay at the old positions.
        header_y = panel_y - 30

        # Clear from moved header through the player rows.
        pygame.draw.rect(WIN, COLORS["LGRAY"], [panel_x, header_y, panel_w, panel_h + 30])

        font_small = Font.SMALL.value["regular"]
        font_normal = Font.NORMAL.value["regular"]

        title = font_normal.render("FF prediction", True, COLORS["BLACK"])
        WIN.blit(title, (panel_x, header_y))

        rows_by_player = {}
        for row in rows:
            try:
                pid = int(row.get("player_id"))
            except Exception:
                continue
            rows_by_player[pid] = row

        def _short(value, max_len: int = 18) -> str:
            txt = str(value)
            if len(txt) <= max_len:
                return txt
            return txt[: max_len - 3] + "..."

        for idx, player in enumerate(game.players):
            # Keep row positions unchanged.
            row_y = 590 + idx * 40
            color = COLORS.get(player.color.upper(), COLORS["BLACK"])

            row = rows_by_player.get(int(player.id))

            player_txt = font_normal.render(f"P{player.id}", True, color)
            WIN.blit(player_txt, (panel_x, row_y))

            if not row:
                info = "-"
            else:
                pred_round = row.get("pred_round", "?")
                pred_turn = row.get("pred_turn", "?")
                strategy = row.get("strategy", "?")
                source = row.get("source_mode", row.get("mode", "light"))
                heavy = bool(row.get("used_heavy", False))

                src_txt = "H" if heavy or str(source) == "heavy" else "L"

                info = (
                    f"R{pred_round}/T{pred_turn} "
                    f"{_short(strategy, 18)} "
                    f"[{src_txt}]"
                )

            info_txt = font_small.render(info, True, COLORS["BLACK"])
            WIN.blit(info_txt, (panel_x + 45, row_y + 3))

    def draw_fast_forward_status(self, game: Game) -> None:
        """
        Draw a compact fast-forward status block used during Execution.

        v014:
        - Shows current FF mode/status.
        - Shows a visible busy message when JUMP/PLAY is processing.
        """
        panel_x = 10
        panel_y = 45
        panel_w = 390
        panel_h = 72

        pygame.draw.rect(WIN, COLORS["LGRAY"], [panel_x, panel_y, panel_w, panel_h])

        font_small = Font.SMALL.value["regular"]
        font_normal = Font.NORMAL.value["regular"]

        mode = str(getattr(game, "ff_button_mode", "JUMP")).upper()
        ff_step = getattr(game, "ff_step_index", 0)
        ff_last_delta = getattr(game, "ff_last_delta", 0.0)
        ff_elapsed = getattr(game, "ff_elapsed_rolls", 0.0)

        current_player = game.current_player
        current_color = COLORS["BLACK"]
        current_label = "None"

        if current_player is not None:
            current_label = f"P{current_player.id} {current_player.color}"
            current_color = COLORS.get(current_player.color.upper(), COLORS["BLACK"])

        pending = getattr(game, "ff_pending_event", None)
        if pending:
            pending_txt = (
                f"Pending: {pending.get('player_color', '?')} "
                f"{pending.get('requested_activity', '?')}"
            )
        else:
            actual = getattr(game, "ff_last_actual_activity", None)
            pending_txt = f"Last: {actual}" if actual else "Pending: None"

        line1 = font_normal.render(
            f"Mode: {mode}   FF Step: {ff_step}   Δ rolls: {ff_last_delta:.2f}   Σ rolls: {ff_elapsed:.2f}",
            True,
            COLORS["BLACK"],
        )

        line2a = font_normal.render("Current: ", True, COLORS["BLACK"])
        line2b = font_normal.render(current_label, True, current_color)
        line2c = font_small.render(pending_txt, True, COLORS["DGRAY"])

        WIN.blit(line1, (panel_x + 5, panel_y + 2))
        WIN.blit(line2a, (panel_x + 5, panel_y + 24))
        WIN.blit(line2b, (panel_x + 65, panel_y + 24))
        WIN.blit(line2c, (panel_x + 185, panel_y + 27))

        if getattr(game, "ff_processing", False):
            busy_txt = getattr(game, "ff_processing_text", "AI is thinking...")
            busy_line = font_normal.render(busy_txt, True, COLORS["RED"])
            WIN.blit(busy_line, (panel_x + 5, panel_y + 48))

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"gui.py | draw_fast_forward_status | mode={mode} "
                    f"step={ff_step} delta={ff_last_delta:.2f} total={ff_elapsed:.2f} "
                    f"processing={getattr(game, 'ff_processing', False)}\n"
                )

    def draw_fast_forward_expected_actions_debug(self, game: Game) -> None:
        """
        TEMP v014 debug overlay.

        Draws the latest fast-forward prediction row per player just right
        of the scoreboard.

        Shows:
            - predicted round / turn
            - chosen strategy
            - expected viable options, e.g. D,C,S
            - compact expected action scores, e.g. D1.1 C2.0 S7.0

        This is display-only. It does not affect prediction or execution.
        """
        if getattr(game, "phase", None) != "Execution":
            return

        rows = list(getattr(game, "ff_debug_prediction_rows", []) or [])

        # Same area we used before, but 15 px higher.
        panel_x = 820
        panel_y = 570
        panel_w = 360
        panel_h = 180

        pygame.draw.rect(WIN, COLORS["LGRAY"], [panel_x, panel_y, panel_w, panel_h])

        font_small = Font.SMALL.value["regular"]
        font_normal = Font.NORMAL.value["regular"]

        def _short(value, max_len: int) -> str:
            txt = str(value)
            if len(txt) <= max_len:
                return txt
            return txt[: max(0, max_len - 3)] + "..."

        def _safe_float(value, default: float = 9999.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _format_expected_scores(row: dict) -> str:
            actions = list(row.get("expected_viable_actions", []) or [])
            parts = []

            for action in actions[:4]:
                code = str(action.get("code", "?"))
                score = _safe_float(action.get("score", 9999.0))

                if score >= 9999.0:
                    score_txt = "∞"
                else:
                    score_txt = f"{score:.1f}"

                parts.append(f"{code}{score_txt}")

            return " ".join(parts) if parts else "-"

        rows_by_player = {}
        for row in rows:
            try:
                pid = int(row.get("player_id"))
            except Exception:
                continue
            rows_by_player[pid] = row

        title = font_normal.render("FF expected actions", True, COLORS["BLACK"])
        WIN.blit(title, (panel_x, panel_y))

        subtitle = font_small.render("R/T  strategy  [opts]  scores", True, COLORS["DGRAY"])
        WIN.blit(subtitle, (panel_x, panel_y + 18))

        for idx, player in enumerate(game.players):
            row_y = 590 + idx * 40

            player_color = COLORS.get(player.color.upper(), COLORS["BLACK"])
            player_txt = font_normal.render(f"P{player.id}", True, player_color)
            WIN.blit(player_txt, (panel_x, row_y))

            row = rows_by_player.get(int(player.id))

            if not row:
                info = "-"
                scores = "-"
            else:
                pred_round = row.get("pred_round", "?")
                pred_turn = row.get("pred_turn", "?")
                strategy = row.get("strategy", "?")
                opts = row.get("expected_viable_codes", "-")
                scores = _format_expected_scores(row)

                info = (
                    f"R{pred_round}/T{pred_turn} "
                    f"{_short(strategy, 15)} "
                    f"[{_short(opts, 7)}]"
                )

            info_txt = font_small.render(info, True, COLORS["BLACK"])
            WIN.blit(info_txt, (panel_x + 45, row_y + 1))

            score_txt = font_small.render(_short(scores, 30), True, COLORS["DGRAY"])
            WIN.blit(score_txt, (panel_x + 45, row_y + 18))

    def set_ai_busy_indicator(self, active: bool, text: str = "AI is thinking...") -> None:
        """
        Show or clear a small busy indicator above the JUMP/PLAY button.
        """
        self.ai_busy_indicator_active = bool(active)
        self.ai_busy_indicator_text = str(text or "AI is thinking...")

        self.draw_ai_busy_indicator(force_visible=True)

        try:
            pygame.display.update()
            pygame.event.pump()
            pygame.time.wait(60)
        except Exception:
            pass

    def draw_ai_busy_indicator(self, force_visible: bool = False) -> None:
        """
        Draw busy text above the JUMP/PLAY button.

        During blocking Markov work, this will usually appear as a static message,
        not a true blink, because the pygame event loop is blocked.
        """
        # Main JUMP/PLAY button rectangle is [20, 470, 130, 40].
        # Do not import PLAY_RECT here; not all versions define it in gui_constants.
        play_rect = pygame.Rect(20, 470, 130, 40)

        rect = pygame.Rect(
            play_rect.x,
            play_rect.y - 28,
            play_rect.width + 210,
            26,
        )

        pygame.draw.rect(WIN, COLORS["LGRAY"], rect)

        if not getattr(self, "ai_busy_indicator_active", False):
            return

        visible = force_visible or ((pygame.time.get_ticks() // 450) % 2 == 0)
        if not visible:
            return

        font = Font.SMALL.value["regular"]
        text = font.render(
            getattr(self, "ai_busy_indicator_text", "AI is thinking..."),
            True,
            COLORS["RED"],
        )
        WIN.blit(text, (rect.x + 4, rect.y + 5))

    def _scoreboard_total_vp(self, player) -> int:
        """
        Display-safe VP calculation.

        Ensures initial-placement settlements count as VP even if the player's
        stored victory_points/points field has not been refreshed yet.
        """
        settlement_vp = len(getattr(player, "settlements", []) or [])
        city_vp = 2 * len(getattr(player, "cities", []) or [])

        board_vp = settlement_vp + city_vp

        stored_vp = 0
        for attr in ("victory_points", "points", "vp"):
            try:
                stored_vp = max(stored_vp, int(getattr(player, attr, 0) or 0))
            except Exception:
                pass

        return max(board_vp, stored_vp)

    def update_scoreboard(self, game: Game) -> None:
        """Render the entire scoreboard with headers and player statistics.

        Args:
            game: The game instance containing player data.
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write(f"{game.sequence_number} | {game.state} | gui_game.py | update_scoreboard\n")

        # Resource_exploration (=pips summary
        self.update_resource_exploration(game.board)

        # Clear full scoreboard area, including headers, resource cards, and dev-card columns.
        pygame.draw.rect(WIN, COLORS["LGRAY"], [110, 540, 705, 240])        
       
        # Header: Small "VP" above C, S, R (Longest Route), A, E
        font_small = Font.SMALL.value["regular"]
        vp_columns = [1, 2, 4, 5, 6] # Indices for C, S, R (Longest Route), A, E
        header_x_positions = [115, 145, 165, 185, 205, 225, 245, 270, 300, 330, 360, 390, 435, 480, 525, 570, 635, 670, 705, 740, 775]
        for i in vp_columns:
            vp_header = font_small.render("VP", True, COLORS["BLACK"])
            vp_rect = vp_header.get_rect(center=(header_x_positions[i] + 10, 550)) # Center for 20-pixel column
            WIN.blit(vp_header, vp_rect)
       
        # Header: Main text and images
        font = Font.NORMAL.value["regular"]
        header_parts = ["VP", "C", "S", "R", "R", "A", "E", "LR", "LA", "RC", "DC"]
        for i, part in enumerate(header_parts):
            text = font.render(part, True, COLORS["BLACK"])
            text_rect = text.get_rect(center=(header_x_positions[i] + 10, 560)) # Center for 20-pixel column
            WIN.blit(text, text_rect)
       
        # RC images (Wheat, Ore, Wood, Brick, Wool) at 40x40, 5 pixels apart
        rc_images = ["FIELD", "MOUNTAIN", "FOREST", "HILL", "PASTURE"]
        rc_x_positions = [390, 435, 480, 525, 570] # 40x40 images + 5-pixel gaps
        for i, img_key in enumerate(rc_images):
            try:
                image = IMAGES[img_key]["40x40"]
                if image is not None:
                    img_rect = image.get_rect(center=(rc_x_positions[i] + 20, 560)) # Center for 40x40
                    WIN.blit(image, img_rect)
                else:
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            f.write(f"gui_game.py | update_scoreboard | Missing RC image: {img_key}\n")
            except KeyError:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui_game.py | update_scoreboard | KeyError: No '40x40' for RC image: {img_key}\n")
       
        # Vertical lines before/after RC, before DC
        pygame.draw.line(WIN, COLORS["BLACK"], (385, 540), (385, 780), 2) # Before RC
        pygame.draw.line(WIN, COLORS["BLACK"], (615, 540), (615, 780), 2) # After RC
        pygame.draw.line(WIN, COLORS["BLACK"], (630, 540), (630, 780), 2) # Before DC
       
        # DC images (VP, Knight, Road, Plenty, Monopoly) at 30x30, 5 pixels apart
        dc_images = ["DC_VPOINT", "DC_KNIGHT", "DC_ROAD", "DC_PLENTY", "DC_MONOPOLY"]
        dc_x_positions = [635, 670, 705, 740, 775] # 30x30 images + 5-pixel gaps
        for i, img_key in enumerate(dc_images):
            try:
                image = IMAGES[img_key]["30x30"]
                if image is not None:
                    img_rect = image.get_rect(center=(dc_x_positions[i] + 15, 560)) # Center for 30x30
                    WIN.blit(image, img_rect)
                else:
                    if MG:
                        with open(FILENAME_MG, "a") as f:
                            f.write(f"gui_game.py | update_scoreboard | Missing DC image: {img_key}\n")
            except KeyError:
                if MG:
                    with open(FILENAME_MG, "a") as f:
                        f.write(f"gui_game.py | update_scoreboard | KeyError: No '30x30' for DC image: {img_key}\n")
       
        # Player rows
        for i, player in enumerate(game.players):
            self._render_scoreboard_row(player, game, 15, 560 + (i + 1) * 40 - 10, 560 + (i + 1) * 40)

        # TEMP v014: show expected viable fast-forward actions next to scoreboard.
        if getattr(game, "phase", None) == "Execution":
            self.draw_fast_forward_expected_actions_debug(game)   
       
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_game.py | update_scoreboard | Rendered scoreboard for {len(game.players)} players\n")

    def _render_scoreboard_row(self, player: Player, game: Game, x: int, name_y: int, stats_y: int) -> None:
        """Render a single player's scoreboard row with statistics and resource counts.

        Args:
            player: The player whose statistics to render.
            game: The game instance containing player data.
            x: X-coordinate for the player name.
            name_y: Y-coordinate for the player name.
            stats_y: Y-coordinate for the player's statistics and resource counts.
        """
        font = Font.NORMAL.value["regular"]
        font_large = Font.LARGE.value["regular"]
        x_positions = [115, 145, 165, 185, 205, 225, 245, 270, 300, 330, 360, 390, 435, 480, 525, 570]
       
        # Player name in color, large font, at x=15
        player_colors = {
            1: COLORS["BLUE"],
            2: COLORS["RED"],
            3: COLORS["WHITE"],
            4: COLORS["ORANGE"]
        }
        player_name = f"Player {player.id}"
        name_text = font_large.render(player_name, True, player_colors.get(player.id, COLORS["BLACK"]))
        WIN.blit(name_text, (x, name_y))
       
        # Player stats
        # v014: compute displayed VP from actual board/player state, not only
        # player.victory_points, because that field is not always refreshed during
        # initial placement / fast-forward execution.

        settlement_vp = len(getattr(player, "settlements", []) or [])
        city_vp = 2 * len(getattr(player, "cities", []) or [])
        longest_route_vp = 2 if getattr(player, "longest_route_tf", False) else 0
        largest_army_vp = 2 if getattr(player, "largest_army_tf", False) else 0

        # dcard_summary rows:
        #   0 victory_point
        #   1 knight
        #   2 two_free_roads
        #   3 year_of_plenty
        #   4 monopoly
        #
        # row format:
        #   [card_name, bought_this_turn, not_played, played]
        extra_vp = 0
        try:
            vp_row = getattr(player, "dcard_summary", [])[0]
            if vp_row and len(vp_row) >= 4:
                extra_vp = int(vp_row[1] or 0) + int(vp_row[2] or 0) + int(vp_row[3] or 0)
        except Exception:
            extra_vp = 0

        computed_total_vp = (
            settlement_vp
            + city_vp
            + longest_route_vp
            + largest_army_vp
            + extra_vp
        )

        stored_vp = 0
        for attr in ("victory_points", "points", "vp"):
            try:
                stored_vp = max(stored_vp, int(getattr(player, attr, 0) or 0))
            except Exception:
                pass

        display_vp = max(computed_total_vp, stored_vp)

        stats = [
            str(display_vp), # VP total
            str(len(player.cities)), # C
            str(len(player.settlements)), # S
            str(len(player.roads)), # R (roads)
            str(longest_route_vp), # R (longest route points)
            str(largest_army_vp), # A
            str(extra_vp), # E, hidden VP dev cards
            str(player.size_longest_route), # LR
            str(player.size_largest_army), # LA
            str(player.number_of_rcards), # RC
            str(player.number_of_dcards), # DC
        ]
        for i, stat in enumerate(stats):
            text = font.render(stat, True, COLORS["BLACK"])
            text_rect = text.get_rect(center=(x_positions[i] + 10, stats_y))
            WIN.blit(text, text_rect)
       
        # Resource cards (Wheat, Ore, Wood, Brick, Wool) for all players
        rc_stats = [
            player.rcards.get(ResourceCard.WHEAT, 0),
            player.rcards.get(ResourceCard.ORE, 0),
            player.rcards.get(ResourceCard.WOOD, 0),
            player.rcards.get(ResourceCard.BRICK, 0),
            player.rcards.get(ResourceCard.WOOL, 0)
        ]
        for i, stat in enumerate(rc_stats):
            text = font.render(str(stat), True, COLORS["BLACK"])
            text_rect = text.get_rect(center=(x_positions[i + 11] + 20, stats_y))
            WIN.blit(text, text_rect)

        # Clear this player's dev-card-stat row before redrawing it.
        pygame.draw.rect(WIN, COLORS["LGRAY"], [630, stats_y - 8, 185, 18])

        # Development-card details under the DC icons.
        # Format follows v045:
        #   bought_this_turn / not_played / played
        #
        # dcard_summary rows:
        #   0 victory_point
        #   1 knight
        #   2 two_free_roads
        #   3 year_of_plenty
        #   4 monopoly
        dc_x_positions = [635, 670, 705, 740, 775]

        for i, row in enumerate(getattr(player, "dcard_summary", [])[:5]):
            if not row or len(row) < 4:
                continue

            bought_this_turn = int(row[1] or 0)
            not_played = int(row[2] or 0)
            played = int(row[3] or 0)

            # Only show non-empty dev-card rows, like v045.
            if bought_this_turn == 0 and not_played == 0 and played == 0:
                continue

            txt = f"{bought_this_turn}/{not_played}/{played}"
            text = Font.SMALL.value["regular"].render(txt, True, COLORS["BLACK"])

            # v045 used approximately:
            # VP 640, Knight 675, Road 710, Plenty 745, Monopoly 780
            WIN.blit(text, (dc_x_positions[i] + 5, stats_y - 5))

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_game.py | _render_scoreboard_row | Player {player.id}: {player_name} {stats} RC: {rc_stats}\n")

    def draw_guidance(self):
        self.human_guidance.draw()

    def draw_guidance_text(self, lines: list[str] | str, y_offset: int = 0, font_size: str = "normal"):
        font = Font.NORMAL.value["regular"] if font_size == "normal" else Font.LARGE.value["regular"]
        """
        Draw one or more lines of guidance text.
        - lines: can be a single string or a list of strings
        - y_offset: extra pixels to shift the whole block downward (default 0)
        """
        if isinstance(lines, str):
            lines = [lines]  # convert single string to list

        # Clear previous text area — make it taller to fit 2 lines safely
        rect_height = 40 + (len(lines) - 1) * 20  # ~30px per extra line
        pygame.draw.rect(WIN, COLORS["LGRAY"], [20, 20 + y_offset, 400, rect_height])

        font = Font.NORMAL.value["regular"]
        y = 28 + y_offset  # base y-position + offset

        for line in lines:
            if line:  # skip empty lines
                surf = font.render(line, True, COLORS["BLACK"])
                WIN.blit(surf, (15, y))
                y += 20  # line spacing (adjust if your font is taller/shorter)

        pygame.display.update()  # optional — can be removed if called from elsewhere

    def handle_confirmation_click(self, pos: Tuple[int, int]) -> str | None:
        """Check if click was on the dynamic OKY / OKN icons."""
        if not self.human_guidance.confirm_center:
            return None

        x, y = self.human_guidance.confirm_center
        
        # OKY is drawn at (x + 35, y - 45), size 40×40
        oky_rect = pygame.Rect(x + 35, y - 45, 40, 40)
        
        # OKN is drawn at (x + 35, y + 10), size 40×40
        okn_rect = pygame.Rect(x + 35, y + 10, 40, 40)

        if oky_rect.collidepoint(pos):
            return "OKY"
        if okn_rect.collidepoint(pos):
            return "OKN"
        
        return None               

    def update_resource_exploration(self, board: Board):
        """Display resource exploration (= pip summary above the scoreboard)."""
        
        # ─── Display Constants ───────────────────────────────────────────────
        AREA_X            = 10
        AREA_Y_START      = 100
        HEADER_Y          = AREA_Y_START          # "Resource Potential:" title
        LABEL_Y           = AREA_Y_START + 25     # Resource names (Wheat, Ore...)
        CURRENT_Y         = AREA_Y_START + 40     # "Current:" row numbers
        APPROX_Y          = AREA_Y_START + 65     # "Remaining:" row numbers
        
        BG_RECT_X         = 5
        BG_RECT_Y         = AREA_Y_START - 10
        BG_RECT_WIDTH     = 400                   # wider to fit longer header
        BG_RECT_HEIGHT    = 135                   # taller for three rows + spacing
        BG_COLOR          = COLORS["LGRAY"]
        
        COL_WIDTH         = 50
        COL_WHEAT         = 150
        COL_ORE           = COL_WHEAT + COL_WIDTH
        COL_WOOD          = COL_ORE   + COL_WIDTH
        COL_BRICK         = COL_WOOD  + COL_WIDTH
        COL_SHEEP         = COL_BRICK + COL_WIDTH
        
        RESOURCE_COLUMNS = {
            "wheat": COL_WHEAT,
            "ore": COL_ORE,
            "wood": COL_WOOD,
            "brick": COL_BRICK,
            "wool": COL_SHEEP,
        }

        RESOURCE_LABELS = {
            "wheat": "Wheat",
            "ore": "Ore",
            "wood": "Wood",
            "brick": "Brick",
            "wool": "Sheep",
        }
        
        FONT_HEADER = Font.NORMAL.value["bold"]
        FONT_NORMAL = Font.NORMAL.value["regular"]
        FONT_SMALL  = Font.SMALL.value["regular"]
        # ─────────────────────────────────────────────────────────────────────

        # Clear background rectangle
        pygame.draw.rect(WIN, BG_COLOR, 
                        (BG_RECT_X, BG_RECT_Y, BG_RECT_WIDTH, BG_RECT_HEIGHT))

        # Header
        header_surf = FONT_HEADER.render("Resource Potential:", 
                                        True, COLORS["BLACK"])
        WIN.blit(header_surf, (AREA_X, HEADER_Y))

        # Resource labels (Wheat, Ore, etc. — now on their own row above numbers)
        for res, cx in RESOURCE_COLUMNS.items():
            label = RESOURCE_LABELS.get(res, res.capitalize())
            label_surf = FONT_SMALL.render(label, True, COLORS["DGRAY"])
            WIN.blit(label_surf, (cx - label_surf.get_width() // 2, LABEL_Y))

        # ── Current factual row ──────────────────────────────────────────────
        current = board.get_current_settlement_pips()
        
        current_text = FONT_NORMAL.render("Current:", True, COLORS["BLACK"])
        WIN.blit(current_text, (AREA_X, CURRENT_Y))

        for res, cx in RESOURCE_COLUMNS.items():
            val = current.get(res, 0.0)
            txt = FONT_NORMAL.render(f"{val:.1f}", True, COLORS["BLACK"])
            WIN.blit(txt, (cx - txt.get_width() // 2, CURRENT_Y))

        # ── Approximation row ────────────────────────────────────────────────
        approx = board.resource_exploration()
        
        approx_text = FONT_NORMAL.render("Remaining:", True, COLORS["BLACK"])
        WIN.blit(approx_text, (AREA_X, APPROX_Y))

        for res, cx in RESOURCE_COLUMNS.items():
            if res not in approx:
                continue
            mi = approx[res]["min"]
            ma = approx[res]["max"]
            if abs(mi - ma) < 0.5:
                display_str = f"{mi:.1f}"
            else:
                display_str = f"{mi:.0f}–{ma:.0f}"
            txt = FONT_NORMAL.render(display_str, True, COLORS["BLACK"])
            WIN.blit(txt, (cx - txt.get_width() // 2, APPROX_Y))

    def clear_execution_guidance(self) -> None:
        """
        Clear any placement-only highlights/guidance when Execution starts.
        """
        try:
            if hasattr(self, "human_guidance") and self.human_guidance:
                self.human_guidance.clear()
        except Exception:
            pass

        self.animate_queue_elements.clear()

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write("gui.py | clear_execution_guidance\n")

    def draw_fast_forward_expected_actions_debug(self, game: Game) -> None:
        """
        TEMP v014 debug overlay.

        Draws the latest fast-forward prediction row per player just right
        of the scoreboard.

        Shows:
            - predicted round / turn
            - chosen strategy
            - expected viable options, e.g. D,C,S
            - compact expected action scores, e.g. D1.1 C2.0 S7.0

        This is display-only. It does not affect prediction or execution.

        v014:
        - Header lines are moved 30 px up.
        - Player rows stay at the original positions.
        """
        if getattr(game, "phase", None) != "Execution":
            return

        rows = list(getattr(game, "ff_debug_prediction_rows", []) or [])

        # Same player-row area as before.
        panel_x = 820
        panel_y = 570
        panel_w = 360
        panel_h = 180

        # Move only the two header lines 30 px up.
        header_y = panel_y - 30

        # Clear from moved header through the original player rows.
        pygame.draw.rect(
            WIN,
            COLORS["LGRAY"],
            [panel_x, header_y, panel_w, panel_h + 30],
        )

        font_small = Font.SMALL.value["regular"]
        font_normal = Font.NORMAL.value["regular"]

        def _short(value, max_len: int) -> str:
            txt = str(value)
            if len(txt) <= max_len:
                return txt
            return txt[: max(0, max_len - 3)] + "..."

        def _safe_float(value, default: float = 9999.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _format_expected_scores(row: dict) -> str:
            actions = list(row.get("expected_viable_actions", []) or [])
            parts = []

            for action in actions[:4]:
                code = str(action.get("code", "?"))
                score = _safe_float(action.get("score", 9999.0))

                if score >= 9999.0:
                    score_txt = "∞"
                else:
                    score_txt = f"{score:.1f}"

                parts.append(f"{code}{score_txt}")

            return " ".join(parts) if parts else "-"

        rows_by_player = {}
        for row in rows:
            try:
                pid = int(row.get("player_id"))
            except Exception:
                continue
            rows_by_player[pid] = row

        # Header lines: moved 30 px up.
        title = font_normal.render("FF expected actions", True, COLORS["BLACK"])
        WIN.blit(title, (panel_x, header_y))

        subtitle = font_small.render("R/T  strategy  [opts]  scores", True, COLORS["DGRAY"])
        WIN.blit(subtitle, (panel_x, header_y + 18))

        for idx, player in enumerate(game.players):
            # Player rows stay unchanged.
            row_y = 590 + idx * 40

            player_color = COLORS.get(player.color.upper(), COLORS["BLACK"])
            player_txt = font_normal.render(f"P{player.id}", True, player_color)
            WIN.blit(player_txt, (panel_x, row_y))

            row = rows_by_player.get(int(player.id))

            if not row:
                info = "-"
                scores = "-"
            else:
                pred_round = row.get("pred_round", "?")
                pred_turn = row.get("pred_turn", "?")
                strategy = row.get("strategy", "?")
                opts = row.get("expected_viable_codes", "-")
                scores = _format_expected_scores(row)

                info = (
                    f"R{pred_round}/T{pred_turn} "
                    f"{_short(strategy, 15)} "
                    f"[{_short(opts, 7)}]"
                )

            info_txt = font_small.render(info, True, COLORS["BLACK"])
            WIN.blit(info_txt, (panel_x + 45, row_y + 1))

            score_txt = font_small.render(_short(scores, 30), True, COLORS["DGRAY"])
            WIN.blit(score_txt, (panel_x + 45, row_y + 18))

    def queue_fast_forward_action(self) -> None:
        """
        Build animation queue from all actions executed by the latest fast-forward PLAY.

        v014:
        - Animate the primary PLAY action.
        - Also animate every successful same-turn chain action.
        - Roads + settlement are both animated for settlement bundles.
        - City upgrades are animated even when they happen inside the same-turn chain.
        - Dev-card buys are pulsed near the dev-card scoreboard area.
        """
        self.animate_queue_elements.clear()

        actor_id = getattr(self.game, "ff_last_actor_id", None)
        details = getattr(self.game, "ff_last_details", {}) or {}

        if actor_id is None:
            return

        player = next((p for p in self.game.players if int(p.id) == int(actor_id)), None)
        if player is None:
            return

        color = COLORS.get(player.color.upper(), COLORS["BLACK"])

        def _road_midpoint(road_id):
            try:
                a, b = int(road_id[0]), int(road_id[1])
            except Exception:
                return None

            pos1 = POSITIONS["intersections"].get(a)
            pos2 = POSITIONS["intersections"].get(b)

            if not pos1 or not pos2:
                return None

            return ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)

        def _queue_one_result(result: dict, dev_index: int = 0) -> int:
            """
            Queue visual animation elements for one successful execution result.

            Returns updated dev_index so multiple dev-card purchases do not all pulse
            in exactly the same spot.
            """
            if not isinstance(result, dict):
                return dev_index

            if not bool(result.get("success", False)):
                return dev_index

            activity = str(result.get("actual_activity", "") or "")

            # ------------------------------------------------------------
            # New settlement bundle: animate roads first, then settlement.
            # ------------------------------------------------------------
            if activity == "new_settlement":
                built_roads = list(result.get("built_roads", []) or [])

                for road_id in built_roads:
                    mid = _road_midpoint(road_id)
                    if mid:
                        self.animate_queue_elements.append((mid, color, 20, "road"))

                inter_id = result.get("chosen_tw")
                try:
                    inter_id = int(inter_id)
                except Exception:
                    inter_id = None

                if inter_id is not None:
                    pos = POSITIONS["intersections"].get(inter_id)
                    if pos:
                        self.animate_queue_elements.append((pos, color, 20, "settlement"))

                return dev_index

            # ------------------------------------------------------------
            # City upgrade.
            # ------------------------------------------------------------
            if activity == "upgrade_to_city":
                inter_id = result.get("chosen_tw")
                try:
                    inter_id = int(inter_id)
                except Exception:
                    inter_id = None

                if inter_id is not None:
                    pos = POSITIONS["intersections"].get(inter_id)
                    if pos:
                        self.animate_queue_elements.append((pos, color, 24, "city"))

                return dev_index

            # ------------------------------------------------------------
            # One dev card.
            # ------------------------------------------------------------
            if activity == "buy_discovery_card":
                # Pulse near the dev-card area of the scoreboard.
                # Offset repeated dev-card buys slightly so multiple buys are visible.
                x = 720 + (dev_index * 18)
                y = 560
                self.animate_queue_elements.append(((x, y), color, 24, "tile"))
                return dev_index + 1

            # ------------------------------------------------------------
            # Legacy / batch dev-card action.
            # ------------------------------------------------------------
            if activity == "buy_4_discovery_cards":
                x = 720 + (dev_index * 18)
                y = 560
                self.animate_queue_elements.append(((x, y), color, 28, "tile"))
                return dev_index + 1

            return dev_index

        # ------------------------------------------------------------
        # 1. Primary PLAY result.
        # ------------------------------------------------------------
        dev_index = 0
        dev_index = _queue_one_result(details, dev_index=dev_index)

        # ------------------------------------------------------------
        # 2. Same-turn chain results.
        #    Each chain row has shape:
        #       {
        #           "chain_step": ...,
        #           "requested_activity": ...,
        #           "result": {...}
        #       }
        # ------------------------------------------------------------
        for chain_row in list(details.get("same_turn_chain_results", []) or []):
            if not isinstance(chain_row, dict):
                continue

            chain_result = dict(chain_row.get("result", {}) or {})
            dev_index = _queue_one_result(chain_result, dev_index=dev_index)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"gui.py | queue_fast_forward_action | actor={actor_id} "
                    f"queue_count={len(self.animate_queue_elements)} "
                    f"queue={self.animate_queue_elements}\n"
                )

    def _redraw_execution_view(self, board: Board) -> None:
        """
        Redraw the board and permanent structures for Execution mode.
        """
        self.draw_board_base(board)
        self.draw_all_permanent_buildings(board)
        self.update_scoreboard(self.game)
        self.update_round_turn(self.game, special=False)
        self.draw_fast_forward_status(self.game)                                             