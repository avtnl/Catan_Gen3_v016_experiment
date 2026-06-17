"""
Handles button rendering for human player interactions in the Catan game.

This module defines the GUIHumanPlayer class, responsible for rendering buttons
for human player actions (e.g., buying settlements, rolling dice) within a panel.
Buttons are conditionally displayed based on game phase, human player status, and modes.

Classes:
    GUIHumanPlayer: Manages button rendering for human player interactions.

Dependencies:
    - pygame: For rendering graphics.
    - gui.gui_constants: For fonts, colors, images, and positions.
    - core.game: For game state.
    - core.player: For player attributes.
    - core.constants: For logging and configuration constants.
"""
import pygame
from core import game
from gui.gui_constants import WIN, COLORS, Font, IMAGES
from core.game import Game
from core.player import ResourceCard
from core.constants import FNFREQ, MG, FILENAME_FREQ, FILENAME_MG, HP_ID, HUMAN_PLAYER, FILENAME_SPEC2

class GUIHumanPlayer:
    """Manages button rendering for human player interactions."""
    
    def __init__(self) -> None:
        """Initialize the GUIHumanPlayer with an empty state.

        Args:
            None
        """
        pass

    def text_buy(self, game: Game, active: bool) -> None:
        """Render 'Buy -->' text with active or inactive styling.

        Args:
            game: The game instance.
            active: Whether the text is active (black) or inactive (gray).
        """
        font = Font.LARGE.value["regular"]
        color = COLORS["BLACK"] if active else COLORS["GRAY"]
        game.gui.set_button("text_buy", active)
        text = font.render("Buy -->", True, color)
        WIN.blit(text, (25, 262))
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | text_buy | Active: {active}\n")

    def text_trade(self, game: Game, active: bool) -> None:
        """Render 'Trade -->' text with active or inactive styling.

        Args:
            game: The game instance.
            active: Whether the text is active (black) or inactive (gray).
        """
        font = Font.LARGE.value["regular"]
        color = COLORS["BLACK"] if active else COLORS["GRAY"]
        game.gui.set_button("text_trade", active)
        text = font.render("Trade -->", True, color)
        WIN.blit(text, (25, 322))
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | text_trade | Active: {active}\n")

    def button_buy_city(self, game: Game, active: bool) -> None:
        """Render 'Buy City' button with image and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, CITY_GREEN) or inactive (gray border, CITY_DGRAY).
        """
        game.gui.set_button("buy_city", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        image_key = "CITY_GREEN" if active else "CITY_DGRAY"
        pygame.draw.rect(WIN, border_color, [140, 260, 40, 40], 2)
        image = IMAGES.get(image_key, {}).get("30x30")
        if image is not None:
            WIN.blit(image, (145, 265))
        else:
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui_human_player.py | button_buy_city | Missing image: {image_key}\n")
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_buy_city | Active: {active}\n")

    def button_buy_settlement(self, game: Game, active: bool) -> None:
        """Render 'Buy Settlement' button with image and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, SETTLEMENT_GREEN) or inactive (gray border, SETTLEMENT_DGRAY).
        """
        game.gui.set_button("buy_settlement", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        image_key = "SETTLEMENT_GREEN" if active else "SETTLEMENT_DGRAY"
        pygame.draw.rect(WIN, border_color, [190, 260, 40, 40], 2)
        image = IMAGES.get(image_key, {}).get("30x30")
        if image is not None:
            WIN.blit(image, (195, 265))
        else:
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui_human_player.py | button_buy_settlement | Missing image: {image_key}\n")
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_buy_settlement | Active: {active}\n")

    def button_buy_road(self, game: Game, active: bool) -> None:
        """Render 'Buy Road' button with image and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, ROAD_GREEN) or inactive (gray border, ROAD_DGRAY).
        """
        game.gui.set_button("buy_road", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        image_key = "ROAD_GREEN" if active else "ROAD_DGRAY"
        pygame.draw.rect(WIN, border_color, [240, 260, 40, 40], 2)
        image = IMAGES.get(image_key, {}).get("30x30")
        if image is not None:
            WIN.blit(image, (245, 265))
        else:
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui_human_player.py | button_buy_road | Missing image: {image_key}\n")
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_buy_road | Active: {active}\n")

    def button_buy_dcard(self, game: Game, active: bool) -> None:
        """Render 'Buy DCard' button with image and border, checking resource availability.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, DCARD_GREEN) or inactive (gray border, DCARD_DGRAY), modified by resource availability.
        """
        human_player = game.players[game.turn - 1]
        resources = [
            human_player.rcards.get(ResourceCard.WHEAT, 0),
            human_player.rcards.get(ResourceCard.ORE, 0),
            human_player.rcards.get(ResourceCard.WOOL, 0)
        ]
        can_buy = (resources[0] >= 1 and resources[1] >= 1 and resources[2] >= 1 and len(game.dcards_stack) > 0)
        game.gui.set_button("buy_dcard", active and can_buy)
        border_color = COLORS["GREEN"] if active and can_buy else COLORS["GRAY"]
        image_key = "DCARD_GREEN" if active and can_buy else "DCARD_DGRAY"
        pygame.draw.rect(WIN, border_color, [290, 260, 40, 40], 2)
        image = IMAGES.get(image_key, {}).get("30x30")
        if image is not None:
            WIN.blit(image, (295, 265))
        else:
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui_human_player.py | button_buy_dcard | Missing image: {image_key}\n")
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_buy_dcard | Active: {active and can_buy}\n")

    def button_twp(self, game: Game, active: bool) -> None:
        """Render 'Trade w/ Player' button with text and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, white text) or inactive (gray border, gray text).
        """
        game.gui.set_button("twp", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        text_color = COLORS["WHITE"] if active else COLORS["GRAY"]
        pygame.draw.rect(WIN, border_color, [200, 320, 60, 40], 2)
        text = Font.LARGE.value["regular"].render("TwP", True, text_color)
        WIN.blit(text, (205, 322))
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_twp | Active: {active}\n")

    def button_twb(self, game: Game, active: bool) -> None:
        """Render 'Trade w/ Bank' button with text and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, white text) or inactive (gray border, gray text).
        """
        game.gui.set_button("twb", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        text_color = COLORS["WHITE"] if active else COLORS["GRAY"]
        pygame.draw.rect(WIN, border_color, [270, 320, 60, 40], 2)
        text = Font.LARGE.value["regular"].render("TwB", True, text_color)
        WIN.blit(text, (275, 322))
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_twb | Active: {active}\n")

    def button_roll_dices(self, game: Game, active: bool) -> None:
        """Render 'Roll Dices' button with text and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, white text) or inactive (gray border, gray text).
        """
        game.gui.set_button("roll_dices", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        text_color = COLORS["WHITE"] if active else COLORS["GRAY"]
        pygame.draw.rect(WIN, border_color, [200, 400, 130, 40], 2)
        text = Font.LARGE.value["regular"].render("Roll Dices", True, text_color)
        WIN.blit(text, (205, 402))
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_roll_dices | Active: {active}\n")

    def button_end_turn(self, game: Game, active: bool) -> None:
        """Render 'End Turn' button with text and border.

        Args:
            game: The game instance.
            active: Whether the button is active (green border, white text) or inactive (gray border, gray text).
        """
        game.gui.set_button("end_turn", active)
        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        text_color = COLORS["WHITE"] if active else COLORS["GRAY"]
        pygame.draw.rect(WIN, border_color, [200, 470, 130, 40], 2)
        text = Font.LARGE.value["regular"].render("End Turn", True, text_color)
        WIN.blit(text, (205, 472))
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_human_player.py | button_end_turn | Active: {active}\n")

    def button_cancel(self, game: Game, active: bool) -> None:
        """Render Cancel button.

        In Execution phase, do not draw the word 'Cancel' because the same
        button area can overlap with the Execution/End Turn button text.
        """
        game.gui.set_button("cancel", active)

        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        text_color = COLORS["WHITE"] if active else COLORS["GRAY"]

        pygame.draw.rect(WIN, border_color, [200, 470, 130, 40], 2)

        # v014: avoid overlaying "Cancel" with "End Turn" during Execution.
        if getattr(game, "phase", None) != "Execution":
            text = Font.LARGE.value["regular"].render("Cancel", True, text_color)
            WIN.blit(text, (205, 472))

        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(
                    f"gui_human_player.py | button_cancel | "
                    f"Active: {active} phase={getattr(game, 'phase', None)}\n"
                )

    def button_next_turn2(self, game: Game, active: bool) -> None:
        """
        Render the main lower-left button.

        Meaning by phase:
            - InitialPlacement -> PLAY
            - Execution + ff_button_mode == "JUMP" -> JUMP
            - Execution + ff_button_mode == "PLAY" -> PLAY
        """
        game.gui.set_button("next_turn2", active)

        rect = [20, 470, 130, 40]

        # Clear/fill the whole button area first so old text disappears
        pygame.draw.rect(WIN, COLORS["LGRAY"], rect)

        border_color = COLORS["GREEN"] if active else COLORS["GRAY"]
        text_color = COLORS["WHITE"] if active else COLORS["GRAY"]

        pygame.draw.rect(WIN, border_color, rect, 2)

        if game.phase == "Execution":
            label = str(getattr(game, "ff_button_mode", "JUMP")).upper()
            if label not in ("JUMP", "PLAY"):
                label = "JUMP"
        else:
            label = "PLAY"

        text = Font.LARGE.value["regular"].render(label, True, text_color)
        text_rect = text.get_rect(center=(rect[0] + rect[2] // 2, rect[1] + rect[3] // 2))
        WIN.blit(text, text_rect)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"gui_human_player.py | button_next_turn2 | "
                    f"Phase: {game.phase} | Label: {label} | Active: {active}\n"
                )

    def show_buttons_HP(self, game: Game, analysis_tf: bool = False) -> None:
        """
        Render the button panel.

        Design:
            - InitialPlacement:
                human may interact only there
                next_turn2 is PLAY
            - Execution:
                AI-only
                next_turn2 becomes JUMP / PLAY
                all manual buy/trade/roll/end-turn controls stay disabled
        """
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a", encoding="utf-8") as f:
                f.write(f"{game.sequence_number} | {game.state} | gui_human_player.py | show_buttons_HP\n")

        # Panel border
        pygame.draw.rect(WIN, COLORS["BLACK"], [10, 250, 330, 270], 2)

        def disable_manual_controls() -> None:
            self.text_buy(game, False)
            self.text_trade(game, False)
            self.button_buy_city(game, False)
            self.button_buy_settlement(game, False)
            self.button_buy_road(game, False)
            self.button_buy_dcard(game, False)
            self.button_twp(game, False)
            self.button_twb(game, False)
            self.button_roll_dices(game, False)
            self.button_end_turn(game, False)
            self.button_cancel(game, False)

        # ------------------------------------------------------------
        # Planning / analysis mode (keep minimal, optional)
        # ------------------------------------------------------------
        if analysis_tf:
            disable_manual_controls()
            self.button_next_turn2(game, True)
            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write("gui_human_player.py | show_buttons_HP | Analysis mode\n")
            pygame.display.update()
            return

        # ------------------------------------------------------------
        # Initial Placement: human can still interact here
        # ------------------------------------------------------------
        if game.phase == "InitialPlacement":
            disable_manual_controls()

            is_placing = False
            if hasattr(game.gui, "human_guidance") and game.gui.human_guidance:
                try:
                    is_placing = game.gui.human_guidance.is_placing()
                except Exception:
                    is_placing = False

            # PLAY is enabled only when not actively placing/confirming
            # (e.g. after a human completed placement, or when AI steps should continue)
            self.button_next_turn2(game, not is_placing)

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"gui_human_player.py | show_buttons_HP | "
                        f"Phase=InitialPlacement | is_placing={is_placing} | "
                        f"next_turn2_active={not is_placing}\n"
                    )

            pygame.display.update()
            return

        # ------------------------------------------------------------
        # Execution: AI-only
        # ------------------------------------------------------------
        if game.phase == "Execution":
            disable_manual_controls()

            ff_engine_exists = hasattr(game, "fast_forward") and game.fast_forward is not None
            game_over = bool(getattr(game, "game_over", False))
            ff_processing = bool(getattr(game, "ff_processing", False))

            # v014:
            # JUMP/PLAY may only be active when:
            # - fast_forward exists
            # - game is not over
            # - no FF/AI processing is currently running
            execution_button_active = (
                ff_engine_exists
                and not game_over
                and not ff_processing
            )

            # next_turn2 now shows JUMP or PLAY depending on game.ff_button_mode.
            # If ff_processing=True, this draws the button gray/inactive.
            self.button_next_turn2(game, execution_button_active)

            # Keep the busy indicator visible if something else redrew the panel.
            if ff_processing and hasattr(game.gui, "draw_ai_busy_indicator"):
                game.gui.draw_ai_busy_indicator(force_visible=True)

            if MG:
                with open(FILENAME_MG, "a", encoding="utf-8") as f:
                    f.write(
                        f"gui_human_player.py | show_buttons_HP | "
                        f"Phase=Execution | ff_button_mode={getattr(game, 'ff_button_mode', 'JUMP')} | "
                        f"ff_pending_event={getattr(game, 'ff_pending_event', None)} | "
                        f"ff_processing={ff_processing} | "
                        f"ff_engine_exists={ff_engine_exists} | "
                        f"game_over={game_over} | "
                        f"button_active={execution_button_active}\n"
                    )

            pygame.display.update()
            return

        # ------------------------------------------------------------
        # Fallback for any other phase
        # ------------------------------------------------------------
        disable_manual_controls()
        self.button_next_turn2(game, False)

        if MG:
            with open(FILENAME_MG, "a", encoding="utf-8") as f:
                f.write(
                    f"gui_human_player.py | show_buttons_HP | "
                    f"Fallback branch for phase={game.phase}\n"
                )

        pygame.display.update()