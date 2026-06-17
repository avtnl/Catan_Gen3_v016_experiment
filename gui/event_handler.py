"""
v010b
Handles mouse click events for the Catan game.

This module defines the EventHandler class, responsible for processing mouse clicks,
including human placement of settlements/roads and confirmation (OKY/OKN) during
InitialPlacement phase.

Dependencies:
    - pygame: For event handling and sound.
    - gui.gui_constants: For button positions and sounds.
    - gui.gui_guidance: For human guidance and confirmation.
    - gui.gui_human_player: For rendering / activating / deactivating human buttons.
    - core.game: For game state.
    - core.constants: For logging and configuration.
"""

import pygame
from typing import Tuple

from gui.gui_constants import SOUNDS
from gui.gui_guidance import PlacementState
from gui.gui_human_player import GUIHumanPlayer
from core.game import Game
from core.constants import FNFREQ, FILENAME_FREQ


class EventHandler:
    """Manages mouse click events for the Catan game."""

    def __init__(self) -> None:
        """Initialize the EventHandler."""
        pass

    def handle_click(self, pos: Tuple[int, int], game: Game) -> bool:
        if FNFREQ == "Y":
            with open(FILENAME_FREQ, "a") as f:
                f.write(
                    f"{game.id} | {game.state} | event_handler.py | "
                    f"handle_click | pos={pos}\n"
                )

        guidance = getattr(game.gui, "human_guidance", None)
        PLAY_RECT = pygame.Rect(20, 470, 130, 40)

        def _play_sound(name: str) -> None:
            snd = SOUNDS.get(name)
            if snd is not None:
                pygame.mixer.Sound.play(snd)

        # ────────────────────────────────────────────────
        # 1. InitialPlacement confirmation clicks: OKY / OKN
        # ────────────────────────────────────────────────
        if game.phase == "InitialPlacement" and guidance is not None:
            if guidance.state in (
                PlacementState.SETTLEMENT_SELECTED,
                PlacementState.ROAD_SELECTED,
            ):
                conf = game.gui.handle_confirmation_click(pos)

                if conf:
                    print(f"Confirmation clicked: {conf}")
                    guidance.on_confirmation(conf)
                    return True

                # Clicked during confirmation but not on OKY / OKN.
                _play_sound("ERROR")
                return True

        # ────────────────────────────────────────────────
        # 2. Main lower-left button:
        #    InitialPlacement -> PLAY
        #    Execution        -> JUMP / PLAY
        # ────────────────────────────────────────────────
        if PLAY_RECT.collidepoint(pos):

            # ------------------------------------------------------------
            # InitialPlacement: PLAY advances setup only when not placing
            # ------------------------------------------------------------
            if game.phase == "InitialPlacement":
                is_placing = guidance is not None and guidance.state != PlacementState.IDLE
                button_active = game.gui.check_button("next_turn2")

                if is_placing:
                    print("PLAY clicked while still placing → rejected")
                    _play_sound("ERROR")
                    return True

                if not button_active:
                    print("PLAY clicked but button is inactive → rejected")
                    _play_sound("ERROR")
                    return True

                print("PLAY clicked → advancing turn")
                _play_sound("BUTTON")

                # v014: show busy feedback during InitialPlacement AI placement/Markov work.
                game.ff_processing = True
                game.ff_processing_text = "AI is choosing placement..."

                # Immediately deactivate before AI / Markov output starts.
                GUIHumanPlayer.button_next_turn2(game.gui, game, active=False)

                if hasattr(game.gui, "set_ai_busy_indicator"):
                    game.gui.set_ai_busy_indicator(True, game.ff_processing_text)

                pygame.display.update()

                try:
                    print("event_handler calling InitialPlacement.advance_turn")
                    game.ip.advance_turn()

                finally:
                    game.ff_processing = False
                    game.ff_processing_text = ""

                    if hasattr(game.gui, "set_ai_busy_indicator"):
                        game.gui.set_ai_busy_indicator(False)

                    # Repaint button panel after placement work.
                    # If the game entered Execution, enter_execution_phase/show_buttons will handle the final state.
                    if getattr(game, "phase", None) == "InitialPlacement":
                        if hasattr(game, "gui_hp") and game.gui_hp:
                            game.gui_hp.show_buttons_HP(game, analysis_tf=False)
                        else:
                            GUIHumanPlayer.button_next_turn2(game.gui, game, active=True)

                    pygame.display.update()

                return True

            # ------------------------------------------------------------
            # Execution: same button is JUMP / PLAY
            # ------------------------------------------------------------
            if game.phase == "Execution":
                ff = getattr(game, "fast_forward", None)

                if ff is None:
                    print("Execution button clicked but game.fast_forward is missing")
                    _play_sound("ERROR")
                    return True

                if getattr(game, "game_over", False):
                    print("Execution button clicked but game_over=True")
                    _play_sound("ERROR")
                    return True

                button_active = game.gui.check_button("next_turn2")

                if not button_active:
                    print("Execution button clicked but button is inactive → rejected")
                    _play_sound("ERROR")
                    return True

                if getattr(game, "ff_processing", False):
                    print("Execution button clicked while ff_processing=True → rejected")
                    _play_sound("ERROR")
                    return True

                mode = str(getattr(game, "ff_button_mode", "JUMP")).upper()
                if mode not in ("JUMP", "PLAY"):
                    mode = "JUMP"
                    game.ff_button_mode = "JUMP"

                print(f"Execution button clicked → mode={mode}")
                _play_sound("BUTTON")

                game.ff_processing = True
                game.ff_processing_text = (
                    "AI is calculating JUMP..."
                    if mode == "JUMP"
                    else "AI is executing PLAY..."
                )

                # Immediately gray out the button while the task is running.
                GUIHumanPlayer.button_next_turn2(game.gui, game, active=False)

                if hasattr(game.gui, "set_ai_busy_indicator"):
                    game.gui.set_ai_busy_indicator(True, game.ff_processing_text)

                pygame.display.update()

                try:
                    if mode == "JUMP":
                        ff.jump_to_next_event()
                        game.gui.update_board(game.board, "FastForwardJump")

                    elif mode == "PLAY":
                        ff.play_staged_event()
                        game.gui.update_board(game.board, "FastForwardLast")

                except Exception as exc:
                    print(f"⚠️ Execution {mode} failed: {exc}")

                finally:
                    game.ff_processing = False
                    game.ff_processing_text = ""

                    if hasattr(game.gui, "set_ai_busy_indicator"):
                        game.gui.set_ai_busy_indicator(False)
                    
                    if hasattr(game, "gui_hp") and game.gui_hp:
                        game.gui_hp.show_buttons_HP(game, analysis_tf=False)
                    else:
                        GUIHumanPlayer.button_next_turn2(game.gui, game, active=True)

                    game.gui.update_scoreboard(game)
                    game.gui.update_round_turn(game, special=False)
                    pygame.display.update()

                return True

            # Any other phase: button area clicked, but no action.
            print(f"Main button clicked in unsupported phase={game.phase}")
            _play_sound("ERROR")
            return True

        # ────────────────────────────────────────────────
        # 3. Board clicks during InitialPlacement
        # ────────────────────────────────────────────────
        if game.phase == "InitialPlacement":
            if game.ip.handle_click(pos):
                return True

            if guidance is not None and guidance.state != PlacementState.IDLE:
                if guidance.on_board_click(pos):
                    return True

        # Execution has no human board interaction.
        return False