"""
Main entry point for the Catan game.
"""
from datetime import datetime
import pygame
from core.game import Game
from gui.gui import GUI
from gui.gui_human_player import GUIHumanPlayer
from gui.gui_constants import WIN, COLORS, POSITIONS, initialize_sounds
from core.initial_placement import InitialPlacement
from gui.event_handler import EventHandler
from gui.gui_guidance import PlacementState
from core.fast_forward import FastForwardEngine

def main():
    """Run the main game loop."""
    pygame.init()
    initialize_sounds()
    clock = pygame.time.Clock()

    today = datetime.now().strftime("%Y%m%d")

    # Initialize game
    game = Game(
        sequence_number=1,
        id_=today,
        phase="InitialPlacement",
        state="None",
        state_1="0",
        state_2="0",
        myplayers=None,
        board_name="Base_Random"
    )
    game.ip = InitialPlacement(game)
    game.fast_forward = FastForwardEngine(game)

    # Initialize GUI & handler
    gui = GUI(round_number=game.round, turn=game.turn, game=game)
    game.gui = gui
    gui_hp = GUIHumanPlayer()
    game.gui_hp = gui_hp
    gui.gui_hp = gui_hp
    event_handler = EventHandler()

    # Initial render
    WIN.fill(COLORS["LGRAY"])
    gui.display_fresh_board(game.board, scoreboard_tf=True)
    gui.update_round_turn(game, special=True)
    gui.update_scoreboard(game)
    gui_hp.show_buttons_HP(game, analysis_tf=False)
    pygame.display.update()

    # Start initial placement sequence
    game.ip.run()

    running = True
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                event_handler.handle_click(event.pos, game)

        # ────────────────────────────────────────────────
        # Continuous rendering + animation (v045 style)
        # ────────────────────────────────────────────────
        # WIN.fill(COLORS["LGRAY"])

        # gui.display_fresh_board(game.board, scoreboard_tf=True)
        # gui.update_round_turn(game, special=False)
        # gui.update_scoreboard(game)
        # gui_hp.show_buttons_HP(game, analysis_tf=False)

        # Dynamic overlays
        game.gui.human_guidance.draw()

        # Continuous animation of last placement (mimics v045)
        game.gui.animate_continuous()

        ### if game.gui.human_guidance.state == PlacementState.CHOOSING_SETTLEMENT:
        ###     print(f"Main loop - pulsing should be visible - queue len = {len(game.gui.animate_queue_elements)}")

        pygame.display.update()
        clock.tick(60)

    # Game over – celebratory final animation
    print("Game over – running final animation sequence")

    gui.animate_queue_elements = []

    for inter in game.board.intersections:
        if inter and inter.occupied_tf:
            pos = POSITIONS["intersections"].get(inter.id)
            if pos:
                kind = "settlement" if inter.face == "Settlement" else "city"
                gui.animate_queue_elements.append((pos, COLORS[inter.color.upper()], 20, kind))

    for road in game.board.roads:
        if road and road.occupied_tf:
            start = POSITIONS["intersections"].get(road.id[0])
            end = POSITIONS["intersections"].get(road.id[1])
            if start and end:
                mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
                gui.animate_queue_elements.append((mid, COLORS[road.color.upper()], 20, "road"))

    if gui.animate_queue_elements and gui.animate_queue_elements!=[]:
        gui._animate_elements(game.board)

    WIN.fill(COLORS["LGRAY"])
    gui.display_fresh_board(game.board, scoreboard_tf=True)
    gui.update_scoreboard(game)
    gui.update_round_turn(game, special=False)
    game.gui.human_guidance.draw()
    pygame.display.update()

    pygame.time.wait(2000)
    pygame.quit()


if __name__ == "__main__":
    main()