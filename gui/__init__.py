"""
GUI package for the Catan game.
This package provides modules for rendering and managing the graphical user interface,
including the game board, scoreboard, human player interaction buttons, and event handling.
Modules:
    gui: Manages board rendering, scoreboard rendering and button states.
    gui_human_player: Manages button rendering for human player actions.
    gui_constants: Defines fonts, colors, images, and positions.
    event_handler: Processes mouse click events for user interactions.
"""
from .gui import GUI
from .gui_human_player import GUIHumanPlayer
from .gui_constants import *
from .gui_guidance import HumanGuidance
from .event_handler import EventHandler