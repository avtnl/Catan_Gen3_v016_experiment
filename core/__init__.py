"""
Core module for the Catan game.
This module imports key classes for game logic, board management, and initial placement phase.
Classes:
    Board: Manages the game board layout and state.
    Player: Represents a player with resources and game statistics.
    Game: Handles game state, player interactions, and turn progression.
    InitialPlacement: Manages the initial placement phase for settlements and roads.
"""
from .board import Board
from .player import Player
from .game import Game
from .initial_placement import InitialPlacement