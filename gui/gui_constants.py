"""
Defines GUI-related constants for the Catan game.

This module provides fonts, colors, images, sounds, and positioning constants for consistent
rendering of the playboard, scoreboard, and buttons.

Classes:
    Font: Enumeration for font sizes with bold support.
    Color: Enumeration for color values.
    PlayerColor: Enumeration for player color mappings.
    Sound: Enumeration for sound effects.
    Image: Enumeration for image assets.

Constants:
    COLORS: Dictionary of color names to RGB tuples.
    REVERSE_COLOR_MAPPING: Dictionary of color mappings for game logic.
    SOUNDS: Dictionary of sound effect names to Pygame Sound objects.
    IMAGES: Dictionary of image asset names to scaled Pygame Surfaces.
    WIN: Pygame display surface for rendering.
    BOARD_OFFSET: Tuple for offsetting board elements.
    PANEL_OFFSET: Tuple for offsetting panel elements.
    PANEL_OFFSET_Y2: Y-offset for secondary panel elements.
    POSITIONS: Dictionary of tile, intersection, button, and panel positions.

Dependencies:
    - pygame: For font, image, and sound handling.
    - pathlib: For file path management.
    - typing: For type hints.
    - enum: For enumerations.
    - core.constants: For logging constants.
"""
import pygame
from pathlib import Path
from typing import Dict, Tuple, Union
from enum import Enum
from core.constants import MG, FILENAME_MG

# Debug print to confirm file loading
print(f"Loading gui_constants.py from: {__file__}")

class Font(Enum):
    """Enumeration for font sizes."""
   
    SMALL = ("Comic Sans MS", 10)
    NORMAL = ("Comic Sans MS", 16)
    LARGE = ("Comic Sans MS", 24)
    @classmethod
    def initialize_fonts(cls) -> None:
        """Initialize Pygame font module and create font objects with bold variants."""
        if not pygame.font.get_init():
            pygame.font.init()
        for font in cls:
            font_name, size = font.value
            font._value_ = {
                "regular": pygame.font.SysFont(font_name, size, bold=False),
                "bold": pygame.font.SysFont(font_name, size, bold=True)
            }

# class Font(Enum):
#     """Enumeration for font sizes."""
  
#     SMALL = ("Comic Sans MS", 10)
#     NORMAL = ("Comic Sans MS", 16)
#     LARGE = ("Comic Sans MS", 24)

#     @classmethod
#     def initialize_fonts(cls) -> None:
#         """Initialize Pygame font module and create font objects with bold variants."""
#         if not pygame.font.get_init():
#             pygame.font.init()
#         # Log all Font enum values before processing
#         if MG:
#             with open(FILENAME_MG, "a") as f:
#                 f.write(f"gui_constants.py | initialize_fonts | Font enum values before: {[f'{font.name}: {font.value}' for font in cls]}\n")
#         for font in cls:
#             print(f"Font {font.name}: {font.value} (before unpacking)")
#             font_name, size = font.value
#             print(f"Font {font.name}: font_name={font_name}, size={size}, type={type(size)} (after unpacking)")
#             if not isinstance(size, int):
#                 if MG:
#                     with open(FILENAME_MG, "a") as f:
#                         f.write(f"gui_constants.py | initialize_fonts | Invalid size type for font {font.name}: {type(size)}, value: {font.value}\n")
#                 raise TypeError(f"Font size for {font.name} must be an integer, got {type(size)}: {size}")
#             font._font_objects = {  # Store in a new attribute
#                 "regular": pygame.font.SysFont(font_name, size, bold=False),
#                 "bold": pygame.font.SysFont(font_name, size, bold=True)
#             }
#             print(f"Font {font.name}: _font_objects={font._font_objects}")
#         if MG:
#             with open(FILENAME_MG, "a") as f:
#                 f.write(f"gui_constants.py | initialize_fonts | Font enum values after: {[f'{font.name}: {font.value}' for font in cls]}\n")

class Color(Enum):
    """Enumeration for color values used in rendering."""
   
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    LGRAY = (200, 200, 200)
    DGRAY = (100, 100, 100)
    GRAY = (169, 169, 169)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    FIELD = (255, 255, 153)
    MOUNTAIN = (139, 69, 19)
    FOREST = (0, 100, 0)
    HILL = (204, 0, 0)
    PASTURE = (173, 255, 47)
    DESERT = (245, 245, 220)
    SEA = (0, 191, 255)

COLORS: Dict[str, Tuple[int, int, int]] = {color.name: color.value for color in Color}
"""Dictionary mapping color names to RGB tuples for rendering."""

REVERSE_COLOR_MAPPING: Dict[str, str] = {
    "Blue": "Orange",
    "Red": "White",
    "White": "Red",
    "Orange": "Blue"
}
"""Dictionary mapping player colors to their opposites for game logic."""

class Sound(Enum):
    """Enumeration for sound effect file paths."""
   
    DICEROLL = "assets/sounds/DiceRoll.wav"
    BUTTON = "assets/sounds/button-click-3.wav"
    BUTTONHP = "assets/sounds/Bell2.wav"
    BUILDROAD = "assets/sounds/BuildRoad.wav"
    FANFARE = "assets/sounds/fanfare-2.wav"
    BELL = "assets/sounds/success-bell.wav"
    ERROR = "assets/sounds/Error-sound.wav"
    DANGER = "assets/sounds/Danger.wav"
    STEAL = "assets/sounds/CashRegister.wav"
    DEAL = "assets/sounds/CashRegister.wav"
    NOTWPFOUND = "assets/sounds/No_TwP_Found.wav"
    TWPFOUND = "assets/sounds/TwP_Found.wav"
    TWPFOUND2 = "assets/sounds/infobleep.wav"
    BUYDCARD = "assets/sounds/BuyDCard2.wav"
    PLAYDCARD = "assets/sounds/PlayDCard.wav"
    NEXTTURN = "assets/sounds/NoGui_NextTurn.wav"
    NEXTGAME = "assets/sounds/NoGui_NextGame.wav"
    MIDGAME = "assets/sounds/NoGui_Midgame.wav"
    ENDGAME = "assets/sounds/NoGui_Endgame.wav"

SOUNDS: Dict[str, pygame.mixer.Sound] = {}
"""Dictionary mapping sound effect names to Pygame Sound objects."""

def initialize_sounds() -> None:
    """Initialize sound effects and populate the SOUNDS dictionary.

    Args:
        None
    """
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    for sound in Sound:
        try:
            SOUNDS[sound.name] = pygame.mixer.Sound(str(Path(sound.value)))
        except FileNotFoundError:
            SOUNDS[sound.name] = None
            if MG:
                with open(FILENAME_MG, "a") as f:
                    f.write(f"gui_constants.py | initialize_sounds | Missing sound file: {sound.value}\n")

class Image(Enum):
    """Enumeration for image asset paths and sizes."""
   
    DICE_1 = ("assets/images/1.png", (75, 75))
    DICE_2 = ("assets/images/2.png", (75, 75))
    DICE_2B = ("assets/images/2b.png", (75, 75))
    DICE_3 = ("assets/images/3.png", (75, 75))
    DICE_3B = ("assets/images/3b.png", (75, 75))
    DICE_4 = ("assets/images/4.png", (75, 75))
    DICE_5 = ("assets/images/5.png", (75, 75))
    DICE_6 = ("assets/images/6.png", (75, 75))
    DICE_6B = ("assets/images/6b.png", (75, 75))
    MOUNTAIN = ("assets/images/Mountain3.png", [(40, 40), (20, 20)])
    FIELD = ("assets/images/Field2.png", [(40, 40), (20, 20)])
    FOREST = ("assets/images/Woods.png", [(40, 40), (20, 20)])
    HILL = ("assets/images/Hills3.png", [(40, 40), (20, 20)])
    PASTURE = ("assets/images/Grass.png", [(40, 40), (20, 20)])
    DESERT = ("assets/images/Desert.png", (40, 40))
    SEA = ("assets/images/Sea3.png", (40, 40))
    PLUS = ("assets/images/Plus.png", (50, 50))
    MIN = ("assets/images/Min.png", (50, 50))
    FINISH = ("assets/images/Finish2.png", (50, 50))
    ROBBER = ("assets/images/Robber.png", (40, 40))
    SETTINGS_ON = ("assets/images/Settings3.png", (40, 40))
    SETTINGS_OFF = ("assets/images/Settings.png", (40, 40))
    QUESTIONMARK = ("assets/images/Questionmark.png", (40, 40))
    DC_VPOINT = ("assets/images/vp.jpg", [(20, 20), (30, 30), (40, 40)])
    DC_KNIGHT = ("assets/images/knight.png", [(20, 20), (30, 30), (40, 40)])
    DC_ROAD = ("assets/images/road.jpg", [(20, 20), (30, 30), (40, 40)])
    DC_PLENTY = ("assets/images/plenty2.png", [(20, 20), (30, 30), (40, 40)])
    DC_MONOPOLY = ("assets/images/monopoly.png", [(20, 20), (30, 30), (40, 40)])
    OKY = ("assets/images/OK.png", (40, 40))
    OKN = ("assets/images/OK_pale.png", (40, 40))
    NOK = ("assets/images/NOK.png", (40, 40))
    CITY_BLUE = ("assets/images/CityBlue4s.png", [(30, 30), (40, 40)])
    SETTLEMENT_BLUE = ("assets/images/BarnBlue5s.png", [(30, 30), (40, 40)])
    CITY_RED = ("assets/images/CityRed4s.png", [(30, 30), (40, 40)])
    SETTLEMENT_RED = ("assets/images/BarnRed4s.png", [(30, 30), (40, 40)])
    CITY_WHITE = ("assets/images/CityWhite3s.png", [(30, 30), (40, 40)])
    SETTLEMENT_WHITE = ("assets/images/BarnWhite2s.png", [(30, 30), (40, 40)])
    CITY_ORANGE = ("assets/images/CityOrange4s.png", [(30, 30), (40, 40)])
    SETTLEMENT_ORANGE = ("assets/images/BarnOrange4s.png", [(30, 30), (40, 40)])
    CITY_GREEN = ("assets/images/CityGreen1s.png", [(40, 40), (30, 30)])
    SETTLEMENT_GREEN = ("assets/images/BarnGreen2s.png", [(40, 40), (30, 30)])
    CITY_DGRAY = ("assets/images/CityDGray1s.png", [(40, 40), (30, 30)])
    SETTLEMENT_DGRAY = ("assets/images/BarnDGray2s.png", [(40, 40), (30, 30)])
    ROAD_GREEN = ("assets/images/RoadGreen.png", [(40, 40), (30, 30)])
    ROAD_DGRAY = ("assets/images/RoadGray.png", [(40, 40), (30, 30)])
    DCARD_GREEN = ("assets/images/DCardGreen.png", [(40, 40), (30, 30)])
    DCARD_DGRAY = ("assets/images/DCardGray.png", [(40, 40), (30, 30)])
    TWP_GREEN = ("assets/images/TwPGreen.png", (40, 40))
    TWP_RED = ("assets/images/TwPRed.png", (40, 40))

IMAGES: Dict[str, Dict[str, pygame.Surface]] = {}
"""Dictionary mapping image asset names to dictionaries of size keys (e.g., '40x40') and scaled Pygame Surfaces."""
for img in Image:
    try:
        path, sizes = img.value
        if isinstance(sizes, list):
            IMAGES[img.name] = {
                f"{size[0]}x{size[1]}": pygame.transform.scale(pygame.image.load(str(Path(path))), size)
                for size in sizes
            }
        else:
            IMAGES[img.name] = {"default": pygame.transform.scale(pygame.image.load(str(Path(path))), sizes)}
    except FileNotFoundError:
        IMAGES[img.name] = {f"{size[0]}x{size[1]}": None for size in sizes} if isinstance(sizes, list) else {"default": None}
        if MG:
            with open(FILENAME_MG, "a") as f:
                f.write(f"gui_constants.py | IMAGES | Missing image file: {path}\n")

WIN = pygame.display.set_mode((1225, 800))
"""Pygame display surface for rendering the game window."""

BOARD_OFFSET: Tuple[int, int] = (230, 50)
"""Tuple specifying the (x, y) offset for board elements."""

PANEL_OFFSET: Tuple[int, int] = (870, 240)
"""Tuple specifying the (x, y) offset for panel elements."""

PANEL_OFFSET_Y2: int = 900
"""Y-offset for secondary panel elements."""

POSITIONS: Dict[str, Union[Dict[int, Tuple[int, int]], Tuple[int, int, int, int]]] = {
    "intersections": {
        3: [280 + BOARD_OFFSET[0], 98],
        4: [320 + BOARD_OFFSET[0], 70],
        5: [360 + BOARD_OFFSET[0], 98],
        6: [400 + BOARD_OFFSET[0], 70],
        7: [440 + BOARD_OFFSET[0], 98],
        8: [480 + BOARD_OFFSET[0], 70],
        9: [520 + BOARD_OFFSET[0], 98],
        13: [240 + BOARD_OFFSET[0], 170],
        14: [280 + BOARD_OFFSET[0], 142],
        15: [320 + BOARD_OFFSET[0], 170],
        16: [360 + BOARD_OFFSET[0], 142],
        17: [400 + BOARD_OFFSET[0], 170],
        18: [440 + BOARD_OFFSET[0], 142],
        19: [480 + BOARD_OFFSET[0], 170],
        20: [520 + BOARD_OFFSET[0], 142],
        21: [560 + BOARD_OFFSET[0], 170],
        23: [200 + BOARD_OFFSET[0], 242],
        24: [240 + BOARD_OFFSET[0], 214],
        25: [280 + BOARD_OFFSET[0], 242],
        26: [320 + BOARD_OFFSET[0], 214],
        27: [360 + BOARD_OFFSET[0], 242],
        28: [400 + BOARD_OFFSET[0], 214],
        29: [440 + BOARD_OFFSET[0], 242],
        30: [480 + BOARD_OFFSET[0], 214],
        31: [520 + BOARD_OFFSET[0], 242],
        32: [560 + BOARD_OFFSET[0], 214],
        33: [600 + BOARD_OFFSET[0], 242],
        34: [200 + BOARD_OFFSET[0], 286],
        35: [240 + BOARD_OFFSET[0], 314],
        36: [280 + BOARD_OFFSET[0], 286],
        37: [320 + BOARD_OFFSET[0], 314],
        38: [360 + BOARD_OFFSET[0], 286],
        39: [400 + BOARD_OFFSET[0], 314],
        40: [440 + BOARD_OFFSET[0], 286],
        41: [480 + BOARD_OFFSET[0], 314],
        42: [520 + BOARD_OFFSET[0], 286],
        43: [560 + BOARD_OFFSET[0], 314],
        44: [600 + BOARD_OFFSET[0], 286],
        46: [240 + BOARD_OFFSET[0], 358],
        47: [280 + BOARD_OFFSET[0], 386],
        48: [320 + BOARD_OFFSET[0], 358],
        49: [360 + BOARD_OFFSET[0], 386],
        50: [400 + BOARD_OFFSET[0], 358],
        51: [440 + BOARD_OFFSET[0], 386],
        52: [480 + BOARD_OFFSET[0], 358],
        53: [520 + BOARD_OFFSET[0], 386],
        54: [560 + BOARD_OFFSET[0], 358],
        58: [280 + BOARD_OFFSET[0], 430],
        59: [320 + BOARD_OFFSET[0], 458],
        60: [360 + BOARD_OFFSET[0], 430],
        61: [400 + BOARD_OFFSET[0], 458],
        62: [440 + BOARD_OFFSET[0], 430],
        63: [480 + BOARD_OFFSET[0], 458],
        64: [520 + BOARD_OFFSET[0], 430],
    },
    "tiles": {
        2: [280 + BOARD_OFFSET[0], 48],
        3: [360 + BOARD_OFFSET[0], 48],
        4: [440 + BOARD_OFFSET[0], 48],
        5: [520 + BOARD_OFFSET[0], 48],
        8: [240 + BOARD_OFFSET[0], 120],
        9: [320 + BOARD_OFFSET[0], 120],
        10: [400 + BOARD_OFFSET[0], 120],
        11: [480 + BOARD_OFFSET[0], 120],
        12: [560 + BOARD_OFFSET[0], 120],
        14: [200 + BOARD_OFFSET[0], 192],
        15: [280 + BOARD_OFFSET[0], 192],
        16: [360 + BOARD_OFFSET[0], 192],
        17: [440 + BOARD_OFFSET[0], 192],
        18: [520 + BOARD_OFFSET[0], 192],
        19: [600 + BOARD_OFFSET[0], 192],
        20: [160 + BOARD_OFFSET[0], 264],
        21: [240 + BOARD_OFFSET[0], 264],
        22: [320 + BOARD_OFFSET[0], 264],
        23: [400 + BOARD_OFFSET[0], 264],
        24: [480 + BOARD_OFFSET[0], 264],
        25: [560 + BOARD_OFFSET[0], 264],
        26: [640 + BOARD_OFFSET[0], 264],
        27: [200 + BOARD_OFFSET[0], 336],
        28: [280 + BOARD_OFFSET[0], 336],
        29: [360 + BOARD_OFFSET[0], 336],
        30: [440 + BOARD_OFFSET[0], 336],
        31: [520 + BOARD_OFFSET[0], 336],
        32: [600 + BOARD_OFFSET[0], 336],
        34: [240 + BOARD_OFFSET[0], 408],
        35: [320 + BOARD_OFFSET[0], 408],
        36: [400 + BOARD_OFFSET[0], 408],
        37: [480 + BOARD_OFFSET[0], 408],
        38: [560 + BOARD_OFFSET[0], 408],
        41: [280 + BOARD_OFFSET[0], 480],
        42: [360 + BOARD_OFFSET[0], 480],
        43: [440 + BOARD_OFFSET[0], 480],
        44: [520 + BOARD_OFFSET[0], 480],
    },
    "buttons": {
        "buy_city": (140, 260, 40, 40),
        "buy_settlement": (190, 260, 40, 40),
        "buy_road": (240, 260, 40, 40),
        "buy_dcard": (290, 260, 40, 40),
        "twp": (200, 320, 60, 40),
        "twb": (270, 320, 60, 40),
        "roll_dices": (200, 400, 130, 40),
        "end_turn": (200, 470, 130, 40),
        "cancel": (200, 470, 130, 40),
        "next_turn2": (20, 470, 130, 40),
        "analysis": (1040, 668, 130, 40),
        "new_game": (900, 668, 130, 40),
        "quit": (1040, 668, 130, 40)
    },
    "panels": {
        "discard_rcards": (10 + PANEL_OFFSET[0], 205 + PANEL_OFFSET[1], 250, 340),
        "twp_panel": (10 + PANEL_OFFSET[0], 205 + PANEL_OFFSET[1], 296, 340),
        "hp_buttons": (10, 250, 330, 270),
    }
}
"""Dictionary mapping element types (intersections, tiles, buttons, panels) to their positions as (x, y) or (x, y, width, height) tuples."""