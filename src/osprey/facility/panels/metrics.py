"""Advance widths of the viewer's text font, for measuring text in a panel.

Written by ``scripts/panel_font_metrics.py`` from IBM Plex Sans, Version 3.005; do not edit.
Each width is in thousandths of an em; a character not listed is measured as
:data:`UNLISTED`, the widest listed one, so an estimate never falls short.
"""

from __future__ import annotations

#: The family the viewer draws a panel's text in.
FONT = "IBM Plex Sans"

#: The unit of :data:`WIDTHS`: thousandths of an em.
PER_EM = 1000

#: Advance width per character, in thousandths of an em.
WIDTHS: dict[str, int] = {
    " ": 236,
    "!": 284,
    '"': 419,
    "#": 713,
    "$": 598,
    "%": 927,
    "&": 694,
    "'": 242,
    "(": 335,
    ")": 335,
    "*": 450,
    "+": 600,
    ",": 272,
    "-": 399,
    ".": 272,
    "/": 383,
    "0": 600,
    "1": 600,
    "2": 600,
    "3": 600,
    "4": 600,
    "5": 600,
    "6": 600,
    "7": 600,
    "8": 600,
    "9": 600,
    ":": 292,
    ";": 292,
    "<": 600,
    "=": 600,
    ">": 600,
    "?": 477,
    "@": 891,
    "A": 641,
    "B": 653,
    "C": 621,
    "D": 671,
    "E": 583,
    "F": 559,
    "G": 695,
    "H": 707,
    "I": 400,
    "J": 510,
    "K": 634,
    "L": 501,
    "M": 812,
    "N": 707,
    "O": 708,
    "P": 606,
    "Q": 708,
    "R": 640,
    "S": 581,
    "T": 572,
    "U": 678,
    "V": 609,
    "W": 891,
    "X": 613,
    "Y": 593,
    "Z": 580,
    "[": 317,
    "\\": 383,
    "]": 317,
    "^": 600,
    "_": 565,
    "`": 600,
    "a": 534,
    "b": 580,
    "c": 503,
    "d": 580,
    "e": 549,
    "f": 324,
    "g": 528,
    "h": 568,
    "i": 250,
    "j": 250,
    "k": 527,
    "l": 272,
    "m": 873,
    "n": 568,
    "o": 560,
    "p": 580,
    "q": 580,
    "r": 367,
    "s": 487,
    "t": 351,
    "u": 568,
    "v": 492,
    "w": 768,
    "x": 507,
    "y": 499,
    "z": 464,
    "{": 343,
    "|": 314,
    "}": 343,
    "~": 600,
    "—": 780,
    "–": 588,
    "·": 326,
    "›": 304,
    "≈": 600,
    "≠": 600,
    "…": 803,
    "→": 820,
    "←": 820,
    "↑": 820,
    "↓": 820,
    "Σ": 569,
    "±": 600,
    "°": 468,
    "µ": 573,
    "×": 600,
    "Δ": 643,
    "²": 346,
    "³": 346,
    "Ω": 716,
}

#: The width of a character :data:`WIDTHS` does not list.
UNLISTED = max(WIDTHS.values())

#: The height of one line of text (the browser's ``line-height: normal``), in
#: thousandths of an em: a label's box lower than this the viewer shrinks.
LINE = 1300

#: The baseline under the top of one line of text, in thousandths of an em: a
#: text centred in its box has its baseline ``ASCENT - LINE / 2`` below the middle.
ASCENT = 1025
