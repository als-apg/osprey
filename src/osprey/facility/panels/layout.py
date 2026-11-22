"""Layout: a view description drawn as a Canopy Screen-IR 1.1 document.

This is the only module of the panels with geometry. :func:`render` walks a
:class:`~osprey.facility.panels.view.View` top to bottom, one section after
another, and draws every item with the pinned viewer's widgets. Colours are
the theme roles of :mod:`~osprey.facility.panels.symbols`.

States are the viewer's to compute. A light's rule over its channels, written from
their health rows (see :func:`ladder`), is a list of arms, each a condition that
holds when the light is in that arm's state; each prop of the light has a rule
(:func:`light_rule`) over the light's channels whose conditions are those arms,
first that holds wins, and set the prop to the state's look. The viewer
shows the state the rows give the live value, and ``unknown`` before a value
arrives. Nothing here reads a value.

Where a view needs a widget the format lacks, the nearest one is drawn and the
node says so in its ``meta.comment`` (``format-request: <name>``); the needs
are :data:`FORMAT_REQUESTS`, and :func:`needs` lists those a document hits. A
light whose rule the format cannot write is drawn ``unknown`` and counted
(:func:`unexpressed`).

Every size, spacing and font size is in the constants block below.
"""

from __future__ import annotations

import contextlib
import contextvars
import itertools
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import Any

from osprey.facility.panels import health, metrics, strip, symbols
from osprey.facility.panels import picture as drawing
from osprey.facility.panels.describe import (
    ENUMERATED_OUTPUTS,
    NO_POSITION,
    PROBE_SECTION,
    RELATION_TITLES,
    WITHHELD,
    action_word,
    press_words,
)
from osprey.facility.panels.describe import light as describe_light
from osprey.facility.panels.health import State
from osprey.facility.panels.model import Citation, PanelGraph, id_tail
from osprey.facility.panels.strip import Placed
from osprey.facility.panels.symbols import (
    BOX_BORDER,
    FIELD_BACKGROUND,
    MUTED_TEXT,
    PAGE_BACKGROUND,
    PAGE_TEXT,
    PANEL_BACKGROUND,
    TRANSPARENT,
)
from osprey.facility.panels.tables import PicturePart
from osprey.facility.panels.view import (
    ENTRIES_IN_ONE_COLUMN,
    Band,
    BlockRow,
    CalibrationRow,
    Calibrations,
    CalibrationTallies,
    CardFoot,
    CardHead,
    CardStrip,
    Chain,
    ChainBlock,
    ChannelCount,
    ChannelGrid,
    CheckRow,
    Connector,
    Control,
    ControlGroup,
    CurvePlot,
    Elevation,
    Fact,
    FarEnd,
    Front,
    FrontGroup,
    Grid,
    GridCell,
    GridRow,
    Group,
    KeyTile,
    Light,
    LimitLadder,
    Link,
    Node,
    Operate,
    PartFacts,
    Picture,
    Placement,
    Probe,
    ProbePart,
    RackUnit,
    RingMember,
    RingStrip,
    Section,
    Strip,
    StripElement,
    Target,
    Tile,
    Totals,
    ValueRow,
    View,
)

__all__ = [
    "FORMAT_REQUESTS",
    "MIN_STRIP_SCALE",
    "OUTLINE_KIND",
    "compiled_rules",
    "light_rule",
    "metric_formula",
    "WIDGETS",
    "Ladder",
    "ladder",
    "aggregates",
    "needs",
    "render",
    "picture_labels",
    "engine_expression",
    "engine_length",
    "stem_levels",
    "strip_labels",
    "text_width",
    "unexpressed",
]

# --- constants: every size, spacing and font size of a panel screen -----------
PAGE_WIDTH = 1200
MARGIN = 32  # the left edge of every section's content
# text sizes: body 13, small 11 and nothing smaller anywhere (a text-label below
# SMALL_SIZE is raised to it), headings small and muted, the page title 20
TITLE_SIZE = 20
HEADING_SIZE = 11
TEXT_SIZE = 13
SMALL_SIZE = 11
CHAR_WIDTH = 0.6  # width of a digit, in parts of the font size: a number's estimate
TEXT_PAD = 10  # added to a measured text width: the label's own padding and a margin
LABEL_INSET = 1  # the viewer's own padding either side of a label's text
HEADER_HEIGHT = 40  # the bar at the top of every screen
HEADER_PAD = 8  # the bar's widgets from its top
PANEL_PAD = 16  # a section's frame around its content: the frame stands in the margin
PANEL_GAP = 12  # between two section frames
ROW = 24  # height of one row of text, a field or a button
ROW_GAP = 4
SECTION_GAP = 16
STAMP_DIGITS = 8  # of the model's content hash, in the header
GAP = 8  # between two widgets on one row
MARK_GAP = 2  # between a title and the reference mark after it
INDENT = 20  # one level of a roll-up tree
LABEL_WIDTH = 200  # a row's label column, however short its section's labels
LABEL_MOST = 360  # a row's label column, however long its section's labels
VALUE_WIDTH = 130
UNIT_WIDTH = 44  # a value's unit, beside it
UNIT_GAP = 2
BUTTON_PAD = 16
MAX_BUTTON_WIDTH = 420
LIGHT_SIZE = 14
MARK_WIDTH = 16
LINE_WIDTH = 1
AXIS_WIDTH = 2
CELL_WIDTH = 110  # a checkout's column
CHECK_FLAGS = 4  # a checkout's flag columns after its two names
NAME_WIDTH = 260  # a checkout's channel and thing columns, however short their names
NAME_LEAST = 160  # a checkout's thing column, when the channel names take the rest
# a card's calibration table: each column's left edge from the content's, and the
# basis chip's width
CAL_COLUMNS = (0, 320, 560, 740, 830, 970)
CAL_CHIP_WIDTH = 170
CAL_CONDITION_LINES = 2  # the most lines a calibration's conditions take
CAL_CELL_LINES = 3  # the most lines a calibration's date, source or history takes
TITLE_LINE = 18  # each further line of a wrapped title at text size
SIGNAL_NAME_LINES = 2  # a signal row's name wraps to this many lines before it is cut
CHIP_INSET = 2  # a basis chip stands this far inside its row, above and below
SET_CELL_WIDTH = 112  # one member of a set overview at the least: its value or name, a light
SET_CELL_MAX = 220  # the widest a cell grows to when the columns are few
SET_ROW_HEAD = 64  # a set overview's row heads: the place, short
SET_BLOCK_GAP = 24  # between two blocks of a set overview's columns
SET_MEMBERS = 200  # a set overview draws blocks until this many cells are drawn
STRIP_WIDTH = PAGE_WIDTH - 2 * MARGIN
MIN_STRIP_SCALE = 0.6  # below this a strip breaks into rows: a bend stays >= 45 px tall
SYMBOL_UNIT = 20  # one shape unit of a symbol, in px, at scale 1: a bend is 75 px tall
STRIP_AXIS = 38  # the axis's room each side: half the tallest symbol, a bend
BADGE = 20  # a device off the axis: its circle
STEM_LOW = 52  # from the axis to the centre of a badge, first level
STEM_HIGH = 84  # second level, for a badge that would touch its neighbour
BADGE_GAP = 2  # between two badges of one level
STRIP_LABEL_TOP = STRIP_AXIS + 18  # a symbol's name, below the axis, with air between
STRIP_LABEL_STEP = 18  # the second line of names, further down
STRIP_LABEL_SIZE = SMALL_SIZE
STRIP_LABEL_WIDTH = 96  # the most a symbol's name may take along the strip
STRIP_LABEL_GAP = 6  # between two names on one line
STRIP_BELOW = STRIP_LABEL_TOP + 2 * STRIP_LABEL_STEP  # room under the axis
LEADER_GAP = 8  # between the names and the leaders
STRIP_VALUE_LANES = 2  # the most lanes of value blocks under a strip
LEADER_DROP = 40  # the leaders' run down to the locator
LOCATOR_PAD = 0.03  # a row's locator reaches this share of its stretch past each end
LOCATOR_LEAST_M = 1.0  # a row's locator spans at least this, m
LOCATOR_BAR = 1  # the to-scale bar: a line
LOCATOR_MARK = 8  # how far a device's mark reaches above the bar
LOCATOR_BIG = 12  # the mark of a magnet
TICK = 5
TICK_LABEL_WIDTH = 48
MAX_TICKS = 16
TICK_STEPS = (1, 2, 5)
TICK_BASE = 10
LEGEND_SYMBOL = 12  # one shape unit of a symbol in the legend: 60 % of the strip's
LEGEND_GAP = 20  # between two entries of the legend
TRAY_PAD = 10
CHIP_HEIGHT = 26
CHIP_GAP = 8
CHIP_LIGHT = 12
LEGEND_ROW = 52
BIG_SIZE = 22  # a key number
TILE_PAD = 10
# -- a drawn card: its title line, key tiles, chain blocks, controls and channel grid
CARD_NAME_SIZE = 22
CARD_NAME_LINE = 34
CARD_KIND_SIZE = 13
CARD_DESCRIPTION_SIZE = 12
KEY_TILE_HEIGHT = 76
KEY_TILES_WIDE = 4  # a row of fewer tiles keeps the width of this many
KEY_TILE_GAP = 10
KEY_VALUE_SIZE = 20
KEY_VALUE_LINE = 34
KEY_WORD_SIZE = 18
BLOCK_LEAST, BLOCK_MOST = 200, 270  # a chain block's width
BLOCK_GAP = 72  # between two blocks: the arrow and its word
BLOCK_TITLE = 20  # a block's title line, over its box
BLOCK_TEXT_SIZE = 12
BLOCK_ROW_TOP = 12  # a block's first line under the box's top
BLOCK_LINE = 1.5  # a block's outline, and an arrow's
ARROW_INSET = 4  # an arrow stops short of the next block
WORD_GAP = 24  # between a name and the muted words after it
DESCRIPTION_LINE = 19
KEY_VALUE_TOP = 30  # a key tile's value under its top
KEY_NOTE_MARGIN = 4  # under the line below a key figure, above the tile's border
CHIP_HEIGHT_SMALL = 20  # the dashed "not served yet" chip
BLOCK_ROW = 26
BLOCK_TOP = 30  # a block's box above its first line
BLOCK_LABEL = 100  # a curve's or a control's label column
BLOCK_LABEL_LEAST = 48  # a block line's label column, however short its labels
BLOCKS_PER_ROW = 5
#: A row of blocks too wide for the card narrows to fit: its arrows to this run, its
#: blocks in proportion down to this width; only then does the chain wrap.
BLOCK_GAP_LEAST = 48
BLOCK_NARROWEST = 168
TITLE_LINES = 2  # a chain block's title, an arrow's word: at most this many lines
ARROW_RUN_MOST = 112  # the longest arrow between two blocks, for a long word
# a card's front: connectors at a fixed pitch on a panel body, what is on each under it
FRONT_PITCH_LEAST, FRONT_PITCH_MOST = 60, 150
FRONT_GROUP_GAP = 24  # between two groups of a face
SOCKET_SIZE = 26
SOCKET_INSET = 7  # the socket's opening inside its outline
FRONT_BODY_PAD = 10  # the panel body round its connectors
FRONT_CABLE = 14  # the cable from the body to what is on the connector
FAR_BOX_HEIGHT = 34  # a far end's box: two small lines
FRONT_EMPTY_WIDTH = 360  # a group with no connector stated: its body with the note
FRONT_LABEL_GUTTER = 170  # the row labels left of a front's values and presses
LOOSE_COLUMNS = 4  # things on no stated port, in columns
RING_COLUMN_LEAST, RING_COLUMN_MOST = 72, 140  # a ring-strip member's column
LADDER_DOT = 14  # the live value on a ladder
FOCUS_RING = 6  # round a part in focus on its owner's drawing
CURVE_WIDTH, CURVE_HEIGHT = 560, 240  # a curve's plot
CURVE_POINT = 6  # a measured point on it
LADDER_TICK = 18  # a set point's mark across the ladder's line
CONTROL_WIDTH = 160
CONTROL_HEIGHT = 28
CONTROL_TAG_HEIGHT = 18
OPERATE_COLUMNS = 3  # press groups side by side; an entry group takes the width
GRID_ROW = ROW + ROW_GAP
GRID_VALUE = 100
GRID_NAME_LEAST = 80  # a grid row's name, however wide its value and unit
PAGE_END = PANEL_GAP + SECTION_GAP  # the canvas ends this far under its lowest frame
# a card's strip: the pipe, the lanes above and below it, the metre ticks
CARD_PIPE = 14
CARD_LANE = 44  # a lane's symbols from the pipe's axis
CARD_SYMBOL_UNIT = 16  # px per symbol shape unit
CARD_DIPOLE_M = 0.8  # a dipole's drawn length when none is stated, m
CARD_DIPOLE_HEIGHT = 3 * CARD_PIPE  # a dipole's box round the pipe
CARD_TICK = 6
CARD_STRIP_VALUE = 90
CARD_TICK_LEAST = 40  # the least px between two labelled ticks
CARD_LEAST_SPAN = 1e-9  # m: a window of no length is drawn as this
HALF_PIXEL = 0.5  # a coordinate this far past a pixel rounds up to the next
CARD_ABOVE_ROOM = 2 * ROW + CARD_LANE  # over the pipe: a lane, a name, a value
CARD_BELOW_ROOM = 3 * ROW + CARD_LANE  # under it: a lane, a name, a value
_STRIP_TRIES = 200  # lines a strip's name or value moves out, at most, to clear the others
_DIPOLE_DRAWN = "dipoles drawn {:g} m long"
_HERE = "{} · here"
_NOWHERE = "where it sits is not stated"
_BY_WHOLE = "in {} by its whole {}"
#: The kinds of control, as the operate block's legend names and explains them.
CONTROL_LEGEND: tuple[tuple[str, str], ...] = (
    ("ACTION", "writes once, when pressed"),
    ("CHOICE", "stays; shaded = the device's status"),
    ("ENTRY", "type, check, Apply"),
    ("READ ONLY", "nothing to press"),
)
_READ_ONLY_TAG = "READ ONLY"
_NOT_SERVED = "not served yet"
#: The drawn card's closing sections, drawn as muted lines.
_FACT_SECTIONS = frozenset({"where", "sources"})
_NOT_STATED_TEXT = "not stated"
_WHERE_WORD = "Where:"
_HIDDEN = "{} driver-internal records not shown"
_MORE = "\u203a"
_FOOT_JOIN = " · "
_OR = "or "
_SOURCES_WORD = "Sources:"
_SETS_WORD = "Sets:"
_MEANING_NOT_STATED = "meaning not stated"
#: The widest number a value of three significant digits prints, for sizing its field.
_WIDEST_NUMBER = "-0.000"
_GRID_NOTE = "hover a row for the full channel name"
_APPLY = "Apply"
_TYPE_FIRST = "type a value first"
_WILL_WRITE = "will write"
#: An entry's Apply button's press value until a number is entered, and its rule's
#: default: text, never a number, so no state of the button writes a number nobody
#: typed. The button is off the canvas in that state (see AWAY).
NOTHING_ENTERED = "nothing entered"
#: The rule condition that holds only for a finite number: text fails the product
#: (the viewer's evaluator refuses text in arithmetic and gives no output), an
#: infinity gives not-a-number, which equals nothing.
_ENTERED = "{0} * 0 == 0"
_NOW = "now"
_VALUE_NOT_STATED = "value not stated"
MIN_TILE_WIDTH = 150  # the band takes the fewest rows whose tiles are this wide
LINE_HEIGHT = 1.5  # a line of text, in parts of its size: the viewer shrinks what overflows
NUMBER_HEIGHT = 36  # a key number's box: one line at the big size, and its border
NUMBER_PAD = 22  # a value display's own padding and border, across
SMALL_LINE = 16  # one line of small text
TILE_LINES = 2  # the most lines a tile's title or caption takes
VALUE_DIGITS = 7  # room for a value's number: "2.6e-10"
COUNT_DIGITS = 3  # the least room for a count's number: what a dash takes
SIGNIFICANT_DIGITS = 3  # a live float prints with this many significant digits
PRECISION_FALLBACK = 3  # its decimals until its precision rule delivers, or with none
FIXED_SMALLEST = 0.001  # below this a value keeps the viewer's own format (an exponent)
COMPACT_SMALLEST = 0.1  # the compact rule prints smaller values as it prints this one
VALUE_SAMPLE = "-0.00123"  # the widest number three significant digits print below 10 000
#: The most characters of a unit Channel Access carries: its units field is 8 bytes, the
#: last a NUL. A longer unit arrives cut ("1/Meter^2" as "1/Meter"), and a pv-label
#: appends what arrives, a false unit; such a channel's number is a static label a
#: formula sets, with the facility's whole unit beside it. That route goes when the
#: format takes the unit from the screen (format request (e)).
CHANNEL_ACCESS_UNIT_CHARS = 7
RING_PAD = 6  # a state ring's distance from its symbol
TAG_HEIGHT = 16  # the word above a ring
VALUE_STEP = 16  # one live value under a symbol's name
BADGE_VALUE_WIDTH = 64  # a value beside a badge
TAG_WIDTH = 64  # room for the longest word a ring carries, at the small size
CARD_LEFT = 600  # a device card's left column, its signals, with the gap to the right one
COLUMN_GAP = 24
CARD_LABEL_WIDTH = 240  # a signal's name with its unit, in the card's table
CARD_ROWS = 40  # signal rows a card draws; the rest are counted by how they are drawn
CARD_SPLIT_ROWS = 12  # a signals or settings group longer than this goes in columns
CARD_COLUMN_MIN = 360  # the narrowest such column
CARD_COMPACT_VALUE = 76  # a value's width in such a column: the name keeps the room
CARD_LABEL_MIN = CARD_COLUMN_MIN - 2 * (CARD_COMPACT_VALUE + GAP) - LIGHT_SIZE - GAP  # a name there
CHAIN_BOX = 150  # one thing of a card's fed-by chain
CHAIN_HEIGHT = 44
CHAIN_ARROW = 40  # the run of an arrow between two boxes of the chain
ROLLUP_LEFT = 700  # a roll-up's tiles, with the gap to its by-system rows
ROLLUP_TILE_WIDTH = 150  # the least width of a child's tile
ROLLUP_TILE_HEIGHT = 64
RING_OUTER = 185  # a ring roll-up's arcs: outer and inner radius
RING_INNER = 120
RING_GAP = 2  # degrees left free at each end of an arc
TURN = 360  # degrees round the ring
RING_MARGIN = 16  # above and below a ring roll-up's ring
RING_LABEL_WIDTH = 60  # an arc's name and mark, centred on the arc
RING_MARK = 12  # the font size of an arc's mark, under its name
SUMMARY_ROW = 36  # one row of a by-system summary
SUMMARY_COLUMNS = 2
PICTURE_SIZE = 544  # a drawing's canvas at the most; less where its column is narrower
PICTURE_PAD = 16  # between the canvas and its label columns
PICTURE_INSET = 8  # between the drawing's reach and the canvas's edge
PART_LABEL_WIDTH = 128  # a part's name, in a column on its side
PART_VALUE_WIDTH = 64  # a part's number beside it
BEAM_POINT = 4  # the beam, at the bore's centre; half a beam-axis cross
COIL_NUMBER_SIZE = 15  # a coil's number, on the yoke ring
BODY_LINE = 1.5  # the outline of a yoke, a coil, a block
POLYGON_BOX = 24  # a drawing polygon's box at the least: a thin band is drawn in a taller box
COIL_WARM_WIDTH = 3  # a coil's outline when its state is warn or bad
PART_DOT = 10  # a part on the body
PART_BLOCK = (16, 11)  # a part off the body: width, height
PART_BLOCK_CORNER = 2
PART_WORD_ROOM = 16  # above the topmost block, for its kind's word
PART_GROUP_DISTANCE = 56  # blocks of one kind this close share one word, px
ARRANGED_BLOCK = 168  # a part's block in the row drawn when no part has a place
RING_PLACE_HEAD = 120  # a ring-strip's place name, heading its row
ENTRY_COLUMNS = 3  # entries of one role past ENTRIES_IN_ONE_COLUMN, side by side
ENTRY_CELL_ROWS = 3  # an entry in a grid: name and value, field and Apply, what it writes
#: The kinds of part drawn as water (a block in the water colour): a kind naming one.
WATER_WORDS = ("manifold", "water", "cooling")
SCALE_BAR_MM = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
SCALE_BAR_SHARE = 0.34  # of the canvas a scale bar takes at the most
PICTURE_SOURCES = 3  # sources a drawing's legend names; the rest are counted
RACK_U = 16  # one rack unit: a 48 U rack is 768 px tall
RACK_WIDTH = 420  # a chassis bar, the rack's inner width
RACK_LABEL_WIDTH = 40  # the unit numbers, left of the rack
RACK_PAD = 6  # the rack's frame round its chassis
RACK_SOURCES = 8  # documents a rack names; the rest are counted
RACK_LANE_LEAST = 120  # the narrowest bar a name is written in; narrower ones are listed
RACK_COLUMN = RACK_LABEL_WIDTH + RACK_WIDTH + 2 * RACK_PAD + 2 * COLUMN_GAP  # the facts' x
# The longest expression the formula engine takes. The viewer compiles each rule
# of a light into one expression, with the light's screen formula inlined once
# per condition (see engine_expression); the engine refuses one longer than
# ``max_expression_len`` (canopy/formula_engine/config.py at 166989aa: 2048).
RULE_EXPRESSION_LIMIT = 2048
TOOLTIP_LINES = 24  # a card's sources line shows this many statements on hover

#: The props a rule may set, per widget type: those the viewer honours under a
#: rule. Read from the viewer runtime at commit 4038c44: its renderer handles
#: ``visible`` (false hides) and the geometry for every node; any other target is
#: handed to the widget, which reads only the props listed, and an unread one is
#: dropped without a word. A type not listed takes only the common ones.
RULE_TARGETS_EVERY = frozenset({"visible", "x", "y", "width", "height"})
RULE_TARGETS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "text-label": frozenset(
            {
                "text",
                "foregroundColor",
                "backgroundColor",
                "fontSize",
                "fontFamily",
                "align",
                "fitText",
            }
        ),
        "pv-label": frozenset(
            {
                "pv",
                "text",
                "foregroundColor",
                "backgroundColor",
                "fontSize",
                "fontFamily",
                "align",
                "fitText",
                "precision",
                "precisionFromPV",
                "showUnits",
                "alarmSensitive",
                "alarmBorder",
            }
        ),  # fmt: skip
        "pv-button": frozenset(
            {
                "invisible",
                "label",
                "onLabel",
                "offLabel",
                "labelType",
                "foregroundColor",
                "backgroundColor",
                "onColor",
                "offColor",
                "fontSize",
                "fontFamily",
                "buttonType",
                "pressValue",
                "releaseValue",
                "transparent",
                "onImage",
                "offImage",
                "pv",
                "readbackPV",
            }
        ),  # fmt: skip
        "pv-led": frozenset({"pv", "onColor", "offColor"}),
        "rectangle": frozenset(
            {"fill", "fillColor", "lineColor", "lineWidth", "lineStyle", "cornerRadius"}
        ),
        "ellipse": frozenset({"fill", "fillColor", "lineColor", "lineWidth", "lineStyle"}),
        "polygon": frozenset(
            {"fill", "fillColor", "lineColor", "lineWidth", "lineStyle", "closed", "points"}
        ),
        "line": frozenset(
            {"lineColor", "lineWidth", "lineStyle", "arrowStart", "arrowEnd", "arrowSize", "points"}
        ),
    }
)
TOOLTIP_CHANNELS = 8  # a light's tooltip names this many channels, else counts them
MOSAIC_LIGHT = 10  # one child's light in an aggregate's mosaic
MOSAIC_GAP = 3
#: A count's lamp while its member is counted: filled in the page's text colour,
#: the strongest fill a tile holds that is no alarm, so "k of N" reads in the lit
#: lamps; a zero is the outline of no state, no value the dashed unknown.
COUNTED_LAMP: Mapping[str, Any] = MappingProxyType(
    {**symbols.look("normal"), "fillColor": PAGE_TEXT, "lineColor": PAGE_TEXT}
)
MAX_TABLE_ROWS = 400  # rows of one table drawn; the rest are counted under it
PROBE_BORE = 240  # the bore's width across a probe picture, in px
PROBE_PAD = 32  # round the bore, for the buttons that stand out of it
PROBE_BOX = 96  # the magnified box the beam dot moves in, at the most
PROBE_BOX_FILL = 0.6  # of the bore's height, at the most
PROBE_DOT = 10
PROBE_LABEL_WIDTH = 184  # a button's letter, bar and value, in a column on its side
PROBE_BAR = (64, 8)  # a button's bar: width, height
PROBE_VALUE_WIDTH = 72
PROBE_READOUT_WIDTH = 96
WAVEFORM_WIDTH = 520
WAVEFORM_HEIGHT = 140
COMMAND_VALUE = 1  # what pressing a command's button writes
#: Where a widget waits, off the canvas, while a rule has not placed it: an entry's
#: Apply button while nothing is entered, a key figure's holder line while its
#: member does not hold the extreme. The viewer applies no rule before its channels
#: answer, so a widget is drawn in its rest state until then (no data, a lost
#: link): a ``visible`` rule whose default is false hides nothing in that state.
#: Parked here, a widget is out of sight with no rule applied, and a geometry rule
#: moves it in while its condition holds.
AWAY = -10_000
# What one screen may bind. The formula engine allows 1,000 subscriptions per
# client (``max_subscriptions_per_client``, canopy/formula_engine/config.py at
# 166989aa), and the viewer subscribes once per rule with a condition; a
# roll-up binding some 28,000 channels on one screen made the Channel Access
# backend report its virtual circuit unresponsive (seen live, 2026-11-17).
SCREEN_RULE_BUDGET = 800
SCREEN_CHANNEL_BUDGET = 4000
INTEGER_RUN = 3  # the fewest successive integer values a rule writes as one range test
BIT_TESTS = 6  # the most bits a channel's rule writes as a table of their values
# --- end constants -------------------------------------------------------------

#: The widget a value is drawn as, by its ``drawn_as`` in ``record_types.csv``:
#: (on a device card's own signals and settings, anywhere else). Writable widgets
#: are in the first column only, and only for a writable channel. A channel with
#: no row (or ``untyped``) is drawn as a ``setpoint`` when writable.
WIDGETS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "value": ("pv-label", "pv-label"),  # an analogue value, with its unit
        "setpoint": ("pv-text-input", "pv-label"),
        "state": ("pv-label", "pv-label"),  # an enum: the label's text is its state
        "command": ("pv-button", "pv-label"),
        "waveform": ("waveform-plot", "pv-label"),
        "text": ("pv-label", "pv-label"),
        "untyped": ("pv-label", "pv-label"),
    }
)
_WRITABLE_WIDGETS = frozenset({"pv-text-input", "pv-button"})

#: Widgets the format lacks, drawn with the nearest one: (name, why).
FORMAT_REQUESTS: tuple[tuple[str, str], ...] = (
    (
        "state light",
        "a light with five states (normal, warn, bad, unknown, disagree); pv-led has on and "
        "off, so a light is a polygon whose rules set fill, colours and line style",
    ),
    (
        "hatched fill",
        "disagree should read as hatched; no drawing widget fills with a hatch, so the light "
        "carries a mark",
    ),
    (
        "table",
        "a grid of a set's members and the checkout are tables; there is no table "
        "container that scrolls or pages, so every cell is placed on its own and a long "
        "table is cut, its remaining rows counted",
    ),
    (
        "clickable text",
        "neutral navigation: a link in a list or a tree reads as text that opens a screen; "
        "the format's navigation buttons take no colour props (related-display-button none, "
        "regular-button only flat), so a link is a text label under an outlined box whose "
        "action opens the screen: two widgets where one would do",
    ),
    (
        "label condition",
        "a health row that names a text label cannot be written as a rule condition: the "
        "viewer gives an enum channel's value as its index, and the label's index is not known",
    ),
    (
        "aggregate state",
        "a roll-up whose rule would be longer than the format's expression limit: the worst "
        "state over many channels, computed by the viewer or a server, drawn as one light; "
        "today it is an aggregate mark, drawn in parts where the screen's budget allows",
    ),
    (
        "server-side aggregate",
        "server-side aggregate over more channels than a screen can bind: a roll-up level "
        "whose summary needs more rules or channels than one screen's budget "
        "(SCREEN_RULE_BUDGET, SCREEN_CHANNEL_BUDGET); today the level says so in a line",
    ),
    (
        "long rule",
        "one channel whose health rows give a rule longer than the format's expression "
        "limit; its light is drawn unknown",
    ),
)
_REQUEST_NAMES = frozenset(name for name, _ in FORMAT_REQUESTS)

_REQUEST_TAG = "format-request: "
_UNKNOWN_TAG = "unknown: "
_AGGREGATE_ROW = "aggregate: row"
_AGGREGATE_TILE = "aggregate: tile"
_AGGREGATE_SYMBOL = "aggregate: symbol"
_AGGREGATE_TEXT = "{} channels, open for detail"
_SUMMARY_TEXT = "{} channels: a live summary of this level needs a server-side aggregate"
_PART = "part {} of {}"
_PICTURE = "picture"
#: The kind of a pictures row that is a point of the class's outline, not a part.
OUTLINE_KIND = "outline"
#: Kinds a drawing's legend does not list: the body's iron and its aperture.
_UNLISTED = frozenset({"yoke", "pole", "bore", "gap"})
_NOT_STATED = "outline not stated"
_NOT_PLACED = "not placed"
_ARRANGEMENT_NOT_STATED = "arrangement not stated"
_SIGNAL_NOT_STATED = "signal not stated"
_SIZE_NOT_STATED = "size not stated: default proportions · schematic, not to scale"
_SCALED_FROM = "scaled from the {}"
_SCHEMATIC = "{} at default proportions"
_BEAM_AXIS = "beam axis"
_ELLIPSIS = "\u2026"
#: While :func:`render` runs with ``cuts``: the list each label cut to fit is added to.
_CUTS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar("cuts", default=None)
_FORMULA_PREFIX = "formula://"
_NUMBERS = frozenset({"value", "setpoint", "untyped"})  # the kinds a label prints as a number
_WHOLE_KINDS = frozenset({"state", "command"})  # an enum, a two-valued output: printed whole
_PRINTED = _NUMBERS | _WHOLE_KINDS  # the kinds a formula can print: numbers
_WHOLE = re.compile(r"(?:==|!=)?\s*-?\d+")  # a health rule's value naming one state, or not it
_TOKEN = re.compile(r"\{(\d+)\}")
_NAVIGATION_TARGET = "tab"
_HOVER_LABEL = "hoverLabel"
_TOOLTIP = "tooltip"
_YES, _NO = "yes", "no"
_NO_RULE = "no rule"
_NO_UNIT = "no unit"
_ROWS_NOT_DRAWN = "rows not drawn: the screen holds a table of at most this many rows"
_CONTESTED = "contested"
_NOTHING = "\u2014"  # what a connector with nothing stated on it shows
_NO_CONNECTOR = "no connector stated for this unit"
_STATE_BAND = "state"
_STATED_WORD = "stated"
_CONTINUED = "{} (continued)"
_PRESS_BAND = "press"
_LADDER_NOW = "now"
_STATED = "STATED"
_NOT_STATED_HEAD = "NOT STATED"
_LADDER_NO_VALUE = "the value is not served: no dot"
_LADDER_LEFT_OUT = "set points not served, left out: {}"
_NOT_TO_SCALE = "schematic, not to scale"
_STRIP_TITLE = f"IN BEAM ORDER → · {_NOT_TO_SCALE}"
_LOOKS_LEGEND = (
    "fill = component type · red ring = fault · amber ring = warning · dashed = no health rule"
)
_LOCATOR_CAPTION = "to scale · position along the beam (m)"
_LOCATOR_STRETCH = " · this row {:.0f} to {:.0f} of {:.0f} to {:.0f}"
_NO_RULE_YET = "no health rule yet"
_TRAY_TITLE = "in our lists, not placed"
_NO_DATA = "no data yet"
_NEEDS_SERVER = "needs a server-side aggregate"
_A_MEMBER_MISSING = "one of {} has no value"
#: The line under a largest or smallest figure: of how many, then (where the tile
#: names it) the member that holds it.
_HOLDS = {"max": "highest of {}", "min": "lowest of {}"}
_NO_OK_STATED = "no ok state stated"
_INFERRED_WORD = "inferred"
_PER_PLACE = "counted per place"
_MANY_VALUES = "{} channels state it: the row names no one"
_DASH = "—"
NO_VALUE = _DASH  # an annotation's text until its formula delivers, and while disconnected
_COUNTS = frozenset({"count_true", "count_false", "count_in_tolerance"})
_VALUES = "values: "
_DROPPED = "{} not shown: the screen's budget is spent"
_CROWDED = "{} not shown: open the card"
_MORE_CHANNELS = "{} more {} channels not drawn"
_MORE_ROWS = "{} more not drawn"
_NOT_AT_A_UNIT = "not at a unit"
_SIDE_BY_SIDE = "side by side at one unit"
_SIDE_BY_SIDE_BAR = "{} devices side by side · listed under the rack"
_BY_CLASS = "by class"
_SOURCES = "sources"
_RACK_COUNTS = "{} at a stated unit · {} not at a unit"
_UNITS_FROM = "{} · {} devices"
_SET_READBACK = "each cell: readback of {}"
_CAL_HEADS = ("parameter", "value in force", "basis", "date", "source", "history")
_NO_VALUE_YET = "no value yet"
_CALIBRATION_SECTION = "calibration"
#: On a card with a probe picture, the sections drawn before the card's columns.
_PICTURE_FIRST = (PROBE_SECTION, _CALIBRATION_SECTION)
_NOT_STARTED = "{} kinds not started"
_MORE_CHIPS = "+{} more"
#: A tally's state as the look of a calibration cell's; at wanted: the wanted basis's.
_TALLY_LOOKS = {"contested": "contested", "none": "none", "below_wanted": "design"}
_INFERRED = "inferred"  # an anchor row's state: its placement is inferred
_PLACEMENT_INFERRED = "button placement inferred, not stated"
_CHIPPED = ("set", "place", "rollup")
_SETS_SECTION = "sets"  # describe.SETS_SECTION: the sets a place or a system holds  # views whose calibration section is chips
_CAL_WHAT = "each cell: {}, deviation from design (the value when no design value)"
#: The states of a calibration view, in the order its legend and counts give them.
_CAL_STATES = ("design", "bench", "survey", "beam", "contested", "none")
_STATE_WORDS = {
    "design": "design",
    "bench": "bench",
    "survey": "survey",
    "beam": "beam",
    "contested": "disagree",
    "none": "no value",
}
_LEGEND = {
    "design": "design only",
    "bench": "measured on the bench",
    "survey": "surveyed",
    "beam": "fitted with the beam",
    "contested": "two sources disagree",
    "none": "no value yet",
}
_SET_MORE = "{} more members: open by place"
_SET_NAMES = "each cell: the member's short name and its light"
_NO_CLASS = "no class stated"
_BY_SYSTEM_TITLE = "by system"
_TILE_OPEN = "{} channels · open"
_SIGNAL_HEADS = ("signal", "setpoint", "readback")
_CHAIN_CAPTION = "power flows right to left"
_THIS_DEVICE = "this device"
_FED_BY = "fed by"
_ALSO = "also"
_CARD_LEFT = frozenset({"signals", "settings", "health", "picture", "sources"})
_CARD_RIGHT = frozenset({"where"})
_READS = "{} rule channel{}"
_MAGNETS = frozenset(
    {"bend", "quadrupole", "quadrupole_f", "quadrupole_d", "sextupole", "skew_quadrupole"}
)
#: The word for each symbol in a strip's legend.
SYMBOL_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "bend": "bend",
        "quadrupole_f": "focusing quadrupole",
        "quadrupole_d": "defocusing quadrupole",
        "quadrupole": "quadrupole",
        "sextupole": "sextupole",
        "skew_quadrupole": "skew quadrupole",
        "corrector": "corrector",
        "bpm": "BPM",
        "neg_pump": "NEG pump",
        "ion_pump": "ion pump",
        "pump": "pump",
        "gauge": "gauge",
        "gate_valve": "gate valve",
        "bellows": "bellows",
        "kicker": "kicker",
        "scraper": "scraper",
        "generic": "other device",
    }
)
_STATE_ORDER: tuple[State, ...] = ("bad", "warn", "disagree")


def text_width(text: str, size: float, *, caps: bool = False) -> float:
    """A box wide enough for *text* at *size* px, measured in the viewer's font
    (:mod:`.metrics`); *caps*: written in capitals."""
    return math.ceil(_em(text.upper() if caps else text) * size / metrics.PER_EM) + TEXT_PAD


def line_height(size: float) -> float:
    """The height of one line of text at *size* px in the viewer's font
    (:data:`.metrics.LINE`): a label's box lower than this the viewer shrinks."""
    return math.ceil(size * metrics.LINE / metrics.PER_EM)


def baseline_drop(size: float) -> float:
    """How far below its box's middle the viewer draws the baseline of a line of
    text at *size* px: the viewer centres the line in the box, and the font puts
    the baseline :data:`.metrics.ASCENT` under the line's top."""
    return size * (metrics.ASCENT - metrics.LINE / 2) / metrics.PER_EM


def on_baseline(top: float, h: float, size: float, h_beside: float, size_beside: float) -> float:
    """The top of a box *h_beside* high whose text at *size_beside* px stands on the
    baseline of a text at *size* px in a box *h* high at *top*."""
    base = top + h / 2 + baseline_drop(size)
    return base - baseline_drop(size_beside) - h_beside / 2


def _line_room(g: Mapping[str, float], size: float) -> Mapping[str, float]:
    """*g* at least one line of *size* high, grown about its middle, so the
    viewer draws the text at the size the file states. It grows by an even
    number of pixels, so a whole box stays centred on whole pixels."""
    need = line_height(size)
    if g["height"] >= need:
        return g
    grow = math.ceil(need - g["height"])
    grow += grow % 2
    return {**g, "y": max(g["y"] - grow / 2, 0.0), "height": g["height"] + grow}


def check_columns(rows: Sequence[CheckRow]) -> tuple[float, float]:
    """A checkout table's channel and thing columns: each as wide as its longest
    name (at least :data:`NAME_WIDTH`), the :data:`CHECK_FLAGS` flag columns after them on the
    page. Where both do not fit, the channel names take the room the thing
    column does not need, down to :data:`NAME_LEAST` for the thing column."""
    room = STRIP_WIDTH - CHECK_FLAGS * CELL_WIDTH
    channel = max([NAME_WIDTH, *(text_width(r.channel, TEXT_SIZE) for r in rows)])
    thing = max([NAME_WIDTH, *(text_width(r.thing or "", TEXT_SIZE) for r in rows)])
    if channel + thing <= room:
        return channel, thing
    thing = max(min(thing, room - channel), NAME_LEAST)
    return room - thing, thing


def _em(text: str) -> int:
    """The advance of *text* in thousandths of an em."""
    return sum(metrics.WIDTHS.get(c, metrics.UNLISTED) for c in text)


# --------------------------------------------------------------------------- rules


@dataclass(frozen=True)
class Ladder:
    """A light's rule: the channels it binds and its arms, first match wins.

    Attributes:
        channels: The bound channels; ``{i}`` in an arm is ``channels[i]``.
        arms: ``(expression, state)`` in order; no arm matching is ``unknown``.
        unwritten: Channels whose rows no condition can say; they add no arm,
            and the light is never ``normal``.
    """

    channels: tuple[str, ...]
    arms: tuple[tuple[str, State], ...]
    unwritten: tuple[str, ...] = ()


def _number(value: float) -> str:
    return repr(value)


def _condition(row_form: health.Form, token: str) -> str | None:
    """A rule condition over *token* for one health row form; ``None`` when none can say it."""
    if row_form.kind == "compare":
        assert row_form.number is not None
        return f"{token} {row_form.op} {_number(row_form.number)}"
    if row_form.kind == "bit":
        assert row_form.bit is not None
        weight = 2**row_form.bit
        return f"{token} == {token} // 1 and ({token} // {weight}) % 2 == {row_form.wanted}"
    if row_form.number is None:
        return None
    return f"{token} == {_number(row_form.number)}"


def _any(conditions: Sequence[str]) -> str | None:
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return " or ".join(f"({c})" for c in conditions)


def _exclusive(a: health.Form, b: health.Form) -> bool:
    """Whether no value can match both forms: two literals of different numbers."""
    return (
        a.kind == b.kind == "literal"
        and a.number is not None
        and b.number is not None
        and a.number != b.number
    )


#: For each arm of a ladder, the states an earlier arm has already ruled out for every
#: channel: where a channel is in one of them, the arm may hold or not.
_RULED_OUT: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "bad": frozenset(),
        "warn": frozenset({"bad"}),
        "disagree": frozenset({"bad", "warn"}),
        "normal": frozenset({"bad", "warn", "disagree"}),
    }
)


def _shown(state: State, item: Light) -> State:
    """The state a light draws: the engine's at the value. Sources that disagree at
    another value do not colour this one: a value every row calls normal is
    normal, one rows give two states is ``disagree`` already. A light with rows
    the engine cannot evaluate is never normal (unknown)."""
    if state != "normal":
        return state
    return "unknown" if item.unsupported else state


def _interval(points: Sequence[float], first: int, last: int, token: str) -> str:
    """The test that a value is in segments *first* to *last* (see :func:`_channel_arms`).

    Segment ``2j + 1`` is the point ``points[j]``; segment ``2j`` is the open
    interval below it, and segment ``2 * len(points)`` the one above the last.
    """
    if first == last and first % 2:
        return f"{token} == {_number(points[first // 2])}"
    low: tuple[float, str] | None = None
    if first % 2:
        low = (points[first // 2], "<=")
    elif first:
        low = (points[first // 2 - 1], "<")
    high: tuple[float, str] | None = None
    if last % 2:
        high = (points[last // 2], "<=")
    elif last < 2 * len(points):
        high = (points[last // 2], "<")
    if low and high:
        return f"{_number(low[0])} {low[1]} {token} {high[1]} {_number(high[0])}"
    if low:
        return f"{token} {'>=' if low[1] == '<=' else '>'} {_number(low[0])}"
    if high:
        return f"{token} {high[1]} {_number(high[0])}"
    return f"{token} == {token}"  # every number; NaN is no number


def _cover(points: Sequence[float], marks: Sequence[str], token: str) -> str | None:
    """The shortest test true on every ``T`` segment and false on every ``F`` one.

    A ``D`` segment may go either way. Each maximal run of segments with no
    ``F`` that holds a ``T`` is one interval; a run of at least
    :data:`INTEGER_RUN` single successive integers is one integer range test.
    """
    runs: list[tuple[int, int]] = []
    start = None
    for k, mark in enumerate([*marks, "F"]):
        if mark != "F" and start is None:
            start = k
        elif mark == "F" and start is not None:
            if "T" in marks[start:k]:
                runs.append((start, k - 1))
            start = None
    terms: list[str] = []
    i = 0
    while i < len(runs):
        j = i
        while j + 1 < len(runs) and _successive(points, runs[j], runs[j + 1]):
            j += 1
        if j - i + 1 >= INTEGER_RUN:
            low, high = points[runs[i][0] // 2], points[runs[j][0] // 2]
            terms.append(
                f"{token} == {token} // 1 and {_number(low)} <= {token} <= {_number(high)}"
            )
        else:
            terms += [_interval(points, a, b, token) for a, b in runs[i : j + 1]]
        i = j + 1
    return " or ".join(terms) if terms else None


def _successive(points: Sequence[float], a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether runs *a* and *b* are single points at successive integers."""
    if a[0] != a[1] or b[0] != b[1] or not a[0] % 2 or not b[0] % 2:
        return False
    p, q = points[a[0] // 2], points[b[0] // 2]
    return p == int(p) and q == p + 1


def _bit_test(token: str, bit: int, wanted: int) -> str:
    shifted = token if bit == 0 else f"{token} // {2**bit}"
    return f"{shifted} % 2 == {wanted}"


def _bits_of(value: int, bits: Sequence[int]) -> tuple[int, ...]:
    return tuple((value >> bit) & 1 for bit in bits)


def _segment_values(
    points: Sequence[float], k: int, bits: Sequence[int]
) -> tuple[float | None, dict[tuple[int, ...], int]]:
    """A value of segment *k* that is no integer (``None`` when it has none) and, for
    each combination of *bits* that an integer in it has, one such integer."""
    period = 2 ** (max(bits) + 1)
    if k % 2:
        point = points[k // 2]
        if point == int(point):
            return None, {_bits_of(int(point), bits): int(point)}
        return point, {}
    low = points[k // 2 - 1] if k else None
    high = points[k // 2] if k < 2 * len(points) else None
    if low is None and high is None:
        other, ints = 1 / 2, range(period)
    elif low is None:
        other = math.floor(high) - 1 / 2  # type: ignore[arg-type]
        last = math.ceil(high) - 1  # type: ignore[arg-type]
        ints = range(last - period + 1, last + 1)
    elif high is None:
        other = math.floor(low) + 1 + 1 / 2
        first = math.floor(low) + 1
        ints = range(first, first + period)
    else:
        other = (low + high) / 2
        while other == int(other):  # halve toward the low end until it is no integer
            other = (low + other) / 2
        first, last = math.floor(low) + 1, math.ceil(high) - 1
        ints = range(first, min(last, first + period - 1) + 1)
    found: dict[tuple[int, ...], int] = {}
    for value in ints:
        found.setdefault(_bits_of(value, bits), value)
    return other, found


def _cubes(
    bits: Sequence[int], true: set[tuple[int, ...]], false: set[tuple[int, ...]], token: str
) -> list[str]:
    """The fewest bit tests, ORed, that hold for every *true* combination and no *false*
    one: a greedy cover by cubes; ``""`` is the cube of every combination."""
    cubes = sorted(itertools.product((None, 0, 1), repeat=len(bits)), key=_fixed)
    allowed = [q for q in cubes if not any(_in_cube(c, q) for c in false)]
    chosen: list[tuple[int | None, ...]] = []
    left = set(true)
    while left:  # the cube that covers most of what is left, fewest bits on a tie
        best = max(allowed, key=lambda q: (sum(_in_cube(c, q) for c in left), -_fixed(q)))
        chosen.append(best)
        left = {c for c in left if not _in_cube(c, best)}
    return [
        " and ".join(
            _bit_test(token, bit, w) for bit, w in zip(bits, q, strict=True) if w is not None
        )
        for q in chosen
    ]


def _both(first: str, second: str) -> str:
    """*first* and *second*; an ``or`` in either is put in parentheses."""
    return " and ".join(f"({t})" if " or " in t else t for t in (first, second) if t)


def _bit_arms(
    graph: PanelGraph, item: Light, token: str, bits: Sequence[int], points: Sequence[float]
) -> dict[str, str | None]:
    """The conditions of a channel with bit tests among its rows.

    A bit test matches integers only. On a segment of the number line (see
    :func:`_channel_arms`) every value that is no integer has one state, and
    an integer's state follows from the segment and its *bits*: each is read
    from :func:`health.channel_state` at a value that has it. A condition is
    the test that the value is no integer and in the segments where that
    state holds, or an integer, in a run of segments, with the fewest bit
    tests that say it; a combination no integer of a segment has may go
    either way.
    """
    combos = list(itertools.product((0, 1), repeat=len(bits)))
    other: list[State | None] = []
    ints: list[dict[tuple[int, ...], State]] = []
    for k in range(2 * len(points) + 1):
        value, found = _segment_values(points, k, bits)
        seen = health.channel_state(graph, item.channel, value) if value is not None else None
        other.append(_shown(seen, item) if seen is not None else None)
        ints.append(
            {
                c: _shown(health.channel_state(graph, item.channel, v), item)
                for c, v in found.items()
            }
        )
    guard = f"{token} == {token} // 1"
    arms: dict[str, str | None] = {}
    for state, ruled_out in _RULED_OUT.items():

        def mark(found: State | None, state=state, ruled_out=ruled_out) -> str:
            if found is None or found in ruled_out:
                return "D"
            return "T" if found == state else "F"

        parts: list[str] = []
        apart = _cover(points, [mark(s) for s in other], token)
        if apart is not None:
            parts.append(_both(f"{token} != {token} // 1", apart))
        # runs of segments whose integers want no conflicting marks, merged left to right
        runs: list[list] = []
        for k, states in enumerate(ints):
            if not states:
                continue
            pattern = {c: mark(states.get(c)) for c in combos}
            last = runs[-1] if runs else None
            if last is not None and not any({last[2][c], pattern[c]} == {"T", "F"} for c in combos):
                last[1] = k
                for c in combos:
                    if pattern[c] != "D":
                        last[2][c] = pattern[c]
            else:
                runs.append([k, k, pattern])
        terms: list[str] = []
        for i, (_first, _last, pattern) in enumerate(runs):
            true = {c for c in combos if pattern[c] == "T"}
            if not true:
                continue
            low = 0 if i == 0 else runs[i - 1][1] + 1
            high = 2 * len(points) if i == len(runs) - 1 else runs[i + 1][0] - 1
            where = _interval(points, low, high, token)
            where = "" if where == f"{token} == {token}" else where
            false = {c for c in combos if pattern[c] == "F"}
            tests = _cubes(bits, true, false, token)
            which = "" if "" in tests else " or ".join(tests)
            terms.append(_both(where, which))
        if terms:
            parts.append(guard if "" in terms else _both(guard, " or ".join(terms)))
        arms[state] = " or ".join(parts) if parts else None
    return arms


def _fixed(cube: tuple[int | None, ...]) -> int:
    return sum(w is not None for w in cube)


def _in_cube(combo: tuple[int, ...], cube: tuple[int | None, ...]) -> bool:
    return all(w is None or w == c for c, w in zip(combo, cube, strict=True))


def _channel_arms(graph: PanelGraph, item: Light, token: str) -> dict[str, str | None] | str:
    """The shortest condition under which one channel is in each state, or why none can be.

    The numbers the channel's rows name cut the number line into segments
    (open intervals and the points between them); the channel's state, as
    the light draws it, is constant on each, and is read from
    :func:`health.channel_state` at one value of each, so the conditions say
    exactly what the engine says. The condition of an arm may hold or not
    where an earlier arm has already ruled the channel out
    (:data:`_RULED_OUT`). A channel with bit tests is :func:`_bit_arms`'s;
    one with more than :data:`BIT_TESTS` bits is written row by row.
    """
    forms = [f for row in graph.rule(item.channel) if (f := health.form(row.value)) is not None]
    if any(f.kind == "literal" and f.number is None for f in forms):
        return "label condition"
    bits = sorted({f.bit for f in forms if f.kind == "bit"})  # type: ignore[type-var]
    if len(bits) > BIT_TESTS:
        return _row_arms(graph, item, token)
    points = sorted({f.number for f in forms if f.number is not None})
    if bits:
        return _bit_arms(graph, item, token, bits, points)  # type: ignore[arg-type]
    samples: list[float] = []
    for j, point in enumerate(points):
        samples.append(point - 1 if j == 0 else (points[j - 1] + point) / 2)
        samples.append(point)
    samples.append(points[-1] + 1 if points else 0)
    states = [_shown(health.channel_state(graph, item.channel, v), item) for v in samples]
    arms: dict[str, str | None] = {}
    for state, ruled_out in _RULED_OUT.items():
        marks = ["T" if s == state else "D" if s in ruled_out else "F" for s in states]
        arms[state] = _cover(points, marks, token)
    return arms


def _row_arms(graph: PanelGraph, item: Light, token: str) -> dict[str, str | None] | str:
    """The condition under which one channel is in each state, written row by row.

    For a channel with a bit test, which no interval of values can say.

    Rows with one condition are taken together, so agreeing sources write it
    once; two conditions no value can meet both of never disagree.
    """
    found: dict[str, tuple[health.Form, set[str]]] = {}
    for row in graph.rule(item.channel):
        row_form = health.form(row.value)
        if row_form is None:
            continue  # the engine never matches it either
        condition = _condition(row_form, token)
        if condition is None:
            return "label condition"
        found.setdefault(condition, (row_form, set()))[1].add(row.state)
    conds = list(found.items())

    def meets(i: int, j: int) -> bool:
        return not _exclusive(conds[i][1][0], conds[j][1][0])

    mixed = [c for c, (_, states) in conds if health.disagree(states)]
    pairs = [
        f"({a}) and ({b})"
        for i, (a, (_, states_a)) in enumerate(conds)
        for j, (b, (_, states_b)) in enumerate(conds)
        if i < j
        and a not in mixed
        and b not in mixed
        and health.disagree(states_a | states_b)
        and meets(i, j)
    ]
    arms: dict[str, str | None] = {"disagree": _any([*mixed, *pairs])}
    for state in ("bad", "warn", "normal"):
        terms = []
        for i, (c, (_, states)) in enumerate(conds):
            if states != {state}:
                continue
            against = _any(
                [d for j, (d, (_, other)) in enumerate(conds) if other != {state} and meets(i, j)]
            )
            terms.append(c if against is None else f"({c}) and not ({against})")
        arms[state] = _any(terms)
    # A light the rows flag is never drawn normal: where its rule says normal it
    # shows what the flag says (disagree), or unknown (rows the engine skips).
    if arms["normal"] is not None and item.disagree:
        arms["disagree"] = _any([c for c in (arms["disagree"], arms["normal"]) if c])
        arms["normal"] = None
    elif item.unsupported:
        arms["normal"] = None
    return arms


def ladder(graph: PanelGraph, lights: Sequence[Light], *, node: bool = False) -> Ladder | str:
    """The rule of one light over *lights*' channels: the worst state among them.

    A channel is ``disagree`` where rows of different states match its value,
    and in the state of the matching rows otherwise; rows the engine cannot
    evaluate are left out, as the engine leaves them out. The light is the
    worst channel (``bad > warn > disagree > unknown > normal``), so it is
    ``normal`` only when every channel is.

    A channel whose rows no condition can say adds no arm, so the others'
    ``bad``, ``warn`` and ``disagree`` still show and the light is never
    ``normal``; when that is every channel, the light cannot be written.

    Args:
        graph: The panel graph.
        lights: The channels the light reads.
        node: The light is a roll-up node's: over the limit it is an
            aggregate even with one channel. Otherwise only a light over
            several channels is.

    Returns:
        The ladder, or the :data:`FORMAT_REQUESTS` name of why the format
        cannot write it.
    """
    too_long = "aggregate state" if node or len(lights) > 1 else "long rule"
    per_channel: list[dict[str, str | None]] = []
    unwritten: list[str] = []
    reason = ""
    written = 0
    for i, item in enumerate(lights):
        arms = _channel_arms(graph, item, f"{{{i}}}")
        if isinstance(arms, str):
            reason = arms
            unwritten.append(item.channel)
            arms = dict.fromkeys(("disagree", "bad", "warn", "normal"))
        per_channel.append(arms)
        written += sum(len(c) for c in arms.values() if c)
        if written > RULE_EXPRESSION_LIMIT:
            return too_long  # every condition is written at least once: a cheap early exit
    if unwritten and len(unwritten) == len(lights):
        return reason
    out: list[tuple[str, State]] = []
    for state in _STATE_ORDER:
        found = [c for arms in per_channel if (c := arms[state])]
        if found:
            out.append((" or ".join(found), state))
    normals = [c for arms in per_channel if (c := arms["normal"])]
    if normals and len(normals) == len(per_channel):
        joined = " and ".join(f"({c})" if " or " in c else c for c in normals)
        out.append((joined, "normal"))
    rule = Ladder(tuple(item.channel for item in lights), tuple(out), tuple(unwritten))
    return too_long if engine_length(rule) > RULE_EXPRESSION_LIMIT else rule


def light_rule(rule: Ladder, values: Mapping[str, Any]) -> dict[str, Any]:
    """One prop's rule of a light: its channels, and its arms as the conditions.

    The first arm that holds gives its state's value, none holding gives
    ``unknown``'s, the default. Successive arms of one value share a
    condition, and the arms after the last one whose value is not the
    default's are left out. The arms are written into the rule itself, not
    into a screen formula: the viewer inlines a formula with longer tokens
    (``{<formula>.<name>}``) for each ``{N}`` that names it, so the
    compiled rule is shorter this way (see :func:`compiled_rules`).
    """
    default = values["unknown"]
    groups: list[tuple[Any, list[str]]] = []
    for condition, state in rule.arms:
        value = values[state]
        if groups and groups[-1][0] == value:
            groups[-1][1].append(condition)
        else:
            groups.append((value, [condition]))
    while groups and groups[-1][0] == default:
        groups.pop()
    conditions = [
        {"expression": " or ".join(f"({c})" if len(arms) > 1 else c for c in arms), "value": value}
        for value, arms in groups
    ]
    pvs = [{"name": channel, "trigger": True} for channel in rule.channels]
    return {"default": default, "pvs": pvs, "conditions": conditions}


def _literal(value: Any) -> str:
    """A rule value as the viewer writes it into the expression (its ``pyLiteral``)."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int | float):
        return str(int(value)) if float(value).is_integer() else repr(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


_WIRE_TOKEN = re.compile(r"\{([^{}]+)\}")


def engine_expression(rule: Mapping[str, Any], formulas: Mapping[str, Mapping[str, Any]]) -> str:
    """The one expression the viewer gives the formula engine for a conditional rule.

    This is the viewer's rule compiler, step for step: a ``formula://`` pv's
    screen formula has each binding name made a ``{<formula>.<name>}`` token
    and is inlined, in parentheses, for that pv's ``{N}`` in every condition;
    the conditions become ``value if (condition) else (...)`` ending in the
    default. The engine refuses it when it is longer than
    :data:`RULE_EXPRESSION_LIMIT`.
    """
    inlined: dict[str, str] = {}
    for index, pv in enumerate(rule.get("pvs", ())):
        name = pv["name"]
        if not name.startswith(_FORMULA_PREFIX):
            continue
        decl = formulas[name[len(_FORMULA_PREFIX) :]]
        wire = decl["expression"]
        keys = sorted(decl.get("bindings", {}), key=len, reverse=True)
        if not _WIRE_TOKEN.search(wire):
            for key in keys:
                wire = re.sub(
                    rf"(?<![A-Za-z0-9_{{]){re.escape(key)}(?![A-Za-z0-9_}}])", f"{{{key}}}", wire
                )
        for key in keys:
            wire = wire.replace(f"{{{key}}}", f"{{{decl['name']}.{key}}}")
        inlined[str(index)] = f"({wire})"

    def condition(text: str) -> str:
        return _WIRE_TOKEN.sub(lambda m: inlined.get(m.group(1), m.group(0)), text)

    out = _literal(rule["default"])
    for item in reversed(rule.get("conditions", ())):
        out = f"{_literal(item['value'])} if ({condition(item['expression'])}) else ({out})"
    return out


def compiled_rules(rule: Ladder) -> list[str]:
    """Every rule a light drawn with *rule* writes (one per prop of any look), as the
    viewer compiles it for the formula engine (:func:`engine_expression`)."""
    looks = {state: symbols.look(state) for state in symbols.STATES}
    return [
        engine_expression(light_rule(rule, {s: looks[s][prop] for s in symbols.STATES}), {})
        for prop in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    ]


def engine_length(rule: Ladder) -> int:
    """The longest expression the engine receives for a light drawn with *rule*."""
    return max(len(text) for text in compiled_rules(rule))


def ring_cost(rule: Ladder | str | None) -> tuple[int, frozenset[str]]:
    """What a strip symbol's ring and tag drawn with *rule* spend of a screen's budget
    (see :func:`light_cost`)."""
    if not isinstance(rule, Ladder) or not rule.arms:
        return 0, frozenset()
    looks = {state: symbols.ring_look(state) for state in symbols.STATES}
    rules = sum(
        bool(light_rule(rule, {s: looks[s][prop] for s in symbols.STATES})["conditions"])
        for prop in (*symbols.RING_PROPS, symbols.TAG_PROP)
    )
    return rules, frozenset(rule.channels)


def light_cost(rule: Ladder | str | None) -> tuple[int, frozenset[str]]:
    """What a light drawn with *rule* spends of a screen's budget: its rules that the
    viewer subscribes (those with a condition), and the channels they bind."""
    if not isinstance(rule, Ladder) or not rule.arms:
        return 0, frozenset()
    looks = {state: symbols.look(state) for state in symbols.STATES}
    rules = sum(
        bool(light_rule(rule, {s: looks[s][prop] for s in symbols.STATES})["conditions"])
        for prop in (*symbols.LIGHT_PROPS, symbols.MARK_PROP)
    )
    return rules, frozenset(rule.channels)


@dataclass(frozen=True)
class _Block:
    """A strip symbol's values as drawn: its short name over them, at *cx* the symbol's
    centre, *left* and *w* the block's span, in *lane*."""

    name: str
    channels: tuple[str, ...]
    cx: float
    left: float
    w: float
    lane: int


@dataclass
class Budget:
    """What is left of one screen's budget (:data:`SCREEN_RULE_BUDGET`,
    :data:`SCREEN_CHANNEL_BUDGET`)."""

    rules: int = field(default_factory=lambda: SCREEN_RULE_BUDGET)
    channels: set[str] = field(default_factory=set)

    def take(self, costs: Sequence[tuple[int, frozenset[str]]]) -> bool:
        """Spend *costs* when all of them fit; whether they did."""
        rules = sum(cost[0] for cost in costs)
        channels = self.channels.union(*(cost[1] for cost in costs))
        if rules > self.rules or len(channels) > SCREEN_CHANNEL_BUDGET:
            return False
        self.rules -= rules
        self.channels = channels
        return True


def tile_positions(
    count: int, first_x: float, line_x: float, line_top: float
) -> tuple[list[tuple[float, float]], float]:
    """Where *count* small lights go: ``(x, dy)`` each, and the height they take.

    The first line starts at *first_x*, each further line at *line_x*, the
    second *line_top* below the first and each next one a tile and a gap
    lower; a line ends at the page's right margin.
    """
    step = MOSAIC_LIGHT + MOSAIC_GAP
    right = MARGIN + STRIP_WIDTH
    out: list[tuple[float, float]] = []
    x, dy = first_x, 0.0
    for _ in range(count):
        if x + MOSAIC_LIGHT > right and x != line_x:
            x, dy = line_x, (line_top if dy == 0 else dy + step)
        out.append((x, dy))
        x += step
    return out, (out[-1][1] + MOSAIC_LIGHT if out else 0.0)


def chunks(graph: PanelGraph, lights: Sequence[Light]) -> list[list[Light]]:
    """*lights* in consecutive parts, each a rule within :data:`RULE_EXPRESSION_LIMIT`.

    The bound is the one :func:`ladder` applies, never a channel count: a part
    grows while the lengths of its channels' own rules fit the limit, and a
    part whose joint rule is still too long is halved. A channel too long on
    its own is a part alone (its light is then ``long rule``).
    """
    found: list[list[Light]] = []
    part: list[Light] = []
    cost = 0
    for item in lights:
        one = ladder(graph, [item])
        length = engine_length(one) if isinstance(one, Ladder) else 0
        if part and cost + length > RULE_EXPRESSION_LIMIT:
            found.append(part)
            part, cost = [], 0
        part.append(item)
        cost += length
    if part:
        found.append(part)

    def fit(part: list[Light]) -> list[list[Light]]:
        if len(part) == 1 or ladder(graph, part, node=True) != "aggregate state":
            return [part]
        half = len(part) // 2
        return fit(part[:half]) + fit(part[half:])

    return [piece for part in found for piece in fit(part)]


def _tolerance_ladder(row: ValueRow) -> Ladder | None:
    """Within the tolerance normal, outside it warn: setpoint ``{0}``, readback ``{1}``."""
    if row.tolerance is None or row.readback is None:
        return None
    term, amount = row.tolerance
    if term == "relative":
        within = f"({{1}} - {{0}}) ** 2 <= ({_number(amount)} * {{0}}) ** 2"
    else:
        within = f"({{1}} - {{0}}) ** 2 <= {_number(amount * amount)}"
    return Ladder((row.channel, row.readback), ((within, "normal"), (f"not ({within})", "warn")))


# --------------------------------------------------------------------------- the document


def _n(value: float) -> float | int:
    """A coordinate rounded to a tenth, integral values as int."""
    rounded = round(value, 1)
    return int(rounded) if rounded == int(rounded) else rounded


def _geometry(x: float, y: float, w: float, h: float) -> dict[str, float | int]:
    return {"x": _n(x), "y": _n(y), "width": _n(w), "height": _n(h)}


def _pixel(value: float) -> int:
    """*value* to the nearest whole pixel, a half up (never to even)."""
    return math.floor(value + HALF_PIXEL)


def snapped(x: float, y: float, w: float, h: float) -> dict[str, int]:
    """A box on whole pixels: each edge rounded, not its origin and size, so two
    boxes that touch still touch and a box of whole size keeps it."""
    x0, y0, x1, y1 = _pixel(x), _pixel(y), _pixel(x + w), _pixel(y + h)
    return {"x": x0, "y": y0, "width": max(x1 - x0, 0), "height": max(y1 - y0, 0)}


@dataclass
class _Screen:
    """One Screen-IR document under construction, ids in build order."""

    doc_id: str
    title: str
    nodes: list[dict[str, Any]] = field(default_factory=list)
    unknown: int = 0
    formulas: list[dict[str, Any]] = field(default_factory=list)
    _widgets: int = 1
    _rules: int = 0
    _item: tuple[str, ...] = ()

    @contextlib.contextmanager
    def item(self, kind: str, thing: str | None = None, order: int | None = None):
        """Every widget added inside names the view item it draws (:data:`ITEM_TAGS`):
        its *kind*, the *thing* it stands for, and its *order* where the view
        states one. The tags go into the widget's ``meta`` comment, which a viewer
        does not read."""
        saved = self._item
        self._item = item_tags(kind, thing, order)
        try:
            yield
        finally:
            self._item = saved

    def add(
        self,
        node: dict[str, Any],
        *,
        rules: list[dict[str, Any]] | None = None,
        tags: Sequence[str] = (),
    ) -> dict[str, Any]:
        self._widgets += 1
        g = node["geometry"]
        if node["type"] == "text-label":
            if node["props"].get("fontSize", SMALL_SIZE) < SMALL_SIZE:
                node["props"]["fontSize"] = SMALL_SIZE  # the floor: nothing smaller anywhere
            g = _line_room(g, node["props"].get("fontSize", TEXT_SIZE))
        box = snapped(g["x"], g["y"], g["width"], g["height"])
        out: dict[str, Any] = {
            "id": f"w-{self._widgets:04d}",
            "type": node["type"],
            "props": node["props"],
            "geometry": box,
        }
        if "points" in out["props"]:  # each point where it was on the page, on a pixel
            out["props"]["points"] = [
                {
                    "x": _pixel(g["x"] + p["x"]) - box["x"],
                    "y": _pixel(g["y"] + p["y"]) - box["y"],
                }
                for p in out["props"]["points"]
            ]
        if rules:
            out["rules"] = rules
        tags = (*tags, *self._item)
        if tags:
            out["meta"] = {"comment": "; ".join(tags)}
        self.nodes.append(out)
        cut = out["props"].get("text") if out["type"] == "text-label" else None
        if isinstance(cut, _Cut) and not rules:
            # a cut label shows its whole text on hover; a link drawn over it
            # later shows its own label instead
            self.add(_hover_box(out["geometry"], cut.whole))
        return out

    def widget(self, kind: str, x, y, w, h, props: dict[str, Any], **kw: Any) -> dict[str, Any]:
        return self.add({"type": kind, "props": props, "geometry": _geometry(x, y, w, h)}, **kw)

    def rule(self, name: str, target: str, ladder: Ladder, values: Mapping[str, Any]) -> dict:
        """One prop's rule: the value of the first arm that holds, else unknown's."""
        self._rules += 1
        return {
            "id": f"r-{self._rules:04d}",
            "name": name,
            "targetProperty": target,
            **light_rule(ladder, values),
        }

    def document(self, height: float) -> dict[str, Any]:
        size = {"width": PAGE_WIDTH, "height": _pixel(height)}
        description = f"Generated from the facility's panel view {self.doc_id}."
        if self.unknown:
            description += (
                f" {self.unknown} light(s) drawn unknown: the format cannot write their rule."
            )
        root = {
            "id": "w-0001",
            "type": "absolute-canvas",
            "props": {**size, "backgroundColor": PAGE_BACKGROUND},
            "geometry": {"x": 0, "y": 0, **size},
            "children": self.nodes,
        }
        return {
            "schemaVersion": "1.1",
            "kind": "screen",
            "id": self.doc_id,
            "metadata": {
                "title": self.title,
                "source": {"type": "osprey-panels"},
                "size": size,
                "description": description,
            },
            "macros": [],
            **({"formulas": self.formulas} if self.formulas else {}),
            "root": root,
        }


def _need(name: str) -> str:
    assert name in _REQUEST_NAMES, name
    return _REQUEST_TAG + name


def needs(doc: Mapping[str, Any]) -> list[str]:
    """The :data:`FORMAT_REQUESTS` names a rendered document hits, each once, in order."""
    found: list[str] = []
    for node in _walk(doc["root"]):
        for tag in node.get("meta", {}).get("comment", "").split("; "):
            if tag.startswith(_REQUEST_TAG) and tag[len(_REQUEST_TAG) :] not in found:
                found.append(tag[len(_REQUEST_TAG) :])
    return found


def aggregates(doc: Mapping[str, Any]) -> int:
    """How many roll-up rows of a rendered document are drawn as an aggregate."""
    return sum(
        _AGGREGATE_ROW in n.get("meta", {}).get("comment", "").split("; ")
        for n in _walk(doc["root"])
    )


def unexpressed(doc: Mapping[str, Any]) -> int:
    """How many lights of a rendered document are drawn unknown because no rule could be written."""
    return sum(
        any(t.startswith(_UNKNOWN_TAG) for t in n.get("meta", {}).get("comment", "").split("; "))
        for n in _walk(doc["root"])
    )


def _walk(node: Mapping[str, Any]):
    yield node
    for child in node.get("children", ()):
        yield from _walk(child)


# --------------------------------------------------------------------------- the page


def _citation_text(citation: Citation | None) -> str:
    if citation is None:
        return ""
    parts = [citation.document, citation.version, citation.state, citation.locator]
    return " ".join(p for p in parts if p)


def _long_group(section: Section) -> bool:
    """A card's signals or settings group of more than :data:`CARD_SPLIT_ROWS` rows."""
    if section.title not in ("signals", "settings"):
        return False
    return sum(isinstance(i, ValueRow) for i in section.items) > CARD_SPLIT_ROWS


class _Page:
    """The drawing of one view: a cursor that moves down the page."""

    def __init__(
        self,
        view: View,
        graph: PanelGraph,
        screens: Callable[[Target], str | None],
        doc_id: str,
        colours: Mapping[str, str] = MappingProxyType({}),
    ) -> None:
        self.view = view
        self.graph = graph
        self.screens = screens
        self.s = _Screen(doc_id, view.title)
        self.y: float = MARGIN
        # writable widgets only on a device card's own signals and settings
        self.on_card = view.id.view == "card" and view.id.kind == "device"
        self.label_w: float = CARD_LABEL_WIDTH  # a signal row's name, in the card's table
        self.label_col: float = LABEL_WIDTH  # a value or fact row's label, in its section
        self.check_cols: tuple[float, float] = (NAME_WIDTH, NAME_WIDTH)  # channel, thing
        self.value_w: float = VALUE_WIDTH
        self.budget = Budget()  # spent by the strip's symbols, chips and summary rows
        self.colours: Mapping[str, str] = colours  # a symbol's type colour, by symbol name
        self.shown: dict[
            str, dict[str | None, str | None]
        ] = {}  # strip values: symbol, signal, unit
        self.dropped = 0  # strip values the budget left out
        self.crowded = 0  # strip values with no room in the lanes under their symbol
        self.strip_order: dict[str, int] = {}  # a strip's things, in the beam order it states
        self.taken: list[tuple[float, float, float, float]] = []  # a strip row's boxes
        self.x0: float = MARGIN  # the left edge of the column drawn in
        self.right: float = PAGE_WIDTH - MARGIN + PANEL_PAD  # where text is cut
        self.lit: set[str] = set()  # channels whose light a card's table drew
        self.facts_bottom: float = 0  # a rack's right column, where it ends
        self.cell_w: float = SET_CELL_WIDTH  # a set overview's cell width
        self.cell_units = "number"  # how a set overview's cells state their unit
        self.one_unit = True  # the set's cells share one unit, stated over the grid
        self.drawn = 0  # a set overview's cells drawn so far
        self.parts_listed: set[str] = set()  # parts a card's picture legend has listed

    # -- primitives -----------------------------------------------------------

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = TEXT_SIZE,
        w: float | None = None,
        h: float = ROW,
        fg=PAGE_TEXT,
        bg=TRANSPARENT,
        align="left",
        tags=(),
    ) -> float:
        """A text label; returns its right edge. Text that would run past the right
        margin, or past a stated width *w*, is cut there, an ellipsis marking the cut:
        no label leaves its box for the viewer to shrink or clip."""
        w = text_width(text, size) if w is None else w
        right = self.right  # the inner edge of a section's frame, or the column's
        if x + w > right:
            w = max(right - x, 0)
        text = _fit(text, w, size)
        props = {
            "text": text,
            "fontSize": size,
            "align": align,
            "foregroundColor": fg,
            "backgroundColor": bg,
        }
        self.s.widget("text-label", x, y, w, h, props, tags=tags)
        return x + w

    def mark(self, x, y, glyph: str) -> float:
        return self.text(x, y, glyph, w=MARK_WIDTH)

    def line(
        self,
        points: Sequence[tuple[float, float]],
        colour,
        width=LINE_WIDTH,
        *,
        tooltip=None,
        arrow: bool = False,
    ) -> None:
        """A line through *points*; its box is never narrower or lower than its stroke.

        A vertical or horizontal line has a box of no width or no height, which
        the viewer may draw as nothing; the box is widened by the stroke and the
        points sit in its middle.
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, y0 = min(xs), min(ys)
        w, h = max(xs) - x0, max(ys) - y0
        dx, dy = max(width - w, 0) / 2, max(width - h, 0) / 2
        props: dict[str, Any] = {
            "lineColor": colour,
            "lineWidth": width,
            "points": [{"x": px - x0 + dx, "y": py - y0 + dy} for px, py in points],
            "arrowStart": False,
            "arrowEnd": arrow,
        }
        if tooltip is not None:
            props[_TOOLTIP] = tooltip
        self.s.widget("line", x0 - dx, y0 - dy, w + 2 * dx, h + 2 * dy, props)

    def box(self, x, y, w, h, *, line=BOX_BORDER, fill=FIELD_BACKGROUND, dashed=False) -> None:
        if dashed:
            points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            node = {
                "type": "polygon",
                "props": {
                    "lineColor": line,
                    "lineWidth": LINE_WIDTH,
                    "lineStyle": "dashed",
                    "fill": False,
                    "points": [{"x": px - x, "y": py - y} for px, py in points],
                },
                "geometry": {"x": x, "y": y, "width": w, "height": h},
            }
            self.s.add(node)
            return
        props = {"lineColor": line, "lineWidth": LINE_WIDTH, "fill": True, "fillColor": fill}
        self.s.widget("rectangle", x, y, w, h, props)

    def button(
        self,
        x: float,
        y: float,
        link: Link,
        *,
        label: str | None = None,
        width: float | None = None,
    ) -> float:
        """A link to another screen, at most *width* wide; returns the right edge of what it drew."""
        text = link.label if label is None else label
        w = min(text_width(text, TEXT_SIZE) + BUTTON_PAD, MAX_BUTTON_WIDTH, self.right - x)
        if width is not None:
            w = min(w, width)
        # Neutral: page text on the field background, outlined in the border
        # role. The format's navigation buttons take no colour, so the text is
        # a label and the outline the box that opens the screen.
        self.text(x, y, text, w=w, bg=FIELD_BACKGROUND, align="center")
        self.nav(x, y, w, ROW, replace(link, label=text), outline=True)
        right = x + w
        if link.basis == "inferred":
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        if link.standing == _CONTESTED:
            right = self.text(right + GAP, y, _CONTESTED, size=SMALL_SIZE)
        return right

    def quiet_link(self, right: float, y: float, link: Link) -> None:
        """A link as muted text ending at *right*: no box, a box that opens it on hover."""
        w = text_width(link.label, SMALL_SIZE)
        self.text(right - w, y, link.label, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        self.nav(right - w, y, w, ROW, link)

    def ref(self, x, y, item: Link | Fact) -> float:
        if isinstance(item, Link):
            return self.button(x, y, item)
        return self.text(x, y, item.text or item.label)

    def light(
        self,
        x: float,
        y: float,
        lights: Sequence[Light],
        *,
        rule: Ladder | str | None = None,
        size: float = LIGHT_SIZE,
        tile: bool = False,
        tooltip: str | None = None,
    ) -> float:
        """One light over *lights*; returns its right edge (after an inferred mark).

        A light whose rule is too long for the format is the aggregate look; a
        *tile* is one in an aggregate's mosaic and carries no inferred mark.
        """
        if rule is None:
            rule = ladder(self.graph, lights)
        look = "aggregate" if rule == "aggregate state" else "unknown"
        shape, mark = symbols.light(x, y + (ROW - size) / 2, size, look)
        self.attach(shape, mark, lights, rule, tile=tile, tooltip=tooltip)
        right = x + size
        if not tile and any(item.inferred for item in lights):
            right = self.mark(right + GAP, y, symbols.INFERRED_MARK)
        return right

    def attach(
        self,
        shape: dict[str, Any],
        mark: dict[str, Any],
        lights: Sequence[Light],
        rule: Ladder | str,
        *,
        tile: bool = False,
        tooltip: str | None = None,
    ) -> None:
        """Add a light's *shape* and *mark*, drawn in the look *rule* starts in, with
        the rules that turn them to the state of *lights*, or the tags saying why
        none can."""
        aggregate = rule == "aggregate state"
        names = ", ".join(item.channel for item in lights[:TOOLTIP_CHANNELS])
        if len(lights) > TOOLTIP_CHANNELS:
            names = f"{len(lights)} channels"
        shape["props"][_TOOLTIP] = names if tooltip is None else tooltip
        tags = [_need("state light")]
        shape_rules: list[dict] = []
        mark_rules: list[dict] = []
        if aggregate:
            tags += [_need("aggregate state"), _AGGREGATE_TILE if tile else _AGGREGATE_ROW]
        elif isinstance(rule, str):
            tags += [_need(rule), _UNKNOWN_TAG + rule]
            self.s.unknown += 1
        else:
            if rule.unwritten:
                # the writable channels' arms are drawn; the unwritten ones add none
                tags.append(_need("label condition"))
            if rule.arms:
                self.rules(shape_rules, mark_rules, names, rule)
        self.s.add(shape, rules=shape_rules, tags=tags)
        mark_tags = [_need("hatched fill")] if mark_rules else []
        self.s.add(mark, rules=mark_rules, tags=mark_tags)

    def no_rule(self, x: float, y: float, size: float = LIGHT_SIZE) -> float:
        """The look of a thing with no rule: a bare outline, no rules, no mark; its right edge."""
        shape, _mark = symbols.light(x, y + (ROW - size) / 2, size, symbols.NONE)
        shape["props"][_TOOLTIP] = _NO_RULE
        self.s.add(shape)
        return x + size

    def rules(self, shape_rules: list, mark_rules: list, names: str, rule: Ladder) -> None:
        """A light's rules: one per shape prop and one for its mark, all from *rule*."""
        looks = {state: symbols.look(state) for state in symbols.STATES}
        for prop in symbols.LIGHT_PROPS:
            values: dict[str, Any] = {state: looks[state][prop] for state in symbols.STATES}
            shape_rules.append(self.s.rule(f"{names} {prop}", prop, rule, values))
        values = {state: looks[state][symbols.MARK_PROP] for state in symbols.STATES}
        mark_rules.append(self.s.rule(f"{names} mark", symbols.MARK_PROP, rule, values))

    def widget_for(self, drawn_as: str | None, writable: bool) -> tuple[str, str]:
        """The widget :data:`WIDGETS` names for *drawn_as*, and the kind it is drawn as."""
        kind = drawn_as if drawn_as not in (None, "untyped") else None
        if kind is None:
            kind = "setpoint" if writable else "untyped"
        on_card, elsewhere = WIDGETS[kind]
        widget = on_card if self.on_card else elsewhere
        if widget in _WRITABLE_WIDGETS and not (self.on_card and writable):
            widget = elsewhere
        return widget, kind

    def precision(self, props: dict[str, Any], value: str, bindings: Mapping[str, str]) -> list:
        """*props* print *value* (an expression over *bindings*) with
        :data:`PRECISION_FALLBACK` decimals; the rule that sets them live to
        :data:`SIGNIFICANT_DIGITS` significant digits, when the budget and the
        expression limit allow it (else none)."""
        props["precision"] = PRECISION_FALLBACK
        expression = precision_expression(value)
        if len(expression) > RULE_EXPRESSION_LIMIT:
            expression = precision_expression(value, compact=True)
        if len(expression) > RULE_EXPRESSION_LIMIT:
            return []
        if not self.budget.take([(1, frozenset(bindings.values()))]):
            return []
        name = f"precision-{len(self.s.formulas)}"
        self.s.formulas.append(
            {"name": name, "kind": "scalar", "expression": expression, "bindings": dict(bindings)}
        )
        self.s._rules += 1
        return [
            {
                "id": f"r-{self.s._rules:04d}",
                "name": "precision",
                "targetProperty": "precision",
                "default": PRECISION_FALLBACK,
                "formula": _FORMULA_PREFIX + name,
            }
        ]

    def field(
        self,
        x: float,
        y: float,
        channel: str,
        *,
        drawn_as: str | None = None,
        writable: bool = False,
        w: float = VALUE_WIDTH,
        units: bool = False,
        within: str | None = None,
    ) -> float:
        """A channel's value in the widget :data:`WIDGETS` names for its ``drawn_as``.

        A writable widget only on a device card, and only for a writable
        channel; anywhere else the value is a label, a number printed to
        :data:`SIGNIFICANT_DIGITS` significant digits, with its unit when *units*.
        A label is widened to print its unit whole, unless *within* holds it to
        *w*: ``"number"`` prints the number alone (the unit is stated once
        elsewhere), ``"unit"`` the number and its unit, the unit's width kept
        clear first. Returns the right edge.
        """
        widget, kind = self.widget_for(drawn_as, writable)
        common = {"pv": channel, "fontSize": TEXT_SIZE, "foregroundColor": PAGE_TEXT}
        if widget == "pv-text-input":
            props = {**common, "backgroundColor": FIELD_BACKGROUND, "alarmSensitiveBorder": False}
            if self.enumerated(channel):  # an operator types 0 or 1, not 1.0000
                props["precision"] = 0
            self.s.widget(widget, x, y, w, ROW, props)
        elif widget == "pv-button":
            props = {
                **common,
                "label": self.command_label(channel),
                "pressValue": COMMAND_VALUE,
                "backgroundColor": FIELD_BACKGROUND,
            }
            self.s.widget(widget, x, y, w, ROW, props)
        elif widget == "waveform-plot":
            curve = {"name": channel, "y_channel": channel, "color": PAGE_TEXT}
            self.plot(x, y, WAVEFORM_WIDTH, WAVEFORM_HEIGHT, channel, [curve])
            self.y += WAVEFORM_HEIGHT - ROW
            return x + WAVEFORM_WIDTH
        else:
            props = {
                **common,
                "backgroundColor": FIELD_BACKGROUND,
                "showUnits": units,  # else the unit stands with the signal's name
                "alarmSensitive": True,
                "alarmSensitiveBorder": False,  # the viewer frames it once; the light says state
                "align": "right",
            }
            rules = []
            own = self.own_kind(channel)
            if own is not None and own in _WHOLE_KINDS:  # a readback shown without its row's kind
                kind = own
            if within is None:
                w = max(w, self.value_span(channel))  # the unit printed whole
            # the viewer prints a label's unit whatever showUnits says, and cuts a long
            # one: a number held to w, or one whose unit is cut, is a formula's text
            # with the whole unit, if any, beside it
            unit = "" if within == "number" else (self.unit_of(channel) or "")
            number_only = within is not None or self.cuts_unit(channel)
            if kind in _PRINTED and number_only and self.budget.take([(1, frozenset({channel}))]):
                unit_w = text_width(unit, TEXT_SIZE) if unit else 0.0
                number_w = max(0.0, w - (UNIT_GAP + unit_w if unit else 0.0))
                expression = self.number_expression(channel, kind)
                self.formula_text(
                    x, y, number_w, ROW, channel, expression, size=TEXT_SIZE, fg=PAGE_TEXT,
                    bg=FIELD_BACKGROUND, align="right",
                )  # fmt: skip
                if unit:
                    self.text(x + number_w + UNIT_GAP, y, unit, w=unit_w)
                return x + w
            if within is not None:  # no formula to spare: the viewer's label, unit and all
                props["showUnits"] = within == "unit"
            if kind in _WHOLE_KINDS or self.prints_whole(channel):
                props["precision"] = 0
            elif kind in _NUMBERS:
                rules = self.precision(props, "{v}", {"v": channel})
            self.s.widget("pv-label", x, y, w, ROW, props, rules=rules)
        return x + w

    def unit_of(self, channel: str) -> str | None:
        """The unit the facility states for *channel*: the one the viewer prints after
        its value; ``None`` with none, or for a channel the graph does not hold."""
        try:
            return self.graph.channel(channel).unit
        except KeyError:
            return None

    def own_kind(self, channel: str) -> str | None:
        """How *channel*'s ``record_types.csv`` row draws it: the ``drawn_as`` it
        states, else its role's default; ``None`` with no row (its kind unknown)."""
        if self.graph.record_type(channel) is None:
            return None
        try:
            return self.graph.channel(channel).drawn_as
        except KeyError:
            return None

    def command_label(self, channel: str) -> str:
        """The word a command's button reads: the word ``state_labels.csv`` states
        for the value the button writes, else the action the channel's name says."""
        stated = self.graph.state_label(channel, COMMAND_VALUE)
        return stated or action_word(channel)

    def enumerated(self, channel: str) -> bool:
        """Whether *channel*'s record is a two- or multi-valued output
        (:data:`ENUMERATED_OUTPUTS`): its value is whole."""
        row = self.graph.record_type(channel)
        return row is not None and row.record_type in ENUMERATED_OUTPUTS

    def cuts_unit(self, channel: str) -> bool:
        """Whether Channel Access cuts *channel*'s unit: it is longer than
        :data:`CHANNEL_ACCESS_UNIT_CHARS`."""
        return len(self.unit_of(channel) or "") > CHANNEL_ACCESS_UNIT_CHARS

    def value_span(self, channel: str) -> float:
        """The width *channel*'s value takes with its unit: one display as wide as
        the widest text the facility states it can show (:meth:`widest_value`), or,
        for a unit Channel Access cuts, the number's display and the whole unit
        beside it."""
        text, word = self.widest_value(channel)
        if word:
            return shown_width(text, TEXT_SIZE)
        unit = self.unit_of(channel)
        if not self.cuts_unit(channel):
            return shown_width(f"{text} {unit}" if unit else text, TEXT_SIZE)
        return shown_width(text, TEXT_SIZE) + UNIT_GAP + text_width(unit or "", TEXT_SIZE)

    def widest_value(self, channel: str) -> tuple[str, bool]:
        """The widest text *channel*'s value display can print, and whether it is a
        word: its longest state word when the facility or ``state_labels.csv``
        states words for it; else the longer of its limits as the display prints
        them (three significant digits) when that is wider than
        :data:`VALUE_SAMPLE`; else :data:`VALUE_SAMPLE`, the widest number three
        significant digits print, since nothing stated bounds the value."""
        try:
            stated = self.graph.channel(channel)
        except KeyError:
            return VALUE_SAMPLE, False
        words = [*self.graph.state_words(channel).values(), *stated.options]
        if words:
            return max(words, key=len), True
        if stated.limits is not None:
            bound = max((_significant(v) for v in stated.limits), key=len)
            if len(bound) > len(VALUE_SAMPLE):
                return bound, False
        return VALUE_SAMPLE, False

    def number_expression(self, channel: str, kind: str | None, decimals: int | None = None):
        """The formula that prints *channel* as a static label's text: as it is for a
        state or a command, to *decimals* when the data states them, else to
        :data:`SIGNIFICANT_DIGITS` significant digits."""
        if kind in _WHOLE_KINDS or self.prints_whole(channel):
            return "{v}"
        if decimals is not None:
            return fixed_expression("{v}", decimals)
        return significant_expression("{v}")

    def formula_text(
        self, x, y, w, h, channel: str, expression: str, *, size, fg, bg, align, tags=()
    ) -> None:
        """A static label whose text a formula rule sets to *expression* over
        *channel*: :data:`NO_VALUE` until the rule delivers and while the channel is
        disconnected, since the rule's output is stale then."""
        name = f"value-{len(self.s.formulas)}"
        self.s.formulas.append(
            {"name": name, "kind": "scalar", "expression": expression, "bindings": {"v": channel}}
        )
        self.s._rules += 1
        rule = {
            "id": f"r-{self.s._rules:04d}",
            "name": f"{channel} value",
            "targetProperty": "text",
            "default": NO_VALUE,
            "formula": _FORMULA_PREFIX + name,
        }
        props = {
            "text": NO_VALUE,
            "fontSize": size,
            "foregroundColor": fg,
            "backgroundColor": bg,
            "align": align,
        }
        self.s.widget("text-label", x, y, w, h, props, rules=[rule], tags=tags)

    def prints_whole(self, channel: str) -> bool:
        """Whether *channel* is a state: every health rule of it names one whole value,
        as itself or with ``==`` or ``!=``; a threshold (``>=5``) is a measure's."""
        rules = self.graph.rule(channel)
        return bool(rules) and all(_WHOLE.fullmatch(r.value.strip()) for r in rules)

    def values_width(self, rows: Sequence[ValueRow], least: float) -> float:
        """The width of a table's value column: *least*, or what :func:`value_width`
        needs for the widest unit its channels print."""
        channels = [c for row in rows for c in (row.channel, row.readback) if c is not None]
        return max(least, *(self.value_span(c) for c in channels))

    def unit_field(
        self, x: float, channel: str, unit: str | None, drawn_as: str | None, writable: bool
    ) -> float:
        """:meth:`field` with the unit once: in the label, or beside an input."""
        if self.widget_for(drawn_as, writable)[0] != "pv-label":
            x = self.field(x, self.y, channel, drawn_as=drawn_as, writable=writable)
            return self.unit(x, self.y, unit)
        w = max(VALUE_WIDTH, self.value_span(channel))
        return self.field(x, self.y, channel, drawn_as=drawn_as, writable=writable, w=w, units=True)

    def unit(self, x: float, y: float, unit: str | None) -> float:
        """A value's unit, its own label beside the value; the right edge (*x* with none)."""
        if not unit:
            return x
        return self.text(x + UNIT_GAP, y, unit, size=SMALL_SIZE, w=UNIT_WIDTH)

    def next_row(self, height: float = ROW) -> None:
        self.y += height + ROW_GAP

    # -- the view -------------------------------------------------------------

    def draw(self) -> dict[str, Any]:
        self.header()
        self.y = HEADER_HEIGHT + SECTION_GAP
        if self.view.title != self.view.id.id and self.title_rest() is None:
            # else the header says it already
            self.text(MARGIN, self.y, self.view.title, size=TITLE_SIZE, w=STRIP_WIDTH, h=ROW + GAP)
            self.y += ROW + GAP
        sections = list(self.view.sections)
        if self.on_card and any(
            isinstance(i, CardHead) for section in sections for i in section.items
        ):
            self.drawn_card(sections)  # the stated height: the last frame's end and a margin
            return self.s.document(self.y + SECTION_GAP)
        if self.on_card:
            if any(s.title == PROBE_SECTION for s in sections):  # the picture is the point
                first = [s for s in sections if s.title in _PICTURE_FIRST]
                for section in first:
                    self.section(section)
                sections = [s for s in sections if s not in first]
            sections = self.card(sections)
        beside: Group | None = None
        arcs: Group | None = None
        if self.view.id.view == "rollup":
            for section in list(sections):
                group = section.items[0] if section.items else None
                if not isinstance(group, Group):
                    continue
                if group.form == "ring":
                    arcs = group
                elif section.title == _BY_SYSTEM_TITLE:
                    beside = group
                else:
                    continue
                sections.remove(section)
        for section in sections:
            tree = section.items[0] if section.items else None
            if self.view.id.view == "rollup" and isinstance(tree, Node):
                self.rollup(tree, beside, arcs)
            else:
                self.section(section)
        return self.s.document(self.y + SECTION_GAP)

    # -- the roll-up ----------------------------------------------------------

    def rollup(self, root: Node, beside: Group | None, arcs: Group | None = None) -> None:
        """A roll-up in one frame: the root's state, then its children as large tiles
        and, right of them, the rows *beside* (the place by system).

        A tile is a child's light, its short name and what it reads, opening its
        screen; all tiles have one height. The budget is spent as :meth:`plan`
        says: a child that fits one rule has its light; any other is ``open``:
        the aggregate mark, its count and a link down to its own screen, where
        its lights are. A tile never holds small lights of its parts. With
        *arcs*, those children are the ring's arcs (:meth:`ring`) and the
        others are tiles under it.
        """
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        width = ROLLUP_LEFT - COLUMN_GAP if beside is not None else STRIP_WIDTH
        self.x0, self.right = MARGIN, MARGIN + width
        self.heading(f"by {self.view.id.by or 'place'}")
        plan, decisions = self.plan(root)
        self.rollup_root(root, plan)
        on_ring = {node.label for node in arcs.nodes} if arcs is not None else set()
        if arcs is not None:
            self.ring(root, arcs, decisions)
        columns = max(1, int((width + GAP) // (ROLLUP_TILE_WIDTH + GAP)))
        w = (width - (columns - 1) * GAP) / columns
        children = [(i, c) for i, c in enumerate(root.children) if c.label not in on_ring]
        y = self.y
        for start in range(0, len(children), columns):
            for column, (i, child) in enumerate(children[start : start + columns]):
                x = MARGIN + column * (w + GAP)
                plan_i = "none" if child.no_rule else decisions.get(i, "open")
                self.rollup_tile(x, y, w, ROLLUP_TILE_HEIGHT, child, root.label, plan_i)
            y += ROLLUP_TILE_HEIGHT + GAP
        bottom = y
        if beside is not None:
            self.x0, self.right = MARGIN + ROLLUP_LEFT, MARGIN + STRIP_WIDTH
            self.y = top + PANEL_PAD
            self.heading(beside.title)
            self.summary(beside, self.x0, self.right - self.x0, 1)
            bottom = max(bottom, self.y)
        self.x0, self.right = MARGIN, PAGE_WIDTH - MARGIN + PANEL_PAD
        self.y = bottom + PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())
        self.y += PANEL_GAP

    def ring(self, root: Node, arcs: Group, decisions: Mapping[int, str]) -> None:
        """The ring: one arc per child on it, clockwise from twelve o'clock in beam
        order, each in the look of its tile (:meth:`rollup_tile`): its light's
        rule, the aggregate mark when over the rule limit, a dashed outline with
        no rule; its short name and mark on it, and a link to its screen. The
        root's id stands in the middle."""
        index = {node.label: i for i, node in enumerate(root.children)}
        cx = (self.x0 + self.right) / 2
        cy = self.y + RING_MARGIN + RING_OUTER
        share = TURN / len(arcs.nodes)
        for number, node in enumerate(arcs.nodes):
            start, end = number * share + RING_GAP, (number + 1) * share - RING_GAP
            i = index.get(node.label)
            plan = "none" if node.no_rule or i is None else decisions.get(i, "open")
            self.arc(cx, cy, start, end, node, root.label, plan)
        w = text_width(root.label, TITLE_SIZE)
        self.text(cx - w / 2, cy - ROW / 2, root.label, size=TITLE_SIZE, w=w)
        self.y = cy + RING_OUTER + RING_MARGIN

    def arc(
        self, cx: float, cy: float, start: float, end: float, node: Node, parent: str, plan: str
    ) -> None:
        """One child on the ring: its arc, its short name and mark at the arc's middle,
        and a box there that opens its screen."""
        lights = [describe_light(self.graph, c) for c in node.channels]
        middle = math.radians((start + end) / 2)
        r = (RING_INNER + RING_OUTER) / 2
        mx, my = cx + r * math.sin(middle), cy - r * math.cos(middle)
        name = self.graph.thing(node.label, "place").short_name  # an arc is a child place
        name = name or node.label.removeprefix(parent + "/")
        name = _cut(name, RING_LABEL_WIDTH, TEXT_SIZE)
        left, top = mx - RING_LABEL_WIDTH / 2, my - ROW
        if plan == "none":
            shape = symbols.arc(cx, cy, RING_INNER, RING_OUTER, start, end, symbols.NONE)
            shape["props"]["lineStyle"] = "dashed"
            shape["props"][_TOOLTIP] = _NO_RULE
            self.s.add(shape)
        else:
            rule = ladder(self.graph, lights, node=True) if plan == "live" else "aggregate state"
            look = "aggregate" if rule == "aggregate state" else "unknown"
            shape = symbols.arc(cx, cy, RING_INNER, RING_OUTER, start, end, look)
            _light, mark = symbols.light(left, top + ROW, RING_LABEL_WIDTH, look)
            mark["geometry"]["height"] = ROW
            mark["props"]["fontSize"] = RING_MARK
            self.attach(shape, mark, lights, rule)
        self.text(left, top, name, w=RING_LABEL_WIDTH, align="center")
        if node.link is not None:
            self.nav(left, top, RING_LABEL_WIDTH, 2 * ROW, replace(node.link, label=node.label))

    def rollup_root(self, root: Node, plan: str | None) -> None:
        """The root's line: its light and what it reads, its parts, or that its summary
        needs a server-side aggregate."""
        count = len(root.channels)
        lights = [describe_light(self.graph, c) for c in root.channels]
        if plan == "summary":
            text = _SUMMARY_TEXT.format(count)
            tags = (_need("server-side aggregate"),)
            self.text(self.x0, self.y, text, size=SMALL_SIZE, tags=tags)
            self.next_row()
            return
        if root.no_rule:
            x = self.no_rule(self.x0, self.y)
            note = _NO_RULE_YET
        elif plan == "live":
            x = self.light(self.x0, self.y, lights, rule=ladder(self.graph, lights, node=True))
            note = _READS.format(count, "" if count == 1 else "s")
        else:
            x = self.light(self.x0, self.y, lights, rule="aggregate state")
            note = _AGGREGATE_TEXT.format(count)
        x = self.text(x + GAP, self.y, f"{root.label} · {note}", size=SMALL_SIZE, fg=MUTED_TEXT)
        height: float = ROW
        if plan == "parts":
            height = self.parts(chunks(self.graph, lights), x + GAP, self.x0 + INDENT)
        self.next_row(height)

    def rollup_tile(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        node: Node,
        parent: str,
        plan: str,
    ) -> None:
        """One child: its light top left with its short name and what it reads under
        it; a child over the rule limit has the aggregate mark, its count and
        ``open``."""
        self.box(x, y, w, h, fill=FIELD_BACKGROUND)
        inner = x + TILE_PAD
        top = y + TILE_PAD / 2
        lights = [describe_light(self.graph, c) for c in node.channels]
        count = len(node.channels)
        if plan == "none":
            right = self.no_rule(inner, top)
            note = _NO_RULE_YET
        elif plan == "live":
            right = self.light(inner, top, lights, rule=ladder(self.graph, lights, node=True))
            note = _READS.format(count, "" if count == 1 else "s")
        else:
            right = self.light(inner, top, lights, rule="aggregate state")
            note = _TILE_OPEN.format(count)
        name = node.label.removeprefix(parent + "/")
        room = x + w - TILE_PAD - (right + GAP)
        below = top + ROW
        if text_width(name, TEXT_SIZE) <= room:
            self.text(right + GAP, top, name, w=text_width(name, TEXT_SIZE))
        else:  # too long for the line: small, in up to two lines broken at its separators
            names = _wrap_label(name, room, SMALL_SIZE, TILE_LINES)
            for i, line in enumerate(names):
                self.small(right + GAP, top + i * SMALL_LINE, line)
            below = max(below, top + len(names) * SMALL_LINE)
        lines = _wrap(note, w - 2 * TILE_PAD, SMALL_SIZE)
        for i, line in enumerate(lines[: TILE_LINES if below == top + ROW else 1]):
            self.small(inner, below + i * SMALL_LINE, line, fg=MUTED_TEXT)
        if node.link is not None:
            self.nav(x, y, w, h, replace(node.link, label=node.label))

    # -- the drawn card -------------------------------------------------------

    def drawn_card(self, sections: list[Section]) -> None:
        """A device's drawn card: the title line, the key tiles, then each section in
        its frame, numbered, the relations as a chain, the operate block and the
        channels grid last; a drawing (picture, probe, calibration) as on the old card."""
        number = 0
        for section in sections:
            first = section.items[0] if section.items else None
            if isinstance(first, CardHead):
                self.title_line(first)
            elif isinstance(first, KeyTile):
                self.key_tiles([i for i in section.items if isinstance(i, KeyTile)])
            elif isinstance(first, CardStrip):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.card_strip, first))
            elif isinstance(first, Chain):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.chain_blocks, first))
            elif isinstance(first, CurvePlot):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.curve_plot, first))
            elif isinstance(first, LimitLadder):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.limit_ladder, first))
            elif isinstance(first, RingStrip):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.ring_strip, first))
            elif isinstance(first, Front):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.card_front, first))
            elif isinstance(first, Operate):
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.operate, first))
            elif isinstance(first, ChannelGrid):
                number += 1
                title = f"{number} · {section.title} ({len(first.rows)})"
                self.framed(title, partial(self.channel_grid, first), note=_GRID_NOTE)
            elif isinstance(first, ChannelCount):
                number += 1
                title = f"{number} · {section.title} ({first.total})"
                self.framed(title, partial(self.channel_count, first))
            elif any(isinstance(i, PartFacts) for i in section.items):
                number += 1
                self.framed(
                    f"{number} · {section.title}", partial(self.part_drawing, section.items)
                )
            elif isinstance(first, Picture | Probe):
                self.section(section)  # a drawing heads itself, by its view
            elif isinstance(first, CardFoot):
                self.card_foot(first)
            elif section.title in _FACT_SECTIONS:
                number += 1
                self.framed(f"{number} · {section.title}", partial(self.fact_lines, section.items))
            else:
                number += 1
                self.section(Section(f"{number} · {section.title}", section.items))

    def card_foot(self, foot: CardFoot) -> None:
        """Muted lines under the last section, no frame and no number: where the
        device is (opening the place), the sets it is in (each opening its set
        view), and its sources, each statement's full provenance on hover."""
        x = self.small(self.x0, self.y, _WHERE_WORD, fg=MUTED_TEXT) + WORD_GAP
        for n, (text, link) in enumerate(foot.where):
            if n:  # "<a> or <b>" reads on; anything else after a dot
                join = " " if text.startswith(_OR) else _FOOT_JOIN
                x = self.small(x, self.y, join, fg=MUTED_TEXT)
            room = self.x0 + STRIP_WIDTH - x
            if room < text_width(_ELLIPSIS, SMALL_SIZE):
                break
            right = self.small(x, self.y, _cut(text, room, SMALL_SIZE), fg=MUTED_TEXT)
            if link is not None:
                self.nav(x, self.y, right - x, SMALL_LINE, link)
            x = right
        self.y += SMALL_LINE + ROW_GAP
        if foot.sets:
            x = self.small(self.x0, self.y, _SETS_WORD, fg=MUTED_TEXT) + WORD_GAP
            for n, link in enumerate(foot.sets):
                if n:
                    x = self.small(x, self.y, _FOOT_JOIN, fg=MUTED_TEXT)
                room = self.x0 + STRIP_WIDTH - x
                if room < text_width(_ELLIPSIS, SMALL_SIZE):
                    break
                right = self.small(x, self.y, _cut(link.label, room, SMALL_SIZE), fg=MUTED_TEXT)
                self.nav(x, self.y, right - x, SMALL_LINE, link)
                x = right
            self.y += SMALL_LINE + ROW_GAP
        line = _cut(f"{_SOURCES_WORD} {foot.sources}", STRIP_WIDTH, SMALL_SIZE)
        right = self.small(self.x0, self.y, line, fg=MUTED_TEXT)
        if foot.provenance:
            shown = list(foot.provenance[:TOOLTIP_LINES])
            if len(foot.provenance) > TOOLTIP_LINES:
                shown.append(f"and {len(foot.provenance) - TOOLTIP_LINES} more")
            props = {
                "fill": False,
                "lineColor": TRANSPARENT,
                "lineWidth": 0,
                _TOOLTIP: "\n".join(shown),
            }
            self.s.widget("rectangle", self.x0, self.y, right - self.x0, SMALL_LINE, props)
        self.y += SMALL_LINE + PANEL_GAP

    def fact_lines(self, items: Sequence[object]) -> None:
        """Where a device is and what it is read from, one muted line each: the place,
        each placement claim with its unit and source, each pose, each source; a
        link opens its screen, a contested or inferred claim says so."""
        for item in items:
            link: Link | None = None
            if isinstance(item, Placement):
                link = item.link
                unit = f"unit {item.unit}" if item.unit is not None else None
                source = _citation_line(item.citation) if item.citation else None
                text = " · ".join(t for t in (item.link.label, unit, source) if t)
            elif isinstance(item, Link):
                link = item
                text = (
                    f"{item.label}: {item.target.id}"
                    if item.label != item.target.id
                    else item.label
                )
            elif isinstance(item, Fact):
                text = f"{item.label} · {item.text}" if item.text != item.label else item.label
                if item.caveat:
                    text += f" ({item.caveat})"
            else:
                continue
            marks = [m for m in (_standing_word(link or item),) if m]
            if marks:
                text += " · " + ", ".join(marks)
            line = _cut(text, STRIP_WIDTH, SMALL_SIZE)
            right = self.small(self.x0, self.y, line, fg=MUTED_TEXT)
            if link is not None:
                self.nav(self.x0, self.y, right - self.x0, SMALL_LINE, link)
            self.y += SMALL_LINE + ROW_GAP

    def framed(self, title: str, draw: Callable[[], None], *, note: str | None = None) -> None:
        """A section's frame and head around what *draw* draws (as :meth:`section`)."""
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        head_y = self.y
        self.heading(title)
        if note:
            x = self.x0 + text_width(title.upper(), HEADING_SIZE) + WORD_GAP
            self.small(x, head_y, note, fg=MUTED_TEXT)
        draw()
        self.y += PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())  # under what it frames
        self.y += PANEL_GAP

    def title_line(self, head: CardHead) -> None:
        """The device's name, large; its class and recipe beside it, muted; the
        recipe's sentence under it."""
        name = self.view.id.id
        x = self.text(MARGIN, self.y, name, size=CARD_NAME_SIZE, h=CARD_NAME_LINE)
        self.text(x + WORD_GAP, self.y + GAP, head.kind, size=CARD_KIND_SIZE, fg=MUTED_TEXT)
        self.y += CARD_NAME_LINE
        if head.nowhere:
            self.small(MARGIN, self.y, _NOWHERE, fg=MUTED_TEXT)
            self.y += SMALL_LINE
        if head.by_whole is not None:
            self.small(MARGIN, self.y, _BY_WHOLE.format(*head.by_whole), fg=MUTED_TEXT)
            self.y += SMALL_LINE
        if head.description:
            for line in _wrap(head.description, STRIP_WIDTH, CARD_DESCRIPTION_SIZE):
                self.text(
                    MARGIN,
                    self.y,
                    line,
                    size=CARD_DESCRIPTION_SIZE,
                    h=DESCRIPTION_LINE,
                    fg=MUTED_TEXT,
                )
                self.y += DESCRIPTION_LINE
        self.y += GAP

    def key_tiles(self, tiles: Sequence[KeyTile]) -> None:
        """The top row: one tile per value, across the frames' width."""
        left = MARGIN - PANEL_PAD
        total = STRIP_WIDTH + 2 * PANEL_PAD
        slots = max(len(tiles), KEY_TILES_WIDE)  # a lone tile keeps a tile's width
        w = (total - (slots - 1) * KEY_TILE_GAP) / slots
        for i, tile in enumerate(tiles):
            with self.s.item("key-tile", None, i):
                self.key_tile(left + i * (w + KEY_TILE_GAP), w, tile)
        self.y += KEY_TILE_HEIGHT + 2 * GAP

    def key_tile(self, x: float, w: float, tile: KeyTile) -> None:
        """One tile of the top row: its title and its value or figure."""
        self.box(x, self.y, w, KEY_TILE_HEIGHT, fill=PANEL_BACKGROUND)
        inner = x + TILE_PAD + GAP / 2
        self.text(
            inner,
            self.y + GAP,
            tile.title.upper(),
            size=HEADING_SIZE,
            h=SMALL_LINE + 2,
            fg=MUTED_TEXT,
        )
        room = w - 2 * TILE_PAD - GAP
        if tile.channels:  # a figure over every channel of the role
            self.key_figure(inner, self.y + KEY_VALUE_TOP, room, tile)
            return
        self.shown_value(
            inner, self.y + KEY_VALUE_TOP, room, tile.channel, tile.shown_as, tile.words,
            unit=tile.unit, scale=tile.scale, size=KEY_VALUE_SIZE, h=KEY_VALUE_LINE,
            note_below=KEY_TILE_HEIGHT - KEY_VALUE_TOP - SMALL_LINE,
        )  # fmt: skip

    def key_figure(self, x: float, y: float, w: float, tile: KeyTile) -> None:
        """A key tile's figure over every channel of its role: "k of N" true, the
        largest, the smallest or the sum, from one formula. A count reads as one:
        "of N" follows the figure on its baseline. Under the figure is one slot:
        while any member has no value the slot says so, and the figure is a dash,
        so no part is shown as whole; but a largest or smallest is the extreme of
        the members that report (:func:`partial_extreme`), drawn muted, as a
        bound, where the formula fits the engine. With values, the largest's or smallest's
        slot names the member that holds it, or says "highest of N" past
        :data:`.describe.HOLDER_MAX` members. A count's slot is a row of small
        lamps, one per member in order, where they fit; each shows its own
        no value by a dashed outline, so the row needs no note, and a member
        with no value is not counted, so the figure is the lit lamps."""
        members = tile.channels
        keys = [f"c{i}" for i in range(len(members))]
        tokens = [f"{{{key}}}" for key in keys]
        counted = tile.shown_as == "count_true"

        def shaped(value: str) -> str:
            scaled = f"({value}) / {_number(tile.scale)}" if tile.scale else value
            shape = (
                fixed_expression("{v}", tile.precision)
                if tile.precision is not None
                else significant_expression("{v}")
            )
            return shape.replace("{v}", f"({scaled})")

        partial = False  # the figure is the extreme of the members that report
        if counted:
            figure = "+".join(f"1*(abs({t})>0)" for t in tokens)  # no value: no count
        elif tile.shown_as == "sum":
            figure = shaped(" + ".join(tokens))
        else:  # max or min: a reduction over one tuple
            figure = partial_extreme(tile.shown_as, tokens, shaped)
            partial = len(figure) <= RULE_EXPRESSION_LIMIT
            if not partial:  # too long to pass over a missing member: a dash then
                figure = shaped(f"{tile.shown_as}(({', '.join(tokens)},))")
        bindings = dict(zip(keys, members, strict=True))
        reads = frozenset(members)
        every: dict[str, Any] = {  # the bare largest member: a number while all report
            "name": "",
            "kind": "scalar",
            "expression": f"max(({', '.join(tokens)},))",
            "bindings": bindings,
        }
        if len(figure) > RULE_EXPRESSION_LIMIT or not self.budget.take([(1 + partial, reads)]):
            self.small(x, y, _NEEDS_SERVER, fg=MUTED_TEXT)
            return
        name = f"figure-{len(self.s.formulas)}"
        self.s.formulas.append(
            {"name": name, "kind": "scalar", "expression": figure, "bindings": bindings}
        )
        every["name"] = f"every-{len(self.s.formulas)}"
        pv = _FORMULA_PREFIX + name
        high = KEY_TILE_HEIGHT - KEY_VALUE_TOP - SMALL_LINE - KEY_NOTE_MARGIN
        if counted:  # the widest count it can print, then "of N" right after it
            beside = f"of {len(members)}"
            number_w = text_width("0" * len(str(len(members))), KEY_VALUE_SIZE)
            beside_x = x + number_w + ROW_GAP
        else:  # the unit in the unit slot at the tile's right
            beside = tile.unit or ""
            beside_w = text_width(beside, SMALL_SIZE) if beside else 0
            number_w = max(w - beside_w - GAP, 0)
            beside_x = x + number_w + GAP
        self.s._rules += 1
        rule = {
            "id": f"r-{self.s._rules:04d}",
            "name": f"{tile.title} figure",
            "targetProperty": "text",
            "default": NO_VALUE,
            "formula": pv,
        }
        rules: list[dict[str, Any]] = [rule]
        if partial:  # muted while a member is missing: the figure is then a bound
            self.s._rules += 1
            rules.append(
                {
                    "id": f"r-{self.s._rules:04d}",
                    "name": f"{tile.title} figure, every member reporting",
                    "targetProperty": "foregroundColor",
                    "pvs": [{"name": _FORMULA_PREFIX + every["name"], "trigger": True}],
                    "conditions": [{"expression": "{0} == {0}", "value": PAGE_TEXT}],
                    "default": MUTED_TEXT,
                }
            )
            self.s.formulas.append(every)
        props = {
            "text": NO_VALUE,
            "fontSize": KEY_VALUE_SIZE,
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
            "align": "left",
            _TOOLTIP: _over_all_tooltip(tile),
        }
        self.s.widget("text-label", x, y, number_w, high, props, rules=rules)
        if beside:
            top = on_baseline(y, high, KEY_VALUE_SIZE, SMALL_LINE, SMALL_SIZE)
            self.small(beside_x, top, beside, fg=MUTED_TEXT)
        below = y + high
        if counted and self.lamp_row(x, below, w, tile):
            return
        # the slot's one line: a missing member while one has no value; with values
        # nothing, or "highest of N" where the tile names no holder. It reads a
        # formula of its own, the bare largest member: the viewer inlines a
        # formula into each condition, and the shaped figure is too long for that
        held = tile.shown_as in _HOLDS and not tile.names
        shown = _HOLDS[tile.shown_as].format(len(members)) if held else ""
        missing = {
            "id": f"r-{self.s._rules + 1:04d}",
            "name": f"{tile.title} every member reports",
            "targetProperty": "text",
            "pvs": [{"name": _FORMULA_PREFIX + every["name"], "trigger": True}],
            "conditions": [{"expression": "{0} == {0}", "value": shown}],
            "default": _A_MEMBER_MISSING.format(len(members)),
        }
        if len(engine_expression(missing, {every["name"]: every})) > RULE_EXPRESSION_LIMIT:
            return  # too many members to say which way: the figure's dash says it alone
        if not partial:
            self.s.formulas.append(every)
        self.s._rules += 1
        note = {
            "text": _A_MEMBER_MISSING.format(len(members)),
            "fontSize": SMALL_SIZE,
            "foregroundColor": MUTED_TEXT,
            "backgroundColor": TRANSPARENT,
            "align": "left",
        }
        if tile.names:  # the holder on the left, the note at the right beside it
            note_w = text_width(_A_MEMBER_MISSING.format(len(members)), SMALL_SIZE)
            self.s.widget(
                "text-label", x + w - note_w, below, note_w, SMALL_LINE, note, rules=[missing]
            )
            self.holder_lines(x, below, w, tile, w - note_w - GAP)
        else:
            self.s.widget("text-label", x, below, w, SMALL_LINE, note, rules=[missing])

    def lamp_row(self, x: float, y: float, w: float, tile: KeyTile) -> bool:
        """A count's slot: one small lamp per member, in order, each with the
        member's name as its tooltip; whether it was drawn. A lamp shows what the
        count counts: filled while its member is not zero, an outline while it is
        zero, unknown with no value; so the lit lamps are the figure. Not drawn
        when the lamps do not fit *w* at the mosaic's size, or their rules not the
        screen's budget."""
        members = tile.channels
        step = MOSAIC_LIGHT + MOSAIC_GAP
        if len(members) * step - MOSAIC_GAP > w:
            return False
        props = symbols.LIGHT_PROPS
        if not self.budget.take([(len(props), frozenset({m})) for m in members]):
            return False
        top = y + (SMALL_LINE - MOSAIC_LIGHT) / 2
        names = tile.names or tuple(members)
        for i, member in enumerate(members):
            self.count_lamp(x + i * step, top, member, names[i])
        return True

    def count_lamp(self, x: float, top: float, channel: str, name: str) -> None:
        """One lamp of a count's row at ``(x, top)``: its shape drawn unknown,
        with a rule per prop over *channel*: counted (not zero) filled in the
        page's text colour, zero an outline, no value unknown's dashed outline.
        The lamp is smaller than a line of the least text, so it carries no
        mark: its outline says unknown (:data:`COUNTED_LAMP`)."""
        looks = (COUNTED_LAMP, symbols.look(symbols.NONE), symbols.look("unknown"))
        counted, zero, unknown = looks
        shape, _mark = symbols.light(x, top, MOSAIC_LIGHT, "unknown")
        shape["props"][_TOOLTIP] = name

        def rule(prop: str) -> dict[str, Any]:
            if counted[prop] == zero[prop]:  # any number: one condition
                conditions = [{"expression": "{0} == {0}", "value": counted[prop]}]
            else:
                conditions = [
                    {"expression": "abs({0}) > 0", "value": counted[prop]},  # no NaN
                    {"expression": "{0} == 0", "value": zero[prop]},
                ]
            self.s._rules += 1
            return {
                "id": f"r-{self.s._rules:04d}",
                "name": f"{name} counted {prop}",
                "targetProperty": prop,
                "pvs": [{"name": channel, "trigger": True}],
                "conditions": conditions,
                "default": unknown[prop],
            }

        self.s.add(shape, rules=[rule(p) for p in symbols.LIGHT_PROPS], tags=[_need("state light")])

    def holder_lines(self, x: float, y: float, w: float, tile: KeyTile, short_w: float) -> None:
        """Under a largest or smallest figure, the member that holds it: one line
        per member, each parked at :data:`AWAY` and moved into the slot by an
        ``x`` rule while its member is the first that holds the extreme among the
        members with a value (:func:`holder_condition`). So one shows at a time,
        a member with no value never removes it, and none shows before the
        channels answer, when the viewer applies no rule. While a member has no
        value the line says the name alone, in *short_w*, and the note beside it
        says one has no value. A name too long for the line is cut so the part
        that tells it from the other members survives (:func:`_cut_apart`); each
        line's tooltip is the member's whole name. Left out when the screen's
        rules do not fit."""
        members = tile.channels
        n = len(members)
        every = " and ".join(f"{{{j}}} == {{{j}}}" for j in range(n))
        lines: list[tuple[str, str, str]] = []
        for k, short in enumerate(tile.names):
            test = holder_condition(tile.shown_as, k, n)
            if max(len(test), len(every)) > RULE_EXPRESSION_LIMIT:
                return
            held = _HOLDS[tile.shown_as].format(n) + " · "
            room = w - text_width(held, SMALL_SIZE) + TEXT_PAD
            text = held + _cut_apart(short, tile.names, room, SMALL_SIZE)
            lines.append((test, text, _cut_apart(short, tile.names, short_w, SMALL_SIZE)))
        if not self.budget.take([(2 * len(lines), frozenset(members))]):
            return
        pvs = [{"name": m, "trigger": True} for m in members]
        for k, (test, text, alone) in enumerate(lines):
            self.s._rules += 2
            rules = [
                {
                    "id": f"r-{self.s._rules - 1:04d}",
                    "name": f"{tile.title} held by {members[k]}",
                    "targetProperty": "x",  # geometry: every widget type honours it
                    "pvs": pvs,
                    "conditions": [{"expression": test, "value": _pixel(x)}],
                    "default": AWAY,
                },
                {
                    "id": f"r-{self.s._rules:04d}",
                    "name": f"{tile.title} held by {members[k]}, every member reporting",
                    "targetProperty": "text",
                    "pvs": pvs,
                    "conditions": [{"expression": every, "value": text}],
                    "default": alone,
                },
            ]
            props = {
                "text": text,
                "fontSize": SMALL_SIZE,
                "foregroundColor": MUTED_TEXT,
                "backgroundColor": TRANSPARENT,
                "align": "left",
                _TOOLTIP: "\n".join(dict.fromkeys((tile.names[k], members[k]))),
            }
            self.s.widget("text-label", AWAY, y, w, SMALL_LINE, props, rules=rules)

    def shown_value(
        self,
        x: float,
        y: float,
        w: float,
        channel: str | None,
        shown_as: str,
        words: Sequence[tuple[str, str]],
        *,
        unit: str | None = None,
        scale: float | None = None,
        size: float = TEXT_SIZE,
        h: float = ROW,
        text: str | None = None,
        note_below: float | None = None,
    ) -> None:
        """One shown value: a number with its unit, a state word, a lamp and its word,
        a stated text, or a dashed chip "not served yet".

        With *note_below* (the height the number then takes), a note that would leave
        the number less than a value's width beside it goes on the line under it."""
        if shown_as == "text":
            self.text(x, y, text or "", size=size, w=w, h=h)
            return
        if channel is None:
            inset = (ROW - CHIP_HEIGHT_SMALL) / 2
            chip_w = min(w, text_width(_NOT_SERVED, SMALL_SIZE) + 2 * GAP)
            self.box(x, y + inset, chip_w, CHIP_HEIGHT_SMALL, line=MUTED_TEXT, dashed=True)
            self.text(
                x + GAP, y + inset, _NOT_SERVED, size=SMALL_SIZE, h=CHIP_HEIGHT_SMALL, fg=MUTED_TEXT
            )
            return
        if shown_as == "lamp":
            top = y + (h - LIGHT_SIZE) / 2
            if meaning_stated(self.graph, channel):  # else the neutral light
                x = self.light(x, top, [describe_light(self.graph, channel)])
            else:
                x = self.no_rule(x, top)
            x += GAP
            w -= LIGHT_SIZE + GAP
            shown_as = "word"
        if shown_as == "word":
            if words and self.budget.take([(1, frozenset({channel}))]):
                self.word(x, y, w, h, channel, words, size=size)
                return
            if not words:  # a number, and the note that no word is stated for it, if it fits
                note_w = text_width(_MEANING_NOT_STATED, SMALL_SIZE)
                room = w - note_w - GAP
                if note_below is not None and room < VALUE_WIDTH:  # the number whole
                    self.number(x, y, w, note_below, channel, None, None, size)
                    self.small(x, y + note_below, _MEANING_NOT_STATED, fg=MUTED_TEXT)
                    return
                number_w = min(VALUE_WIDTH, room) if room >= GRID_VALUE / 2 else w
                right = self.number(x, y, number_w, h, channel, None, None, size)
                if number_w < w:
                    self.text(right + GAP, y, _MEANING_NOT_STATED, size=SMALL_SIZE, w=note_w, h=h, fg=MUTED_TEXT)  # fmt: skip
                return
        self.number(x, y, w, h, channel, unit, scale, size)

    def number(
        self, x, y, w, h, channel: str, unit: str | None, scale: float | None, size: float
    ) -> float:
        """A live number: a label of the channel, or with a divisor a formula's text
        and the unit beside it. Returns the right edge."""
        if scale not in (None, 1.0) and scale:
            expression = significant_expression(f"({{v}} / {scale:g})")
            uw = text_width(unit or "", size) + GAP if unit else 0
            self.formula_text(x, y, w - uw, h, channel, expression, size=size, fg=PAGE_TEXT, bg=TRANSPARENT, align="left")  # fmt: skip
            if unit:
                self.text(x + w - uw + GAP / 2, y, unit, size=size, h=h)
            return float(x + w)
        # the widest text it can print, with its unit, at the largest size that fits w
        widest, word = self.widest_value(channel)
        printed = (
            widest if word or not self.unit_of(channel) else f"{widest} {self.unit_of(channel)}"
        )
        need = shown_width(printed, size)
        if need > w and size > SMALL_SIZE:
            size = max(SMALL_SIZE, math.floor(size * (w - NUMBER_PAD) / (need - NUMBER_PAD)))
        props = {
            "pv": channel,
            "fontSize": size,
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
            "showUnits": True,
            "alarmSensitive": True,
            "alarmSensitiveBorder": False,
            "align": "left",
        }
        rules = (
            self.precision(props, "{v}", {"v": channel}) if not self.prints_whole(channel) else []
        )
        if self.prints_whole(channel) or self.enumerated(channel):
            props["precision"] = 0
        self.s.widget("pv-label", x, y, w, h, props, rules=rules or None)
        return float(x + w)

    def word(self, x, y, w, h, channel: str, words: Sequence[tuple[str, str]], *, size) -> None:
        """A state shown as its stated word: a label whose text a rule sets from the
        value; the dash while no value holds."""
        self.s._rules += 1
        rule = {
            "id": f"r-{self.s._rules:04d}",
            "name": f"{channel} word",
            "targetProperty": "text",
            "pvs": [{"name": channel, "trigger": True}],
            "conditions": [
                {"expression": f"{{0}} == {value}", "value": word} for value, word in words
            ],
            "default": NO_VALUE,
        }
        props = {
            "text": NO_VALUE,
            "fontSize": size,
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
            "align": "left",
            _TOOLTIP: channel,
        }
        self.s.widget("text-label", x, y, w, h, props, rules=[rule])

    def _strip_nodes(self, member: Any, x: float, y: float, scale: float) -> list[dict[str, Any]]:
        """A card strip member's symbol at (*x*, *y*): a bend a box round the pipe
        its stated length long, cut at the strip's ends, any other its symbol."""
        if member.symbol != "bend":
            return symbols.draw(member.symbol, x, y, unit=CARD_SYMBOL_UNIT)
        half, h = CARD_DIPOLE_M * scale / 2, CARD_DIPOLE_HEIGHT
        left = max(x - half, self.x0)
        right = max(min(x + half, self.x0 + STRIP_WIDTH), left)
        props: dict[str, Any] = {
            "lineColor": symbols.FOCUS if member.focus else PAGE_TEXT,
            "lineWidth": BLOCK_LINE,
            "fill": False,
        }
        geometry = {"x": left, "y": y - h / 2, "width": right - left, "height": h}
        return [{"type": "rectangle", "props": props, "geometry": geometry}]

    def card_strip(self, strip: CardStrip) -> None:
        """The beam pipe across the card, to scale over the strip's window; each member
        as its symbol on the pipe or in its lane above or below it, its name and its
        value; the device in the focus colour; metre ticks under it; a dashed bracket
        where only a place around the device is placed."""
        lo, hi = strip.window
        scale = STRIP_WIDTH / max(hi - lo, CARD_LEAST_SPAN)  # px per m

        def at(s_m: float) -> float:
            return self.x0 + (s_m - lo) * scale

        # Every member's symbol, name and value, placed first with the pipe's axis at 0:
        # a name or a value that would cover another's moves one line further out,
        # so neighbours closer together than their names never draw over each other.
        placed: list[tuple[float, float, float, float]] = []  # boxes taken: x0, y0, x1, y1

        def clear(x0: float, y0: float, w: float, h: float) -> bool:
            return all(x0 + w <= a or b <= x0 or y0 + h <= c or d <= y0 for a, c, b, d in placed)

        def free(x0: float, y0: float, w: float, h: float, step: float) -> float:
            for _ in range(_STRIP_TRIES):
                if clear(x0, y0, w, h):
                    break
                y0 += step
            placed.append((x0, y0, x0 + w, y0 + h))
            return y0

        drawn: list[tuple[Any, float, float, list[float]]] = []
        for member in strip.members:
            x = at(member.s_m)
            y = 0.0
            if member.lane != "beam":
                y = CARD_LANE if member.lane == "below" else -CARD_LANE
            nodes = self._strip_nodes(member, x, y, scale)
            left = min(n["geometry"]["x"] for n in nodes)
            top = min(n["geometry"]["y"] for n in nodes)
            right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in nodes)
            bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in nodes)
            placed.append((left, top, right, bottom))
            drawn.append((member, x, y, [left, top, right, bottom]))
        texts: list[tuple[Any, float, float, float, float | None, float]] = []
        for member, x, _y, (_left, top, _right, bottom) in drawn:
            label = _HERE.format(member.label) if member.focus else member.label
            lw = min(text_width(label, SMALL_SIZE), CARD_STRIP_VALUE * 2)
            lx = min(max(x - lw / 2, self.x0), self.x0 + STRIP_WIDTH - lw)
            up = member.lane == "above" or member.lane == "beam" and member.symbol == "bend"
            sign = -1 if up else 1
            ty = top - SMALL_LINE - ROW_GAP if up else bottom + ROW_GAP
            ty = free(lx, ty, lw, SMALL_LINE, sign * SMALL_LINE)
            vx: float | None = None
            vy = 0.0
            if member.channel is not None and member.symbol != "bend":
                vx = min(
                    max(x - CARD_STRIP_VALUE / 2, self.x0), self.x0 + STRIP_WIDTH - CARD_STRIP_VALUE
                )
                vy = free(
                    vx, ty - ROW if up else ty + SMALL_LINE, CARD_STRIP_VALUE, ROW, sign * ROW
                )
            texts.append((member, lx, ty, lw, vx, vy))
        highest = min([b for _a, b, _c, _d in placed] + [-CARD_ABOVE_ROOM])
        lowest = max([d for _a, _b, _c, d in placed] + [CARD_BELOW_ROOM - ROW])
        axis = self.y - highest
        self.box(self.x0, axis - CARD_PIPE / 2, STRIP_WIDTH, CARD_PIPE, fill=FIELD_BACKGROUND)
        if strip.bracket is not None:
            b0, b1, _label = strip.bracket
            left, right = at(max(b0, lo)), at(min(b1, hi))
            reach = CARD_LANE - CARD_PIPE  # clear of the lanes' symbols
            self.box(left, axis - reach, right - left, 2 * reach, line=symbols.FOCUS, dashed=True)
        dipoles = False
        for member, x, y, (left, top, right, bottom) in drawn:
            if member.lane != "beam":
                self.line([(x, axis), (x, axis + y)], MUTED_TEXT)  # its stem to the pipe
            dipoles = dipoles or member.symbol == "bend"
            for node in self._strip_nodes(member, x, axis + y, scale):
                if member.focus:
                    node["props"]["lineColor"] = symbols.FOCUS
                self.s.add(node)
            if member.link is not None:
                self.nav(left, axis + top, right - left, bottom - top, member.link)
        for member, lx, ty, lw, vx, vy in texts:
            label = _HERE.format(member.label) if member.focus else member.label
            colour = symbols.FOCUS if member.focus else PAGE_TEXT
            self.text(lx, axis + ty, _cut(label, lw, SMALL_SIZE), size=SMALL_SIZE, w=lw, h=SMALL_LINE, fg=colour)  # fmt: skip
            if vx is not None:
                self.field(vx, axis + vy, member.channel, w=CARD_STRIP_VALUE, units=True, within="unit")  # fmt: skip
        ticks = axis + lowest + ROW
        step = 1.0
        while step * scale < CARD_TICK_LEAST:
            step *= 2
        first = math.ceil(lo / step) * step
        mark = first
        while mark <= hi + CARD_LEAST_SPAN:
            x = at(mark)
            self.line([(x, ticks), (x, ticks + CARD_TICK)], MUTED_TEXT)
            word = f"{mark:g} m"
            ww = text_width(word, SMALL_SIZE)
            self.small(min(max(x - ww / 2, self.x0), self.x0 + STRIP_WIDTH - ww), ticks + CARD_TICK, word, fg=MUTED_TEXT)  # fmt: skip
            mark += step
        self.y = ticks + CARD_TICK + SMALL_LINE + GAP
        if strip.bracket is not None:  # under the ticks, from the bracket's left
            b0, _b1, label = strip.bracket
            lw = text_width(label, SMALL_SIZE)
            lx = min(max(at(max(b0, lo)), self.x0), self.x0 + STRIP_WIDTH - lw)
            self.text(lx, self.y, label, size=SMALL_SIZE, w=lw, h=SMALL_LINE, fg=symbols.FOCUS)
            self.y += SMALL_LINE + ROW_GAP
        notes = [strip.caption or "", _DIPOLE_DRAWN.format(CARD_DIPOLE_M) if dipoles else ""]
        line = "; ".join(n for n in notes if n)
        if line:
            for text in _prose(line, STRIP_WIDTH, SMALL_SIZE):
                self.small(self.x0, self.y, text, fg=MUTED_TEXT)
                self.y += SMALL_LINE

    def curve_plot(self, curve: CurvePlot) -> None:
        """The curve's measured points as dots joined by a line, in a frame with the
        least and most x and y at its corners and the axes' units; with no point,
        a dashed frame and the note."""
        x0, top = self.x0 + BLOCK_LABEL, self.y
        if not curve.points:
            self.box(x0, top, CURVE_WIDTH, CURVE_HEIGHT / 2, line=MUTED_TEXT, dashed=True)
            self.text(x0 + GAP, top, curve.note or "", size=SMALL_SIZE, w=CURVE_WIDTH - 2 * GAP, h=CURVE_HEIGHT / 2, fg=MUTED_TEXT)  # fmt: skip
            self.y = top + CURVE_HEIGHT / 2 + ROW_GAP
            return
        xs, ys = [p[0] for p in curve.points], [p[1] for p in curve.points]
        lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)

        def at(point: tuple[float, float]) -> tuple[float, float]:
            fx = (point[0] - lo_x) / ((hi_x - lo_x) or 1)
            fy = (point[1] - lo_y) / ((hi_y - lo_y) or 1)
            return x0 + fx * CURVE_WIDTH, top + (1 - fy) * CURVE_HEIGHT

        self.box(x0, top, CURVE_WIDTH, CURVE_HEIGHT, line=BOX_BORDER, fill=PANEL_BACKGROUND)
        if len(curve.points) > 1:
            self.line([at(p) for p in curve.points], symbols.FOCUS, BLOCK_LINE)
        for point in curve.points:
            px, py = at(point)
            props = {"lineColor": symbols.FOCUS, "lineWidth": 1, "fill": True, "fillColor": symbols.FOCUS, _TOOLTIP: f"{point[0]:g}, {point[1]:g}"}  # fmt: skip
            self.s.widget("ellipse", px - CURVE_POINT / 2, py - CURVE_POINT / 2, CURVE_POINT, CURVE_POINT, props)  # fmt: skip
        y_unit, x_unit = curve.y_unit or "", curve.x_unit or ""
        self.small(self.x0, top, f"{hi_y:g} {y_unit}".strip(), fg=MUTED_TEXT)
        self.small(self.x0, top + CURVE_HEIGHT - SMALL_LINE, f"{lo_y:g} {y_unit}".strip(), fg=MUTED_TEXT)  # fmt: skip
        self.y = top + CURVE_HEIGHT + ROW_GAP
        self.small(x0, self.y, f"{lo_x:g} {x_unit}".strip(), fg=MUTED_TEXT)
        hi = f"{hi_x:g} {x_unit}".strip()
        self.small(x0 + CURVE_WIDTH - text_width(hi, SMALL_SIZE), self.y, hi, fg=MUTED_TEXT)
        self.y += SMALL_LINE + ROW_GAP
        for text in (
            f"{len(curve.points)} points from {curve.source}" if curve.source else None,
            curve.note,
        ):
            if text:
                self.small(self.x0, self.y, _cut(text, STRIP_WIDTH, SMALL_SIZE), fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.y += ROW_GAP

    def limit_ladder(self, ladder: LimitLadder) -> None:
        """The set points as marks on one line, evenly in their order (not to
        scale), each its title over it and its live value under it; the live value
        a dot in the gap between the set points it lies between (a rule over the
        value and every set point), hollow while no value is known; the set points
        not served named under it."""
        top = self.y + SMALL_LINE + ROW_GAP
        line_y = top + LADDER_TICK / 2
        m = len(ladder.marks)
        step = STRIP_WIDTH / (m + 1)
        self.line([(self.x0, line_y), (self.x0 + STRIP_WIDTH, line_y)], PAGE_TEXT, BLOCK_LINE)
        for k, (title, channel) in enumerate(ladder.marks):
            x = self.x0 + (k + 1) * step
            self.text(x - step / 2, self.y, _cut(title, step, SMALL_SIZE), size=SMALL_SIZE, w=step, h=SMALL_LINE, fg=MUTED_TEXT, align="center")  # fmt: skip
            self.line([(x, top), (x, top + LADDER_TICK)], symbols.ALARM_MINOR, BLOCK_LINE * 2, tooltip=channel)  # fmt: skip
            self.field(x - VALUE_WIDTH / 2, top + LADDER_TICK + GAP, channel, w=VALUE_WIDTH)
        bottom = top + LADDER_TICK + GAP + ROW + ROW_GAP
        if ladder.value is not None and m:
            bindings = {"v": ladder.value} | {f"t{k}": c for k, (_t, c) in enumerate(ladder.marks)}
            above = " + ".join(f"(1 if {{v}} > {{t{k}}} else 0)" for k in range(m))
            x_expr = f"{_number(self.x0 + step / 2 - LADDER_DOT / 2)} + {_number(step)} * ({above})"
            if self.budget.take([(2, frozenset(bindings.values()))]):
                name = f"ladder-x-{len(self.s.formulas)}"
                self.s.formulas.append(
                    {"name": name, "kind": "scalar", "expression": x_expr, "bindings": bindings}
                )
                self.s._rules += 2
                rules = [
                    {
                        "id": f"r-{self.s._rules - 1:04d}",
                        "name": f"{ladder.value} between its set points",
                        "targetProperty": "x",
                        "default": self.x0,
                        "formula": _FORMULA_PREFIX + name,
                    },
                    {
                        "id": f"r-{self.s._rules:04d}",
                        "name": f"{ladder.value} known",
                        "targetProperty": "fill",
                        "pvs": [{"name": ladder.value, "trigger": True}],
                        "conditions": [{"expression": "{0} == {0}", "value": True}],
                        "default": False,
                    },
                ]
                props = {
                    "lineColor": symbols.FOCUS,
                    "lineWidth": BLOCK_LINE,
                    "fill": False,
                    "fillColor": symbols.FOCUS,
                    _TOOLTIP: f"{ladder.value}: between the set points it lies between",
                }
                self.s.widget("ellipse", self.x0, line_y - LADDER_DOT / 2, LADDER_DOT, LADDER_DOT, props, rules=rules)  # fmt: skip
        self.y = bottom
        if ladder.value is not None:
            x = (
                self.small(self.x0, self.y + (ROW - SMALL_LINE) / 2, _LADDER_NOW, fg=MUTED_TEXT)
                + GAP
            )
            self.field(x, self.y, ladder.value, w=VALUE_WIDTH)
            self.y += ROW + ROW_GAP
        notes = []
        if ladder.value is None:
            notes.append(_LADDER_NO_VALUE)
        if ladder.left_out:
            notes.append(_LADDER_LEFT_OUT.format(", ".join(ladder.left_out)))
        for text in [*notes, *([ladder.caption] if ladder.caption else [])]:
            for line in _prose(text, STRIP_WIDTH, SMALL_SIZE):
                self.small(self.x0, self.y, line, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.y += ROW_GAP

    def part_drawing(self, items: Sequence[object]) -> None:
        """A part's owner drawn with the part in focus, in the left column; beside
        it what is stated about the part (each opening the other thing) and, dashed,
        what is not."""
        top = self.y
        for item in items:
            if isinstance(item, Picture):
                self.picture(item)
        bottom = self.y
        self.y = top
        x = self.x0 + CARD_LEFT if any(isinstance(i, Picture) for i in items) else self.x0
        width = self.x0 + STRIP_WIDTH - x
        for item in items:
            if not isinstance(item, PartFacts):
                continue
            self.small(x, self.y, _STATED, fg=MUTED_TEXT)
            self.y += SMALL_LINE + ROW_GAP
            for fact in item.stated:
                label = fact.label if isinstance(fact, Link) else f"{fact.label}: {fact.text}"
                right = self.text(x, self.y, _cut(label, width, TEXT_SIZE), w=min(width, text_width(label, TEXT_SIZE)))  # fmt: skip
                if isinstance(fact, Link):
                    self.nav(
                        x, self.y, right - x, ROW, replace(fact, label=f"open {fact.target.id}")
                    )
                self.y += ROW
            if not item.stated:
                self.small(x, self.y, _NOT_STATED_TEXT, fg=MUTED_TEXT)
                self.y += SMALL_LINE
            self.y += GAP
            if item.not_stated:
                self.small(x, self.y, _NOT_STATED_HEAD, fg=MUTED_TEXT)
                self.y += SMALL_LINE + ROW_GAP
            for text in item.not_stated:
                self.box(x, self.y, width, ROW, line=MUTED_TEXT, dashed=True)
                self.text(x + GAP, self.y, _cut(text, width - 2 * GAP, SMALL_SIZE), size=SMALL_SIZE, w=width - 2 * GAP, fg=MUTED_TEXT)  # fmt: skip
                self.y += ROW + ROW_GAP
        self.y = max(self.y, bottom)

    def ring_strip(self, ring: RingStrip) -> None:
        """One column per member in its order, wrapping into rows: its name, then the
        value or lamp of its role (a bare outline where it plays none: not served);
        the card's own device in the focus colour, every other opens its card."""
        whole = self.ok_count(ring) if ring.counted else True
        runs = list(ring.places) or [("", len(ring.members))]
        head = RING_PLACE_HEAD if ring.places else 0
        room = STRIP_WIDTH - head
        n = max(k for _place, k in runs) if ring.places else len(ring.members)
        cw = max(RING_COLUMN_LEAST, min(RING_COLUMN_MOST, room / max(n, 1)))
        per = max(1, int(room // cw))
        height = SMALL_LINE + ROW + 2 * GAP
        if any(m.inferred or m.stated for m in ring.members):
            height += SMALL_LINE  # room for the word under the value
        start = 0
        for place, k in runs:  # a row of columns per place, its name heading it
            if place:
                self.small(
                    self.x0, self.y + GAP / 2, _cut(place, head - GAP, SMALL_SIZE), fg=MUTED_TEXT
                )
                lamps = _ring_lamps(ring.members[start : start + k])
                if not whole and lamps:  # the count per place, under its name
                    self.ok_label(self.x0, self.y + GAP / 2 + SMALL_LINE + ROW_GAP, lamps, place)
            self.ring_columns(ring.members[start : start + k], self.x0 + head, cw, per, height)
            start += k
        if ring.caption:
            for text in _prose(ring.caption, STRIP_WIDTH, SMALL_SIZE):
                self.small(self.x0, self.y, text, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.y += ROW_GAP

    def ok_formula(self, channels: Sequence[str]) -> tuple[str, dict[str, str]] | None:
        """How many of *channels*' lights are ok now: a sum of one term per light, its
        arms in order up to its ok arm, first match wins as the light; ``None`` when
        no light states an ok state."""
        terms: list[str] = []
        bindings: dict[str, str] = {}
        for i, channel in enumerate(channels):
            rule = ladder(self.graph, [describe_light(self.graph, channel)])
            if isinstance(rule, str):
                continue
            states = [st for _c, st in rule.arms]
            if "normal" not in states:
                continue  # no ok state stated: it counts none
            arms = rule.arms[: states.index("normal") + 1]  # after the ok arm: none ok
            key = f"m{i}"
            bindings[key] = channel
            term = "0"
            for condition, state in reversed(arms):
                term = f"({int(state == 'normal')} if {condition} else {term})"
            terms.append(term.replace("{0}", f"{{{key}}}"))
        return (" + ".join(terms), bindings) if terms else None

    def ok_label(self, x: float, y: float, channels: Sequence[str], name: str) -> bool:
        """At (*x*, *y*): "n of m" lights of *channels* ok now, from one formula;
        False, and nothing drawn, when it cannot be written (too long, over budget,
        no ok state stated)."""
        found = self.ok_formula(channels)
        if found is None:
            return False
        expression, bindings = found
        if len(expression) > RULE_EXPRESSION_LIMIT or not self.budget.take(
            [(1, frozenset(bindings.values()))]
        ):
            return False
        formula = f"ok-{len(self.s.formulas)}"
        self.s.formulas.append(
            {"name": formula, "kind": "scalar", "expression": expression, "bindings": bindings}
        )
        self.s._rules += 1
        rule = {
            "id": f"r-{self.s._rules:04d}",
            "name": f"{name} count",
            "targetProperty": "text",
            "default": NO_VALUE,
            "formula": _FORMULA_PREFIX + formula,
        }
        w = math.ceil(COUNT_DIGITS * CHAR_WIDTH * TEXT_SIZE + NUMBER_PAD)
        props = {"text": NO_VALUE, "fontSize": TEXT_SIZE, "foregroundColor": PAGE_TEXT,
                 "backgroundColor": TRANSPARENT, "align": "right"}  # fmt: skip
        self.s.widget("text-label", x, y - (ROW - SMALL_LINE) / 2, w, ROW, props, rules=[rule])
        self.small(x + w + GAP, y, f"of {len(channels)}", fg=MUTED_TEXT)
        return True

    def ok_count(self, ring: RingStrip) -> bool:
        """Over a ring-strip: how many of its lamps are ok now, of how many, from one
        formula over the same lamps' rules; when that formula cannot be written, the
        line says the count is per place (each place's row then counts its own) or
        that it needs a server-side aggregate. Returns whether it counted them all."""
        lamps = _ring_lamps(ring.members)
        x = self.small(self.x0, self.y, f"{ring.counted}:", fg=MUTED_TEXT) + GAP
        whole = self.ok_label(x, self.y, lamps, ring.counted or "")
        if not whole:
            if self.ok_formula(lamps) is None:
                note = _NO_OK_STATED
            else:
                note = _PER_PLACE if ring.places else _NEEDS_SERVER
            self.small(x, self.y, f"{note} ({len(lamps)} lamps)", fg=MUTED_TEXT)
        self.y += SMALL_LINE + ROW_GAP
        return whole

    def ring_columns(
        self, members: Sequence[RingMember], x0: float, cw: float, per: int, height: float
    ) -> None:
        """Members as columns from *x0*, *per* to a row, wrapping."""
        for i, member in enumerate(members):
            r, c = divmod(i, per)
            x, y = x0 + c * cw, self.y + r * (height + GAP)
            w = cw - GAP
            if member.focus:
                props = {"lineColor": symbols.FOCUS, "lineWidth": 2, "fill": True, "fillColor": symbols.FOCUS_TINT}  # fmt: skip
                self.s.widget("rectangle", x, y, w, height, props)
            else:
                self.box(x, y, w, height, line=BOX_BORDER, fill=PANEL_BACKGROUND)
            colour = symbols.FOCUS if member.focus else PAGE_TEXT
            self.text(x, y + GAP / 2, _cut(member.label, w, SMALL_SIZE), size=SMALL_SIZE, w=w, h=SMALL_LINE, fg=colour, align="center")  # fmt: skip
            if member.link is None and member.label != member.thing:  # its id on hover
                self.s.widget(
                    "rectangle", x, y, w, SMALL_LINE + GAP,
                    {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0, _TOOLTIP: member.thing},
                )  # fmt: skip
            vy = y + SMALL_LINE + GAP
            if not member.rows:
                self.no_rule(x + (w - LIGHT_SIZE) / 2, vy)
            for row in member.rows[:1]:
                self.shown_value(
                    x + GAP / 2, vy, w - GAP, row.channel, row.shown_as, row.words,
                    unit=row.unit, scale=row.scale, size=SMALL_SIZE, h=ROW,
                )  # fmt: skip
            if member.inferred:  # under its value, its rule on hover
                self.inferred_word(x + GAP / 2, vy + ROW, member.inferred)
            elif member.stated:  # the few stated among inferred ones
                self.small(x + GAP / 2, vy + ROW, _STATED_WORD, fg=MUTED_TEXT)
            why = f" · {_INFERRED_WORD}: {member.inferred}" if member.inferred else ""
            if member.link is not None:
                self.nav(x, y, w, height, replace(member.link, label=f"open {member.thing}{why}"))
        self.y += -(-len(members) // per) * (height + GAP)

    def card_front(self, front: Front) -> None:
        """A device's faces: its connectors at one pitch on a panel body, group by
        group in the unit's own order, wrapping into more lines; under each the cable
        to what is on it (a box that opens its card; dashed "nothing stated" where
        nothing is; every thing, dashed and "contested", where a side has two),
        then the section's values and the connector's own presses. A group with no
        connector stated is an empty body with its note. Then the line of what the
        presses write, the things on no stated port by class, and the caption."""
        slots = sum(max(len(g.connectors), 1) for g in front.groups)
        labelled = any(c.rows or c.controls or c.lamp for g in front.groups for c in g.connectors)
        gutter = FRONT_LABEL_GUTTER if labelled else 0
        room = STRIP_WIDTH - gutter - FRONT_GROUP_GAP * max(len(front.groups) - 1, 0)
        pitch = max(FRONT_PITCH_LEAST, min(FRONT_PITCH_MOST, room / max(slots, 1)))
        width = STRIP_WIDTH - gutter
        line: list[tuple[FrontGroup, list[Connector]]] = []
        x = 0.0
        for group in front.groups:
            if line and (_reading(group) or _reading(line[0][0])):  # a reading: its own line
                self.front_line(line, pitch, gutter)
                line, x = [], 0.0
            if not group.connectors:
                empty = FRONT_EMPTY_WIDTH
                if line and x + FRONT_GROUP_GAP + empty > width:
                    self.front_line(line, pitch, gutter)
                    line, x = [], 0.0
                x += (FRONT_GROUP_GAP if line else 0) + empty
                line.append((group, []))
                continue
            x += FRONT_GROUP_GAP if line else 0
            part: list[Connector] = []
            line.append((group, part))
            for connector in group.connectors:
                if x + pitch > width and (part or len(line) > 1):
                    if not part:
                        line.pop()
                    self.front_line(line, pitch, gutter)
                    part = []
                    rest = replace(group, title=_CONTINUED.format(group.title))
                    line, x = [(rest, part)], 0.0
                part.append(connector)
                x += pitch
        if line:
            self.front_line(line, pitch, gutter)
        if not front.groups:
            self.small(self.x0, self.y, _NO_CONNECTOR, fg=MUTED_TEXT)
            self.y += SMALL_LINE + ROW_GAP
        if front.writes:
            for text in _prose(front.writes, STRIP_WIDTH, SMALL_SIZE):
                self.small(self.x0, self.y, text, fg=MUTED_TEXT)
                self.y += SMALL_LINE
            self.y += ROW_GAP
        for typed in front.entries:  # the typed values per connector, under the front
            self.entry_grid(typed)
        if front.loose:
            self.front_loose(front)
        if front.caption:
            for text in _prose(front.caption, STRIP_WIDTH, SMALL_SIZE):
                self.small(self.x0, self.y, text, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.y += ROW_GAP

    def front_line(
        self, line: list[tuple[FrontGroup, list[Connector]]], pitch: float, gutter: float
    ) -> None:
        """One line of a front: each group's title over its part of the body, the
        connectors, under each what is on it; then the values and presses in bands,
        each band's label at the left. A line of readings (connectors nothing is
        looked for on) is a table instead: no body, socket or cable, the connectors'
        names as column heads, the lamps first."""
        top = self.y
        reading = _reading(line[0][0])
        body_top = top + SMALL_LINE + ROW_GAP
        if reading:
            body_h = SMALL_LINE + ROW_GAP
        else:
            body_h = 2 * FRONT_BODY_PAD + SOCKET_SIZE + GAP + LIGHT_SIZE + SMALL_LINE
        under = body_top + body_h + (0 if reading else FRONT_CABLE)
        x = self.x0 + gutter
        bottom = under
        columns: list[tuple[float, Connector]] = []
        for k, (group, connectors) in enumerate(line):
            x += FRONT_GROUP_GAP if k else 0
            width = pitch * len(connectors) if connectors else FRONT_EMPTY_WIDTH
            title = group.title if group.face in ("", "front") else f"{group.face} · {group.title}"
            room = (
                width if k < len(line) - 1 else self.x0 + STRIP_WIDTH - x
            )  # the last: to the edge
            self.text(x, top, _cut(title, room, SMALL_SIZE), size=SMALL_SIZE, w=room, h=SMALL_LINE, fg=MUTED_TEXT)  # fmt: skip
            if reading:
                rule_y = under - ROW_GAP / 2
                self.line([(x, rule_y), (x + width, rule_y)], BOX_BORDER)
            else:
                self.box(x, body_top, width, body_h, line=BOX_BORDER, fill=PANEL_BACKGROUND)
            if not connectors:
                self.box(x + GAP, body_top + FRONT_BODY_PAD, width - 2 * GAP, SOCKET_SIZE, line=MUTED_TEXT, dashed=True)  # fmt: skip
                self.text(
                    x + GAP, body_top + FRONT_BODY_PAD + SOCKET_SIZE + GAP, group.empty or "",
                    size=SMALL_SIZE, w=width - 2 * GAP, h=SMALL_LINE, fg=MUTED_TEXT,
                )  # fmt: skip
                x += width
                continue
            for connector in connectors:
                if reading:
                    head = connector.label
                    if text_width(head, SMALL_SIZE) > pitch - GAP:  # its own word, as on a body
                        head = head.rsplit(" ", 1)[-1]
                    self.text(
                        x, body_top, _cut(head, pitch - GAP, SMALL_SIZE),
                        size=SMALL_SIZE, w=pitch, h=SMALL_LINE, align="center",
                    )  # fmt: skip
                else:
                    bottom = max(bottom, self.connector(x, body_top, pitch, connector, under))
                columns.append((x, connector))
                x += pitch
        self.y = bottom
        self.front_bands(columns, pitch, reading)
        self.y += ROW_GAP + GAP

    def front_bands(
        self, columns: list[tuple[float, Connector]], pitch: float, reading: bool
    ) -> None:
        """Under a line of connectors: a band per value the section's roles give
        them, then per press, each its label (and unit) at the left and each
        connector's own in its column; a reading's lamps are the first band."""
        w = pitch - GAP
        label_w = FRONT_LABEL_GUTTER - GAP
        mid = (ROW - SMALL_LINE) / 2
        if reading and any(c.lamp for _x, c in columns):
            self.small(self.x0, self.y + mid, _STATE_BAND, fg=MUTED_TEXT)
            for x, connector in columns:
                if connector.lamp is not None:
                    self.shown_value(
                        x + GAP / 2, self.y, w, connector.lamp, "lamp",
                        self.graph_words(connector.lamp), size=SMALL_SIZE, h=ROW,
                    )  # fmt: skip
            self.y += ROW + ROW_GAP
        for cells in _bands(columns):
            label = _band_label([row.label for _x, row in cells])
            unit = next((row.unit for _x, row in cells if row.unit), None)
            title = f"{label} ({unit})" if unit else label
            self.small(self.x0, self.y + mid, _cut(title, label_w, SMALL_SIZE), fg=MUTED_TEXT)
            names = sorted({row.label for _x, row in cells})
            if len(names) > 1 or _cut(title, label_w, SMALL_SIZE) != title:  # whole, on hover
                self.s.widget(
                    "rectangle", self.x0, self.y, label_w, ROW,
                    {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0,
                     _TOOLTIP: "; ".join(names) if len(names) > 1 else title},
                )  # fmt: skip
            for x, row in cells:
                bx = x + GAP / 2
                self.shown_value(
                    bx, self.y, w, row.channel, row.shown_as, row.words, scale=row.scale,
                    size=SMALL_SIZE, h=ROW, text=row.text,
                )  # fmt: skip
                self.s.widget(
                    "rectangle", bx, self.y, w, ROW,
                    {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0,
                     _TOOLTIP: f"{row.label}: {row.channel or row.text or ''}"},
                )  # fmt: skip
            self.y += ROW + ROW_GAP
        presses: dict[str, list[tuple[float, Control]]] = {}
        for x, connector in columns:
            for control in connector.controls:
                presses.setdefault(control.label, []).append((x, control))
        for n, pressed in enumerate(presses.values()):
            if not n:
                self.small(self.x0, self.y + mid, _PRESS_BAND, fg=MUTED_TEXT)
            for x, control in pressed:
                self.press(x + GAP / 2, self.y, control, True, w, ROW)
            self.y += ROW + ROW_GAP

    def connector(
        self, x: float, body_top: float, pitch: float, connector: Connector, under: float
    ) -> float:
        """One connector in its slot: socket, lamp, label on the body; under it the
        cable and what is on it per side (its values and presses are in the bands
        under the line, :meth:`front_bands`). Returns the bottom."""
        cx = x + pitch / 2
        sy = body_top + FRONT_BODY_PAD
        self.box(cx - SOCKET_SIZE / 2, sy, SOCKET_SIZE, SOCKET_SIZE, line=PAGE_TEXT, fill=PANEL_BACKGROUND)  # fmt: skip
        inner = SOCKET_SIZE - 2 * SOCKET_INSET
        self.box(
            cx - inner / 2, sy + SOCKET_INSET, inner, inner / 2, line=PAGE_TEXT, fill=MUTED_TEXT
        )
        ly = sy + SOCKET_SIZE + GAP / 2
        if connector.lamp is not None:
            if meaning_stated(self.graph, connector.lamp):
                self.light(cx - LIGHT_SIZE / 2, ly - (ROW - LIGHT_SIZE) / 2, [describe_light(self.graph, connector.lamp)])  # fmt: skip
            else:
                self.no_rule(cx - LIGHT_SIZE / 2, ly - (ROW - LIGHT_SIZE) / 2)
        label = connector.label if text_width(connector.label, SMALL_SIZE) <= pitch - GAP else (
            connector.label.rsplit(" ", 1)[-1]
        )  # fmt: skip
        self.text(
            x, ly + LIGHT_SIZE + GAP / 2, _cut(label, pitch, SMALL_SIZE), size=SMALL_SIZE,
            w=pitch, h=SMALL_LINE, align="center",
        )  # fmt: skip
        body_bottom = under - FRONT_CABLE
        self.line([(cx, body_bottom), (cx, under)], MUTED_TEXT, BLOCK_LINE)
        y = under
        w = pitch - GAP
        bx = x + GAP / 2
        for side in connector.sides:
            if not side:
                self.box(bx, y, w, FAR_BOX_HEIGHT, line=MUTED_TEXT, dashed=True)
                self.text(bx, y, _NOTHING, size=SMALL_SIZE, w=w, h=FAR_BOX_HEIGHT, fg=MUTED_TEXT, align="center")  # fmt: skip
                y += FAR_BOX_HEIGHT + ROW_GAP
                continue
            contested = len(side) > 1
            for end in side:
                self.far_box(bx, y, w, end, contested)
                y += FAR_BOX_HEIGHT + ROW_GAP
            if contested:
                self.text(bx, y, _CONTESTED, size=SMALL_SIZE, w=w, h=SMALL_LINE, fg=MUTED_TEXT, align="center")  # fmt: skip
                y += SMALL_LINE + ROW_GAP
        return y

    def inferred_word(self, x: float, y: float, rule: str) -> float:
        """The word "inferred", muted, its rule on hover; returns its right edge."""
        w = text_width(_INFERRED_WORD, SMALL_SIZE)
        self.small(x, y, _INFERRED_WORD, fg=MUTED_TEXT)
        self.s.widget(
            "rectangle", x, y, w, SMALL_LINE,
            {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0, _TOOLTIP: rule},
        )  # fmt: skip
        return x + w

    def far_box(self, x: float, y: float, w: float, end: FarEnd, contested: bool) -> None:
        """What is on a connector: its id in two lines (the part before its last
        ``:``, then the rest; with its own port, the id and the port), opening its
        card."""
        if contested:
            self.box(x, y, w, FAR_BOX_HEIGHT, line=symbols.ALARM_MINOR, dashed=True)
        else:
            self.box(x, y, w, FAR_BOX_HEIGHT, line=PAGE_TEXT, fill=PANEL_BACKGROUND)
        head, _, tail = end.thing.rpartition(":")
        first, second = (end.thing, f"port {end.port}") if end.port else (head, tail or end.thing)
        half = FAR_BOX_HEIGHT / 2
        if end.inferred:  # the word on the box, its rule on hover
            first = f"{first} · {_INFERRED_WORD}"
        self.text(x, y + 1, _cut(first, w, SMALL_SIZE), size=SMALL_SIZE, w=w, h=half, fg=MUTED_TEXT, align="center")  # fmt: skip
        self.text(x, y + half - 1, _cut(second, w, SMALL_SIZE), size=SMALL_SIZE, w=w, h=half, align="center")  # fmt: skip
        why = f" · {_INFERRED_WORD}: {end.inferred}" if end.inferred else ""
        if end.link is not None:
            self.nav(x, y, w, FAR_BOX_HEIGHT, replace(end.link, label=f"open {end.thing}{why}"))
        else:
            self.s.widget(
                "rectangle", x, y, w, FAR_BOX_HEIGHT,
                {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0, _TOOLTIP: end.thing + why},
            )  # fmt: skip

    def front_loose(self, front: Front) -> None:
        """Things the face's relations name on no stated port: the note, then per
        class its title and count and its things in columns, each with its value."""
        self.small(self.x0, self.y, front.loose_note or "", fg=MUTED_TEXT)
        self.y += SMALL_LINE + ROW_GAP
        cw = (STRIP_WIDTH - GAP * (LOOSE_COLUMNS - 1)) / LOOSE_COLUMNS
        for cls, things in front.loose:
            self.text(self.x0, self.y, f"{cls} ({len(things)})", size=SMALL_SIZE, h=SMALL_LINE)
            self.y += SMALL_LINE + ROW_GAP
            per = -(-len(things) // LOOSE_COLUMNS)
            top = self.y
            for i, (end, rows) in enumerate(things):
                c, r = divmod(i, per)
                bx, by = self.x0 + c * (cw + GAP), top + r * (FAR_BOX_HEIGHT + ROW_GAP)
                self.box(bx, by, cw, FAR_BOX_HEIGHT, line=PAGE_TEXT, fill=PANEL_BACKGROUND)
                name_w = cw / 2 if rows else cw - GAP
                self.text(bx + GAP, by, _cut(end.thing, name_w, TEXT_SIZE), w=name_w, h=FAR_BOX_HEIGHT)  # fmt: skip
                if end.link is not None:
                    self.nav(
                        bx, by, cw, FAR_BOX_HEIGHT, replace(end.link, label=f"open {end.thing}")
                    )
                for row in rows[:1]:
                    self.shown_value(
                        bx + cw / 2, by + (FAR_BOX_HEIGHT - ROW) / 2, cw / 2 - GAP, row.channel,
                        row.shown_as, row.words, size=SMALL_SIZE,
                    )  # fmt: skip
            self.y = top + per * (FAR_BOX_HEIGHT + ROW_GAP) + GAP

    def chain_blocks(self, chain: Chain) -> None:
        """Blocks left to right, each its title over its box of lines, the arrow and
        its word to the next. Up to :data:`BLOCKS_PER_ROW` share a row, narrowed to
        fit the width (:meth:`chain_rows`); one row is centred, more rows start under
        the first block, the arrow running on from the last block of a row down to
        the first of the next. Nothing is cut: an arrow's word takes the run it needs
        (two lines, above and under the arrow, or its last name segment with the whole
        word on hover, :func:`arrow_lines`), and a title that does not fit its block
        takes two lines, the row's titles all as high as its highest."""
        words = [arrow_lines(b.arrow) if b.arrow else ([], False) for b in chain.blocks]
        needs = [_arrow_need(lines) for lines, _short in words[:-1]]
        rows = self.chain_rows([self.block_width(b) for b in chain.blocks], needs)
        left: tuple[float, float, float, str | None] | None = None  # a row's last block
        n = 0
        for widths, gaps in rows:
            total = sum(widths) + sum(gaps)
            x = self.x0 + (max(0.0, (STRIP_WIDTH - total) / 2) if len(rows) == 1 else 0)
            top = self.y
            band = max(
                _title_lines(b.title, w)
                for b, w in zip(chain.blocks[n : n + len(widths)], widths, strict=True)
            ) * BLOCK_TITLE  # fmt: skip
            ya = top + band + BLOCK_TOP
            if left is not None:  # the arrow on from the row above
                lx, ly, lw, word = left
                mid = top - ROW_GAP
                into = min(gaps[0] if gaps else BLOCK_GAP_LEAST, x - self.x0 + BLOCK_GAP_LEAST)
                path = [
                    (lx + lw / 2, ly),
                    (lx + lw / 2, mid),
                    (x - min(into, x) / 2, mid),  # in the margin, left of the block
                    (x - min(into, x) / 2, ya),
                    (x - ARROW_INSET, ya),
                ]
                self.line(path, PAGE_TEXT, BLOCK_LINE, arrow=True)
                if word:
                    self.small(lx + lw / 2 + GAP, ly + GAP / 2, word, fg=MUTED_TEXT)
            bottom = top
            for k, w in enumerate(widths):
                block = chain.blocks[n]
                lines, short = words[n]
                n += 1
                thing = block.link.target.id if block.link is not None else None
                with self.s.item("chain-block", thing, n - 1):
                    end = self.chain_block(x, top, w, block, band)
                    for other in block.stack:  # the other things the hop reaches, under it
                        end = self.chain_block(x, end + GAP, w, other)
                bottom = max(bottom, end)
                if k != len(widths) - 1:
                    gap = gaps[k]
                    start, stop = x + w + ARROW_INSET / 2, x + w + gap - ARROW_INSET
                    self.line([(start, ya), (stop, ya)], PAGE_TEXT, BLOCK_LINE, arrow=True)
                    for i, line in enumerate(lines):  # one over the arrow, one under it
                        ww = text_width(line, SMALL_SIZE)
                        ly = ya - SMALL_LINE - ARROW_INSET if i == 0 else ya + ARROW_INSET
                        self.small(x + w + (gap - ww) / 2, ly, line, fg=MUTED_TEXT)
                    if short and block.arrow:  # the whole word on hover
                        self.s.widget(
                            "rectangle", x + w, ya - SMALL_LINE - ARROW_INSET, gap,
                            len(lines) * SMALL_LINE + 2 * ARROW_INSET,
                            {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0,
                             _TOOLTIP: block.arrow},
                        )  # fmt: skip
                else:
                    left = (x, end, w, block.arrow)
                x += w + (gaps[k] if k < len(gaps) else 0)
            self.y = bottom + GAP + (2 * ROW_GAP if n < len(chain.blocks) else 0)
        if chain.foot:
            for line in _prose(chain.foot, STRIP_WIDTH, SMALL_SIZE):
                self.small(self.x0, self.y, line, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.y += ROW_GAP

    @staticmethod
    def chain_rows(
        widths: list[float], needs: list[float] | None = None
    ) -> list[tuple[list[float], list[float]]]:
        """The rows of a chain: each its blocks' widths and the run of each arrow
        between them. A row takes up to :data:`BLOCKS_PER_ROW` blocks at their own
        widths and :data:`BLOCK_GAP`; a row too wide gives each arrow the run its
        word needs (*needs*, by arrow; :data:`BLOCK_GAP_LEAST` at least) and narrows
        its blocks in proportion, none below :data:`BLOCK_NARROWEST` (a word longer
        than :data:`BLOCK_GAP` lengthens its arrow in any row); a row that
        still does not fit gives its last block to the next row."""
        need = list(needs or [])
        need += [BLOCK_GAP_LEAST] * (len(widths) - len(need))
        rows: list[tuple[list[float], list[float]]] = []
        at = 0
        while at < len(widths):
            take = min(BLOCKS_PER_ROW, len(widths) - at)
            while True:
                row = widths[at : at + take]
                full = [max(BLOCK_GAP, g) for g in need[at : at + take - 1]]
                if sum(row) + sum(full) <= STRIP_WIDTH:
                    rows.append((row, full))
                    break
                least = [max(BLOCK_GAP_LEAST, g) for g in need[at : at + take - 1]]
                room = STRIP_WIDTH - sum(least)
                factor = room / sum(row)
                narrowed = [max(BLOCK_NARROWEST, w * factor) for w in row]
                if sum(narrowed) <= room or take == 1:
                    rows.append((narrowed, least))
                    break
                take -= 1
            at += take
        return rows

    def block_width(self, block: ChainBlock) -> float:
        if block.stack:  # a column as wide as its widest block
            return max(self.block_width(replace(b, stack=())) for b in (block, *block.stack))
        need = text_width(block.title, BLOCK_TEXT_SIZE) + 2 * GAP
        label = _block_label(block)
        return max(BLOCK_LEAST, min(BLOCK_MOST, max(need, label + self.block_value(block))))

    def block_value(self, block: ChainBlock) -> float:
        """The value column a block's lines need: the widest text a line shows, or a
        number of three significant digits with its unit, whole; and its margin."""
        need = 0.0
        for row in block.rows:
            if row.shown_as == "text":
                need = max(need, text_width(row.text or "", BLOCK_TEXT_SIZE) + WORD_GAP)
            elif row.shown_as == "value" and row.channel is not None:
                # the widest text the display prints, in its fixed-width face, whole
                unit = row.unit or self.channel_unit(row.channel)
                widest, word = self.widest_value(row.channel)
                printed = widest if word or not unit else f"{widest} {unit}"
                need = max(need, shown_width(printed, BLOCK_TEXT_SIZE) + GAP)
            elif row.channel is not None:  # a word, or a lamp and its word
                light = LIGHT_SIZE + GAP if row.shown_as == "lamp" else 0
                said = [w for _value, w in row.words] or [_WIDEST_NUMBER]
                said.append(self.widest_value(row.channel)[0])
                word_w = max(text_width(w, BLOCK_TEXT_SIZE) for w in said)
                need = max(need, light + word_w + GAP)
        return need

    def channel_unit(self, channel: str) -> str | None:
        """The unit the facility states for *channel*, which the viewer prints beside it."""
        try:
            return self.graph.channel(channel).unit
        except KeyError:
            return None

    def chain_block(
        self, x: float, top: float, w: float, block: ChainBlock, band: float | None = None
    ) -> float:
        """One block: its title (in the focus colour for the card's own device, a link
        to the thing's card otherwise), its box, its lines; a hop nothing states is a
        dashed box, a contested one a dashed box in the minor colour. Returns its
        bottom."""
        colour = symbols.FOCUS if block.focus else (MUTED_TEXT if block.dashed else PAGE_TEXT)
        titles = _wrap(block.title, w, BLOCK_TEXT_SIZE, TITLE_LINES)
        if band is None:
            band = len(titles) * BLOCK_TITLE
        for i, title in enumerate(titles):
            ty = top + band - (len(titles) - i) * BLOCK_TITLE  # the last line on the box
            self.text(x, ty, title, size=BLOCK_TEXT_SIZE, w=w, h=BLOCK_TITLE, fg=colour)
        if block.link is not None:
            self.nav(x, top, w, band, replace(block.link, label=f"open {block.link.label}"))
        box_top = top + band + ROW_GAP
        # the value column takes what its lines need, the labels the rest of the box,
        # as wide as the longest of them; a label longer than that takes two lines
        label = max(min(_block_label(block), w - GAP - self.block_value(block)), BLOCK_LABEL_LEAST)
        labels = [_wrap_label(row.label, label - GAP, SMALL_SIZE, 2) for row in block.rows]
        rows_h = [BLOCK_ROW + (len(lines) - 1) * SMALL_LINE for lines in labels] or [BLOCK_ROW]
        height = BLOCK_TOP + sum(rows_h) - BLOCK_ROW_TOP
        if block.dashed or block.standing == _CONTESTED:
            line = symbols.ALARM_MINOR if block.standing == _CONTESTED else MUTED_TEXT
            self.box(x, box_top, w, height, line=line, dashed=True)
        else:
            props = {
                "lineColor": symbols.FOCUS if block.focus else PAGE_TEXT,
                "lineWidth": 2 if block.focus else BLOCK_LINE,
                "fill": True,
                "fillColor": symbols.FOCUS_TINT if block.focus else PANEL_BACKGROUND,
            }
            self.s.widget("rectangle", x, box_top, w, height, props)
        bottom = box_top + height
        if block.standing == _CONTESTED:
            self.small(x + GAP, bottom + ROW_GAP, _CONTESTED, fg=MUTED_TEXT)
            bottom += SMALL_LINE + ROW_GAP
        if block.inferred:
            self.inferred_word(x + GAP, bottom + ROW_GAP, block.inferred)
            bottom += SMALL_LINE + ROW_GAP
        line_h = BLOCK_ROW - ROW_GAP
        if not block.rows:
            self.text(
                x + GAP, box_top + BLOCK_ROW_TOP, _NOT_STATED_TEXT,
                size=SMALL_SIZE, w=w - 2 * GAP, h=line_h, fg=MUTED_TEXT,
            )  # fmt: skip
        y = box_top + BLOCK_ROW_TOP
        for row, lines, row_h in zip(block.rows, labels, rows_h, strict=False):
            for i, piece in enumerate(lines):
                self.text(x + GAP, y + i * SMALL_LINE, piece, size=SMALL_SIZE, w=label - GAP, h=line_h, fg=MUTED_TEXT)  # fmt: skip
            vx, vw = x + label, w - label - GAP
            self.shown_value(
                vx, y, vw, row.channel, row.shown_as, row.words,
                unit=row.unit, scale=row.scale, size=BLOCK_TEXT_SIZE, h=line_h, text=row.text,
            )  # fmt: skip
            y += row_h
        return bottom

    def operate(self, operate: Operate) -> None:
        """The legend of the kinds of control, then each group: its tag, its title,
        its controls in a row, the line of what it writes; with no group, its note."""
        if not operate.groups:
            self.small(self.x0, self.y, operate.note or "", fg=MUTED_TEXT)
            self.y += SMALL_LINE + ROW_GAP
            return
        x = self.x0
        for kind, meaning in CONTROL_LEGEND:
            x = self.control_tag(x, self.y, kind)
            x = self.small(x + GAP / 2, self.y, meaning, fg=MUTED_TEXT) + WORD_GAP
        self.y += CONTROL_TAG_HEIGHT + 2 * GAP
        presses = [g for g in operate.groups if g.kind != "entry"]
        for start in range(0, len(presses), OPERATE_COLUMNS):
            row = presses[start : start + OPERATE_COLUMNS]
            width = (STRIP_WIDTH - COLUMN_GAP * (OPERATE_COLUMNS - 1)) / OPERATE_COLUMNS
            if len(row) == 1:
                width = STRIP_WIDTH
            top, bottom = self.y, self.y
            for k, group in enumerate(row):
                self.y = top
                self.control_group(group, self.x0 + k * (width + COLUMN_GAP), width)
                bottom = max(bottom, self.y)
            self.y = bottom
        for group in operate.groups:
            if group.kind == "entry" and len(group.controls) > ENTRIES_IN_ONE_COLUMN:
                self.entry_grid(group)  # many of one role: a grid, not a long list
            elif group.kind == "entry":  # a row each: name, now, field, Apply
                self.control_group(group, self.x0, STRIP_WIDTH)

    def control_group(self, group: ControlGroup, x0: float, width: float) -> None:
        """One group in a column from *x0*, *width* wide: its tag and title, its
        controls (an action with a status beside it shows the live word), the line
        of what it writes."""
        withheld = group.kind == WITHHELD
        x = self.control_tag(x0, self.y, _READ_ONLY_TAG if withheld else group.kind.upper())
        if group.title:
            self.small(
                x + GAP, self.y, _cut(group.title, x0 + width - x - GAP, SMALL_SIZE), fg=MUTED_TEXT
            )
        self.y += CONTROL_TAG_HEIGHT + ROW_GAP + 2
        if withheld:  # where the presses would stand: each channel's value, read only
            for control in group.controls:
                tx = self.small(x0, self.y + (ROW - SMALL_LINE) / 2, control.label, fg=MUTED_TEXT)
                room = max(x0 + width - tx - GAP, GRID_VALUE / 2)
                self.shown_value(
                    tx + GAP, self.y, room, control.channel, "word",
                    self.graph_words(control.channel), size=BLOCK_TEXT_SIZE,
                )  # fmt: skip
                self.y += ROW + ROW_GAP
            self.small(x0, self.y, group.writes, fg=MUTED_TEXT)
            self.s.widget(
                "rectangle", x0, self.y, width, SMALL_LINE,
                {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0,
                 _TOOLTIP: group.rule or ""},
            )  # fmt: skip
            self.y += SMALL_LINE + GAP
            return
        if group.kind == "entry":
            for control in group.controls:
                self.entry(control)
        else:
            x = x0
            for control in group.controls:
                if x > x0 and x + CONTROL_WIDTH > x0 + width:
                    x = x0
                    self.y += CONTROL_HEIGHT + GAP
                self.press(x, self.y, control, group.kind == "choice")
                x += CONTROL_WIDTH + GAP
                # what it is now, beside it: a status channel of its own, not the command
                status = control.status
                if group.kind == "action" and status is not None and status != control.channel:
                    room = min(VALUE_WIDTH, x0 + width - x)
                    if room > GRID_VALUE / 2:
                        self.shown_value(
                            x, self.y + (CONTROL_HEIGHT - ROW) / 2, room, status, "word",
                            self.graph_words(status), size=BLOCK_TEXT_SIZE,
                        )  # fmt: skip
                        x += room + GAP
            self.y += CONTROL_HEIGHT + ROW_GAP
        for line in _prose(group.writes, width, SMALL_SIZE):
            self.small(x0, self.y, line, fg=MUTED_TEXT)
            self.y += SMALL_LINE
        self.y += GAP

    def graph_words(self, channel: str) -> tuple[tuple[str, str], ...]:
        return tuple(self.graph.state_words(channel).items())

    def control_tag(self, x: float, y: float, kind: str) -> float:
        """A small tag saying what kind of control follows. Returns its right edge."""
        colour = (
            symbols.CONTROL_ACTION
            if kind == "ACTION"
            else (MUTED_TEXT if kind == "READ ONLY" else PAGE_TEXT)
        )
        w = text_width(kind, SMALL_SIZE) + 2 * GAP
        self.box(
            x, y, w, CONTROL_TAG_HEIGHT, line=colour, fill=TRANSPARENT, dashed=kind == "READ ONLY"
        )
        self.text(x, y, kind, size=SMALL_SIZE, w=w, h=CONTROL_TAG_HEIGHT, fg=colour, align="center")
        return x + w

    def press(
        self,
        x: float,
        y: float,
        control: Control,
        choice: bool,
        w: float = CONTROL_WIDTH,
        h: float = CONTROL_HEIGHT,
    ) -> None:
        """An action button (filled) or a choice's segment (outlined, shaded while the
        status reads its value); a press whose value no source states is a dashed box."""
        if control.press_value is None:
            self.box(x, y, w, h, line=MUTED_TEXT, dashed=True)
            label = f"{control.label}: {_VALUE_NOT_STATED}"
            self.text(x, y, _cut(label, w, SMALL_SIZE), size=SMALL_SIZE, w=w, h=h, fg=MUTED_TEXT, align="center")  # fmt: skip
            return
        value = control.press_value
        props: dict[str, Any] = {
            "pv": control.channel,
            "label": control.label,
            "pressValue": int(value) if float(value).is_integer() else value,
            "fontSize": BLOCK_TEXT_SIZE,
            _TOOLTIP: f"writes {control.channel} = {press_words(control)}",
        }
        rules = None
        if choice:
            props |= {"backgroundColor": PANEL_BACKGROUND, "foregroundColor": PAGE_TEXT}
            status = control.status or control.channel
            if self.budget.take([(1, frozenset({status}))]):
                self.s._rules += 1
                rules = [
                    {
                        "id": f"r-{self.s._rules:04d}",
                        "name": f"{status} selected",
                        "targetProperty": "backgroundColor",
                        "pvs": [{"name": status, "trigger": True}],
                        "conditions": [
                            {
                                "expression": f"{{0}} == {value:g}",
                                "value": symbols.CONTROL_CHOICE_SELECTED,
                            }
                        ],
                        "default": PANEL_BACKGROUND,
                    }
                ]
            self.box(x - 1, y - 1, w + 2, h + 2, fill=TRANSPARENT)
        else:
            props |= {
                "backgroundColor": symbols.CONTROL_ACTION,
                "foregroundColor": symbols.CONTROL_ACTION_TEXT,
            }
        self.s.widget("pv-button", x, y, w, h, props, rules=rules)

    def entry(self, control: Control) -> None:
        """Type, check, Apply: the name, the live value (now), a field that writes a
        local value on Enter (new), Apply, which writes the new value to the
        channel, and what it will write (:meth:`apply_controls`)."""
        x = self.x0
        self.text(x, self.y, control.label, size=BLOCK_TEXT_SIZE, w=2 * BLOCK_LABEL, h=ROW)
        x += 2 * BLOCK_LABEL + GAP
        self.small(x, self.y + (ROW - SMALL_LINE) / 2, _NOW, fg=MUTED_TEXT)
        x += text_width(_NOW, SMALL_SIZE) + GAP
        now_w = max(VALUE_WIDTH, self.value_span(control.channel))
        self.number(x, self.y, now_w, ROW, control.channel, None, None, TEXT_SIZE)
        x += now_w + 2 * GAP
        self.apply_controls(control, x, self.y)
        self.y += ROW + ROW_GAP

    def entry_cell(self, control: Control, x: float, y: float, w: float) -> None:
        """One entry of a grid, *w* wide: its name and live value, then the field and
        Apply, then what Apply will write."""
        now_w = min(max(VALUE_WIDTH, self.value_span(control.channel)), w / 2)
        name_w = w - now_w - GAP
        self.text(x, y, _cut(control.label, name_w, SMALL_SIZE), size=SMALL_SIZE, w=name_w, h=ROW)
        self.number(x + name_w + GAP, y, now_w, ROW, control.channel, None, None, TEXT_SIZE)
        self.apply_controls(control, x, y + ROW + ROW_GAP, (x, y + 2 * (ROW + ROW_GAP)))

    def entry_grid(self, group: ControlGroup) -> None:
        """Many entries of one role: a sub-heading with the role's title, then the
        entries in :data:`ENTRY_COLUMNS` columns, filled row by row, then the line of
        what they write, every channel's name on hover."""
        if group.title:
            self.small(self.x0, self.y, group.title.upper(), fg=MUTED_TEXT)
            self.y += SMALL_LINE + ROW_GAP
        cw = (STRIP_WIDTH - (ENTRY_COLUMNS - 1) * COLUMN_GAP) / ENTRY_COLUMNS
        high = ENTRY_CELL_ROWS * (ROW + ROW_GAP) + GAP
        for n, control in enumerate(group.controls):
            x = self.x0 + (n % ENTRY_COLUMNS) * (cw + COLUMN_GAP)
            self.entry_cell(control, x, self.y + (n // ENTRY_COLUMNS) * high, cw)
        self.y += -(-len(group.controls) // ENTRY_COLUMNS) * high
        self.writes_line(group, STRIP_WIDTH)

    def writes_line(self, group: ControlGroup, width: float) -> None:
        """A group's muted line of what it writes; over many channels, their names
        on hover."""
        top = self.y
        for line in _prose(group.writes, width, SMALL_SIZE):
            self.small(self.x0, self.y, line, fg=MUTED_TEXT)
            self.y += SMALL_LINE
        if len(group.controls) > ENTRIES_IN_ONE_COLUMN:
            names = ", ".join(dict.fromkeys(c.channel for c in group.controls))
            self.s.widget(
                "rectangle", self.x0, top, width, self.y - top,
                {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0, _TOOLTIP: names},
            )  # fmt: skip
        self.y += GAP

    def apply_controls(
        self, control: Control, x: float, y: float, will_at: tuple[float, float] | None = None
    ) -> None:
        """From (*x*, *y*): a field that writes a local value on Enter (new), Apply,
        which writes the new value to the channel, then (at *will_at*, else beside)
        what it will write.

        The local starts as empty text: nothing entered. Until it holds a finite
        number the Apply button is off the canvas, with "type a value first" in its
        place, invisible (the viewer leaves it out of the page, so the keyboard
        cannot reach it either) and its press value is text, not a number; a
        geometry rule brings it in, a rule makes it visible and a formula gives it
        the typed number, all from that local only. What it will write comes from
        that same formula: when the field's text is no number, both keep the last
        number, so the line says what a press writes. The viewer cannot clear the
        local after a write, so the typed value stays in the field."""
        local = (
            "loc://"
            + re.sub(r"[^A-Za-z0-9]+", "_", f"{self.s.doc_id}_{control.channel}")
            + '<VString>("")'
        )
        self.box(x - 1, y - 1, VALUE_WIDTH + 2, ROW + 2, line=MUTED_TEXT, fill=TRANSPARENT)
        props = {
            "pv": local,
            "fontSize": TEXT_SIZE,
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": FIELD_BACKGROUND,
            "alarmSensitiveBorder": False,
            _TOOLTIP: f"new value for {control.channel}: Enter keeps it here, nothing is written",
        }
        self.s.widget("pv-text-input", x, y, VALUE_WIDTH, ROW, props)
        x += VALUE_WIDTH + GAP
        width = 2 * BLOCK_GAP
        self.text(x, y, _TYPE_FIRST, size=SMALL_SIZE, w=width, h=ROW, fg=MUTED_TEXT)
        name = f"draft-{len(self.s.formulas)}"
        self.s.formulas.append(
            {"name": name, "kind": "scalar", "expression": "{v} + 0", "bindings": {"v": local}}
        )
        first = self.s._rules + 1
        self.s._rules += len(("x", "invisible", "pressValue"))
        rules = [
            {
                "id": f"r-{first:04d}",
                "name": f"{control.channel} apply shown once a number is entered",
                "targetProperty": "x",
                "pvs": [{"name": local, "trigger": True}],
                "conditions": [{"expression": _ENTERED, "value": x}],
                "default": AWAY,
            },
            {
                "id": f"r-{first + 1:04d}",
                "name": f"{control.channel} apply in the page once a number is entered",
                "targetProperty": "invisible",
                "pvs": [{"name": local, "trigger": True}],
                "conditions": [{"expression": _ENTERED, "value": False}],
                "default": True,
            },
            {
                "id": f"r-{first + 2:04d}",
                "name": f"{control.channel} apply value",
                "targetProperty": "pressValue",
                "default": NOTHING_ENTERED,
                "formula": _FORMULA_PREFIX + name,
            },
        ]
        target = control.channel  # the one name the button writes and its words name
        button = {
            "pv": target,
            "label": _APPLY,
            "pressValue": NOTHING_ENTERED,
            "invisible": True,  # not in the page, so not in its tab order, until a number
            "fontSize": BLOCK_TEXT_SIZE,
            "backgroundColor": symbols.CONTROL_ACTION,
            "foregroundColor": symbols.CONTROL_ACTION_TEXT,
            _TOOLTIP: f"writes the new value to {target}",
        }
        self.s.widget("pv-button", AWAY, y, width, ROW, button, rules=rules)
        # what Apply writes, from the same formula as its press value: when the
        # field's text is no number, both keep the last number, so the line still
        # says what a press writes
        x, y = will_at if will_at is not None else (x + width + GAP, y)
        self.small(x, y + (ROW - SMALL_LINE) / 2, _WILL_WRITE, fg=MUTED_TEXT)
        x += text_width(_WILL_WRITE, SMALL_SIZE) + GAP
        self.s._rules += 1
        shown = {
            "id": f"r-{self.s._rules:04d}",
            "name": f"{control.channel} what apply writes",
            "targetProperty": "text",
            "default": NOTHING_ENTERED,
            "formula": _FORMULA_PREFIX + name,
        }
        props = {
            "text": NOTHING_ENTERED,
            "fontSize": TEXT_SIZE,
            "align": "left",
            "foregroundColor": PAGE_TEXT,
            "backgroundColor": TRANSPARENT,
            _TOOLTIP: f"what Apply writes to {target}",
        }
        self.s.widget("text-label", x, y, VALUE_WIDTH, ROW, props, rules=[shown])

    def channel_grid(self, grid: ChannelGrid) -> None:
        """Every channel not drawn above, in columns filled top to bottom: its light
        when a rule states its health, its short name, its value as a card prints
        one (a state word where one is stated); hovering a row shows the full name."""
        columns = max(1, grid.columns)
        cw = STRIP_WIDTH / columns
        per = -(-len(grid.rows) // columns)
        top = self.y
        for i, (short, channel) in enumerate(grid.rows):
            with self.s.item("grid-row", None, i):
                self.grid_row(i, short, channel, cw, per, top)
        self.y = top + per * GRID_ROW + ROW_GAP
        self.grid_rest(grid)

    def grid_row(self, i: int, short: str, channel: str, cw: float, per: int, top: float) -> None:
        """One row of the channel grid: light, short name, value, and the full name
        on hover."""
        c, r = divmod(i, per)
        x, y = self.x0 + c * cw, top + r * GRID_ROW
        span = max(GRID_VALUE, self.value_span(channel))
        value_w = min(span, cw - GRID_NAME_LEAST - 2 * GAP)  # the name keeps its least
        vx = x + cw - value_w - GAP
        name_x = x
        lit = [describe_light(self.graph, channel)] if self.graph.rule(channel) else []
        rule = ladder(self.graph, lit) if lit else None
        if lit and not meaning_stated(self.graph, channel):  # a rule, no stated meaning
            name_x = self.no_rule(x, y + (ROW - LIGHT_SIZE) / 2) + GAP
        elif isinstance(rule, Ladder) and self.budget.take([light_cost(rule)]):
            name_x = self.light(x, y + (ROW - LIGHT_SIZE) / 2, lit, rule=rule) + GAP

        name_w = vx - GAP - name_x
        self.text(
            name_x, y, _cut(short, name_w, SMALL_SIZE), size=SMALL_SIZE, w=name_w, fg=MUTED_TEXT
        )
        words = tuple(self.graph.state_words(channel).items())
        if words and self.budget.take([(1, frozenset({channel}))]):
            self.word(vx, y, value_w, ROW, channel, words, size=TEXT_SIZE)
        else:
            drawn_as = self.graph.channel(channel).drawn_as
            held = "unit" if span > value_w else None
            self.field(vx, y, channel, drawn_as=drawn_as, w=value_w, units=True, within=held)
        self.s.widget(
            "rectangle", x, y, cw - GAP, ROW,
            {"fill": False, "lineColor": TRANSPARENT, "lineWidth": 0, _TOOLTIP: channel},
        )  # fmt: skip

    def grid_rest(self, grid: ChannelGrid) -> None:
        """Under the channel grid: the count of the channels it does not list, and
        the link to the full list."""
        if grid.hidden:
            note = _HIDDEN.format(grid.hidden)
            self.small(self.x0, self.y, note, fg=MUTED_TEXT)
            if grid.full is not None:  # where they are listed, read only
                x = self.x0 + text_width(note, SMALL_SIZE) + GAP
                link_text = f"{grid.full.label} {_MORE}"
                lw = text_width(link_text, SMALL_SIZE)
                self.small(x, self.y, link_text, fg=symbols.FOCUS)
                self.nav(x, self.y, lw, SMALL_LINE, grid.full)
            self.y += SMALL_LINE + ROW_GAP

    def channel_count(self, count: ChannelCount) -> None:
        """For a device with more channels than a card lists: the counts by owner in
        one muted line, and the link to the full list at its right."""
        link_text = f"{count.link.label} {_MORE}"
        lw = text_width(link_text, TEXT_SIZE)
        line = _cut(" · ".join(count.parts), STRIP_WIDTH - lw - 2 * GAP, SMALL_SIZE)
        self.small(self.x0, self.y + (ROW - SMALL_LINE) / 2, line, fg=MUTED_TEXT)
        x = self.x0 + STRIP_WIDTH - lw
        self.text(x, self.y, link_text, w=lw, fg=symbols.FOCUS)
        self.nav(x, self.y, lw, ROW, count.link)
        self.y += ROW + ROW_GAP

    # -- the device card ------------------------------------------------------

    def card(self, sections: list[Section]) -> list[Section]:
        """A device card's two columns in one frame: signals, settings, the health
        not shown in the table and the sources left; the fed-by chain, the other
        relations as chips under "also", and where it is right. Returns the other
        sections (a picture), drawn full width under it."""
        relations = {title for pair in RELATION_TITLES.values() for title in pair}
        left = [s for s in sections if s.title in _CARD_LEFT]
        wide = [s for s in left if _long_group(s)]
        right = [s for s in sections if s.title in relations | _CARD_RIGHT]
        below = [s for s in sections if s not in left and s not in right]
        left = [s for s in left if s not in wide]
        if not left and not right and not wide:
            return below
        first = len(self.s.nodes)
        top = self.y
        start = top + PANEL_PAD
        self.y, self.x0, self.right = start, MARGIN, MARGIN + CARD_LEFT - COLUMN_GAP
        for section in left:
            self.card_section(section)
        bottom = self.y
        right_x = MARGIN + CARD_LEFT if left else MARGIN  # no left column: right moves over
        self.y, self.x0, self.right = start, right_x, MARGIN + STRIP_WIDTH
        chips: list[tuple[str, Link | None]] = []
        for section in right:
            if section.title == _FED_BY:
                self.chain(section)
            elif section.title in relations:
                chips += [_chip(section.title, item) for item in section.items]
            else:
                self.heading(section.title)
                self.items(section)
                self.y += ROW_GAP
        if chips:
            self.also(chips)
        bottom = max(bottom, self.y)
        self.x0, self.right = MARGIN, MARGIN + STRIP_WIDTH
        self.y = bottom
        for section in wide:
            self.wide_group(section)
        bottom = self.y
        self.x0, self.right = MARGIN, PAGE_WIDTH - MARGIN + PANEL_PAD
        self.y = bottom + PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())  # under what it frames
        self.y += PANEL_GAP
        return below

    def card_section(self, section: Section) -> None:
        """One block of a card's left column."""
        items = list(section.items)
        if section.title == "health":  # a light the table drew is not repeated
            items = [i for i in items if not (isinstance(i, Light) and i.channel in self.lit)]
            if not items:
                return
        if section.title != _PICTURE:  # a picture heads itself, by its view
            self.heading(section.title)
        if section.title in ("signals", "settings"):
            rows = [i for i in items if isinstance(i, ValueRow)]
            saved = (self.label_w, self.value_w)
            self.value_w = self.values_width(rows[:CARD_ROWS], self.value_w)
            room = self.right - self.x0 - 2 * (self.value_w + GAP) - LIGHT_SIZE - GAP
            self.label_w = min(self.label_w, room)  # the name wraps; the unit is whole
            if section.title == "signals":
                self.signal_heads()
            for row in rows[:CARD_ROWS]:
                self.signal_row(row)
            self.label_w, self.value_w = saved
            rest = Counter(row.drawn_as or "untyped" for row in rows[CARD_ROWS:])
            for kind, count in sorted(rest.items()):
                text = _MORE_CHANNELS.format(count, kind)
                self.text(self.x0, self.y, text, size=SMALL_SIZE, fg=MUTED_TEXT)
                self.next_row()
            for item in items:
                if not isinstance(item, ValueRow):
                    self.item(item)
        else:
            for item in items[:CARD_ROWS]:
                self.item(item)
            if len(items) > CARD_ROWS:
                text = _MORE_ROWS.format(len(items) - CARD_ROWS)
                self.text(self.x0, self.y, text, size=SMALL_SIZE, fg=MUTED_TEXT)
                self.next_row()
        self.y += ROW_GAP

    def wide_group(self, section: Section) -> None:
        """A signals or settings group of more than :data:`CARD_SPLIT_ROWS` rows across the
        card, filled column by column, the group's heading over them all and the
        table's heads over each. Values are :data:`CARD_COMPACT_VALUE` wide, or as
        wide as :func:`value_width` needs for the widest unit there; the name keeps
        :data:`CARD_LABEL_MIN`, so a column is at least :data:`CARD_COLUMN_MIN` wide
        and wider when its values need it, and there are as many as fit. The rows
        past :data:`CARD_ROWS` are counted under them, as in one column."""
        rows = [i for i in section.items if isinstance(i, ValueRow)]
        shown = rows[:CARD_ROWS]
        value_w = self.values_width(shown, CARD_COMPACT_VALUE)
        least = max(CARD_COLUMN_MIN, CARD_LABEL_MIN + 2 * (value_w + GAP) + LIGHT_SIZE + GAP)
        count = max(1, int((STRIP_WIDTH + COLUMN_GAP) // (least + COLUMN_GAP)))
        count = min(count, len(shown)) or 1
        width = (STRIP_WIDTH - (count - 1) * COLUMN_GAP) / count
        per = math.ceil(len(shown) / count)
        self.heading(section.title)
        top = self.y
        bottom = top
        saved = (self.label_w, self.value_w)
        self.value_w = value_w
        self.label_w = width - 2 * (self.value_w + GAP) - LIGHT_SIZE - GAP
        for n in range(count):
            self.x0 = MARGIN + n * (width + COLUMN_GAP)
            self.y = top
            if section.title == "signals":
                self.signal_heads()
            for row in shown[n * per : (n + 1) * per]:
                self.signal_row(row)
            bottom = max(bottom, self.y)
        self.label_w, self.value_w = saved
        self.x0, self.y = MARGIN, bottom
        rest = Counter(row.drawn_as or "untyped" for row in rows[CARD_ROWS:])
        for kind, number in sorted(rest.items()):
            self.text(
                self.x0, self.y, _MORE_CHANNELS.format(number, kind), size=SMALL_SIZE, fg=MUTED_TEXT
            )
            self.next_row()
        for item in section.items:
            if not isinstance(item, ValueRow):
                self.item(item)
        self.y += ROW_GAP

    def signal_heads(self) -> None:
        """The table's column heads, over the name, the setpoint and the readback."""
        for word, x in zip(_SIGNAL_HEADS, self.signal_columns(), strict=True):
            self.text(x, self.y, word.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW

    def signal_columns(self) -> tuple[float, float, float]:
        name = self.x0
        setpoint = name + self.label_w + GAP
        return name, setpoint, setpoint + self.value_w + GAP

    def signal_row(self, row: ValueRow) -> None:
        """A row of the signals table: the name with its unit (wrapped to at most
        :data:`SIGNAL_NAME_LINES` lines, cut only past them), the setpoint under its
        head, the readback (or the readable channel) under its, then the light of the
        deviation or of the readback's rule."""
        name, setpoint, readback = self.signal_columns()
        units = " / ".join(dict.fromkeys(u for u in (row.unit, row.readback_unit) if u))
        raw = row.label in (row.channel, row.readback)  # no signal name: the channel's own
        word = _short(row.label) if raw else row.label
        label = f"{word} ({units})" if units else word
        lines = _wrap(label, self.label_w, TEXT_SIZE, lines=SIGNAL_NAME_LINES)
        for n, line in enumerate(lines):
            w = min(text_width(line, TEXT_SIZE), self.label_w)
            self.text(name, self.y + n * TITLE_LINE, line, w=w)
        extra = (len(lines) - 1) * TITLE_LINE
        if raw:  # the full name, muted, under the short one, broken after a colon
            fulls = _wrap_name(row.label, self.label_w, SMALL_SIZE, SIGNAL_NAME_LINES)
            for n, line in enumerate(fulls):
                self.small(name, self.y + ROW + extra + n * SMALL_LINE, line, fg=MUTED_TEXT)
            extra += (len(fulls) - 1) * SMALL_LINE
        shown = row.readback
        if row.readback is not None:
            self.field(
                setpoint, self.y, row.channel, drawn_as=row.drawn_as, writable=True, w=self.value_w
            )
            self.field(readback, self.y, row.readback, w=self.value_w)
        elif row.writable:
            self.field(
                setpoint, self.y, row.channel, drawn_as=row.drawn_as, writable=True, w=self.value_w
            )
        else:
            shown = row.channel
            self.field(readback, self.y, row.channel, drawn_as=row.drawn_as, w=self.value_w)
        x = readback + self.value_w + GAP
        deviation = _tolerance_ladder(row)
        if deviation is not None:
            x = self.light(x, self.y, [Light(row.channel)], rule=deviation)
        elif shown is not None and self.graph.rule(shown):
            x = self.light(x, self.y, [describe_light(self.graph, shown)])
            self.lit.add(shown)
        if row.inferred:
            self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        self.next_row((ROW + SMALL_LINE if raw else ROW) + extra)

    def chain(self, section: Section) -> None:
        """The fed-by chain: the device's box, and each feeder's box right of it with an
        arrow into the device, the way the power flows; the arrows of several feeders
        join on one bus before the device."""
        self.heading(section.title)
        x = self.x0
        top = self.y
        name = self.view.alias or self.view.id.id
        self.chain_box(x, top, _THIS_DEVICE, name, None)
        feeder_x = x + CHAIN_BOX + CHAIN_ARROW
        y = top
        for item in section.items:
            label = item.label if isinstance(item, Link) else getattr(item, "text", "")
            link = item if isinstance(item, Link) else None
            kind = ""
            if link is not None:
                try:
                    kind = self.graph.thing(link.target.id, link.target.kind).cls or ""
                except KeyError:
                    kind = ""
            if getattr(item, "standing", None) == _CONTESTED:
                kind = f"{kind} · {_CONTESTED}" if kind else _CONTESTED
            if getattr(item, "basis", None) == "inferred":
                kind = f"{kind} {symbols.INFERRED_MARK}".strip()
            self.chain_box(feeder_x, y, kind, label, link)
            bus = x + CHAIN_BOX + CHAIN_ARROW / 2
            mid, here = top + CHAIN_HEIGHT / 2, y + CHAIN_HEIGHT / 2
            points = [(feeder_x, here), (bus, here), (bus, mid), (x + CHAIN_BOX, mid)]
            self.line(points if here != mid else [points[0], points[-1]], MUTED_TEXT, arrow=True)
            y += CHAIN_HEIGHT + GAP
        self.y = max(y, top + CHAIN_HEIGHT + GAP)
        self.text(self.x0, self.y, _CHAIN_CAPTION, size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW + ROW_GAP

    def chain_box(self, x: float, y: float, kind: str, name: str, link: Link | None) -> None:
        """One box of the chain: what it is, small, over its name; opening its card."""
        self.box(x, y, CHAIN_BOX, CHAIN_HEIGHT)
        inner = CHAIN_BOX - GAP
        if kind:
            kind = _cut(kind, inner, SMALL_SIZE)
            self.text(x + GAP / 2, y, kind, size=SMALL_SIZE, w=inner, h=SMALL_LINE, fg=MUTED_TEXT)
        name = _cut(name, inner, SMALL_SIZE)
        w = text_width(name, SMALL_SIZE)
        self.text(x + GAP / 2, y + CHAIN_HEIGHT - ROW, name, size=SMALL_SIZE, w=w)
        if link is not None:
            self.nav(x, y, CHAIN_BOX, CHAIN_HEIGHT, link)

    def also(self, chips: Sequence[tuple[str, Link | None]]) -> None:
        """Every other relation as a chip, "<relation>: <thing>", wrapping in the column."""
        self.heading(_ALSO)
        x = self.x0
        for label, link in chips:
            room = self.right - self.x0 - GAP
            label = _cut(label, room, SMALL_SIZE)
            w = text_width(label, SMALL_SIZE) + GAP
            if x > self.x0 and x + w > self.right:
                x = self.x0
                self.y += CHIP_HEIGHT + CHIP_GAP
            x = self.chip(x, self.y, label, link) + CHIP_GAP
        self.y += CHIP_HEIGHT + CHIP_GAP

    def header(self) -> None:
        """The bar at the top: the screen's thing, the way out to the places around it,
        and the model the screen was made from, right-aligned."""
        self.s.widget(
            "rectangle",
            0,
            0,
            PAGE_WIDTH,
            HEADER_HEIGHT,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": FIELD_BACKGROUND,
            },
        )
        y = HEADER_PAD
        # the stamp first, as wide as any hash makes it, so the line left of it
        # stops at the same place on every screen and is cut there, not under it
        widest = max("0123456789abcdef", key=lambda c: metrics.WIDTHS[c])
        stamp_w = text_width(f"model {widest * STAMP_DIGITS}", SMALL_SIZE)
        stamp = f"model {self.graph.content_hash[:STAMP_DIGITS]}"
        stamp_x = PAGE_WIDTH - MARGIN - stamp_w
        self.text(stamp_x, y, stamp, size=SMALL_SIZE, w=stamp_w, fg=MUTED_TEXT, align="right")
        page_right, self.right = self.right, stamp_x - GAP
        x: float = MARGIN - GAP
        for crumb in self.view.crumbs:
            if x + GAP + BUTTON_PAD + MARK_WIDTH > self.right:
                break  # no room for one more step of the way out
            x = self.button(x + GAP, y, crumb)
            x = self.text(x + GAP, y, "›", w=MARK_WIDTH, fg=MUTED_TEXT)
        x = self.text(x + GAP, y, self.view.id.id)
        if self.view.alias is not None and x + 2 * GAP < self.right:  # the short name, a chip
            w = min(text_width(self.view.alias, SMALL_SIZE), self.right - x - GAP)
            self.box(x + GAP, y, w, ROW, fill=PANEL_BACKGROUND)
            x = self.text(x + GAP, y, self.view.alias, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        rest = self.title_rest()
        if rest and x + 2 * GAP < self.right:  # what the screen shows of it, muted
            self.text(x + GAP, y, rest, fg=MUTED_TEXT)
        self.right = page_right

    def title_rest(self) -> str | None:
        """What the title says after the screen's id, when it starts with the id; the
        header bar carries it. ``None`` when the title is something else."""
        title, thing = self.view.title, self.view.id.id
        if title.startswith(thing + " "):
            return title.removeprefix(thing).strip()
        return None

    def heading(self, text: str) -> None:
        """A section's heading: small capitals in the muted role."""
        self.text(self.x0, self.y, text.upper(), size=HEADING_SIZE, fg=MUTED_TEXT)
        self.y += ROW + ROW_GAP

    def section(self, section: Section) -> None:
        """A section in its frame: the frame stands in the margin around what it holds."""
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        if not (section.items and isinstance(section.items[0], Strip | Band | Picture)):
            self.heading(section.title)  # a picture heads itself, by its view
        labels = [i.label for i in section.items if isinstance(i, ValueRow | Fact)]
        self.label_col = min(
            max([LABEL_WIDTH, *(text_width(t, TEXT_SIZE) for t in labels)]), LABEL_MOST
        )
        self.check_cols = check_columns([i for i in section.items if isinstance(i, CheckRow)])
        self.items(section)
        self.label_col = LABEL_WIDTH
        self.check_cols = (NAME_WIDTH, NAME_WIDTH)
        self.y += PANEL_PAD - ROW_GAP
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD,
            top,
            STRIP_WIDTH + 2 * PANEL_PAD,
            self.y - top,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": PANEL_BACKGROUND,
            },
        )
        self.s.nodes.insert(first, self.s.nodes.pop())  # under what it frames
        self.y += PANEL_GAP

    def items(self, section: Section) -> None:
        tallies = [i for i in section.items if isinstance(i, CalibrationTallies)]
        if tallies:
            self.tallies(tallies[0])
            return
        if section.title in (_CALIBRATION_SECTION, _SETS_SECTION) and self.view.id.view in _CHIPPED:
            # calibration views, or how far a place's are: one row of chips
            entries = [
                (i.label, i, None) if isinstance(i, Link) else (i.text, None, None)
                for i in section.items
                if isinstance(i, Link | Fact)
            ]
            self.chips(self.x0, entries)
            return
        previous: object = None
        rows = left_out = 0
        for item in section.items:
            if isinstance(item, CheckRow):
                if not isinstance(previous, CheckRow):
                    self.check_header()
                    rows = 0
                rows += 1
                if rows > MAX_TABLE_ROWS:
                    left_out += 1
                    previous = item
                    continue
            self.item(item)
            previous = item
        if left_out:
            self.left_out(left_out)

    def left_out(self, count: int) -> None:
        """The note under a table cut at :data:`MAX_TABLE_ROWS`."""
        text = f"{count} more {_ROWS_NOT_DRAWN}"
        self.text(self.x0, self.y, text, w=STRIP_WIDTH, tags=(_need("table"),))
        self.next_row()

    def item(self, item: object) -> None:
        draws: dict[type, Callable[[Any], object]] = {
            ValueRow: self.value_row,
            Light: self.light_row,
            Link: self.link_row,
            Fact: self.fact_row,
            Placement: self.placement_row,
            Grid: self.grid,
            Strip: self.strip,
            Picture: self.picture,
            Probe: self.probe,
            Elevation: self.elevation,
            Node: self.node,
            Group: self.group,
            Band: self.band,
            CheckRow: self.check_row,
            Totals: self.totals,
            Calibrations: self.calibrations,
        }
        draws[type(item)](item)

    # -- rows -----------------------------------------------------------------

    def value_row(self, row: ValueRow) -> None:
        """A value and its readback, each with its unit once: a label prints the
        channel's own, an input has it beside it."""
        x = self.text(self.x0, self.y, row.label, w=self.label_col)
        x = self.unit_field(x + GAP, row.channel, row.unit, row.drawn_as, row.writable)
        if row.readback is not None:
            x = self.unit_field(x + GAP, row.readback, row.readback_unit, None, False)
        deviation = _tolerance_ladder(row)
        if deviation is not None:
            x = self.light(x + GAP, self.y, [Light(row.channel)], rule=deviation)
        if row.inferred:
            self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        self.next_row()

    def light_row(self, item: Light) -> None:
        x = self.light(self.x0, self.y, [item])
        self.text(x + GAP, self.y, item.channel)
        self.next_row()

    def link_row(self, link: Link) -> None:
        self.button(self.x0, self.y, link, width=self.right - self.x0)
        self.next_row()

    def fact_row(self, fact: Fact) -> None:
        x = self.text(self.x0, self.y, fact.label, w=self.label_col)
        x = self.text(x + GAP, self.y, fact.text)
        if fact.basis == "inferred":
            x = self.mark(x + GAP, self.y, symbols.INFERRED_MARK)
        if fact.standing == _CONTESTED:
            x = self.text(x + GAP, self.y, _CONTESTED, size=SMALL_SIZE)
        note = " · ".join(t for t in (fact.caveat, _citation_text(fact.citation)) if t)
        self.next_row(self.row_note(x + GAP, note))

    def row_note(self, x: float, note: str) -> float:
        """A row's small note from *x* to the right edge, in as many lines as it
        needs (:func:`_prose`); the row's height."""
        if not note:
            return ROW
        lines = _prose(note, self.right - x, SMALL_SIZE)
        if len(lines) == 1:
            self.text(x, self.y, note, size=SMALL_SIZE)
            return ROW
        top = self.y + (ROW - SMALL_LINE) / 2
        for n, line in enumerate(lines):
            self.text(x, top + n * SMALL_LINE, line, size=SMALL_SIZE, h=SMALL_LINE)
        return max(ROW, top - self.y + len(lines) * SMALL_LINE)

    def placement_row(self, item: Placement) -> None:
        x = self.button(self.x0, self.y, item.link, width=self.right - self.x0)
        parts = []
        if item.unit is not None:
            parts.append(f"unit {item.unit}")
        if item.size_u is not None:
            parts.append(f"{item.size_u} U")
        if item.citation is not None:
            parts.append(_citation_text(item.citation))
        self.next_row(self.row_note(x + GAP, " · ".join(parts)))

    def grid(self, grid: Grid) -> None:
        """A set's members as an equal grid: a row per place under its short name, the
        rows grouped under the top-level place they are in (a heading each, a gap
        between), the members in beam order under column heads: the short name
        a column's members share, else ``#1`` … ``#n`` (:func:`_column_head`); one
        row of heads over all groups when they read alike, else a row under each
        group's heading; a line over them says what a cell shows, with its unit. A
        cell is the member's readback of the set's primary pair or its short name, and
        one light: the deviation light when the pair has one, else the member's.
        It opens the member's card and nothing in it is writable. Few columns
        widen to at most :data:`SET_CELL_MAX`; columns past the page's width go
        on in a block below, :data:`SET_BLOCK_GAP` under the one before, with
        only the rows that have members in it. A group whose members all sit in
        one row too long for a block is no grid: it is a flow (:meth:`set_flow`)
        after the grid groups. Cells are drawn while fewer than
        :data:`SET_MEMBERS` are; the members of the rest are counted in a line
        under them."""
        tags = (_need("table"),)
        count = sum(len(row.cells) for row in grid.rows)
        words = [f"{count} members", f"{len(grid.rows)} rows"]
        if grid.kind is not None:  # how many members are in each state
            states = Counter(
                cell.calibration.state
                for row in grid.rows
                for cell in row.cells
                if cell.calibration is not None
            )
            words += [
                f"{n} {_STATE_WORDS[state]}"
                for state, n in sorted(states.items(), key=lambda kv: _CAL_STATES.index(kv[0]))
            ]
        line = " · ".join(words)
        self.text(MARGIN, self.y, line.upper(), size=SMALL_SIZE, fg=MUTED_TEXT, tags=tags)
        self.next_row()
        room = STRIP_WIDTH - SET_ROW_HEAD
        fit = max(1, int((room + GAP) // (SET_CELL_WIDTH + GAP)))
        groups = self.grid_groups(grid.rows)
        flows = [(t, rows[0]) for t, rows in groups if len(rows) == 1 and len(rows[0].cells) > fit]
        flowing = {t for t, _row in flows}
        gridded = [(t, rows) for t, rows in groups if t not in flowing]
        width = max((len(r.cells) for _t, rows in gridded for r in rows), default=0)
        shown = max(1, min(fit, width or fit))
        self.cell_w = min(SET_CELL_MAX, (room - (shown - 1) * GAP) / shown)
        what = self.grid_what(grid)
        self.one_unit = len(self.readback_units(grid.rows)) <= 1  # stated in that line
        w = text_width(what, SMALL_SIZE)
        x_what = self.set_x(0) if gridded else MARGIN
        self.text(x_what, self.y, what, size=SMALL_SIZE, w=w, fg=MUTED_TEXT, tags=tags)
        self.y += ROW
        self.drawn = 0
        left_out = 0
        for first in range(0, width, fit):
            here = [(t, [r for r in rows if r.cells[first : first + fit]]) for t, rows in gridded]
            here = [(t, rows) for t, rows in here if rows]  # a row with none here is not drawn
            if self.drawn >= SET_MEMBERS:
                left_out += sum(len(r.cells[first : first + fit]) for _t, rs in here for r in rs)
                continue
            if first:
                self.y += SET_BLOCK_GAP - ROW_GAP
            columns = range(first, min(first + fit, width))
            heads = {t: [_column_head([(t, rows)], c) for c in columns] for t, rows in here}
            shared = len({tuple(h) for h in heads.values()}) == 1
            if shared:  # every group's columns read alike: one row of heads over all
                self.column_heads(first, heads[here[0][0]], tags)
            for title, rows in here:
                self.y += ROW_GAP
                self.group_heading(title, rows)
                self.y += ROW
                if not shared:  # each group under its own heads
                    self.column_heads(first, heads[title], tags)
                for row in rows:
                    cells = row.cells[first : first + fit]
                    if isinstance(row.label, Link):
                        short = _short(row.label.label)
                        w = min(text_width(short, SMALL_SIZE), SET_ROW_HEAD - GAP)
                        self.text(MARGIN, self.y, _cut(short, w, SMALL_SIZE), size=SMALL_SIZE, w=w)
                        self.nav(MARGIN, self.y, w, ROW, row.label)
                    tall: float = ROW
                    for column, cell in enumerate(cells):
                        tall = max(tall, self.grid_cell(self.set_x(column), self.y, cell))
                        self.drawn += 1
                    self.y += tall + ROW_GAP
        for title, row in flows:
            left_out += self.set_flow(title, row)
        if left_out:
            self.y += ROW_GAP
            self.text(MARGIN, self.y, _SET_MORE.format(left_out), size=SMALL_SIZE, fg=MUTED_TEXT)
            self.next_row()
        if grid.kind is not None:
            self.calibration_legend()

    def calibration_legend(self) -> None:
        """What a calibration view's cell looks like in each state: a sample cell each."""
        self.y += ROW_GAP
        x: float = MARGIN
        for state in _CAL_STATES:
            word = _LEGEND[state]
            w = text_width(word, SMALL_SIZE) + GAP
            self.calibration_box(x, self.y, w, state)
            self.text(x + GAP / 2, self.y, word, size=SMALL_SIZE, w=w - GAP)
            x += w + CHIP_GAP
        self.next_row()

    def calibration_box(self, x: float, y: float, w: float, state: str, h: float = ROW) -> None:
        """A calibration cell's box: tinted by its basis, outlined in the basis role;
        dashed and bare with no value."""
        props: dict[str, Any] = {"lineWidth": LINE_WIDTH}
        if state in symbols.BASIS_TINTS:
            props |= {
                "lineColor": symbols.BASIS_ROLES[state],
                "fill": True,
                "fillColor": symbols.BASIS_TINTS[state],
                "lineStyle": "solid",
            }
        else:
            props |= {"lineColor": MUTED_TEXT, "fill": False, "lineStyle": "dashed"}
        self.s.widget("rectangle", x, y, w, h, props)

    def set_flow(self, title: str, row: GridRow) -> int:
        """A group whose members all sit in one row too long for a block: its heading,
        then its cells in lines at full width, no column heads, no row head. Cells
        past :data:`SET_MEMBERS` drawn on the screen are not drawn; returns how many."""
        self.y += SET_BLOCK_GAP - ROW_GAP
        self.group_heading(title, [row])
        if isinstance(row.label, Link):
            w = text_width(title.upper(), HEADING_SIZE)
            self.nav(MARGIN, self.y, w, ROW, row.label)
        self.y += ROW
        per = max(1, int((STRIP_WIDTH + GAP) // (self.cell_w + GAP)))
        room = max(0, SET_MEMBERS - self.drawn)
        cells = row.cells[:room]
        tall: float = ROW
        for k, cell in enumerate(cells):
            if k and not k % per:
                self.y += tall + ROW_GAP
                tall = ROW
            tall = max(tall, self.grid_cell(MARGIN + (k % per) * (self.cell_w + GAP), self.y, cell))
            self.drawn += 1
        if cells:
            self.y += tall + ROW_GAP
        return len(row.cells) - len(cells)

    def readback_units(self, rows: Sequence[GridRow]) -> set[str | None]:
        """The units of the readbacks the cells of *rows* show."""
        return {
            self.unit_of(cell.pair.readback)
            for row in rows
            for cell in row.cells
            if cell.pair is not None and cell.pair.readback is not None
        }

    def group_heading(self, title: str, rows: Sequence[GridRow]) -> None:
        """A group's heading over its rows of cells. It sets how the group's cells state
        their unit: not at all when the set's line states the one unit; once, after
        the heading, when the group's members share one; else in each cell, clear of
        the light (:meth:`field`)."""
        x = self.text(MARGIN, self.y, title.upper(), size=HEADING_SIZE, fg=MUTED_TEXT)
        units = self.readback_units(rows)
        self.cell_units = "number" if self.one_unit or len(units) <= 1 else "unit"
        (unit,) = units if len(units) == 1 and not self.one_unit else (None,)
        if unit:
            self.text(x + GAP, self.y, unit, size=HEADING_SIZE, fg=MUTED_TEXT)

    def grid_what(self, grid: Grid) -> str:
        """What a set overview's cells show: the readback of the primary pair's signal
        with the readback's unit, or the members' short names; in a calibration view,
        the kind."""
        if grid.kind is not None:
            return _CAL_WHAT.format(grid.kind.lower())
        pairs = [
            cell.pair for row in grid.rows for cell in row.cells
            if cell.pair is not None and cell.pair.readback is not None
        ]  # fmt: skip
        if grid.signal is None or not pairs:
            return _SET_NAMES
        units = {self.unit_of(p.readback) for p in pairs if p.readback is not None}
        what = _SET_READBACK.format(grid.signal.replace("_", " "))
        (unit,) = units if len(units) == 1 else (None,)  # several: each cell states its own
        return f"{what} ({unit})" if unit else what

    def column_heads(self, first: int, heads: Sequence[str], tags) -> None:
        """A row of column heads, each centred over its column, from column *first*."""
        for column, head in enumerate(heads, start=first):
            x = self.set_x(column - first)
            w = text_width(head, SMALL_SIZE)
            x_head = x + (self.cell_w - w) / 2
            self.text(x_head, self.y, head, size=SMALL_SIZE, w=w, fg=MUTED_TEXT, tags=tags)
        self.y += ROW

    def grid_groups(self, rows: Sequence[GridRow]) -> list[tuple[str, list[GridRow]]]:
        """The rows grouped by the top-level place of each row's place, in row order; the
        rows of no place under their own fact."""
        groups: dict[str, list[GridRow]] = {}
        for row in rows:
            if isinstance(row.label, Link):
                top = row.label.target.id
                with contextlib.suppress(KeyError):
                    while (up := self.graph.thing(top, "place").place) is not None:
                        top = up
            else:
                top = row.label.text
            groups.setdefault(top, []).append(row)
        return list(groups.items())

    def set_x(self, column: int) -> float:
        """The left edge of a set overview's *column* within its block."""
        return MARGIN + SET_ROW_HEAD + column * (self.cell_w + GAP)

    def grid_cell(self, x0: float, y: float, cell: GridCell) -> float:
        """One member: the readback in its field, else (with no light) the value its
        card's first key tile shows, else the short name in a box, and one light at
        the right end: the deviation light when the pair has one, else the member's;
        the whole cell opens its card. Returns the cell's height: a row, or the lines
        a calibration value of several numbers takes."""
        w = self.cell_w
        if cell.calibration is not None:  # a calibration view: no light, static text
            found = cell.calibration
            h = self.held_height(found.channels, w - GAP) if found.channels else ROW
            self.calibration_box(x0, y, w, found.state, h=h)
            if found.channels:
                self.held(x0 + GAP / 2, y, found.channels, w - GAP)
            else:
                text = _cut(found.text, w - GAP, TEXT_SIZE)
                self.text(x0 + GAP / 2, y, text, w=min(text_width(text, TEXT_SIZE), w - GAP))
            if isinstance(cell.member, Link):
                member = cell.member
                if not found.channels and text_width(found.text, TEXT_SIZE) > w - GAP:
                    member = Link(f"{member.label}: {found.text}", member.target)  # cut: whole
                self.nav(x0, y, w, h, member)
            return h
        right = x0 + w - CHIP_LIGHT
        pair = cell.pair
        deviation = _tolerance_ladder(pair) if pair is not None else None
        if pair is not None and deviation is not None:
            self.light(right, y, [Light(pair.channel)], rule=deviation, size=CHIP_LIGHT, tile=True)
        elif cell.channels:
            lights = [describe_light(self.graph, c) for c in cell.channels]
            rule = ladder(self.graph, lights, node=True)
            self.light(right, y, lights, rule=rule, size=CHIP_LIGHT, tile=True)
        else:
            self.no_rule(right, y, CHIP_LIGHT)
        end = right - GAP / 2
        if pair is not None and pair.readback is not None:
            # the viewer frames it; the number (and its unit, when the set's differ) in it
            self.field(x0, y, pair.readback, w=end - x0, within=self.cell_units)
        elif cell.tile is not None:  # no light: the value its card shows first
            tile = cell.tile
            self.box(x0, y, end - x0, ROW, fill=FIELD_BACKGROUND)
            self.shown_value(
                x0 + GAP / 2, y, end - x0 - GAP, tile.channel, tile.shown_as, tile.words,
                unit=tile.unit, scale=tile.scale, size=SMALL_SIZE,
            )  # fmt: skip
        else:
            self.box(x0, y, end - x0, ROW, fill=FIELD_BACKGROUND)
            name = self.member_name(cell.member)
            name = _cut(name, end - x0 - GAP, SMALL_SIZE)
            nw = text_width(name, SMALL_SIZE)
            self.text(x0 + GAP / 2, y, name, size=SMALL_SIZE, w=min(nw, end - x0 - GAP))
        if isinstance(cell.member, Link):
            self.nav(x0, y, w, ROW, cell.member)
        return ROW

    def calibrations(self, table: Calibrations) -> None:
        """A device's calibration block: column heads, then a row per kind (see
        :meth:`calibration_row`); conditions that several rows state alike once,
        under the rows: after half a row of space and a thin rule, each such text
        behind a reference mark (1, 2, ...) that also follows the title of every
        row it holds for, so it reads as theirs and not as the last row's note.
        Read-only."""
        for word, dx in zip(_CAL_HEADS, CAL_COLUMNS, strict=True):
            self.text(MARGIN + dx, self.y, word.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW
        # conditions that several rows state alike are stated once, under the rows
        shared = Counter(row.conditions for row in table.rows if row.conditions)
        once = [
            text
            for text in dict.fromkeys(row.conditions for row in table.rows)
            if text is not None and shared[text] > 1
        ]
        marks = {text: str(n) for n, text in enumerate(once, start=1)}
        for row in table.rows:
            if row.conditions in marks:
                self.calibration_row(replace(row, conditions=None), mark=marks[row.conditions])
            else:
                self.calibration_row(row)
        if not once:
            return
        self.y += (ROW + ROW_GAP) / 2
        self.line([(MARGIN, self.y), (MARGIN + STRIP_WIDTH, self.y)], MUTED_TEXT, LINE_WIDTH)
        self.y += ROW_GAP
        for text in once:
            mark = marks[text]
            indent = text_width(mark, SMALL_SIZE) + GAP / 2
            self.small(MARGIN, self.y, mark)
            # the mark points at the rows: their titles are not said again
            for line in _prose(text, STRIP_WIDTH - indent, SMALL_SIZE):
                self.small(MARGIN + indent, self.y, line, fg=MUTED_TEXT)
                self.y += SMALL_LINE
            self.y += ROW_GAP

    def calibration_row(self, row: CalibrationRow, mark: str | None = None) -> None:
        """One kind: its title (a link to the kind's view over a set), wrapped and
        never cut, and after it the *mark* of conditions stated under the table;
        the value in force, or the live channels, with the note and the
        conditions muted under it (at most :data:`CAL_CONDITION_LINES` lines, the
        whole text on hover when cut); the basis chip; date, source and history."""
        x = [MARGIN + dx for dx in CAL_COLUMNS]
        widths = [b - a - GAP for a, b in itertools.pairwise([*x, MARGIN + STRIP_WIDTH])]
        top = self.y
        room = widths[0] - (text_width(mark, SMALL_SIZE) + MARK_GAP if mark else 0)
        titles = _wrap(row.title, room, TEXT_SIZE, lines=len(row.title.split()))
        for n, line in enumerate(titles):
            w = min(text_width(line, TEXT_SIZE), room)
            right = self.text(x[0], top + n * TITLE_LINE, line, w=w)
        if mark:  # raised beside the title's last line, as a reference mark
            self.small(right + MARK_GAP, top + (len(titles) - 1) * TITLE_LINE, mark)
        if row.link is not None:
            wide = max(min(text_width(line, TEXT_SIZE), widths[0]) for line in titles)
            self.nav(x[0], top, wide, ROW + (len(titles) - 1) * TITLE_LINE, row.link)
        under = top + ROW
        if row.channels:
            under = self.held(x[1], top, row.channels, widths[1])
        elif row.shown is None:
            self.small(x[1], top + (ROW - SMALL_LINE) / 2, _NO_VALUE_YET, fg=MUTED_TEXT)
        else:
            value = _cut(row.shown, widths[1], TEXT_SIZE)
            self.text(x[1], top, value, w=min(text_width(value, TEXT_SIZE), widths[1]))
        if row.note:
            self.small(x[1], under, _cut(row.note, widths[1], SMALL_SIZE), fg=MUTED_TEXT)
            under += SMALL_LINE
        if row.conditions:
            lines = _wrap(row.conditions, widths[1], SMALL_SIZE, lines=CAL_CONDITION_LINES)
            for n, line in enumerate(lines):
                self.small(x[1], under + n * SMALL_LINE, line, fg=MUTED_TEXT)
            under += len(lines) * SMALL_LINE
        _name_x, _value_x, chip_x, *rest_x = x
        under = max(under, self.basis_chip(chip_x, top, row))
        _w0, _w1, _w2, *rest_w = widths
        for left, width, text in zip(
            rest_x, rest_w, (row.date, row.source, row.history), strict=True
        ):
            if text:  # in a few lines, broken at its separators; whole on hover
                lines = _wrap_label(text, width, SMALL_SIZE, CAL_CELL_LINES)
                y0 = top + (ROW - SMALL_LINE) / 2
                for n, line in enumerate(lines):
                    self.small(left, y0 + n * SMALL_LINE, line)
                under = max(under, y0 + len(lines) * SMALL_LINE)
        self.y = max(under, top + ROW + (len(titles) - 1) * TITLE_LINE) + ROW_GAP

    def held_height(self, channels: tuple[str, ...], w: float) -> float:
        """How tall :meth:`held` draws *channels* within *w*: its lines of fields."""
        left, lines = 0.0, 1
        for channel in channels:
            need = value_width(self.unit_of(channel), TEXT_SIZE)
            if left > 0 and left + need > w:
                left, lines = 0.0, lines + 1
            left += need + GAP
        return lines * ROW + (lines - 1) * ROW_GAP

    def held(self, x: float, y: float, channels: tuple[str, ...], w: float) -> float:
        """The channels holding a calibration value live, side by side within *w*, a
        gap between each two: one field for a scalar, one per number of a tuple, each
        as wide as its number and whole unit need (:func:`value_width`). A field that
        does not fit the line starts the next one. Returns the bottom."""
        left, top = x, y
        for channel in channels:
            need = value_width(self.unit_of(channel), TEXT_SIZE)
            if left > x and left + need > x + w:
                left, top = x, top + ROW + ROW_GAP
            left = self.field(left, top, channel, w=need) + GAP
        return top + ROW

    def basis_chip(self, x: float, y: float, row: CalibrationRow) -> float:
        """The basis chip: an outline in the basis role (the disagree role when
        contested), dashed and muted for a value still wanted, page text for a
        value held live; a text too long for one line in two. Returns its bottom."""
        if row.channels:
            colour, dashed = PAGE_TEXT, False
        elif row.shown is None:
            colour, dashed = MUTED_TEXT, True
        else:
            colour, dashed = symbols.BASIS_ROLES.get(row.state, MUTED_TEXT), False
        lines = _wrap_label(row.chip, CAL_CHIP_WIDTH - GAP, SMALL_SIZE, TILE_LINES)
        w = min(max(text_width(t, SMALL_SIZE) for t in lines) + GAP, CAL_CHIP_WIDTH)
        h = ROW + (len(lines) - 1) * SMALL_LINE
        props = {
            "lineColor": colour,
            "lineWidth": LINE_WIDTH,
            "fill": False,
            "lineStyle": "dashed" if dashed else "solid",
        }
        self.s.widget("rectangle", x, y + CHIP_INSET, w, h - 2 * CHIP_INSET, props)
        if len(lines) == 1:
            self.text(x + GAP / 2, y, lines[0], size=SMALL_SIZE, w=w - GAP, fg=colour)
        else:
            first = y + (ROW - SMALL_LINE) / 2
            for n, line in enumerate(lines):
                self.text(x + GAP / 2, first + n * SMALL_LINE, line, size=SMALL_SIZE, w=w - GAP, h=SMALL_LINE, fg=colour)  # fmt: skip
        return y + h

    def member_name(self, member: Link | Fact) -> str:
        """A member's short name (:attr:`Thing.short`); a member that is no thing of
        the graph by its label's last part."""
        if isinstance(member, Fact):
            return member.text
        try:
            return self.graph.thing(member.target.id, member.target.kind).short
        except KeyError:
            return id_tail(member.label)

    def check_header(self) -> None:
        tags = (_need("table"),)
        channel_w, thing_w = self.check_cols
        x = self.text(MARGIN, self.y, "channel", size=SMALL_SIZE, w=channel_w, tags=tags)
        x = self.text(x, self.y, "thing", size=SMALL_SIZE, w=thing_w, tags=tags)
        for title in ("on device", "typed", "rule", "source"):
            x = self.text(x, self.y, title, size=SMALL_SIZE, w=CELL_WIDTH, tags=tags)
        self.next_row()

    def check_row(self, row: CheckRow) -> None:
        channel_w, thing_w = self.check_cols
        x = self.text(MARGIN, self.y, row.channel, w=channel_w)
        x = self.text(x, self.y, row.thing or "", w=thing_w)
        for flag in (row.on_device, row.typed, row.has_rule, row.has_source):
            x = self.text(x, self.y, _YES if flag else _NO, w=CELL_WIDTH)
        self.next_row()

    def totals(self, totals: Totals) -> None:
        text = (
            f"{totals.channels} channels · {totals.on_device} on a device · "
            f"{totals.typed} typed · {totals.has_rule} with a rule · "
            f"{totals.has_source} with a source"
        )
        self.text(MARGIN, self.y, text, w=STRIP_WIDTH)
        self.next_row()

    def node(self, node: Node, depth: int = 0, plan: str | None = None) -> None:
        """A roll-up node's row; the root's direct children follow it, one level only.

        The screen has one :class:`Budget`, spent by the root's row first
        (:meth:`plan`): a row is ``live`` (one light), drawn in ``parts``
        (one small light per part, each within the limit), or ``open``: the
        aggregate mark, its channel count and a link to its own screen. A
        root that is not live with children shows them instead, under a
        line that says its summary needs a server-side aggregate.
        """
        x: float = MARGIN + depth * INDENT
        lights = [describe_light(self.graph, c) for c in node.channels]
        rule: Ladder | str | None = None
        decisions: dict[int, str] = {}
        if depth == 0:
            plan, decisions = self.plan(node)
        if node.no_rule:
            x = self.no_rule(x, self.y)
        elif plan == "summary":
            tags = (_need("server-side aggregate"),)
            self.s.widget(
                "text-label",
                x,
                self.y,
                STRIP_WIDTH - x + MARGIN,
                ROW,
                {"text": _SUMMARY_TEXT.format(len(node.channels)), "fontSize": SMALL_SIZE},
                tags=tags,
            )
            self.next_row()
            for i, child in enumerate(node.children):
                self.node(child, 1, decisions.get(i))
            return
        else:
            rule = ladder(self.graph, lights, node=True) if plan == "live" else "aggregate state"
            x = self.light(x, self.y, lights, rule=rule)
        if node.link is not None:
            x = self.button(x + GAP, self.y, node.link, label=node.label)
        else:
            x = self.text(x + GAP, self.y, node.label)
        height: float = ROW
        if not node.no_rule and plan != "live":
            text = _AGGREGATE_TEXT.format(len(node.channels))
            if plan == "open" and node.link is not None:
                x = self.button(x + GAP, self.y, Link(text, node.link.target))
            else:
                x = self.text(x + GAP, self.y, text, size=SMALL_SIZE)
        if plan == "parts":
            height = self.parts(chunks(self.graph, lights), x + GAP, MARGIN + (depth + 1) * INDENT)
        self.next_row(height)
        for i, child in enumerate(node.children if depth == 0 else ()):
            self.node(child, 1, decisions.get(i))

    def plan(self, root: Node) -> tuple[str, dict[int, str]]:
        """How a roll-up screen spends its budget: the root's plan and each child's.

        In order: the root's own light when it fits one rule; then each child
        that fits one rule gets its light; every other child is ``open``, a
        link down to its own screen. A root that is not live is in
        parts when it has no children and they fit, else a ``summary``.
        """
        budget = self.budget  # the screen's: what is left goes to the rows beside

        def ladder_of(node: Node) -> Ladder | str:
            lights = [describe_light(self.graph, c) for c in node.channels]
            return ladder(self.graph, lights, node=True)

        def part_costs(node: Node) -> list[tuple[int, frozenset[str]]]:
            lights = [describe_light(self.graph, c) for c in node.channels]
            return [
                light_cost(ladder(self.graph, part, node=len(part) > 1))
                for part in chunks(self.graph, lights)
            ]

        decisions: dict[int, str] = {}
        if root.no_rule:
            plan = "none"
        else:
            rule = ladder_of(root)
            if isinstance(rule, Ladder) and budget.take([light_cost(rule)]):
                plan = "live"
            elif isinstance(rule, str) and rule != "aggregate state":
                plan = "live"  # drawn unknown and counted: it binds nothing
            elif not root.children and budget.take(part_costs(root)):
                plan = "parts"
            else:
                plan = "summary" if root.children else "open"
        for i, child in enumerate(root.children):
            if child.no_rule:
                continue
            rule = ladder_of(child)
            if (isinstance(rule, Ladder) and budget.take([light_cost(rule)])) or (
                isinstance(rule, str) and rule != "aggregate state"
            ):
                decisions[i] = "live"  # a rule that cannot be written is drawn unknown
            else:
                decisions[i] = "open"  # its own screen has its lights
        return plan, decisions

    def parts(self, parts: Sequence[Sequence[Light]], x0: float, line_x: float) -> float:
        """The parts of a node too long for one rule: a small light each, part i of n.

        The first line follows the row's label; the next lines start at
        *line_x* below it. Returns the row's height.
        """
        places, height = tile_positions(len(parts), x0, line_x, ROW)
        for number, (part, (x, dy)) in enumerate(zip(parts, places, strict=True), 1):
            rule = ladder(self.graph, part, node=len(part) > 1)
            top = self.y + dy - (ROW - MOSAIC_LIGHT) / 2 if dy else self.y
            tooltip = _PART.format(number, len(parts))
            self.light(x, top, part, rule=rule, size=MOSAIC_LIGHT, tile=True, tooltip=tooltip)
        return max(ROW, height)

    def nav(self, x, y, w, h, link: Link, *, outline: bool = False) -> None:
        """A box over what is drawn there that opens *link*'s screen.

        Every link is one: a button's text is a label under an *outline* in
        the border role, a symbol or a tile has a transparent box that shows
        the link's label while the pointer is on it. A screen the write does not
        produce (*screens* gives no file for it) gets no box: what is drawn there
        names it as text.
        """
        file = self.screens(link.target)
        if file is None:
            return
        props: dict[str, Any] = {
            "lineColor": BOX_BORDER if outline else TRANSPARENT,
            "lineWidth": LINE_WIDTH if outline else 0,
            "fill": False,
            _TOOLTIP: link.label,
            "actions": [
                {
                    "type": "open_display",
                    "file": file,
                    "target": _NAVIGATION_TARGET,
                }
            ],
        }
        if not outline:
            props[_HOVER_LABEL] = link.label
        self.s.widget("rectangle", x, y, w, h, props, tags=(_need("clickable text"),))

    # -- the strip ------------------------------------------------------------

    def strip(self, item: Strip) -> None:
        """A place in beam order: the title line, each row of symbols with its locator,
        the legend of the symbols drawn, the tray, and the count of the rest."""
        self.strip_title()
        self.strip_order = {e.thing: i for i, e in enumerate(item.elements)}
        if item.elements:
            widths = {kind: units * SYMBOL_UNIT for kind, units in symbols.WIDTH_UNITS.items()}
            for row in _strip_rows(item.elements, widths):
                self.strip_row(row, widths, item.extent_m)
            self.values_line()
            self.legend(item.elements)
        if item.tray:
            self.tray(item)
        elif item.others is not None:
            self.quiet_link(MARGIN + STRIP_WIDTH, self.y, item.others)
            self.next_row()

    def strip_title(self) -> None:
        """What the strip is, left; what its looks mean, right."""
        self.text(MARGIN, self.y, _STRIP_TITLE)
        w = text_width(_LOOKS_LEGEND, SMALL_SIZE)
        self.text(
            MARGIN + STRIP_WIDTH - w, self.y, _LOOKS_LEGEND, size=SMALL_SIZE, w=w, fg=MUTED_TEXT
        )
        self.next_row()

    def strip_row(
        self,
        row: Sequence[StripElement],
        widths: Mapping[str, float],
        extent: tuple[float, float] | None,
    ) -> None:
        drawn = [
            e if e.symbol in symbols.WIDTH_UNITS else replace(e, symbol="generic") for e in row
        ]
        placed, _scale = strip.arrange_scaled(drawn, widths, STRIP_WIDTH, fill=True)
        beam = [p for p in placed if p.lane != "above"]
        badges = [p for p in placed if p.lane == "above"]
        # a ringed badge needs its ring's room beside its neighbour's
        ringed = any(p.channels for p in badges)
        levels = stem_levels(
            [p.x for p in badges], BADGE + BADGE_GAP + (2 * RING_PAD if ringed else 0)
        )
        # the upper lane only when something stands in it
        tags = stem_levels([p.x for p in beam], TAG_WIDTH + STRIP_LABEL_GAP)
        above: float = STRIP_AXIS  # room over the axis: rings' words, then badges with theirs
        if any(p.channels for p in beam):
            above += RING_PAD + (max(tags) + 1) * TAG_HEIGHT
        if badges:
            stem = STEM_HIGH if any(levels) else STEM_LOW
            ruled = any(p.channels for p in badges)
            above = max(above, stem + BADGE / 2 + (RING_PAD + TAG_HEIGHT if ruled else 0))
        axis = self.y + above + GAP
        self.line([(MARGIN, axis), (MARGIN + STRIP_WIDTH, axis)], PAGE_TEXT, AXIS_WIDTH)
        self.taken = []  # boxes a badge's name and values must clear: tags, badges
        for p, tag_level in zip(beam, tags, strict=True):
            x = MARGIN + p.x
            with self.s.item("strip-element", p.thing, self.strip_order.get(p.thing)):
                self.symbol(p, x, axis, p.width / symbols.WIDTH_UNITS[p.symbol], tag_level)
                if p.link is not None:
                    half = p.width / 2
                    link = replace(p.link, label=p.thing)
                    self.nav(x - half, axis - STRIP_AXIS, p.width, 2 * STRIP_AXIS, link)
        centres = []
        for p, level in zip(badges, levels, strict=True):
            x = MARGIN + p.x
            cy = axis - (STEM_HIGH if level else STEM_LOW)
            with self.s.item("strip-element", p.thing, self.strip_order.get(p.thing)):
                self.line([(x, cy + BADGE / 2), (x, axis)], MUTED_TEXT)
                self.badge(p, x, cy)
            self.taken.append((x - BADGE / 2, cy - BADGE / 2, BADGE, BADGE))
            centres.append((p, x, cy))
        for p, x, cy in centres:
            with self.s.item("strip-element", p.thing):
                self.badge_text(p, x, cy)
        blocks = self.value_blocks(beam)
        lines = 0
        bare = [p for p in beam if id(p) not in blocks]
        for text, left, w, level in strip_labels(bare):
            top = axis + STRIP_LABEL_TOP + level * STRIP_LABEL_STEP
            self.text(MARGIN + left, top, text, size=STRIP_LABEL_SIZE, w=w, h=STRIP_LABEL_STEP)
            lines = max(lines, level + 1)
        names = axis + STRIP_LABEL_TOP + lines * STRIP_LABEL_STEP
        below = self.draw_blocks(blocks, axis, names)
        self.y = max(axis + STRIP_AXIS, below) + LEADER_GAP
        if extent is not None:
            self.locator(beam, extent)
        else:
            self.y += ROW_GAP

    def value_blocks(self, beam: Sequence[Placed]) -> dict[int, _Block]:
        """The symbols whose values the strip shows, by ``id`` of the placed symbol: each
        a block of its short name over its values, centred on the symbol and as wide as
        the name and :func:`value_width` of each unit need, in one of
        :data:`STRIP_VALUE_LANES` lanes (:func:`value_lanes`). A block that fits no lane
        is left out and counted in :attr:`crowded`; one the budget refuses, in
        :attr:`dropped`; either symbol keeps its bare name."""
        wanted = []
        for p in beam:
            if not p.annotations:
                continue
            name = p.short if p.short is not None else _short(p.thing)
            w = max(text_width(name, STRIP_LABEL_SIZE), text_width(VALUE_SAMPLE, SMALL_SIZE))
            left = min(max(MARGIN + p.x - w / 2, MARGIN), MARGIN + STRIP_WIDTH - w)
            wanted.append((p, name, left, w))
        lanes = value_lanes([(left, w, p.symbol) for p, _name, left, w in wanted])
        blocks: dict[int, _Block] = {}
        for (p, name, left, w), lane in zip(wanted, lanes, strict=True):
            if lane is None:
                self.crowded += len(p.annotations)
                continue
            channels = self.affordable(p)
            if channels:
                blocks[id(p)] = _Block(name, tuple(channels), MARGIN + p.x, left, w, lane)
        return blocks

    def draw_blocks(self, blocks: Mapping[int, _Block], axis: float, top: float) -> float:
        """*blocks* from *top* down, each lane under the one before, a gap of
        :data:`ROW_GAP` above each; a thin leader from the axis to each block of a
        lower lane, at its symbol, jogging into the gap beside a block above it
        (:func:`leader_path`). Returns the bottom."""
        heights: dict[int, float] = {}
        for block in blocks.values():
            height = (1 + len(block.channels)) * VALUE_STEP
            heights[block.lane] = max(heights.get(block.lane, 0), height)
        tops, y = {}, top + ROW_GAP
        for lane in sorted(heights):
            tops[lane] = y
            y += heights[lane] + ROW_GAP
        for block in blocks.values():
            y0 = tops[block.lane]
            if block.lane:
                above = [(b.left, b.w) for b in blocks.values() if b.lane < block.lane]
                path = leader_path(block.cx, axis + STRIP_AXIS, y0, tops[0], above)
                self.line(path, MUTED_TEXT)
            self.text(
                block.left, y0, block.name, size=STRIP_LABEL_SIZE, w=block.w, h=VALUE_STEP,
                align="center",
            )  # fmt: skip
            for i, channel in enumerate(block.channels, start=1):
                self.value(block.left, y0 + i * VALUE_STEP, block.w, channel)
        return y - ROW_GAP if heights else top

    def affordable(self, element: Placed) -> list[str]:
        """The channels of *element*'s values, when the screen's budget allows them all;
        none, counted in :attr:`dropped`, when it does not."""
        channels = [channel for channel, _unit in element.annotations]
        if not channels:
            return []
        if not self.budget.take([(len(channels), frozenset(channels))]):
            self.dropped += len(channels)
            return []
        for channel, unit in element.annotations:
            units = self.shown.setdefault(element.symbol, {})
            signal = self.graph.channel(channel).signal
            if units.get(signal) is None:  # the first unit stated for the signal
                units[signal] = unit
        return channels

    def value(self, x: float, y: float, w: float, channel: str, align: str = "center") -> None:
        """A channel's live value on the strip: a static label, small and muted, whose
        text a formula rule sets to the value at :data:`SIGNIFICANT_DIGITS`
        significant digits (:func:`significant_expression`). The viewer appends no
        unit to it (the strip's values line names the units); it shows
        :data:`NO_VALUE` until the rule delivers and while the channel is
        disconnected, since the rule's output is stale then."""
        self.formula_text(
            x, y, w, VALUE_STEP, channel, significant_expression("{v}"), size=SMALL_SIZE,
            fg=MUTED_TEXT, bg=TRANSPARENT, align=align,
        )  # fmt: skip

    def values_line(self) -> None:
        """What the values on the strip are, and how many the budget left out."""
        if self.shown:
            parts = []
            for kind in symbols.KINDS:
                if kind in self.shown:
                    signals = ", ".join(
                        f"{signal} ({unit})" if unit else str(signal)
                        for signal, unit in self.shown[kind].items()
                    )
                    parts.append(f"{SYMBOL_WORDS[kind]}: {signals}")
            line = _VALUES
            for part in parts:  # one line per what fits, a part never split
                joined = part if line == _VALUES else f"{line} · {part}"
                if line != _VALUES and text_width(joined, SMALL_SIZE) > STRIP_WIDTH:
                    self.text(MARGIN, self.y, line, size=SMALL_SIZE, fg=MUTED_TEXT)
                    self.y += ROW
                    line = part
                else:
                    line = _VALUES + part if line == _VALUES else joined
            self.text(MARGIN, self.y, line, size=SMALL_SIZE, fg=MUTED_TEXT)
            self.next_row()
        for count, words in ((self.crowded, _CROWDED), (self.dropped, _DROPPED)):
            if count:
                self.text(MARGIN, self.y, not_shown(count, words), size=SMALL_SIZE, fg=MUTED_TEXT)
                self.next_row()

    def badge(self, element: Placed, cx: float, cy: float) -> None:
        """A device off the axis: a circle with its kind's letter, filled with its
        type colour, its state ringed (:meth:`marked`)."""
        circle, letter = symbols.badge(cx, cy, BADGE, element.symbol, SMALL_SIZE)
        fill = self.colours.get(element.symbol)
        circle["props"].update(
            {
                "fillColor": fill if fill is not None else PANEL_BACKGROUND,
                "lineColor": PAGE_TEXT,
                "lineStyle": "dashed" if not element.channels else "solid",
                _TOOLTIP: element.thing,
            }
        )
        if fill is not None:  # a letter on a type colour reads in the page's ground
            letter["props"]["foregroundColor"] = PAGE_BACKGROUND
        self.marked(element, [circle])
        self.s.add(letter)
        if element.link is not None:
            link = replace(element.link, label=element.thing)
            self.nav(cx - BADGE / 2, cy - BADGE / 2, BADGE, BADGE, link)

    def badge_text(self, element: Placed, cx: float, cy: float) -> None:
        """A badge's short name right of it, its values under the name; each left out
        only when it would touch a tag, a badge or another badge's text, or leave
        the strip."""
        x = cx + BADGE / 2 + GAP / 2
        name = element.short if element.short is not None else _short(element.thing)
        w = text_width(name, SMALL_SIZE)
        box = (x, cy - VALUE_STEP, w, VALUE_STEP)
        if self.free(box):
            self.text(x, cy - VALUE_STEP, name, size=SMALL_SIZE, w=w, h=VALUE_STEP)
            self.taken.append(box)
        for i, channel in enumerate(self.affordable(element)):
            top = cy + i * VALUE_STEP
            w = max(BADGE_VALUE_WIDTH, text_width(VALUE_SAMPLE, SMALL_SIZE))
            box = (x, top, w, VALUE_STEP)
            if self.free(box):
                self.value(x, top, w, channel, align="left")
                self.taken.append(box)

    def free(self, box: tuple[float, float, float, float]) -> bool:
        """Whether *box* stays inside the strip and touches nothing in :attr:`taken`."""
        x, y, w, h = box
        if x + w > MARGIN + STRIP_WIDTH:
            return False
        return all(
            x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y
            for ox, oy, ow, oh in self.taken
        )

    def symbol_rule(self, element: Placed) -> tuple[Ladder | None, list[str]]:
        """The live rule of a strip element (the worst of its channels, as a roll-up
        node reads them) and the needs it hits; ``None`` when it has no live rule."""
        if not element.channels:
            return None, []
        lights = [describe_light(self.graph, c) for c in element.channels]
        rule = ladder(self.graph, lights, node=True)
        if isinstance(rule, Ladder) and rule.arms and not self.budget.take([ring_cost(rule)]):
            rule = "aggregate state"
        tags = [_need("state light")]
        if rule == "aggregate state":
            return None, [*tags, _need("aggregate state"), _AGGREGATE_SYMBOL]
        if isinstance(rule, str):
            self.s.unknown += 1
            return None, [*tags, _need(rule), _UNKNOWN_TAG + rule]
        if rule.unwritten:
            tags.append(_need("label condition"))
        return (rule if rule.arms else None), tags

    def symbol(self, element: Placed, x: float, y: float, unit: float, level: int = 0) -> None:
        """A device's symbol: filled with its type colour, never with its state; its
        state is a ring around it and a word above (:meth:`marked`)."""
        shapes = symbols.coloured(
            element.symbol,
            x,
            y,
            unit=unit,
            fill=self.colours.get(element.symbol),
            dashed=not element.channels,
        )
        for node in shapes:
            node["props"][_TOOLTIP] = element.thing
        self.marked(element, shapes, level)

    def marked(self, element: Placed, shapes: list[dict], level: int = 0) -> None:
        """*shapes*, and around them the ring and the tag that show the worst state of
        the element's channels live. A thing with no rule gets neither (its outline is
        dashed); one whose rule cannot be written or is past the budget gets them in
        the aggregate or unknown look, with no rules."""
        for node in shapes:
            self.s.add(node)
        if not element.channels:
            return
        left = min(n["geometry"]["x"] for n in shapes)
        top = min(n["geometry"]["y"] for n in shapes)
        right = max(n["geometry"]["x"] + n["geometry"]["width"] for n in shapes)
        bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in shapes)
        live, tags = self.symbol_rule(element)
        static = "aggregate" if _AGGREGATE_SYMBOL in tags else "unknown"
        outline = symbols.ring(left, top, right, bottom, RING_PAD, static)
        outline["props"][_TOOLTIP] = element.thing
        above = top - RING_PAD - level * TAG_HEIGHT
        low, high = MARGIN + TAG_WIDTH / 2, MARGIN + STRIP_WIDTH - TAG_WIDTH / 2
        centre = min(max((left + right) / 2, low), high)  # the word stays on the page
        word = symbols.tag(centre, above, TAG_WIDTH, TAG_HEIGHT, SMALL_SIZE, static)
        g = word["geometry"]
        self.taken.append((g["x"], g["y"], g["width"], g["height"]))
        rules: list[dict] = []
        word_rules: list[dict] = []
        if live is not None:
            looks = {state: symbols.ring_look(state) for state in symbols.STATES}
            rules = [
                self.s.rule(
                    f"{element.thing} ring {prop}",
                    prop,
                    live,
                    {st: looks[st][prop] for st in symbols.STATES},
                )
                for prop in symbols.RING_PROPS
            ]
            prop = symbols.TAG_PROP
            word_rules = [
                self.s.rule(
                    f"{element.thing} tag",
                    prop,
                    live,
                    {st: looks[st][prop] for st in symbols.STATES},
                )
            ]
        self.s.add(outline, rules=rules, tags=tags)
        self.s.add(word, rules=word_rules)

    def locator(self, beam: Sequence[Placed], extent: tuple[float, float]) -> None:
        """The to-scale bar under a row: its caption, a mark per symbol where the
        symbol truly is, a leader from each symbol's name down to its mark, and
        the metres. The bar spans the row's own stretch of the beam, not the whole
        place's *extent*, so a row's leaders run down, not across the page; the
        caption says which stretch of the whole it is."""
        whole_lo, whole_hi = extent
        found = [p.s_m for p in beam]
        lo, hi = (min(found), max(found)) if found else extent
        pad = max((hi - lo) * LOCATOR_PAD, LOCATOR_LEAST_M / 2)
        lo, hi = max(lo - pad, min(whole_lo, lo)), min(hi + pad, max(whole_hi, hi))
        span = hi - lo
        caption = _LOCATOR_CAPTION + _LOCATOR_STRETCH.format(lo, hi, whole_lo, whole_hi)
        self.text(MARGIN, self.y, caption, size=SMALL_SIZE, fg=MUTED_TEXT)
        leaders_top = self.y + ROW
        bar = leaders_top + LEADER_DROP + LOCATOR_BIG
        self.line([(MARGIN, bar), (MARGIN + STRIP_WIDTH, bar)], MUTED_TEXT, LOCATOR_BAR)
        shifted = [StripElement(p.thing, None, p.s_m - lo, p.symbol, p.lane, None) for p in beam]
        marks = strip.locator(shifted, span, STRIP_WIDTH)
        for p, (thing, x) in zip(beam, marks, strict=True):
            big = p.symbol in _MAGNETS
            reach = LOCATOR_BIG if big else LOCATOR_MARK
            colour, width = (PAGE_TEXT, AXIS_WIDTH) if big else (MUTED_TEXT, LINE_WIDTH)
            self.line([(MARGIN + x, bar - reach), (MARGIN + x, bar)], colour, width, tooltip=thing)
            self.line([(MARGIN + p.x, leaders_top), (MARGIN + x, bar - reach)], MUTED_TEXT)
        top = bar + LOCATOR_BAR
        for metres, x in _ticks(lo, hi):
            self.line([(MARGIN + x, top), (MARGIN + x, top + TICK)], MUTED_TEXT)
            label = f"{metres:g}"
            w = text_width(label, SMALL_SIZE)
            self.text(MARGIN + x - w / 2, top + TICK, label, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
        self.y = top + TICK + ROW + ROW_GAP

    def legend(self, elements: Sequence[StripElement]) -> None:
        """Each symbol the strip draws, as it is drawn, and its name."""
        present = {e.symbol if e.symbol in symbols.WIDTH_UNITS else "generic" for e in elements}
        above = {e.symbol for e in elements if e.lane == "above"}
        x: float = MARGIN
        cy = self.y + LEGEND_ROW / 2
        for kind in symbols.KINDS:
            if kind not in present:
                continue
            word = SYMBOL_WORDS[kind]
            w = BADGE if kind in above else symbols.WIDTH_UNITS[kind] * LEGEND_SYMBOL
            room = w + GAP + text_width(word, SMALL_SIZE)
            if x + room > MARGIN + STRIP_WIDTH:
                x = MARGIN
                self.y += LEGEND_ROW
                cy = self.y + LEGEND_ROW / 2
            if kind in above:
                drawn = symbols.badge(x + w / 2, cy, BADGE, kind, SMALL_SIZE)
                fill = self.colours.get(kind)
                drawn[0]["props"].update(
                    {
                        "fillColor": fill if fill is not None else PANEL_BACKGROUND,
                        "lineColor": PAGE_TEXT,
                    }
                )
                if fill is not None:
                    drawn[1]["props"]["foregroundColor"] = PAGE_BACKGROUND
            else:
                drawn = symbols.coloured(
                    kind, x + w / 2, cy, unit=LEGEND_SYMBOL, fill=self.colours.get(kind)
                )
                top = min(n["geometry"]["y"] for n in drawn)
                bottom = max(n["geometry"]["y"] + n["geometry"]["height"] for n in drawn)
                for node in drawn:  # a quadrupole sits off its axis: centre it on the row
                    node["geometry"]["y"] += cy - (top + bottom) / 2
            for node in drawn:
                self.s.add(node)
            right = self.text(x + w + GAP, cy - ROW / 2, word, size=SMALL_SIZE, fg=MUTED_TEXT)
            x = right + LEGEND_GAP
        self.y += LEGEND_ROW + ROW_GAP

    def tallies(self, found: CalibrationTallies) -> None:
        """A place's calibration chips in ONE line: "<title> <good> / <total>", tinted
        by the worst state present, each opening its kind's view; the kinds with no
        value yet in one muted chip last; the chips past the line folded, from the
        end, into "+n more", which opens the place's roll-up."""
        words = [f"{t.title} {t.good} / {t.total}" for t in found.tallies]
        widths = [text_width(w, SMALL_SIZE) + GAP for w in words]
        last = _NOT_STARTED.format(found.not_started) if found.not_started else None
        room: float = STRIP_WIDTH + CHIP_GAP  # each chip takes its width and a gap
        if last is not None:
            room -= text_width(last, SMALL_SIZE) + GAP + CHIP_GAP
        shown = len(words)
        while shown and sum(widths[:shown]) + CHIP_GAP * shown > room - (
            0
            if shown == len(words)
            else text_width(_MORE_CHIPS.format(len(words) - shown), SMALL_SIZE) + GAP + CHIP_GAP
        ):
            shown -= 1
        x: float = MARGIN
        y = self.y
        for tally, word, w in zip(found.tallies[:shown], words, widths, strict=False):
            self.tally_box(x, y, w, _TALLY_LOOKS.get(tally.state, tally.basis), word)
            self.text(x + GAP / 2, y, word, size=SMALL_SIZE, h=CHIP_HEIGHT)
            if tally.link is not None:
                self.nav(x, y, w, CHIP_HEIGHT, replace(tally.link, label=word))
            x += w + CHIP_GAP
        if shown < len(words):
            word = _MORE_CHIPS.format(len(words) - shown)
            w = text_width(word, SMALL_SIZE) + GAP
            self.tally_box(x, y, w, None, word)
            self.text(x + GAP / 2, y, word, size=SMALL_SIZE, h=CHIP_HEIGHT)
            if found.more is not None:
                self.nav(x, y, w, CHIP_HEIGHT, replace(found.more, label=word))
            x += w + CHIP_GAP
        if last is not None:
            w = text_width(last, SMALL_SIZE) + GAP
            self.tally_box(x, y, w, "none", last)
            self.text(x + GAP / 2, y, last, size=SMALL_SIZE, h=CHIP_HEIGHT, fg=MUTED_TEXT)
        self.y += CHIP_HEIGHT + CHIP_GAP

    def tally_box(self, x: float, y: float, w: float, look: str | None, tooltip: str) -> None:
        """A calibration chip's box in the look of a calibration cell's state (a basis,
        ``contested``, ``none``); a plain chip with ``None``."""
        props: dict[str, Any] = {"lineWidth": LINE_WIDTH, _TOOLTIP: tooltip}
        if look is not None and look in symbols.BASIS_TINTS:
            props |= {
                "lineColor": symbols.BASIS_ROLES[look],
                "fill": True,
                "fillColor": symbols.BASIS_TINTS[look],
                "lineStyle": "solid",
            }
        elif look == "none":
            props |= {"lineColor": MUTED_TEXT, "fill": False, "lineStyle": "dashed"}
        else:
            props |= {"lineColor": BOX_BORDER, "fill": True, "fillColor": FIELD_BACKGROUND}
        self.s.widget("rectangle", x, y, w, CHIP_HEIGHT, props)

    def chip(
        self, x: float, y: float, label: str, link: Link | None, node: Node | None = None
    ) -> float:
        """A name in a box, with its node's light when given, opening *link*; its right edge."""
        lit = node is not None
        w = text_width(label, SMALL_SIZE) + (CHIP_LIGHT + GAP if lit else 0) + GAP
        self.s.widget(
            "rectangle",
            x,
            y,
            w,
            CHIP_HEIGHT,
            {
                "lineColor": BOX_BORDER,
                "lineWidth": LINE_WIDTH,
                "fill": True,
                "fillColor": FIELD_BACKGROUND,
            },
        )
        inner = x + GAP / 2
        if node is not None:
            light_y = y + (CHIP_HEIGHT - ROW) / 2
            if node.no_rule:
                self.no_rule(inner, light_y, CHIP_LIGHT)
            else:
                self.node_light(inner, light_y, node, CHIP_LIGHT)
            inner += CHIP_LIGHT + GAP / 2
        self.text(inner, y, label, size=SMALL_SIZE, h=CHIP_HEIGHT)
        if link is not None:
            self.nav(x, y, w, CHIP_HEIGHT, replace(link, label=link.label))
        return x + w

    def node_light(self, x: float, y: float, node: Node, size: float = LIGHT_SIZE) -> float:
        """One light over a node's channels, as a roll-up reads them; the aggregate look
        when the rule is too long for the format or past the screen's budget."""
        lights = [describe_light(self.graph, c) for c in node.channels]
        rule = ladder(self.graph, lights, node=True)
        if isinstance(rule, Ladder) and rule.arms and not self.budget.take([light_cost(rule)]):
            rule = "aggregate state"
        return self.light(x, y, lights, rule=rule, size=size, tile=True)

    def chips(self, x0: float, entries: Sequence[tuple[str, Link | None, Node | None]]) -> None:
        """Chips that wrap across the content width, from *x0*."""
        x = x0
        for label, link, node in entries:
            w = text_width(label, SMALL_SIZE) + (CHIP_LIGHT + GAP if node is not None else 0) + GAP
            if x > x0 and x + w > MARGIN + STRIP_WIDTH:
                x = x0
                self.y += CHIP_HEIGHT + CHIP_GAP
            x = self.chip(x, self.y, label, link, node) + CHIP_GAP
        self.y += CHIP_HEIGHT + CHIP_GAP

    def tray(self, item: Strip) -> None:
        """What our lists hold and the strip cannot place, in a dashed frame of chips."""
        top = self.y
        self.y += TRAY_PAD
        self.text(MARGIN + TRAY_PAD, self.y, _TRAY_TITLE.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        if item.others is not None:
            self.quiet_link(MARGIN + STRIP_WIDTH - TRAY_PAD, self.y, item.others)
        self.y += ROW
        # the heading says "not placed": a reason only when it says more
        entries = [
            (e.thing if e.reason == NO_POSITION else f"{e.thing} · {e.reason}", e.link, None)
            for e in item.tray
        ]
        self.chips(MARGIN + TRAY_PAD, entries)
        self.y += TRAY_PAD - CHIP_GAP
        self.box(MARGIN, top, STRIP_WIDTH, self.y - top, line=MUTED_TEXT, dashed=True)
        self.y += GAP

    def band(self, band: Band) -> None:
        """The key numbers: an even grid of tiles across the content width, in the fewest
        rows whose tiles are :data:`MIN_TILE_WIDTH` wide, the earlier rows one tile
        longer when the count does not divide; each row spans the full width."""
        n = len(band.tiles)
        rows = next(
            r for r in range(1, n + 1) if _tile_width(math.ceil(n / r)) >= MIN_TILE_WIDTH or r == n
        )
        base, extra = divmod(n, rows)
        sizes = [base + 1] * extra + [base] * (rows - extra)
        widths = [_tile_width(size) for size in sizes]
        inner = min(widths) - 2 * TILE_PAD
        titles = max(len(_wrap(t.title.upper(), inner, SMALL_SIZE, caps=True)) for t in band.tiles)
        captions = max(len(_wrap(self.caption(t), inner, SMALL_SIZE)) for t in band.tiles)
        height = 2 * TILE_PAD + (titles + captions) * SMALL_LINE + NUMBER_HEIGHT + 2 * ROW_GAP
        i = 0
        for size in sizes:
            for x, w in spread(size, MARGIN, STRIP_WIDTH, GAP):
                self.tile(x, self.y, w, height, band.tiles[i], f"m{i}", titles)
                i += 1
            self.y += height + GAP

    @staticmethod
    def caption(item: Tile) -> str:
        """The line under a tile's number as the data gives it (:meth:`tile` may replace it)."""
        if not item.channels and not item.pairs:
            return _NO_DATA
        return item.caption or ""

    def tile(
        self, x: float, y: float, w: float, h: float, item: Tile, name: str, titles: int
    ) -> None:
        """One key number: its title, the number (a channel, or a screen formula over
        the channels its row selects) with "of N" or its unit on its baseline, and up to
        two lines under it. A count of losses turns the tile's border bad while it is
        above zero. Text is wrapped, never left to the viewer to shrink."""
        tags = (f"tile: {item.title}",)
        frame = {
            "lineColor": BOX_BORDER,
            "lineWidth": LINE_WIDTH,
            "fill": True,
            "fillColor": FIELD_BACKGROUND,
        }
        left = x + TILE_PAD
        top = y + TILE_PAD
        inner = w - 2 * TILE_PAD
        number_top = top + titles * SMALL_LINE + ROW_GAP
        caption = self.caption(item)
        pv: str | None = None
        total = len(item.pairs) if item.measure == "count_in_tolerance" else len(item.channels)
        formula = None
        extra_tags: tuple[str, ...] = ()
        if not item.channels and not item.pairs:
            pass
        elif item.measure == "value":
            if len(item.channels) == 1:
                pv = item.channels[0]
                if not self.budget.take([(0, frozenset(item.channels))]):
                    pv, caption = None, _NEEDS_SERVER
                    extra_tags = (_need("server-side aggregate"),)
            else:
                caption = _MANY_VALUES.format(len(item.channels))
        else:
            formula = metric_formula(name, item)
            reads = frozenset(formula["bindings"].values())
            fits = len(formula["expression"]) <= RULE_EXPRESSION_LIMIT
            if fits and self.budget.take([(1, reads)]):
                self.s.formulas.append(formula)
                pv = _FORMULA_PREFIX + name
            else:
                formula, caption = None, _NEEDS_SERVER
                extra_tags = (_need("server-side aggregate"),)
        rules: list[dict] = []
        if formula is not None and item.measure in ("count_false", "count_in_tolerance"):
            lost = "{0} > 0" if item.measure == "count_false" else f"{{0}} < {total}"
            rule = {
                "pvs": [{"name": pv, "trigger": True}],
                "conditions": [{"expression": lost, "value": symbols.ALARM_MAJOR}],
                "default": BOX_BORDER,
            }
            if len(engine_expression(rule, {name: formula})) <= RULE_EXPRESSION_LIMIT:
                self.s._rules += 1
                rules = [
                    {
                        "id": f"r-{self.s._rules:04d}",
                        "name": f"{item.title} border",
                        "targetProperty": "lineColor",
                        **rule,
                    }
                ]
        self.s.widget("rectangle", x, y, w, h, frame, rules=rules, tags=(*tags, *extra_tags))
        for i, line in enumerate(_wrap(item.title.upper(), inner, SMALL_SIZE, caps=True)):
            w_caps = caps_width(line, SMALL_SIZE)
            self.text(
                left, top + i * SMALL_LINE, line, size=SMALL_SIZE, w=w_caps, h=SMALL_LINE,
                fg=MUTED_TEXT, tags=tags,
            )  # fmt: skip
        counted = item.measure in _COUNTS
        own_unit = formula is None and pv is not None  # a channel: the viewer prints its unit
        cut = pv is not None and own_unit and not counted and self.cuts_unit(pv)
        if cut and pv is not None:  # the number a formula's, the unit whole beside it
            expression = self.number_expression(pv, None, item.precision)
            cut = self.budget.take([(1, frozenset({pv}))])
        beside = f"of {total}" if counted else ("" if own_unit else (item.unit or ""))
        if cut:
            beside = self.unit_of(pv) or ""  # type: ignore[arg-type]
        beside_w = text_width(beside, SMALL_SIZE) if beside else 0
        if pv is None:
            dash_w = text_width(_DASH, BIG_SIZE)
            self.text(
                left, number_top, _DASH, size=BIG_SIZE, w=dash_w, h=NUMBER_HEIGHT, fg=MUTED_TEXT,
                tags=tags,
            )  # fmt: skip
        else:
            digits = max(len(str(total)), COUNT_DIGITS) if counted else VALUE_DIGITS
            number_w: float = math.ceil(digits * CHAR_WIDTH * BIG_SIZE + NUMBER_PAD)
            size: float = BIG_SIZE
            room = inner - beside_w - GAP
            if not counted:
                unit = self.unit_of(pv) if own_unit and not cut else None
                number_w = max(number_w, value_width(unit, BIG_SIZE))
                if number_w > room:  # a smaller number, the unit whole: never cut
                    size = value_size(unit, room)
                    number_w = value_width(unit, size)
            number_w = min(number_w, room) if counted else number_w
            if cut:
                self.formula_text(
                    left, number_top, number_w, NUMBER_HEIGHT, pv, expression, size=size,
                    fg=PAGE_TEXT, bg=TRANSPARENT, align="left", tags=tags,
                )  # fmt: skip
            props = {
                "pv": pv,
                "fontSize": size,
                "foregroundColor": PAGE_TEXT,
                "backgroundColor": TRANSPARENT,
                "showUnits": False,
                "alarmSensitive": False,
                "alarmBorder": False,
                "alarmSensitiveBorder": False,
            }
            precision: list[dict] = []
            if counted:
                props["precision"] = 0
            elif item.precision is not None:  # stated by the data
                props["precision"] = item.precision
            elif formula is None and self.prints_whole(pv):
                props["precision"] = 0
            elif formula is not None:
                value = f"({formula['expression']})"
                precision = self.precision(props, value, formula["bindings"])
            else:
                precision = self.precision(props, "{v}", {"v": pv})
            if not cut:
                self.s.widget(
                    "pv-label", left, number_top, number_w, NUMBER_HEIGHT, props, tags=tags,
                    rules=precision,
                )  # fmt: skip
            if beside:  # on the number's baseline: its line's middle below the number's
                base = number_top + NUMBER_HEIGHT - SMALL_LINE - ROW_GAP
                self.small(left + number_w + ROW_GAP, base, beside, fg=MUTED_TEXT, tags=tags)
        below = number_top + NUMBER_HEIGHT + ROW_GAP
        for i, line in enumerate(_wrap(caption, inner, SMALL_SIZE)):
            self.small(left, below + i * SMALL_LINE, line, fg=MUTED_TEXT, tags=tags)

    def small(self, x: float, y: float, text: str, *, fg=PAGE_TEXT, tags=()) -> float:
        """One line of small text, its box as wide as the text."""
        w = text_width(text, SMALL_SIZE)
        return self.text(x, y, text, size=SMALL_SIZE, w=w, h=SMALL_LINE, fg=fg, tags=tags)

    def group(self, group: Group) -> None:
        """Things of one kind: chips that wrap, or summary rows in columns."""
        if group.form == "rows":
            self.summary(group)
            return
        self.text(MARGIN, self.y, group.title.upper(), size=SMALL_SIZE, fg=MUTED_TEXT)
        self.y += ROW
        entries: list[tuple[str, Link | None, Node | None]] = [
            (n.label, n.link, n) for n in group.nodes
        ]
        if group.more is not None:
            entries.append((group.more.label, group.more, None))
        self.chips(MARGIN, entries)

    def summary(
        self,
        group: Group,
        x0: float = MARGIN,
        width: float = STRIP_WIDTH,
        columns: int = SUMMARY_COLUMNS,
    ) -> None:
        """One row per node: its light, its name, and right-aligned what it reads."""
        column = (width - (columns - 1) * GAP) / columns
        per = max(1, math.ceil(len(group.nodes) / columns))
        top = self.y
        for i, node in enumerate(group.nodes):
            x = x0 + (i // per) * (column + GAP)
            y = top + (i % per) * SUMMARY_ROW
            h = SUMMARY_ROW - ROW_GAP
            self.s.widget(
                "rectangle",
                x,
                y,
                column,
                h,
                {
                    "lineColor": BOX_BORDER,
                    "lineWidth": LINE_WIDTH,
                    "fill": True,
                    "fillColor": FIELD_BACKGROUND,
                },
            )
            row_y = y + (h - ROW) / 2
            if node.no_rule:
                self.no_rule(x + GAP, row_y)
            else:
                self.node_light(x + GAP, row_y, node)
            self.text(x + GAP + LIGHT_SIZE + GAP, row_y, node.label)
            count = len(node.channels)
            note = _NO_RULE_YET if node.no_rule else _READS.format(count, "" if count == 1 else "s")
            w = text_width(note, SMALL_SIZE)
            self.text(x + column - GAP - w, row_y, note, size=SMALL_SIZE, w=w, fg=MUTED_TEXT)
            if node.link is not None:
                self.nav(x, y, column, h, node.link)
        self.y = top + per * SUMMARY_ROW + ROW_GAP

    # -- picture and elevation ------------------------------------------------

    def picture(self, picture: Picture) -> None:
        """A device class's drawing in one view, across the card's left column.

        With a body (:func:`osprey.facility.panels.picture.body`), the magnet as
        seen from its end, painted by material (:meth:`body_shape`): the yoke with
        its cavity cut out, broad poles with rounded tips short of the bore,
        copper coil blocks along their flanks, corrector windings as thin bands
        outside them, the bore as a dark disc with the beam point and the word
        "beam"; each coil's number on the yoke ring at its angle. A sensor whose
        row states its coil is a dot on that coil (its light when a health row
        reads its channel, else neutral); one with no place a hollow ring in a row
        of its kind beside the drawing. Water and terminal parts are small blocks
        in a column beside the yoke under one word (:meth:`picture_block`). No
        leaders: a part is named by its tooltip and in the legend; a value its
        channel prints stands beside it. Colour inside the drawing is material;
        state is the only other colour.

        With no body, the parts at their positions in a dashed frame that says the
        outline is not stated (:meth:`bare_picture`). A scale bar under the
        drawing, with "size not stated" when a default proportion is drawn; under
        it the legend (:meth:`picture_legend`)."""
        shape_rows = [p for p in picture.parts if p.shape]
        outline = [p for p in picture.parts if p.kind == OUTLINE_KIND]
        parts = [p for p in picture.parts if not p.shape and p.kind != OUTLINE_KIND]
        self.heading(f"{picture.view} view")
        room = min(self.right - self.x0, CARD_LEFT - COLUMN_GAP)
        canvas = min(PICTURE_SIZE, room - 2 * PICTURE_PAD)
        x0 = self.x0 + (room - canvas) / 2
        if not shape_rows and not outline and not any(p.positioned for p in parts):
            self.unarranged(picture, parts)  # no frame round nothing
            return
        if not shape_rows and not outline:
            scale = self.bare_picture(picture, parts, x0, canvas)
            self.scale_bar(x0, scale, canvas, schematic_note(picture.parts, _NOT_STATED))
            self.picture_legend(picture)
            return
        body = drawing.body(shape_rows, outline, parts)
        top = self.y
        points = [pt for shape in body.shapes for pt in _reach(shape)]
        points += [mark.at for mark in body.coils]
        for dot in body.dots:
            (a, b), r = dot.at, dot.radius
            points += [(a - r, b - r), (a + r, b + r)]
        for block in body.blocks:
            (a, b), (w, h) = block.at, block.size
            points += [(a - w / 2, b - h / 2), (a + w / 2, b + h / 2)]
        xs, ys = [a for a, _b in points], [b for _a, b in points]
        words = [(w, text_width(w.text, SMALL_SIZE)) for w in body.words]
        # the scale fits the drawing and its words to the canvas's width, no taller
        # than it is wide
        width_mm = max(xs) - min(xs)
        height_mm = max(ys) - min(ys)
        scale = (canvas - 2 * PICTURE_INSET) / (max(width_mm, height_mm) or 1.0)
        for _round in range(2):  # the words' room, at that scale
            lo = min([min(xs)] + [w.at[0] - (0 if w.left else tw / 2 / scale) for w, tw in words])
            hi = max([max(xs)] + [w.at[0] + (tw / scale if w.left else tw / 2 / scale) for w, tw in words])  # fmt: skip
            up = max([max(ys)] + [w.at[1] + SMALL_LINE / 2 / scale for w, _tw in words])
            down = min(ys)
            scale = (canvas - 2 * PICTURE_INSET) / (max(hi - lo, up - down) or 1.0)
        mid_x = (lo + hi) / 2
        height = (up - down) * scale + 2 * PICTURE_INSET
        cx = x0 + canvas / 2

        def px(point: tuple[float, float]) -> tuple[float, float]:
            return cx + (point[0] - mid_x) * scale, top + PICTURE_INSET + (up - point[1]) * scale

        for shape in body.shapes:
            self.body_shape(shape, [px(pt) for pt in shape.points])
        if body.beam is not None:
            self.beam_point(body, px)
        for mark in body.coils:  # the facility's number, on the yoke ring at its angle
            mx, my = px(mark.at)
            w = text_width(mark.number, COIL_NUMBER_SIZE)
            self.text(mx - w / 2, my - ROW / 2, mark.number, size=COIL_NUMBER_SIZE, w=w, align="center")  # fmt: skip
        for block in body.blocks:
            self.picture_block(block, px, scale)
        for word, tw in words:
            wx, wy = px(word.at)
            left = wx if word.left else wx - tw / 2
            self.text(
                left,
                wy - SMALL_LINE / 2,
                word.text,
                size=SMALL_SIZE,
                w=tw,
                h=SMALL_LINE,
                align="left" if word.left else "center",
                fg=MUTED_TEXT,
            )
        for dot in body.dots:
            x, y = px(dot.at)
            self.part_dot(dot.row, x, y, size=2 * dot.radius * scale, hollow=not dot.placed)
            if dot.row.part in picture.focus:  # the card's own part, on its owner
                ring = 2 * dot.radius * scale + 2 * FOCUS_RING
                props = {
                    "lineColor": symbols.FOCUS,
                    "lineWidth": 2,
                    "fill": False,
                    _TOOLTIP: dot.row.part,
                }
                self.s.widget("ellipse", x - ring / 2, y - ring / 2, ring, ring, props)
            if dot.placed and self.shows_number(dot.row):
                self.part_value(dot.row, x, y, cx)
        self.y = top + height + GAP
        self.scale_bar(x0, scale, canvas, schematic_note(picture.parts, scale_note(body)))
        self.picture_legend(picture)

    def unarranged(self, picture: Picture, parts: list[PicturePart]) -> None:
        """A drawing none of whose parts has a place: no frame, but a compact row of
        blocks, each the part's name over its signal's light and value, wrapping to
        the column's width, its height fitted; "arrangement not stated" under it,
        then the rows' sources."""
        per = max(1, int((self.right - self.x0 + GAP) // (ARRANGED_BLOCK + GAP)))
        high = 2 * ROW
        for n, part in enumerate(parts):
            x = self.x0 + (n % per) * (ARRANGED_BLOCK + GAP)
            y = self.y + (n // per) * (high + GAP)
            self.box(x, y, ARRANGED_BLOCK, high, fill=PANEL_BACKGROUND)
            inner = ARRANGED_BLOCK - 2 * GAP
            name = _cut(part.part, inner, SMALL_SIZE)
            self.text(x + GAP, y, name, size=SMALL_SIZE, w=inner, h=ROW, fg=MUTED_TEXT)
            self.parts_listed.add(part.part)
            channel = part.channel
            if channel is None:
                self.small(x + GAP, y + ROW, _SIGNAL_NOT_STATED, fg=MUTED_TEXT)
                continue
            vx = x + GAP
            if self.graph.rule(channel):
                if meaning_stated(self.graph, channel):
                    vx = self.light(vx, y + ROW, [describe_light(self.graph, channel)])
                else:
                    vx = self.no_rule(vx, y + ROW)
                vx += GAP
            if self.shows_number(part):
                self.field(vx, y + ROW, channel, w=x + ARRANGED_BLOCK - GAP - vx, units=True)
        rows = -(-len(parts) // per)
        self.y += rows * (high + GAP)
        self.small(
            self.x0,
            self.y,
            schematic_note(picture.parts, _ARRANGEMENT_NOT_STATED) or _ARRANGEMENT_NOT_STATED,
            fg=MUTED_TEXT,
        )
        self.y += SMALL_LINE + GAP
        self.picture_legend(picture)  # the parts are named above: their sources only

    def bare_picture(
        self, picture: Picture, parts: list[PicturePart], x0: float, canvas: float
    ) -> float:
        """Parts with no body stated, at their positions in a dashed frame with the
        beam axis as a cross: a dot each (a water part a block), its kind's word
        over each group of them. A part with no position is not placed: a hollow
        ring in a row of its kind in the frame's top left corner. Returns the
        scale, px per mm."""
        top = self.y
        cx, cy = x0 + canvas / 2, top + canvas / 2
        unplaced = [p for p in parts if not p.positioned]
        parts = [p for p in parts if p.positioned]
        points = [_picture_xy(picture.view, p) for p in parts] + [(0.0, 0.0)]
        xs, ys = [a for a, _b in points], [b for _a, b in points]
        mid = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        reach = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1
        scale = (canvas / 2 - 2 * PICTURE_INSET - PART_WORD_ROOM) / reach

        def px(point: tuple[float, float]) -> tuple[float, float]:
            return cx + (point[0] - mid[0]) * scale, cy - (point[1] - mid[1]) * scale

        half = canvas / 2 - PICTURE_INSET / 2
        self.box(cx - half, cy - half, 2 * half, 2 * half, line=MUTED_TEXT, dashed=True)
        kinds: dict[str, list[PicturePart]] = {}
        for part in unplaced:
            kinds.setdefault(part.kind, []).append(part)
        for row, (kind, members) in enumerate(kinds.items()):
            y = cy - half + PICTURE_INSET + PART_DOT / 2 + row * SMALL_LINE
            x = cx - half + PICTURE_INSET + PART_DOT / 2
            for member in members:
                self.part_dot(member, x, y, hollow=True)
                x += PART_DOT + GAP / 2
            word = kind.replace("_", " ")
            self.small(x, y - SMALL_LINE / 2, word, fg=MUTED_TEXT)
        ax, ay = px((0.0, 0.0))
        arm = BEAM_POINT
        for ends in (((ax - arm, ay), (ax + arm, ay)), ((ax, ay - arm), (ax, ay + arm))):
            self.line(list(ends), MUTED_TEXT, tooltip=_BEAM_AXIS)
        blocks: dict[str, list[tuple[float, float]]] = {}
        drawn: set[tuple[int, int, str]] = set()
        for part in parts:
            x, y = px(_picture_xy(picture.view, part))
            here = (round(x), round(y), part.kind)
            if not _water(part.kind):
                self.part_dot(part, x, y)
            elif here not in drawn:  # one block per point
                self.part_block(part, x, y)
            if here not in drawn:
                blocks.setdefault(part.kind, []).append((x, y))
            drawn.add(here)
            if self.shows_number(part):
                self.part_value(part, x, y, cx)
        for kind, spots in blocks.items():
            for group in _groups(spots, PART_GROUP_DISTANCE):
                gx = sum(x for x, _y in group) / len(group)
                gy = min(y for _x, y in group) - PART_BLOCK[1] / 2 - GAP / 2
                word = kind.replace("_", " ")
                w = text_width(word, SMALL_SIZE)
                self.text(
                    gx - w / 2,
                    gy - SMALL_LINE,
                    word,
                    size=SMALL_SIZE,
                    w=w,
                    h=SMALL_LINE,
                    align="center",
                    fg=MUTED_TEXT,
                )
        self.y = top + canvas + GAP
        return scale

    def body_shape(self, shape: drawing.Shape, points: list[tuple[float, float]]) -> None:
        """One shape of a drawing's body at *points* (px), painted by its material: the
        yoke dark steel (its outline dashed and muted when no row states it), its
        cavity the panel's ground, a pole steel, a coil copper (its outline set by
        the state of the coil row's channel when a health row reads it), a winding
        a blue band (a second one purple), the bore the dark bore colour."""
        row = shape.row
        line, fill, width = {
            "yoke": (symbols.STEEL_EDGE, symbols.STEEL_DARK, BODY_LINE),
            "window": (PANEL_BACKGROUND, PANEL_BACKGROUND, 0),
            "pole": (symbols.STEEL_EDGE, symbols.STEEL, LINE_WIDTH),
            "coil": (symbols.COPPER_EDGE, symbols.COPPER, BODY_LINE),
            "winding": (symbols.WINDING, symbols.WINDING, 0),
            "winding2": (symbols.WINDING_2, symbols.WINDING_2, 0),
            "bore": (symbols.STEEL_EDGE, symbols.BORE, LINE_WIDTH),
        }[shape.role]
        props: dict[str, Any] = {"lineColor": line, "lineWidth": width, "fill": True}
        props |= {"fillColor": fill, _TOOLTIP: shape.name}
        if shape.dashed:
            props |= {"lineColor": MUTED_TEXT, "lineStyle": "dashed"}
        rules: list[dict[str, Any]] = []
        channel = row.channel if row is not None else None
        if shape.role == "coil" and channel is not None and self.graph.rule(channel):
            rule = ladder(self.graph, [describe_light(self.graph, channel)])
            if not isinstance(rule, str) and rule.arms:
                warm = {"normal": symbols.COPPER_EDGE, "warn": symbols.ALARM_MINOR}
                warm["bad"] = symbols.ALARM_MAJOR
                colours: dict[str, Any] = {
                    st: warm.get(st, symbols.COPPER_EDGE) for st in symbols.STATES
                }
                widths: dict[str, Any] = {
                    st: COIL_WARM_WIDTH if st in ("warn", "bad") else BODY_LINE
                    for st in symbols.STATES
                }
                rules.append(self.s.rule(f"{channel} lineColor", "lineColor", rule, colours))
                rules.append(self.s.rule(f"{channel} lineWidth", "lineWidth", rule, widths))
        if shape.form == "ellipse":
            (x1, y1), (x2, y2) = points
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            self.s.widget("ellipse", x, y, w, h, props, rules=rules)
            return
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        x, y = min(xs), min(ys)
        props["points"] = [{"x": _n(a - x), "y": _n(b - y)} for a, b in points]
        # a box no lower than a row: the viewer sets a lower one on the text baseline,
        # below where its points say (the points keep their place in a taller box)
        h = max(max(ys) - y, POLYGON_BOX)
        self.s.widget("polygon", x, y, max(xs) - x, h, props, rules=rules)

    def picture_block(self, block: drawing.Block, px: Callable, scale: float) -> None:
        """A water or terminal part beside the yoke: a small rounded block (water, or
        steel with its label), dashed when its row states no place."""
        x, y = px(block.at)
        w, h = block.size[0] * scale, block.size[1] * scale
        water = block.role == "water"
        fill = symbols.WATER if water else symbols.STEEL
        line = fill if water else symbols.STEEL_EDGE
        props: dict[str, Any] = {"lineColor": line, "lineWidth": BODY_LINE, "fill": True}
        props |= {"fillColor": fill, "cornerRadius": PART_BLOCK_CORNER, _TOOLTIP: block.row.part}
        if not block.placed:
            props |= {"fill": False, "lineStyle": "dashed"}
            if not water:
                props["lineColor"] = MUTED_TEXT
        self.s.widget("rectangle", x - w / 2, y - h / 2, w, h, props)
        if block.label:
            tw = text_width(block.label, SMALL_SIZE)
            self.text(x - tw / 2, y - SMALL_LINE / 2, block.label, size=SMALL_SIZE, w=tw, h=SMALL_LINE, align="center")  # fmt: skip

    def beam_point(self, body: drawing.Body, px: Callable) -> None:
        """The beam: a point, and the word "beam" under it inside the bore when it
        has the room; a bore too small for it (pole tips close round it) has the
        point alone, its tooltip saying "beam"."""
        bx, by = px(body.beam)
        dot = BEAM_POINT
        fill = {"lineColor": PAGE_TEXT, "lineWidth": LINE_WIDTH, "fill": True}
        fill |= {"fillColor": PAGE_TEXT, _TOOLTIP: "beam"}
        self.s.widget("ellipse", bx - dot / 2, by - dot / 2, dot, dot, fill)
        bores = [s for s in body.shapes if s.role == "bore"]
        radius = 0.0
        if bores:
            (x1, y1), (x2, y2) = (px(pt) for pt in bores[0].points)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if 2 * min(w, h) < max(w, h):  # a flat gap: the point says it
                return
            radius = min(w, h) / 2
        word = "beam"
        w = text_width(word, SMALL_SIZE)
        if bores and radius < dot + SMALL_LINE + GAP:
            return
        below = by + dot if radius >= dot + SMALL_LINE + GAP else by + max(radius, dot) + 2
        self.text(
            bx - w / 2,
            below,
            word,
            size=SMALL_SIZE,
            w=w,
            h=SMALL_LINE,
            align="center",
            fg=MUTED_TEXT,
        )

    def part_dot(
        self,
        part: PicturePart,
        x: float,
        y: float,
        *,
        size: float = PART_DOT,
        hollow: bool = False,
    ) -> None:
        """A sensor at (*x*, *y*), *size* across: its light when a health row reads its
        channel, else a neutral dot; a neutral ring when *hollow* (its place not
        stated)."""
        channel = part.channel
        if channel is not None and self.graph.rule(channel) and not hollow:
            light = describe_light(self.graph, channel)
            self.light(x - size / 2, y - ROW / 2, [light], size=size, tooltip=part.part)
            return
        props: dict[str, Any] = {"lineColor": PANEL_BACKGROUND, "lineWidth": BODY_LINE}
        props |= {"fill": True, "fillColor": MUTED_TEXT, _TOOLTIP: part.part}
        if hollow:
            props |= {"lineColor": MUTED_TEXT, "lineWidth": AXIS_WIDTH, "fill": False}
        half = size / 2
        self.s.widget("ellipse", x - half, y - half, size, size, props)

    def part_block(self, part: PicturePart, x: float, y: float) -> None:
        """A part off the body at (*x*, *y*): a small rounded block, water for a kind
        :data:`WATER_WORDS` names, steel otherwise."""
        fill = symbols.WATER if _water(part.kind) else symbols.STEEL
        w, h = PART_BLOCK
        props = {"lineColor": fill, "lineWidth": LINE_WIDTH, "fill": True, "fillColor": fill}
        props |= {"cornerRadius": PART_BLOCK_CORNER, _TOOLTIP: part.part}
        self.s.widget("rectangle", x - w / 2, y - h / 2, w, h, props)

    def part_value(self, part: PicturePart, x: float, y: float, cx: float) -> None:
        """The value a part's channel prints, a readback box beside the part, on the
        side away from the centre *cx*."""
        channel = part.channel or ""
        w = max(PART_VALUE_WIDTH, self.value_span(channel))
        side = PART_DOT / 2 + GAP / 2
        vx = x + side if x >= cx else x - side - w
        self.field(vx, y - ROW / 2, channel, w=w, units=True)

    def shows_number(self, part: PicturePart) -> bool:
        """Whether a part shows its channel's value beside it: a number, or a state
        with no light to show it."""
        channel = part.channel
        if channel is None:
            return False
        return self.own_kind(channel) in _NUMBERS or not self.graph.rule(channel)

    def scale_bar(self, x: float, scale: float, canvas: float, note: str | None) -> None:
        """A bar of a round number of millimetres under a drawing at *scale* (px per
        mm), its length written beside it; then *note*, muted, beside it when it fits
        before the right edge, else wrapped on the lines under the bar."""
        left = x
        most = canvas * SCALE_BAR_SHARE / scale
        mm = max((m for m in SCALE_BAR_MM if m <= most), default=None)
        y = self.y + SMALL_LINE / 2
        if mm is not None:
            length = mm * scale
            tip = f"scale: {mm:g} mm"
            self.line([(x, y), (x + length, y)], MUTED_TEXT, tooltip=tip)
            for end in (x, x + length):
                self.line([(end, y - TICK / 2), (end, y + TICK / 2)], MUTED_TEXT, tooltip=tip)
            x = self.small(x + length + GAP, self.y, f"{mm:g} mm", fg=MUTED_TEXT) + GAP
        if note and x + GAP + text_width(note, SMALL_SIZE) <= self.right:
            self.small(x + GAP, self.y, note, fg=MUTED_TEXT)
        elif note:
            for line in _prose(note, self.right - left, SMALL_SIZE):
                self.y += SMALL_LINE
                self.small(left, self.y, line, fg=MUTED_TEXT)
        self.y += SMALL_LINE + GAP

    def picture_legend(self, picture: Picture) -> None:
        """The drawing's legend: per kind of part (coils and corrector windings with
        them), its members once by number; then the sources of the rows, once each.
        A part a former view of the class listed is not listed again."""
        kinds: dict[str, list[PicturePart]] = {}
        for part in picture.parts:
            iron = part.shape and part.kind in _UNLISTED  # the drawing, not a part
            if iron or part.kind == OUTLINE_KIND or part.part in self.parts_listed:
                continue
            self.parts_listed.add(part.part)
            kinds.setdefault(part.kind, []).append(part)
        width = self.right - self.x0
        for kind, members in kinds.items():
            word = kind.replace("_", " ")
            x = self.text(
                self.x0,
                self.y,
                word,
                size=SMALL_SIZE,
                w=PART_LABEL_WIDTH,
                h=SMALL_LINE,
                fg=MUTED_TEXT,
            )
            names = []
            for n, member in enumerate(members, start=1):
                number = drawing.member_number(word, member.part, n)
                own = member.part.startswith(word + " ")  # its name is the kind and a number
                names.append(number if own else f"{number} ({member.part})")
            text = ", ".join(names)
            loose = [n for n, m in zip(names, members, strict=True) if not m.placed]
            if loose and len(loose) == len(members):
                text += f" ({_NOT_PLACED})"
            elif loose:
                text += f" ({', '.join(loose)} {_NOT_PLACED})"
            lines = _prose(text, width - PART_LABEL_WIDTH - GAP, SMALL_SIZE)
            for n, line in enumerate(lines):
                self.text(x + GAP, self.y + n * SMALL_LINE, line, size=SMALL_SIZE, h=SMALL_LINE)
            self.y += max(1, len(lines)) * SMALL_LINE + ROW_GAP
        sources = list(dict.fromkeys(_source(p) for p in picture.parts if _source(p)))
        for source in sources[:PICTURE_SOURCES]:
            for line in _prose(f"source {source}", self.right - self.x0, SMALL_SIZE):
                self.text(self.x0, self.y, line, size=SMALL_SIZE, h=SMALL_LINE, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        if len(sources) > PICTURE_SOURCES:
            more = _MORE_ROWS.format(len(sources) - PICTURE_SOURCES)
            self.text(self.x0, self.y, more, size=SMALL_SIZE, h=SMALL_LINE, fg=MUTED_TEXT)
            self.y += SMALL_LINE
        self.y += ROW_GAP

    def probe(self, probe: Probe) -> None:
        """A pickup's cross-section, looking downstream, to scale: the bore, each button
        at its angle and size, lettered in a column on its side with its signal's
        bar and value on a leader (no two cross, :func:`picture_labels`); the beam
        dot in a magnified box (:meth:`beam`); a line of readouts under it; the
        plots at right. A button whose placement is inferred is drawn muted, and a
        note says so. Never turned by a roll: roll is a calibration value."""
        parts = {p.element.shape: p for p in probe.parts}
        bore = parts.get("bore")
        top = self.y
        canvas = PROBE_BORE + 2 * PROBE_PAD
        left_col = MARGIN
        x0 = left_col + PROBE_LABEL_WIDTH + PICTURE_PAD
        right_col = x0 + canvas + PICTURE_PAD
        cx, cy = x0 + canvas / 2, top + canvas / 2
        reach = bore.element.radius_mm if bore is not None else None
        anchors = [p for p in probe.parts if p.element.shape == "anchor"]
        reach = reach or max((p.element.radius_mm or 0 for p in anchors), default=0) or 1
        scale = PROBE_BORE / (2 * reach)
        box: float = PROBE_BOX
        if bore is not None:
            row = bore.element
            w = 2 * (row.radius_mm or 0) * scale
            h = row.size_mm * scale if row.size_mm is not None else w
            box = min(box, PROBE_BOX_FILL * h)
            props = {"lineColor": PAGE_TEXT, "lineWidth": LINE_WIDTH, "fill": False}
            props[_TOOLTIP] = row.element
            self.s.widget("ellipse", cx - w / 2, cy - h / 2, w, h, props)
        placed = []
        for part in anchors:
            row = part.element
            turn = math.radians(row.angle_deg or 0)
            px = cx + (row.radius_mm or 0) * scale * math.cos(turn)
            py = cy - (row.radius_mm or 0) * scale * math.sin(turn)
            size = (row.size_mm or 0) * scale
            muted = row.state == _INFERRED
            props = {
                "lineColor": MUTED_TEXT if muted else PAGE_TEXT,
                "lineWidth": LINE_WIDTH,
                "lineStyle": "dashed" if muted else "solid",
                "fill": True,
                "fillColor": FIELD_BACKGROUND,
                _TOOLTIP: row.element,
            }
            self.s.widget("ellipse", px - size / 2, py - size / 2, size, size, props)
            placed.append((row.element, px, py))
        ends = (left_col + PROBE_LABEL_WIDTH, right_col)
        labels = picture_labels(placed, cx, top, top + canvas, ends)
        for part, (_name, px, py), (side, label_y) in zip(anchors, placed, labels, strict=True):
            self.anchor_label(part, side, ends, label_y, (px, py))
        marker = parts.get("marker")
        if marker is not None:
            self.beam(marker, cx, cy, box)
        bottom = max([top + canvas, *(y + ROW for _side, y in labels)])
        plots: dict[str, list[ProbePart]] = {}
        for part in probe.parts:
            if part.element.shape == "plot":
                plots.setdefault(part.element.element, []).append(part)
        if plots:
            px = right_col + PROBE_LABEL_WIDTH + PICTURE_PAD
            w = MARGIN + STRIP_WIDTH - px
            h = (canvas - GAP * (len(plots) - 1)) / len(plots)
            for n, (name, curves) in enumerate(plots.items()):
                self.probe_plot(px, top + n * (h + GAP), w, h, name, curves)
        self.y = bottom + GAP
        self.readouts(probe, marker, x0)
        if probe.inferred:
            self.small(x0, self.y, _PLACEMENT_INFERRED, fg=MUTED_TEXT)
            self.y += SMALL_LINE
        self.y += ROW_GAP

    def anchor_label(
        self,
        part: ProbePart,
        side: str,
        ends: tuple[float, float],
        y: float,
        at: tuple[float, float],
    ) -> None:
        """A button's letter, its signal's bar and value, on its side, on a leader that
        ends a gap short of them; the letter nearest the button, the value outermost."""
        row = part.element
        fg = MUTED_TEXT if row.state == _INFERRED else PAGE_TEXT
        bar_w, bar_h = PROBE_BAR
        pieces: list[tuple[str, float]] = []
        if row.label:
            pieces.append(("letter", MARK_WIDTH))
        if part.channel is not None:
            pieces += [("bar", bar_w), ("value", PROBE_VALUE_WIDTH)]
        if not pieces:
            return
        end = ends[0] if side == "left" else ends[1]
        self.line([at, (end, y + ROW / 2)], MUTED_TEXT)
        if side == "left":
            pieces.reverse()
        width = sum(w for _name, w in pieces) + GAP * len(pieces)
        x = end - width if side == "left" else end + GAP
        for name, w in pieces:
            if name == "letter":
                self.text(x, y, row.label or "", w=w, fg=fg, align="center")
            elif name == "bar":
                props = {"pv": part.channel, "min": row.min, "max": row.max}
                props["orientation"] = "horizontal"
                props = {k: v for k, v in props.items() if v is not None}
                self.s.widget("pv-progress-bar", x, y + (ROW - bar_h) / 2, w, bar_h, props)
            else:
                self.field(x, y, part.channel, drawn_as="value", w=w)  # type: ignore[arg-type]
            x += w + GAP

    def beam(self, marker: ProbePart, cx: float, cy: float, box: float) -> None:
        """The magnified box of ± ``range_mm`` at the centre, and the beam dot in it:
        its x, y, width and height each a screen formula over both position
        channels, the place clamped to the box. Its own size is 0, so while
        either channel gives no value the dot is not drawn at all."""
        row = marker.element
        reach = row.range_mm or 1
        half = box / 2
        self.box(
            cx - half,
            cy - half,
            box,
            box,
            line=MUTED_TEXT,
            fill=TRANSPARENT,
            dashed=True,
        )
        label = f"±{reach:g} mm"
        w = text_width(label, SMALL_SIZE)
        self.small(cx + half - w, cy - half - SMALL_LINE, label, fg=MUTED_TEXT)
        if marker.channel is None or marker.channel_y is None:
            return
        k = half / reach
        bindings = {"x": marker.channel, "y": marker.channel_y}
        both = "0 * {x} + 0 * {y}"

        def clamp(token: str) -> str:  # min and max over one tuple: the engine's reductions
            low = f"max(({{{token}}}, {_number(-reach)},))"
            return f"min(({low}, {_number(reach)},))"

        place = {
            "x": f"{_number(cx - PROBE_DOT / 2)} + {_number(k)} * {clamp('x')} + 0 * {{y}}",
            "y": f"{_number(cy - PROBE_DOT / 2)} - {_number(k)} * {clamp('y')} + 0 * {{x}}",
            "width": f"{PROBE_DOT} + {both}",
            "height": f"{PROBE_DOT} + {both}",
        }
        if not self.budget.take([(len(place), frozenset(bindings.values()))]):
            return
        rules = []
        for prop, expression in place.items():
            name = f"beam-{prop}-{len(self.s.formulas)}"
            self.s.formulas.append(
                {"name": name, "kind": "scalar", "expression": expression, "bindings": bindings}
            )
            self.s._rules += 1
            rules.append(
                {
                    "id": f"r-{self.s._rules:04d}",
                    "name": f"beam {prop}",
                    "targetProperty": prop,
                    "default": 0,
                    "formula": _FORMULA_PREFIX + name,
                }
            )
        props = {"lineColor": PAGE_TEXT, "lineWidth": LINE_WIDTH, "fill": True}
        props |= {"fillColor": PAGE_TEXT, _TOOLTIP: row.element}
        self.s.widget("ellipse", cx, cy, 0, 0, props, rules=rules)

    def readouts(self, probe: Probe, marker: ProbePart | None, x: float) -> None:
        """One line under the drawing: the beam's x and y, then each readout row's value."""
        values: list[tuple[str, str, str | None]] = []
        if marker is not None:
            for name, channel in (("x", marker.channel), ("y", marker.channel_y)):
                if channel is not None:
                    values.append((name, channel, self.graph.channel(channel).unit))
        for part in probe.parts:
            if part.element.shape == "readout" and part.channel is not None:
                unit = self.graph.channel(part.channel).unit
                values.append((part.element.label or "", part.channel, unit))
        if not values:
            return
        for name, channel, unit in values:  # the label prints the unit: none beside it
            x = self.text(x, self.y, name) + GAP / 2
            w = max(PROBE_READOUT_WIDTH, value_width(unit, TEXT_SIZE))
            x = self.field(x, self.y, channel, drawn_as="value", w=w, units=True) + 2 * GAP
        self.y += ROW + ROW_GAP

    def probe_plot(
        self, x: float, y: float, w: float, h: float, name: str, curves: list[ProbePart]
    ) -> None:
        """One plot of a probe picture: a curve per row, named by its label; the title
        says how many points the data states one record holds."""
        counts = {p.element.count for p in curves if p.element.count is not None}
        title = name if len(counts) != 1 else f"{name} · {next(iter(counts)):,.0f} points"
        lines = [
            {
                "name": part.element.label or part.channel,
                "y_channel": part.channel,
                "color": symbols.CURVES[n % len(symbols.CURVES)],
            }
            for n, part in enumerate(p for p in curves if p.channel is not None)
        ]
        if lines:
            self.plot(x, y, w, h, title, lines)

    def plot(self, x: float, y: float, w: float, h: float, title: str, curves: list) -> None:
        """A waveform plot under its title. The title is a label of the page's own,
        not the plot's: the viewer draws a plot's title in a weight the page's font
        set has no face for."""
        self.text(x, y, title, size=SMALL_SIZE, w=w, h=SMALL_LINE, fg=MUTED_TEXT)
        self.s.widget("waveform-plot", x, y + SMALL_LINE, w, h - SMALL_LINE, {"curves": curves})

    def elevation(self, elevation: Elevation) -> None:
        """A rack as it stands: its frame, each chassis a bar at its true unit spanning
        the rack, the unit numbers left, the name inside the bar and one light over
        the unit's channels at its right end, opening the card. Right of it the
        classes counted and the sources; under it the devices not at a unit."""
        tops = [_unit_top(unit) for unit in elevation.units]
        height = max(tops, default=0)
        # a unit numbered below 1 (a rack's floor, U0) stands inside the frame too
        lowest = min([1, *(unit.position or 0 for unit in elevation.units)])
        rows = height - lowest + 1
        frame_x = MARGIN + RACK_LABEL_WIDTH
        bars_x = frame_x + RACK_PAD
        top = self.y
        rack_top = top + RACK_PAD
        self.s.widget(
            "rectangle",
            frame_x,
            top,
            RACK_WIDTH + 2 * RACK_PAD,
            rows * RACK_U + 2 * RACK_PAD,
            {"lineColor": PAGE_TEXT, "lineWidth": AXIS_WIDTH, "fill": False},
        )
        labelled: set[int | None] = set()
        lanes = rack_lanes(elevation.units)
        # chassis too many side by side for their names: one bar over their units,
        # and each in a list under the rack with its own light
        crowded: list[RackUnit] = []
        spans: list[list[int]] = []  # the crowded bars' units, lowest and highest
        # the units numbered by where they stand: top down, then left to right
        rank = sorted(range(len(tops)), key=lambda k: (-tops[k], lanes[k][0]))
        numbered = {k: n for n, k in enumerate(rank)}
        for k, (unit, unit_top, (lane, count)) in enumerate(
            zip(elevation.units, tops, lanes, strict=True)
        ):
            order = numbered[k]
            w = RACK_WIDTH / count  # chassis whose units overlap share the rack's width
            x = bars_x + lane * w
            y = rack_top + (height - unit_top) * RACK_U
            h = (unit.size_u or 1) * RACK_U
            if unit.position not in labelled:  # at the unit stated: the bar's lowest
                labelled.add(unit.position)
                label = f"U{unit.position}"
                self.text(
                    MARGIN, y + h - RACK_U, label, size=SMALL_SIZE, w=RACK_LABEL_WIDTH - GAP,
                    h=RACK_U,
                )  # fmt: skip
            if w < RACK_LANE_LEAST:
                crowded.append(unit)
                continue
            with self.s.item("rack-unit", _unit_thing(unit), order):
                self.chassis(x, y, w, h, unit)
        crowded.sort(key=lambda u: u.position or 0)
        for unit in crowded:
            low, high = unit.position or 0, _unit_top(unit)
            if spans and low <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], high)
                spans[-1][2] += 1
            else:
                spans.append([low, high, 1])
        for low, high, n in spans:
            y = rack_top + (height - high) * RACK_U
            h = (high - low + 1) * RACK_U
            self.box(bars_x, y, RACK_WIDTH, h, fill=FIELD_BACKGROUND, dashed=True)
            word = _cut(_SIDE_BY_SIDE_BAR.format(n), RACK_WIDTH - GAP, SMALL_SIZE)
            self.text(bars_x + GAP / 2, y, word, size=SMALL_SIZE, w=RACK_WIDTH - GAP, h=RACK_U, fg=MUTED_TEXT)  # fmt: skip
        bottom = top + rows * RACK_U + 2 * RACK_PAD
        self.rack_facts(elevation, top)
        self.y = max(bottom, self.facts_bottom) + SECTION_GAP
        if crowded:
            self.not_at_a_unit(crowded, heading=_SIDE_BY_SIDE)
        if elevation.unpositioned:
            self.not_at_a_unit(elevation.unpositioned)

    def chassis(self, x: float, y: float, w: float, h: float, unit: RackUnit) -> None:
        """One bar: its name inside, the light at its right end, a link over it."""
        self.box(x, y, w, h, fill=FIELD_BACKGROUND, dashed=unit.conflicting)
        light_x = x + w - CHIP_LIGHT - GAP / 2 - _mark_room(unit)
        light_y = y + (min(h, RACK_U) - ROW) / 2
        self.unit_light(light_x, light_y, unit)
        name = unit.link.label if isinstance(unit.link, Link) else unit.link.text
        if unit.standing == _CONTESTED:
            name = f"{name} · {_CONTESTED}"
        x_text = x + GAP / 2
        if unit.conflicting:  # the mark, then air before the name
            x_text = self.text(x_text, y, symbols.CONFLICT_MARK, size=SMALL_SIZE, h=RACK_U) + GAP
        room = light_x - GAP / 2 - x_text
        name = _cut(name, room, SMALL_SIZE)
        if name:
            w_name = min(text_width(name, SMALL_SIZE), room)
            self.text(x_text, y, name, size=SMALL_SIZE, w=w_name, h=RACK_U)
        if isinstance(unit.link, Link):
            self.nav(x, y, w, h, unit.link)

    def unit_light(self, x: float, y: float, unit: RackUnit) -> None:
        """One light over a unit's channels (and its parts'), the no-rule look with none;
        an inferred mark after it takes :func:`_mark_room`."""
        if not unit.lights:
            self.no_rule(x, y, CHIP_LIGHT)
        else:
            self.light(x, y, unit.lights, size=CHIP_LIGHT)

    def rack_facts(self, elevation: Elevation, top: float) -> None:
        """Right of the rack: the classes of its devices counted, how many stand at a
        unit, and the documents the units come from."""
        x0, right = self.x0, self.right
        self.x0, self.right = MARGIN + RACK_COLUMN, MARGIN + STRIP_WIDTH
        self.y = top
        every = [*elevation.units, *elevation.unpositioned]
        classes: Counter[str] = Counter()
        for device in dict.fromkeys(u.link.target.id for u in every if isinstance(u.link, Link)):
            try:
                cls = self.graph.thing(device, "device").cls
            except KeyError:
                cls = None
            classes[cls or _NO_CLASS] += 1
        if classes:
            self.heading(_BY_CLASS)
            for cls, count in sorted(classes.items(), key=lambda kv: (-kv[1], kv[0])):
                self.text(self.x0, self.y, cls, size=SMALL_SIZE, h=SMALL_LINE)
                n = str(count)
                w = text_width(n, SMALL_SIZE)
                self.text(self.right - w, self.y, n, size=SMALL_SIZE, w=w, h=SMALL_LINE)
                self.y += SMALL_LINE
            self.y += ROW_GAP
        counts = _RACK_COUNTS.format(len(elevation.units), len(elevation.unpositioned))
        self.text(self.x0, self.y, counts, size=SMALL_SIZE, fg=MUTED_TEXT)
        self.next_row()
        documents: Counter[str] = Counter()
        for unit in every:  # a document per unit once, however many of its rows say it
            documents.update(
                list(
                    dict.fromkeys(
                        " ".join(p for p in (c.document, c.version) if p) for c in unit.citations
                    )
                )
            )
        if documents:
            self.heading(_SOURCES)
            ranked = sorted(documents.items(), key=lambda kv: (-kv[1], kv[0]))
            for document, count in ranked[:RACK_SOURCES]:
                text = _cut(_UNITS_FROM.format(document, count), self.right - self.x0, SMALL_SIZE)
                self.text(self.x0, self.y, text, size=SMALL_SIZE, h=SMALL_LINE, fg=MUTED_TEXT)
                self.y += SMALL_LINE
            if len(ranked) > RACK_SOURCES:
                more = _MORE_ROWS.format(len(ranked) - RACK_SOURCES)
                self.text(self.x0, self.y, more, size=SMALL_SIZE, h=SMALL_LINE, fg=MUTED_TEXT)
                self.y += SMALL_LINE
        self.facts_bottom = self.y
        self.x0, self.right = x0, right

    def not_at_a_unit(self, entries: Sequence[RackUnit], heading: str = _NOT_AT_A_UNIT) -> None:
        """The devices in the rack with no unit (or, under another *heading*, those too
        many side by side for their names): a framed list of chips, each with its
        light and the unit a claim gives, under the rack."""
        first = len(self.s.nodes)
        top = self.y
        self.y += PANEL_PAD
        self.heading(heading)
        x: float = MARGIN
        for unit in entries:
            name = unit.link.label if isinstance(unit.link, Link) else unit.link.text
            label = f"{name} · unit {unit.unit_text}" if unit.unit_text else name
            if unit.position is not None:
                label = f"{name} · U{unit.position}"
            mark = _mark_room(unit)
            w = text_width(label, SMALL_SIZE) + CHIP_LIGHT + 2 * GAP + mark
            if x > MARGIN and x + w > MARGIN + STRIP_WIDTH:
                x = MARGIN
                self.y += CHIP_HEIGHT + CHIP_GAP
            self.box(x, self.y, w, CHIP_HEIGHT, fill=FIELD_BACKGROUND)
            self.unit_light(x + GAP / 2, self.y + (CHIP_HEIGHT - ROW) / 2, unit)
            self.text(x + CHIP_LIGHT + GAP + mark, self.y, label, size=SMALL_SIZE, h=CHIP_HEIGHT)
            if isinstance(unit.link, Link):
                self.nav(x, self.y, w, CHIP_HEIGHT, unit.link)
            x += w + CHIP_GAP
        self.y += CHIP_HEIGHT + PANEL_PAD
        self.s.widget(
            "rectangle",
            MARGIN - PANEL_PAD / 2,
            top,
            STRIP_WIDTH + PANEL_PAD,
            self.y - top,
            {"lineColor": BOX_BORDER, "lineWidth": LINE_WIDTH, "fill": False},
        )
        self.s.nodes.insert(first, self.s.nodes.pop())
        self.y += GAP


def rack_lanes(units: Sequence[RackUnit]) -> list[tuple[int, int]]:
    """Each unit's lane and the lanes of its cluster: units whose spans overlap form a
    cluster, each takes the first lane free over its span, in order. A half-width
    unit (its ``side`` stated) takes the left or right of two lanes, so two halves
    of one unit stand side by side and one alone fills its own half."""
    spans = [(unit.position or 0, _unit_top(unit)) for unit in units]
    order = sorted(range(len(units)), key=lambda i: (spans[i][0], i))
    lanes = [0] * len(units)
    cluster = [0] * len(units)
    clusters: list[list[int]] = []
    reach = -math.inf
    for i in order:
        low, high = spans[i]
        if low > reach:
            clusters.append([])
        reach = max(reach, high) if low <= reach else high
        members = clusters[-1]
        taken = {lanes[j] for j in members if spans[j][1] >= low}
        wanted = _SIDE_LANES.get(units[i].side or "")
        if wanted is not None and wanted not in taken:
            lanes[i] = wanted
        else:
            lanes[i] = next(n for n in itertools.count() if n not in taken)
        members.append(i)
        cluster[i] = len(clusters) - 1
    width = [
        max(
            max(lanes[j] for j in members) + 1,
            len(_SIDE_LANES) if any(units[j].side for j in members) else 1,
        )
        for members in clusters
    ]
    return [(lanes[i], width[cluster[i]]) for i in range(len(units))]


def _ring_lamps(members: Sequence[RingMember]) -> list[str]:
    """The channels of the members' lamps: the first row of each, when a lamp."""
    return [r.channel for m in members for r in m.rows[:1] if r.shown_as == "lamp" and r.channel]


#: The lane of a half-width unit's stated side, of two.
_SIDE_LANES: Mapping[str, int] = MappingProxyType({"L": 0, "R": 1})


def _unit_top(unit: RackUnit) -> int:
    """The highest unit a chassis takes: its stated unit is its lowest."""
    return (unit.position or 0) + int(unit.size_u or 1) - 1


def picture_labels(
    parts: Sequence[tuple[str, float, float]],
    centre: float,
    top: float,
    bottom: float,
    anchors: tuple[float, float] | None = None,
    heights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Where each part's name goes: ``(side, y)`` per ``(name, x, y)`` part, in order.

    A part left of *centre* is named in the left column, any other in the
    right one. In each column the names go in order of their part's ``y``,
    each at its part's height or just under the name above it, so no two
    overlap; a column that runs past *bottom* is pushed back up, never
    above *top*. With *anchors* (the x where the left and the right
    column's leaders end), two leaders of a column that cross swap their
    names' places until none cross: each swap shortens the leaders, so it ends.
    With *heights* (a slot's height per part, :data:`ROW` when not given), a
    slot is that tall; after the swaps the column is spaced again, so that no
    two slots overlap.
    """
    tall = list(heights) if heights is not None else [ROW] * len(parts)
    slots: list[tuple[str, float]] = [("", 0.0)] * len(parts)
    for side in ("left", "right"):
        mine = [i for i, (_n, x, _y) in enumerate(parts) if (x < centre) == (side == "left")]
        mine.sort(key=lambda i: (parts[i][2], i))
        ys: list[float] = []
        for k, i in enumerate(mine):
            want = parts[i][2] - ROW / 2
            ys.append(max(want, ys[-1] + tall[mine[k - 1]] + ROW_GAP) if ys else want)
        for k in range(len(ys) - 1, -1, -1):  # push back up from the bottom
            own = tall[mine[k]]
            limit = bottom - own if k == len(ys) - 1 else ys[k + 1] - own - ROW_GAP
            ys[k] = min(ys[k], limit)
        if ys and ys[0] < top:  # a column taller than the picture runs on downward
            ys[0] = top
            for k in range(1, len(ys)):
                ys[k] = max(ys[k], ys[k - 1] + tall[mine[k - 1]] + ROW_GAP)
        if anchors is not None:
            ax = anchors[0] if side == "left" else anchors[1]
            _uncross(mine, ys, parts, ax)
        if heights is not None:  # a swap may have put a tall slot where a short one was
            order = sorted(range(len(mine)), key=lambda k: ys[k])
            for n in range(1, len(order)):
                above, here = order[n - 1], order[n]
                ys[here] = max(ys[here], ys[above] + tall[mine[above]] + ROW_GAP)
        for i, y in zip(mine, ys, strict=True):
            slots[i] = (side, y)
    return slots


def _uncross(
    mine: list[int], ys: list[float], parts: Sequence[tuple[str, float, float]], ax: float
) -> None:
    """Swap the label heights *ys* of the parts *mine* until no two leaders from a part to
    (*ax*, its label's middle) cross."""

    def leader(k: int) -> tuple[tuple[float, float], tuple[float, float]]:
        _n, px, py = parts[mine[k]]
        return (px, py), (ax, ys[k] + ROW / 2)

    for _round in range(len(mine) ** 2 + 1):
        swapped = False
        for a, b in itertools.combinations(range(len(mine)), 2):
            if _crossing(*leader(a), *leader(b)):
                ys[a], ys[b] = ys[b], ys[a]
                swapped = True
        if not swapped:
            return


def _crossing(p1, p2, q1, q2) -> bool:
    """Whether the segments p1-p2 and q1-q2 cross at a point inside both."""

    def turn(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = turn(q1, q2, p1), turn(q1, q2, p2)
    d3, d4 = turn(p1, p2, q1), turn(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _water(kind: str) -> bool:
    """Whether a kind of part is drawn as water: one :data:`WATER_WORDS` names."""
    return any(word in kind for word in WATER_WORDS)


def _groups(
    spots: Sequence[tuple[float, float]], distance: float
) -> list[list[tuple[float, float]]]:
    """*spots* in groups: two within *distance* of each other (in turn) are in one."""
    groups: list[list[tuple[float, float]]] = []
    for spot in spots:
        near = [g for g in groups if any(math.dist(spot, other) <= distance for other in g)]
        merged = [spot, *(other for g in near for other in g)]
        groups = [g for g in groups if g not in near] + [merged]
    return groups


def _reach(shape: drawing.Shape) -> list[tuple[float, float]]:
    """The points of a body's shape that bound it: a polygon's corners, an ellipse's
    four extremes."""
    if shape.form != "ellipse":
        return list(shape.points)
    (a1, b1), (a2, b2) = shape.points
    return [(a1, (b1 + b2) / 2), (a2, (b1 + b2) / 2), ((a1 + a2) / 2, b1), ((a1 + a2) / 2, b2)]


def _source(part: PicturePart) -> str:
    """Where a pictures row is stated, its locator left out: document, version, state."""
    return " ".join(t for t in (part.document, part.version, part.state) if t)


def _picture_xy(view: str, part) -> tuple[float, float]:
    """A row's drawn coordinates, in mm: x, y on an end view; z, y on a side view;
    the beam axis where the row states none."""
    across = part.z_mm if view == "side" else part.x_mm
    return (across or 0.0, part.y_mm or 0.0)


def _ticks(lo: float, hi: float) -> list[tuple[float, float]]:
    """The metre ticks of a locator of ``lo`` to ``hi`` metres: (metres, x)."""
    span = hi - lo
    if span <= 0:
        return [(lo, STRIP_WIDTH / 2)]
    exponent = math.floor(math.log10(span / MAX_TICKS)) if span > MAX_TICKS else 0
    step = next(
        base * TICK_BASE**e
        for e in range(exponent, exponent + 2)
        for base in TICK_STEPS
        if span / (base * TICK_BASE**e) <= MAX_TICKS
    )
    first = math.ceil(lo / step)
    out = []
    k = first
    while k * step <= hi:
        metres = k * step
        out.append((metres, (metres - lo) / span * STRIP_WIDTH))
        k += 1
    return out


def _short(thing: str) -> str:
    """A symbol's name: the last component of its id."""
    return id_tail(thing)


def _column_head(groups: Sequence[tuple[str, Sequence[GridRow]]], column: int) -> str:
    """A set grid's column head: the short name every member in the column states
    when they all state one and the same, else the column's number."""
    names = {
        row.cells[column].short_name
        for _title, rows in groups
        for row in rows
        if column < len(row.cells)
    }
    if len(names) == 1 and None not in names:
        return str(next(iter(names)))
    return f"#{column + 1}"


def arrow_lines(word: str) -> tuple[list[str], bool]:
    """An arrow's word in the run of :data:`ARROW_RUN_MOST`: whole on one line, else
    broken at spaces over two (over and under the arrow), else its last name segment
    (after the last ``:``); and whether it was shortened (then the whole word is on
    hover)."""
    room = ARROW_RUN_MOST - GAP
    if text_width(word, SMALL_SIZE) <= room:
        return [word], False
    lines = _wrap(word, room, SMALL_SIZE, TITLE_LINES)
    if " ".join(lines) == " ".join(word.split()):
        return lines, False
    last = word.rsplit(":", 1)[-1].strip() or word
    if text_width(last, SMALL_SIZE) <= room:
        return [last], True
    return _wrap(last, room, SMALL_SIZE, TITLE_LINES), True


def _arrow_need(lines: list[str]) -> float:
    """The run an arrow needs for its word's lines."""
    widest = max((text_width(line, SMALL_SIZE) for line in lines), default=0.0)
    return min(ARROW_RUN_MOST, max(BLOCK_GAP_LEAST, widest + GAP))


def _title_lines(title: str, width: float) -> int:
    return len(_wrap(title, width, BLOCK_TEXT_SIZE, TITLE_LINES))


def meaning_stated(graph: PanelGraph, channel: str) -> bool:
    """Whether a light for *channel* may take a colour: a health rule holds for it
    and what its values mean is stated (state words for a state, the rule's own
    thresholds for a number). Every light of a card (a tile, a chain block, the
    grid) asks this one question; with no, the light is the neutral outline."""
    if not graph.rule(channel):
        return False
    if graph.state_words(channel):
        return True
    try:
        drawn_as = graph.channel(channel).drawn_as
    except KeyError:
        return False
    return drawn_as in ("value", "setpoint")


def _mark_room(unit: RackUnit) -> float:
    """The room the inferred mark after a unit's light takes: none when no light is inferred."""
    return MARK_WIDTH + GAP if any(item.inferred for item in unit.lights) else 0.0


def _block_label(block: ChainBlock) -> float:
    """The label column of a block's lines: its longest label, and the gap before it."""
    widths = [text_width(row.label, SMALL_SIZE) for row in block.rows]
    return max([BLOCK_LABEL_LEAST, *widths]) + GAP


#: The view items a widget names in its ``meta`` comment, each with the way the
#: items read on a screen in the order the view states for them: ``across``, left
#: to right and on in rows top to bottom; ``down``, top to bottom (side by side
#: left to right) and on in columns left to right.
ITEM_TAGS: Mapping[str, str] = MappingProxyType(
    {
        "rack-unit": "down",
        "strip-element": "across",
        "key-tile": "across",
        "chain-block": "across",
        "grid-row": "down",
    }
)


def item_tags(kind: str, thing: str | None = None, order: int | None = None) -> tuple[str, ...]:
    """The tags that name one view item: ``item: <kind>``, then ``thing: <id>``
    and ``order: <n>`` where they are given."""
    if kind not in ITEM_TAGS:
        raise ValueError(f"no item kind {kind!r}")
    tags = [f"item: {kind}"]
    if thing is not None:
        tags.append(f"thing: {thing}")
    if order is not None:
        tags.append(f"order: {order}")
    return tuple(tags)


def _unit_thing(unit: RackUnit) -> str | None:
    """The thing a rack unit stands for: its link's screen's thing, else none."""
    return unit.link.target.id if isinstance(unit.link, Link) else None


def _hover_box(g: Mapping[str, float], text: str) -> dict[str, Any]:
    """A transparent box over *g* that shows *text* while the pointer is on it."""
    props = {"lineColor": TRANSPARENT, "lineWidth": 0, "fill": False, _TOOLTIP: text}
    return {"type": "rectangle", "props": {**props, _HOVER_LABEL: text}, "geometry": dict(g)}


def _fit(text: str, w: float, size: float) -> str:
    """*text* as a label *w* wide shows it: whole when its letters and the viewer's
    inset fit, else cut (:func:`_cut`)."""
    if _em(text) * size / metrics.PER_EM + 2 * LABEL_INSET <= w:
        return text
    return _cut(text, w + TEXT_PAD - 2 * LABEL_INSET, size)


class _Cut(str):
    """A text cut to fit, which remembers the whole: a label drawn from it gets a
    box that shows the whole text on hover (:meth:`_Screen.add`)."""

    whole: str

    def __new__(cls, kept: str, whole: str) -> _Cut:
        made = super().__new__(cls, kept)
        made.whole = whole
        return made


def _cut(
    text: str, width: float, size: float, *, caps: bool = False, whole: str | None = None
) -> str:
    """*text* cut to fit *width* at *size* by :func:`text_width`, an ellipsis marking
    the cut; *caps*: written in capitals; *whole*: the text it is a part of, shown
    on hover (else *text*)."""
    if text_width(text, size, caps=caps) <= width:
        return text
    cuts = _CUTS.get()
    if cuts is not None:
        cuts.append(text)
    whole = whole or str(getattr(text, "whole", text))
    for end in range(len(text) - 1, 0, -1):
        kept = text[:end].rstrip() + _ELLIPSIS
        if text_width(kept, size, caps=caps) <= width:
            return _Cut(kept, whole)
    return ""


def _cut_apart(text: str, siblings: Sequence[str], width: float, size: float) -> str:
    """*text*, one of *siblings*, cut to fit *width* at *size* so the part that
    tells it from the others survives: whole while it fits; else the words the
    siblings share at the front go first, word by word from their end
    ("front…differing end"),
    then the shared words at the end ("…differing…"), and only then the
    differing part itself, from its front; with nothing shared it is cut in the
    middle. The shared parts end and start at a
    separator (any character not a letter or digit), so a cut never splits a
    word the siblings both use. The whole shows on hover (:class:`_Cut`)."""
    if text_width(text, size) <= width:
        return text
    cuts = _CUTS.get()
    if cuts is not None:
        cuts.append(text)
    others = [s for s in siblings if s != text] or [text]
    front = min(len(os.path.commonprefix([text, s])) for s in others)
    back = min(len(os.path.commonprefix([text[::-1], s[::-1]])) for s in others)
    while front and text[front - 1].isalnum():
        front -= 1
    back = min(back, len(text) - front)
    while back and text[len(text) - back].isalnum():
        back -= 1
    head, differing, tail = text[:front], text[front : len(text) - back], text[len(text) - back :]
    lead = _ELLIPSIS if head else ""
    words = [k for k in range(len(head) - 1, 0, -1) if not head[k - 1].isalnum()]
    tries = [head[:k] + _ELLIPSIS + differing + tail for k in [*words, 0] if head]
    tries += [lead + differing + tail[:k] + _ELLIPSIS for k in range(len(tail) - 1, -1, -1)]
    if head or tail:  # the differing part alone, from its front
        tries += [
            lead + differing[k:] + (_ELLIPSIS if tail else "") for k in range(1, len(differing))
        ]
    else:  # nothing shared: cut in the middle, the front and the end kept
        tries += [
            text[: (len(text) - k + 1) // 2] + _ELLIPSIS + text[len(text) - (len(text) - k) // 2 :]
            for k in range(1, len(text))
        ]
    for kept in tries:
        if text_width(kept, size) <= width:
            return _Cut(kept, text)
    return ""


def _chip(title: str, item: object) -> tuple[str, Link | None]:
    """A card's chip for a related thing: "<relation>: <thing>", its card when it has one."""
    if isinstance(item, Link):
        return f"{title}: {item.label}", item
    if isinstance(item, Placement):
        return f"{title}: {item.link.label}", item.link
    if isinstance(item, Fact):
        return f"{title}: {item.text}", None
    return title, None


def caps_width(text: str, size: float) -> float:
    """:func:`text_width` of *text* written in capitals."""
    return text_width(text, size, caps=True)


def _citation_line(citation: Citation) -> str:
    return " ".join(
        p for p in (citation.document, citation.version, citation.state, citation.locator) if p
    )


def _standing_word(item: object) -> str | None:
    """``contested`` or ``inferred`` for a claim that is, else ``None``."""
    if getattr(item, "standing", None) == _CONTESTED:
        return _CONTESTED
    if getattr(item, "basis", None) == "inferred":
        return "inferred"
    return None


def _bands(columns: Sequence[tuple[float, Connector]]) -> list[list[tuple[float, BlockRow]]]:
    """The value bands under a line of connectors: by place in each connector's
    rows when every connector with rows has as many and the labels at each place
    end in the same words (each connector's roles in the same order), else by the
    rows' label."""
    counts = {len(c.rows) for _x, c in columns if c.rows}
    if len(counts) == 1:
        (n,) = counts
        placed = [[(x, c.rows[k]) for x, c in columns if c.rows] for k in range(n)]
        if all(_alike([row.label for _x, row in cells]) for cells in placed):
            return placed
    by_label: dict[str, list[tuple[float, BlockRow]]] = {}
    for x, connector in columns:
        for row in connector.rows:
            by_label.setdefault(row.label, []).append((x, row))
    return list(by_label.values())


def _alike(labels: Sequence[str]) -> bool:
    """Whether *labels* end in the same word."""
    return len({label.rsplit(" ", 1)[-1] for label in labels}) == 1


def _band_label(labels: Sequence[str]) -> str:
    """A band's label: the words its rows' labels end in alike ("voltage" for
    "phase voltage" and "circuit voltage"); with none alike, the labels joined."""
    split = [label.split() for label in dict.fromkeys(labels)]
    common: list[str] = []
    for words in zip(*(reversed(w) for w in split), strict=False):
        if len(set(words)) != 1:
            break
        common.insert(0, words[0])
    if common or len(split) == 1:
        return " ".join(common) or labels[0]
    return " / ".join(" ".join(w) for w in split)


def _reading(group: FrontGroup) -> bool:
    """Whether a front group is readings: connectors, each with no far side."""
    return bool(group.connectors) and all(not c.sides for c in group.connectors)


def schematic_note(parts: Sequence[PicturePart], note: str | None) -> str | None:
    """A picture's caption *note*, with "schematic, not to scale" after it when a
    row of the picture states itself ``schematic`` (and the note does not say so)."""
    if not any(p.state == "schematic" for p in parts) or (note and _NOT_TO_SCALE in note):
        return note
    return f"{note} · {_NOT_TO_SCALE}" if note else _NOT_TO_SCALE


def scale_note(body: drawing.Body) -> str | None:
    """What the caption under a drawing's scale bar says: nothing when every shape
    states its size; the default proportions when no row states one; else the kind
    of row the drawing is scaled from, and the kinds drawn at the default proportion."""
    if body.scaled_from is None:
        return _SIZE_NOT_STATED
    if not body.schematic:
        return None
    kinds = ", ".join(kind.replace("_", " ") for kind in body.schematic)
    return f"{_SCALED_FROM.format(body.scaled_from)}; {_SCHEMATIC.format(kinds)}"


def _wrap(
    text: str, width: float, size: float, lines: int = TILE_LINES, caps: bool = False
) -> list[str]:
    """*text* in at most *lines* lines that each fit *width* at *size*, broken at
    spaces; the last line cut with an ellipsis when the rest does not fit. *caps*:
    measured as capitals (:func:`caps_width`)."""
    out: list[str] = []
    words = text.split()
    while words and len(out) < lines:
        line = words.pop(0)
        while words and text_width(f"{line} {words[0]}", size, caps=caps) <= width:
            line = f"{line} {words.pop(0)}"
        if len(out) == lines - 1 and words:
            line = _cut(" ".join([line, *words]), width, size, caps=caps, whole=text)
            words = []
        out.append(_cut(line, width, size, caps=caps))
    return out or [""]


def _prose(text: str, width: float, size: float) -> list[str]:
    """A muted line of prose (a caption, a note, what a press writes, a source) in
    as many lines as it needs, so none is cut: the page grows instead. It breaks
    after a space, and inside a long word after a slash, comma or semicolon; only
    a piece wider than the line by itself is cut."""
    pieces = re.findall(r"[^\s/,;]*[\s/,;]+|[^\s/,;]+$", text.strip())
    out: list[str] = []
    line = ""
    for piece in pieces:
        if line and text_width((line + piece).rstrip(), size) > width:
            out.append(_cut(line.rstrip(), width, size))
            line = ""
        line += piece
    if line.strip():
        out.append(_cut(line.rstrip(), width, size))
    return out or [""]


def _wrap_label(text: str, width: float, size: float, lines: int) -> list[str]:
    """A name in at most *lines* lines that each fit *width* at *size*, broken after a
    space, colon, hyphen or underscore; the last line cut with an ellipsis when the
    rest does not fit. A piece too wide for a line by itself breaks where a small
    letter meets a capital (``ElectronicsChassis``)."""
    pieces = [
        part
        for piece in re.findall(r"[^\s:_-]*[\s:_-]|[^\s:_-]+$", text)
        for part in (
            re.split(r"(?<=[a-z])(?=[A-Z])", piece) if text_width(piece, size) > width else [piece]
        )
    ]
    out: list[str] = []
    while pieces and len(out) < lines:
        line = pieces.pop(0)
        while pieces and text_width(line + pieces[0], size) <= width:
            line += pieces.pop(0)
        if len(out) == lines - 1 and pieces:
            line, pieces = line + "".join(pieces), []
        out.append(_cut(line.strip(), width, size, whole=text))
    return out or [""]


def _wrap_name(text: str, width: float, size: float, lines: int) -> list[str]:
    """A channel name in at most *lines* lines that each fit *width* at *size*, broken
    after a colon; the last line cut with an ellipsis when the rest does not fit."""
    pieces = re.findall(r"[^:]*:|[^:]+$", text)
    out: list[str] = []
    while pieces and len(out) < lines:
        line = pieces.pop(0)
        while pieces and text_width(line + pieces[0], size) <= width:
            line += pieces.pop(0)
        if len(out) == lines - 1 and pieces:
            line, pieces = line + "".join(pieces), []
        out.append(_cut(line, width, size, whole=text))
    return out or [""]


def _tile_width(per_row: int) -> int:
    """The width of each of *per_row* tiles spanning the content width: whole
    pixels, the rest given to the gaps (:func:`spread`)."""
    return (STRIP_WIDTH - (per_row - 1) * GAP) // per_row


def spread(count: int, left: float, width: float, gap: float) -> list[tuple[int, int]]:
    """``(x, width)`` of *count* boxes of one whole width filling *width* from
    *left*, at least *gap* apart: what whole widths leave over widens the first
    gaps by a pixel each, so the row ends exactly at its right edge."""
    if count <= 0:
        return []
    each = int((width - (count - 1) * gap) // count)
    rest = int(round(width - count * each - (count - 1) * gap))
    out, x = [], int(round(left))
    for i in range(count):
        out.append((x, each))
        x += each + int(gap) + (1 if i < rest else 0)
    return out


def strip_labels(placed: Sequence[Placed]) -> list[tuple[str, float, float, int]]:
    """Each symbol's name under it: ``(text, left, width, level)``, in beam order.

    The name is the last component of the id, cut to the symbol's budget
    (:data:`STRIP_LABEL_WIDTH`, or its width when wider) and centred on it. It
    goes on the first of two levels where it clears the last name there by
    :data:`STRIP_LABEL_GAP`; when neither is clear, on the level that ends
    first, cut to the room left of its centre's reach. A name with no room is
    left out (the symbol's box still shows the id). By :func:`text_width`,
    so no two names of one level overlap.
    """
    ends = [-math.inf, -math.inf]
    found: list[tuple[str, float, float, int]] = []
    for p in placed:
        name = p.short if p.short is not None else _short(p.thing)
        text = _cut(name, max(p.width, STRIP_LABEL_WIDTH), STRIP_LABEL_SIZE)
        w = text_width(text, STRIP_LABEL_SIZE)
        left = p.x - w / 2
        level = next((n for n in (0, 1) if left >= ends[n] + STRIP_LABEL_GAP), None)
        if level is None:
            level = 0 if ends[0] <= ends[1] else 1
            left = ends[level] + STRIP_LABEL_GAP
            text = _cut(name, p.x + w / 2 - left, STRIP_LABEL_SIZE)
            if not text:
                continue
            w = text_width(text, STRIP_LABEL_SIZE)
        ends[level] = left + w
        found.append((text, left, w, level))
    return found


def precision_expression(value: str, *, compact: bool = False) -> str:
    """The decimals that print *value* (a formula expression) with
    :data:`SIGNIFICANT_DIGITS` significant digits: that many decimals for zero, and
    ``-1`` below :data:`FIXED_SMALLEST`, which the viewer ignores, so such a value
    keeps the viewer's own format. *compact* names *value* once, for one too long to
    repeat: below :data:`COMPACT_SMALLEST` it prints as that does."""
    digits = SIGNIFICANT_DIGITS
    if compact:
        floor = _number(COMPACT_SMALLEST)
        return f"max((0, {digits - 1} - floor(log10(max((abs({value}), {floor},)))),))"
    return (
        f"{digits} if {value} == 0 else (-1 if abs({value}) < {_number(FIXED_SMALLEST)} "
        f"else max((0, {digits - 1} - floor(log10(abs({value}))),)))"
    )  # max over one tuple: the formula engine's max is a reduction


def shown_width(text: str, size: float) -> float:
    """The width of a value display that prints *text* at *size*: the characters,
    in the display's fixed-width face, and its own padding and frame."""
    return math.ceil(len(text) * CHAR_WIDTH * size + NUMBER_PAD)


def _significant(value: float) -> str:
    """*value* as a display prints it to :data:`SIGNIFICANT_DIGITS` significant digits."""
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    return f"{value:.{max(0, SIGNIFICANT_DIGITS - 1 - exponent)}f}"


def value_width(unit: str | None, size: float) -> float:
    """The width of a value display that prints :data:`VALUE_SAMPLE` and *unit* at
    *size*: the characters, and the display's own padding and frame."""
    text = f"{VALUE_SAMPLE} {unit}" if unit else VALUE_SAMPLE
    return math.ceil(len(text) * CHAR_WIDTH * size + NUMBER_PAD)


def value_size(unit: str | None, width: float) -> float:
    """The largest font size at which :func:`value_width` of *unit* fits *width*; never
    under :data:`SMALL_SIZE`, the floor of all text."""
    text = f"{VALUE_SAMPLE} {unit}" if unit else VALUE_SAMPLE
    return max(SMALL_SIZE, (width - NUMBER_PAD) / (len(text) * CHAR_WIDTH))


def not_shown(count: int, words: str) -> str:
    """*words* (:data:`_CROWDED` or :data:`_DROPPED`) for *count* values."""
    return words.format(f"{count} value" if count == 1 else f"{count} values")


def fixed_expression(value: str, decimals: int) -> str:
    """*value* (a formula expression) rounded to *decimals* decimals, as a number."""
    return f"floor({value} * pow(10, {decimals}) + 0.5) / pow(10, {decimals})"


def significant_expression(value: str) -> str:
    """*value* (a formula expression) rounded to :data:`SIGNIFICANT_DIGITS` significant
    digits, as a number: the formula engine has no ``round``, so ``floor(x + 0.5)``; a
    power of ten divides when the last digit is a decimal, so the number prints short."""
    exponent = f"(floor(log10(abs({value}))) - {SIGNIFICANT_DIGITS - 1})"
    down = f"floor({value} * pow(10, -{exponent}) + 0.5) / pow(10, -{exponent})"
    up = f"floor({value} / pow(10, {exponent}) + 0.5) * pow(10, {exponent})"
    return f"0 if {value} == 0 else ({down} if {exponent} < 0 else {up})"


def leader_path(
    cx: float, start: float, end: float, lane_top: float, above: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """A leader's points from *start* down to *end* at *cx*. Where *cx* runs through
    one of the blocks *above* (``(left, width)``, the lane from *lane_top* down), it
    jogs in the gap over that lane to the middle of the gap beside the block, nearer
    side first, runs down there, and jogs back in the gap over its own block."""
    inside = [(left, w) for left, w in above if left < cx < left + w]
    if not inside:
        return [(cx, start), (cx, end)]
    left, w = inside[0]
    half = STRIP_LABEL_GAP / 2
    sides = sorted((left - half, left + w + half), key=lambda x: abs(x - cx))
    clear = [x for x in sides if not any(a < x < a + b for a, b in above)]
    jog = (clear or sides)[0]
    upper, lower = lane_top - ROW_GAP / 2, end - ROW_GAP / 2
    return [(cx, start), (cx, upper), (jog, upper), (jog, lower), (cx, lower), (cx, end)]


def value_lanes(spans: Sequence[tuple[float, float, str]]) -> list[int | None]:
    """The lane of each ``(left, width, kind)`` span in beam order, one of
    :data:`STRIP_VALUE_LANES`: the lane its kind took first when the span clears the
    last one there by :data:`STRIP_LABEL_GAP`, else the other lane that it clears;
    ``None`` when it clears none."""
    ends = [-math.inf] * STRIP_VALUE_LANES
    kept: dict[str, int] = {}
    out: list[int | None] = []
    for left, w, kind in spans:
        first = kept.get(kind, 0)
        order = [first, *(n for n in range(STRIP_VALUE_LANES) if n != first)]
        lane = next((n for n in order if left >= ends[n] + STRIP_LABEL_GAP), None)
        if lane is not None:
            kept.setdefault(kind, lane)
            ends[lane] = left + w
        out.append(lane)
    return out


def partial_extreme(measure: str, tokens: Sequence[str], shaped: Callable[[str], str]) -> str:
    """The figure of a largest or smallest (*measure* ``max`` or ``min``) over
    *tokens* that passes over a member with no value: each member with no value
    (``{t} != {t}``) stands in as the measure's far end, minus or plus infinity,
    which never wins; *shaped* rounds the reduction for print. With no member
    reporting it is no number, so the label keeps its dash: the engine's
    reductions are numpy's, which give no number over one that is missing."""
    far = "-inf" if measure == "max" else "inf"
    each = ", ".join(f"where({t} == {t}, {t}, {far})" for t in tokens)
    none = " * ".join(f"({t} != {t})" for t in tokens)
    return f"where({none}, nan, {shaped(f'{measure}(({each},))')})"


def holder_condition(measure: str, k: int, n: int) -> str:
    """The condition under which member *k* of *n* is the first that holds the
    extreme *measure* (``max`` or ``min``) of rule pvs ``{0}`` .. ``{n-1}`` among
    the members that have a value: it has a value, no member before it is as far
    or farther, and none after it is farther; a member with no value (``{j} !=
    {j}``) is passed over. Of two members both holding it the first wins (the
    tie rule), so at most one condition holds, and exactly one while any member
    has a value."""
    first, rest = (">=", ">") if measure == "max" else ("<=", "<")
    terms = [f"{{{k}}} == {{{k}}}"]
    for j in range(n):
        if j == k:
            continue
        beaten = first if j < k else rest  # an earlier member wins a tie
        terms.append(f"not ({{{j}}} {beaten} {{{k}}})")
    return " and ".join(terms)


def _over_all_tooltip(tile: KeyTile) -> str:
    """A key figure's tooltip: what it counts or reduces, then its members by name,
    one per line, the rest counted past :data:`TOOLTIP_LINES`."""
    members = tile.names or tile.channels
    head = f"{tile.shown_as.replace('_', ' ')} over {len(members)} channels"
    shown = list(members[:TOOLTIP_LINES])
    if len(members) > TOOLTIP_LINES:
        shown.append(f"and {len(members) - TOOLTIP_LINES} more")
    return "\n".join([head, *shown])


def metric_formula(name: str, item: Tile) -> dict[str, Any]:
    """The screen formula of a key number: its expression over ``{key}`` tokens and the
    channel each key binds. Counts are sums of ``1 * (test)``, so a true test counts one
    whatever type the engine gives a comparison."""
    bindings: dict[str, str] = {}
    if item.measure == "count_in_tolerance":
        terms = []
        for i, (setpoint, readback, term, amount) in enumerate(item.pairs):
            bindings[f"s{i}"], bindings[f"r{i}"] = setpoint, readback
            gap = f"abs({{s{i}}} - {{r{i}}})"
            bound = _number(amount) if term == "absolute" else f"{_number(amount)} * abs({{s{i}}})"
            terms.append(f"1 * ({gap} <= {bound})")
        expression = " + ".join(terms)
    else:
        keys = [f"c{i}" for i in range(len(item.channels))]
        bindings = dict(zip(keys, item.channels, strict=True))
        tokens = [f"{{{key}}}" for key in keys]
        if item.measure == "count_true":
            expression = "+".join(f"1*(abs({t})>0)" for t in tokens)  # no value: no count
        elif item.measure == "count_false":
            expression = " + ".join(f"1 * ({t} == 0)" for t in tokens)
        elif item.measure == "rms":
            expression = f"sqrt(({' + '.join(f'{t} ** 2' for t in tokens)}) / {len(tokens)})"
        elif item.measure == "sum":
            expression = " + ".join(tokens)
        else:  # max, min: a reduction over one tuple
            expression = f"{item.measure}(({', '.join(tokens)},))"
    return {"name": name, "kind": "scalar", "expression": expression, "bindings": bindings}


def stem_levels(xs: Sequence[float], room: float = BADGE + BADGE_GAP) -> list[int]:
    """The level of each of a row of things at *xs* (in order), each *room* wide: the
    low one, unless it would touch the last one there, then the high one; when both
    would, the level whose last thing is further away. Badges' stems, rings' tags."""
    last = [-math.inf, -math.inf]
    out = []
    for x in xs:
        level = next((n for n in (0, 1) if x - last[n] >= room), None)
        if level is None:
            level = 0 if last[0] <= last[1] else 1
        last[level] = x
        out.append(level)
    return out


def _strip_rows(
    elements: Sequence[StripElement], widths: Mapping[str, float]
) -> list[list[StripElement]]:
    """The strip in rows, each as long as fits at no less than :data:`MIN_STRIP_SCALE`."""

    def fits(chunk: Sequence[StripElement]) -> bool:
        return strip.arrange_scaled(chunk, widths, STRIP_WIDTH, fill=True)[1] >= MIN_STRIP_SCALE

    rows: list[list[StripElement]] = []
    start = 0
    while start < len(elements):
        lo, hi = start + 1, len(elements)  # one element always makes a row
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if fits(elements[start:mid]):
                lo = mid
            else:
                hi = mid - 1
        rows.append(list(elements[start:lo]))
        start = lo
    return rows


def render(
    view: View,
    *,
    graph: PanelGraph,
    screens: Callable[[Target], str | None],
    doc_id: str,
    colours: Mapping[str, str] = MappingProxyType({}),
    cuts: list[str] | None = None,
) -> dict[str, Any]:
    """The view as a Screen-IR 1.1 document.

    Args:
        view: What the screen shows.
        graph: The panel graph, for the health rows the lights' rules are written from.
        screens: The screen file a link to a target opens, relative to this
            screen's directory.
        doc_id: The document's id; the caller names it, as it names the files
            links open (:func:`osprey.facility.panels.emit.screen_ids`).
        colours: The fill of each symbol on a strip, by symbol name (the
            drawing settings' ``colours``); a symbol not named is filled with
            the panel's ground.
        cuts: When given, each label the layout cuts to fit is added to it, as it
            was before the cut.

    Returns:
        The document, as JSON-ready data. Equal inputs give equal documents.
    """
    token = _CUTS.set(cuts)
    try:
        return _Page(view, graph, screens, doc_id, colours).draw()
    finally:
        _CUTS.reset(token)
