.. _reference-osprey-panels:

=============
osprey-panels
=============

The command of osprey-panels, the panel engine: it writes the operator
screens of a facility from a built ``facility.json`` and the facility's panel
rule tables, as Canopy Screen-IR documents with an index. The package carries
none of the framework and depends on jsonschema only. ``pip install
osprey-panels`` installs it and its ``osprey-panels`` command;
``python -m osprey_panels`` runs the same command.

**Experimental.** The commands, the rule tables under ``panels/``, the
``draw.json`` settings and the screens they write may change in any release,
without a compatibility path.

How the package versions, and which ``facility.json`` versions and Screen-IR
version each release reads and writes, is in the Stability section of the
package's
`README <https://github.com/als-apg/osprey/blob/main/packages/osprey-panels/README.md>`_.
This page is built from the same tree as the rest of this site, so it can
describe a release newer or older than the one you have installed; the
package's
`changelog <https://github.com/als-apg/osprey/blob/main/packages/osprey-panels/CHANGELOG.md>`_
says which release added what.

Every input is named on the command line; nothing is read from a profile or a
checkout. Reports go to stdout, warnings and errors to stderr. The command
exits 0 when it did what it was asked, 1 on a failing problem or a refused
input and 2 on a usage error. ``osprey-panels <command> --help`` lists every
option.

Inputs and screens
==================

``write`` and ``check`` read the facility file ``--facility`` names (a built
``facility.json``; neither builds it) and the rule tables in the directory
``--tables`` names, and draw with the settings in ``draw.json`` beside the
tables (``--draw FILE`` names another; with neither, the empty settings, and
one line says so). ``--theme FILE`` names a theme file whose state colours a
colour in ``draw.json`` must stay clear of; without it, the engine's fallback
state colours only.

The optional ``write_under`` lists place ids a write keeps to: the home, those places and
the places within them, the devices placed there, each device in no place one
relation joins to one of those, and each group with one of those devices as a
member. A written screen links only to files of the same write: it names a
screen the write does not produce as text, without a link, and
``index.json`` lists those screens under ``outside``. Under it the home lists
only what the write keeps,
the kept places that lie in no kept place at the top and one line naming the
places the write keeps to, and the not-placed indexes only kept devices; a
group is cut to its kept members, and a group cut to the same members as
another is written once (the one all of whose members are kept, else the
first by id); the other holds no members under the scope, so screens name the
kept one. ``check`` reads the same key: a screen outside it may
be absent, and only the health rules of what it keeps must be read. The
screens are
Canopy Screen-IR documents, one file per screen, in nine view kinds:
``home`` (the one screen above all others: the top places, those that hold
a device of a model first as the machines and the others below, every set under
its class and every group with no set view, and what is in no place, by
class, each class opening a page of its devices, so every screen is reached
from it), ``card`` (one device, its signals, picture, relations and calibration
records), ``channels`` (every channel of a device with more than a card
lists, grouped by role, then record type: the one plain list), ``set`` (the members of a group as a grid), ``calibration`` (one
calibration kind over a set), ``place`` (a place's devices in beam order
with a band of key numbers: positions are each device's pose in its model's
beam frame, else its ``s`` along its model. A model's beam frame is the
``positions.csv`` frame in which most of its devices' rows state an ``s_m``
and assume nothing (the devices with no model count as one more model), so
two rings each have their own; a tie goes to the frame with the most such
rows of all devices; a frame where fewer are stated is a local frame; one strip keeps to the beam path most of its
devices are on, the others wait in its tray (a device's card says which
position it uses and where it is read from, and gives the facility file's
``s`` beside a pose that differs from it by more than 1 cm), and a stretch across a ring's
start is drawn in one piece; a card's strip window reads on across the start
the same way. A ring is a model the facility file marks ``periodic``, at its
``length``, and a beam frame takes the length of its model), ``rack`` (a rack elevation, chassis at their
unit; a half-width unit, whose ``located_in`` claim states ``side`` ``L`` or
``R``, fills that half, two halves of one unit side by side), ``rollup`` (the health of a place and everything in it, level by
level; a place with at least two children along a beam path the facility
file marks ``periodic``, going round it (no gap between their positions half
the ring or more, so one sector's places are no ring), draws them as arcs of
a ring, in beam order) and ``checkout`` (every setpoint beside its readback).

Every screen but the home leads back to it first in its crumbs, then out
through the places around it (to each place's view where it has one). The
ways down meet at a device's card: by place (a place view's racks open their
elevations, its sets their set views) and by system (the roll-up of one
system within a place opens the sets of its devices); a set opens its
members' cards, and a card names its sets on a line of their own above its
sources. A place's card draws at most 60 lights of its own channels and
counts the rest, which its roll-up reads. Controls of one group that share
a word are each named by what tells their channel apart ("Enable CH2"), and
a device whose presses all stand on its front has no operate block.

A set is a group whose ``kind`` is ``family`` (the default; a ``section`` is
never a set) and whose members are all devices of one class; ``check``
notes a family that is no set only because some members are no device of a
stated class (``unset_group``, naming them). A set of more than 96 members
is paged by its rows, each page linking the others; a member with no health
light shows the value its card's first key tile shows. A device whose card
lists counts rather than its channels gets a roll-up of its own, paged at
400 rule-bearing channels, which its node in the place roll-up opens: every
rule-bearing channel is read by exactly one roll-up page.

The rule tables are CSV files, each held to its header, and each may be
absent. What the facility file states is the default, and a table states
only what it does not: a setpoint's readback is its channel's ``pair`` and
nothing else (a pair the build derives is marked inferred, with its rule), a thing's short name what ``names.csv`` states,
else its ``label``, else the last ``:``- or ``/``-separated part of its id, a
channel is drawn by its ``role`` (a setpoint as an entry
beside its readback, a readback as a value; ``record_types.csv`` names its
record type and, where its ``drawn_as`` is not empty, any other drawing), and a channel on a device that states a
``signal`` plays that signal as its role on the device's card, shown as an
entry, a word or a value by how it is drawn, a state that ``panels/health.csv`` rows
rule as a lamp; such a role says what the channel is, never how it is pressed:
a writable channel no row names is offered as any such channel is (a choice
of its stated words, an action, an entry) (``channel_roles.csv`` names
walked scopes, labels, presses and channels with no signal; a row's stated
``label`` is the word the card shows for the channel on every tile, value row
and group title, and stays in front of what tells two channels of one role
apart, while a choice's or an action's label is the word on its press). ``check``
prints one ``redundant: <table>: <n> rows`` line per table whose rows only
restate the facility file or these defaults with no column of them read for
anything else (a
``channel_roles.csv`` row without which every device it matches plays the same
roles in the same order, read the same, so never a press), so a
deployment can delete them, and one ``redundant: record_types.csv drawn_as``
line for the rows whose ``drawn_as`` is their role's: their record type is
read, so the row stays and only that value may be left empty.
The tables: ``positions.csv``
(where a thing sits), ``record_types.csv``,
``panels/health.csv`` (the state a channel's value means: ``normal``,
``warn``, ``bad``; a value no row names is ``unknown``, and one that two
sources give two states is ``disagree``),
``excluded.csv``, ``pictures.csv``, ``names.csv``, ``metrics.csv``,
``probe_pictures.csv``, ``calibration_kinds.csv``, ``calibrations.csv``,
``calibration_curves.csv``, ``state_labels.csv`` (the word each value of a
discrete channel means; a command's button reads the word for the value it
writes, else the action its channel's name says), ``units.csv`` (a channel's
unit and the divisor its raw value is shown through), ``channel_roles.csv``
(what a channel is to a device card, by class and name pattern; its ``scope``
is ``device``, ``parts``, or a relation type of the vocabulary that runs from
a device to a device, such as ``read_out_by``, ``controlled_by`` or
``interlocked_by``, walked from the device and its pattern filled from the
relation's port) and
``card_recipes.csv`` (a class's card recipe, optionally for the devices under a
place, a part by its whole's place, or whose id matches a pattern; the first
row that takes a device wins),
``card_sections.csv`` (a recipe's drawn sections in order) and ``chains.csv``
(the hops of a recipe's chains: each walks one relation in one direction from
the device or an earlier hop, keeps the things its ``to`` filter takes, and
shows its roles in the block; a hop that reaches nothing is a dashed "not
stated" block, two answers are stacked as contested) and ``symbols.csv`` (how
every strip, a card's and a place's, draws a device of a class, optionally
only those whose id matches a pattern: one of the drawing's symbols and a
lane above, on or below the beam pipe; a place's strip has one lane off the
beam; a class no row names is not drawn on a strip). ``draw.json`` keeps
only ``colours`` and ``annotate`` (both keyed by symbol) and
``write_under``; a rack is a place some ``located_in`` claim gives a unit in. A strip section draws the devices of its
classes within its window of the device along the beam, to scale, each with
the value of the section's role (a window of 3 m either side when the section
states none); a device placed only by a place around it is drawn in a dashed
bracket over that place's stretch (the place's own position and those of the
devices in it), the window widened to hold it, and a card
whose device is placed by neither says "where it sits is not stated".
``fronts.csv`` (a recipe's connectors: one row per connector or numbered range
``Outlet{n}``, by face, group and order) draws a front section: the connectors
at one pitch on a panel body, numbered by a stated count, else the served
channels that match the row's ``channel``, else the sockets that exist as things
of their own (``<device>:<connector>``), else the ports the relation states, in
number order; under each, what is on it (the thing whose ``far_relation`` names
that port, ``in``; the device's own relation from it, ``out``; both sides of a
socket thing, ``socket``), a dashed "—" where nothing is stated and every thing,
dashed and "contested", where a side has two; its state lamp; under the line,
in bands each labelled at the left (the role's title and unit, "press" for
the buttons; where every connector has as many rows and the labels at each
place end alike, one band per place, labelled with the words they share),
the section's
roles of that connector and its presses (drawn there, not again under
operate). A row's optional ``roles`` column (``;``-separated) names the section
roles its connectors carry; empty means every section role. A role binds to a
connector only through a row that lists it, so two groups numbered alike
(outlets 1-8, phases 1-3) never take each other's channels; on a connector
with no number the listed roles bind the device's channel of that role whose
name carries the connector's name. A ``{n}`` row with ``roles`` and no
``channel`` is numbered by the last number of the device's channels of its
first role, with no lamp; a device that serves none gets no such group. A
connector with no number whose row lists ``roles`` is left out on a device
where neither its lamp nor any listed role binds, and a group all of whose
connectors are left out is dropped. ``far_relation`` and ``far_direction``
are given together or both left empty; a row with neither is a reading, and
a group whose connectors wrap onto a second line names it "<group>
(continued)"; a group of readings is a table on a line of its own: no socket, cable or far end, the connectors'
names as column heads, the lamps first in a "state" band, then a band per
role. Clients the relation names on no stated port
are drawn apart by class, unless the row's optional ``apart`` column says
``no``: then only the clients on a stated port are drawn; a range with nothing in it is left out. ``write``
prints per recipe the connectors with something stated, contested or nothing,
the readings apart, and the devices with every connector that is not a reading
stated, some, none, or readings only. A ``ring-strip`` section draws a group
(``same place``: the devices of the card's recipe in the device's place; or
``<relation> <direction>``: the devices that relation names) one column each,
by ring position, else in number order with "in number order: positions not
stated", each with the value or lamp of the first of the section's roles it
plays (an empty ring where it plays none), the device in focus and every other
opening its card; values are shown as numbers, since no table states a scale
for a bar. When the members have places, they run by place, a row each
under the place's name, at the finest level of the place tree that gives two
to eight places, in ring order (in number order without positions), those
with no place last; over the strip, "giving <first role> now: n of m" counts
the lamps ok now with one formula over the same lamps' rules, or, when that
formula is too long, each place's row counts its own. Inside a place a
member shows the last segment of its id (two alike keep what follows the part
of their ids they share), its full id on hover. A chain hop, a front
connector's far end or a ring-strip member that rests on a relation claim
with ``basis: inferred`` says "inferred" beside it, the claim's ``rule`` on
hover ("the rule is not stated" when it has none); when every member of a
ring-strip is inferred, its caption says it once ("all 12 inferred: <rule>")
and no member does; when most are, the caption says "inferred unless marked
stated: <rule>" and only the others are marked "stated". A ``ladder`` section draws the section's first role against the
others as set points, evenly in their order (not to scale), each with its live
value, and the live value as a dot in the gap between the set points it lies
between; a set point not served is left out and named, and ``write`` counts
the ladders drawn whole. A ``section`` row with a ``focus`` relation
(``part_of``) draws a part's card: the thing that relation names, drawn as its
class's picture with the part outlined in the focus colour (the picture rows
whose ``signal`` matches a channel of the part), and beside it what is stated
about the part (its relations, each opening the other thing) and, dashed, what
is not (its owner, its place on the drawing, a drawing at all, each hop of the
recipe's chains that found nothing). A ``curve`` section draws the points of
the claim in force of the calibration kind its ``members`` names (an
excitation curve) as dots joined by a line, the least and most of each axis
at its corners and the claim's source under it; the operating point is not
drawn, since no table states the conversion from set point to current. A
device with no such curve says none is stated. The claim in force is the
newest accepted one: a contested claim is never in force nor a base value; it
is named "contested, not in force" beside the value in force, alone it makes
the row and its cell "contested" with no value, and a contested curve is not
drawn. The calibration block is the
``calibration-strip``. A device with no place of its own and an accepted
``part_of`` to a placed whole stands where the whole is: in its crumbs, on
strips at the whole's position, and its card says "in <place> by its whole
<whole>". A device that is an endpoint of channels another device holds (a
family's shared supply) shows that supply's value roles read-only under
"shared supply <id>", with a link to the supply's card; the supply's presses
stay on its own card. A device gets a card when it has a channel its card shows (one on it, one it
is an endpoint of, or one a ``channel_roles.csv`` row walks to from it) or a
drawing (a ``card_recipes.csv`` row or a ``pictures.csv`` row that applies to
it); a pose or a relation alone gives none. ``cards.csv`` (action, class,
under, devices, attribute, when, owner_relation, owner_direction, owner_to,
reason, source) holds the exceptions: ``action`` ``keep`` gives the devices a
row selects a card, ``drop`` takes theirs. ``class`` ``*`` matches any class,
``under`` holds a device whose place is it or lies within it, and a part by
its whole's place (a part with no place of its own sits where its whole sits), ``devices`` is an id pattern, ``attribute`` is ``name=value``, a
value the device's ``attributes`` map states (a list value matches each of its
items), and ``when`` is ``no-channel`` (no channel its card shows),
``no-drawing`` (no recipe, no picture) or both joined by ``;``. Every selector a
row states must hold, and the first row that selects a device decides. A
``drop`` row may name an owner by a relation and a direction (a ``keep`` row
may not): the device is drawn on its owner's card, a link to it opens that
card, and ``index.json`` lists it under ``drawn_on`` with the owner's screen. A
device with no card and no owner found gets no screen and no entry: a related
card names it without a link. ``check`` counts the devices with no card, those
the default alone leaves without one, and how many devices each row decides. A device card is drawn:
its title line, up to five key tiles (the ``metrics.csv`` rows of view
``card`` whose ``level`` names its recipe, else the roles exactly one of the
device's own channels plays, a neighbour's channel its roles reach not
counted, but never a role whose word is a ``nameplate`` signal, which says what the device is, not how
it is (a ``channel_roles.csv`` row giving the channel another word overrides
it); a tile's measure is
``value`` or ``word`` of the role's first channel, or over every channel of
the role on the device ``count_true`` shown "k of N", ``max`` or ``sum``, a
dash and "one of N has no value" while any member has none), its relations as
a chain of blocks (the device's own block showing only roles one channel of
it plays: a role played per connector or member is not rows there), its
drawings (a drawing none of whose parts has a place is a compact row of
blocks, each the part's name over its signal's light and value, with
"arrangement not stated", not an empty frame; a picture any of whose rows
has state ``schematic``, or that is drawn wholly at the default proportions,
says "schematic, not to scale" in its caption, while a stated scale keeps its
bar; a light is ``disagree`` only at
a value its rows give two states, so a value every row calls normal is
normal), an operate block (its control roles, else its
writable channels: a choice of stated states, an action, or a typed value
applied with Apply, never a text record (``drawn_as: text``: Apply writes
only a number, so it stays in the channels view), a line beside Apply saying what it will write from the
same formula as its press value; more than six entries of one role are a grid
of three columns under the role's title, their line naming no channel (each
on hover), and an entry of a front's role on a connector is set under the
front, named by the connector; press groups side by side, up to three; a choice with one
stated value is an action with the live status beside it; "no operator
actions stated" when there is none), every channel the card is about in a
grid (its own, and of a device its roles reach the channels beside the ones a
role names; up to 60; a device with more shows the counts by
owner and opens its ``channels`` view), then one muted line for where it is
and one for its sources (the count of statements and their documents, each
statement on hover). A channel ``channel_roles.csv`` gives the role
``internal`` (a driver's own record) is never offered and never listed on a
card, only counted. A relation the chain names is drawn once per thing it
reaches after the ``to`` filter (the place for ``located_in``, a device for
the rest); only two distinct things after the filter make the hop contested,
both named (a claim's own standing does not); a hop after a step that reached
several things walks from each and unites what they reach. A button writes
only a value a source states (a role's press value or the value of its state
word) or one marked inferred (a role's inferred press value, 1 for a command
whose states do not name 1), and an inferred value is said so wherever the
card names it ("(value inferred, not stated in the database)"); a press with
no value is left out. A writable channel of another thing that the card
reaches only through a relation claim with ``basis: inferred`` (a walked
relation such as ``read_out_by``, or ``part_of``) is never offered as an
action, a choice or an entry: it is shown read only where it would have
stood, under "join inferred, command not offered" with the claim's rule on
hover; the device's own channels are not affected. ``check`` counts the
presses stated, inferred, left out, and withheld because their join is
inferred. Apply stays off the canvas and out of the page (so the keyboard
cannot reach it either), with "type a value first" in its place, until the
field holds a number, so no press writes a value nobody typed. A light takes a
colour only where its meaning is stated (a state word, or a number's
thresholds); else it is the neutral outline, on a tile, in a chain block and in
the grid alike. An arrow's word and a block's title are never cut: the arrow
takes the run its word needs, a word may take two lines or its last name
segment with the whole word on hover, and a title two lines. ``check`` also counts the default-recipe
cards with no key tile (no card metric and no role that shows a value). A class with no recipe row is drawn with the default recipe (the
sections stated for the recipe ``default``, else a chain of its place, power,
network and controller); ``write`` prints how many device cards each recipe
drew, per chain, how many were drawn whole and the hop most often missing, and
per strip, how many devices are placed, placed by their place or not placed. A table that breaks its schema
prints the file and line and exits 1. A ``pictures.csv`` row with a place id
in its optional ``under`` column draws only the devices of its class in that
place (and one with a ``devices`` id pattern only the devices it matches,
with ``!`` first every device but those), so two machines with devices of one class name each state their own
drawing; a device under no row's place gets no drawing and no note. Health
is computed by the viewer from rules written into the screen; nothing is
read from the control system while screens are written.

Commands
========

``osprey-panels write --facility FILE --tables DIRECTORY --out DIRECTORY [--draw FILE] [--theme FILE] [--kind KIND]... [--under PLACE_ID] [--top N]``
   Write the screens under ``--out`` as ``<kind>/<screen id>.screen.json``
   with an ``index.json`` that lists them. Equal inputs give equal bytes: no
   file holds a time, an absolute path or a host name, and each screen's
   description ends with the facility's hash and the panels format number.
   Files an earlier run listed and that are no screen any more are deleted;
   nothing else is touched, and a non-empty ``--out`` without an
   ``index.json`` is refused. ``--kind`` (repeatable) and ``--under`` write
   only the screens that match; links to the others are still written. The
   command prints the screen count and size per view kind, the class summary
   (see ``osprey-panels classes``) and the ``--top`` largest screens (default 10).
   Every screen is checked against the vendored Screen-IR schema before any
   file is written: when one fails, the run names it, writes nothing and
   exits 1.

   Each entry of ``index.json`` names its screen's ``class``, and the index's
   ``classes`` table states, per class, its key, its ``standard`` screen, its
   member count, how many members are ``off_standard``, how many have a ``cut``
   label, a ``layout`` hash (of the standard's structure, the format and the
   drawing settings), a ``drawing`` digest (of the standard's written file
   without the model stamp, which moves when the screen is drawn otherwise and
   not when only the facility file's content hash does, taken by the rule the
   index states as ``drawing_rule``) and a ``sample``;
   ``class_summary`` holds the counts
   a gate reads, with the classes of one screen per view kind. Two screens are
   of one class when they have one key: for a device's card and its channel
   list, the device's class, the card recipe it is drawn from and the set of
   signal roles the device's channels play (a role played by two channels
   counts once; the driver-internal role and channels with no role are not in
   it), and, for a device with no channels of its own, ``own_channels: none``, so it is
   not classed with one whose channels play no stated role. Where cards
   of one such class drawn by the default recipe show different blocks (a card
   one can act from and a read-only one, a card with or without key tiles),
   the key also holds ``blocks``, the kinds of block the card shows by presence
   (a channel list drawn as a grid or as a count is one block), and the class
   splits into one per kind of card; a class whose cards all show the same
   blocks keeps its key and its id. For a
   calibration overview, its calibration kind and its structure (a gain and an
   offset are judged apart even where they are drawn alike); for every other
   screen (a set, a place, a rack, a roll-up, a checkout, the card of a place or
   a group), its view kind, the kind of thing it shows and its structure. The
   device types of a set's members are names, not a layout, and are not in its
   key.

   Sameness is measured on what a screen shows, not on its file. Its structure
   is its sections and the kinds of item, block and relation they hold. Left
   out are text and numbers; how many items a repeated block holds and their
   order (a size); the parts that state what is known rather than what is drawn,
   such as a source, a unit, a stated position or a link to the thing's screen
   (a gap where one is missing); and data-state flags such as inferred,
   contested or writable. The standard is the member whose structure most
   members share (a tie goes to the smaller structure hash), and the ``layout``
   hash covers the standard's structure, the panels format and the drawing
   settings: it changes when the class's typical screen changes, not when a
   member is added or renamed.

   An entry off its standard lists how under ``differs``, in words: a section,
   a kind of item, a block or a relation present or missing (``no tiles
   section``, ``relations: a value row, which the standard has not``). What
   varies from the standard without making another screen is under
   ``varies``: sizes (``foot: where: 4, the standard 3``), gaps, flags and
   drawing variants (a channel list drawn as a count above its size). Each
   entry also states its own ``sizes``, ``gaps`` (per gap the count, such as
   ``unit`` or ``where without a link``, for a checker to total per class),
   ``flags``, its longest ``label`` in characters, and under ``cuts`` the labels
   the layout cut to fit. A rack screen numbers every unit, draws a free one as
   a slot and keeps the lists of the rack in one column beside it, down to the
   rack's foot and never past it: those at no unit, then the classes and
   sources, then those side by side at one unit, level with it and joined to it
   by a line, the bar over them saying whether they are listed beside the rack
   or under it. The list of those at no unit takes the room the rest leaves;
   what does not fit beside stands under the rack across the full width: the
   rest of the list under its heading continued, then each group side by side
   that did not fit, its units on its heading in place of the line. A list
   takes as many columns as fit with each as wide as its own longest name; a
   name longer than the whole width keeps the part that tells it from the
   others, the whole on hover. A rack's list is filled downward in
   balanced columns (two columns of a band differ by one row at most), a larger
   gap after every fourth row, in bands of twenty rows read down then across; a
   list of one or two rows fills across. An operate panel draws each control as one row on one
   set of columns: tag, name, now, field or button, what it does; groups whose
   rows end at their button or value (actions, choices, read-only values) stand
   up to three side by side, and a group of many such rows takes up to three
   columns, filled downward, as do a choice's segments that wrap. Said once:
   the kind's tag stands at the head of a run of one kind (groups of one kind
   that follow each other, each with its line of what it writes, are one run),
   and a row whose button says its name has no name beside it. Every list
   ordered by name takes the numbers in a name by value (``2`` before ``10``);
   it only orders, it groups nothing. A thing's channels come in that order,
   so a role's rows, a block's lines and a card's channel list follow it. Frames in a row whose content needs at most half
   the width each stand in two columns when they have one role
   (live values, references, an action block, a picture with its curve; two
   groups of a channel list only as stated groups of one grouping, a group of
   no stated kind alone), read downward, split where the taller column is least
   tall, so short frames stack beside a long one; a frame keeps its own height
   unless matching its neighbour's foot adds at most a third of it, and the page
   under a shorter column stays plain; content that stretches (a curve's plot) is drawn
   after the next frame and, where that one takes at most half, fills the other
   half. A screen shows what it is for first: sections of
   nothing but links follow the rest, key numbers with no data yet fold into
   one line, on the band's last row where it fits there, else under it, and a roll-up's summed state is larger than what it sums, a
   child it does not sum set apart in a dashed outline on the same row where
   all fit one row at their content's width, its counts one muted line on the
   heading's line. Every widget that draws a view item names it in its
   ``meta`` comment, which a viewer does not read: ``item: <kind>``, then
   ``thing: <id>`` where the item stands for one thing and ``order: <n>`` where
   the view states an order (a rack unit's by where it stands in the rack, not
   by the order its units are listed). The kinds are ``rack-unit`` and ``grid-row``,
   which read down (top to bottom, then on in the next column), and
   ``strip-element``, ``key-tile`` and ``chain-block``, which read across (left
   to right, then on in the next row). The list is complete: a key figure's
   members (its lamps, its holder names) are parts of its tile, not items of
   their own. A class's ``drawing`` digest leaves these comments out (rule 2;
   an index with no ``drawing_rule`` was written by rule 1, which kept them in,
   so between the two every class's digest differs once). Every widget stands
   on whole pixels, each edge rounded where it is written, and every screen
   ends one margin under its lowest frame. A section's heading line carries
   what qualifies it: a count, a one-line legend, or the one row of chips that
   fits beside it. The layout measures text in the viewer's own font
   (IBM Plex Sans): a column takes the width its longest label needs (a
   checkout's channel and thing columns their longest names), a long
   label in a chain block takes two lines, a muted line of prose (a caption, a
   note, what a press writes, a source, also beside a fact) takes as many lines
   as it needs, conditions that several calibration rows state alike are stated
   once under the rows, set off by half a row of space and a thin rule and
   tied to their rows by a reference mark (1, 2, ...) after each row's title
   and at the start of the line, which says the text alone, a name with no separator breaks where a small letter
   meets a capital, and a text still longer than its box is cut there with an
   ellipsis and listed. The engine writes the whole text as a tooltip on a box
   over the cut label (a link drawn over it carries that whole text too in its tooltip, several cut labels joined one per line, a text another line holds left out; its one-line hover label is the link's own label, at most 40 characters and cut
   with an ellipsis past that, the basis of an inferred relation said in the
   tooltip only); a viewer that
   shows tooltips shows it, and no viewer has to shrink or clip a label. A
   value display is as wide as the widest text the facility
   states it can show: its longest state word, else its limits as the display
   prints them when they are wider than the stand-in ``-0.00123``, else that
   stand-in, with the unit where the display prints it. Every text label's box
   is at least one line of the font high (a light's mark among them), grown
   about its middle where it was lower. A class's ``sample`` names the members a reviewer
   opens: the standard, the smallest, the largest, the one with the longest
   label, the first off standard, and one member on each side of every
   variation that changes the drawing (a part drawn with or without a stated
   position, a channel list drawn as a grid or a count). Each member of the
   sample is ``{"file", "as"}``: its file and what it stands for, one or more of
   ``standard``, ``smallest``, ``largest``, ``longest labels`` and ``off
   standard``, and the drawing sides it is the first to show (``x_mm missing``,
   ``channel list drawn as ChannelCount``).
   A device's card and channel list also state, under ``others``, what they
   show of other things on purpose: each thing, the kind of item that draws it
   (``strip member``, ``chain block``, ...) and the channels of it the screen
   binds; a row that names no thing (a channel list's row of a device the
   roles reach) is placed by its channel's owner. A checker can then tell a
   neighbour's channel from a channel on the wrong screen.

``osprey-panels check --facility FILE --tables DIRECTORY --out DIRECTORY [--draw FILE] [--theme FILE] [--strict] [--max-bytes N] [--limit N]``
   Regenerate every screen and compare it with the files under ``--out``;
   nothing is written. It prints the count of each problem kind and the first
   ``--limit`` of each (default 20), and exits 1 on any failing problem:
   ``stale``, ``missing``, ``orphan``, ``dangling_link``, ``unknown_channel``,
   ``stamp``, ``invalid``, ``schema`` (a screen the vendored Screen-IR
   schema refuses), ``incomplete``, ``oversize`` (over
   ``--max-bytes``), ``unrolled`` (a rule channel no roll-up reads),
   ``static_rollup``, ``long_formula`` (an expression over the viewer's
   length limit), ``over_budget`` (more rules or channels than one screen
   may bind) or ``stale_class`` (a screen's class or what its entry states about
   it, or the class table or its summary, in ``index.json`` other than the
   model gives now). A link outside the written set, a card or checkout over its
   budget, a probe row whose signal no channel states, and a ``pictures.csv``
   row whose signal binds no channel (``unbound_picture``) or channels of two
   devices (``ambiguous_picture``), and a strip badge's stem drawn through
   another thing's state word (``crossed_word``) are notes, and fail only with
   ``--strict``.
   Before the problems it prints the device classes by how ``pictures.csv``
   draws them (with a drawing, with parts only, with none) and, per class, the
   parts not placed (no position and no angle): information, never a problem.
   After the problems it prints the class summary from the index. A clean set
   prints ``The written screens match the model.``

``osprey-panels classes --out DIRECTORY [--limit N] [--differs]``
   Print the class table of the ``index.json`` under ``--out``; nothing else is
   read. One row per class, the largest first: its members, those off the
   standard, those with a cut label, the class and its standard's file; the
   first ``--limit`` rows (default 30, ``0`` for all). Then one row per view
   kind: its screens, classes, classes of one screen, screens off their
   standard and screens with a cut label. ``--differs`` adds every screen off
   its standard, and how. The last line is the summary a gate reads:
   ``classes: N classes (S of more than one screen), M screens off their
   standard, C with a label cut; single: <view> <count>, ...``. Exits 1 when
   the index is missing or holds no class table.

Where a screen needs a widget the Screen-IR format lacks (a five-state
light, a hatched fill, a scrolling table, clickable text, a condition on an
enum label, a viewer- or server-side aggregate state, a rule longer than
the expression limit), the nearest widget is drawn and the node's
``meta.comment`` names the need as ``format-request: <name>``;
``osprey_panels.layout.FORMAT_REQUESTS`` lists them.
