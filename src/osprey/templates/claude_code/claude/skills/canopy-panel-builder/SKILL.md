---
name: canopy-panel-builder
description: Build, edit, port and share Canopy control-system screens (panels) with the canopy MCP tools. Use it to create a new panel or screen, change one, convert a legacy .bob, .opi, .edl or .ui display, screenshot a screen, or push a screen to the shared library.
---

# Canopy panel builder

A Canopy screen is one JSON file in the Screen-IR format. The canopy MCP server
stores screens, checks them, renders them and serves them to the viewer. You
build screens through its tools. Do not edit the files of a screen by hand
while the server is running: the tools check every change, and a hand edit
skips that check.

Work in this order: find the widgets and the PV names, write the screen, read
the findings, look at a screenshot, fix, repeat.

## Where screens live

The server writes to one directory, set by `CANOPY_DRAFTS_DIR` when the server
is started. This text calls it your screens root. Everything you create is a
draft in that root:

- A screen with id `rf/status` is the file
  `<your screens root>/screens/rf/status.screen.json`.
- An id is one or more segments joined by `/`. A segment holds letters,
  digits, `_` and `-`. The segments before the last are directories.
- Images that a screen shows are stored below `screens/` too, beside the
  screen that refers to them.

A server may also serve a shared library. It is a second root that no tool
can change. Its screens have ids of the form `library/<id>`. You can read,
check and screenshot them. To change one, read it with `get_screen` and save
the document under an id of your own with `new_screen`. When a library is
served, `library` is not allowed as the first segment of your own ids.

Your screens root is whatever the person who started the server chose. It can
be a private store that only you see, or a directory of a source tree that is
under version control. The tools behave the same in both. What differs is how
a screen reaches other people: see "Before you commit" and "Push to the shared
library" below.

## The Screen-IR file format

A screen is a JSON object with these keys:

- `schemaVersion`: `"1.1"`, or `"1.0"` for a document without theme colour
  roles.
- `kind`: `"screen"`, or `"template"` for a part that is only ever embedded.
- `id`: the name of the screen.
- `metadata`: `{title, source, size}`. `source` is an object with a `type`
  string that says where the screen came from. `size` is `{width, height}` in
  pixels.
- `macros`: a list of `{name, default}`. A screen must give every macro a
  default. A template must give none. Props refer to a macro as `${NAME}` or
  `$(NAME)`.
- `root`: the top widget node. Every other widget is below it.

A widget node is:

- `id`: unique in the document.
- `type`: a widget type from `list_widgets`.
- `props`: the properties of that widget type, from `get_widget`.
- `geometry`: `{x, y, width, height}` in pixels.
- `children`: the nodes inside it, for a container.
- `rules`: optional rules that set one prop from PV values.
- `warnings`: notes the converter left on a ported node.

A small screen:

```json
{
  "schemaVersion": "1.1",
  "kind": "screen",
  "id": "status",
  "metadata": {
    "title": "Status",
    "source": { "type": "hand-authored" },
    "size": { "width": 400, "height": 200 }
  },
  "macros": [{ "name": "DEV", "default": "SR01" }],
  "root": {
    "id": "root",
    "type": "absolute-canvas",
    "props": {},
    "geometry": { "x": 0, "y": 0, "width": 400, "height": 200 },
    "children": [
      {
        "id": "title",
        "type": "text-label",
        "props": { "text": "Status of ${DEV}" },
        "geometry": { "x": 10, "y": 10, "width": 200, "height": 20 }
      }
    ]
  }
}
```

Take the exact rules from the server, not from this example. `get_schema`
returns the JSON Schema of the document, `list_widgets` the widget types, and
`get_widget` the props of one type. Do not guess a type or a prop name.

### References to other files

A screen names other files in three places:

- the `file` prop of an `embedded-display` node, which shows another screen
  inside this one;
- the `file` of an `open_display` action, which opens another screen;
- the image props, such as `src` of an `image-view`.

A reference to a screen is written as `<id>.screen.json`, relative to the
directory of the document that holds it: `parts/knob.screen.json` from
`rf/status` is the screen `rf/parts/knob`. Keep a screen and everything it
refers to in one directory tree. Do not start a reference with `/`, `screens/`
or `library/`, and do not point from your root into the library or back.
`check` reports a reference it cannot follow as `unresolved_ref` with a
`reason`, and a reference into the other root as `cross_root_ref`.

An image reference is either a URL (`http:`, `https:`, `data:`) or a path to
an image stored beside the screen. `convert_legacy` copies the images it
finds beside a legacy file. No tool uploads a single image.

## The tools

The tools marked "reads" change nothing. The tools marked "writes" create,
change or delete drafts, or file a draft for the shared library. A server may
be started without the "writes" tools. If they are missing from your tool
list, this server is read-only: say so, and do not look for another way to
write.

| Tool | Kind | Use it to |
|---|---|---|
| `list_widgets` | reads | See every widget type a screen may use. Start here. |
| `get_widget` | reads | Get the props of one widget type and their schema, before you use the type. |
| `get_schema` | reads | Get the JSON Schema of a screen document. |
| `search_pvs` | reads | Find PV names by a glob pattern such as `SR:C01:*`. |
| `read_pv` | reads | Read one PV once. Use it to confirm that a PV name exists and connects. |
| `check` | reads | Get the findings for a document or a stored screen without saving anything. |
| `list_drafts` | reads | List your drafts, followed by the library screens. |
| `get_screen` | reads | Read one screen and its `rev`. |
| `screenshot` | reads | Render a screen as a PNG and look at it. |
| `list_collections` | reads | List the collections: named sets of screens and tools. Answer with collection names and member counts, render `open_url` as a link, and never quote field names to the operator. |
| `new_screen` | writes | Save a whole document as a new draft, or replace a draft. |
| `add_node` | writes | Append one widget node to a container of a draft. |
| `patch_node` | writes | Change the props, geometry, rules or warnings of one node. |
| `remove_node` | writes | Remove one node and its children. |
| `delete_draft` | writes | Delete a draft. |
| `convert_legacy` | writes | Port a `.bob`, `.opi`, `.edl` or `.ui` panel and its sub-screens into drafts. |
| `request_push` | writes | Ask for a draft to be added to the shared library. |

Each tool's own description has the full argument list and the shape of its
result. This table says when to reach for which.

No tool writes to a PV. `read_pv` and `search_pvs` only read.

## Build a screen

1. Call `list_widgets`, then `get_widget` for each type you plan to use.
2. Find the PVs. Use the names the user gave you, or `search_pvs`. Confirm
   each name with `read_pv` before you put it in a screen. An empty search
   result is not proof that a PV does not exist: when `skipped` is not empty,
   the PV directory could not be asked.
3. Write the whole document and save it with `new_screen`. Or call `check`
   with the document first, which saves nothing.
4. Read the result. Findings with severity `error` refuse the save, and
   nothing is written. Fix every error and save again. Warnings do not block
   a save. Read them anyway and fix the ones that are real.
5. Call `screenshot` and look at the image. Check the layout, overlaps and
   clipped text.
6. Make small changes with `patch_node`, `add_node` and `remove_node`. Take a
   screenshot after each round of edits.
7. Show the user the screen. With a `set_workspace` tool, send
   `set_workspace` with `[{"cmd": "open", "args": {"ref": <ref>}}]`, the
   `ref` from the result; the `canopy-operator` skill says how. On
   `no_window` or `window_hidden`, open the host's Canopy panel without a
   url and send it again, once; with no host, ask the user to open or focus
   the Canopy viewer. Without a `set_workspace` tool, or when that did not
   work, give the user the `url` from the result. It opens the screen in the
   viewer.

When the user wants several screens shown together in a given arrangement
(two screens side by side, one screen as tabs with several macro sets, a
Trace below a screen), open them with one `open_set` command with
`as: "layout"` instead of one `open` each; "Load an arrangement in one call"
in the `canopy-operator` skill gives the form.

A finding has `nodeId`, `code`, `path` (a JSON pointer into the document),
`severity` and `message`. A finding inside an embedded screen also has
`docId`. The `skipped` list in a `check` result names checks that could not
run. An empty finding list beside a non-empty `skipped` list proves nothing
about those checks.

A write-capable widget or a `write_pv` action draws the warning
`write_not_relayed` when the server relays no writes: the screen saves and
shows its values, but its writes are inert while this server relays no
writes. Tell the user so when the screen is meant to write.

## Plots

Three widget types draw PV data. Each entry of their `props.curves` list
names its PVs with one key, and nothing else draws a line:

- `waveform-plot` draws one array (waveform) PV, named by `y_channel`,
  against its index or against an `x_channel` array.
- `xy-plot` draws `y_channel` against `x_channel`. Two array PVs give a
  curve; two scalar PVs give one marker at the current position.
- `pv-array-plot` draws the scalar PVs listed in `y_channels` as one series,
  one sample per PV, such as the positions of a row of BPMs.

A scalar PV in `y_channel` draws a single point, not a line. A time series of
a scalar PV, or one scalar PV against another over time, is not a widget: it
is the Trace tab's job. Answer such a request with the Trace instruction: tell
the user to open the viewer's Trace tab and enter the PV name and time range
there, or the two PV names on its correlation page. Do not build a plot widget
for it. If the viewer has no Trace tab, say that a screen cannot show it.

A live view of one or more array PVs, overlays of several of them, or their
delta to a reference trace is not a screen either: it is the Waveform tab's
job. Tell the user to open the viewer's Waveform tab and enter the PV names
there. Build a `waveform-plot` only for a screen that shows the array
beside other controls. A screen that is little more than a `waveform-plot`
draws the `waveform_tab` warning. If the viewer has no Waveform tab, a
`waveform-plot` screen is the way to show the array.

## Edit with `rev` and `base_rev`

Every screen has a `rev`, the hash of its file. Every write returns the new
`rev`, and `get_screen` and `list_drafts` return the current one.

Every edit takes the `rev` you last saw as `base_rev`. If the screen changed
since then, the edit is refused with the reason `stale_rev` and nothing is
written. The user can edit the same screen in the viewer while you work, so
this happens. When it does, read
the screen again with `get_screen`, look at what changed, and apply your edit
to the new document with the new `rev`. Never retry with a guessed `rev`.

Change a draft that exists with `patch_node`, `add_node` and `remove_node`.
`new_screen` is for a new draft; with `overwrite=true` it replaces a whole
draft and is not the tool for a change to a few props. Read the draft once
with `get_screen` before a round of edits, not before every call.

- `patch_node` merges `set` into one node as a JSON Merge Patch. Objects merge
  key by key, `null` deletes a key, and a list replaces the old list. It
  cannot change `type`, `id` or `children`: use `remove_node` and `add_node`.
- `new_screen` on an existing id needs `overwrite=true` and `base_rev`.
- `delete_draft` needs `base_rev`.

A write to a `library/<id>` screen, or to a root the server cannot write, is
refused with the reason `read_only_store`.

## Port a legacy panel

`convert_legacy` turns a `.bob`, `.opi`, `.edl` or `.ui` file into a draft and
converts the sub-screens it refers to. Give it the id for the top screen and
either the absolute `path` of the file or its `content`.

The result has a `queue`: the work the converter left for you. Go through it.

- `unknown_widget`: a legacy widget with no Canopy equivalent. Replace it
  with `remove_node` and `add_node`.
- `unresolved_ref`: a sub-screen or image that was not found. Pass the
  missing source in `siblings`, or fix the reference.
- `warning`: a converter note on a node. Check the node, then clear the note
  with `patch_node` and `set: {"warnings": []}`.
- `dropped_prop`: legacy properties the format could not carry. Tell the
  user about the ones that change what the screen does.

Converted drafts are saved even when they have findings, so run
`check` on each draft the result lists and take a screenshot. Running
`convert_legacy` again is safe. Replacing a draft that you have changed since
needs `overwrite=true` and `base_revs`.

## Screenshots and mock mode

`screenshot` renders the screen in the server's own viewer and returns the PNG
with `{width, height, pv}`. Pass `macros` to render a screen with other macro
values, and `theme` (`light` or `dark`) to pick the colour scheme.

`pv` is `connected` only when the viewer reached live PVs. Otherwise it is
`disconnected`.

A server started with `CANOPY_PV_MODE=mock` has no control system behind its
viewer. The viewer then draws every PV from a mock service. In mock mode:

- `screenshot` reports `pv: disconnected`. Blank or placeholder values in the
  image do not mean the screen is broken. Judge the layout, not the values.
- `read_pv` may fail and `search_pvs` and `check` may report `skipped`,
  because there is nothing to ask. Keep the PV names the user gave you and
  tell the user that they are not verified.

If `screenshot` answers that screenshots are unavailable, the server has no
browser to render with. Say so, show the user the screen as step 7 of
"Build a screen" says, and go on with `check`.

## Before you commit

This section applies when your screens root is a directory of a source tree
that carries a report, the files `LIBRARY-REPORT.json` and `LIBRARY-REPORT.md`
beside `screens/`. The report lists what the screens of the tree can write,
run and open, and a reviewer reads it. It must match the screens.

After your last change and before the commit, run:

```sh
python -m canopy_mcp.report <root> --write
```

`<root>` is the directory that holds `screens/`. Run the command in the
Python environment the server runs in. Then commit the changed files below
`screens/` together with both report files. If the command refuses, it names
the file and the reason. Fix that and run it again. Do not edit the report
files by hand.

`python -m canopy_mcp.report <root> --check` tells you whether the report
matches without writing anything.

Skip this section when your screens root is a private store that is not
committed.

## Push to the shared library

Use `request_push` when the server offers push and the user asks for a screen
to be shared. The call files a request. It publishes nothing. An administrator
reviews the request and accepts or rejects it outside this session.

- `id` is your draft. `target_dir` is the library directory it should land
  in, in id form, such as `sr/rf`. `note` tells the reviewer in plain words
  what the screen is and why it belongs in the library.
- The request holds a copy of the screen, of every screen it embeds or opens,
  and of every stored image they show. All of them must lie in `source_dir`,
  which defaults to the directory of `id`.
- Run `check` first and fix every error. A screen with errors is refused.

The result has a `rid`, the name of the request. Give the `rid` to the user.
The reviewer needs it.

A refusal comes back as `{refused: true, reason, message}` and files nothing.

- `push_unavailable`: this server takes no pushes. Tell the user. Do not
  retry. If your screens root is a source tree, the way to share a screen is
  the commit described above.
- `push_queue_full`: 8 requests already wait for review. Tell the user. Do
  not retry until the reviewer has worked through some of them.
- `read_only_store` or `library_source`: the id is a library screen. Push
  your own copy.
- `closure_refused`: `refusals` lists references and rules that would not
  mean the same in the library, such as an unresolved reference, a macro in a
  reference, or a reference that leaves `source_dir`. Fix them, or push from
  a directory further up.
- `grammar_refused`: `refusals` lists names and contents the library does not
  take, such as an SVG image, a file that is too large, or names that differ
  only in case.
- `check_errors`: `errors` lists the error findings of each pushed screen.

## When a call fails

- A refusal is an answer, not a fault. Read `reason` and `message`, fix the
  cause, and call again. Do not repeat a refused call unchanged.
- A tool error about an argument means the call was wrong: an id with a
  forbidden character, a missing draft, or two sources given to `check`.
- Never work around a refusal by writing files into the screens root
  yourself.
- Tell the user what you saved, under which id, and what is left: open
  warnings, unverified PV names, a `rid` waiting for review.
