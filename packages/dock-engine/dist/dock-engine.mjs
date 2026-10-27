// @als-apg/dock-engine: the dock layout engine, one dependency-free ESM file.
// Generated from packages/dock-engine/ts/ by packages/dock-engine/scripts/build.mjs.
// Do not edit; rebuild with `node packages/dock-engine/scripts/build.mjs`
// (`npm run build:dock-engine`).
//#region layout
/**
* The named reasons a layout is refused. The values are the cross-repo
* vocabulary of the C-CASES `validate` cases: canopy and OSPREY validators
* return the same strings.
*/
var LAYOUT_REASONS = {
	notLayout: "not_layout",
	unknownField: "unknown_field",
	badSplit: "bad_split",
	badTileId: "bad_tile_id",
	duplicateTile: "duplicate_tile",
	badKey: "bad_key",
	emptyTabs: "empty_tabs",
	duplicateKey: "duplicate_key",
	showingNotInTabs: "showing_not_in_tabs",
	splitTooFewItems: "split_too_few_items",
	badSize: "bad_size",
	unknownFrom: "unknown_from",
	oldSlotShape: "old_slot_shape",
	oldKeyShape: "old_key_shape",
	memberMissing: "member_missing"
};
/** Sibling fractions must sum to 1 within this. */
var SIZE_TOLERANCE = .001;
var SPLIT_FIELDS = /* @__PURE__ */ new Set([
	"split",
	"items",
	"size"
]);
var DOCK_TILE_FIELDS = /* @__PURE__ */ new Set([
	"tile",
	"tabs",
	"showing",
	"size",
	"from"
]);
var COLLECTION_TILE_FIELDS = /* @__PURE__ */ new Set([
	"tabs",
	"showing",
	"size"
]);
var TILE_ID = /^t(\d+)$/;
function isTile(node) {
	return !Object.hasOwn(node, "split");
}
var Refused = class extends Error {
	reason;
	where;
	constructor(reason, where) {
		super(`${where}: ${reason}`);
		this.reason = reason;
		this.where = where;
	}
};
function isPlainObject(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
/** The old leaf shapes are named before anything else, wherever they sit. */
function findOldShape(value, where) {
	if (Array.isArray(value)) {
		value.forEach((item, i) => findOldShape(item, `${where}[${i}]`));
		return;
	}
	if (!isPlainObject(value)) return;
	if (Object.hasOwn(value, "slot")) throw new Refused(LAYOUT_REASONS.oldSlotShape, where);
	if (Object.hasOwn(value, "key") && !Object.hasOwn(value, "tabs")) throw new Refused(LAYOUT_REASONS.oldKeyShape, where);
	if (Array.isArray(value.items)) findOldShape(value.items, `${where}.items`);
}
function checkKey(key, walk, where) {
	if (walk.form === "dock") {
		if (typeof key !== "string" || key === "") throw new Refused(LAYOUT_REASONS.badKey, where);
		return key;
	}
	const isIndex = typeof key === "number" && Number.isInteger(key) && key >= 0;
	if (walk.members !== void 0) {
		if (!isIndex || key >= walk.members) throw new Refused(LAYOUT_REASONS.badKey, where);
		return key;
	}
	if (isIndex || typeof key === "string" && key !== "") return key;
	throw new Refused(LAYOUT_REASONS.badKey, where);
}
function checkSizeValue(size, form, where) {
	if (size === void 0) return;
	if (!(typeof size === "number" && Number.isFinite(size) && size > 0 && (form === "collection" || size <= 1.001))) throw new Refused(LAYOUT_REASONS.badSize, where);
}
/** Dock form: given fractions sum to 1, or leave a remainder for the missing ones. */
function checkSiblingFractions(items, where) {
	let given = 0;
	let missing = 0;
	for (const item of items) if (typeof item.size === "number") given += item.size;
	else missing += 1;
	if (!(missing === 0 ? Math.abs(given - 1) <= .001 : given < .999)) throw new Refused(LAYOUT_REASONS.badSize, where);
}
function checkFields(node, allowed, where) {
	const unknown = Object.keys(node).filter((field) => !allowed.has(field)).sort();
	if (unknown.length > 0) throw new Refused(LAYOUT_REASONS.unknownField, `${where}.${unknown[0]}`);
}
function checkTile(node, walk, where) {
	checkFields(node, walk.form === "dock" ? DOCK_TILE_FIELDS : COLLECTION_TILE_FIELDS, where);
	let id = "";
	if (walk.form === "dock") {
		if (typeof node.tile !== "string" || node.tile === "") throw new Refused(LAYOUT_REASONS.badTileId, `${where}.tile`);
		id = node.tile;
		if (walk.tiles.has(id)) throw new Refused(LAYOUT_REASONS.duplicateTile, `${where}.tile`);
		walk.tiles.set(id, where);
	}
	if (!Array.isArray(node.tabs)) throw new Refused(LAYOUT_REASONS.emptyTabs, `${where}.tabs`);
	if (node.tabs.length === 0) throw new Refused(LAYOUT_REASONS.emptyTabs, `${where}.tabs`);
	const tabs = [];
	node.tabs.forEach((raw, i) => {
		const at = `${where}.tabs[${i}]`;
		const key = checkKey(raw, walk, at);
		if (walk.keys.has(key)) throw new Refused(LAYOUT_REASONS.duplicateKey, at);
		walk.keys.add(key);
		tabs.push(key);
	});
	const showingRequired = walk.form === "dock";
	if (node.showing !== void 0 || showingRequired) {
		if (!tabs.includes(node.showing)) throw new Refused(LAYOUT_REASONS.showingNotInTabs, `${where}.showing`);
	}
	checkSizeValue(node.size, walk.form, `${where}.size`);
	if (node.from !== void 0) {
		if (typeof node.from !== "string" || node.from === id) throw new Refused(LAYOUT_REASONS.unknownFrom, `${where}.from`);
		walk.froms.push({
			from: node.from,
			tile: id,
			where: `${where}.from`
		});
	}
}
function checkNode(value, walk, where) {
	if (!isPlainObject(value)) throw new Refused(LAYOUT_REASONS.notLayout, where);
	if (!(Object.hasOwn(value, "split") || Object.hasOwn(value, "items"))) {
		if (!(Object.hasOwn(value, "tabs") || Object.hasOwn(value, "tile"))) throw new Refused(LAYOUT_REASONS.notLayout, where);
		checkTile(value, walk, where);
		return;
	}
	checkFields(value, SPLIT_FIELDS, where);
	if (value.split !== "row" && value.split !== "column") throw new Refused(LAYOUT_REASONS.badSplit, `${where}.split`);
	const items = value.items;
	if (!Array.isArray(items) || items.length < 2) throw new Refused(LAYOUT_REASONS.splitTooFewItems, `${where}.items`);
	checkSizeValue(value.size, walk.form, `${where}.size`);
	items.forEach((item, i) => checkNode(item, walk, `${where}.items[${i}]`));
	if (walk.form === "dock") checkSiblingFractions(items, `${where}.items`);
}
/**
* Check a layout against C-LAYOUT. The first broken rule wins, in tree order;
* the result names it with a stable reason and the path where it broke.
*/
function validateLayout(input, options = {}) {
	const walk = {
		form: options.form ?? "dock",
		members: options.form === "collection" ? options.members : void 0,
		keys: /* @__PURE__ */ new Set(),
		tiles: /* @__PURE__ */ new Map(),
		froms: []
	};
	try {
		findOldShape(input, "layout");
		checkNode(input, walk, "layout");
		for (const { from, where } of walk.froms) if (!walk.tiles.has(from)) throw new Refused(LAYOUT_REASONS.unknownFrom, where);
		if (walk.members !== void 0) {
			for (let i = 0; i < walk.members; i += 1) if (!walk.keys.has(i)) throw new Refused(LAYOUT_REASONS.memberMissing, "layout");
		}
	} catch (error) {
		if (error instanceof Refused) return {
			valid: false,
			reason: error.reason,
			where: error.where
		};
		throw error;
	}
	return { valid: true };
}
function isAnySplit(node) {
	return Object.hasOwn(node, "split");
}
/** Dock form: missing sizes share the remainder; collection form: missing weight is 1. Then sum to 1. */
function fractionsOf$1(items, form) {
	const given = items.map((item) => typeof item.size === "number" && Number.isFinite(item.size) && item.size > 0 ? item.size : void 0);
	const missing = given.filter((size) => size === void 0).length;
	let fill = 1;
	if (form === "dock" && missing > 0) {
		const remainder = 1 - given.reduce((total, size) => total + (size ?? 0), 0);
		fill = remainder > 1e-9 ? remainder / missing : 1 / items.length;
	}
	const raw = given.map((size) => size ?? fill);
	const total = raw.reduce((sum, size) => sum + size, 0);
	return raw.map((size) => size / total);
}
function withSize$1(node, size) {
	const copy = { ...node };
	if (size === void 0) delete copy.size;
	else copy.size = size;
	return copy;
}
function normalizeNode(node, form, tiles) {
	if (!isAnySplit(node)) {
		const out = {
			...node,
			tabs: [...node.tabs]
		};
		if (out.showing === void 0) out.showing = out.tabs[0];
		if (out.from !== void 0 && (!tiles.has(out.from) || out.from === out.tile)) delete out.from;
		return out;
	}
	const fractions = fractionsOf$1(node.items, form);
	const items = [];
	node.items.forEach((item, i) => {
		const share = fractions[i] ?? 0;
		const child = normalizeNode(item, form, tiles);
		if (isAnySplit(child) && child.split === node.split) for (const grandchild of child.items) items.push(withSize$1(grandchild, share * (grandchild.size ?? 0)));
		else items.push(withSize$1(child, share));
	});
	if (items.length === 1 && items[0] !== void 0) return withSize$1(items[0], node.size);
	return withSize$1({
		split: node.split,
		items
	}, node.size);
}
function collectTileIds(node, out) {
	if (isAnySplit(node)) for (const item of node.items) collectTileIds(item, out);
	else if (typeof node.tile === "string") out.push(node.tile);
}
function normalizeAny(layout, form) {
	const ids = [];
	collectTileIds(layout, ids);
	return withSize$1(normalizeNode(layout, form, new Set(ids)), void 0);
}
/**
* The normalised form of a dock layout, as a fresh tree: same-direction splits
* flattened, one-item splits collapsed, every size filled in and siblings
* summing to 1, a `from` whose tile is gone cleared. The root has no size.
*/
function normalize(layout) {
	return normalizeAny(layout, "dock");
}
/**
* A collection layout read into fractions: weights normalised among siblings
* (a missing weight counts 1), every `showing` filled in (default `tabs[0]`),
* and the same flattening and collapsing as `normalize`.
*/
function normalizeCollection(layout) {
	return normalizeAny(layout, "collection");
}
/** Every tile id, in tree order. */
function tileIdsOf(layout) {
	const ids = [];
	if (layout !== null) collectTileIds(layout, ids);
	return ids;
}
/** Every key, in tree order (tiles in order, then their tabs in order). */
function keysOf(layout) {
	const keys = [];
	const visit = (node) => {
		if (isAnySplit(node)) node.items.forEach(visit);
		else keys.push(...node.tabs);
	};
	if (layout !== null) visit(layout);
	return keys;
}
/**
* The id a new tile gets: `t<k>`, k = 1 + the largest numeric suffix among the
* layout's `t<n>` ids (`t1` for an empty layout). Only the engine mints.
*/
function mintTileId(layout) {
	let largest = 0;
	for (const id of tileIdsOf(layout)) {
		const match = TILE_ID.exec(id);
		if (match?.[1] !== void 0) largest = Math.max(largest, Number(match[1]));
	}
	return `t${largest + 1}`;
}
//#endregion
//#region geometry
/** The axis a split of this direction lays its items along. */
function axisOf(direction) {
	return direction === "row" ? "width" : "height";
}
/**
* The smallest px a subtree can take along `axis` under dockview's minimum
* constraints, with `minPx` per tile.
*/
function minimumSize(node, axis, minPx) {
	if (isTile(node)) return minPx;
	const mins = node.items.map((item) => minimumSize(item, axis, minPx));
	return axisOf(node.split) === axis ? mins.reduce((a, b) => a + b, 0) : Math.max(...mins);
}
/**
* The fractions of a split's items: given sizes kept, missing sizes share the
* remainder equally, then scaled to sum to 1 (the C-LAYOUT reading of sizes).
*/
function itemFractions(items) {
	const given = items.filter((item) => typeof item.size === "number" && item.size > 0);
	const givenSum = given.reduce((sum, item) => sum + item.size, 0);
	const missing = items.length - given.length;
	const share = missing > 0 ? Math.max(0, 1 - givenSum) / missing : 0;
	const raw = items.map((item) => typeof item.size === "number" && item.size > 0 ? item.size : share);
	const total = raw.reduce((a, b) => a + b, 0);
	return total > 0 ? raw.map((r) => r / total) : items.map(() => 1 / items.length);
}
/**
* Lay `span` px out over items with these fractions and minimums, exactly as
* dockview's splitview does: round and clamp each, then hand the difference
* out from the last item to the first, each clamped to its minimum. When the
* minimums exceed the span the items stay at their minimums (overflow).
*/
function layoutSizes(fractions, span, minimums) {
	const sizes = fractions.map((f, i) => Math.max(Math.round(f * span), minimums[i] ?? 0));
	let delta = span - sizes.reduce((a, b) => a + b, 0);
	for (let i = sizes.length - 1; delta !== 0 && i >= 0; i--) {
		const next = Math.max(sizes[i] + delta, minimums[i] ?? 0);
		delta -= next - sizes[i];
		sizes[i] = next;
	}
	return sizes;
}
/**
* px sizes (as dockview reports them for one split's items) -> C-LAYOUT
* fractions. Each fraction is rounded to 4 decimals and the last absorbs the
* rounding, so the result sums to exactly 1. A zero or empty total gives
* equal shares.
*/
function fractionsOf(pxSizes) {
	const n = pxSizes.length;
	if (n === 0) return [];
	const total = pxSizes.reduce((a, b) => a + Math.max(0, b), 0);
	const rounded = (total > 0 ? pxSizes.map((px) => Math.max(0, px) / total) : pxSizes.map(() => 1 / n)).map((f) => Math.round(f * 1e4) / 1e4);
	const head = rounded.slice(0, -1).reduce((a, b) => a + b, 0);
	rounded[n - 1] = Math.round((1 - head) * 1e4) / 1e4;
	return rounded;
}
function tileIds(node, out = []) {
	if (isTile(node)) out.push(node.tile);
	else for (const item of node.items) tileIds(item, out);
	return out;
}
/**
* Each tile's rect and each interior border of `layout` laid out on
* `dockRect`, with dockview's `minPx` minimum on every tile.
*/
function geometry(layout, dockRect, minPx) {
	const tiles = {};
	const borders = [];
	const place = (node, rect, path) => {
		if (isTile(node)) {
			tiles[node.tile] = rect;
			return;
		}
		const axis = axisOf(node.split);
		const minimums = node.items.map((item) => minimumSize(item, axis, minPx));
		const span = axis === "width" ? rect.width : rect.height;
		const sizes = layoutSizes(itemFractions(node.items), span, minimums);
		let offset = axis === "width" ? rect.x : rect.y;
		node.items.forEach((item, i) => {
			const size = sizes[i];
			if (i > 0) {
				const line = axis === "width" ? {
					x1: offset,
					y1: rect.y,
					x2: offset,
					y2: rect.y + rect.height
				} : {
					x1: rect.x,
					y1: offset,
					x2: rect.x + rect.width,
					y2: offset
				};
				borders.push({
					split: node.split,
					path,
					index: i,
					before: tileIds(node.items[i - 1]),
					after: tileIds(item),
					line
				});
			}
			offset += size;
		});
		offset = axis === "width" ? rect.x : rect.y;
		node.items.forEach((item, i) => {
			const size = sizes[i];
			const child = axis === "width" ? {
				x: offset,
				y: rect.y,
				width: size,
				height: rect.height
			} : {
				x: rect.x,
				y: offset,
				width: rect.width,
				height: size
			};
			place(item, child, [...path, i]);
			offset += size;
		});
	};
	place(layout, { ...dockRect }, []);
	return {
		tiles,
		borders,
		feasible: dockRect.width >= minimumSize(layout, "width", minPx) && dockRect.height >= minimumSize(layout, "height", minPx)
	};
}
//#endregion
//#region drop
/** The direction a side's split runs in ('left'/'right': a row). */
function directionOf(side) {
	return side === "left" || side === "right" ? "row" : "column";
}
function isBefore(side) {
	return side === "left" || side === "top";
}
function nodeAt(layout, path) {
	let node = layout;
	for (const i of path) node = node.items[i];
	return node;
}
function pathOfTile(layout, id) {
	if (layout === null) return null;
	if (isTile(layout)) return layout.tile === id ? [] : null;
	for (let i = 0; i < layout.items.length; i++) {
		const sub = pathOfTile(layout.items[i], id);
		if (sub !== null) return [i, ...sub];
	}
	return null;
}
/** The tile with this id, or null. */
function findTile(layout, id) {
	const path = pathOfTile(layout, id);
	return path === null ? null : nodeAt(layout, path);
}
/** The tile holding this key, or null. */
function tileOf(layout, key) {
	if (layout === null) return null;
	if (isTile(layout)) return layout.tabs.includes(key) ? layout : null;
	for (const item of layout.items) {
		const found = tileOf(item, key);
		if (found !== null) return found;
	}
	return null;
}
/** Replace the node at `path` with `node` (a fresh spine; untouched subtrees are shared). */
function replaceAt(layout, path, node) {
	if (path.length === 0) return node;
	const split = layout;
	const [head, ...rest] = path;
	const items = split.items.map((item, i) => i === head ? replaceAt(item, rest, node) : item);
	return {
		...split,
		items
	};
}
function withSize(node, size) {
	const copy = { ...node };
	if (size === void 0) delete copy.size;
	else copy.size = size;
	return copy;
}
/** The rect of the node at `path` laid out on the dock, as geometry() lays it out. */
function rectAt(layout, path, dock) {
	let rect = {
		x: 0,
		y: 0,
		width: dock.px.width,
		height: dock.px.height
	};
	let node = layout;
	for (const i of path) {
		const split = node;
		const sizes = itemPx(split, rect, dock.minPx);
		const offset = sizes.slice(0, i).reduce((a, b) => a + b, 0);
		rect = split.split === "row" ? {
			x: rect.x + offset,
			y: rect.y,
			width: sizes[i],
			height: rect.height
		} : {
			x: rect.x,
			y: rect.y + offset,
			width: rect.width,
			height: sizes[i]
		};
		node = split.items[i];
	}
	return rect;
}
function itemPx(split, rect, minPx) {
	const axis = axisOf(split.split);
	const minimums = split.items.map((item) => minimumSize(item, axis, minPx));
	return layoutSizes(itemFractions(split.items), axis === "width" ? rect.width : rect.height, minimums);
}
function spanOf(rect, axis) {
	return axis === "width" ? rect.width : rect.height;
}
/** Set a split's item sizes from fractions (rounded to 4 decimals, summing to 1). */
function sized$1(split, fractions) {
	const rounded = fractionsOf(fractions);
	return {
		...split,
		items: split.items.map((item, i) => withSize(item, rounded[i]))
	};
}
/**
* Remove a tile per the C-LAYOUT leave rule: its space goes to the adjacent
* sibling subtree holding its `from` tile, else the previous sibling, else the
* next. The result is normalised (a one-item split collapses, a stale `from` is
* cleared). `neighbour` is the `from` tile when that subtree grew, else the
* adjacent tile of the sibling that did; `side` is the side of it the tile sat on.
*/
function removeTile(layout, id) {
	const path = pathOfTile(layout, id);
	if (layout === null || path === null) return {
		layout,
		neighbour: null,
		side: null
	};
	if (path.length === 0) return {
		layout: null,
		neighbour: null,
		side: null
	};
	const parentPath = path.slice(0, -1);
	const index = path[path.length - 1];
	const parent = nodeAt(layout, parentPath);
	const from = parent.items[index].from;
	const holds = (j) => from !== void 0 && parent.items[j] !== void 0 && tileIdsOf(parent.items[j]).includes(from);
	let donor;
	if (holds(index - 1)) donor = index - 1;
	else if (holds(index + 1)) donor = index + 1;
	else donor = index > 0 ? index - 1 : index + 1;
	const fractions = itemFractions(parent.items);
	fractions[donor] = fractions[donor] + fractions[index];
	fractions.splice(index, 1);
	const items = parent.items.filter((_, j) => j !== index);
	const next = replaceAt(layout, parentPath, sized$1({
		...parent,
		items
	}, fractions));
	const donorIds = tileIdsOf(parent.items[donor]);
	const neighbour = from !== void 0 && donorIds.includes(from) ? from : donor < index ? donorIds[donorIds.length - 1] : donorIds[0];
	const before = donor < index;
	const side = parent.split === "row" ? before ? "right" : "left" : before ? "bottom" : "top";
	return {
		layout: normalize(next),
		neighbour,
		side
	};
}
/**
* Remove a tab from the tile holding it. Its tile's last tab removes the tile
* (removeTile); a showing tab hands `showing` to the next tab, else the previous.
*/
function removeTab(layout, key) {
	const tile = tileOf(layout, key);
	if (layout === null || tile === null) return {
		layout,
		neighbour: null,
		side: null
	};
	if (tile.tabs.length === 1) return removeTile(layout, tile.tile);
	const index = tile.tabs.indexOf(key);
	const tabs = tile.tabs.filter((k) => k !== key);
	const showing = tile.showing === key ? tile.tabs[index + 1] ?? tile.tabs[index - 1] : tile.showing;
	return {
		layout: replaceAt(layout, pathOfTile(layout, tile.tile), {
			...tile,
			tabs,
			showing
		}),
		neighbour: null,
		side: null
	};
}
/**
* Split tile `id` at `edge`, putting `tile` on that side; the two halves share
* the tile's span equally. Clamped to `dock.minPx`: when a half would be
* smaller, both halves get minPx and the leave-rule neighbour of the target
* (along that axis) gives the difference. Null when that is infeasible or the
* tile is unknown. The caller sets `from` on `tile`.
*/
function splitTile(layout, id, edge, tile, dock) {
	const path = pathOfTile(layout, id);
	if (path === null) return null;
	const direction = directionOf(edge);
	const axis = axisOf(direction);
	const target = nodeAt(layout, path);
	let out = replaceAt(layout, path, {
		split: direction,
		items: isBefore(edge) ? [withSize(tile, .5), withSize(target, .5)] : [withSize(target, .5), withSize(tile, .5)],
		...sizeOf(target)
	});
	let x = path;
	while (x.length > 0 && nodeAt(layout, x.slice(0, -1)).split !== direction) x = x.slice(0, -1);
	const span = spanOf(rectAt(layout, path, dock), axis);
	const need = 2 * dock.minPx;
	if (Math.round(span / 2) >= dock.minPx) return normalize(out);
	if (x.length === 0) return null;
	const parentPath = x.slice(0, -1);
	const index = x[x.length - 1];
	const parent = nodeAt(layout, parentPath);
	const holds = (j) => target.from !== void 0 && parent.items[j] !== void 0 && tileIdsOf(parent.items[j]).includes(target.from);
	let donor;
	if (holds(index - 1)) donor = index - 1;
	else if (holds(index + 1)) donor = index + 1;
	else donor = index > 0 ? index - 1 : index + 1;
	const px = itemPx(parent, rectAt(layout, parentPath, dock), dock.minPx);
	const deficit = need - px[index];
	if (px[donor] - deficit < minimumSize(parent.items[donor], axis, dock.minPx)) return null;
	px[index] = px[index] + deficit;
	px[donor] = px[donor] - deficit;
	const resized = sized$1(nodeAt(out, parentPath), px);
	out = replaceAt(out, parentPath, resized);
	return normalize(out);
}
function sizeOf(node) {
	return node.size === void 0 ? {} : { size: node.size };
}
/**
* C-DROP border resolution: the nearest ancestor split along the side's axis
* in which the tile's subtree has a sibling on that side. The new item goes
* between those two siblings, at `index` of that split. Null at the dock edge.
*/
function resolveBorder(layout, border) {
	const path = pathOfTile(layout, border.tile);
	if (path === null) return null;
	const direction = directionOf(border.side);
	for (let depth = path.length - 1; depth >= 0; depth--) {
		const parentPath = path.slice(0, depth);
		const parent = nodeAt(layout, parentPath);
		if (parent.split !== direction) continue;
		const i = path[depth];
		if (isBefore(border.side) && i > 0) return {
			path: parentPath,
			index: i
		};
		if (!isBefore(border.side) && i + 1 < parent.items.length) return {
			path: parentPath,
			index: i + 1
		};
	}
	return null;
}
/**
* Insert `node` into the split at `slot.path` at `slot.index`, taking 1/n of
* it (n items after); the others keep their sizes relative to each other.
* `exclude` drops one direct item of that split first (an item moving within
* the same split keeps the ratios of before the drag). Null when a tile would
* get less than `dock.minPx` and its neighbour cannot give the difference.
*/
function insertAtBorder(layout, slot, node, dock, exclude) {
	const parent = nodeAt(layout, slot.path);
	let index = slot.index;
	const kept = parent.items.map((item, i) => ({
		item,
		i
	})).filter(({ i }) => i !== exclude);
	if (exclude !== void 0 && exclude < index) index -= 1;
	const n = kept.length + 1;
	const keptFractions = itemFractions(kept.map(({ item }) => item));
	const share = 1 / n;
	const items = kept.map(({ item }) => item);
	const fractions = keptFractions.map((f) => f * (1 - share));
	items.splice(index, 0, node);
	fractions.splice(index, 0, share);
	const span = spanOf(rectAt(layout, slot.path, dock), axisOf(parent.split));
	const clamped = clampShare(items, fractions, index, span, dock.minPx, axisOf(parent.split));
	if (clamped === null) return null;
	return normalize(replaceAt(layout, slot.path, sized$1({
		...parent,
		items
	}, clamped)));
}
/**
* The item at `index` gets at least minPx, its previous (else next) sibling
* giving the difference; null when any item would end below its minimum.
*/
function clampShare(items, fractions, index, span, minPx, axis) {
	const px = fractions.map((f) => f * span);
	if (px[index] < minPx) {
		const donor = index > 0 ? index - 1 : index + 1;
		const deficit = minPx - px[index];
		px[index] = minPx;
		px[donor] = px[donor] - deficit;
	}
	for (let i = 0; i < items.length; i++) if (px[i] + .5 < minimumSize(items[i], axis, minPx)) return null;
	return px.map((p) => p / span);
}
/**
* Put `node` along the dock's outer edge on `side`, spanning the full width or
* height and taking one third of the dock along that axis; the rest share two
* thirds in their current proportions. Null when that leaves the rest below
* their minimums.
*/
function insertOuter(layout, side, node, dock) {
	if (layout === null) return normalize(withSize(node, void 0));
	const direction = directionOf(side);
	const axis = axisOf(direction);
	const span = axis === "width" ? dock.px.width : dock.px.height;
	const items = !isTile(layout) && layout.split === direction ? [...layout.items] : [withSize(layout, void 0)];
	const fractions = itemFractions(items).map((f) => f * (2 / 3));
	const index = isBefore(side) ? 0 : items.length;
	items.splice(index, 0, node);
	fractions.splice(index, 0, 1 / 3);
	const clamped = clampShare(items, fractions, index, span, dock.minPx, axis);
	if (clamped === null) return null;
	return normalize(sized$1({
		split: direction,
		items
	}, clamped));
}
var MOVING = "\0moving";
function unchanged$1(layout) {
	return {
		layout,
		changed: false
	};
}
function result(before, after) {
	return sameLayout(before, after) ? unchanged$1(before) : {
		layout: after,
		changed: true
	};
}
function sameLayout(a, b) {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-6;
	if (a === null || b === null || typeof a !== "object" || typeof b !== "object") return a === b;
	if (Array.isArray(a) !== Array.isArray(b)) return false;
	const ka = Object.keys(a).filter((k) => a[k] !== void 0);
	const kb = Object.keys(b).filter((k) => b[k] !== void 0);
	if (ka.length !== kb.length) return false;
	return ka.every((k) => sameLayout(a[k], b[k]));
}
function resolveSource(layout, source) {
	if ("tile" in source) {
		const home = findTile(layout, source.tile);
		if (home === null) return null;
		return {
			tabs: [...home.tabs],
			showing: home.showing,
			home,
			wholeTile: true,
			keepsId: true
		};
	}
	const key = "tab" in source ? source.tab : source.new;
	const home = tileOf(layout, key);
	if (home === null) return "tab" in source ? null : {
		tabs: [key],
		showing: key,
		home: null,
		wholeTile: false,
		keepsId: false
	};
	return {
		tabs: [key],
		showing: key,
		home,
		wholeTile: home.tabs.length === 1,
		keepsId: false
	};
}
/** The layout with the moving item taken out (rule 5 for a whole tile). */
function takeOut(layout, moving) {
	if (moving.home === null) return layout;
	if (moving.wholeTile) return removeTile(layout, moving.home.tile).layout;
	return removeTab(layout, moving.tabs[0]).layout;
}
function newTile$1(before, moving, id) {
	return {
		tile: moving.keepsId && moving.home !== null ? moving.home.tile : id ?? mintTileId(before),
		tabs: [...moving.tabs],
		showing: moving.showing
	};
}
/**
* Plan a drop of `source` on `target`. The result's `changed` is false for a
* self-drop, an infeasible zone, or a source or target the layout does not hold.
*/
function planDrop(layout, source, target, dock) {
	const moving = resolveSource(layout, source);
	if (moving === null) return unchanged$1(layout);
	if ("outer" in target) {
		const node = newTile$1(layout, moving);
		const out = insertOuter(takeOut(layout, moving), target.outer, node, dock);
		return out === null ? unchanged$1(layout) : result(layout, out);
	}
	if (layout === null) return unchanged$1(layout);
	if ("border" in target) return dropOnBorder(layout, moving, target.border, dock);
	const into = findTile(layout, target.tile);
	if (into === null) return unchanged$1(layout);
	const self = moving.home !== null && moving.home.tile === into.tile;
	if ("edge" in target) {
		if (self && moving.wholeTile) return unchanged$1(layout);
		const node = {
			...newTile$1(layout, moving),
			from: into.tile
		};
		const rest = takeOut(layout, moving);
		if (rest === null) return unchanged$1(layout);
		const out = splitTile(rest, into.tile, target.edge, node, dock);
		return out === null ? unchanged$1(layout) : result(layout, out);
	}
	if (self && moving.wholeTile) return unchanged$1(layout);
	let index = target.zone === "tab" ? target.index : into.tabs.length;
	if (self) {
		const at = into.tabs.indexOf(moving.tabs[0]);
		if (index > at) index -= 1;
	}
	const rest = takeOut(layout, moving);
	const host = findTile(rest, into.tile);
	const tabs = [...host.tabs];
	tabs.splice(Math.max(0, Math.min(index, tabs.length)), 0, ...moving.tabs);
	return result(layout, normalize(replaceAt(rest, pathOfTile(rest, into.tile), {
		...host,
		tabs,
		showing: moving.showing
	})));
}
function dropOnBorder(layout, moving, border, dock) {
	const slot = resolveBorder(layout, border);
	if (slot === null) return unchanged$1(layout);
	const parent = nodeAt(layout, slot.path);
	const leaving = moving.wholeTile ? moving.home.tile : null;
	const isLeaving = (item) => leaving !== null && item !== void 0 && isTile(item) && item.tile === leaving;
	if (isLeaving(parent.items[slot.index - 1]) || isLeaving(parent.items[slot.index])) return unchanged$1(layout);
	const id = mintTileId(layout);
	let work = layout;
	if (moving.home !== null && !moving.wholeTile) work = removeTab(work, moving.tabs[0]).layout;
	const direct = parent.items.findIndex(isLeaving);
	const exclude = direct >= 0 ? direct : void 0;
	const node = newTile$1(layout, moving, id);
	const placed = moving.keepsId && exclude === void 0 ? {
		...node,
		tile: MOVING
	} : node;
	let out = insertAtBorder(work, slot, placed, dock, exclude);
	if (out === null) return unchanged$1(layout);
	if (leaving !== null && exclude === void 0) {
		out = removeTile(out, leaving).layout;
		if (placed.tile === MOVING) out = renameTile(out, MOVING, leaving);
	}
	return result(layout, out);
}
function renameTile(layout, from, to) {
	const path = pathOfTile(layout, from);
	return replaceAt(layout, path, {
		...nodeAt(layout, path),
		tile: to
	});
}
/**
* The new key takes the old key's place in the same tile at the same index;
* `showing` follows; no size changes (screen replace, back/forward).
*/
function replaceKey(layout, oldKey, newKey) {
	const tile = tileOf(layout, oldKey);
	if (layout === null || tile === null || oldKey === newKey) return unchanged$1(layout);
	const tabs = tile.tabs.map((k) => k === oldKey ? newKey : k);
	const showing = tile.showing === oldKey ? newKey : tile.showing;
	return {
		layout: replaceAt(layout, pathOfTile(layout, tile.tile), {
			...tile,
			tabs,
			showing
		}),
		changed: true
	};
}
//#endregion
//#region open
var TOO_MANY_FOR_SIDE = "too_many_for_side";
function sized(node, size) {
	const out = { ...node };
	delete out.size;
	return size === void 0 ? out : {
		...out,
		size
	};
}
/** A fresh tree with tile `id` replaced by `fn(tile)`. */
function replaceTile$1(layout, id, fn) {
	if (isTile(layout)) return layout.tile === id ? fn(layout) : layout;
	return {
		...layout,
		items: layout.items.map((item) => replaceTile$1(item, id, fn))
	};
}
/** The focused tile, else the first tile in tree order. */
function focusedOf(layout, focusedTile) {
	return (focusedTile === null ? null : findTile(layout, focusedTile)) ?? findTile(layout, tileIdsOf(layout)[0]);
}
function idSeries(layout, count) {
	const first = Number(mintTileId(layout).slice(1));
	return Array.from({ length: count }, (_, i) => `t${first + i}`);
}
function newTile(id, tabs) {
	return {
		tile: id,
		tabs,
		showing: tabs[0]
	};
}
/** Open one item by rule 7; `focus` names the tile that shows it. */
function openItem(layout, key, ctx) {
	if (layout === null) return {
		layout: newTile("t1", [key]),
		changed: true,
		focus: "t1"
	};
	const picked = ctx.pickTarget?.(layout, key) ?? null;
	if (picked !== null) {
		const out = planDrop(layout, { new: key }, picked, ctx.dock);
		const holder = out.changed ? tileOf(out.layout, key) : null;
		if (holder !== null) return {
			layout: out.layout,
			changed: true,
			focus: holder.tile
		};
	}
	const home = tileOf(layout, key);
	if (home !== null) {
		if (home.showing === key) return {
			layout,
			changed: false,
			focus: home.tile
		};
		return {
			layout: replaceTile$1(layout, home.tile, (tile) => ({
				...tile,
				showing: key
			})),
			changed: true,
			focus: home.tile
		};
	}
	const focused = focusedOf(layout, ctx.focusedTile);
	const out = planDrop(layout, { new: key }, {
		tile: focused.tile,
		zone: "centre"
	}, ctx.dock);
	return {
		layout: out.layout,
		changed: out.changed,
		focus: focused.tile
	};
}
function openAsTabs(layout, members, ctx) {
	if (layout === null) return {
		layout: newTile("t1", members),
		changed: true,
		focus: "t1"
	};
	const focused = focusedOf(layout, ctx.focusedTile);
	const id = mintTileId(layout);
	const split = splitTile(layout, focused.tile, "right", {
		...newTile(id, members),
		from: focused.tile
	}, ctx.dock);
	if (split !== null) return {
		layout: split,
		changed: true,
		focus: id
	};
	return {
		layout: replaceTile$1(layout, focused.tile, (tile) => ({
			...tile,
			tabs: [...tile.tabs, ...members],
			showing: members[0]
		})),
		changed: true,
		focus: focused.tile
	};
}
function rowOf(cells, size) {
	return sized({
		split: "row",
		items: cells.map((cell) => sized(cell, 1 / cells.length))
	}, size);
}
function openAsSide(layout, members, ctx) {
	const { width, height } = ctx.dock.px;
	const minPx = ctx.dock.minPx;
	if (layout === null) {
		const ids = idSeries(null, members.length);
		return {
			layout: normalize(rowOf(members.map((m, i) => newTile(ids[i], [m])), void 0)),
			changed: true,
			focus: ids[0]
		};
	}
	const focused = focusedOf(layout, ctx.focusedTile);
	const rect = geometry(layout, {
		x: 0,
		y: 0,
		width,
		height
	}, minPx).tiles[focused.tile];
	const n = members.length + 1;
	const ids = idSeries(layout, members.length);
	const cells = [sized(focused, void 0), ...members.map((m, i) => newTile(ids[i], [m]))];
	let block = null;
	if (rect.width / n >= minPx) block = rowOf(cells, focused.size);
	else {
		const cols = Math.ceil(Math.sqrt(n));
		const rows = Math.ceil(n / cols);
		if (rect.width / cols >= minPx && rect.height / rows >= minPx) {
			const items = [];
			for (let r = 0; r < rows; r++) items.push(rowOf(cells.slice(r * cols, (r + 1) * cols), 1 / rows));
			block = sized({
				split: "column",
				items
			}, focused.size);
		}
	}
	if (block === null) {
		if (ctx.caller === "person") return {
			...openAsTabs(layout, members, ctx),
			notice: TOO_MANY_FOR_SIDE
		};
		return {
			layout,
			changed: false,
			refusal: TOO_MANY_FOR_SIDE
		};
	}
	const placed = block;
	return {
		layout: normalize(replaceTile$1(layout, focused.tile, () => placed)),
		changed: true,
		focus: ids[0]
	};
}
/** The collection with skipped keys pruned; emptied tiles and splits dropped. */
function prune(node, skip) {
	if ("split" in node) {
		const items = node.items.map((item) => prune(item, skip)).filter((item) => item !== null);
		return items.length === 0 ? null : {
			...node,
			items
		};
	}
	const tabs = node.tabs.filter((key) => !skip.has(key));
	if (tabs.length === 0) return null;
	const { showing, ...rest } = node;
	return showing !== void 0 && tabs.includes(showing) ? {
		...rest,
		tabs,
		showing
	} : {
		...rest,
		tabs
	};
}
/** A normalised collection as a dock tree, tile ids minted from t1 in tree order. */
function toDock(node, next) {
	if ("split" in node) return sized({
		split: node.split,
		items: node.items.map((item) => toDock(item, next))
	}, node.size);
	return sized({
		tile: `t${next.n++}`,
		tabs: [...node.tabs],
		showing: node.showing
	}, node.size);
}
function openAsLayout(layout, collection, skipped) {
	const pruned = prune(collection, new Set(skipped));
	if (pruned === null) return {
		layout,
		changed: false,
		skipped
	};
	const dock = normalize(toDock(normalizeCollection(pruned), { n: 1 }));
	return {
		layout: dock,
		changed: true,
		focus: tileIdsOf(dock)[0],
		record: {
			kind: "layout",
			layout
		},
		skipped
	};
}
/**
* Open a set of members `as` tabs, side by side, or by its collection layout.
* Members in `ctx.skip` are left out and reported in `skipped`.
*/
function openSet(layout, members, as, collection, ctx) {
	const skip = new Set(ctx.skip ?? []);
	const skipped = members.filter((key) => skip.has(key));
	if (as === "layout" && collection !== void 0) return openAsLayout(layout, collection, skipped);
	const open = new Set(keysOf(layout));
	const fresh = members.filter((key) => !skip.has(key) && !open.has(key));
	const extra = skipped.length > 0 ? { skipped } : {};
	if (fresh.length === 0) return {
		layout,
		changed: false,
		...extra
	};
	return {
		...as === "side" ? openAsSide(layout, fresh, ctx) : openAsTabs(layout, fresh, ctx),
		...extra
	};
}
//#endregion
//#region close
/** Rounded to the 4 decimals sizes are stored at. */
function round4(value) {
	return Math.round(value * 1e4) / 1e4;
}
/** The split directly holding tile `id` and its index there, or null (root or unknown). */
function parentOf(layout, id) {
	if (isTile(layout)) return null;
	for (let i = 0; i < layout.items.length; i++) {
		const item = layout.items[i];
		if (isTile(item)) {
			if (item.tile === id) return {
				split: layout,
				index: i
			};
		} else {
			const found = parentOf(item, id);
			if (found !== null) return found;
		}
	}
	return null;
}
/** A fresh tree with `parent` (found by identity) replaced by `next`. */
function replaceSplit(layout, parent, next) {
	if (layout === parent) return next;
	if (isTile(layout)) return layout;
	return {
		...layout,
		items: layout.items.map((item) => replaceSplit(item, parent, next))
	};
}
function unchanged(layout) {
	return {
		layout,
		changed: false,
		record: null
	};
}
/** A fresh tree with tile `id` replaced by `fn(tile)`. */
function replaceTile(layout, id, fn) {
	if (isTile(layout)) return layout.tile === id ? fn(layout) : layout;
	return {
		...layout,
		items: layout.items.map((item) => replaceTile(item, id, fn))
	};
}
/** Drop `closing` from `tile`, showing `keep` when the showing tab goes. */
function closeWithin(layout, tile, keep, closing) {
	if (closing.size === 0) return unchanged(layout);
	const records = tile.tabs.flatMap((key, index) => closing.has(key) ? [{
		kind: "tab",
		key,
		tile: tile.tile,
		index,
		showing: tile.showing === key
	}] : []);
	return {
		layout: replaceTile(layout, tile.tile, (t) => ({
			...t,
			tabs: t.tabs.filter((key) => !closing.has(key)),
			showing: closing.has(t.showing) ? keep : t.showing
		})),
		changed: true,
		record: {
			kind: "tabs",
			records
		}
	};
}
/** Close one tab (rule 8); its tile's last tab closes the tile. */
function closeTab(layout, key) {
	const tile = tileOf(layout, key);
	if (layout === null || tile === null) return unchanged(layout);
	if (tile.tabs.length === 1) return closeTile(layout, tile.tile);
	const removal = removeTab(layout, key);
	const record = {
		kind: "tab",
		key,
		tile: tile.tile,
		index: tile.tabs.indexOf(key),
		showing: tile.showing === key
	};
	return {
		layout: removal.layout,
		changed: true,
		record
	};
}
/** Close a tile and every tab in it, by the leave rule. */
function closeTile(layout, id) {
	const tile = findTile(layout, id);
	if (layout === null || tile === null) return unchanged(layout);
	const removal = removeTile(layout, id);
	const parent = parentOf(layout, id);
	const record = {
		kind: "tile",
		tile: tile.tile,
		tabs: [...tile.tabs],
		showing: tile.showing,
		neighbour: removal.neighbour,
		side: removal.side
	};
	if (parent !== null) record.share = round4(itemFractions(parent.split.items)[parent.index]);
	return {
		layout: removal.layout,
		changed: true,
		record
	};
}
/** Close every other tab of the key's tile (rule 15). */
function closeOthers(layout, key) {
	const tile = tileOf(layout, key);
	if (layout === null || tile === null) return unchanged(layout);
	return closeWithin(layout, tile, key, new Set(tile.tabs.filter((k) => k !== key)));
}
/** Close the tabs right of the key in its tile (rule 15). */
function closeRight(layout, key) {
	const tile = tileOf(layout, key);
	if (layout === null || tile === null) return unchanged(layout);
	return closeWithin(layout, tile, key, new Set(tile.tabs.slice(tile.tabs.indexOf(key) + 1)));
}
/** Close the whole dock; the record holds it. */
function closeAll(layout) {
	if (layout === null) return unchanged(layout);
	return {
		layout: null,
		changed: true,
		record: {
			kind: "layout",
			layout
		}
	};
}
/** Insert `key` into tile `id` at `index` (clamped); shown when `show`. */
function insertTab(layout, id, key, index, show) {
	return replaceTile(layout, id, (t) => {
		const tabs = [...t.tabs];
		tabs.splice(Math.min(index, tabs.length), 0, key);
		return {
			...t,
			tabs,
			showing: show ? key : t.showing
		};
	});
}
/** Show `key` in the tile holding it. */
function show(layout, key) {
	return replaceTile(layout, tileOf(layout, key).tile, (t) => ({
		...t,
		showing: key
	}));
}
function reopenTab(layout, record, ctx) {
	const home = layout === null ? null : findTile(layout, record.tile);
	if (layout === null || home === null || keysOf(layout).includes(record.key)) {
		const out = openItem(layout, record.key, ctx);
		return {
			layout: out.layout,
			changed: out.changed,
			focus: out.focus ?? null
		};
	}
	return {
		layout: insertTab(layout, home.tile, record.key, record.index, true),
		changed: true,
		focus: home.tile
	};
}
function reopenTabs(layout, group, ctx) {
	let next = layout;
	let changed = false;
	let focus = null;
	const ordered = [...group.records].sort((a, b) => a.index - b.index);
	for (const record of ordered) {
		const home = next === null ? null : findTile(next, record.tile);
		if (next !== null && home !== null && !keysOf(next).includes(record.key)) {
			next = insertTab(next, home.tile, record.key, record.index, false);
			focus = home.tile;
			changed = true;
		} else {
			const out = openItem(next, record.key, ctx);
			next = out.layout;
			focus = out.focus ?? focus;
			changed = changed || out.changed;
		}
	}
	const shown = ordered.find((r) => r.showing);
	if (next !== null && shown !== void 0 && tileOf(next, shown.key) !== null) {
		next = show(next, shown.key);
		changed = true;
	}
	return {
		layout: next,
		changed,
		focus
	};
}
/**
* After an edge split put tile `id` beside `neighbour`, give `id` `share` of the
* split holding both, taken from the neighbour only. Clamped so each keeps
* `dock.minPx`; when the two together have less than twice that (the split
* already borrowed from elsewhere) the split is left as it is.
*/
function restoreShare(layout, id, neighbour, share, dock) {
	const held = parentOf(layout, id);
	const other = parentOf(layout, neighbour);
	if (held === null || other === null || held.split !== other.split) return layout;
	const { split, index } = held;
	const axis = axisOf(split.split);
	const rects = geometry(layout, {
		x: 0,
		y: 0,
		width: dock.px.width,
		height: dock.px.height
	}, dock.minPx).tiles;
	const pairPx = rects[id][axis] + rects[neighbour][axis];
	if (pairPx < 2 * dock.minPx) return layout;
	const fractions = itemFractions(split.items);
	const pair = fractions[index] + fractions[other.index];
	const wantPx = share / pair * pairPx;
	const mine = round4(Math.min(Math.max(wantPx, dock.minPx), pairPx - dock.minPx) / pairPx * pair);
	const items = split.items.map((item, i) => {
		if (i === index) return {
			...item,
			size: mine
		};
		if (i === other.index) return {
			...item,
			size: round4(pair - mine)
		};
		return item;
	});
	return replaceSplit(layout, split, {
		...split,
		items
	});
}
function reopenTile(layout, record, ctx) {
	const tabs = record.tabs.filter((key) => !keysOf(layout).includes(key));
	if (tabs.length === 0) return {
		layout,
		changed: false
	};
	const showing = tabs.includes(record.showing) ? record.showing : tabs[0];
	if (layout === null) return {
		layout: {
			tile: record.tile,
			tabs,
			showing
		},
		changed: true,
		focus: record.tile
	};
	const id = tileIdsOf(layout).includes(record.tile) ? mintTileId(layout) : record.tile;
	if (record.neighbour !== null && record.side !== null && findTile(layout, record.neighbour)) {
		const tile = {
			tile: id,
			tabs,
			showing,
			from: record.neighbour
		};
		const split = splitTile(layout, record.neighbour, record.side, tile, ctx.dock);
		if (split !== null) return {
			layout: typeof record.share === "number" && record.share > 0 && record.share < 1 ? restoreShare(split, id, record.neighbour, record.share, ctx.dock) : split,
			changed: true,
			focus: id
		};
	}
	let next = openItem(layout, tabs[0], ctx).layout;
	const holder = tileOf(next, tabs[0]);
	for (const key of tabs.slice(1)) next = insertTab(next, holder.tile, key, Number.POSITIVE_INFINITY, false);
	return {
		layout: show(next, showing),
		changed: true,
		focus: holder.tile
	};
}
/** Reopen a close record (rule 8), falling back to rule 7 through openItem. */
function reopen(layout, record, ctx) {
	switch (record.kind) {
		case "tab": return reopenTab(layout, record, ctx);
		case "tabs": return reopenTabs(layout, record, ctx);
		case "tile": return reopenTile(layout, record, ctx);
		case "layout": return {
			layout: record.layout,
			changed: JSON.stringify(record.layout) !== JSON.stringify(layout)
		};
	}
}
//#endregion
//#region hitTest
/** The tile floor every dock enforces. */
var MIN_PX = 40;
/** Smallest edge depth; larger tiles use EDGE_RATIO of their content. */
var EDGE_PX = 80;
/** Edge depth as a share of the tile content along that axis. */
var EDGE_RATIO = .31;
/** Depth of the dock-edge strips that mean an outer drop. */
var STRIP_PX = 48;
function inside(r, p) {
	return p.x >= r.x && p.x < r.x + r.width && p.y >= r.y && p.y < r.y + r.height;
}
function insideClosed(r, p) {
	return p.x >= r.x && p.x <= r.x + r.width && p.y >= r.y && p.y <= r.y + r.height;
}
function findTile$1(layout, id) {
	if (isTile(layout)) return layout.tile === id ? layout : null;
	for (const item of layout.items) {
		const found = findTile$1(item, id);
		if (found) return found;
	}
	return null;
}
function tileOfKey(layout, key) {
	if (isTile(layout)) return layout.tabs.includes(key) ? layout : null;
	for (const item of layout.items) {
		const found = tileOfKey(item, key);
		if (found) return found;
	}
	return null;
}
/**
* The tile a source moves as a whole: a `{tile}` source, or a tab alone in its
* tile (`{new}` of an already open key acts as `{tab}`). Null when the source
* moves a single tab out of a multi-tab tile, or is genuinely new.
*/
function movingTile(layout, source) {
	if ("tile" in source) return findTile$1(layout, source.tile) ? source.tile : null;
	const tile = tileOfKey(layout, "tab" in source ? source.tab : source.new);
	return tile && tile.tabs.length === 1 ? tile.tile : null;
}
/** The tile's content: its rect minus its bar when the bar lies inside it. */
function contentOf(rect, bars) {
	if (!bars) return rect;
	const bar = bars.barRect;
	const barBottom = bar.y + bar.height;
	if (bar.y < rect.y - 1 || bar.y >= rect.y + rect.height || barBottom <= rect.y) return rect;
	const top = Math.min(barBottom, rect.y + rect.height);
	return {
		x: rect.x,
		y: top,
		width: rect.width,
		height: rect.y + rect.height - top
	};
}
/** Halving `span` px leaves each half at least MIN_PX. */
function halvable(span) {
	return Math.round(span / 2) >= 40;
}
/** The tab gap at `x`: the count of tabs whose centre lies left of it. */
function tabIndex(tabRects, x) {
	return tabRects.filter((r) => r.x + r.width / 2 < x).length;
}
function barHit(layout, tabRects, point) {
	for (const id of tileIdsOf(layout)) {
		const bars = tabRects[id];
		if (!bars || !inside(bars.barRect, point)) continue;
		return {
			tile: id,
			zone: "tab",
			index: tabIndex(bars.tabRects, point.x)
		};
	}
	return null;
}
/** The edge band depth along a content span: max(EDGE_PX, EDGE_RATIO), at most half. */
function depthOf(span) {
	return Math.min(Math.max(80, span * EDGE_RATIO), span / 2);
}
function edgeHit(rect, content, point) {
	if (halvable(rect.width)) {
		const depth = depthOf(content.width);
		if (point.x - content.x < depth) return "left";
		if (point.x - content.x > content.width - depth) return "right";
	}
	if (halvable(rect.height)) {
		const depth = depthOf(content.height);
		if (point.y - content.y < depth) return "top";
		if (point.y - content.y > content.height - depth) return "bottom";
	}
	return null;
}
/** The dock-edge strip under `point` in tile `rect`, or null; see rule 3. */
function stripHit(dockRect, rect, content, point) {
	if (point.x < dockRect.x + 48) return "left";
	if (point.x > dockRect.x + dockRect.width - 48) return "right";
	if (point.y > dockRect.y + dockRect.height - 48) return "bottom";
	if (rect.y <= dockRect.y + 1 && point.y < content.y + 48) return "top";
	return null;
}
/**
* The Target a drop of `source` at `point` would mean, or null: off the dock
* or a self-drop. Tile rects come from `geometry` on `dockRect` with the 40 px
* floor; bars and tabs from the dock's `tabRects`.
*/
function hitTest(layout, dockRect, tabRects, point, source) {
	const self = movingTile(layout, source);
	const unlessSelf = (target) => "tile" in target && target.tile === self ? null : target;
	const bar = barHit(layout, tabRects, point);
	if (bar) return unlessSelf(bar);
	if (!insideClosed(dockRect, point)) return null;
	const g = geometry(layout, dockRect, 40);
	const id = Object.keys(g.tiles).find((t) => inside(g.tiles[t], point)) ?? Object.keys(g.tiles).find((t) => insideClosed(g.tiles[t], point));
	if (id === void 0) return null;
	const rect = g.tiles[id];
	const content = contentOf(rect, tabRects[id]);
	if (point.y < content.y) return unlessSelf({
		tile: id,
		zone: "tab",
		index: tabIndex(tabRects[id]?.tabRects ?? [], point.x)
	});
	const strip = stripHit(dockRect, rect, content, point);
	if (strip) return self !== null && isTile(layout) ? null : { outer: strip };
	const edge = edgeHit(rect, content, point);
	return unlessSelf(edge ? {
		tile: id,
		edge
	} : {
		tile: id,
		zone: "centre"
	});
}
//#endregion
//#region resultRect
/** The insertion gap at `index` in a tile's measured bar, or null with no bar. */
function markerAt(tabRects, tile, index) {
	const bar = tabRects[tile];
	if (bar === void 0) return null;
	const tabs = bar.tabRects;
	const at = index === null ? tabs.length : Math.max(0, Math.min(index, tabs.length));
	let x;
	if (at < tabs.length) x = tabs[at].x;
	else if (tabs.length > 0) x = tabs[tabs.length - 1].x + tabs[tabs.length - 1].width;
	else x = bar.barRect.x;
	return {
		x,
		y: bar.barRect.y,
		height: bar.barRect.height
	};
}
/** A key no layout holds: the stand-in whose landing the edge highlight shows. */
var PROBE = "\0highlight";
/**
* The `edge` half of tile `id` as an edge split lays it out: the landed rect
* of a new item dropped there, so a split that flattens into its parent rounds
* the same way. Null when the tile's span along that axis cannot be halved.
*/
function halfOf(layout, dockRect, dock, id, edge) {
	const rect = geometry(layout, dockRect, 40).tiles[id];
	if (!halvable(edge === "left" || edge === "right" ? rect.width : rect.height)) return null;
	const after = planDrop(layout, { new: PROBE }, {
		tile: id,
		edge
	}, dock).layout;
	const landed = tileOf(after, PROBE);
	return landed === null ? null : { ...geometry(after, dockRect, 40).tiles[landed.tile] };
}
/** The key or tile id whose holder the highlight follows. */
function holderOf(after, source) {
	if ("tile" in source) return source.tile;
	return tileOf(after, "tab" in source ? source.tab : source.new)?.tile ?? null;
}
/** The new strip an outer drop of `source` lands in, or null when it would not land. */
function stripOf(layout, dockRect, dock, source, side) {
	const result = planDrop(layout, source, { outer: side }, dock);
	if (!result.changed || result.layout === null) return null;
	const holder = holderOf(result.layout, source);
	const rect = holder === null ? void 0 : geometry(result.layout, dockRect, 40).tiles[holder];
	return rect === void 0 ? null : { ...rect };
}
/**
* The part of the target tile the dragged item lands in after dropping
* `source` on `target`, with the tab insertion marker for centre and tab
* targets, or the new strip of an outer drop; null when the drop would not
* land or has no pointer zone.
*/
function resultRect(layout, dockRect, tabRects, source, target) {
	const dock = {
		px: {
			width: dockRect.width,
			height: dockRect.height
		},
		minPx: 40
	};
	if ("outer" in target) {
		const strip = stripOf(layout, dockRect, dock, source, target.outer);
		return strip === null ? null : {
			rect: strip,
			marker: null
		};
	}
	if (!("tile" in target)) return null;
	const rect = geometry(layout, dockRect, 40).tiles[target.tile];
	if (rect === void 0) return null;
	if ("zone" in target) {
		const index = target.zone === "tab" ? target.index : null;
		return {
			rect: { ...rect },
			marker: markerAt(tabRects, target.tile, index)
		};
	}
	if (!planDrop(layout, source, target, dock).changed) return null;
	const half = halfOf(layout, dockRect, dock, target.tile, target.edge);
	return half === null ? null : {
		rect: half,
		marker: null
	};
}
//#endregion
export { EDGE_PX, EDGE_RATIO, LAYOUT_REASONS, MIN_PX, SIZE_TOLERANCE, STRIP_PX, TOO_MANY_FOR_SIDE, axisOf, closeAll, closeOthers, closeRight, closeTab, closeTile, directionOf, findTile, fractionsOf, geometry, halvable, hitTest, insertAtBorder, insertOuter, isTile, itemFractions, keysOf, layoutSizes, minimumSize, mintTileId, normalize, normalizeCollection, openItem, openSet, planDrop, removeTab, removeTile, reopen, replaceKey, resolveBorder, resultRect, splitTile, tileIdsOf, tileOf, validateLayout };
