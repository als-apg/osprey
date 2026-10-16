"""ARIEL text embedding module.

This module generates text embeddings for logbook entries to enable semantic search.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.database.embedding_input_key_migration import (
    INPUT_KEY_COLUMN_SQL,
)
from osprey.services.ariel_search.database.migrations import model_to_table_name
from osprey.services.ariel_search.enhancement._offload import run_blocking
from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule, HealthResult
from osprey.services.ariel_search.enhancement.provider_resolver import (
    ResolvedProvider,
    resolve_provider,
    resolve_reachable_base_url,
)
from osprey.services.ariel_search.enhancement.text_embedding.migration import (
    TextEmbeddingMigration,
)
from osprey.services.ariel_search.vocabulary.model import match_tokens
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

    from osprey.models.providers.base import BaseProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.migrations import BaseMigration
    from osprey.services.ariel_search.models import EnhancedLogbookEntry
    from osprey.services.ariel_search.vocabulary.entry import EntryExpander
    from osprey.services.ariel_search.vocabulary.expand import EntryExpansion, Insertion
    from osprey.services.ariel_search.vocabulary.model import Token

logger = get_logger("ariel")

#: Input limit, in tokens, used for a model whose entry names no ``max_input_tokens``. It is
#: the smallest window among the models the Ollama provider lists (``all-minilm`` and
#: ``mxbai-embed-large`` serve 512), so an unstated limit is never larger than the model's.
DEFAULT_MAX_INPUT_TOKENS = 512

#: Config key the module's provider comes from when ``configure`` is given none.
_PROVIDER_KEY = "ariel.enhancement_modules.text_embedding.provider"

#: Tokens held back from a model's input limit for the special tokens a tokenizer wraps every
#: input in (``nomic-embed-text`` adds two).
RESERVED_TOKENS = 8

#: Embedding tables already reported as lacking an ``input_key`` column in this process.
_INPUT_KEY_ABSENT_LOGGED: set[str] = set()


def max_input_tokens(model_config: Mapping[str, Any]) -> int:
    """Return the input limit, in tokens, of one configured embedding model.

    Args:
        model_config: One entry of ``ariel.enhancement_modules.text_embedding.models``.

    Returns:
        The entry's ``max_input_tokens``, or ``DEFAULT_MAX_INPUT_TOKENS`` when it is absent or
        null.

    Raises:
        ValueError: If the limit is not an integer greater than ``RESERVED_TOKENS``.
    """
    value = model_config.get("max_input_tokens")
    if value is None:
        return DEFAULT_MAX_INPUT_TOKENS
    if isinstance(value, int) and not isinstance(value, bool) and value > RESERVED_TOKENS:
        return value
    raise ValueError(
        "ariel.enhancement_modules.text_embedding.models: max_input_tokens for "
        f"{model_config.get('name')!r} must be an integer greater than {RESERVED_TOKENS} "
        f"(got {value!r})"
    )


def fit_to_input_limit(text: str, max_input_tokens: int) -> str:
    """Cut a text so it fits an embedding model's input limit, keeping its start.

    The returned text is at most ``max_input_tokens - RESERVED_TOKENS`` UTF-8 bytes long. A
    tokenizer emits at most one token per character (WordPiece, measured on
    ``nomic-embed-text`` with punctuation, digits, CJK, accented and compatibility characters)
    or one per byte (byte-level BPE), so a cut in UTF-8 bytes fits every text, whereas a
    characters-per-token ratio fits only the text it was measured on.

    Args:
        text: The text to embed.
        max_input_tokens: The model's input limit, in tokens.

    Returns:
        ``text`` itself when it fits, otherwise its longest prefix that fits. A character the
        cut would split is dropped whole.
    """
    return _utf8_prefix(text, max_input_tokens - RESERVED_TOKENS)


#: Version of the rule that turns an entry into its expander-less embedding input. It
#: changes only when the bytes embedded without an expander change.
INPUT_RULE = 1


def running_input_key(expander: EntryExpander | None) -> str:
    """Return the build key of vectors embedded now: the input rule plus the expander's key.

    Args:
        expander: The entry expander in use, or ``None`` when entries are embedded unexpanded.

    Returns:
        ``r<INPUT_RULE>``, followed by ``:<expander.key>`` when an expander is given.
    """
    key = f"r{INPUT_RULE}"
    if expander is not None:
        key += f":{expander.key}"
    return key


@dataclass(frozen=True)
class EmbeddingInput:
    """The text an entry is embedded from.

    Attributes:
        text: The embedding input.
        kept_chars: How many code points of the entry's own text ``text`` contains; inserted
            terms and picture text are not counted.
    """

    text: str
    kept_chars: int


def embedding_input(raw_text: str, attachment_text: str | None, max_input_tokens: int) -> str:
    """Build the text an entry is embedded from, without entry expansion.

    Args:
        raw_text: The entry's text.
        attachment_text: The entry's composed picture text, or ``None``.
        max_input_tokens: The model's input limit, in tokens.

    Returns:
        ``build_embedding_input(raw_text, attachment_text, max_input_tokens).text`` — ``raw_text``
        itself when it has no picture text and fits.
    """
    return build_embedding_input(raw_text, attachment_text, max_input_tokens).text


def build_embedding_input(
    raw_text: str,
    attachment_text: str | None,
    max_input_tokens: int,
    *,
    expander: EntryExpander | None = None,
) -> EmbeddingInput:
    """Build the text an entry is embedded from: its own text, then its picture text.

    The whole input stays within ``max_input_tokens - RESERVED_TOKENS`` UTF-8 bytes. With no
    picture text the entry gets the whole budget; otherwise the picture text, cut to a quarter
    of the budget, follows a newline and the entry gets the room left.

    Without an expander, or when the entry's text cut to its room has no term to insert, the
    entry part is that byte cut, exactly as embedded without expansion. Otherwise the entry's
    text is cut at a whitespace word end and its inserted terms follow their words: an entry
    that fits its room whole is never cut, and one that does not keeps at least the shortest
    whitespace-aligned prefix of three quarters of the room. When no word end lies between
    three quarters of the room and the room, the byte cut at the room is final and only terms
    anchored inside it are inserted. Terms are inserted in text order while they fit; the rest
    are dropped whole.

    Args:
        raw_text: The entry's text.
        attachment_text: The entry's composed picture text, or ``None``.
        max_input_tokens: The model's input limit, in tokens.
        expander: Inserts vocabulary terms into the entry's text, or ``None``.

    Returns:
        The embedding input and how many code points of ``raw_text`` it keeps. A character a
        byte cut would split is dropped whole. ``text`` is ``raw_text`` itself when nothing
        is cut or added.
    """
    budget = max_input_tokens - RESERVED_TOKENS
    att = attachment_text or ""
    if att.strip():
        att = _utf8_prefix(att, budget // 4)
        room = budget - len(att.encode("utf-8")) - 1
        suffix = "\n" + att
    else:
        room = budget
        suffix = ""

    raw_room = _utf8_prefix(raw_text, room)
    entry, kept = raw_room, len(raw_room)
    if expander is not None:
        tokens = match_tokens(raw_text)
        scanned = expander.expand(raw_text, limit=len(raw_room), tokens=tokens)
        if scanned.insertions:
            floor = _expansion_floor(raw_text, raw_room, room)
            cut, expansion = _fit_expanded(
                raw_text, tokens, expander, room=room, floor=floor, scanned=scanned
            )
            entry, kept = _render_expanded(raw_text, cut, expansion.insertions, room), cut
    return EmbeddingInput(text=entry + suffix if suffix else entry, kept_chars=kept)


def _expansion_floor(raw: str, raw_room: str, room: int) -> int:
    """Return the code-point offset an expanded entry is never cut below.

    Args:
        raw: The entry's text.
        raw_room: ``_utf8_prefix(raw, room)``.
        room: The UTF-8 byte budget for the entry's text and its insertions.

    Returns:
        ``len(raw)`` when the entry fits the room whole; otherwise the shortest
        whitespace-aligned prefix of at least three quarters of the room, or the byte cut at
        the room when no word end lies between the two.
    """
    if raw_room is raw:
        return len(raw)
    least = room * 3 // 4
    pos = size = 0
    for match in _WORD.finditer(raw):
        end = match.end()
        if end > len(raw_room):
            break
        size += len(raw[pos:end].encode("utf-8"))
        pos = end
        if size >= least:
            return end
    return len(raw_room)


def _render_expanded(raw: str, cut: int, insertions: tuple[Insertion, ...], room: int) -> str:
    """Render ``raw[:cut]`` with its insertions, kept in text order while they fit the room.

    Args:
        raw: The entry's text.
        cut: Code-point offset the entry's text is cut at.
        insertions: The insertions of a scan bounded by ``cut``, in text order.
        room: The UTF-8 byte budget for the entry's text and its insertions.

    Returns:
        The rendered text; ``raw[:cut]`` with the first insertion that overflows the room and
        every later one dropped whole.
    """
    used = len(raw[:cut].encode("utf-8"))
    parts: list[str] = []
    pos = 0
    for insertion in insertions:
        if insertion.anchor > cut:
            break
        size = _insertion_bytes(insertion)
        if used + size > room:
            break
        used += size
        parts.append(raw[pos : insertion.anchor])
        parts.append(" " + insertion.text)
        pos = insertion.anchor
    if not parts:
        return raw[:cut]
    parts.append(raw[pos:cut])
    return "".join(parts)


def _utf8_prefix(text: str, budget: int) -> str:
    """Return the longest prefix of ``text`` that is at most ``budget`` UTF-8 bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    return encoded[: max(budget, 0)].decode("utf-8", errors="ignore")


#: A whitespace-delimited word; its ``end()`` is a word end a cut may sit on.
_WORD = re.compile(r"\S+")

#: Re-scans that confirm a walked cut, each moving the cut one word back at most.
_CONFIRM_SCANS = 3


def _insertion_bytes(insertion: Insertion) -> int:
    """Return the UTF-8 bytes one insertion adds to the text: its separating space and text."""
    return len((" " + insertion.text).encode("utf-8"))


def _expanded_bytes(raw: str, cut: int, insertions: tuple[Insertion, ...]) -> int:
    """Return the UTF-8 bytes of ``raw[:cut]`` plus every insertion anchored at or before it."""
    added = sum(_insertion_bytes(ins) for ins in insertions if ins.anchor <= cut)
    return len(raw[:cut].encode("utf-8")) + added


def _fit_expanded(
    raw: str,
    tokens: list[Token],
    expander: EntryExpander,
    *,
    room: int,
    floor: int,
    scanned: EntryExpansion | None = None,
) -> tuple[int, EntryExpansion]:
    """Pick where an entry is cut so its text plus its inserted terms fits a byte room.

    The cut is a code-point offset into ``raw``. Candidates are the whitespace word ends
    from the room end back to ``floor``, and ``floor`` itself. One walk over the insertions
    of a scan bounded by the room picks the largest candidate whose raw bytes plus
    ``len(" " + text)`` bytes per insertion anchored at or before it fit ``room``; when
    none does, the cut is ``floor``. The cut is then confirmed by at most three re-scans
    with ``limit = cut``: a re-scan whose whole expansion fits ends the search; otherwise
    the cut moves to the previous word end while it lies above ``floor``. The cut never
    goes below ``floor``, so an entry the caller had to cut keeps at least ``floor``
    of its text and its insertions take at most ``room`` minus those bytes.

    The returned expansion is that of the last re-scan. It may still overflow the room
    at ``floor``; the caller keeps its insertions in text order while they fit and drops
    the rest whole, never cutting ``raw`` below the returned cut.

    Args:
        raw: The entry's text.
        tokens: ``match_tokens(raw)``, shared by every scan.
        expander: Scans ``raw`` as ``expander.expand(raw, limit=cut, tokens=tokens)``.
        room: The UTF-8 byte budget for the entry's text and its insertions.
        floor: Code-point offset the cut never goes below; ``len(raw)`` for an entry
            whose raw text fits the room whole.
        scanned: The scan of ``raw`` bounded by the room, when the caller already ran
            it; otherwise it is run here.

    Returns:
        The cut and the expansion of the last re-scan.
    """
    room_end = max(len(_utf8_prefix(raw, room)), floor)
    if scanned is None:
        scanned = expander.expand(raw, limit=room_end, tokens=tokens)
    word_ends = [m.end() for m in _WORD.finditer(raw) if floor < m.end() <= room_end]

    cut = floor
    if word_ends:
        insertions = scanned.insertions
        kept = len(insertions)
        added = sum(_insertion_bytes(ins) for ins in insertions)
        for end in reversed(word_ends):
            while kept and insertions[kept - 1].anchor > end:
                kept -= 1
                added -= _insertion_bytes(insertions[kept])
            if len(raw[:end].encode("utf-8")) + added <= room:
                cut = end
                break

    expansion = scanned
    for _ in range(_CONFIRM_SCANS):
        expansion = expander.expand(raw, limit=cut, tokens=tokens)
        if _expanded_bytes(raw, cut, expansion.insertions) <= room:
            break
        if cut <= floor:
            break
        previous = [end for end in word_ends if end < cut]
        cut = max(previous[-1], floor) if previous else floor
    return cut, expansion


class TextEmbeddingModule(BaseEnhancementModule):
    """Generate text embeddings for logbook entries.

    Supports multiple embedding models, each with its own dedicated table.
    The provider name references api.providers for api_key and base_url.
    """

    def __init__(self) -> None:
        """Initialize the module."""
        self._provider: BaseProvider | None = None
        self._resolved: ResolvedProvider | None = None
        self._reachable_base_url: str | None = None
        self._models: list[dict[str, Any]] = []
        self._provider_key: str = _PROVIDER_KEY
        self._tables_exist: bool | None = None  # Cached result of table existence check
        self._expander: EntryExpander | None = None
        self._input_key_columns: dict[str, bool] = {}

    @property
    def name(self) -> str:
        """Return module identifier."""
        return "text_embedding"

    @property
    def migration(self) -> type[BaseMigration]:
        """Return migration class for this module."""
        return TextEmbeddingMigration

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the module with settings from config.yml.

        Resolves the provider through :func:`resolve_provider` (default
        ``ollama``) without touching the network: a server that is down still
        configures, and :meth:`health_check` reports it.

        Args:
            config: The enhancement_modules.text_embedding config dict
                   containing 'provider' (provider name string or inline config dict),
                   'provider_key' (the config key the provider came from) and
                   'models' list.

        Raises:
            ValueError: If a model's ``max_input_tokens`` is malformed.
            ModuleConfigError: If the provider is unknown, serves no embeddings,
                or its adapter refuses the base URL (the message names
                ``provider_key``).
        """
        self._models = config.get("models", [])
        for model_config in self._models:
            max_input_tokens(model_config)
        self._provider_key = config.get("provider_key") or _PROVIDER_KEY
        self._provider = None
        self._reachable_base_url = None
        self._resolved = None
        self._resolved = self._resolve(config.get("provider"))

    def bind(self, config: ARIELConfig) -> None:
        """Bind the entry expander the module's inputs are built with.

        Args:
            config: The parsed ARIEL configuration; an unusable vocabulary leaves entries
                embedded unexpanded.
        """
        self._expander = config.entry_expander_or_none()

    def _resolve(self, provider: str | dict[str, Any] | None) -> ResolvedProvider:
        """Resolve the module's provider (default ``ollama``) under its config key."""
        return resolve_provider(provider, provider_key=self._provider_key, default="ollama")

    def _resolution(self) -> ResolvedProvider:
        """The provider resolved in ``configure`` (the default resolved now if it was not)."""
        if self._resolved is None:
            self._resolved = self._resolve(None)
        return self._resolved

    def _get_provider(self) -> BaseProvider:
        """Return the embedding provider adapter, instantiated once.

        Returns:
            The adapter instance of the class resolved in ``configure``.
        """
        if self._provider is None:
            self._provider = self._resolution().instance

        return self._provider

    def _resolved_class(self) -> type[BaseProvider]:
        """The provider class resolved in ``configure``."""
        return self._resolution().cls

    def _base_url(self, *, refresh: bool = False) -> str | None:
        """The base URL embedding calls use.

        The configured URL, swapped for the reachable one for a provider that
        resolves its fallback outside calls (blocking on the first call, then
        cached; ``refresh`` re-checks it).
        """
        resolved = self._resolution()
        if resolved.base_url is None:
            return None
        if self._reachable_base_url is None or refresh:
            self._reachable_base_url = resolve_reachable_base_url(
                resolved.cls, resolved.base_url, refresh=refresh
            )
        return self._reachable_base_url

    async def _check_tables_exist(self, conn: AsyncConnection) -> bool:
        """Check if embedding tables exist (cached after first call).

        Returns:
            True if at least one configured model's table exists
        """
        if self._tables_exist is not None:
            return self._tables_exist

        for model_config in self._models:
            table_name = model_to_table_name(model_config["name"])
            result = await conn.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                [table_name],
            )
            row = await result.fetchone()
            if row and row[0]:
                self._tables_exist = True
                return True

        logger.warning(
            "Embedding tables do not exist (pgvector migration was skipped). "
            "Skipping text embedding enhancement."
        )
        self._tables_exist = False
        return False

    async def _has_input_key_column(self, table_name: str, conn: AsyncConnection) -> bool:
        """Report whether ``table_name`` has an ``input_key`` column (probed once per table).

        A table without the column is logged once per process.

        Args:
            table_name: The model's embedding table.
            conn: Database connection the probe runs on.

        Returns:
            True when the column exists.
        """
        present = self._input_key_columns.get(table_name)
        if present is None:
            result = await conn.execute(INPUT_KEY_COLUMN_SQL, [table_name])
            present = await result.fetchone() is not None
            self._input_key_columns[table_name] = present
        if not present and table_name not in _INPUT_KEY_ABSENT_LOGGED:
            _INPUT_KEY_ABSENT_LOGGED.add(table_name)
            logger.warning(
                "Embedding table %s has no input_key column; its entries are embedded "
                "unexpanded until `osprey ariel migrate` adds it",
                table_name,
            )
        return present

    async def enhance(
        self,
        entry: EnhancedLogbookEntry,
        conn: AsyncConnection,
    ) -> None:
        """Generate embeddings for entry and store in database.

        Lazy-loads the embedding provider on first call.
        Embeds the entry's text, with vocabulary terms inserted when an entry expander is
        bound, followed by its picture text (``attachment_text``), cut to the model's input
        limit so the start of each is embedded.

        Args:
            entry: The entry to enhance
            conn: Database connection from pool
        """
        if not self._models:
            logger.warning("No embedding models configured, skipping text embedding")
            return

        if not await self._check_tables_exist(conn):
            return

        provider = self._get_provider()
        raw_text = entry.get("raw_text", "")
        attachment_text = entry.get("attachment_text")

        if not raw_text.strip() and not (attachment_text or "").strip():
            logger.debug(f"Skipping empty entry {entry.get('entry_id')}")
            return

        errors: list[str] = []
        for model_config in self._models:
            try:
                model_name = model_config["name"]
                table_name = model_to_table_name(model_name)
                has_key = await self._has_input_key_column(table_name, conn)
                expander = self._expander if has_key else None

                limit = max_input_tokens(model_config)
                built = build_embedding_input(raw_text, attachment_text, limit, expander=expander)
                text = built.text
                if built.kept_chars < len(raw_text):
                    logger.info(
                        "Entry %s is %d characters; only the first %d were embedded with %s "
                        "(its input limit is %d tokens)",
                        entry.get("entry_id"),
                        len(raw_text),
                        built.kept_chars,
                        model_name,
                        limit,
                    )

                api_key = self._resolution().api_key
                embed_kwargs: dict[str, Any] = {}
                if self._resolved_class().truncates_to_dimensions:
                    embed_kwargs["dimensions"] = model_config["dimension"]

                embeddings = provider.execute_embedding(
                    texts=[text],
                    model_id=model_name,
                    base_url=self._base_url(),
                    api_key=api_key,
                    **embed_kwargs,
                )

                if embeddings and len(embeddings) > 0:
                    await self._store_embedding(
                        entry_id=entry["entry_id"],
                        model_name=model_name,
                        embedding=embeddings[0],
                        conn=conn,
                        input_key=running_input_key(expander) if has_key else None,
                    )
                else:
                    errors.append(f"{model_name}: empty embedding result")

            except Exception as e:
                logger.warning(
                    f"Failed to generate embedding for entry {entry.get('entry_id')} "
                    f"with model {model_config.get('name')}: {e}"
                )
                errors.append(f"{model_config.get('name')}: {e}")
                continue

        if errors and len(errors) == len(self._models):
            raise RuntimeError(
                f"All embedding models failed for entry {entry.get('entry_id')}: "
                + "; ".join(errors)
            )

    async def _store_embedding(
        self,
        entry_id: str,
        model_name: str,
        embedding: list[float],
        conn: AsyncConnection,
        input_key: str | None = None,
    ) -> None:
        """Store embedding in model-specific table.

        Args:
            entry_id: Entry ID
            model_name: Model name for table lookup
            embedding: Embedding vector
            conn: Database connection
            input_key: Build key of the input the vector was embedded from, written to the
                table's ``input_key`` column; ``None`` for a table without that column.
        """
        table_name = model_to_table_name(model_name)

        if input_key is not None:
            await conn.execute(
                f"""
                INSERT INTO {table_name} (entry_id, embedding, input_key)
                VALUES (%s, %s, %s)
                ON CONFLICT (entry_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    input_key = EXCLUDED.input_key,
                    created_at = NOW()
                """,
                [entry_id, embedding, input_key],
            )
            return

        await conn.execute(
            f"""
            INSERT INTO {table_name} (entry_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (entry_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                created_at = NOW()
            """,
            [entry_id, embedding],
        )

    async def health_check(self) -> HealthResult:
        """Check if module is ready.

        Asks the embedding provider for its verdict on the first configured
        model (the adapter's own health model when none is), off the event loop.

        Returns:
            The adapter's ``check_embedding_health`` verdict, unchanged; a probe
            that raises reports the reason its exception classifies to, else
            ``unreachable``.
        """
        return await run_blocking(self._probe_health)

    def _probe_health(self) -> HealthResult:
        """The blocking half of :meth:`health_check`."""
        from osprey.models.providers.health import failure_reason

        try:
            provider = self._get_provider()
            return provider.check_embedding_health(
                api_key=self._resolution().api_key,
                base_url=self._base_url(refresh=True),
                model_id=self._models[0]["name"] if self._models else None,
            )
        except Exception as e:
            return HealthResult(False, str(e), failure_reason(e) or "unreachable")
