"""Vendored English full-text-search stopword list.

The words below are a verbatim copy of PostgreSQL's
``src/backend/snowball/stopwords/english.stop`` (installed as
``/usr/share/postgresql/<major>/tsearch_data/english.stop``), read from
PostgreSQL **15.4** — the server the ARIEL test suite runs against
(``ankane/pgvector``). The list is stable across PostgreSQL majors; it is
vendored here so vocabulary validation needs no database connection.

Why it matters: ARIEL's keyword search hardcodes the ``'english'`` text-search
configuration, so ``plainto_tsquery('english', …)`` silently discards every one
of these words. A vocabulary canonical (or a direction-enabled form) that is one
of them expands to a term that can never match anything, so the loader warns.
"""

#: The 127 words PostgreSQL's ``english`` text-search configuration discards.
_ENGLISH_FTS_STOPWORDS: frozenset[str] = frozenset(
    {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which", "who",
        "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the",
        "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now",
    }
)  # fmt: skip

__all__ = ["_ENGLISH_FTS_STOPWORDS"]
