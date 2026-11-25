"""The framework's facility-tree tables, pinned as literals.

Every framework table that names an entry of the facility tree, or a path
under ``data/`` beside it, is held here to its exact value. A table computed
from a declaration must keep producing these values; a change to one is a
visible, reviewed edit of this file.

The expected values are module constants so other tests can import them.
"""

from __future__ import annotations

from osprey.cli import scaffold_pull
from osprey.cli.templates import preset_data
from osprey.facility import settings as facility_settings
from osprey.facility.views import benchmarks
from osprey.services.code_review import settings as code_review_settings

QUERY_SOURCES = "benchmarks/channel_finder"

IN_CONTEXT_QUERY_FILE = "in_context_queries.json"

TREE_QUERY_FILE = "tree_queries.json"

VOCABULARY_SOURCE = "data/facility/ariel/vocabulary.yml"

KNOWLEDGE_ROOT = "data/facility/knowledge"

FACILITY_DATA_DIRNAME = "facility"

CODE_REVIEW_DEFAULT_PROMPT = "agent/code_review/reviewer.md"


def test_the_benchmark_query_sources_directory() -> None:
    assert benchmarks.QUERY_SOURCES == QUERY_SOURCES


def test_the_in_context_mode_reads_its_own_query_file() -> None:
    assert benchmarks.query_source_name("in_context") == IN_CONTEXT_QUERY_FILE


def test_every_other_mode_reads_the_tree_query_file() -> None:
    assert benchmarks.query_source_name("hierarchical") == TREE_QUERY_FILE
    assert benchmarks.query_source_name(None) == TREE_QUERY_FILE


def test_the_ariel_vocabulary_source() -> None:
    assert facility_settings.VOCABULARY_SOURCE == VOCABULARY_SOURCE


def test_the_knowledge_root_scaffold_pull_reads() -> None:
    assert scaffold_pull._KNOWLEDGE_ROOT == KNOWLEDGE_ROOT


def test_the_preset_facility_data_directory_name() -> None:
    assert preset_data.FACILITY_DATA_DIRNAME == FACILITY_DATA_DIRNAME


def test_the_code_review_default_prompt() -> None:
    assert code_review_settings.DEFAULT_PROMPT == CODE_REVIEW_DEFAULT_PROMPT
