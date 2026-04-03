"""Migration service — business logic for OSPREY project migrations."""

from .engine import (
    FileCategory,
    calculate_file_hash,
    classify_file,
    detect_project_settings,
    generate_merge_prompt,
    generate_migration_directory,
    load_manifest,
    migrate_claude_code_config,
    perform_migration_analysis,
    read_file_content,
)

__all__ = [
    "FileCategory",
    "calculate_file_hash",
    "classify_file",
    "detect_project_settings",
    "generate_merge_prompt",
    "generate_migration_directory",
    "load_manifest",
    "migrate_claude_code_config",
    "perform_migration_analysis",
    "read_file_content",
]
