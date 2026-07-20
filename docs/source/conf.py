# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

# Keep warnings visible to catch documentation issues
# Do NOT suppress warnings - we want to see import problems

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Add project root and src directories
project_root = os.path.abspath("../..")
src_root = os.path.abspath("../../src")

sys.path.insert(0, project_root)
sys.path.insert(0, src_root)

# Framework is now the only package we need
# NO backwards compatibility for old paths

# -- Project information -----------------------------------------------------


# Function to get version from git
def get_version_from_git():
    """Get the current version from git tags, with GitHub Actions support."""
    try:
        # In GitHub Actions, check if we're building for a specific tag
        github_ref = os.environ.get("GITHUB_REF", "")
        if github_ref.startswith("refs/tags/v"):
            # Extract version from GitHub ref (e.g., refs/tags/v0.7.2 -> 0.7.2)
            version = github_ref.replace("refs/tags/v", "")
            print(f"📋 Using version from GitHub tag: {version}")
            return version

        # Fallback to git describe for local builds
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            # Remove 'v' prefix if present
            version = result.stdout.strip().lstrip("v")
            print(f"📋 Using version from git describe: {version}")
            return version
        else:
            print("⚠️  No git tags found, using fallback version")
            return "0.0.0-dev"
    except (subprocess.SubprocessError, FileNotFoundError):
        print("⚠️  Git not available, using fallback version")
        return "0.0.0-dev"


project = "Osprey Framework"
copyright = "2026, Osprey Developer Team"
author = "Osprey Developer Team"
release = get_version_from_git()

# -- General configuration ---------------------------------------------------

# Add custom extensions directory to path
sys.path.insert(0, os.path.abspath("_ext"))

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate API docs
    "sphinx.ext.autosummary",  # Auto-generate summary tables
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.intersphinx",  # Link to other projects
    "sphinx.ext.githubpages",  # GitHub Pages support
    "myst_parser",  # Markdown support
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx.ext.graphviz",  # Graph visualization
    "sphinx.ext.todo",  # TODO notes
    "sphinx_design",  # Design components (cards, tabs, etc.)
    "sphinxcontrib.mermaid",  # Mermaid diagram support
    "workflow_autodoc",  # Custom: Auto-document workflow files
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output ------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Theme options for PyData Sphinx Theme - Clean Original Style
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/als-apg/osprey",
            "icon": "fa-brands fa-github",
        },
    ],
    # Using clean text-only logo for proper spacing
    "logo": {
        "text": "Osprey Framework",
    },
    "show_toc_level": 2,
    "navbar_align": "left",
    # Enable edit button in secondary sidebar
    "use_edit_page_button": True,
    # Configure secondary sidebar items - clean layout with TOC and edit button only
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    # Version switcher configuration
    "switcher": {
        "json_url": "https://als-apg.github.io/osprey/_static/versions.json",
        "version_match": release,
    },
    # Add version switcher to navbar
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
}

# Repository information for edit buttons
html_context = {
    "github_user": "als-apg",
    "github_repo": "osprey",
    "github_version": "main",
    "doc_path": "docs/source",
}

# HTML settings - Clean original theme style (no conflicting logo settings)
# html_logo = "_static/logo.svg"  # Commented out to avoid conflict with logo.text
html_favicon = "_static/logo.svg"
html_sourcelink_suffix = ""
html_last_updated_fmt = ""

# Disable the default Sphinx "Show Source" link since we use the theme's sourcelink component
html_show_sourcelink = False

# Ensure indices are generated
html_use_index = True
html_domain_indices = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = ["custom.css"]


# -- Autodoc configuration --------------------------------------------------

# EXPLICIT MOCK IMPORTS
# These are external dependencies that we intentionally do NOT install in CI
# to keep the documentation build lightweight and fast. Each module listed here
# represents a conscious decision to mock rather than install the real dependency.
#
# If a module fails to import and is NOT in this list, the build will fail loudly,
# indicating that we need to either:
# 1. Add it to [project.optional-dependencies].docs in pyproject.toml (if it's essential for docs)
# 2. Add it to this mock list (if it's an optional heavy dependency)
# 3. Fix the import structure in the actual code

autodoc_mock_imports = [
    # Heavy API client libraries - interfaces documented, implementations mocked
    "openai",
    "anthropic",
    "google",
    "google.generativeai",
    "google.genai",
    "google.genai.types",
    "ollama",
    "litellm",
    # Data science stack - too heavy for docs CI, interfaces documented
    "pandas",
    "numpy",
    "matplotlib",
    "plotly",
    "seaborn",
    "scikit-learn",
    "scipy",
    # Database clients - connection logic mocked, interfaces documented
    "pymongo",
    "neo4j",
    "qdrant_client",
    "psycopg",
    "psycopg.rows",
    "psycopg_pool",
    # Container and deployment tools - not needed for documentation
    "docker",
    "podman",
    "python-dotenv",
    "dotenv",
    # EPICS control system - specialized scientific software
    "epics",
    "pyepics",
    "p4p",
    "pvaccess",
    # Notebook format library - not needed for static documentation
    "nbformat",
    # Development tools - not needed for static documentation
    "pytest",
    "jupyter",
    "notebook",
    "ipykernel",
    # Network and async libraries - interfaces documented, implementations mocked
    "aiohttp",
    "websockets",
]

# IMPORTANT: If you see import errors for modules NOT in the above list,
# that means we need to decide whether to install them (add to [project.optional-dependencies].docs in pyproject.toml)
# or mock them (add to the list above). DO NOT add modules to this list without
# understanding why they're failing to import.

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Enhanced autodoc settings following API guide best practices
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autoclass_content = "both"  # Class + __init__ docstrings
autodoc_member_order = "bysource"  # Preserve logical order

# Napoleon configuration for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Autosummary configuration ----------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_imported_members = True

# Handle import failures explicitly - do NOT suppress warnings
autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True

# Ensure we see all import issues clearly
autodoc_warningiserror = False  # Set to True to fail on autodoc warnings

# -- Intersphinx configuration ----------------------------------------------

# Disabled due to firewall/proxy restrictions
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'pandas': ('https://pandas.pydata.org/docs/', None),
#     'numpy': ('https://numpy.org/doc/stable/', None),
#     'ray': ('https://docs.ray.io/en/latest/', None),
# }
intersphinx_mapping = {}

# -- MyST configuration -----------------------------------------------------

myst_enable_extensions = [
    "deflist",
    "tasklist",
    "colon_fence",
    "substitution",
    "dollarmath",
]


def _screenshot_caption_prolog():
    """Build ``|captured_<name>|`` substitutions for every screenshot recipe.

    Delegates to :func:`docs.screenshots.recipes.caption_substitutions` (which is
    unit-tested): the set is derived from the recipe *registry*, not from
    ``manifest.json`` presence, so every substitution is always defined and the
    docs build never fails on a fresh clone with no captured manifest.
    """
    try:
        from docs.screenshots.recipes import caption_substitutions
    except Exception:  # pragma: no cover - registry import must never break the build
        return ""
    return "\n".join(
        f".. |{name}| replace:: {value}" for name, value in caption_substitutions().items()
    )


# Make version (and per-screenshot capture provenance) available as RST substitutions
rst_prolog = f"""
.. |version| replace:: {release}
.. |release| replace:: v{release}
{_screenshot_caption_prolog()}
"""

# -- Todo configuration -----------------------------------------------------

todo_include_todos = True

# -- Copy button configuration ----------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Sphinx Design configuration -------------------------------------------

# Enable sphinx-design components
sd_fontawesome_latex = True

# -- Mermaid configuration -------------------------------------------------

# Use client-side rendering (no CLI needed)
mermaid_output_format = "raw"

# Mermaid version to use
mermaid_version = "11.8.0"

# Size each diagram to its own aspect ratio. The extension's default is a flat
# `height: 500px` on every rendered SVG, which letterboxes a wide `flowchart LR`
# into a near-square box (the multi-user diagram is 6.5:1 but rendered at
# 0.98:1, ~68% of it empty). "auto" lets the browser derive height from the
# SVG's viewBox instead.
mermaid_height = "auto"

# Light/dark theming is handled by the extension itself: it detects the
# page theme, and a MutationObserver on `data-theme` re-renders every
# diagram when the reader toggles. Use `mermaid_init_config` if this ever
# needs custom theme variables.
