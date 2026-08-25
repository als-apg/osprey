# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
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

# Imported here rather than at the top of the file: the sys.path setup above is what
# makes ``osprey`` importable when the docs build from an uninstalled checkout.
from osprey.version import get_running_version, is_release  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Osprey Framework"
copyright = "2026, Osprey Developer Team"
author = "Osprey Developer Team"

# The docs version comes from the package's single source of truth. Anything that is not
# a clean tagged release publishes as the literal "dev": that substring is what the
# pydata-sphinx-theme development banner and the switcher's development entry key on,
# via ``DOCUMENTATION_OPTIONS.VERSION`` (which Sphinx fills from ``release``).
release = get_running_version() if is_release() else "dev"
version = release

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
    "sphinx_reredirects",  # Old-URL redirects for moved pages
    "workflow_autodoc",  # Custom: Auto-document workflow files
]

# Old page path -> new location. Keys are docnames (no suffix) of pages that no
# longer exist; values are PAGE-RELATIVE targets with the `.html` suffix, e.g.
# "cli-reference/index": "../reference/cli.html". Populated as pages move.
redirects: dict[str, str] = {
    # CLI reference -> the new top-level Reference section
    "cli-reference/index": "../reference/cli.html",
    # Build & deploy
    "how-to/containerize-project": "deploy-project/project-image.html",
    "how-to/deploy-project": "deploy-project/index.html",
    "how-to/configure-providers": "llm-providers/configure-providers.html",
    "how-to/run-open-models": "llm-providers/run-open-models.html",
    # Operate: web terminal
    "how-to/use-web-terminal": "web-terminal/index.html",
    "how-to/send-feedback": "web-terminal/send-feedback.html",
    "how-to/multi-user": "web-terminal/multi-user/index.html",
    "how-to/multi-user/index": "../web-terminal/multi-user/index.html",
    "how-to/multi-user/login": "../web-terminal/multi-user/login.html",
    "how-to/multi-user/tiers": "../web-terminal/multi-user/tiers.html",
    # Operate: agent interfaces
    "how-to/use-cli-chat": "agent-interfaces/cli-agent.html",
    "how-to/non_interactive_query": "agent-interfaces/cli-agent.html",
    "how-to/cli-agent": "agent-interfaces/cli-agent.html",
    "how-to/event-dispatch": "agent-interfaces/event-dispatch.html",
    "how-to/add-mcp-server": "agent-interfaces/add-mcp-server.html",
    "how-to/chat-bridges/index": "../agent-interfaces/chat-bridges/index.html",
    "how-to/chat-bridges/nextcloud-talk": "../agent-interfaces/chat-bridges/nextcloud-talk.html",
    "how-to/chat-bridges/google-chat": "../agent-interfaces/chat-bridges/google-chat.html",
    "how-to/chat-bridges/add-a-channel": "../../contributing/extending-osprey.html",
    "how-to/agent-interfaces/chat-bridges/add-a-channel": "../../../contributing/extending-osprey.html",
    # Operate: health and monitoring
    "how-to/configure-health-checks": "health-and-monitoring/configure-health-checks.html",
    "how-to/monitor-agent": "health-and-monitoring/monitor-agent.html",
    "how-to/health-json-contract": "../reference/contracts/health-json.html",
    # Operate: control systems
    "how-to/add-connector": "control-systems/use-connectors.html",
    "how-to/use-virtual-accelerator": "control-systems/use-virtual-accelerator.html",
    "how-to/switch-control-target": "control-systems/switch-control-target.html",
    "how-to/protected-set": "control-systems/protected-set.html",
    # Facility services
    "how-to/use-facility-knowledge": "facility-knowledge/index.html",
    "how-to/use-facility-graph": "facility-knowledge/use-facility-graph.html",
    "how-to/okf-bundle": "facility-knowledge/okf-bundle.html",
    "how-to/facility-rules": "facility-knowledge/facility-rules.html",
    "how-to/search-sidecar": "facility-knowledge/search-sidecar.html",
    "how-to/ariel/osprey-integration": "../../reference/contracts/ariel.html",
    # Developer material
    "how-to/use-python-executor": "../architecture/python-executor.html",
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output ------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Every build - old snapshot or /latest/ - points its canonical link at the stable root
# (the numpy/pandas convention), so search engines rank the root rather than a snapshot.
# A /latest/-only page's canonical 404s until the next release, which engines treat as a
# hint to ignore rather than as an error.
html_baseurl = "https://als-apg.github.io/osprey/"

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
        "json_url": html_baseurl + "_static/versions.json",
        "version_match": release,
    },
    # Banner keyed off ``release``: an old snapshot gets "old version", ``dev`` gets
    # "unstable development version", and the ``preferred`` entry gets no banner.
    "show_version_warning_banner": True,
    # Add version switcher to navbar
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
}

# Repository information for edit buttons. ``github_version`` stays on a branch name no
# matter which version is being published, because GitHub's /edit/ route needs a branch.
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


# Make version (and per-screenshot capture provenance) available as RST substitutions.
# A development build has no tag to prefix, so |release| reads "dev", never "vdev".
_release_label = f"v{release}" if is_release() else "dev"

rst_prolog = f"""
.. |version| replace:: {release}
.. |release| replace:: {_release_label}
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
