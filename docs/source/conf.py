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

# Add project root and src directories. The osprey-connectors and osprey-facility
# workspace packages keep their own src trees; the API reference under
# reference/connectors-api/ autodocs the connectors, and the framework imports both.
project_root = os.path.abspath("../..")
src_root = os.path.abspath("../../src")
connectors_src_root = os.path.abspath("../../packages/osprey-connectors/src")
facility_src_root = os.path.abspath("../../packages/osprey-facility/src")

sys.path.insert(0, project_root)
sys.path.insert(0, connectors_src_root)
sys.path.insert(0, facility_src_root)
sys.path.insert(0, src_root)

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
    "sphinx.ext.todo",  # the todo directive
    "sphinx_design",  # Design components (cards, tabs, etc.)
    "sphinx_reredirects",  # Old-URL redirects for moved pages
    "workflow_autodoc",  # Custom: Auto-document workflow files
    "port_table",  # Custom: Render the host-port layout as a table
]

# Graphviz diagrams render as SVG (crisp at any zoom, transparent background)
graphviz_output_format = "svg"


# Old page path -> new location. Keys are docnames (no suffix) of pages that no
# longer exist; values are PAGE-RELATIVE targets with the `.html` suffix, e.g.
# "cli-reference/index": "../reference/cli.html". Populated as pages move.
redirects: dict[str, str] = {
    # The control-assistant preset became `demo`
    "getting-started/control-assistant": "demo.html",
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
    # The Operate pages on health and monitoring
    "how-to/configure-health-checks": "health-and-monitoring/configure-health-checks.html",
    "how-to/monitor-agent": "health-and-monitoring/agent-record.html",
    "how-to/health-and-monitoring/monitor-agent": "agent-record.html",
    "how-to/health-and-monitoring/reconstruct-session": "agent-record.html",
    "how-to/health-and-monitoring/read-the-record": "agent-record.html",
    "how-to/health-json-contract": "../reference/contracts/health-json.html",
    # Operate: control systems
    "how-to/add-connector": "control-systems/use-connectors.html",
    "how-to/use-virtual-accelerator": "control-systems/use-virtual-accelerator.html",
    "how-to/switch-control-target": "control-systems/switch-control-target.html",
    "how-to/run-scenarios": "control-systems/use-virtual-accelerator.html",
    "how-to/protected-set": "../reference/configuration/config.html",
    # Facility services
    "how-to/use-facility-knowledge": "facility-knowledge/index.html",
    "how-to/use-facility-graph": "facility-knowledge/use-facility-graph.html",
    "how-to/okf-bundle": "facility-knowledge/okf-bundle.html",
    "how-to/facility-rules": "facility-knowledge/facility-rules.html",
    "how-to/search-sidecar": "ariel/search-sidecar.html",
    "how-to/ariel/osprey-integration": "../../reference/contracts/ariel.html",
    # Integrations
    "how-to/measure-with-pyaml": "pyaml/index.html",
    "how-to/register-a-measurement-kind": "pyaml/register-a-measurement-kind.html",
    # Developer material
    "how-to/use-python-executor": "../architecture/python-executor.html",
    # Contract reference pages
    "reference/contracts/simulation-bundle": "../configuration/scenario-files.html",
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Link check configuration -----------------------------------------------

# Addresses that answer to a reader but not to a checker, so `make linkcheck`
# neither reports nor blocks on them, and nothing wider: a loopback address names
# the reader's own deployment, which is not running where the docs are built;
# the next hosts refuse an automated request with 403 and open normally in a
# browser; a DOI is a permanent identifier by construction; the blog host opens
# in a browser but times out a checker run from a build machine. Anything else
# that goes red is a link a reader cannot follow.
linkcheck_ignore = [
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?(/|$)",
    r"^https://claude\.ai/",
    r"^https://api\.cborg\.lbl\.gov",
    r"^https://doi\.org/",
    r"^https://cloud\.google\.com/blog/",
]

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
    "google",
    "google.generativeai",
    "google.genai",
    "google.genai.types",
    "litellm",
    # Data science stack - too heavy for docs CI, interfaces documented. numpy,
    # pandas and scipy are not mocked: osprey-connectors depends on numpy and
    # pandas directly and on scipy through lume-base and beamphysics, so the docs
    # extra installs all three, and the simulation modules (and beamphysics)
    # compute with them at import time, which a mock cannot answer.
    "matplotlib",
    "plotly",
    "seaborn",
    "scikit-learn",
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
    "aioca",
    "epicscorelibs",
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
# A Google-style "Attributes:" section renders as a field list, not as attribute
# entries: autodoc already documents a dataclass's fields, and both at once
# describe each field twice.
napoleon_use_ivar = True
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

# -- Redirect stubs ---------------------------------------------------------


def _missing_redirect_stubs(outdir, redirects):
    """The redirect keys whose stub page is absent from *outdir*, in key order."""
    return [key for key in redirects if not os.path.isfile(os.path.join(outdir, f"{key}.html"))]


def _check_redirect_stubs(app, exception):
    """Fail the html build when a redirect key has no stub page.

    sphinx_reredirects writes one stub page per ``redirects`` key on
    html-collect-pages, which the html builder emits before build-finished. A
    stub that never got written means a moved page's old URL 404s for every
    reader who bookmarked it. Other builders write no stubs, and a build that
    already failed keeps its own error.
    """
    if exception is not None or app.builder.name != "html":
        return
    missing = _missing_redirect_stubs(app.outdir, app.config.redirects)
    if missing:
        from sphinx.errors import SphinxError

        raise SphinxError("redirect stubs missing from the build: " + " ".join(missing))


# -- Published facility.json schemas ---------------------------------------

#: Where the osprey-facility package keeps its generated schemas; the docs serve
#: copies taken at build time, so no schema file is committed twice.
FACILITY_SCHEMA_DIR = os.path.join(facility_src_root, "osprey_facility", "schema", "_generated")

#: The directory, below the html output, the facility schemas are served from.
FACILITY_SCHEMA_SERVED = "schemas/facility"


def _published_schemas(generated_dir):
    """Map each served schema path (below the html root) to its file in *generated_dir*.

    The current schema is served at ``<N>.schema.json`` for the N of its
    ``schema`` const, each frozen ``facility-<N>.final.schema.json`` at
    ``<N>.schema.json``, and each floor ``facility-<M>.schema.json`` at
    ``<M>.floor.schema.json``. Two files claiming one served path is an error.
    """
    import json
    import re

    served = {}

    def claim(name, source):
        path = f"{FACILITY_SCHEMA_SERVED}/{name}"
        if path in served:
            raise ValueError(f"{path} is claimed by {served[path]} and {source}")
        served[path] = source

    current = os.path.join(generated_dir, "facility.schema.json")
    with open(current, encoding="utf-8") as handle:
        header = json.load(handle)["properties"]["schema"]["const"]
    claim(f"{header.rsplit('/', 1)[-1]}.schema.json", current)
    for name in sorted(os.listdir(generated_dir)):
        final = re.fullmatch(r"facility-(\d+)\.final\.schema\.json", name)
        floor = re.fullmatch(r"facility-(\d+)\.schema\.json", name)
        if final:
            claim(f"{final.group(1)}.schema.json", os.path.join(generated_dir, name))
        elif floor:
            claim(f"{floor.group(1)}.floor.schema.json", os.path.join(generated_dir, name))
    return served


def _copy_published_schemas(outdir, generated_dir=FACILITY_SCHEMA_DIR):
    """Copy the facility schemas into *outdir* at their served paths; return the map."""
    import shutil

    served = _published_schemas(generated_dir)
    for path, source in served.items():
        target = os.path.join(outdir, *path.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    return served


def _publish_schemas(app, exception):
    """Serve the facility.json schemas from the html build, at the URL their ``$id`` names."""
    if exception is not None or app.builder.name != "html":
        return
    _copy_published_schemas(app.outdir)


def setup(app):
    app.connect("build-finished", _check_redirect_stubs)
    app.connect("build-finished", _publish_schemas)
