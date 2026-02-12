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
copyright = "2025, Osprey Developer Team"
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

# Add JavaScript files for execution plan viewer
html_js_files = ["js/execution_plan_viewer.js"]

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
    "langgraph.checkpoint.postgres",
    # Specialized infrastructure - interfaces documented, implementations mocked
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_core.messages",
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
    # Framework services that depend on complex infrastructure
    # Note: The services themselves can now import, but their dependencies are mocked above
    # Internal modules that have import issues during docs build
    "container_manager",
    "loader",
    # Framework modules that fail due to registry/config dependencies
    "framework.infrastructure.task_extraction_node",
    "framework.infrastructure.orchestration_node",
    "framework.infrastructure.error_node.ErrorType",
    "framework.services.python_executor.PythonExecutorConfig",
    # Capability error classes that depend on registry
    "framework.capabilities.memory",
    "framework.capabilities.time_range_parsing",
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

# Make version available as substitution variables for RST files
rst_prolog = f"""
.. |version| replace:: {release}
.. |release| replace:: v{release}
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

# Mermaid initialization with dynamic light/dark theme support
mermaid_init_js = """
// Disable automatic initialization to control rendering manually
mermaid.initialize({ startOnLoad: false });

// Function to detect current theme from PyData Sphinx Theme
function getCurrentTheme() {
    return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'base';
}

// Function to add CSS overrides for text colors
function addMermaidCSS() {
    const cssId = 'mermaid-text-fix';
    if (!document.getElementById(cssId)) {
        const css = document.createElement('style');
        css.id = cssId;
        css.textContent = `
            /* Force text colors in Mermaid diagrams */
            html[data-theme="dark"] .mermaid text,
            html[data-theme="dark"] .mermaid .nodeLabel,
            html[data-theme="dark"] .mermaid .edgeLabel,
            html[data-theme="dark"] .mermaid .labelText,
            html[data-theme="dark"] .mermaid .actor-label,
            html[data-theme="dark"] .mermaid .messageText,
            html[data-theme="dark"] .mermaid .noteText,
            html[data-theme="dark"] .mermaid tspan,
            html[data-theme="dark"] .mermaid .label foreignObject div,
            html[data-theme="dark"] .mermaid .label span {
                fill: #ffffff !important;
                color: #ffffff !important;
            }

            html[data-theme="light"] .mermaid text,
            html[data-theme="light"] .mermaid .nodeLabel,
            html[data-theme="light"] .mermaid .edgeLabel,
            html[data-theme="light"] .mermaid .labelText,
            html[data-theme="light"] .mermaid .actor-label,
            html[data-theme="light"] .mermaid .messageText,
            html[data-theme="light"] .mermaid .noteText,
            html[data-theme="light"] .mermaid tspan,
            html[data-theme="light"] .mermaid .label foreignObject div,
            html[data-theme="light"] .mermaid .label span {
                fill: #333333 !important;
                color: #333333 !important;
            }

            /* Remove any filters that might interfere */
            .mermaid svg {
                filter: none !important;
            }
        `;
        document.head.appendChild(css);
    }
}

// Function to render all Mermaid diagrams with current theme
function renderMermaidDiagrams() {
    const currentTheme = getCurrentTheme();

    // Add CSS overrides
    addMermaidCSS();

    // Configure Mermaid with current theme
    const mermaidConfig = {
        startOnLoad: false,
        theme: currentTheme,
        flowchart: {
            htmlLabels: true,
            curve: "basis",
            useMaxWidth: true
        },
        sequence: {
            mirrorActors: false,
            messageAlign: "center",
            useMaxWidth: true,
            showSequenceNumbers: false
        },
        gantt: {
            numberSectionStyles: 4,
            axisFormat: "%m/%d/%Y",
            topAxis: false
        },
        journey: {
            diagramMarginX: 50,
            diagramMarginY: 10,
            leftMargin: 150,
            width: 150,
            height: 50,
            boxMargin: 10,
            boxTextMargin: 5,
            noteMargin: 10,
            messageMargin: 35,
            messageAlign: "center"
        },
        gitgraph: {
            mainBranchName: "main",
            showBranches: true,
            showCommitLabel: true
        },
        pie: {
            textPosition: 0.75
        },
        timeline: {
            padding: 5,
            useMaxWidth: true
        },
        mindmap: {
            padding: 10,
            useMaxWidth: true
        }
    };

    // Apply theme-specific customizations
    if (currentTheme === 'dark') {
        mermaidConfig.themeVariables = {
            // Main colors
            primaryColor: "#4a90e2",
            primaryTextColor: "#ffffff",
            primaryBorderColor: "#4a90e2",
            lineColor: "#ffffff",

            // Text colors - comprehensive coverage
            textColor: "#ffffff",
            secondaryColor: "#ffffff",
            tertiaryColor: "#ffffff",

            // Node text colors
            nodeTextColor: "#ffffff",
            edgeLabelBackground: "#1a202c",

            // Background colors
            background: "#1a202c",
            mainBkg: "#2d3748",
            secondBkg: "#4a5568",
            sectionBkgColor: "#2d3748",
            altSectionBkgColor: "#1a202c",

            // Grid and borders
            gridColor: "#4a5568",

            // Flowchart specific
            clusterBkg: "#2d3748",
            clusterBorder: "#4a5568",

            // Sequence diagram specific
            actorBkg: "#2d3748",
            actorBorder: "#4a5568",
            actorTextColor: "#ffffff",
            actorLineColor: "#4a5568",
            signalColor: "#ffffff",
            signalTextColor: "#ffffff",
            labelBoxBkgColor: "#2d3748",
            labelBoxBorderColor: "#4a5568",
            labelTextColor: "#ffffff",
            loopTextColor: "#ffffff",
            noteBkgColor: "#2d3748",
            noteTextColor: "#ffffff",
            noteBorderColor: "#4a5568",

            // State diagram specific
            fillType0: "#2d3748",
            fillType1: "#4a5568",
            fillType2: "#5a6c7d",
            fillType3: "#6a7c8d",
            fillType4: "#7a8c9d",
            fillType5: "#8a9cad",
            fillType6: "#9aacbd",
            fillType7: "#aabccd",

            // Class diagram specific
            classText: "#ffffff",

            // Git diagram specific
            git0: "#4a90e2",
            git1: "#f6ad55",
            git2: "#68d391",
            git3: "#fc8181",
            git4: "#b794f6",
            git5: "#f687b3",
            git6: "#4fd1c7",
            git7: "#cbd5e0",
            gitBranchLabel0: "#ffffff",
            gitBranchLabel1: "#ffffff",
            gitBranchLabel2: "#ffffff",
            gitBranchLabel3: "#ffffff",
            gitBranchLabel4: "#ffffff",
            gitBranchLabel5: "#ffffff",
            gitBranchLabel6: "#ffffff",
            gitBranchLabel7: "#ffffff",

            // Journey diagram specific
            taskBkgColor: "#2d3748",
            taskTextColor: "#ffffff",
            taskTextLightColor: "#ffffff",
            taskTextOutsideColor: "#ffffff",
            taskTextClickableColor: "#4a90e2",
            activeTaskBkgColor: "#4a90e2",
            activeTaskBorderColor: "#4a90e2",
            gridColor: "#4a5568",
            section0: "#2d3748",
            section1: "#4a5568",
            section2: "#5a6c7d",
            section3: "#6a7c8d"
        };
    } else {
        mermaidConfig.themeVariables = {
            // Main colors
            primaryColor: "#1f77b4",
            primaryTextColor: "#333333",
            primaryBorderColor: "#1f77b4",
            lineColor: "#333333",

            // Text colors
            textColor: "#333333",
            secondaryColor: "#333333",
            tertiaryColor: "#333333",

            // Node text colors
            nodeTextColor: "#333333",
            edgeLabelBackground: "#ffffff",

            // Background colors
            background: "#ffffff",
            mainBkg: "#ffffff",
            secondBkg: "#f8f9fa",
            sectionBkgColor: "#f9f9f9",
            altSectionBkgColor: "#ffffff",

            // Grid and borders
            gridColor: "#e0e0e0"
        };
    }

    // Initialize Mermaid with current configuration
    mermaid.initialize(mermaidConfig);

    // Find all Mermaid diagrams and render them
    const diagrams = document.querySelectorAll('.mermaid');
    diagrams.forEach((diagram, index) => {
        // Preserve original content if not already stored
        if (!diagram.dataset.originalContent) {
            diagram.dataset.originalContent = diagram.textContent.trim();
        }

        // Generate unique ID for this diagram
        const diagramId = `mermaid-diagram-${index}`;

        // Render the diagram
        mermaid.render(diagramId, diagram.dataset.originalContent)
            .then(result => {
                diagram.innerHTML = result.svg;
            })
            .catch(error => {
                console.error('Mermaid rendering error:', error);
                diagram.innerHTML = '<div class="mermaid-error">Error rendering diagram</div>';
            });
    });
}

// Initial render when page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderMermaidDiagrams);
} else {
    renderMermaidDiagrams();
}

// Watch for theme changes using MutationObserver
const themeObserver = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
            // Small delay to ensure theme change is complete
            setTimeout(renderMermaidDiagrams, 100);
        }
    });
});

// Start observing theme changes
themeObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme']
});
"""
