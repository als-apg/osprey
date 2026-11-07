# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional targets:"
	@echo "  test-build          Test documentation build, warnings as errors (like CI)"
	@echo "  test-deps           Install documentation dependencies"
	@echo "  pre-commit-test     Quick test, warnings as errors (assumes deps installed)"
	@echo "  screenshots         Regenerate default (standalone) doc screenshots"
	@echo "  screenshots-<name>  Regenerate one recipe by name (e.g. screenshots-display_menu)"
	@echo "  doc-videos          Record and encode every docs video (VIDEOOPTS=\"--video landing --theme dark\")"
	@echo "  demo-video          Record and encode the landing-page demo video only"
	@echo "  upload-doc-videos   Upload the videos not uploaded yet to the docs-media-vVERSION release (VERSION=YYYY.M.P, or a pre-release like 2026.9.0b4)"

.PHONY: help Makefile test-build test-deps pre-commit-test screenshots screenshots-% doc-videos demo-video upload-doc-videos upload-demo-video

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

LIVEHTML_PORT ?= 8008

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" --port $(LIVEHTML_PORT) $(SPHINXOPTS) $(O)

# Regenerate committed documentation screenshots from the declarative registry
# (docs/screenshots). Run from the repo root so `docs.screenshots` resolves as a
# namespace package. The default target captures only the container-free
# standalone recipes; pass SCREENSHOTOPTS=--stack (containers) or
# SCREENSHOTOPTS=--agentic (live Claude session) to opt into the heavier ones.
PYTHON        ?= python
SCREENSHOTOPTS ?=

screenshots:
	cd .. && $(PYTHON) -m docs.screenshots $(SCREENSHOTOPTS)

screenshots-%:
	cd .. && $(PYTHON) -m docs.screenshots --only $* $(SCREENSHOTOPTS)

# Record every docs video (docs/screenshots/videos.py) into docs/demo-video/,
# with posters and manifest in source/_static/demo/ (needs ffmpeg, Playwright
# chromium, containers and a live Claude session; skips otherwise).
# VIDEOOPTS="--video NAME --theme dark" narrows it; demo-video is the landing
# demo alone. upload-doc-videos checks the takes against the manifest,
# publishes the videos not uploaded yet as assets of the docs-media-vVERSION
# release (one release per OSPREY version) and names that release in the
# manifest; upload-demo-video is its old name.
VIDEOOPTS ?=
UPLOADOPTS ?=
VERSION ?=

doc-videos:
	cd .. && $(PYTHON) -m docs.screenshots video $(VIDEOOPTS)

demo-video:
	cd .. && $(PYTHON) -m docs.screenshots video --video landing $(VIDEOOPTS)

upload-doc-videos upload-demo-video:
	@test -n "$(VERSION)" || { echo "usage: make $@ VERSION=YYYY.M.P[aN|bN|rcN]"; exit 2; }
	cd .. && $(PYTHON) -m docs.screenshots upload --version $(VERSION) $(UPLOADOPTS)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Install documentation dependencies (like GitHub Actions)
test-deps:
	@echo "Installing documentation dependencies..."
	cd .. && uv sync --extra docs

# Test documentation build as CI does: every warning is an error.
test-build pre-commit-test: SPHINXOPTS += -W --keep-going
test-build: test-deps clean
	@echo "Testing documentation build (warnings fail it, as in CI)..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Documentation build successful: $(BUILDDIR)/html/index.html"

# Quick pre-commit test (faster, assumes deps installed)
pre-commit-test: clean
	@echo "Quick documentation build test..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Pre-commit test passed!"

clean-build:
	rm -rf "$(BUILDDIR)"/*
