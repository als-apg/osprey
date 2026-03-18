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
	@echo "  test-build          Test documentation build (like GitHub Actions)"
	@echo "  test-deps           Install documentation dependencies"
	@echo "  pre-commit-test     Quick test (assumes deps installed)"

.PHONY: help Makefile test-build test-deps pre-commit-test

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Install documentation dependencies (like GitHub Actions)
test-deps:
	@echo "Installing documentation dependencies..."
	cd .. && uv sync --extra docs

# Test documentation build (replicates GitHub Actions)
test-build: test-deps clean
	@echo "\033[1;34mTesting documentation build (GitHub Actions simulation)...\033[0m"
	@echo "   This shows the same errors/warnings as CI"
	@echo ""
	@FORCE_COLOR=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee build.log
	@echo ""
	@echo "\033[1;36mBUILD SUMMARY\033[0m"
	@echo "\033[1;36m================\033[0m"
	@if grep -E "(WARNING|ERROR):" build.log > build_issues.log 2>/dev/null; then \
		echo "\033[1;33mIssues found ($$(wc -l < build_issues.log 2>/dev/null || echo 0)):\033[0m" ; \
		echo "" ; \
		sed 's/WARNING:/\x1b[33mWARNING:\x1b[0m/g; s/ERROR:/\x1b[31mERROR:\x1b[0m/g' build_issues.log ; \
		echo "" ; \
		echo "\033[1;34mFull log saved to: build.log\033[0m" ; \
	else \
		echo "\033[1;32mNo warnings or errors found!\033[0m" ; \
	fi
	@echo ""
	@if [ -f "$(BUILDDIR)/html/index.html" ]; then \
		echo "\033[1;32mDocumentation build successful!\033[0m" ; \
		echo "\033[1;34mBuilt files are in: $(BUILDDIR)/html/\033[0m" ; \
		echo "\033[1;34mOpen $(BUILDDIR)/html/index.html in your browser\033[0m" ; \
	else \
		echo "\033[1;31mDocumentation build failed!\033[0m" ; \
		echo "\033[1;33mCheck the issues above and build.log for details\033[0m" ; \
		exit 1 ; \
	fi
	@rm -f build_issues.log

# Quick pre-commit test (faster, assumes deps installed)
pre-commit-test: clean
	@echo "Quick documentation build test..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Pre-commit test passed!"

clean-build:
	rm -rf "$(BUILDDIR)"/*
