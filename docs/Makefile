# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Shared module paths (updated for new structure)
SHARED_JS_DIR = ../src/osprey/templates/services/open-webui/functions/shared
STATIC_JS_DIR = source/_static/js

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional targets:"
	@echo "  copy-shared-modules  Copy shared JavaScript modules from Open WebUI functions"
	@echo "  update-modules       Force update of shared modules and rebuild docs"
	@echo "  test-build          Test documentation build (like GitHub Actions)"
	@echo "  test-deps           Install documentation dependencies"
	@echo "  pre-commit-test     Quick test (assumes deps installed)"

.PHONY: help Makefile copy-shared-modules update-modules test-build test-deps pre-commit-test

# Copy shared modules to static directory
copy-shared-modules:
	@echo "Copying shared modules to documentation static files..."
	@mkdir -p $(STATIC_JS_DIR)
	@cp $(SHARED_JS_DIR)/execution_plan_viewer.js $(STATIC_JS_DIR)/
	@echo "Shared modules copied successfully."

# Force update of shared modules and rebuild HTML
update-modules: copy-shared-modules
	@echo "Rebuilding documentation with updated modules..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Enhanced build target that includes shared modules
html: copy-shared-modules
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Enhanced livehtml target that includes shared modules
livehtml: copy-shared-modules
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile copy-shared-modules
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Install documentation dependencies (like GitHub Actions)
test-deps:
	@echo "📦 Installing documentation dependencies..."
	cd .. && uv sync --extra docs

# Test documentation build (replicates GitHub Actions)
test-build: test-deps clean copy-shared-modules
	@echo "\033[1;34m🔨 Testing documentation build (GitHub Actions simulation)...\033[0m"
	@echo "   This shows the same errors/warnings as CI"
	@echo ""
	@# Capture build output and filter warnings/errors with color support
	@FORCE_COLOR=1 $(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) 2>&1 | tee build.log
	@echo ""
	@echo "\033[1;36m📊 BUILD SUMMARY\033[0m"
	@echo "\033[1;36m================\033[0m"
	@# Extract and display ALL warnings and errors with color
	@if grep -E "(WARNING|ERROR):" build.log > build_issues.log 2>/dev/null; then \
		echo "\033[1;33m⚠️  Issues found ($$(wc -l < build_issues.log 2>/dev/null || echo 0)):\033[0m" ; \
		echo "" ; \
		sed 's/WARNING:/\x1b[33mWARNING:\x1b[0m/g; s/ERROR:/\x1b[31mERROR:\x1b[0m/g' build_issues.log ; \
		echo "" ; \
		echo "\033[1;34m📄 Full log saved to: build.log\033[0m" ; \
	else \
		echo "\033[1;32m✅ No warnings or errors found!\033[0m" ; \
	fi
	@echo ""
	@# Check if build succeeded
	@if [ -f "$(BUILDDIR)/html/index.html" ]; then \
		echo "\033[1;32m✅ Documentation build successful!\033[0m" ; \
		echo "\033[1;34m📂 Built files are in: $(BUILDDIR)/html/\033[0m" ; \
		echo "\033[1;34m🌐 Open $(BUILDDIR)/html/index.html in your browser\033[0m" ; \
	else \
		echo "\033[1;31m❌ Documentation build failed!\033[0m" ; \
		echo "\033[1;33m🔧 Check the issues above and build.log for details\033[0m" ; \
		exit 1 ; \
	fi
	@# Clean up temp files
	@rm -f build_issues.log

# Quick pre-commit test (faster, assumes deps installed)
pre-commit-test: clean copy-shared-modules
	@echo "🚀 Quick documentation build test..."
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "✅ Pre-commit test passed!"

clean-build:
	rm -rf "$(BUILDDIR)"/*
	rm -rf $(STATIC_JS_DIR)
