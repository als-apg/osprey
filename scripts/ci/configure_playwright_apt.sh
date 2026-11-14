#!/usr/bin/env bash
# Prepare apt for `playwright install --with-deps` on a GitHub-hosted Ubuntu runner.
#
# Azure's apt mirror some days serves the ~140 MB of system debs at 30-100 kB/s, so
# this prefers archive.ubuntu.com over it, keeps the downloaded debs in
# ~/apt-archives (which the workflow caches) and gives each fetch a short timeout.
# The apt.conf snippet's name must sort after the image's `zz-retries`.
#
# Usage: scripts/ci/configure_playwright_apt.sh   (needs sudo; Linux runners only)
set -euo pipefail

mirrors=/etc/apt/apt-mirrors.txt
sudo sed -i -E \
  -e 's#^(http://azure\.archive\.ubuntu\.com/ubuntu/\t)priority:1$#\1priority:2#' \
  -e 's#^(https?://archive\.ubuntu\.com/ubuntu/\t)priority:2$#\1priority:1#' \
  "$mirrors"
cat "$mirrors"
grep -Eq '^https?://archive\.ubuntu\.com/ubuntu/[[:space:]]+priority:1$' "$mirrors" \
  || echo "::warning::apt-mirrors.txt not in the expected shape; apt keeps the image's mirror order"
mkdir -p "$HOME/apt-archives/partial"
sudo tee /etc/apt/apt.conf.d/zzz-playwright-deps-cache >/dev/null <<EOF
Dir::Cache::Archives "$HOME/apt-archives";
APT::Keep-Downloaded-Packages "true";
Acquire::http::Timeout "15";
Acquire::Retries "3";
EOF
