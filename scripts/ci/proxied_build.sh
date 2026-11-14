#!/usr/bin/env bash
# Prove that a template image build routes apt and pip through a facility's
# proxy, from the proxy's access log rather than the build's exit status: the
# runner has open egress, so a build whose lowercase-proxy bridge was deleted
# still succeeds.
#
# The template Dockerfiles bridge the uppercase proxy variables a facility hands
# a build into the lowercase ones apt reads. The build therefore gets ONLY the
# uppercase build-args, and runs under podman: BuildKit copies HTTP_PROXY into
# http_proxy inside RUN, which would satisfy the log assertion with the bridge
# deleted. `probe` re-checks on every run that podman does not do the same.
#
# ci.yml's dockerfile-e2e runs the subcommands as five steps, in this order:
#   start   run tinyproxy (LogLevel Info: the request lines read below are
#           logged at Info) and wait for its port with a bare TCP connect, since
#           a probe that fetched through it would write the log under test.
#           Needs PROXY_BASE_IMAGE.
#   probe   build a one-line Dockerfile with the uppercase build-args and fail
#           unless the uppercase value, and no lowercase one, reaches RUN.
#           Needs PROBE_BASE_IMAGE; only its pull is retried.
#   build   build the web-terminal auth sidecar through the proxy. OSPREY_VERSION
#           defaults to the installed osprey's; OSPREY_DEV=1 lets an unreleased
#           pin fall back to an unpinned install.
#   assert  fail unless both deb.debian.org (the bridge) and PyPI (the
#           framework install) requests reached the proxy.
#   stop    print the log's tail and remove the proxy; never fails.
#
# Usage: scripts/ci/proxied_build.sh start|probe|build|assert|stop
#   OSPREY_PROXY_DIR   the config and access-log directory (default /tmp/osprey-proxy).
#   OSPREY_PROXY_PORT  the port on 127.0.0.1 (default 8888).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
proxy_dir="${OSPREY_PROXY_DIR:-/tmp/osprey-proxy}"
port="${OSPREY_PROXY_PORT:-8888}"
proxy_url="http://127.0.0.1:$port"
access_log="$proxy_dir/access.log"
container=osprey-ci-proxy
sidecar_context=src/osprey/templates/modules/web_terminals/auth_sidecar

start() {
  : "${PROXY_BASE_IMAGE:?set PROXY_BASE_IMAGE to the image tinyproxy is installed into}"
  rm -rf "$proxy_dir"
  mkdir -p "$proxy_dir"
  cat > "$proxy_dir/tinyproxy.conf" <<CONF
Port $port
Timeout 600
MaxClients 100
LogLevel Info
LogFile "/proxy/access.log"
CONF
  : > "$access_log"
  chmod 666 "$access_log"
  docker run -d --name "$container" --network host \
    -v "$proxy_dir:/proxy" "$PROXY_BASE_IMAGE" \
    sh -c 'apt-get update -qq \
           && apt-get install -y -qq --no-install-recommends tinyproxy \
           && exec tinyproxy -d -c /proxy/tinyproxy.conf'
  for _ in $(seq 60); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      echo "✅ tinyproxy is listening on 127.0.0.1:$port"
      return 0
    fi
    sleep 2
  done
  echo "❌ tinyproxy never came up on 127.0.0.1:$port"
  docker logs "$container" || true
  return 1
}

probe() {
  : "${PROBE_BASE_IMAGE:?set PROBE_BASE_IMAGE to the base image of the probe build}"
  local ctx out rc attempt pulled=
  ctx="$(mktemp -d)"
  printf 'FROM %s\nRUN env | sort | grep -i proxy || echo "NO-PROXY-ENV"\n' \
    "$PROBE_BASE_IMAGE" > "$ctx/Dockerfile"
  # Retry only the base-image pull: the measured build runs once from local
  # storage, so the property under test is never retried into passing.
  for attempt in 1 2 3; do
    if podman pull "$PROBE_BASE_IMAGE"; then
      pulled=yes
      break
    fi
    echo "pull attempt $attempt of 3 failed; retrying in $((attempt * 5))s"
    sleep "$((attempt * 5))"
  done
  if [ -z "$pulled" ]; then
    echo "❌ could not fetch the probe base image in 3 attempts — a registry"
    echo "   failure, which says nothing about the proxy contract below"
    return 1
  fi
  # `|| rc=$?`: under `set -e` a bare assignment would abort before the output is printed.
  rc=0
  out="$(podman build --no-cache --pull=never --http-proxy=false \
    --build-arg "HTTP_PROXY=$proxy_url" \
    --build-arg "HTTPS_PROXY=$proxy_url" \
    "$ctx" 2>&1)" || rc=$?
  rm -rf "$ctx"
  printf '%s\n' "$out"
  if [ "$rc" -ne 0 ]; then
    echo "❌ the probe build failed (podman exit $rc); its output is above"
    return 1
  fi
  if ! grep -qF "HTTP_PROXY=$proxy_url" <<<"$out"; then
    echo "❌ the uppercase HTTP_PROXY build-arg never reached the RUN environment"
    return 1
  fi
  if grep -q 'http_proxy=' <<<"$out"; then
    echo "❌ this build runtime injects a lowercase proxy variable of its own —"
    echo "   the access-log assertion would then pass with the Dockerfile's"
    echo "   lowercase bridge deleted, i.e. it would prove nothing."
    return 1
  fi
  echo "✅ uppercase in, uppercase only — the bridge is the sole path to apt"
}

build() {
  local version="${OSPREY_VERSION:-}" ctx
  if [ -z "$version" ]; then
    version="$(cd "$repo_root" && uv run --no-sync python -c 'import osprey; print(osprey.__version__)')"
  fi
  echo "OSPREY_VERSION=$version"
  ctx="$(mktemp -d)"
  cp -a "$repo_root/$sidecar_context/." "$ctx/"
  : > "$ctx/osprey-local-requirements.txt"
  # `--http-proxy=false` drops buildah's copy of the host's proxy variables, so
  # the build-args are the only ones.
  podman build --no-cache --network=host --http-proxy=false \
    --build-arg "HTTP_PROXY=$proxy_url" \
    --build-arg "HTTPS_PROXY=$proxy_url" \
    --build-arg "OSPREY_VERSION=$version" \
    --build-arg OSPREY_DEV=1 \
    --tag osprey-auth-sidecar:proxied-ci \
    "$ctx"
  rm -rf "$ctx"
}

assert_log() {
  local apt_hits pip_hits
  if [ ! -s "$access_log" ]; then
    echo "❌ proxy access log missing or empty at $access_log — nothing was proxied at all"
    docker logs "$container" || true
    return 1
  fi
  apt_hits="$(grep -F 'deb.debian.org' "$access_log" || true)"
  pip_hits="$(grep -E 'pypi\.org|files\.pythonhosted\.org' "$access_log" || true)"
  echo "── apt traffic seen by the proxy ──────────────────────────────"
  head -n 20 <<<"${apt_hits:-<none>}"
  echo "── pip traffic seen by the proxy ──────────────────────────────"
  head -n 20 <<<"${pip_hits:-<none>}"
  if [ -z "$apt_hits" ]; then
    echo "❌ no deb.debian.org request reached the proxy: the lowercase-proxy"
    echo "   bridge in the auth-sidecar Dockerfile is not doing its job."
    return 1
  fi
  if [ -z "$pip_hits" ]; then
    echo "❌ no pypi.org / files.pythonhosted.org request reached the proxy:"
    echo "   the framework install bypassed it."
    return 1
  fi
  echo "✅ both apt and pip traffic went through the proxy"
}

stop() {
  if [ -f "$access_log" ]; then
    echo "── last 40 proxy access-log lines ─────────────────────────────"
    tail -n 40 "$access_log" || true
  fi
  docker rm -f "$container" > /dev/null 2>&1 || true
}

case "${1:-}" in
  start) start ;;
  probe) probe ;;
  build) build ;;
  assert) assert_log ;;
  stop) stop ;;
  *)
    echo "usage: $0 start|probe|build|assert|stop" >&2
    exit 2
    ;;
esac
