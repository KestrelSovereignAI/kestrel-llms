#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
usage: scripts/llm_provider_packages.sh <command>

commands:
  list       Print provider package directories
  build      Build every provider package and meta-package
  clean      Remove provider package build artifacts
  compile    Compile provider source packages
  test       Run provider unit tests
  verify     Run clean, compile, build, and test
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

list_packages() {
  find "$repo_root/providers" -mindepth 1 -maxdepth 1 -type d -name 'kestrel-*' | sort
}

clean_packages() {
  rm -rf "$repo_root/dist" "$repo_root/build" "$repo_root"/*.egg-info
  while IFS= read -r pkg; do
    rm -rf "$pkg/dist" "$pkg/build" "$pkg"/*.egg-info
    find "$pkg" -type d -name __pycache__ -prune -exec rm -rf {} +
  done < <(list_packages)
}

compile_packages() {
  local srcs=()
  while IFS= read -r pkg; do
    if [[ -d "$pkg/src" ]]; then
      srcs+=("$pkg/src")
    fi
  done < <(list_packages)
  if [[ ${#srcs[@]} -gt 0 ]]; then
    uv run python -m compileall "${srcs[@]}"
  fi
}

build_packages() {
  while IFS= read -r pkg; do
    echo "==> Building ${pkg#$repo_root/}"
    (cd "$pkg" && uv build)
  done < <(list_packages)
}

test_packages() {
  uv run --group test pytest "$repo_root/tests/unit/providers"
}

cmd="${1:-}"
case "$cmd" in
  list)
    list_packages
    ;;
  clean)
    clean_packages
    ;;
  compile)
    compile_packages
    ;;
  build)
    build_packages
    ;;
  test)
    test_packages
    ;;
  verify)
    clean_packages
    compile_packages
    build_packages
    test_packages
    ;;
  *)
    usage
    exit 2
    ;;
esac
