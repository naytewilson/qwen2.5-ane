#!/usr/bin/env bash
# Portable Git ignore regression: generated artifact names must ignore directories AND symlinks.
# Run from any checkout: bash path/to/test-gitignore-symlinks.sh [repo-root]
set -euo pipefail

repo=${1:-$(git rev-parse --show-toplevel)}
repo=$(cd "$repo" && pwd -P)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/gitignore-symlinks.XXXXXXXX")
trap 'rm -rf "$scratch"' EXIT
failures=0
checks=0
files=0

is_artifact() {
  case "${1##*/}" in
    .build|.swiftpm|DerivedData|node_modules|dist|build|target|build-san|web-build|models_build|.venv-*|.venv|venv|.next|.turbo|.gradle|*.mlpackage|*.mlmodelc|*.app) return 0 ;;
    *) return 1 ;;
  esac
}

# Control: the original directory-only spelling MUST fail for a symlink.
mkdir -p "$scratch/control"
git -C "$scratch/control" init -q
printf '.build/\n' > "$scratch/control/.gitignore"
ln -s non-existent "$scratch/control/.build"
if git -C "$scratch/control" check-ignore -q -- .build; then
  echo 'FAIL: the negative control unexpectedly ignored a symlink' >&2
  exit 1
fi

while IFS= read -r -d '' tracked; do
  case "$tracked" in .gitignore|*/.gitignore) ;; *) continue ;; esac
  [ -f "$repo/$tracked" ] || continue
  files=$((files + 1))
  fixture="$scratch/fixture-$files"
  mkdir -p "$fixture"
  git -C "$fixture" init -q
  cp "$repo/$tracked" "$fixture/.gitignore"

  case_checks=0
  lineno=0
  while IFS= read -r rule || [ -n "$rule" ]; do
    lineno=$((lineno + 1))
    rule=${rule%$'\r'}
    case "$rule" in ''|\#*|\!*) continue ;; esac
    normalized=${rule#/}
    basename=${normalized%/}
    if ! is_artifact "$basename"; then continue; fi

    if [ "$normalized" != "$basename" ]; then
      echo "FAIL: $tracked:$lineno ignores only a directory: $rule" >&2
      failures=$((failures + 1))
      continue
    fi

    # Materialize one representative path for patterns containing wildcard directories.
    relative=${normalized//\*\*/probe}
    relative=${relative//\*/probe}
    case "$relative" in *\[*|*\]*|*\?*) continue ;; esac
    mkdir -p "$fixture/$(dirname "$relative")"
    ln -s non-existent "$fixture/$relative"
    if ! git -C "$fixture" check-ignore -q -- "$relative"; then
      echo "FAIL: $tracked:$lineno does not ignore symlink $relative" >&2
      failures=$((failures + 1))
    elif [ -n "$(git -C "$fixture" status --porcelain=v1 -- "$relative")" ]; then
      echo "FAIL: $tracked:$lineno symlink dirties the checkout: $relative" >&2
      failures=$((failures + 1))
    fi
    rm "$fixture/$relative"
    mkdir -p "$fixture/$relative"
    if ! git -C "$fixture" check-ignore -q -- "$relative/"; then
      echo "FAIL: $tracked:$lineno does not ignore directory $relative" >&2
      failures=$((failures + 1))
    fi
    rmdir "$fixture/$relative"
    checks=$((checks + 1))
    case_checks=$((case_checks + 1))
  done < "$repo/$tracked"
  if [ "$case_checks" -gt 0 ] && git -C "$fixture" check-ignore -q -- "unrelated-source.swift"; then
    echo "FAIL: $tracked unexpectedly ignores ordinary source" >&2
    failures=$((failures + 1))
  fi
done < <(git -C "$repo" ls-files -z)

[ "$checks" -gt 0 ] || {
  echo 'FAIL: no expected artifact patterns were tested' >&2
  exit 1
}
if [ "$failures" -ne 0 ]; then
  echo "FAIL: $failures ignore-rule violations ($checks positives, $files ignore files)" >&2
  exit 1
fi
echo "PASS: $checks artifact symlink+directory rules, $files tracked ignore files, negative control"
