#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NUC="$ROOT/environment/pretrained/resnet101-5d3b4d8f.pth"
CELL="$ROOT/environment/torch_cache/hub/checkpoints/resnet50-0676ba61.pth"
mkdir -p "$(dirname "$NUC")" "$(dirname "$CELL")"

fetch() {
  local url="$1" target="$2" sha_prefix="$3"
  local actual
  actual=""
  if [[ -f "$target" ]]; then
    actual="$(shasum -a 256 "$target" | awk '{print substr($1, 1, 8)}')"
  fi
  if [[ "$actual" != "$sha_prefix" ]]; then
    curl --fail --location --show-error --retry 4 --retry-all-errors -C - "$url" -o "$target"
    actual="$(shasum -a 256 "$target" | awk '{print substr($1, 1, 8)}')"
  fi
  if [[ "$actual" != "$sha_prefix" ]]; then
    echo "SHA256 prefix mismatch for $target: $actual; expected $sha_prefix" >&2
    exit 2
  fi
}

fetch 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth' "$NUC" 5d3b4d8f
fetch 'https://download.pytorch.org/models/resnet50-0676ba61.pth' "$CELL" 0676ba61
echo "Verified official ImageNet initialisation weights in environment/{pretrained,torch_cache}."
