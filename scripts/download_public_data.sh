#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw_public"
mkdir -p "$RAW/bioimage_archive"

download() {
  local url="$1" target="$2" expected_size="$3"
  if [[ -f "$target" ]] && [[ "$(stat -f '%z' "$target")" -eq "$expected_size" ]]; then
    echo "Already complete: $target"
    return
  fi
  echo "Downloading/resuming: $target"
  curl --fail --location --show-error --silent --retry 4 --retry-all-errors -C - "$url" -o "$target"
  local actual_size
  actual_size="$(stat -f '%z' "$target")"
  if [[ "$actual_size" -ne "$expected_size" ]]; then
    echo "Incomplete $target: $actual_size bytes; expected $expected_size" >&2
    exit 2
  fi
}

download 'https://zenodo.org/api/records/10535022/files/HAPPYData-20240119T100520Z-001.zip/content' \
  "$RAW/HAPPYData-20240119T100520Z-001.zip" 2141793908
download 'https://zenodo.org/api/records/10535022/files/HAPPYData-20240119T100520Z-002.zip/content' \
  "$RAW/HAPPYData-20240119T100520Z-002.zip" 452945061
md5 -q "$RAW/HAPPYData-20240119T100520Z-001.zip" | grep -qx 'e8af7d9a349051e209c9d982bc44edcf'
md5 -q "$RAW/HAPPYData-20240119T100520Z-002.zip" | grep -qx 'e39adcab4ea865f8d498d62ced6b1ca6'
unzip -t "$RAW/HAPPYData-20240119T100520Z-001.zip"
unzip -t "$RAW/HAPPYData-20240119T100520Z-002.zip"

download 'https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/045/S-BIAD1045/Files/96-2019-09-05%2022.47.40.ndpi' \
  "$RAW/bioimage_archive/96-2019-09-05 22.47.40.ndpi" 2686927912
download 'https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/045/S-BIAD1045/Files/59B.svs' \
  "$RAW/bioimage_archive/59B.svs" 2929818169
echo "All four public data files passed their available size/checksum checks."
