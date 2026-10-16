#!/usr/bin/env python3
"""Verify and safely extract the official HAPPY Zenodo archives."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "HAPPYData-20240119T100520Z-001.zip": (2_141_793_908, "e8af7d9a349051e209c9d982bc44edcf"),
    "HAPPYData-20240119T100520Z-002.zip": (452_945_061, "e39adcab4ea865f8d498d62ced6b1ca6"),
}
RAW = ROOT / "data" / "raw_public"
OUT = ROOT / "data" / "prepared"
EXPECTED_DIRS = ("annotations", "datasets", "embeddings", "graph_splits", "slides", "trained_models")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    base = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (destination / info.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {info.filename}")
        zf.extractall(destination)


def merge_tree(source: Path, destination: Path) -> None:
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item.stat().st_size or md5(target) != md5(item):
                raise RuntimeError(f"Conflicting duplicate path while merging archives: {relative}")
            item.unlink()
        else:
            shutil.move(str(item), str(target))


def main() -> int:
    verified: list[Path] = []
    for name, (expected_size, expected_md5) in ARCHIVES.items():
        path = RAW / name
        if not path.is_file():
            print(f"Missing archive: {path}", file=sys.stderr)
            return 2
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            print(f"Incomplete archive {name}: {actual_size} bytes, expected {expected_size}", file=sys.stderr)
            return 2
        actual_md5 = md5(path)
        if actual_md5 != expected_md5:
            print(f"MD5 mismatch for {name}: {actual_md5}, expected {expected_md5}", file=sys.stderr)
            return 2
        try:
            result = subprocess.run(["unzip", "-t", str(path)], capture_output=True, text=True)
            if result.returncode:
                print(result.stdout + result.stderr, file=sys.stderr)
                return result.returncode
        except FileNotFoundError:
            with zipfile.ZipFile(path) as zf:
                bad_file = zf.testzip()
                if bad_file:
                    print(f"CRC check failed for {bad_file} in {name}", file=sys.stderr)
                    return 2
        verified.append(path)

    if OUT.exists():
        print(f"Refusing to overwrite existing extracted data: {OUT}", file=sys.stderr)
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".happy-extract-", dir=OUT.parent) as temp:
        temp_root = Path(temp)
        combined = temp_root / "combined"
        combined.mkdir()
        for archive in verified:
            stage = temp_root / archive.stem
            stage.mkdir()
            safe_extract(archive, stage)
            # Both Zenodo zips place their payload under a common HAPPYData/ root.
            # Merge its contents so consumers see data/prepared/{annotations,datasets,...}.
            payload = stage / "HAPPYData"
            if not payload.is_dir():
                print(f"Expected HAPPYData/ payload root missing in {archive.name}", file=sys.stderr)
                return 3
            merge_tree(payload, combined)

        missing = [name for name in EXPECTED_DIRS if not (combined / name).is_dir()]
        if missing:
            print("Archive layout did not match the published record; missing: " + ", ".join(missing), file=sys.stderr)
            return 3
        shutil.move(str(combined), str(OUT))
    print(f"Verified and extracted both official Zenodo archives to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
