#!/usr/bin/env python3
"""Cross-platform downloader for Zenodo archives and pretrained weights."""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_public"
RAW.mkdir(parents=True, exist_ok=True)

ZENODO_FILES = [
    {
        "url": "https://zenodo.org/api/records/10535022/files/HAPPYData-20240119T100520Z-001.zip/content",
        "target": RAW / "HAPPYData-20240119T100520Z-001.zip",
        "expected_size": 2_141_793_908,
        "expected_md5": "e8af7d9a349051e209c9d982bc44edcf",
    },
    {
        "url": "https://zenodo.org/api/records/10535022/files/HAPPYData-20240119T100520Z-002.zip/content",
        "target": RAW / "HAPPYData-20240119T100520Z-002.zip",
        "expected_size": 452_945_061,
        "expected_md5": "e39adcab4ea865f8d498d62ced6b1ca6",
    },
]

PRETRAINED_WEIGHTS = [
    {
        "url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "target": ROOT / "environment" / "pretrained" / "resnet101-5d3b4d8f.pth",
        "sha_prefix": "5d3b4d8f",
    },
    {
        "url": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        "target": ROOT / "environment" / "torch_cache" / "hub" / "checkpoints" / "resnet50-0676ba61.pth",
        "sha_prefix": "0676ba61",
    },
]


def download_with_resume(url: str, target: Path, expected_size: int | None = None, max_retries: int = 50) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    retries = 0
    while True:
        current_size = target.stat().st_size if target.exists() else 0
        if expected_size and current_size == expected_size:
            print(f"Already complete: {target.name} ({current_size:,} bytes)")
            return
        if expected_size and current_size > expected_size:
            print(f"Target {target.name} larger than expected, restarting...")
            target.unlink()
            current_size = 0

        req = urllib.request.Request(url)
        mode = "wb"
        if current_size > 0:
            req.add_header("Range", f"bytes={current_size}-")
            mode = "ab"
            print(f"Resuming {target.name} from {current_size:,} bytes...")
        else:
            print(f"Downloading {target.name} from {url}...")

        try:
            with urllib.request.urlopen(req, timeout=45) as resp, target.open(mode) as out_file:
                if mode == "ab" and resp.status == 200:
                    print("Server does not support Range; downloading from start...")
                    out_file.close()
                    out_file = target.open("wb")
                    current_size = 0

                total = expected_size or (int(resp.headers.get("Content-Length", 0)) + current_size)
                downloaded = current_size
                last_report = time.time()
                chunk_size = 1024 * 1024  # 1 MB

                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_report > 3:
                        pct = (downloaded / total * 100) if total else 0
                        print(f"  {target.name}: {downloaded:,} / {total:,} bytes ({pct:.1f}%)", flush=True)
                        last_report = time.time()

            current_size = target.stat().st_size
            if expected_size and current_size < expected_size:
                print(f"Connection closed early at {current_size:,} / {expected_size:,} bytes. Resuming in 2s...")
                time.sleep(2)
                continue
            pct = (current_size / total * 100) if total else 100
            print(f"  Finished {target.name}: {current_size:,} bytes ({pct:.1f}%)")
            break
        except urllib.error.HTTPError as e:
            if e.code == 416:
                print(f"Server returned 416 (Range Not Satisfiable); {target.name} is already complete.")
                break
            retries += 1
            print(f"Download interrupted ({e}). Retry {retries}/{max_retries} in 3s...", flush=True)
            if retries >= max_retries:
                raise
            time.sleep(3)
        except Exception as e:
            retries += 1
            print(f"Download interrupted ({e}). Retry {retries}/{max_retries} in 3s...", flush=True)
            if retries >= max_retries:
                raise
            time.sleep(3)


def check_md5(path: Path, expected_md5: str) -> bool:
    print(f"Verifying MD5 for {path.name}...")
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_md5:
        print(f"MD5 mismatch for {path.name}: {actual} != {expected_md5}", file=sys.stderr)
        return False
    print(f"MD5 check passed for {path.name} ({actual})")
    return True


def check_sha_prefix(path: Path, expected_prefix: str) -> bool:
    print(f"Verifying SHA256 prefix for {path.name}...")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()[:len(expected_prefix)]
    if actual != expected_prefix:
        print(f"SHA256 mismatch for {path.name}: {actual} != {expected_prefix}", file=sys.stderr)
        return False
    print(f"SHA256 check passed for {path.name}")
    return True


def main() -> int:
    # 1. Download pretrained weights
    print("\n=== Pretrained ImageNet Weights ===")
    for item in PRETRAINED_WEIGHTS:
        if item["target"].exists() and check_sha_prefix(item["target"], item["sha_prefix"]):
            print(f"Pretrained weights already valid: {item['target'].name}")
            continue
        download_with_resume(item["url"], item["target"])
        if not check_sha_prefix(item["target"], item["sha_prefix"]):
            return 1

    # 2. Download Zenodo archives
    print("\n=== Zenodo Dataset Archives ===")
    for item in ZENODO_FILES:
        download_with_resume(item["url"], item["target"], item["expected_size"])
        if not check_md5(item["target"], item["expected_md5"]):
            return 1

    print("\nAll required datasets and pretrained weights are successfully downloaded and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
