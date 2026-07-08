#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:34:00 2026

@author: eafpres
"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--image-root",
    required=True,
    help="Original image root, e.g. /mnt/x/common_images"
  )
  parser.add_argument(
    "--coco-jsons",
    required=True,
    nargs="+",
    help="One or more COCO annotation JSON files"
  )
  parser.add_argument(
    "--cache-root",
    required=True,
    help="Local cache root for copied images"
  )
  parser.add_argument(
    "--clear-existing",
    action="store_true",
    help="Delete existing cache-root before copying"
  )
  return parser.parse_args()


def load_required_files(coco_jsons):
  required = set()
  for coco_json in coco_jsons:
    with open(coco_json, "r", encoding="utf-8") as f:
      data = json.load(f)
    for image in data.get("images", []):
      file_name = image.get("file_name")
      if file_name:
        required.add(file_name)
  return sorted(required)


def copy_required_images(image_root, cache_root, required_files):
  missing = []
  copied = 0
  skipped = 0
  for file_name in tqdm(
    required_files,
    desc="caching images",
    ncols=120,
    colour="blue"
  ):
    src = image_root / file_name
    dst = cache_root / file_name
    if not src.exists():
      missing.append(file_name)
      continue
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
      skipped += 1
      continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1
  return copied, skipped, missing


def main():
  args = parse_args()
  image_root = Path(args.image_root)
  cache_root = Path(args.cache_root)
  if args.clear_existing and cache_root.exists():
    shutil.rmtree(cache_root)
  cache_root.mkdir(parents=True, exist_ok=True)
  required_files = load_required_files(args.coco_jsons)
  print(f"required images: {len(required_files):,}")
  copied, skipped, missing = copy_required_images(
    image_root=image_root,
    cache_root=cache_root,
    required_files=required_files
  )
  print(f"copied: {copied:,}")
  print(f"skipped existing: {skipped:,}")
  print(f"missing: {len(missing):,}")
  if missing:
    missing_path = cache_root / "missing_images.txt"
    missing_path.write_text("\n".join(missing), encoding="utf-8")
    print(f"missing list written to: {missing_path}")


if __name__ == "__main__":
  main()
