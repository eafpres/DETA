#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:48:07 2026

@author: eafpres

Flatten COCO image file_name paths to basenames only."""
#
# libraries
#
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
#
# helpers
#
def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    argparse.Namespace: Parsed args.
  """
  parser = argparse.ArgumentParser(
    description = (
      "Rewrite COCO images[].file_name to basename only, removing "
      "folder prefixes."
    )
  )
  parser.add_argument(
    "--input-json",
    required = True,
    type = str,
    help = "Path to input COCO JSON."
  )
  parser.add_argument(
    "--output-json",
    required = True,
    type = str,
    help = "Path to output COCO JSON."
  )
  parser.add_argument(
    "--fail-on-duplicates",
    action = "store_true",
    help = (
      "Fail if flattened basenames are not unique. Recommended."
    )
  )
  parser.add_argument(
    "--dry-run",
    action = "store_true",
    help = "Report what would change without writing output."
  )
  return parser.parse_args()

def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON file.

  Args:
    path: Input path.

  Returns:
    Dict[str, Any]: Parsed JSON.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def save_json(data: Dict[str, Any], path: Path) -> None:
  """Write JSON file.

  Args:
    data: JSON payload.
    path: Output path.
  """
  path.parent.mkdir(parents = True, exist_ok = True)
  with path.open("w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)

def flatten_file_names(
  coco: Dict[str, Any],
  fail_on_duplicates: bool,
) -> Dict[str, Any]:
  """Flatten image file_name values to basename only.

  Args:
    coco: COCO dataset.
    fail_on_duplicates: Whether to fail on basename collisions.

  Returns:
    Dict[str, Any]: Updated COCO dataset.
  """
  images: List[Dict[str, Any]] = coco.get("images", [])
  basenames = [
    Path(str(image_rec["file_name"])).name
    for image_rec in images
  ]
  counts = Counter(basenames)
  dupes = sorted([
    name for name, count in counts.items()
    if count > 1
  ])
  if fail_on_duplicates and len(dupes) > 0:
    sample = dupes[:20]
    raise ValueError(
      "duplicate basenames after flattening; examples: "
      + ", ".join(sample)
    )
  for image_rec in images:
    image_rec["file_name"] = Path(str(image_rec["file_name"])).name
  return coco

def main() -> None:
  """Run file_name flattening."""
  args = parse_args()
  input_json = Path(args.input_json).resolve()
  output_json = Path(args.output_json).resolve()
  coco = load_json(input_json)
  original_images = coco.get("images", [])
  original_names = [
    str(image_rec["file_name"])
    for image_rec in original_images
  ]
  coco = flatten_file_names(
    coco = coco,
    fail_on_duplicates = args.fail_on_duplicates,
  )
  flattened_names = [
    str(image_rec["file_name"])
    for image_rec in coco.get("images", [])
  ]
  changed = sum(
    old != new
    for old, new in zip(original_names, flattened_names)
  )
  print(f"images: {len(flattened_names)}")
  print(f"changed file_name values: {changed}")
  if len(flattened_names) > 0:
    print(f"sample before: {original_names[0]}")
    print(f"sample after:  {flattened_names[0]}")
  if not args.dry_run:
    save_json(coco, output_json)
    print(f"wrote: {output_json}")

if __name__ == "__main__":
  main()
