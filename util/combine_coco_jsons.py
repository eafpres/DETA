#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:30:33 2026

@author: eafpres
"""

import argparse
import json
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def resolve_path(path, must_exist = False):
  """Resolve a path relative to the DETA repo root.
  Args:
    path: Input path string.
    must_exist: Whether the path must already exist.
  Returns:
    pathlib.Path: Resolved path.
  """
  path = Path(path).expanduser()
  if not path.is_absolute():
    path = ROOT / path
  path = path.resolve()
  if must_exist and not path.exists():
    raise FileNotFoundError(f"path does not exist: {path}")
  return path
def load_json(path):
  """Load a JSON file.
  Args:
    path: Input JSON path.
  Returns:
    dict: Parsed JSON.
  """
  with Path(path).open("r", encoding = "utf-8") as f:
    return json.load(f)
def save_json(data, path):
  """Save a JSON file.
  Args:
    data: JSON-compatible object.
    path: Output JSON path.
  """
  path = Path(path)
  path.parent.mkdir(parents = True, exist_ok = True)
  with path.open("w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)
def category_signature(data):
  """Create a category signature.
  Args:
    data: COCO dictionary.
  Returns:
    list[tuple[int, str]]: Category id/name pairs.
  """
  return sorted(
    (cat["id"], cat["name"])
    for cat in data.get("categories", [])
  )
def validate_categories(base, other, other_path):
  """Validate category consistency.
  Args:
    base: Base COCO dictionary.
    other: Other COCO dictionary.
    other_path: Other file path for error text.
  """
  base_sig = category_signature(base)
  other_sig = category_signature(other)
  if base_sig != other_sig:
    raise ValueError(
      "category mismatch for "
      f"{other_path}\nbase={base_sig}\nother={other_sig}"
    )
def make_unique_file_name(file_name, seen, source_label, mode):
  """Handle file-name collisions.
  Args:
    file_name: Original file name.
    seen: Set of seen file names.
    source_label: Input file stem.
    mode: Collision handling mode.
  Returns:
    str | None: Output file name, or None if skipped.
  """
  if file_name not in seen:
    seen.add(file_name)
    return file_name
  if mode == "error":
    raise ValueError(f"duplicate file_name found: {file_name}")
  if mode == "skip":
    return None
  if mode == "prefix_source":
    candidate = f"{source_label}__{file_name}"
  elif mode == "prefix_counter":
    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    idx = 1
    candidate = f"{stem}__dup{idx}{suffix}"
    while candidate in seen:
      idx += 1
      candidate = f"{stem}__dup{idx}{suffix}"
  else:
    raise ValueError(f"unknown collision mode: {mode}")
  if candidate in seen:
    raise ValueError(f"duplicate after collision handling: {candidate}")
  seen.add(candidate)
  return candidate
def category_counts(data):
  """Count annotations by category id.
  Args:
    data: COCO dictionary.
  Returns:
    collections.Counter: Annotation counts by category id.
  """
  return Counter(
    ann["category_id"]
    for ann in data.get("annotations", [])
  )
def print_summary(label, data):
  """Print a COCO summary.
  Args:
    label: Summary label.
    data: COCO dictionary.
  """
  counts = category_counts(data)
  cat_id_to_name = {
    cat["id"]: cat["name"]
    for cat in data.get("categories", [])
  }
  print(f"\n{label}")
  print(f"  images: {len(data.get('images', []))}")
  print(f"  annotations: {len(data.get('annotations', []))}")
  for cat_id, name in sorted(cat_id_to_name.items()):
    print(f"  {cat_id}: {name}: {counts.get(cat_id, 0)}")
def combine_coco(files, collision_mode):
  """Combine multiple COCO JSON files.
  Args:
    files: Input JSON paths.
    collision_mode: File-name collision handling mode.
  Returns:
    dict: Combined COCO dictionary.
  """
  base = load_json(files[0])
  combined = {
    "info": dict(base.get("info", {})),
    "licenses": base.get("licenses", []),
    "categories": base.get("categories", []),
    "images": [],
    "annotations": []
  }
  combined["info"]["description"] = (
    str(combined["info"].get("description", "COCO dataset"))
    + " | combined"
  )
  seen_file_names = set()
  next_image_id = 0
  next_ann_id = 0
  for file_path in files:
    data = load_json(file_path)
    validate_categories(base, data, file_path)
    print_summary(str(file_path), data)
    source_label = Path(file_path).stem
    image_id_map = {}
    for image in data.get("images", []):
      out_file_name = make_unique_file_name(
        file_name = image["file_name"],
        seen = seen_file_names,
        source_label = source_label,
        mode = collision_mode
      )
      if out_file_name is None:
        continue
      new_image = dict(image)
      old_image_id = image["id"]
      new_image["id"] = next_image_id
      new_image["file_name"] = out_file_name
      image_id_map[old_image_id] = next_image_id
      combined["images"].append(new_image)
      next_image_id += 1
    for ann in data.get("annotations", []):
      old_image_id = ann["image_id"]
      if old_image_id not in image_id_map:
        continue
      new_ann = dict(ann)
      new_ann["id"] = next_ann_id
      new_ann["image_id"] = image_id_map[old_image_id]
      combined["annotations"].append(new_ann)
      next_ann_id += 1
  return combined
def parse_args():
  """Parse command-line arguments.
  Returns:
    argparse.Namespace: Parsed arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input-jsons",
    nargs = "+",
    required = True
  )
  parser.add_argument("--output-json", required = True)
  parser.add_argument(
    "--collision-mode",
    choices = ["error", "skip", "prefix_source", "prefix_counter"],
    default = "error"
  )
  return parser.parse_args()
def main():
  """Combine COCO JSON files."""
  args = parse_args()
  input_jsons = [
    resolve_path(path, must_exist = True)
    for path in args.input_jsons
  ]
  output_json = resolve_path(args.output_json, must_exist = False)
  combined = combine_coco(
    files = input_jsons,
    collision_mode = args.collision_mode
  )
  save_json(combined, output_json)
  print_summary("combined", combined)
  print(f"\nwrote: {output_json}")
if __name__ == "__main__":
  main()
