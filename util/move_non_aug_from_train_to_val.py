#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:26:33 2026

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
    path: Input path.
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
    path: JSON path.
  Returns:
    dict: Parsed JSON.
  """
  with Path(path).open("r", encoding = "utf-8") as f:
    return json.load(f)
def save_json(data, path):
  """Save a JSON file.
  Args:
    data: JSON-compatible object.
    path: Output path.
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
    list[tuple[int, str]]: Sorted category id/name pairs.
  """
  return sorted(
    (cat["id"], cat["name"])
    for cat in data.get("categories", [])
  )
def validate_categories(train, val):
  """Validate that train and val categories match.
  Args:
    train: Train COCO dictionary.
    val: Validation COCO dictionary.
  """
  train_sig = category_signature(train)
  val_sig = category_signature(val)
  if train_sig != val_sig:
    raise ValueError(
      f"category mismatch\ntrain={train_sig}\nval={val_sig}"
    )
def get_annotations_by_image_id(data):
  """Group annotations by image id.
  Args:
    data: COCO dictionary.
  Returns:
    dict[int, list[dict]]: Annotations by image id.
  """
  out = {}
  for ann in data.get("annotations", []):
    out.setdefault(ann["image_id"], []).append(ann)
  return out
def make_subset(data, keep_image_ids):
  """Create a COCO subset from selected image ids.
  Args:
    data: Input COCO dictionary.
    keep_image_ids: Image ids to keep.
  Returns:
    dict: COCO subset with remapped ids.
  """
  keep_image_ids = set(keep_image_ids)
  ann_by_image_id = get_annotations_by_image_id(data)
  images_out = []
  annotations_out = []
  old_to_new_image_id = {}
  next_image_id = 0
  next_ann_id = 0
  for image in data.get("images", []):
    if image["id"] not in keep_image_ids:
      continue
    old_image_id = image["id"]
    new_image = dict(image)
    new_image["id"] = next_image_id
    old_to_new_image_id[old_image_id] = next_image_id
    images_out.append(new_image)
    next_image_id += 1
  for image in data.get("images", []):
    old_image_id = image["id"]
    if old_image_id not in old_to_new_image_id:
      continue
    for ann in ann_by_image_id.get(old_image_id, []):
      new_ann = dict(ann)
      new_ann["id"] = next_ann_id
      new_ann["image_id"] = old_to_new_image_id[old_image_id]
      annotations_out.append(new_ann)
      next_ann_id += 1
  return {
    "info": dict(data.get("info", {})),
    "licenses": data.get("licenses", []),
    "categories": data.get("categories", []),
    "images": images_out,
    "annotations": annotations_out
  }
def append_to_val(val, moved):
  """Append moved images/annotations to validation COCO.
  Args:
    val: Existing validation COCO dictionary.
    moved: Moved COCO subset.
  Returns:
    dict: Combined validation COCO dictionary.
  """
  out = {
    "info": dict(val.get("info", {})),
    "licenses": val.get("licenses", []),
    "categories": val.get("categories", []),
    "images": [],
    "annotations": []
  }
  out["info"]["description"] = (
    str(out["info"].get("description", "COCO dataset"))
    + " | plus non-aug train images"
  )
  next_image_id = 0
  next_ann_id = 0
  seen_names = set()
  for source in [val, moved]:
    image_id_map = {}
    for image in source.get("images", []):
      file_name = image["file_name"]
      if file_name in seen_names:
        raise ValueError(f"duplicate val file_name: {file_name}")
      seen_names.add(file_name)
      new_image = dict(image)
      old_image_id = image["id"]
      new_image["id"] = next_image_id
      image_id_map[old_image_id] = next_image_id
      out["images"].append(new_image)
      next_image_id += 1
    for ann in source.get("annotations", []):
      if ann["image_id"] not in image_id_map:
        continue
      new_ann = dict(ann)
      new_ann["id"] = next_ann_id
      new_ann["image_id"] = image_id_map[ann["image_id"]]
      out["annotations"].append(new_ann)
      next_ann_id += 1
  return out
def print_summary(label, data):
  """Print a COCO summary.
  Args:
    label: Summary label.
    data: COCO dictionary.
  """
  cat_id_to_name = {
    cat["id"]: cat["name"]
    for cat in data.get("categories", [])
  }
  counts = Counter(
    ann["category_id"]
    for ann in data.get("annotations", [])
  )
  print(f"\n{label}")
  print(f"  images: {len(data.get('images', []))}")
  print(f"  annotations: {len(data.get('annotations', []))}")
  for cat_id, name in sorted(cat_id_to_name.items()):
    print(f"  {cat_id}: {name}: {counts.get(cat_id, 0)}")
def parse_args():
  """Parse command-line arguments.
  Returns:
    argparse.Namespace: Parsed command-line arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--train-combined-json", required = True)
  parser.add_argument("--val-json", required = True)
  parser.add_argument("--out-train-json", required = True)
  parser.add_argument("--out-val-json", required = True)
  parser.add_argument("--aug-prefix", default = "aug_")
  return parser.parse_args()
def main():
  """Move non-aug train images into validation JSON."""
  args = parse_args()
  train_path = resolve_path(args.train_combined_json, must_exist = True)
  val_path = resolve_path(args.val_json, must_exist = True)
  out_train_path = resolve_path(args.out_train_json, must_exist = False)
  out_val_path = resolve_path(args.out_val_json, must_exist = False)
  train = load_json(train_path)
  val = load_json(val_path)
  validate_categories(train, val)
  aug_image_ids = []
  non_aug_image_ids = []
  for image in train.get("images", []):
    if image["file_name"].startswith(args.aug_prefix):
      aug_image_ids.append(image["id"])
    else:
      non_aug_image_ids.append(image["id"])
  train_aug_only = make_subset(train, aug_image_ids)
  moved_non_aug = make_subset(train, non_aug_image_ids)
  val_plus_non_aug = append_to_val(val, moved_non_aug)
  save_json(train_aug_only, out_train_path)
  save_json(val_plus_non_aug, out_val_path)
  print_summary("input train_combined", train)
  print_summary("input val", val)
  print_summary("output train aug only", train_aug_only)
  print_summary("moved non-aug train images", moved_non_aug)
  print_summary("output val plus non-aug", val_plus_non_aug)
  print(f"\nwrote: {out_train_path}")
  print(f"wrote: {out_val_path}")
if __name__ == "__main__":
  main()
