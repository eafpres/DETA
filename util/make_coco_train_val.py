#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:55:11 2026

@author: eafpres
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def resolve_path(path, must_exist = False):
  """Resolve a path relative to the DETA repo root.
  Args:
    path: Input path string or None.
    must_exist: Whether the resolved path must exist.
  Returns:
    pathlib.Path | None: Resolved path or None.
  """
  if path is None:
    return None
  path = Path(path).expanduser()
  if not path.is_absolute():
    path = ROOT / path
  path = path.resolve()
  if must_exist and not path.exists():
    raise FileNotFoundError(f"path does not exist: {path}")
  return path
def parse_class_list(value):
  """Parse a comma-separated class list.
  Args:
    value: Comma-separated class names.
  Returns:
    list[str]: Parsed class names.
  """
  return [item.strip() for item in value.split(",") if item.strip()]
def load_json(path):
  """Load a JSON file.
  Args:
    path: Path to a JSON file.
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
def load_classes(args):
  """Load target class names from args.
  Args:
    args: Parsed command-line arguments.
  Returns:
    list[str]: Requested class names.
  """
  classes = []
  if args.classes:
    classes.extend(parse_class_list(args.classes))
  if args.classes_file:
    with Path(args.classes_file).open("r", encoding = "utf-8") as f:
      classes.extend([line.strip() for line in f if line.strip()])
  classes = list(dict.fromkeys(classes))
  if not classes:
    raise ValueError("provide --classes or --classes-file")
  return classes
def make_category_mapping(data, keep_names, start_category_id):
  """Create filtered categories and id mapping.
  Args:
    data: Input COCO dictionary.
    keep_names: Class names to retain.
    start_category_id: First output category id.
  Returns:
    tuple[list[dict], dict, dict]: Categories and id maps.
  """
  old_id_to_cat = {
    cat["id"]: cat
    for cat in data.get("categories", [])
  }
  old_name_to_cat = {
    cat["name"]: cat
    for cat in data.get("categories", [])
  }
  missing = [name for name in keep_names if name not in old_name_to_cat]
  if missing:
    raise ValueError(f"classes not found in input categories: {missing}")
  categories = []
  old_id_to_new_id = {}
  new_id_to_name = {}
  for new_idx, name in enumerate(keep_names, start = start_category_id):
    old_cat = old_name_to_cat[name]
    old_id_to_new_id[old_cat["id"]] = new_idx
    new_id_to_name[new_idx] = name
    categories.append({
      "id": new_idx,
      "name": name,
      "category_id": old_cat.get("category_id"),
      "supercategory": old_cat.get("supercategory")
    })
  return categories, old_id_to_new_id, new_id_to_name
def flatten_name(image, seen, collision_mode):
  """Flatten an image path to a basename.
  Args:
    image: COCO image dictionary.
    seen: Set of already used output file names.
    collision_mode: Collision handling mode.
  Returns:
    str: Flattened file name.
  """
  old_file_name = image["file_name"]
  name = Path(old_file_name).name
  if name not in seen:
    seen.add(name)
    return name
  dataset = str(image.get("dataset", "dataset"))
  safe_dataset = dataset.replace("/", "_").replace("\\", "_")
  if collision_mode == "error":
    raise ValueError(f"duplicate flattened file_name: {name}")
  if collision_mode == "prefix_dataset":
    name = f"{safe_dataset}__{name}"
  elif collision_mode == "prefix_id":
    name = f"{image['id']}__{name}"
  else:
    raise ValueError(f"unknown collision mode: {collision_mode}")
  if name in seen:
    raise ValueError(f"duplicate after collision handling: {name}")
  seen.add(name)
  return name
def build_filtered_records(data, old_id_to_new_id, keep_names, args):
  """Filter annotations and image records.
  Args:
    data: Input COCO dictionary.
    old_id_to_new_id: Mapping from old category id to new category id.
    keep_names: Class names to retain.
    args: Parsed command-line arguments.
  Returns:
    tuple[list[dict], dict, dict, Counter]: Images, anns, paths, counts.
  """
  old_image_by_id = {
    image["id"]: image
    for image in data.get("images", [])
  }
  kept_anns_by_old_image_id = defaultdict(list)
  dropped_counts = Counter()
  kept_counts = Counter()
  ann_id = 0
  for ann in data.get("annotations", []):
    old_cat_id = ann["category_id"]
    if old_cat_id not in old_id_to_new_id:
      dropped_counts[old_cat_id] += 1
      continue
    new_ann = dict(ann)
    new_ann["id"] = ann_id
    new_ann["category_id"] = old_id_to_new_id[old_cat_id]
    kept_anns_by_old_image_id[ann["image_id"]].append(new_ann)
    kept_counts[old_id_to_new_id[old_cat_id]] += 1
    ann_id += 1
  kept_images = []
  old_to_new_image_id = {}
  old_to_flat_name = {}
  seen_names = set()
  new_image_id = 0
  keep_name_set = set(keep_names)
  for old_image_id, image in sorted(old_image_by_id.items()):
    image_has_kept_ann = old_image_id in kept_anns_by_old_image_id
    dataset_name = image.get("dataset")
    keep_empty = False
    if args.keep_empty_images == "all":
      keep_empty = True
    elif args.keep_empty_images == "listed_dataset":
      keep_empty = dataset_name in keep_name_set
    if not image_has_kept_ann and not keep_empty:
      continue
    new_image = dict(image)
    new_image["id"] = new_image_id
    new_image["file_name"] = flatten_name(
      image = image,
      seen = seen_names,
      collision_mode = args.collision_mode
    )
    old_to_new_image_id[old_image_id] = new_image_id
    old_to_flat_name[old_image_id] = new_image["file_name"]
    kept_images.append(new_image)
    new_image_id += 1
  kept_annotations = []
  ann_id = 0
  for old_image_id in old_to_new_image_id:
    for ann in kept_anns_by_old_image_id.get(old_image_id, []):
      new_ann = dict(ann)
      new_ann["id"] = ann_id
      new_ann["image_id"] = old_to_new_image_id[old_image_id]
      kept_annotations.append(new_ann)
      ann_id += 1
  return kept_images, kept_annotations, old_to_flat_name, kept_counts
def primary_split_key(image, anns_by_image_id, new_id_to_name, mode):
  """Get a split key for one image.
  Args:
    image: COCO image dictionary.
    anns_by_image_id: Mapping from image id to annotations.
    new_id_to_name: Mapping from category id to category name.
    mode: Stratification mode.
  Returns:
    str: Split key.
  """
  if mode == "none":
    return "__all__"
  if mode == "dataset":
    return str(image.get("dataset", "__missing_dataset__"))
  anns = anns_by_image_id.get(image["id"], [])
  if not anns:
    return "__empty__"
  cat_ids = sorted({ann["category_id"] for ann in anns})
  return new_id_to_name.get(cat_ids[0], str(cat_ids[0]))
def split_images(images, annotations, new_id_to_name, val_frac, seed, mode):
  """Split images into train and validation groups.
  Args:
    images: Filtered COCO image records.
    annotations: Filtered COCO annotation records.
    new_id_to_name: Mapping from category id to name.
    val_frac: Validation fraction.
    seed: Random seed.
    mode: Stratification mode.
  Returns:
    tuple[set, set]: Train and validation image ids.
  """
  if not 0.0 < val_frac < 1.0:
    raise ValueError("--val-frac must be between 0 and 1")
  rng = random.Random(seed)
  anns_by_image_id = defaultdict(list)
  for ann in annotations:
    anns_by_image_id[ann["image_id"]].append(ann)
  groups = defaultdict(list)
  for image in images:
    key = primary_split_key(
      image = image,
      anns_by_image_id = anns_by_image_id,
      new_id_to_name = new_id_to_name,
      mode = mode
    )
    groups[key].append(image["id"])
  train_ids = set()
  val_ids = set()
  for _, image_ids in sorted(groups.items()):
    rng.shuffle(image_ids)
    if len(image_ids) == 1:
      train_ids.update(image_ids)
      continue
    n_val = int(round(len(image_ids) * val_frac))
    n_val = max(1, min(n_val, len(image_ids) - 1))
    val_ids.update(image_ids[:n_val])
    train_ids.update(image_ids[n_val:])
  return train_ids, val_ids
def subset_coco(data, image_ids, categories):
  """Create a COCO subset for selected image ids.
  Args:
    data: Filtered COCO dictionary.
    image_ids: Selected image ids.
    categories: Output categories.
  Returns:
    dict: COCO subset dictionary.
  """
  image_ids = set(image_ids)
  old_to_new_image_id = {}
  images_out = []
  for new_id, image in enumerate(
    [img for img in data["images"] if img["id"] in image_ids]
  ):
    old_to_new_image_id[image["id"]] = new_id
    new_image = dict(image)
    new_image["id"] = new_id
    images_out.append(new_image)
  annotations_out = []
  ann_id = 0
  for ann in data["annotations"]:
    if ann["image_id"] not in old_to_new_image_id:
      continue
    new_ann = dict(ann)
    new_ann["id"] = ann_id
    new_ann["image_id"] = old_to_new_image_id[ann["image_id"]]
    annotations_out.append(new_ann)
    ann_id += 1
  return {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "categories": categories,
    "images": images_out,
    "annotations": annotations_out
  }
def copy_or_link_images(source_data, subset, images_root, out_dir, mode):
  """Copy or symlink images into a flat output folder.
  Args:
    source_data: Original COCO dictionary.
    subset: Output COCO subset.
    images_root: Root folder containing original images.
    out_dir: Output image folder.
    mode: copy, symlink, or none.
  """
  if mode == "none":
    return
  if images_root is None or out_dir is None:
    raise ValueError("--images-root and image output dirs are required")
  images_root = Path(images_root)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents = True, exist_ok = True)
  original_by_flat = {}
  for image in source_data.get("images", []):
    original_by_flat[Path(image["file_name"]).name] = image["file_name"]
  for image in subset["images"]:
    flat_name = image["file_name"]
    original_rel = original_by_flat.get(flat_name, flat_name)
    src = images_root / original_rel
    dst = out_dir / flat_name
    if not src.exists():
      raise FileNotFoundError(f"missing source image: {src}")
    if dst.exists():
      continue
    if mode == "copy":
      shutil.copy2(src, dst)
    elif mode == "symlink":
      dst.symlink_to(src.resolve())
    else:
      raise ValueError(f"unknown copy mode: {mode}")
def print_summary(label, subset):
  """Print a COCO subset summary.
  Args:
    label: Subset label.
    subset: COCO subset dictionary.
  """
  cat_id_to_name = {
    cat["id"]: cat["name"]
    for cat in subset["categories"]
  }
  counts = Counter(ann["category_id"] for ann in subset["annotations"])
  print(f"\n{label}")
  print(f"  images: {len(subset['images'])}")
  print(f"  annotations: {len(subset['annotations'])}")
  for cat_id, name in sorted(cat_id_to_name.items()):
    print(f"  {cat_id}: {name}: {counts.get(cat_id, 0)}")
def parse_args():
  """Parse command-line arguments.
  Returns:
    argparse.Namespace: Parsed arguments.
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-json", required = True)
  parser.add_argument("--out-dir", required = True)
  parser.add_argument("--classes", default = None)
  parser.add_argument("--classes-file", default = None)
  parser.add_argument("--val-frac", type = float, default = 0.2)
  parser.add_argument("--seed", type = int, default = 123)
  parser.add_argument("--start-category-id", type = int, default = 0)
  parser.add_argument(
    "--keep-empty-images",
    choices = ["never", "listed_dataset", "all"],
    default = "listed_dataset"
  )
  parser.add_argument(
    "--stratify-by",
    choices = ["dataset", "category", "none"],
    default = "dataset"
  )
  parser.add_argument(
    "--collision-mode",
    choices = ["error", "prefix_dataset", "prefix_id"],
    default = "prefix_dataset"
  )
  parser.add_argument(
    "--copy-mode",
    choices = ["none", "copy", "symlink"],
    default = "none"
  )
  parser.add_argument("--images-root", default = None)
  parser.add_argument("--train-images-dir", default = None)
  parser.add_argument("--val-images-dir", default = None)
  return parser.parse_args()
def main():
  """Create filtered flat COCO train and validation files."""
  args = parse_args()
  input_json = resolve_path(args.input_json, must_exist = True)
  out_dir = resolve_path(args.out_dir, must_exist = False)
  images_root = resolve_path(args.images_root, must_exist = args.copy_mode != "none")
  train_images_dir = resolve_path(args.train_images_dir, must_exist = False)
  val_images_dir = resolve_path(args.val_images_dir, must_exist = False)
  train_json = out_dir / "train.json"
  val_json = out_dir / "val.json"
  data = load_json(input_json)
  keep_names = load_classes(args)
  categories, old_id_to_new_id, new_id_to_name = make_category_mapping(
    data = data,
    keep_names = keep_names,
    start_category_id = args.start_category_id
  )
  images, annotations, _, _ = build_filtered_records(
    data = data,
    old_id_to_new_id = old_id_to_new_id,
    keep_names = keep_names,
    args = args
  )
  filtered = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "categories": categories,
    "images": images,
    "annotations": annotations
  }
  train_ids, val_ids = split_images(
    images = images,
    annotations = annotations,
    new_id_to_name = new_id_to_name,
    val_frac = args.val_frac,
    seed = args.seed,
    mode = args.stratify_by
  )
  train = subset_coco(
    data = filtered,
    image_ids = train_ids,
    categories = categories
  )
  val = subset_coco(
    data = filtered,
    image_ids = val_ids,
    categories = categories
  )
  save_json(train, train_json)
  save_json(val, val_json)
  copy_or_link_images(
    source_data = data,
    subset = train,
    images_root = images_root,
    out_dir = train_images_dir,
    mode = args.copy_mode
  )
  copy_or_link_images(
    source_data = data,
    subset = val,
    images_root = images_root,
    out_dir = val_images_dir,
    mode = args.copy_mode
  )
  print(f"\nwrote: {train_json}")
  print(f"wrote: {val_json}")
  print_summary("train", train)
  print_summary("val", val)
if __name__ == "__main__":
  main()
