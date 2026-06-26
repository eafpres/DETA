#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:50:42 2026

@author: eafpres
"""

import argparse
import json
from pathlib import Path
def build_category_maps(categories):
#
# build original category lookup maps.
#
  name_to_id = {
    category["name"]: category["id"]
    for category in categories
  }
  id_to_name = {
    category["id"]: category["name"]
    for category in categories
  }
  return name_to_id, id_to_name
def parse_reassignments(reassign_class_args):
#
# parse source target pairs from cli.
#
  reassignments = {}
  for source_name, target_name in reassign_class_args:
    reassignments[source_name] = target_name
  return reassignments
def validate_requested_classes(name_to_id, remove_class_names, reassignments):
#
# validate remove classes.
#
  missing_remove_names = sorted([
    class_name
    for class_name in remove_class_names
    if class_name not in name_to_id
  ])
  if missing_remove_names:
    raise ValueError(
      "These --remove-class values were not found in categories: "
      f"{missing_remove_names}"
    )
#
# validate reassignment sources and targets.
#
  missing_reassign_sources = sorted([
    source_name
    for source_name in reassignments
    if source_name not in name_to_id
  ])
  missing_reassign_targets = sorted([
    target_name
    for target_name in reassignments.values()
    if target_name not in name_to_id
  ])
  if missing_reassign_sources:
    raise ValueError(
      "These --reassign-class source values were not found in categories: "
      f"{missing_reassign_sources}"
    )
  if missing_reassign_targets:
    raise ValueError(
      "These --reassign-class target values were not found in categories: "
      f"{missing_reassign_targets}"
    )
def filter_coco_classes(
  input_json,
  output_json,
  remove_class_names,
  reassignments,
  remove_empty_images,
  drop_unused_categories
):
#
# load coco json.
#
  with open(input_json, "r", encoding = "utf-8") as f:
    data = json.load(f)
  remove_class_names = set(remove_class_names)
  name_to_id, id_to_name = build_category_maps(data["categories"])
  validate_requested_classes(name_to_id, remove_class_names, reassignments)
#
# build category reassignment map from original ids to original ids.
#
  reassign_id_map = {
    name_to_id[source_name]: name_to_id[target_name]
    for source_name, target_name in reassignments.items()
  }
#
# reassign annotation category ids before any class removal.
#
  reassigned_annotations = []
  reassigned_count = 0
  for annotation in data["annotations"]:
    old_category_id = annotation["category_id"]
    new_category_id = reassign_id_map.get(old_category_id, old_category_id)
    if new_category_id != old_category_id:
      annotation = annotation.copy()
      annotation["category_id"] = new_category_id
      reassigned_count += 1
    reassigned_annotations.append(annotation)
  data["annotations"] = reassigned_annotations
#
# remove requested classes after reassignment.
#
  remove_category_ids = {
    name_to_id[class_name]
    for class_name in remove_class_names
  }
  kept_annotations = [
    annotation
    for annotation in data["annotations"]
    if annotation["category_id"] not in remove_category_ids
  ]
#
# optionally drop categories that no longer have annotations.
#
  used_category_ids = {
    annotation["category_id"]
    for annotation in kept_annotations
  }
  kept_categories = []
  for category in data["categories"]:
    category_id = category["id"]
    if category_id in remove_category_ids:
      continue
    if drop_unused_categories and category_id not in used_category_ids:
      continue
    kept_categories.append(category.copy())
#
# optionally remove images with zero remaining annotations.
#
  original_image_count = len(data["images"])
  if remove_empty_images:
    used_image_ids = {
      annotation["image_id"]
      for annotation in kept_annotations
    }
    kept_images = [
      image
      for image in data["images"]
      if image["id"] in used_image_ids
    ]
  else:
    kept_images = data["images"]
#
# remap remaining category ids to contiguous zero-based ids.
#
  old_to_new_id = {
    category["id"]: new_id
    for new_id, category in enumerate(kept_categories)
  }
  final_categories = []
  for category in kept_categories:
    category = category.copy()
    category["id"] = old_to_new_id[category["id"]]
    final_categories.append(category)
  final_annotations = []
  dropped_orphan_annotation_count = 0
  for annotation in kept_annotations:
    old_category_id = annotation["category_id"]
    if old_category_id not in old_to_new_id:
      dropped_orphan_annotation_count += 1
      continue
    annotation = annotation.copy()
    annotation["category_id"] = old_to_new_id[old_category_id]
    final_annotations.append(annotation)
  data["categories"] = final_categories
  data["annotations"] = final_annotations
  data["images"] = kept_images
#
# save filtered json.
#
  with open(output_json, "w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)
#
# print summary.
#
  print(f"input json: {input_json}")
  print(f"output json: {output_json}")
  print(f"removed classes: {sorted(remove_class_names)}")
  print(f"removed original category ids: {sorted(remove_category_ids)}")
  print(f"reassignments: {reassignments}")
  print(f"reassigned annotations: {reassigned_count:,}")
  print(f"original images: {original_image_count:,}")
  print(f"final images: {len(data['images']):,}")
  print(f"final categories: {len(data['categories']):,}")
  print(f"final annotations: {len(data['annotations']):,}")
  print(f"dropped orphan annotations: {dropped_orphan_annotation_count:,}")
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input-json",
    required = True,
    help = "Input COCO JSON."
  )
  parser.add_argument(
    "--output-json",
    required = True,
    help = "Output COCO JSON."
  )
  parser.add_argument(
    "--remove-class",
    action = "append",
    default = [],
    help = "Class name to remove. Can be passed multiple times."
  )
  parser.add_argument(
    "--reassign-class",
    nargs = 2,
    action = "append",
    default = [],
    metavar = ("SOURCE_CLASS", "TARGET_CLASS"),
    help = "Reassign source class annotations to target class."
  )
  parser.add_argument(
    "--remove-empty-images",
    action = "store_true",
    help = "Remove images with zero remaining annotations."
  )
  parser.add_argument(
    "--keep-unused-categories",
    action = "store_true",
    help = "Keep categories with zero remaining annotations."
  )
  args = parser.parse_args()
  reassignments = parse_reassignments(args.reassign_class)
  filter_coco_classes(
    input_json = Path(args.input_json),
    output_json = Path(args.output_json),
    remove_class_names = args.remove_class,
    reassignments = reassignments,
    remove_empty_images = args.remove_empty_images,
    drop_unused_categories = not args.keep_unused_categories
  )
if __name__ == "__main__":
  main()
