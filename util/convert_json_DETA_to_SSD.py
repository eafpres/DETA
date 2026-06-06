#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 13:35:34 2026

@author: eafpres
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Set
#
# constants
#
WELD_CATEGORY = {
  "id": 1,
  "name": "weld_region",
  "supercategory": None,
}
#
# helpers
#
def parse_args() -> argparse.Namespace:
  """Parse command-line arguments.

  Returns:
    argparse.Namespace: Parsed args.
  """
  parser = argparse.ArgumentParser(
    description = "Convert multi-class COCO JSON to one-class weld-region COCO"
  )
  parser.add_argument(
    "--input-json",
    required = True,
    type = str,
    help = "Path to source COCO JSON"
  )
  parser.add_argument(
    "--output-json",
    required = True,
    type = str,
    help = "Path to output COCO JSON"
  )
  parser.add_argument(
    "--drop-images-without-annotations",
    action = "store_true",
    help = (
      "Drop images that have no annotations after conversion. "
      "By default they are kept as negatives."
    )
  )
  return parser.parse_args()

def load_json(path: Path) -> Dict[str, Any]:
  """Load JSON from disk.

  Args:
    path: JSON path.

  Returns:
    Dict[str, Any]: Parsed JSON.
  """
  with path.open("r", encoding = "utf-8") as f:
    return json.load(f)

def save_json(data: Dict[str, Any], path: Path) -> None:
  """Write JSON to disk.

  Args:
    data: Output payload.
    path: Output path.
  """
  path.parent.mkdir(parents = True, exist_ok = True)
  with path.open("w", encoding = "utf-8") as f:
    json.dump(data, f, indent = 2)

def convert_to_weld_region(
  coco: Dict[str, Any],
  drop_images_without_annotations: bool,
) -> Dict[str, Any]:
  """Convert a multi-class COCO file to one-class weld-region COCO.

  Args:
    coco: Source COCO dict.
    drop_images_without_annotations: Whether to remove images with no anns.

  Returns:
    Dict[str, Any]: Converted COCO dict.
  """
  out: Dict[str, Any] = {
    "info": copy.deepcopy(coco.get("info", {})),
    "licenses": copy.deepcopy(coco.get("licenses", [])),
    "categories": [copy.deepcopy(WELD_CATEGORY)],
    "images": [],
    "annotations": [],
  }
#
# convert annotations to one class
#
  kept_image_ids: Set[int] = set()
  next_ann_id = 1
  for ann in coco.get("annotations", []):
    image_id = int(ann["image_id"])
    bbox = [float(v) for v in ann["bbox"]]
    area = float(ann.get("area", bbox[2] * bbox[3]))
    out_ann = {
      "id": next_ann_id,
      "image_id": image_id,
      "category_id": 1,
      "bbox": bbox,
      "area": area,
      "iscrowd": int(ann.get("iscrowd", 0)),
    }
    if "segmentation" in ann:
      out_ann["segmentation"] = copy.deepcopy(ann["segmentation"])
    out["annotations"].append(out_ann)
    kept_image_ids.add(image_id)
    next_ann_id += 1
#
# keep or filter images
#
  for img in coco.get("images", []):
    image_id = int(img["id"])
    if drop_images_without_annotations and image_id not in kept_image_ids:
      continue
    out["images"].append(copy.deepcopy(img))
  return out

def main() -> None:
  """Run conversion."""
  args = parse_args()
  input_json = Path(args.input_json).resolve()
  output_json = Path(args.output_json).resolve()
  coco = load_json(input_json)
  out = convert_to_weld_region(
    coco = coco,
    drop_images_without_annotations = args.drop_images_without_annotations,
  )
  save_json(out, output_json)
#
# report
#
  print(f"input json: {input_json}")
  print(f"output json: {output_json}")
  print(f"images: {len(out['images'])}")
  print(f"annotations: {len(out['annotations'])}")
  print(f"categories: {out['categories']}")

if __name__ == "__main__":
  main()
