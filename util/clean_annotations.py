#!/usr/bin/env python3
#
# libraries
#
import json
from pathlib import Path
from typing import Any, Dict
#
# config
#
coco_json_path = Path(
  "/mnt/e/glow-ai/aits/weld_quality/data/original_0204_val/augmented/annotations.json"
)
images_root = Path(
  "/mnt/e/glow-ai/aits/weld_quality/data/original_0204_val/augmented"
)
output_json_path = coco_json_path
#
# load coco
#
with coco_json_path.open("r", encoding = "utf-8") as f:
  coco: Dict[str, Any] = json.load(f)
#
# keep only images that still exist on disk
#
kept_images = []
kept_image_ids = set()
removed_images = []
for image_rec in coco.get("images", []):
  rel_path = Path(str(image_rec["file_name"]))
  abs_path = images_root / rel_path
  if abs_path.exists():
    kept_images.append(image_rec)
    kept_image_ids.add(int(image_rec["id"]))
  else:
    removed_images.append(str(image_rec["file_name"]))
#
# keep only annotations whose image still exists
#
kept_annotations = []
removed_annotation_count = 0
for ann in coco.get("annotations", []):
  if int(ann["image_id"]) in kept_image_ids:
    kept_annotations.append(ann)
  else:
    removed_annotation_count += 1
#
# update coco
#
coco["images"] = kept_images
coco["annotations"] = kept_annotations
#
# save
#
with output_json_path.open("w", encoding = "utf-8") as f:
  json.dump(coco, f, indent = 2)
#
# report
#
print(f"updated json: {output_json_path}")
print(f"removed images: {len(removed_images)}")
print(f"removed annotations: {removed_annotation_count}")
if removed_images:
  print("sample removed image paths:")
  for item in removed_images[:10]:
    print(f"  {item}")