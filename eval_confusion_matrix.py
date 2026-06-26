#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:25:21 2026

@author: eafpres
"""

import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
CHECKPOINT_PATH = (
  '~/DETA/DETA/exps/public/damages_deta_swin_ft_e1/checkpoint.pth'
)
OUTPUT_DIR = '~/DETA/DETA/exps/public/damages_deta_swin_ft_e1_eval'
COCO_PATH = '/mnt/c/damage_model/data'
COCO_VAL_IMAGES = '/mnt/x/common_images'
COCO_VAL_ANN = '/mnt/c/damage_model/data/val/segmentation.json'
DEVICE = 'cuda'
BATCH_SIZE = 1
NUM_WORKERS = 2
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.50
LABEL_MODE = 'auto'
MAX_IMAGES = None
EXCLUDED_CLASSES = ["Damaged_mirror",
                    "Damaged_glass",
                    "Missing_part",
                    "Severe_damage",
                    "Rear_collision"
                    ]
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))
from datasets import build_dataset
from models import build_model
import util.misc as utils
def parse_args():
  parser = argparse.ArgumentParser(
    description = (
      'Evaluate a trained DETA checkpoint on a COCO validation set and '
      'write class metrics and confusion matrices.'
    )
  )
  parser.add_argument('--checkpoint', default = CHECKPOINT_PATH)
  parser.add_argument('--output-dir', default = OUTPUT_DIR)
  parser.add_argument('--coco-path', default = COCO_PATH)
  parser.add_argument('--coco-val-images', default = COCO_VAL_IMAGES)
  parser.add_argument('--coco-val-ann', default = COCO_VAL_ANN)
  parser.add_argument('--device', default = DEVICE)
  parser.add_argument('--batch-size', default = BATCH_SIZE, type = int)
  parser.add_argument('--num-workers', default = NUM_WORKERS, type = int)
  parser.add_argument(
    '--confidence-threshold',
    default = CONFIDENCE_THRESHOLD,
    type = float
  )
  parser.add_argument(
    '--iou-threshold',
    default = IOU_THRESHOLD,
    type = float
  )
  parser.add_argument(
    '--label-mode',
    choices = ['auto', 'category_id', 'zero_based'],
    default = LABEL_MODE
  )
  parser.add_argument(
    '--exclude-classes',
    nargs = '*',
    default = None
  )
  parser.add_argument('--max-images', default = MAX_IMAGES, type = int)
  return parser.parse_args()
def load_checkpoint(checkpoint_path):
  return torch.load(
    checkpoint_path,
    map_location = 'cpu',
    weights_only = False
  )
def configure_args(checkpoint_args, cli_args):
  checkpoint_args.device = cli_args.device
  checkpoint_args.batch_size = cli_args.batch_size
  checkpoint_args.num_workers = cli_args.num_workers
  checkpoint_args.output_dir = cli_args.output_dir
  checkpoint_args.eval = True
  checkpoint_args.distributed = False
  checkpoint_args.coco_path = cli_args.coco_path
  checkpoint_args.coco_val_images = cli_args.coco_val_images
  checkpoint_args.coco_val_ann = cli_args.coco_val_ann
  return checkpoint_args
def get_coco_dataset(dataset):
  current_dataset = dataset
  for _ in range(10):
    if hasattr(current_dataset, 'coco'):
      return current_dataset
    if hasattr(current_dataset, 'dataset'):
      current_dataset = current_dataset.dataset
      continue
    break
  raise RuntimeError('Unable to locate the COCO dataset object.')
def xywh_to_xyxy(bbox):
  x, y, width, height = bbox
  return [
    float(x),
    float(y),
    float(x + width),
    float(y + height)
  ]
def box_iou(boxes_a, boxes_b):
  if len(boxes_a) == 0 or len(boxes_b) == 0:
    return np.zeros((len(boxes_a), len(boxes_b)), dtype = np.float64)
  boxes_a = np.asarray(boxes_a, dtype = np.float64)
  boxes_b = np.asarray(boxes_b, dtype = np.float64)
  top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
  bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
  widths_heights = np.clip(
    bottom_right - top_left,
    a_min = 0,
    a_max = None
  )
  intersection = widths_heights[:, :, 0] * widths_heights[:, :, 1]
  area_a = (
    (boxes_a[:, 2] - boxes_a[:, 0]) *
    (boxes_a[:, 3] - boxes_a[:, 1])
  )
  area_b = (
    (boxes_b[:, 2] - boxes_b[:, 0]) *
    (boxes_b[:, 3] - boxes_b[:, 1])
  )
  union = area_a[:, None] + area_b[None, :] - intersection
  return np.divide(
    intersection,
    union,
    out = np.zeros_like(intersection),
    where = union > 0
  )
def greedy_matches(gt_boxes, pred_boxes, iou_threshold):
  ious = box_iou(gt_boxes, pred_boxes)
  candidates = []
  for gt_idx in range(ious.shape[0]):
    for pred_idx in range(ious.shape[1]):
      iou = float(ious[gt_idx, pred_idx])
      if iou >= iou_threshold:
        candidates.append((iou, gt_idx, pred_idx))
  candidates.sort(reverse = True)
  matched_gt = set()
  matched_pred = set()
  matches = []
  for iou, gt_idx, pred_idx in candidates:
    if gt_idx in matched_gt or pred_idx in matched_pred:
      continue
    matched_gt.add(gt_idx)
    matched_pred.add(pred_idx)
    matches.append((gt_idx, pred_idx, iou))
  return matches, matched_gt, matched_pred
def resolve_label_mode(category_ids, num_logits, requested_mode):
  if requested_mode != 'auto':
    return requested_mode
  if num_logits == len(category_ids):
    return 'zero_based'
  if all(0 <= category_id < num_logits for category_id in category_ids):
    return 'category_id'
  raise RuntimeError(
    'Unable to infer label mapping. Set LABEL_MODE to category_id or '
    'zero_based.'
  )
def map_prediction_label(raw_label, category_ids, label_mode):
  if label_mode == 'zero_based':
    if 0 <= raw_label < len(category_ids):
      return category_ids[raw_label]
    return None
  if raw_label in category_ids:
    return raw_label
  return None
def resolve_excluded_category_ids(categories, excluded_classes):
  category_id_by_name = {
    str(category['name']).casefold(): int(category['id'])
    for category in categories
  }
  valid_category_ids = {
    int(category['id'])
    for category in categories
  }
  excluded_category_ids = set()
  unknown_classes = []
  for excluded_class in excluded_classes:
    excluded_class = str(excluded_class)
    class_key = excluded_class.casefold()
    if class_key in category_id_by_name:
      excluded_category_ids.add(category_id_by_name[class_key])
      continue
    try:
      category_id = int(excluded_class)
    except ValueError:
      unknown_classes.append(excluded_class)
      continue
    if category_id not in valid_category_ids:
      unknown_classes.append(excluded_class)
      continue
    excluded_category_ids.add(category_id)
  if unknown_classes:
    raise RuntimeError(
      'Unknown excluded classes or category IDs: ' +
      ', '.join(unknown_classes)
    )
  return excluded_category_ids
def safe_ratio(numerator, denominator):
  if denominator == 0:
    return 0.0
  return numerator / denominator
def bbox_as_json(bbox):
  return json.dumps([round(float(value), 4) for value in bbox])
def append_match_row(
  rows,
  image_id,
  file_name,
  match_type,
  actual_class,
  predicted_class,
  iou,
  score,
  gt_bbox,
  pred_bbox
):
  rows.append({
    'image_id': image_id,
    'file_name': file_name,
    'match_type': match_type,
    'actual_class': actual_class,
    'predicted_class': predicted_class,
    'iou': iou,
    'score': score,
    'gt_bbox_xyxy': '' if gt_bbox is None else bbox_as_json(gt_bbox),
    'pred_bbox_xyxy': '' if pred_bbox is None else bbox_as_json(pred_bbox)
  })
def plot_matrix(matrix_df, output_path, title, value_format):
  display_matrix_df = matrix_df.T
  labels = display_matrix_df.columns.tolist()
  matrix = display_matrix_df.to_numpy()
  figure_size = max(8, 1.2 * len(labels))
  fig, ax = plt.subplots(figsize = (figure_size, figure_size))
  ax.set(
    xticks = np.arange(len(labels)),
    yticks = np.arange(len(labels)),
    xticklabels = labels,
    yticklabels = display_matrix_df.index.tolist(),
    xlabel = 'actual class',
    ylabel = 'predicted class',
    title = title
  )
  plt.setp(
    ax.get_xticklabels(),
    rotation = 45,
    ha = 'right',
    rotation_mode = 'anchor'
  )
  for row_idx in range(matrix.shape[0]):
    for col_idx in range(matrix.shape[1]):
      value = matrix[row_idx, col_idx]
      ax.text(
        col_idx,
        row_idx,
        format(value, value_format),
        ha = 'center',
        va = 'center'
      )
  ax.set_xlim(-0.5, len(labels) - 0.5)
  ax.set_ylim(len(labels) - 0.5, -0.5)
  ax.set_xticks(
    np.arange(-0.5, len(labels), 1),
    minor = True
  )
  ax.set_yticks(
    np.arange(-0.5, len(labels), 1),
    minor = True
  )
  ax.grid(which = 'minor')
  ax.tick_params(which = 'minor', bottom = False, left = False)
  fig.tight_layout()
  fig.savefig(output_path, dpi = 160)
  plt.close(fig)
def main():
  cli_args = parse_args()
  excluded_classes = (
    EXCLUDED_CLASSES
    if cli_args.exclude_classes is None
    else cli_args.exclude_classes
  )
  output_dir = Path(cli_args.output_dir).expanduser()
  output_dir.mkdir(parents = True, exist_ok = True)
  checkpoint_path = Path(cli_args.checkpoint).expanduser()
  checkpoint = load_checkpoint(checkpoint_path)
  if 'args' not in checkpoint:
    raise RuntimeError(
      'Checkpoint does not contain training arguments under checkpoint["args"].'
    )
  args = configure_args(checkpoint['args'], cli_args)
  device = torch.device(args.device)
  model, _, postprocessors = build_model(args)
  model.load_state_dict(checkpoint['model'], strict = True)
  model.to(device)
  model.eval()
  dataset_val = build_dataset(image_set = 'val', args = args)
  coco_dataset = get_coco_dataset(dataset_val)
  coco = coco_dataset.coco
  all_category_ids = sorted(coco.getCatIds())
  categories = coco.loadCats(all_category_ids)
  category_name_by_id = {
    int(category['id']): category['name']
    for category in categories
  }
  excluded_category_ids = resolve_excluded_category_ids(
    categories,
    excluded_classes
  )
  category_ids = [
    category_id
    for category_id in all_category_ids
    if category_id not in excluded_category_ids
  ]
  if not category_ids:
    raise RuntimeError('All classes were excluded from evaluation.')
  class_names = [
    category_name_by_id[category_id]
    for category_id in category_ids
  ]
  excluded_class_names = [
    category_name_by_id[category_id]
    for category_id in sorted(excluded_category_ids)
  ]
  if excluded_class_names:
    print(
      'excluded classes: ' + ', '.join(excluded_class_names),
      flush = True
    )
  else:
    print('excluded classes: none', flush = True)
  background_name = 'background'
  matrix_names = class_names + [background_name]
  matrix = pd.DataFrame(
    0,
    index = matrix_names,
    columns = matrix_names,
    dtype = int
  )
  matrix.index.name = 'actual_class'
  data_loader = DataLoader(
    dataset_val,
    batch_size = args.batch_size,
    shuffle = False,
    collate_fn = utils.collate_fn,
    num_workers = args.num_workers,
    pin_memory = True
  )
  rows = []
  label_mode = None
  skipped_unknown_predictions = 0
  skipped_excluded_predictions = 0
  processed_images = 0
  with torch.inference_mode():
    for samples, targets in data_loader:
      samples = samples.to(device)
      outputs = model(samples)
      original_sizes = torch.stack([
        target['orig_size']
        for target in targets
      ]).to(device)
      results = postprocessors['bbox'](outputs, original_sizes)
      if label_mode is None:
        label_mode = resolve_label_mode(
          all_category_ids,
          int(outputs['pred_logits'].shape[-1]),
          cli_args.label_mode
        )
        print(f'label mode: {label_mode}', flush = True)
      for target, result in zip(targets, results):
        if (
          cli_args.max_images is not None and
          processed_images >= cli_args.max_images
        ):
          break
        image_id = int(target['image_id'].item())
        image_info = coco.loadImgs([image_id])[0]
        file_name = image_info.get('file_name', '')
        annotation_ids = coco.getAnnIds(imgIds = [image_id])
        annotations = coco.loadAnns(annotation_ids)
        annotations = [
          annotation
          for annotation in annotations
          if (
            int(annotation.get('iscrowd', 0)) == 0 and
            int(annotation['category_id']) not in excluded_category_ids
          )
        ]
        gt_boxes = []
        gt_category_ids = []
        for annotation in annotations:
          bbox = xywh_to_xyxy(annotation['bbox'])
          if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
          gt_boxes.append(bbox)
          gt_category_ids.append(int(annotation['category_id']))
        pred_boxes = []
        pred_category_ids = []
        pred_scores = []
        for bbox, raw_label, score in zip(
          result['boxes'].detach().cpu().tolist(),
          result['labels'].detach().cpu().tolist(),
          result['scores'].detach().cpu().tolist()
        ):
          if float(score) < cli_args.confidence_threshold:
            continue
          category_id = map_prediction_label(
            int(raw_label),
            all_category_ids,
            label_mode
          )
          if category_id is None:
            skipped_unknown_predictions += 1
            continue
          if category_id in excluded_category_ids:
            skipped_excluded_predictions += 1
            continue
          pred_boxes.append([float(value) for value in bbox])
          pred_category_ids.append(category_id)
          pred_scores.append(float(score))
        matches, matched_gt, matched_pred = greedy_matches(
          gt_boxes,
          pred_boxes,
          cli_args.iou_threshold
        )
        for gt_idx, pred_idx, iou in matches:
          actual_name = category_name_by_id[gt_category_ids[gt_idx]]
          predicted_name = category_name_by_id[pred_category_ids[pred_idx]]
          matrix.loc[actual_name, predicted_name] += 1
          match_type = (
            'true_positive'
            if actual_name == predicted_name
            else 'misclassified'
          )
          append_match_row(
            rows,
            image_id,
            file_name,
            match_type,
            actual_name,
            predicted_name,
            iou,
            pred_scores[pred_idx],
            gt_boxes[gt_idx],
            pred_boxes[pred_idx]
          )
        for gt_idx, gt_category_id in enumerate(gt_category_ids):
          if gt_idx in matched_gt:
            continue
          actual_name = category_name_by_id[gt_category_id]
          matrix.loc[actual_name, background_name] += 1
          append_match_row(
            rows,
            image_id,
            file_name,
            'false_negative',
            actual_name,
            background_name,
            np.nan,
            np.nan,
            gt_boxes[gt_idx],
            None
          )
        for pred_idx, pred_category_id in enumerate(pred_category_ids):
          if pred_idx in matched_pred:
            continue
          predicted_name = category_name_by_id[pred_category_id]
          matrix.loc[background_name, predicted_name] += 1
          append_match_row(
            rows,
            image_id,
            file_name,
            'false_positive',
            background_name,
            predicted_name,
            np.nan,
            pred_scores[pred_idx],
            None,
            pred_boxes[pred_idx]
          )
        processed_images += 1
        if processed_images % 50 == 0:
          print(f'processed images: {processed_images}', flush = True)
      if (
        cli_args.max_images is not None and
        processed_images >= cli_args.max_images
      ):
        break
  metric_rows = []
  for class_name in class_names:
    true_positive = int(matrix.loc[class_name, class_name])
    false_positive = int(matrix.loc[:, class_name].sum() - true_positive)
    false_negative = int(matrix.loc[class_name, :].sum() - true_positive)
    ground_truth = int(matrix.loc[class_name, :].sum())
    predicted = int(matrix.loc[:, class_name].sum())
    precision = safe_ratio(
      true_positive,
      true_positive + false_positive
    )
    recall = safe_ratio(
      true_positive,
      true_positive + false_negative
    )
    f1 = safe_ratio(2.0 * precision * recall, precision + recall)
    metric_rows.append({
      'class': class_name,
      'ground_truth': ground_truth,
      'predicted': predicted,
      'true_positive': true_positive,
      'false_positive': false_positive,
      'false_negative': false_negative,
      'precision': precision,
      'recall': recall,
      'f1': f1
    })
  metrics_df = pd.DataFrame(metric_rows)
  normalized_matrix = matrix.div(
    matrix.sum(axis = 1).replace(0, np.nan),
    axis = 0
  ).fillna(0.0)
  normalized_matrix.index.name = 'actual_class'
  metrics_df.to_csv(output_dir / 'class_metrics.csv', index = False)
  matrix.to_csv(output_dir / 'confusion_matrix_counts.csv')
  normalized_matrix.to_csv(
    output_dir / 'confusion_matrix_normalized.csv'
  )
  pd.DataFrame(rows).to_csv(
    output_dir / 'prediction_matches.csv',
    index = False
  )
  plot_matrix(
    matrix,
    output_dir / 'confusion_matrix_counts.png',
    'Detection confusion matrix: counts',
    'd'
  )
  plot_matrix(
    normalized_matrix,
    output_dir / 'confusion_matrix_normalized.png',
    'Detection confusion matrix: row-normalized',
    '.2f'
  )
  print('', flush = True)
  print('class metrics:', flush = True)
  print(metrics_df.to_string(index = False), flush = True)
  print('', flush = True)
  print(f'processed images: {processed_images}', flush = True)
  print(
    f'skipped predictions with unknown category mapping: '
    f'{skipped_unknown_predictions}',
    flush = True
  )
  print(
    f'skipped predictions for excluded classes: '
    f'{skipped_excluded_predictions}',
    flush = True
  )
  print(f'outputs written to: {output_dir}', flush = True)
if __name__ == '__main__':
  main()
