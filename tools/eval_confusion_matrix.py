#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:25:21 2026

@author: eafpres
"""

import argparse
import json
import sys
import warnings
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
IOU_THRESHOLD = 0.50
LABEL_MODE = 'auto'
MAX_IMAGES = None
EXCLUDED_CLASSES = [
  'Damaged_mirror',
  'Damaged_glass',
  'Missing_part',
  'Severe_damage',
  'Rear_collision'
]
OPTIMIZE_THRESHOLDS = True
THRESHOLD_OPTIMIZATION_METRIC = 'f1'
DEFAULT_CONFIDENCE_THRESHOLD = 0.40
MIN_CONFIDENCE_THRESHOLD = 0.05
MAX_CONFIDENCE_THRESHOLD = 0.95
CONFIDENCE_THRESHOLD_STEP = 0.01
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))
from datasets import build_dataset
from models import build_model
import util.misc as utils
def parse_args():
  parser = argparse.ArgumentParser(
    description = (
      'Evaluate a trained DETA checkpoint on a COCO validation set, '
      'optimize per-class confidence thresholds, and write metrics.'
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
  parser.add_argument('--iou-threshold', default = IOU_THRESHOLD, type = float)
  parser.add_argument(
    '--label-mode',
    choices = ['auto', 'category_id', 'zero_based'],
    default = LABEL_MODE
  )
  parser.add_argument('--exclude-classes', nargs = '*', default = None)
  parser.add_argument('--max-images', default = MAX_IMAGES, type = int)
  parser.add_argument(
    '--optimize-thresholds',
    action = argparse.BooleanOptionalAction,
    default = OPTIMIZE_THRESHOLDS
  )
  parser.add_argument(
    '--threshold-optimization-metric',
    choices = ['accuracy', 'f1', 'precision', 'recall'],
    default = THRESHOLD_OPTIMIZATION_METRIC
  )
  parser.add_argument(
    '--default-confidence-threshold',
    default = DEFAULT_CONFIDENCE_THRESHOLD,
    type = float
  )
  parser.add_argument(
    '--min-confidence-threshold',
    default = MIN_CONFIDENCE_THRESHOLD,
    type = float
  )
  parser.add_argument(
    '--max-confidence-threshold',
    default = MAX_CONFIDENCE_THRESHOLD,
    type = float
  )
  parser.add_argument(
    '--confidence-threshold-step',
    default = CONFIDENCE_THRESHOLD_STEP,
    type = float
  )
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
    warnings.warn(
      'Unknown excluded classes or category IDs: ' +
      ', '.join(unknown_classes),
      RuntimeWarning
    )
  return excluded_category_ids
def safe_ratio(numerator, denominator):
  if denominator == 0:
    return 0.0
  return numerator / denominator
def calculate_metrics(true_positive, false_positive, false_negative):
  precision = safe_ratio(
    true_positive,
    true_positive + false_positive
  )
  recall = safe_ratio(
    true_positive,
    true_positive + false_negative
  )
  f1 = safe_ratio(
    2.0 * precision * recall,
    precision + recall
  )
  accuracy = safe_ratio(
    true_positive,
    true_positive + false_positive + false_negative
  )
  return {
    'true_positive': int(true_positive),
    'false_positive': int(false_positive),
    'false_negative': int(false_negative),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
  }
def create_threshold_grid(cli_args):
  if cli_args.confidence_threshold_step <= 0:
    raise ValueError('confidence threshold step must be greater than zero.')
  if cli_args.min_confidence_threshold < 0:
    raise ValueError('minimum confidence threshold cannot be negative.')
  if cli_args.max_confidence_threshold > 1:
    raise ValueError('maximum confidence threshold cannot exceed 1.')
  if (
    cli_args.min_confidence_threshold >
    cli_args.max_confidence_threshold
  ):
    raise ValueError(
      'minimum confidence threshold cannot exceed maximum threshold.'
    )
  thresholds = np.arange(
    cli_args.min_confidence_threshold,
    cli_args.max_confidence_threshold +
    cli_args.confidence_threshold_step / 2.0,
    cli_args.confidence_threshold_step
  )
  thresholds = np.append(
    thresholds,
    cli_args.default_confidence_threshold
  )
  return sorted({
    round(float(threshold), 10)
    for threshold in thresholds
    if 0 <= threshold <= 1
  })
def evaluate_class_threshold(
  image_records,
  category_id,
  confidence_threshold,
  iou_threshold
):
  true_positive = 0
  false_positive = 0
  false_negative = 0
  for image_record in image_records:
    gt_boxes = [
      gt_box
      for gt_box, gt_category_id in zip(
        image_record['gt_boxes'],
        image_record['gt_category_ids']
      )
      if gt_category_id == category_id
    ]
    pred_boxes = [
      pred_box
      for pred_box, pred_category_id, pred_score in zip(
        image_record['pred_boxes'],
        image_record['pred_category_ids'],
        image_record['pred_scores']
      )
      if (
        pred_category_id == category_id and
        pred_score >= confidence_threshold
      )
    ]
    matches, _, _ = greedy_matches(
      gt_boxes,
      pred_boxes,
      iou_threshold
    )
    true_positive += len(matches)
    false_positive += len(pred_boxes) - len(matches)
    false_negative += len(gt_boxes) - len(matches)
  return calculate_metrics(
    true_positive,
    false_positive,
    false_negative
  )
def get_threshold_rank(metrics, threshold, optimization_metric):
  if optimization_metric == 'precision':
    return (
      metrics['precision'],
      metrics['recall'],
      metrics['f1'],
      metrics['accuracy'],
      threshold
    )
  if optimization_metric == 'recall':
    return (
      metrics['recall'],
      metrics['precision'],
      metrics['f1'],
      metrics['accuracy'],
      -threshold
    )
  if optimization_metric == 'accuracy':
    return (
      metrics['accuracy'],
      metrics['f1'],
      metrics['precision'],
      metrics['recall'],
      threshold
    )
  return (
    metrics['f1'],
    metrics['accuracy'],
    metrics['precision'],
    metrics['recall'],
    threshold
  )
def optimize_class_thresholds(
  image_records,
  category_ids,
  category_name_by_id,
  cli_args
):
  threshold_grid = create_threshold_grid(cli_args)
  threshold_by_category_id = {}
  selected_rows = []
  search_rows = []
  for category_id in category_ids:
    class_name = category_name_by_id[category_id]
    best_rank = None
    best_row = None
    for threshold in threshold_grid:
      metrics = evaluate_class_threshold(
        image_records,
        category_id,
        threshold,
        cli_args.iou_threshold
      )
      row = {
        'category_id': category_id,
        'class': class_name,
        'threshold': threshold,
        'optimized_for': cli_args.threshold_optimization_metric,
        **metrics
      }
      search_rows.append(row)
      rank = get_threshold_rank(
        metrics,
        threshold,
        cli_args.threshold_optimization_metric
      )
      if best_rank is None or rank > best_rank:
        best_rank = rank
        best_row = row
    threshold_by_category_id[category_id] = best_row['threshold']
    selected_rows.append(best_row)
    print(
      f'optimized threshold: {class_name} = '
      f'{best_row["threshold"]:.2f} '
      f'({cli_args.threshold_optimization_metric} = '
      f'{best_row[cli_args.threshold_optimization_metric]:.4f})',
      flush = True
    )
  return (
    threshold_by_category_id,
    pd.DataFrame(selected_rows),
    pd.DataFrame(search_rows)
  )
def create_default_thresholds(
  image_records,
  category_ids,
  category_name_by_id,
  cli_args
):
  threshold_by_category_id = {
    category_id: cli_args.default_confidence_threshold
    for category_id in category_ids
  }
  rows = []
  for category_id in category_ids:
    metrics = evaluate_class_threshold(
      image_records,
      category_id,
      cli_args.default_confidence_threshold,
      cli_args.iou_threshold
    )
    rows.append({
      'category_id': category_id,
      'class': category_name_by_id[category_id],
      'threshold': cli_args.default_confidence_threshold,
      'optimized_for': 'disabled',
      **metrics
    })
  return (
    threshold_by_category_id,
    pd.DataFrame(rows),
    pd.DataFrame()
  )
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
  threshold,
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
    'threshold': threshold,
    'gt_bbox_xyxy': '' if gt_bbox is None else bbox_as_json(gt_bbox),
    'pred_bbox_xyxy': '' if pred_bbox is None else bbox_as_json(pred_bbox)
  })
def create_confusion_matrix(
  image_records,
  category_ids,
  category_name_by_id,
  threshold_by_category_id,
  iou_threshold
):
  class_names = [
    category_name_by_id[category_id]
    for category_id in category_ids
  ]
  background_name = 'background'
  matrix_names = class_names + [background_name]
  matrix = pd.DataFrame(
    0,
    index = matrix_names,
    columns = matrix_names,
    dtype = int
  )
  matrix.index.name = 'actual_class'
  rows = []
  for image_record in image_records:
    gt_boxes = image_record['gt_boxes']
    gt_category_ids = image_record['gt_category_ids']
    pred_boxes = []
    pred_category_ids = []
    pred_scores = []
    pred_thresholds = []
    for pred_box, pred_category_id, pred_score in zip(
      image_record['pred_boxes'],
      image_record['pred_category_ids'],
      image_record['pred_scores']
    ):
      threshold = threshold_by_category_id[pred_category_id]
      if pred_score < threshold:
        continue
      pred_boxes.append(pred_box)
      pred_category_ids.append(pred_category_id)
      pred_scores.append(pred_score)
      pred_thresholds.append(threshold)
    matches, matched_gt, matched_pred = greedy_matches(
      gt_boxes,
      pred_boxes,
      iou_threshold
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
        image_record['image_id'],
        image_record['file_name'],
        match_type,
        actual_name,
        predicted_name,
        iou,
        pred_scores[pred_idx],
        pred_thresholds[pred_idx],
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
        image_record['image_id'],
        image_record['file_name'],
        'false_negative',
        actual_name,
        background_name,
        np.nan,
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
        image_record['image_id'],
        image_record['file_name'],
        'false_positive',
        background_name,
        predicted_name,
        np.nan,
        pred_scores[pred_idx],
        pred_thresholds[pred_idx],
        None,
        pred_boxes[pred_idx]
      )
  return matrix, pd.DataFrame(rows)
def create_final_metrics(
  matrix,
  class_names,
  category_id_by_name,
  threshold_by_category_id
):
  rows = []
  for class_name in class_names:
    true_positive = int(matrix.loc[class_name, class_name])
    false_positive = int(matrix.loc[:, class_name].sum() - true_positive)
    false_negative = int(matrix.loc[class_name, :].sum() - true_positive)
    metrics = calculate_metrics(
      true_positive,
      false_positive,
      false_negative
    )
    rows.append({
      'category_id': category_id_by_name[class_name],
      'class': class_name,
      'threshold': threshold_by_category_id[
        category_id_by_name[class_name]
      ],
      'ground_truth': int(matrix.loc[class_name, :].sum()),
      'predicted': int(matrix.loc[:, class_name].sum()),
      **metrics
    })
  return pd.DataFrame(rows)
def plot_matrix(matrix_df, output_path, title, value_format):
  display_matrix_df = matrix_df.T
  x_labels = display_matrix_df.columns.tolist()
  y_labels = display_matrix_df.index.tolist()
  matrix = display_matrix_df.to_numpy()
  figure_size = max(8, 1.2 * len(x_labels))
  fig, ax = plt.subplots(figsize = (figure_size, figure_size))
  ax.set(
    xticks = np.arange(len(x_labels)),
    yticks = np.arange(len(y_labels)),
    xticklabels = x_labels,
    yticklabels = y_labels,
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
      ax.text(
        col_idx,
        row_idx,
        format(matrix[row_idx, col_idx], value_format),
        ha = 'center',
        va = 'center'
      )
  ax.set_xlim(-0.5, len(x_labels) - 0.5)
  ax.set_ylim(len(y_labels) - 0.5, -0.5)
  ax.set_xticks(
    np.arange(-0.5, len(x_labels), 1),
    minor = True
  )
  ax.set_yticks(
    np.arange(-0.5, len(y_labels), 1),
    minor = True
  )
  ax.grid(which = 'minor')
  ax.tick_params(which = 'minor', bottom = False, left = False)
  fig.tight_layout()
  fig.savefig(output_path, dpi = 160)
  plt.close(fig)
def collect_image_records(
  model,
  postprocessors,
  data_loader,
  coco,
  all_category_ids,
  excluded_category_ids,
  cli_args,
  device
):
  image_records = []
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
        annotation_ids = coco.getAnnIds(imgIds = [image_id])
        annotations = coco.loadAnns(annotation_ids)
        gt_boxes = []
        gt_category_ids = []
        for annotation in annotations:
          category_id = int(annotation['category_id'])
          if int(annotation.get('iscrowd', 0)) != 0:
            continue
          if category_id in excluded_category_ids:
            continue
          bbox = xywh_to_xyxy(annotation['bbox'])
          if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
          gt_boxes.append(bbox)
          gt_category_ids.append(category_id)
        pred_boxes = []
        pred_category_ids = []
        pred_scores = []
        for bbox, raw_label, score in zip(
          result['boxes'].detach().cpu().tolist(),
          result['labels'].detach().cpu().tolist(),
          result['scores'].detach().cpu().tolist()
        ):
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
        image_records.append({
          'image_id': image_id,
          'file_name': image_info.get('file_name', ''),
          'gt_boxes': gt_boxes,
          'gt_category_ids': gt_category_ids,
          'pred_boxes': pred_boxes,
          'pred_category_ids': pred_category_ids,
          'pred_scores': pred_scores
        })
        processed_images += 1
        if processed_images % 50 == 0:
          print(f'processed images: {processed_images}', flush = True)
      if (
        cli_args.max_images is not None and
        processed_images >= cli_args.max_images
      ):
        break
  return (
    image_records,
    processed_images,
    skipped_unknown_predictions,
    skipped_excluded_predictions
  )
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
  category_id_by_name = {
    category_name: category_id
    for category_id, category_name in category_name_by_id.items()
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
  data_loader = DataLoader(
    dataset_val,
    batch_size = args.batch_size,
    shuffle = False,
    collate_fn = utils.collate_fn,
    num_workers = args.num_workers,
    pin_memory = True
  )
  (
    image_records,
    processed_images,
    skipped_unknown_predictions,
    skipped_excluded_predictions
  ) = collect_image_records(
    model,
    postprocessors,
    data_loader,
    coco,
    all_category_ids,
    excluded_category_ids,
    cli_args,
    device
  )
  if cli_args.optimize_thresholds:
    (
      threshold_by_category_id,
      selected_thresholds_df,
      threshold_search_df
    ) = optimize_class_thresholds(
      image_records,
      category_ids,
      category_name_by_id,
      cli_args
    )
  else:
    (
      threshold_by_category_id,
      selected_thresholds_df,
      threshold_search_df
    ) = create_default_thresholds(
      image_records,
      category_ids,
      category_name_by_id,
      cli_args
    )
  matrix, matches_df = create_confusion_matrix(
    image_records,
    category_ids,
    category_name_by_id,
    threshold_by_category_id,
    cli_args.iou_threshold
  )
  metrics_df = create_final_metrics(
    matrix,
    class_names,
    category_id_by_name,
    threshold_by_category_id
  )
  normalized_matrix = matrix.div(
    matrix.sum(axis = 1).replace(0, np.nan),
    axis = 0
  ).fillna(0.0)
  normalized_matrix.index.name = 'actual_class'
  metrics_df.to_csv(output_dir / 'class_metrics.csv', index = False)
  selected_thresholds_df.to_csv(
    output_dir / 'class_thresholds.csv',
    index = False
  )
  if not threshold_search_df.empty:
    threshold_search_df.to_csv(
      output_dir / 'threshold_search_results.csv',
      index = False
    )
  output_matrix = matrix.T
  output_matrix.index.name = 'predicted_class'
  output_matrix.columns.name = 'actual_class'
  output_normalized_matrix = normalized_matrix.T
  output_normalized_matrix.index.name = 'predicted_class'
  output_normalized_matrix.columns.name = 'actual_class'
  output_matrix.to_csv(
    output_dir / 'confusion_matrix_counts.csv'
  )
  output_normalized_matrix.to_csv(
    output_dir / 'confusion_matrix_normalized.csv'
  )
  matches_df.to_csv(
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
  print('selected class thresholds:', flush = True)
  print(selected_thresholds_df.to_string(index = False), flush = True)
  print('', flush = True)
  print('final class metrics:', flush = True)
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
