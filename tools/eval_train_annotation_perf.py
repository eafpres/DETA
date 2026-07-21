#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate DETA train-set image-level annotation performance.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
CHECKPOINT_PATH = (
  '~/DETA/DETA/exps/public/damages_deta_swin_ft_e1/checkpoint.pth'
)
OUTPUT_DIR = '~/DETA/DETA/exps/public/damages_deta_swin_ft_e1_train_perf'
COCO_PATH = '/mnt/c/damage_model/data'
COCO_TRAIN_IMAGES = '/mnt/x/common_images'
COCO_TRAIN_ANN = '/mnt/c/damage_model/data/hard_negatives/train_w_hard_negatives.json'
DEVICE = 'cuda'
BATCH_SIZE = 1
NUM_WORKERS = 2
IOU_THRESHOLD = 0.50
LABEL_MODE = 'auto'
MAX_IMAGES = None
TOP_K_IMAGES = 250
EXPORT_IMAGES = True
EXCLUDED_CLASSES = [
  'Damaged_mirror',
  'Damaged_glass',
  'Missing_part',
  'Severe_damage',
  'Rear_collision'
]
OPTIMIZE_THRESHOLDS = False
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
      'Evaluate a trained DETA checkpoint against train COCO annotations, '
      'write per-image detection metrics, and export the poorest images.'
    )
  )
  parser.add_argument('--checkpoint', default = CHECKPOINT_PATH)
  parser.add_argument('--output-dir', default = OUTPUT_DIR)
  parser.add_argument('--coco-path', default = COCO_PATH)
  parser.add_argument('--coco-train-images', default = COCO_TRAIN_IMAGES)
  parser.add_argument('--coco-train-ann', default = COCO_TRAIN_ANN)
  parser.add_argument('--device', default = DEVICE)
  parser.add_argument('--batch-size', default = BATCH_SIZE, type = int)
  parser.add_argument('--num-workers', default = NUM_WORKERS, type = int)
  parser.add_argument('--iou-threshold', default = IOU_THRESHOLD, type = float)
  parser.add_argument('--label-mode', choices = ['auto', 'category_id', 'zero_based'], default = LABEL_MODE)
  parser.add_argument('--exclude-classes', nargs = '*', default = None)
  parser.add_argument('--max-images', default = MAX_IMAGES, type = int)
  parser.add_argument('--top-k-images', default = TOP_K_IMAGES, type = int)
  parser.add_argument('--export-images', action = argparse.BooleanOptionalAction, default = EXPORT_IMAGES)
  parser.add_argument('--thresholds-csv', default = None)
  parser.add_argument('--optimize-thresholds', action = argparse.BooleanOptionalAction, default = OPTIMIZE_THRESHOLDS)
  parser.add_argument('--threshold-optimization-metric', choices = ['accuracy', 'f1', 'precision', 'recall'], default = THRESHOLD_OPTIMIZATION_METRIC)
  parser.add_argument('--default-confidence-threshold', default = DEFAULT_CONFIDENCE_THRESHOLD, type = float)
  parser.add_argument('--min-confidence-threshold', default = MIN_CONFIDENCE_THRESHOLD, type = float)
  parser.add_argument('--max-confidence-threshold', default = MAX_CONFIDENCE_THRESHOLD, type = float)
  parser.add_argument('--confidence-threshold-step', default = CONFIDENCE_THRESHOLD_STEP, type = float)
  return parser.parse_args()
def load_checkpoint(checkpoint_path):
  return torch.load(checkpoint_path, map_location = 'cpu', weights_only = False)
def configure_args(checkpoint_args, cli_args):
  checkpoint_args.device = cli_args.device
  checkpoint_args.batch_size = cli_args.batch_size
  checkpoint_args.num_workers = cli_args.num_workers
  checkpoint_args.output_dir = cli_args.output_dir
  checkpoint_args.eval = True
  checkpoint_args.distributed = False
  checkpoint_args.coco_path = cli_args.coco_path
  checkpoint_args.coco_val_images = cli_args.coco_train_images
  checkpoint_args.coco_val_ann = cli_args.coco_train_ann
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
  return [float(x), float(y), float(x + width), float(y + height)]
def box_iou(boxes_a, boxes_b):
  if len(boxes_a) == 0 or len(boxes_b) == 0:
    return np.zeros((len(boxes_a), len(boxes_b)), dtype = np.float64)
  boxes_a = np.asarray(boxes_a, dtype = np.float64)
  boxes_b = np.asarray(boxes_b, dtype = np.float64)
  top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
  bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
  widths_heights = np.clip(bottom_right - top_left, a_min = 0, a_max = None)
  intersection = widths_heights[:, :, 0] * widths_heights[:, :, 1]
  area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
  area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
  union = area_a[:, None] + area_b[None, :] - intersection
  return np.divide(intersection, union, out = np.zeros_like(intersection), where = union > 0)
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
  raise RuntimeError('Unable to infer label mapping. Set LABEL_MODE to category_id or zero_based.')
def map_prediction_label(raw_label, category_ids, label_mode):
  if label_mode == 'zero_based':
    if 0 <= raw_label < len(category_ids):
      return category_ids[raw_label]
    return None
  if raw_label in category_ids:
    return raw_label
  return None
def resolve_excluded_category_ids(categories, excluded_classes):
  category_id_by_name = {str(category['name']).casefold(): int(category['id']) for category in categories}
  valid_category_ids = {int(category['id']) for category in categories}
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
    warnings.warn('Unknown excluded classes or category IDs: ' + ', '.join(unknown_classes), RuntimeWarning)
  return excluded_category_ids
def safe_ratio(numerator, denominator):
  if denominator == 0:
    return 0.0
  return numerator / denominator
def calculate_metrics(true_positive, false_positive, false_negative):
  precision = safe_ratio(true_positive, true_positive + false_positive)
  recall = safe_ratio(true_positive, true_positive + false_negative)
  f1 = safe_ratio(2.0 * precision * recall, precision + recall)
  accuracy = safe_ratio(true_positive, true_positive + false_positive + false_negative)
  return {'true_positive': int(true_positive), 'false_positive': int(false_positive), 'false_negative': int(false_negative), 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
def create_threshold_grid(cli_args):
  if cli_args.confidence_threshold_step <= 0:
    raise ValueError('confidence threshold step must be greater than zero.')
  if cli_args.min_confidence_threshold < 0:
    raise ValueError('minimum confidence threshold cannot be negative.')
  if cli_args.max_confidence_threshold > 1:
    raise ValueError('maximum confidence threshold cannot exceed 1.')
  if cli_args.min_confidence_threshold > cli_args.max_confidence_threshold:
    raise ValueError('minimum confidence threshold cannot exceed maximum threshold.')
  thresholds = np.arange(cli_args.min_confidence_threshold, cli_args.max_confidence_threshold + cli_args.confidence_threshold_step / 2.0, cli_args.confidence_threshold_step)
  thresholds = np.append(thresholds, cli_args.default_confidence_threshold)
  return sorted({round(float(threshold), 10) for threshold in thresholds if 0 <= threshold <= 1})
def evaluate_class_threshold(image_records, category_id, confidence_threshold, iou_threshold):
  true_positive = 0
  false_positive = 0
  false_negative = 0
  for image_record in image_records:
    gt_boxes = [gt_box for gt_box, gt_category_id in zip(image_record['gt_boxes'], image_record['gt_category_ids']) if gt_category_id == category_id]
    pred_boxes = [pred_box for pred_box, pred_category_id, pred_score in zip(image_record['pred_boxes'], image_record['pred_category_ids'], image_record['pred_scores']) if pred_category_id == category_id and pred_score >= confidence_threshold]
    matches, _, _ = greedy_matches(gt_boxes, pred_boxes, iou_threshold)
    true_positive += len(matches)
    false_positive += len(pred_boxes) - len(matches)
    false_negative += len(gt_boxes) - len(matches)
  return calculate_metrics(true_positive, false_positive, false_negative)
def get_threshold_rank(metrics, threshold, optimization_metric):
  if optimization_metric == 'precision':
    return (metrics['precision'], metrics['recall'], metrics['f1'], -abs(float(threshold) - 0.5))
  if optimization_metric == 'recall':
    return (metrics['recall'], metrics['precision'], metrics['f1'], -abs(float(threshold) - 0.5))
  if optimization_metric == 'accuracy':
    return (metrics['accuracy'], metrics['f1'], metrics['precision'], metrics['recall'], -abs(float(threshold) - 0.5))
  return (metrics['f1'], metrics['precision'], metrics['recall'], -abs(float(threshold) - 0.5))
def optimize_class_thresholds(image_records, category_ids, category_name_by_id, cli_args):
  threshold_grid = create_threshold_grid(cli_args)
  threshold_by_category_id = {}
  selected_rows = []
  search_rows = []
  for category_id in category_ids:
    class_name = category_name_by_id[category_id]
    best_rank = None
    best_row = None
    for threshold in threshold_grid:
      metrics = evaluate_class_threshold(image_records, category_id, threshold, cli_args.iou_threshold)
      row = {'category_id': category_id, 'class': class_name, 'threshold': threshold, 'optimized_for': cli_args.threshold_optimization_metric, **metrics}
      search_rows.append(row)
      rank = get_threshold_rank(metrics, threshold, cli_args.threshold_optimization_metric)
      if best_rank is None or rank > best_rank:
        best_rank = rank
        best_row = row
    threshold_by_category_id[category_id] = best_row['threshold']
    selected_rows.append(best_row)
    print(f'optimized threshold: {class_name} = {best_row["threshold"]:.2f} ({cli_args.threshold_optimization_metric} = {best_row[cli_args.threshold_optimization_metric]:.4f})', flush = True)
  return threshold_by_category_id, pd.DataFrame(selected_rows), pd.DataFrame(search_rows)
def create_default_thresholds(category_ids, category_name_by_id, cli_args):
  rows = []
  threshold_by_category_id = {category_id: cli_args.default_confidence_threshold for category_id in category_ids}
  for category_id in category_ids:
    rows.append({'category_id': category_id, 'class': category_name_by_id[category_id], 'threshold': cli_args.default_confidence_threshold, 'source': 'default'})
  return threshold_by_category_id, pd.DataFrame(rows), pd.DataFrame()
def load_thresholds_csv(thresholds_csv, category_ids, category_name_by_id, cli_args):
  thresholds_path = Path(thresholds_csv).expanduser()
  thresholds_df = pd.read_csv(thresholds_path)
  if 'threshold' not in thresholds_df.columns:
    raise ValueError('thresholds csv must contain a threshold column.')
  if 'category_id' not in thresholds_df.columns and 'class' not in thresholds_df.columns:
    raise ValueError('thresholds csv must contain category_id or class.')
  category_id_by_name = {name: category_id for category_id, name in category_name_by_id.items()}
  threshold_by_category_id = {category_id: cli_args.default_confidence_threshold for category_id in category_ids}
  rows = []
  for _, row in thresholds_df.iterrows():
    if 'category_id' in thresholds_df.columns and not pd.isna(row.get('category_id')):
      category_id = int(row['category_id'])
    else:
      category_id = category_id_by_name[str(row['class'])]
    if category_id not in category_ids:
      continue
    threshold_by_category_id[category_id] = float(row['threshold'])
  for category_id in category_ids:
    rows.append({'category_id': category_id, 'class': category_name_by_id[category_id], 'threshold': threshold_by_category_id[category_id], 'source': str(thresholds_path)})
  return threshold_by_category_id, pd.DataFrame(rows), pd.DataFrame()
def bbox_as_json(bbox):
  return json.dumps([round(float(value), 4) for value in bbox])
def collect_image_records(model, postprocessors, data_loader, coco, all_category_ids, excluded_category_ids, cli_args, device):
  image_records = []
  label_mode = None
  skipped_unknown_predictions = 0
  skipped_excluded_predictions = 0
  processed_images = 0
  with torch.inference_mode():
    for samples, targets in data_loader:
      samples = samples.to(device)
      outputs = model(samples)
      original_sizes = torch.stack([target['orig_size'] for target in targets]).to(device)
      results = postprocessors['bbox'](outputs, original_sizes)
      if label_mode is None:
        label_mode = resolve_label_mode(all_category_ids, int(outputs['pred_logits'].shape[-1]), cli_args.label_mode)
        print(f'label mode: {label_mode}', flush = True)
      for target, result in zip(targets, results):
        if cli_args.max_images is not None and processed_images >= cli_args.max_images:
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
        for bbox, raw_label, score in zip(result['boxes'].detach().cpu().tolist(), result['labels'].detach().cpu().tolist(), result['scores'].detach().cpu().tolist()):
          category_id = map_prediction_label(int(raw_label), all_category_ids, label_mode)
          if category_id is None:
            skipped_unknown_predictions += 1
            continue
          if category_id in excluded_category_ids:
            skipped_excluded_predictions += 1
            continue
          pred_boxes.append([float(value) for value in bbox])
          pred_category_ids.append(category_id)
          pred_scores.append(float(score))
        image_records.append({'image_id': image_id, 'file_name': image_info.get('file_name', ''), 'width': image_info.get('width', np.nan), 'height': image_info.get('height', np.nan), 'gt_boxes': gt_boxes, 'gt_category_ids': gt_category_ids, 'pred_boxes': pred_boxes, 'pred_category_ids': pred_category_ids, 'pred_scores': pred_scores})
        processed_images += 1
        if processed_images % 50 == 0:
          print(f'processed images: {processed_images}', flush = True)
      if cli_args.max_images is not None and processed_images >= cli_args.max_images:
        break
  return image_records, processed_images, skipped_unknown_predictions, skipped_excluded_predictions
def filter_predictions(image_record, threshold_by_category_id):
  pred_boxes = []
  pred_category_ids = []
  pred_scores = []
  pred_thresholds = []
  for pred_box, pred_category_id, pred_score in zip(image_record['pred_boxes'], image_record['pred_category_ids'], image_record['pred_scores']):
    threshold = threshold_by_category_id[pred_category_id]
    if pred_score < threshold:
      continue
    pred_boxes.append(pred_box)
    pred_category_ids.append(pred_category_id)
    pred_scores.append(pred_score)
    pred_thresholds.append(threshold)
  return pred_boxes, pred_category_ids, pred_scores, pred_thresholds
def append_detail_row(rows, image_record, match_type, actual_class, predicted_class, iou, score, threshold, gt_bbox, pred_bbox):
  rows.append({'image_id': image_record['image_id'], 'file_name': image_record['file_name'], 'match_type': match_type, 'actual_class': actual_class, 'predicted_class': predicted_class, 'iou': iou, 'score': score, 'threshold': threshold, 'gt_bbox_xyxy': '' if gt_bbox is None else bbox_as_json(gt_bbox), 'pred_bbox_xyxy': '' if pred_bbox is None else bbox_as_json(pred_bbox)})
def evaluate_image_records(image_records, category_name_by_id, threshold_by_category_id, iou_threshold):
  image_rows = []
  detail_rows = []
  for image_record in image_records:
    gt_boxes = image_record['gt_boxes']
    gt_category_ids = image_record['gt_category_ids']
    pred_boxes, pred_category_ids, pred_scores, pred_thresholds = filter_predictions(image_record, threshold_by_category_id)
    matches, matched_gt, matched_pred = greedy_matches(gt_boxes, pred_boxes, iou_threshold)
    true_positive = 0
    misclassified = 0
    matched_ious = []
    for gt_idx, pred_idx, iou in matches:
      actual_class = category_name_by_id[gt_category_ids[gt_idx]]
      predicted_class = category_name_by_id[pred_category_ids[pred_idx]]
      if actual_class == predicted_class:
        true_positive += 1
        match_type = 'true_positive'
        matched_ious.append(float(iou))
      else:
        misclassified += 1
        match_type = 'misclassified'
      append_detail_row(detail_rows, image_record, match_type, actual_class, predicted_class, float(iou), pred_scores[pred_idx], pred_thresholds[pred_idx], gt_boxes[gt_idx], pred_boxes[pred_idx])
    for gt_idx, gt_category_id in enumerate(gt_category_ids):
      if gt_idx in matched_gt:
        continue
      append_detail_row(detail_rows, image_record, 'false_negative', category_name_by_id[gt_category_id], 'background', np.nan, np.nan, np.nan, gt_boxes[gt_idx], None)
    for pred_idx, pred_category_id in enumerate(pred_category_ids):
      if pred_idx in matched_pred:
        continue
      append_detail_row(detail_rows, image_record, 'false_positive', 'background', category_name_by_id[pred_category_id], np.nan, pred_scores[pred_idx], pred_thresholds[pred_idx], None, pred_boxes[pred_idx])
    false_negative = (len(gt_boxes) - len(matched_gt)) + misclassified
    false_positive = (len(pred_boxes) - len(matched_pred)) + misclassified
    metrics = calculate_metrics(true_positive, false_positive, false_negative)
    error_count = false_positive + false_negative
    mean_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    score_sum_fp = sum(pred_scores[pred_idx] for pred_idx in range(len(pred_scores)) if pred_idx not in matched_pred)
    image_rows.append({'image_id': image_record['image_id'], 'file_name': image_record['file_name'], 'width': image_record['width'], 'height': image_record['height'], 'ground_truth': len(gt_boxes), 'predicted': len(pred_boxes), 'matched': len(matches), 'misclassified': misclassified, 'error_count': error_count, 'mean_matched_iou': mean_matched_iou, 'score_sum_fp': score_sum_fp, **metrics})
  image_df = pd.DataFrame(image_rows)
  detail_df = pd.DataFrame(detail_rows)
  image_df = image_df.sort_values(['f1', 'error_count', 'false_negative', 'false_positive', 'mean_matched_iou'], ascending = [True, False, False, False, True])
  return image_df, detail_df
def resolve_image_path(images_root, file_name):
  file_path = Path(file_name)
  if file_path.is_absolute() and file_path.exists():
    return file_path
  return Path(images_root).expanduser() / file_name
def draw_box(draw, bbox, outline, label, width = 3, draw_label = True):
  x0, y0, x1, y1 = [float(value) for value in bbox]
  draw.rectangle([x0, y0, x1, y1], outline = outline, width = width)
  if not draw_label:
    return
  try:
    font = ImageFont.load_default()
  except Exception:
    font = None
  text_pos = (x0 + 2, max(0, y0 - 12))
  draw.text(text_pos, label, fill = outline, font = font)
def export_poor_images(image_df, detail_df, output_dir, images_root, top_k_images):
  image_output_dir = output_dir / 'poorest_images'
  image_output_dir.mkdir(parents = True, exist_ok = True)
  poorest_df = image_df.head(top_k_images).copy()
  for rank, (_, image_row) in enumerate(poorest_df.iterrows(), start = 1):
    image_path = resolve_image_path(images_root, image_row['file_name'])
    if not image_path.exists():
      warnings.warn(f'image not found: {image_path}', RuntimeWarning)
      continue
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    image_details = detail_df.loc[detail_df['image_id'] == image_row['image_id']]
    for _, detail_row in image_details.iterrows():
      if detail_row['match_type'] == 'true_positive':
        if detail_row['gt_bbox_xyxy']:
          draw_box(draw, json.loads(detail_row['gt_bbox_xyxy']), 'lime', f'GT/PRED {detail_row["actual_class"]}')
        continue
      if detail_row['match_type'] == 'misclassified':
        if detail_row['gt_bbox_xyxy']:
          draw_box(draw, json.loads(detail_row['gt_bbox_xyxy']), 'orange', f'GT {detail_row["actual_class"]}')
        if detail_row['pred_bbox_xyxy']:
          draw_box(draw, json.loads(detail_row['pred_bbox_xyxy']), 'yellow', f'PRED {detail_row["predicted_class"]}')
        continue
      if detail_row['match_type'] == 'false_negative' and detail_row['gt_bbox_xyxy']:
        gt_bbox = json.loads(detail_row['gt_bbox_xyxy'])
        draw_box(draw, gt_bbox, 'yellow', '', width = 7, draw_label = False)
        draw_box(draw, gt_bbox, 'red', f'FN {detail_row["actual_class"]}', width = 3)
      if detail_row['match_type'] == 'false_positive' and detail_row['pred_bbox_xyxy']:
        draw_box(draw, json.loads(detail_row['pred_bbox_xyxy']), 'cyan', f'FP {detail_row["predicted_class"]} {detail_row["score"]:.2f}')
    safe_name = Path(str(image_row['file_name'])).name
    output_name = f'{rank:06d}_f1_{image_row["f1"]:.3f}_err_{int(image_row["error_count"])}_{safe_name}'
    image.save(image_output_dir / output_name)
def main():
  cli_args = parse_args()
  excluded_classes = EXCLUDED_CLASSES if cli_args.exclude_classes is None else cli_args.exclude_classes
  output_dir = Path(cli_args.output_dir).expanduser()
  output_dir.mkdir(parents = True, exist_ok = True)
  checkpoint_path = Path(cli_args.checkpoint).expanduser()
  checkpoint = load_checkpoint(checkpoint_path)
  if 'args' not in checkpoint:
    raise RuntimeError('Checkpoint does not contain training arguments under checkpoint["args"].')
  args = configure_args(checkpoint['args'], cli_args)
  device = torch.device(args.device)
  model, _, postprocessors = build_model(args)
  model.load_state_dict(checkpoint['model'], strict = True)
  model.to(device)
  model.eval()
  dataset_eval = build_dataset(image_set = 'val', args = args)
  coco_dataset = get_coco_dataset(dataset_eval)
  coco = coco_dataset.coco
  all_category_ids = sorted(coco.getCatIds())
  categories = coco.loadCats(all_category_ids)
  category_name_by_id = {int(category['id']): category['name'] for category in categories}
  excluded_category_ids = resolve_excluded_category_ids(categories, excluded_classes)
  category_ids = [category_id for category_id in all_category_ids if category_id not in excluded_category_ids]
  if not category_ids:
    raise RuntimeError('All classes were excluded from evaluation.')
  excluded_class_names = [category_name_by_id[category_id] for category_id in sorted(excluded_category_ids)]
  print('excluded classes: ' + ', '.join(excluded_class_names) if excluded_class_names else 'excluded classes: none', flush = True)
  data_loader = DataLoader(dataset_eval, batch_size = args.batch_size, shuffle = False, collate_fn = utils.collate_fn, num_workers = args.num_workers, pin_memory = True)
  image_records, processed_images, skipped_unknown_predictions, skipped_excluded_predictions = collect_image_records(model, postprocessors, data_loader, coco, all_category_ids, excluded_category_ids, cli_args, device)
  if cli_args.thresholds_csv is not None:
    threshold_by_category_id, selected_thresholds_df, threshold_search_df = load_thresholds_csv(cli_args.thresholds_csv, category_ids, category_name_by_id, cli_args)
  elif cli_args.optimize_thresholds:
    threshold_by_category_id, selected_thresholds_df, threshold_search_df = optimize_class_thresholds(image_records, category_ids, category_name_by_id, cli_args)
  else:
    threshold_by_category_id, selected_thresholds_df, threshold_search_df = create_default_thresholds(category_ids, category_name_by_id, cli_args)
  image_df, detail_df = evaluate_image_records(image_records, category_name_by_id, threshold_by_category_id, cli_args.iou_threshold)
  image_df.to_csv(output_dir / 'train_image_performance.csv', index = False)
  detail_df.to_csv(output_dir / 'train_prediction_matches.csv', index = False)
  selected_thresholds_df.to_csv(output_dir / 'class_thresholds_used.csv', index = False)
  if not threshold_search_df.empty:
    threshold_search_df.to_csv(output_dir / 'threshold_search_results.csv', index = False)
  if cli_args.export_images:
    export_poor_images(image_df, detail_df, output_dir, cli_args.coco_train_images, cli_args.top_k_images)
  print('', flush = True)
  print('poorest train images:', flush = True)
  print(image_df.head(min(cli_args.top_k_images, 25)).to_string(index = False), flush = True)
  print('', flush = True)
  print(f'processed images: {processed_images}', flush = True)
  print(f'skipped predictions with unknown category mapping: {skipped_unknown_predictions}', flush = True)
  print(f'skipped predictions for excluded classes: {skipped_excluded_predictions}', flush = True)
  print(f'outputs written to: {output_dir}', flush = True)
if __name__ == '__main__':
  main()
