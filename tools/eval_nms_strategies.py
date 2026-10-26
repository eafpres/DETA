#!/usr/bin/env python3
"""Compare baseline, per-class hard NMS, and Gaussian Soft-NMS."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))
from datasets import build_dataset
from models import build_model
from util import box_ops
import util.misc as utils
from tools.eval_confusion_matrix import calculate_metrics
from tools.eval_confusion_matrix import create_confusion_matrix
from tools.eval_confusion_matrix import create_final_metrics
from tools.eval_confusion_matrix import create_threshold_grid
from tools.eval_confusion_matrix import evaluate_class_threshold
from tools.eval_confusion_matrix import get_coco_dataset
from tools.eval_confusion_matrix import get_threshold_rank
from tools.eval_confusion_matrix import optimize_class_thresholds
from tools.eval_confusion_matrix import plot_matrix
from tools.eval_confusion_matrix import xywh_to_xyxy
#
# Raw model outputs are cached before suppression so every strategy operates on
# identical predictions and the expensive neural-network pass runs only once.
#
def parse_args():
  """Parse command-line options.
  Returns:
    argparse.Namespace: Validated evaluation configuration.
  """
  parser = argparse.ArgumentParser(
    description = (
      'Jointly tune per-class hard-NMS/confidence thresholds and compare '
      'them with tuned Gaussian Soft-NMS.'
    )
  )
  parser.add_argument('--checkpoint', required = True)
  parser.add_argument('--output-dir', required = True)
  parser.add_argument('--coco-path', required = True)
  parser.add_argument('--coco-val-images', required = True)
  parser.add_argument('--coco-val-ann', required = True)
  parser.add_argument('--device', default = 'cuda')
  parser.add_argument('--batch-size', default = 2, type = int)
  parser.add_argument('--num-workers', default = 2, type = int)
  parser.add_argument('--iou-threshold', default = 0.50, type = float)
  parser.add_argument('--top-k', default = 100, type = int)
  parser.add_argument('--baseline-nms-iou', default = 0.40, type = float)
  parser.add_argument(
    '--hard-nms-grid',
    default = (
      '0.20,0.25,0.30,0.32,0.34,0.35,0.36,0.38,'
      '0.40,0.45,0.48,0.50,0.55,0.60,0.65,0.70'
    )
  )
  parser.add_argument(
    '--soft-nms-sigma-grid',
    default = '0.10,0.20,0.30,0.40,0.50,0.60,0.75,1.00'
  )
  parser.add_argument('--coordinate-passes', default = 3, type = int)
  parser.add_argument('--default-confidence-threshold', default = 0.40, type = float)
  parser.add_argument('--min-confidence-threshold', default = 0.05, type = float)
  parser.add_argument('--max-confidence-threshold', default = 0.95, type = float)
  parser.add_argument('--confidence-threshold-step', default = 0.01, type = float)
  parser.add_argument('--threshold-optimization-metric', default = 'f1')
  parser.add_argument(
    '--reuse-raw-cache',
    action = argparse.BooleanOptionalAction,
    default = True
  )
  return parser.parse_args()
def parse_float_grid(value, option_name):
  """Parse a comma-separated positive float grid.
  Args:
    value: Comma-separated numeric values.
    option_name: Option name used in validation errors.
  Returns:
    list[float]: Sorted unique values.
  Raises:
    ValueError: If the grid is empty or contains invalid values.
  """
  try:
    values = sorted({float(item.strip()) for item in value.split(',')})
  except ValueError as error:
    raise ValueError(f'{option_name} must contain only numbers.') from error
  if not values:
    raise ValueError(f'{option_name} cannot be empty.')
  if any(not math.isfinite(item) or item <= 0 for item in values):
    raise ValueError(f'{option_name} values must be finite and positive.')
  return values
def file_identity(path):
  """Return inexpensive file identity metadata for cache validation.
  Args:
    path: File whose size and modification time identify the input.
  Returns:
    dict: Resolved path, byte count, and nanosecond modification time.
  """
  resolved = Path(path).expanduser().resolve()
  stat = resolved.stat()
  return {
    'path': str(resolved),
    'size': int(stat.st_size),
    'mtime_ns': int(stat.st_mtime_ns)
  }
def configure_checkpoint_args(checkpoint_args, cli_args):
  """Apply evaluation dataset and runtime settings to checkpoint arguments.
  Args:
    checkpoint_args: Training arguments stored in the checkpoint.
    cli_args: Current evaluation arguments.
  Returns:
    argparse.Namespace: A detached, evaluation-ready argument object.
  """
  model_args = copy.deepcopy(checkpoint_args)
  model_args.device = cli_args.device
  model_args.batch_size = cli_args.batch_size
  model_args.num_workers = cli_args.num_workers
  model_args.output_dir = cli_args.output_dir
  model_args.eval = True
  model_args.distributed = False
  model_args.coco_path = cli_args.coco_path
  model_args.coco_val_images = cli_args.coco_val_images
  model_args.coco_val_ann = cli_args.coco_val_ann
  return model_args
def load_raw_cache(cache_path, expected_identity):
  """Load raw outputs only when their source inputs still match.
  Args:
    cache_path: Serialized raw-output cache.
    expected_identity: Current checkpoint and annotation identities.
  Returns:
    dict | None: Cache payload when valid, otherwise ``None``.
  """
  if not cache_path.is_file():
    return None
  payload = torch.load(cache_path, map_location = 'cpu', weights_only = False)
  if payload.get('identity') != expected_identity:
    print('raw cache inputs changed; regenerating cache', flush = True)
    return None
  print(f'loaded raw model outputs: {cache_path}', flush = True)
  return payload
def collect_raw_outputs(cli_args, expected_identity, cache_path):
  """Run the model once and retain unsuppressed boxes and class scores.
  Args:
    cli_args: Evaluation configuration.
    expected_identity: Source identity stored beside cached outputs.
    cache_path: Destination for the reusable raw-output cache.
  Returns:
    dict: Categories and per-image unsuppressed predictions/annotations.
  """
  checkpoint = torch.load(
    Path(cli_args.checkpoint).expanduser(),
    map_location = 'cpu',
    weights_only = False
  )
  if 'args' not in checkpoint:
    raise RuntimeError('Checkpoint does not contain checkpoint["args"].')
  model_args = configure_checkpoint_args(checkpoint['args'], cli_args)
  device = torch.device(cli_args.device)
  model, _, _ = build_model(model_args)
  model.load_state_dict(checkpoint['model'], strict = True)
  model.to(device)
  model.eval()
  dataset = build_dataset(image_set = 'val', args = model_args)
  coco_dataset = get_coco_dataset(dataset)
  coco = coco_dataset.coco
  category_ids = sorted(int(item) for item in coco.getCatIds())
  categories = coco.loadCats(category_ids)
  category_name_by_id = {
    int(category['id']): str(category['name'])
    for category in categories
  }
  loader = DataLoader(
    dataset,
    batch_size = cli_args.batch_size,
    shuffle = False,
    collate_fn = utils.collate_fn,
    num_workers = cli_args.num_workers,
    pin_memory = True
  )
  raw_records = []
  processed = 0
  with torch.inference_mode():
    for samples, targets in loader:
      outputs = model(samples.to(device))
      scores = outputs['pred_logits'].sigmoid().detach().cpu()
      normalized_boxes = box_ops.box_cxcywh_to_xyxy(
        outputs['pred_boxes']
      ).detach().cpu()
      for target, image_scores, image_boxes in zip(
        targets,
        scores,
        normalized_boxes
      ):
        image_id = int(target['image_id'].item())
        image_info = coco.loadImgs([image_id])[0]
        image_height, image_width = [
          float(value)
          for value in target['orig_size'].tolist()
        ]
        scale = torch.tensor(
          [image_width, image_height, image_width, image_height],
          dtype = image_boxes.dtype
        )
        annotation_ids = coco.getAnnIds(imgIds = [image_id])
        annotations = coco.loadAnns(annotation_ids)
        gt_boxes = []
        gt_category_ids = []
        for annotation in annotations:
          if int(annotation.get('iscrowd', 0)) != 0:
            continue
          bbox = xywh_to_xyxy(annotation['bbox'])
          if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
          gt_boxes.append(bbox)
          gt_category_ids.append(int(annotation['category_id']))
        raw_records.append({
          'image_id': image_id,
          'file_name': image_info.get('file_name', ''),
          'gt_boxes': gt_boxes,
          'gt_category_ids': gt_category_ids,
          'boxes': (image_boxes * scale).contiguous(),
          'scores': image_scores.contiguous()
        })
        processed += 1
        if processed % 50 == 0:
          print(f'raw inference images: {processed}', flush = True)
  payload = {
    'identity': expected_identity,
    'category_ids': category_ids,
    'category_name_by_id': category_name_by_id,
    'raw_records': raw_records,
    'num_classes': int(raw_records[0]['scores'].shape[1])
  }
  if payload['num_classes'] != len(category_ids):
    raise RuntimeError(
      'This analysis requires compact zero-based model labels matching the '
      f'{len(category_ids)} categories; model emitted {payload["num_classes"]}.'
    )
  torch.save(payload, cache_path)
  print(f'wrote raw model outputs: {cache_path}', flush = True)
  return payload
def box_iou_one_to_many(box, boxes):
  """Calculate IoU between one box and an array of boxes.
  Args:
    box: Reference XYXY box.
    boxes: Candidate XYXY boxes.
  Returns:
    np.ndarray: IoU for each candidate.
  """
  if len(boxes) == 0:
    return np.empty((0,), dtype = np.float64)
  top_left = np.maximum(box[:2], boxes[:, :2])
  bottom_right = np.minimum(box[2:], boxes[:, 2:])
  widths_heights = np.clip(bottom_right - top_left, 0.0, None)
  intersection = widths_heights[:, 0] * widths_heights[:, 1]
  box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
  areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(
    boxes[:, 3] - boxes[:, 1],
    0.0,
    None
  )
  union = box_area + areas - intersection
  return np.divide(
    intersection,
    union,
    out = np.zeros_like(intersection),
    where = union > 0
  )
def gaussian_soft_nms(boxes, scores, sigma, minimum_score, top_k):
  """Apply class-local Gaussian Soft-NMS.
  Args:
    boxes: Candidate XYXY boxes.
    scores: Candidate class confidences.
    sigma: Gaussian decay scale in ``exp(-(IoU ** 2) / sigma)``.
    minimum_score: Lowest score relevant to the confidence search.
    top_k: Maximum candidates needed before global top-k selection.
  Returns:
    tuple[np.ndarray, np.ndarray]: Original indices and decayed scores.
  """
  valid_indices = np.flatnonzero(scores >= minimum_score).astype(np.int64)
  if len(valid_indices) == 0:
    return valid_indices, np.empty((0,), dtype = np.float32)
  work_indices = valid_indices.copy()
  work_scores = scores[work_indices].astype(np.float64, copy = True)
  selected_indices = []
  selected_scores = []
  while len(work_indices) > 0 and len(selected_indices) < top_k:
    best_position = int(np.argmax(work_scores))
    best_score = float(work_scores[best_position])
    if best_score < minimum_score:
      break
    best_index = int(work_indices[best_position])
    selected_indices.append(best_index)
    selected_scores.append(best_score)
    keep_mask = np.ones(len(work_indices), dtype = bool)
    keep_mask[best_position] = False
    remaining_indices = work_indices[keep_mask]
    remaining_scores = work_scores[keep_mask]
    if len(remaining_indices) == 0:
      break
    overlaps = box_iou_one_to_many(
      boxes[best_index],
      boxes[remaining_indices]
    )
    remaining_scores *= np.exp(-(overlaps ** 2) / sigma)
    work_indices = remaining_indices
    work_scores = remaining_scores
  return (
    np.asarray(selected_indices, dtype = np.int64),
    np.asarray(selected_scores, dtype = np.float32)
  )
def precompute_hard_candidates(raw_records, threshold_grid, minimum_score, top_k):
  """Precompute class-local hard-NMS survivors for every grid point.
  Args:
    raw_records: Unsuppressed per-image model outputs.
    threshold_grid: Candidate hard-NMS IoU thresholds.
    minimum_score: Lowest confidence relevant to later evaluation.
    top_k: Per-class candidate cap that preserves the global top-k result.
  Returns:
    dict: Threshold-keyed image/class candidate tuples.
  """
  cache = {}
  for threshold in threshold_grid:
    print(f'precomputing hard NMS IoU {threshold:.2f}', flush = True)
    threshold_records = []
    for raw_record in raw_records:
      class_candidates = []
      boxes = raw_record['boxes']
      scores_matrix = raw_record['scores']
      for class_index in range(scores_matrix.shape[1]):
        class_scores = scores_matrix[:, class_index]
        valid_indices = torch.nonzero(
          class_scores >= minimum_score,
          as_tuple = False
        ).flatten()
        if valid_indices.numel() == 0:
          class_candidates.append((
            np.empty((0,), dtype = np.int64),
            np.empty((0,), dtype = np.float32)
          ))
          continue
        kept_positions = nms(
          boxes[valid_indices],
          class_scores[valid_indices],
          threshold
        )[:top_k]
        kept_indices = valid_indices[kept_positions]
        class_candidates.append((
          kept_indices.numpy().astype(np.int64, copy = False),
          class_scores[kept_indices].numpy().astype(np.float32, copy = False)
        ))
      threshold_records.append(class_candidates)
    cache[threshold] = threshold_records
  return cache
def precompute_soft_candidates(raw_records, sigma_grid, minimum_score, top_k):
  """Precompute class-local Gaussian Soft-NMS candidates for every sigma.
  Args:
    raw_records: Unsuppressed per-image model outputs.
    sigma_grid: Candidate Gaussian decay scales.
    minimum_score: Lowest confidence relevant to later evaluation.
    top_k: Per-class candidate cap that preserves the global top-k result.
  Returns:
    dict: Sigma-keyed image/class candidate tuples.
  """
  cache = {}
  for sigma in sigma_grid:
    print(f'precomputing Gaussian Soft-NMS sigma {sigma:.2f}', flush = True)
    sigma_records = []
    for raw_record in raw_records:
      boxes = raw_record['boxes'].numpy()
      scores_matrix = raw_record['scores'].numpy()
      class_candidates = []
      for class_index in range(scores_matrix.shape[1]):
        class_candidates.append(gaussian_soft_nms(
          boxes,
          scores_matrix[:, class_index],
          sigma,
          minimum_score,
          top_k
        ))
      sigma_records.append(class_candidates)
    cache[sigma] = sigma_records
  return cache
def build_image_records(raw_records, candidate_records_by_class, category_ids, top_k):
  """Combine class-local candidates and reproduce DETA's global top-k step.
  Args:
    raw_records: Unsuppressed predictions and ground truth.
    candidate_records_by_class: Per-class candidate source for each image.
    category_ids: COCO category IDs in raw-logit order.
    top_k: Maximum predictions retained per image.
  Returns:
    list[dict]: Records accepted by the existing threshold evaluator.
  """
  image_records = []
  for image_index, raw_record in enumerate(raw_records):
    candidate_boxes = []
    candidate_category_ids = []
    candidate_scores = []
    for class_index, category_id in enumerate(category_ids):
      indices, scores = candidate_records_by_class[
        class_index
      ][image_index]
      if len(indices) == 0:
        continue
      boxes = raw_record['boxes'][indices].numpy().tolist()
      candidate_boxes.extend(boxes)
      candidate_category_ids.extend([category_id] * len(indices))
      candidate_scores.extend(float(score) for score in scores)
    order = np.argsort(
      np.asarray(candidate_scores, dtype = np.float64),
      kind = 'stable'
    )[::-1][:top_k]
    image_records.append({
      'image_id': raw_record['image_id'],
      'file_name': raw_record['file_name'],
      'gt_boxes': raw_record['gt_boxes'],
      'gt_category_ids': raw_record['gt_category_ids'],
      'pred_boxes': [candidate_boxes[index] for index in order],
      'pred_category_ids': [
        candidate_category_ids[index]
        for index in order
      ],
      'pred_scores': [candidate_scores[index] for index in order]
    })
  return image_records
def hard_sources_for_config(hard_cache, nms_by_class, class_count):
  """Transpose cached hard-NMS results into class-major candidate sources.
  Args:
    hard_cache: Threshold-keyed image/class candidate cache.
    nms_by_class: Hard-NMS IoU threshold for each raw class index.
    class_count: Number of model classes.
  Returns:
    list[list[tuple]]: Class-major candidates indexed by image.
  """
  image_count = len(next(iter(hard_cache.values())))
  return [
    [
      hard_cache[nms_by_class[class_index]][image_index][class_index]
      for image_index in range(image_count)
    ]
    for class_index in range(class_count)
  ]
def soft_sources_for_sigma(soft_cache, sigma, class_count):
  """Transpose a Soft-NMS cache entry into class-major candidate sources.
  Args:
    soft_cache: Sigma-keyed image/class candidate cache.
    sigma: Selected Gaussian decay scale.
    class_count: Number of model classes.
  Returns:
    list[list[tuple]]: Class-major candidates indexed by image.
  """
  image_count = len(soft_cache[sigma])
  return [
    [
      soft_cache[sigma][image_index][class_index]
      for image_index in range(image_count)
    ]
    for class_index in range(class_count)
  ]
def optimize_one_class(image_records, category_id, cli_args):
  """Optimize confidence for one class using the established tie-break rule.
  Args:
    image_records: Post-NMS predictions and annotations.
    category_id: Class being optimized.
    cli_args: Confidence grid and matching configuration.
  Returns:
    dict: Best threshold and metrics for the class.
  """
  best_rank = None
  best_result = None
  for threshold in create_threshold_grid(cli_args):
    metrics = evaluate_class_threshold(
      image_records,
      category_id,
      threshold,
      cli_args.iou_threshold
    )
    rank = get_threshold_rank(
      metrics,
      threshold,
      cli_args.threshold_optimization_metric
    )
    if best_rank is None or rank > best_rank:
      best_rank = rank
      best_result = {'threshold': threshold, **metrics}
  return best_result
def summarize_metrics(metrics_df):
  """Create aggregate micro and macro metrics from classwise counts.
  Args:
    metrics_df: Final per-class metric rows.
  Returns:
    dict: Aggregate counts and precision/recall/F1 values.
  """
  true_positive = int(metrics_df['true_positive'].sum())
  false_positive = int(metrics_df['false_positive'].sum())
  false_negative = int(metrics_df['false_negative'].sum())
  micro = calculate_metrics(true_positive, false_positive, false_negative)
  return {
    'ground_truth': int(metrics_df['ground_truth'].sum()),
    'predicted': int(metrics_df['predicted'].sum()),
    **micro,
    'micro_f1': micro['f1'],
    'macro_f1': float(metrics_df['f1'].mean())
  }
def evaluate_strategy(
  strategy_name,
  image_records,
  category_ids,
  category_name_by_id,
  cli_args,
  output_dir = None
):
  """Optimize confidences and calculate final detection metrics.
  Args:
    strategy_name: Human-readable strategy identifier.
    image_records: Post-NMS predictions and annotations.
    category_ids: Ordered included category IDs.
    category_name_by_id: Category names keyed by ID.
    cli_args: Confidence and evaluation settings.
    output_dir: Optional artifact directory for the full result.
  Returns:
    dict: Thresholds, class metrics, aggregate metrics, and records.
  """
  print(f'evaluating strategy: {strategy_name}', flush = True)
  thresholds, threshold_df, search_df = optimize_class_thresholds(
    image_records,
    category_ids,
    category_name_by_id,
    cli_args
  )
  class_names = [category_name_by_id[item] for item in category_ids]
  category_id_by_name = {
    name: category_id
    for category_id, name in category_name_by_id.items()
  }
  matrix, matches_df = create_confusion_matrix(
    image_records,
    category_ids,
    category_name_by_id,
    thresholds,
    cli_args.iou_threshold
  )
  metrics_df = create_final_metrics(
    matrix,
    class_names,
    category_id_by_name,
    thresholds
  )
  summary = summarize_metrics(metrics_df)
  if output_dir is not None:
    output_dir.mkdir(parents = True, exist_ok = True)
    metrics_df.to_csv(output_dir / 'class_metrics.csv', index = False)
    threshold_df.to_csv(output_dir / 'class_thresholds.csv', index = False)
    search_df.to_csv(output_dir / 'threshold_search_results.csv', index = False)
    matches_df.to_csv(output_dir / 'prediction_matches.csv', index = False)
    output_matrix = matrix.T
    output_matrix.index.name = 'predicted_class'
    output_matrix.columns.name = 'actual_class'
    output_matrix.to_csv(output_dir / 'confusion_matrix_counts.csv')
    normalized = matrix.div(
      matrix.sum(axis = 1).replace(0, np.nan),
      axis = 0
    ).fillna(0.0)
    normalized_output = normalized.T
    normalized_output.index.name = 'predicted_class'
    normalized_output.columns.name = 'actual_class'
    normalized_output.to_csv(output_dir / 'confusion_matrix_normalized.csv')
    plot_matrix(
      matrix,
      output_dir / 'confusion_matrix_counts.png',
      f'{strategy_name}: counts',
      'd'
    )
    plot_matrix(
      normalized,
      output_dir / 'confusion_matrix_normalized.png',
      f'{strategy_name}: row-normalized',
      '.2f'
    )
  return {
    'name': strategy_name,
    'thresholds': thresholds,
    'threshold_df': threshold_df,
    'metrics_df': metrics_df,
    'summary': summary,
    'image_records': image_records
  }
def rank_class_result(result, nms_threshold, baseline_nms, metric_name):
  """Rank joint confidence/NMS settings with deterministic tie breaks.
  Args:
    result: Optimized class metrics and confidence threshold.
    nms_threshold: Candidate hard-NMS overlap threshold.
    baseline_nms: Current NMS threshold preferred when metrics tie.
    metric_name: Primary metric used for confidence optimization.
  Returns:
    tuple: Lexicographically ordered selection rank.
  """
  base_rank = get_threshold_rank(
    result,
    result['threshold'],
    metric_name
  )
  return (*base_rank, -abs(nms_threshold - baseline_nms))
def select_per_class_hard_nms(
  raw_records,
  hard_cache,
  hard_grid,
  category_ids,
  category_name_by_id,
  cli_args
):
  """Coordinate-optimize classwise hard-NMS and confidence thresholds.
  Args:
    raw_records: Unsuppressed model outputs.
    hard_cache: Precomputed hard-NMS candidates.
    hard_grid: Candidate NMS thresholds.
    category_ids: Ordered category IDs matching raw logits.
    category_name_by_id: Category names keyed by ID.
    cli_args: Evaluation and coordinate-search configuration.
  Returns:
    tuple[list[float], pd.DataFrame]: Selected thresholds and search history.
  """
  class_count = len(category_ids)
  selected = [cli_args.baseline_nms_iou] * class_count
  history = []
  for pass_index in range(cli_args.coordinate_passes):
    changed = False
    print(f'per-class hard-NMS coordinate pass {pass_index + 1}', flush = True)
    for class_index, category_id in enumerate(category_ids):
      best_rank = None
      best_nms = None
      for nms_threshold in hard_grid:
        candidate_config = list(selected)
        candidate_config[class_index] = nms_threshold
        sources = hard_sources_for_config(
          hard_cache,
          candidate_config,
          class_count
        )
        image_records = build_image_records(
          raw_records,
          sources,
          category_ids,
          cli_args.top_k
        )
        result = optimize_one_class(image_records, category_id, cli_args)
        rank = rank_class_result(
          result,
          nms_threshold,
          cli_args.baseline_nms_iou,
          cli_args.threshold_optimization_metric
        )
        history.append({
          'pass': pass_index + 1,
          'category_id': category_id,
          'class': category_name_by_id[category_id],
          'nms_iou': nms_threshold,
          **result
        })
        if best_rank is None or rank > best_rank:
          best_rank = rank
          best_nms = nms_threshold
      if best_nms != selected[class_index]:
        changed = True
      selected[class_index] = best_nms
      print(
        f'joint optimum: {category_name_by_id[category_id]} '
        f'NMS={best_nms:.2f}',
        flush = True
      )
    if not changed:
      print('per-class hard-NMS search converged', flush = True)
      break
  return selected, pd.DataFrame(history)
def class_comparison(results, category_ids, category_name_by_id):
  """Create a side-by-side per-class metric table.
  Args:
    results: Evaluated strategy result dictionaries.
    category_ids: Ordered category IDs.
    category_name_by_id: Category names keyed by ID.
  Returns:
    pd.DataFrame: Wide per-class comparison table.
  """
  rows = []
  for category_id in category_ids:
    row = {
      'category_id': category_id,
      'class': category_name_by_id[category_id]
    }
    for result in results:
      metric_row = result['metrics_df'].set_index('category_id').loc[category_id]
      prefix = result['name']
      for column in (
        'threshold',
        'true_positive',
        'false_positive',
        'false_negative',
        'precision',
        'recall',
        'f1'
      ):
        row[f'{prefix}_{column}'] = metric_row[column]
    rows.append(row)
  return pd.DataFrame(rows)
def main():
  """Run raw inference, tune both NMS strategies, and write comparisons."""
  cli_args = parse_args()
  hard_grid = parse_float_grid(cli_args.hard_nms_grid, '--hard-nms-grid')
  soft_grid = parse_float_grid(
    cli_args.soft_nms_sigma_grid,
    '--soft-nms-sigma-grid'
  )
  if cli_args.baseline_nms_iou not in hard_grid:
    hard_grid.append(cli_args.baseline_nms_iou)
    hard_grid.sort()
  if any(item > 1 for item in hard_grid):
    raise ValueError('--hard-nms-grid values cannot exceed 1.')
  output_dir = Path(cli_args.output_dir).expanduser()
  output_dir.mkdir(parents = True, exist_ok = True)
  cache_path = output_dir / 'raw_model_outputs.pt'
  identity = {
    'checkpoint': file_identity(cli_args.checkpoint),
    'annotations': file_identity(cli_args.coco_val_ann),
    'images': str(Path(cli_args.coco_val_images).expanduser())
  }
  payload = (
    load_raw_cache(cache_path, identity)
    if cli_args.reuse_raw_cache
    else None
  )
  if payload is None:
    payload = collect_raw_outputs(
      cli_args,
      identity,
      cache_path
    )
  raw_records = payload['raw_records']
  category_ids = [int(item) for item in payload['category_ids']]
  category_name_by_id = {
    int(category_id): name
    for category_id, name in payload['category_name_by_id'].items()
  }
  class_count = len(category_ids)
  hard_cache = precompute_hard_candidates(
    raw_records,
    hard_grid,
    cli_args.min_confidence_threshold,
    cli_args.top_k
  )
  baseline_config = [cli_args.baseline_nms_iou] * class_count
  baseline_sources = hard_sources_for_config(
    hard_cache,
    baseline_config,
    class_count
  )
  baseline_records = build_image_records(
    raw_records,
    baseline_sources,
    category_ids,
    cli_args.top_k
  )
  baseline = evaluate_strategy(
    'hard_nms_0_40',
    baseline_records,
    category_ids,
    category_name_by_id,
    cli_args,
    output_dir / 'hard_nms_0_40'
  )
  selected_hard, hard_coordinate_df = select_per_class_hard_nms(
    raw_records,
    hard_cache,
    hard_grid,
    category_ids,
    category_name_by_id,
    cli_args
  )
  hard_sources = hard_sources_for_config(
    hard_cache,
    selected_hard,
    class_count
  )
  hard_records = build_image_records(
    raw_records,
    hard_sources,
    category_ids,
    cli_args.top_k
  )
  per_class_hard = evaluate_strategy(
    'per_class_hard_nms',
    hard_records,
    category_ids,
    category_name_by_id,
    cli_args,
    output_dir / 'per_class_hard_nms'
  )
  hard_coordinate_df.to_csv(
    output_dir / 'per_class_hard_nms_coordinate_search.csv',
    index = False
  )
  soft_cache = precompute_soft_candidates(
    raw_records,
    soft_grid,
    cli_args.min_confidence_threshold,
    cli_args.top_k
  )
  soft_results = []
  soft_grid_rows = []
  soft_class_rows = []
  for sigma in soft_grid:
    sources = soft_sources_for_sigma(soft_cache, sigma, class_count)
    records = build_image_records(
      raw_records,
      sources,
      category_ids,
      cli_args.top_k
    )
    result = evaluate_strategy(
      f'gaussian_soft_nms_sigma_{sigma:.2f}',
      records,
      category_ids,
      category_name_by_id,
      cli_args
    )
    result['sigma'] = sigma
    soft_results.append(result)
    soft_grid_rows.append({'sigma': sigma, **result['summary']})
    for _, row in result['metrics_df'].iterrows():
      soft_class_rows.append({'sigma': sigma, **row.to_dict()})
  best_soft = max(
    soft_results,
    key = lambda item: (
      item['summary']['micro_f1'],
      item['summary']['macro_f1'],
      -abs(item['sigma'] - 0.50)
    )
  )
  best_soft = evaluate_strategy(
    'gaussian_soft_nms',
    best_soft['image_records'],
    category_ids,
    category_name_by_id,
    cli_args,
    output_dir / 'gaussian_soft_nms'
  ) | {'sigma': best_soft['sigma']}
  pd.DataFrame(soft_grid_rows).to_csv(
    output_dir / 'gaussian_soft_nms_sigma_search.csv',
    index = False
  )
  pd.DataFrame(soft_class_rows).to_csv(
    output_dir / 'gaussian_soft_nms_sigma_class_metrics.csv',
    index = False
  )
  results = [baseline, per_class_hard, best_soft]
  comparison_rows = []
  baseline_micro_f1 = baseline['summary']['micro_f1']
  baseline_macro_f1 = baseline['summary']['macro_f1']
  for result in results:
    comparison_rows.append({
      'strategy': result['name'],
      **result['summary'],
      'delta_micro_f1_vs_baseline': (
        result['summary']['micro_f1'] - baseline_micro_f1
      ),
      'delta_macro_f1_vs_baseline': (
        result['summary']['macro_f1'] - baseline_macro_f1
      )
    })
  comparison_df = pd.DataFrame(comparison_rows)
  comparison_df.to_csv(output_dir / 'strategy_comparison.csv', index = False)
  class_comparison(
    results,
    category_ids,
    category_name_by_id
  ).to_csv(output_dir / 'class_strategy_comparison.csv', index = False)
  parameter_rows = []
  for class_index, category_id in enumerate(category_ids):
    parameter_rows.append({
      'category_id': category_id,
      'class': category_name_by_id[category_id],
      'baseline_hard_nms_iou': cli_args.baseline_nms_iou,
      'baseline_confidence': baseline['thresholds'][category_id],
      'per_class_hard_nms_iou': selected_hard[class_index],
      'per_class_hard_confidence': per_class_hard['thresholds'][category_id],
      'gaussian_soft_nms_sigma': best_soft['sigma'],
      'gaussian_soft_nms_confidence': best_soft['thresholds'][category_id]
    })
  pd.DataFrame(parameter_rows).to_csv(
    output_dir / 'selected_parameters.csv',
    index = False
  )
  report = {
    'inputs': identity,
    'settings': {
      'evaluation_iou': cli_args.iou_threshold,
      'top_k': cli_args.top_k,
      'confidence_grid': {
        'minimum': cli_args.min_confidence_threshold,
        'maximum': cli_args.max_confidence_threshold,
        'step': cli_args.confidence_threshold_step
      },
      'hard_nms_grid': hard_grid,
      'soft_nms_sigma_grid': soft_grid,
      'gaussian_formula': 'score * exp(-(iou ** 2) / sigma)',
      'coordinate_pass_limit': cli_args.coordinate_passes
    },
    'selected_per_class_hard_nms': {
      category_name_by_id[category_id]: selected_hard[class_index]
      for class_index, category_id in enumerate(category_ids)
    },
    'selected_gaussian_soft_nms_sigma': best_soft['sigma'],
    'comparison': comparison_rows
  }
  (output_dir / 'analysis_report.json').write_text(
    json.dumps(report, indent = 2),
    encoding = 'utf-8'
  )
  print('', flush = True)
  print('strategy comparison:', flush = True)
  print(comparison_df.to_string(index = False), flush = True)
  print('', flush = True)
  print(f'outputs written to: {output_dir}', flush = True)
if __name__ == '__main__':
  main()
