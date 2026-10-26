#!/usr/bin/env python3
"""Evaluate higher-resolution and tiled inference for the damage detector."""
import argparse
import copy
import gc
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as vision_functional
REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
  sys.path.insert(0, str(REPO_DIR))
from models import build_model
from util import box_ops
import util.misc as utils
from datasets.transforms import resize as deta_resize
from tools.eval_nms_strategies import build_image_records
from tools.eval_nms_strategies import evaluate_strategy
from tools.eval_nms_strategies import hard_sources_for_config
from tools.eval_nms_strategies import parse_float_grid
from tools.eval_nms_strategies import precompute_hard_candidates
from tools.eval_nms_strategies import select_per_class_hard_nms
#
# Tile inference is restricted to localized damage classes because large region
# labels need full-vehicle context and become ambiguous when clipped by a tile.
#
DEFAULT_TILE_CLASSES = [
  'Surface_abrasion',
  'Nick',
  'Deformation',
  'Crack',
  'Gap',
  'Hole'
]
def parse_args():
  """Parse scale, tiling, model, and threshold-search options.
  Returns:
    argparse.Namespace: Evaluation configuration.
  """
  parser = argparse.ArgumentParser(
    description = (
      'Compare higher-resolution full-frame inference and localized-class '
      'tile fusion against the established v4 baseline.'
    )
  )
  parser.add_argument('--checkpoint', required = True)
  parser.add_argument('--baseline-raw-cache', required = True)
  parser.add_argument('--baseline-results-dir', required = True)
  parser.add_argument('--output-dir', required = True)
  parser.add_argument('--image-root', required = True)
  parser.add_argument('--device', default = 'cuda')
  parser.add_argument('--view-batch-size', default = 1, type = int)
  parser.add_argument('--full-scales', default = '1024,1200')
  parser.add_argument('--tile-short-side', default = 800, type = int)
  parser.add_argument('--tile-overlap', default = 0.20, type = float)
  parser.add_argument('--minimum-tile-side', default = 256, type = int)
  parser.add_argument('--tile-classes', nargs = '*', default = DEFAULT_TILE_CLASSES)
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
  parser.add_argument('--coordinate-passes', default = 3, type = int)
  parser.add_argument('--default-confidence-threshold', default = 0.40, type = float)
  parser.add_argument('--min-confidence-threshold', default = 0.05, type = float)
  parser.add_argument('--max-confidence-threshold', default = 0.95, type = float)
  parser.add_argument('--confidence-threshold-step', default = 0.01, type = float)
  parser.add_argument('--threshold-optimization-metric', default = 'f1')
  parser.add_argument(
    '--reuse-variant-caches',
    action = argparse.BooleanOptionalAction,
    default = True
  )
  return parser.parse_args()
def parse_integer_grid(value, option_name):
  """Parse positive comma-separated integer values.
  Args:
    value: Comma-separated integer text.
    option_name: Option name used in validation errors.
  Returns:
    list[int]: Sorted unique positive values.
  Raises:
    ValueError: If a value is not a positive integer.
  """
  try:
    values = sorted({int(item.strip()) for item in value.split(',')})
  except ValueError as error:
    raise ValueError(f'{option_name} must contain integers.') from error
  if not values or any(item <= 0 for item in values):
    raise ValueError(f'{option_name} values must be positive.')
  return values
def configure_model_args(checkpoint_args, cli_args):
  """Create inference-only checkpoint arguments.
  Args:
    checkpoint_args: Training arguments serialized in the checkpoint.
    cli_args: Current runtime configuration.
  Returns:
    argparse.Namespace: Detached model-building arguments.
  """
  model_args = copy.deepcopy(checkpoint_args)
  model_args.device = cli_args.device
  model_args.eval = True
  model_args.distributed = False
  return model_args
def load_model(cli_args):
  """Load the trained model without its fixed postprocessor.
  Args:
    cli_args: Model checkpoint and device configuration.
  Returns:
    torch.nn.Module: Evaluation-mode model on the requested device.
  """
  checkpoint = torch.load(
    Path(cli_args.checkpoint).expanduser(),
    map_location = 'cpu',
    weights_only = False
  )
  if 'args' not in checkpoint:
    raise RuntimeError('Checkpoint does not contain checkpoint["args"].')
  model_args = configure_model_args(checkpoint['args'], cli_args)
  model, _, _ = build_model(model_args)
  model.load_state_dict(checkpoint['model'], strict = True)
  model.to(torch.device(cli_args.device))
  model.eval()
  return model
def scaled_maximum_side(short_side):
  """Preserve DETA's 800-to-1333 maximum-side ratio at larger scales.
  Args:
    short_side: Requested resized short side.
  Returns:
    int: Proportional long-side cap.
  """
  return int(round(short_side * 1333.0 / 800.0))
def preprocess_image(image, short_side):
  """Resize and normalize a PIL view using DETA validation semantics.
  Args:
    image: RGB PIL image or tile.
    short_side: Requested short-side resolution.
  Returns:
    torch.Tensor: Normalized CHW image tensor.
  """
  resized, _ = deta_resize(
    image,
    None,
    short_side,
    scaled_maximum_side(short_side)
  )
  tensor = vision_functional.to_tensor(resized)
  return vision_functional.normalize(
    tensor,
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
  )
def create_grid_regions(width, height, grid_size, overlap):
  """Create evenly spaced overlapping tile rectangles.
  Args:
    width: Original image width.
    height: Original image height.
    grid_size: Number of rows and columns.
    overlap: Fractional overlap relative to one tile dimension.
  Returns:
    list[tuple[int, int, int, int]]: XYXY integer crop regions.
  """
  denominator = grid_size - (grid_size - 1) * overlap
  tile_width = min(width, int(math.ceil(width / denominator)))
  tile_height = min(height, int(math.ceil(height / denominator)))
  x_starts = np.linspace(0, width - tile_width, grid_size).round().astype(int)
  y_starts = np.linspace(0, height - tile_height, grid_size).round().astype(int)
  return [
    (int(x), int(y), int(x + tile_width), int(y + tile_height))
    for y in y_starts
    for x in x_starts
  ]
def select_tile_regions(width, height, mode, overlap, minimum_tile_side):
  """Select fixed 2x2 or size-gated adaptive tile regions.
  Args:
    width: Original image width.
    height: Original image height.
    mode: ``two_by_two`` or ``adaptive_three_by_three``.
    overlap: Fractional adjacent-tile overlap.
    minimum_tile_side: Smallest allowed native tile dimension.
  Returns:
    tuple[list[tuple], int]: Crop regions and selected grid size.
  """
  candidate_grids = [2] if mode == 'two_by_two' else [3, 2]
  for grid_size in candidate_grids:
    regions = create_grid_regions(width, height, grid_size, overlap)
    first = regions[0]
    tile_width = first[2] - first[0]
    tile_height = first[3] - first[1]
    if min(tile_width, tile_height) >= minimum_tile_side:
      return regions, grid_size
  return [], 1
def resolve_image_path(image_root, file_name):
  """Resolve a COCO filename beneath the configured image cache.
  Args:
    image_root: Root containing validation images.
    file_name: COCO image filename, potentially with subdirectories.
  Returns:
    Path: Existing image path.
  Raises:
    FileNotFoundError: If the image cannot be located.
  """
  candidate = image_root / file_name
  if candidate.is_file():
    return candidate
  flat_candidate = image_root / Path(file_name).name
  if flat_candidate.is_file():
    return flat_candidate
  raise FileNotFoundError(f'Unable to locate image: {file_name}')
def infer_views(model, device, image, regions, short_side, batch_size):
  """Run model inference for image regions and map boxes to original space.
  Args:
    model: Evaluation-mode detector.
    device: Torch inference device.
    image: Full-resolution RGB PIL image.
    regions: Original-coordinate XYXY view rectangles.
    short_side: DETA resize target for every view.
    batch_size: Number of views processed together.
  Returns:
    tuple[torch.Tensor, torch.Tensor]: Concatenated original-space boxes and scores.
  """
  output_boxes = []
  output_scores = []
  for start in range(0, len(regions), batch_size):
    batch_regions = regions[start:start + batch_size]
    tensors = []
    view_sizes = []
    for left, top, right, bottom in batch_regions:
      view = image.crop((left, top, right, bottom))
      tensors.append(preprocess_image(view, short_side))
      view_sizes.append((right - left, bottom - top, left, top))
    nested = utils.nested_tensor_from_tensor_list(tensors).to(device)
    with torch.inference_mode():
      outputs = model(nested)
    normalized_boxes = box_ops.box_cxcywh_to_xyxy(
      outputs['pred_boxes']
    ).detach().cpu()
    score_matrices = outputs['pred_logits'].sigmoid().detach().cpu()
    for boxes, scores, view_size in zip(
      normalized_boxes,
      score_matrices,
      view_sizes
    ):
      view_width, view_height, left, top = view_size
      scale = torch.tensor(
        [view_width, view_height, view_width, view_height],
        dtype = boxes.dtype
      )
      mapped = boxes * scale
      mapped[:, [0, 2]] += left
      mapped[:, [1, 3]] += top
      mapped[:, [0, 2]].clamp_(0, image.width)
      mapped[:, [1, 3]].clamp_(0, image.height)
      output_boxes.append(mapped)
      output_scores.append(scores)
  return torch.cat(output_boxes, dim = 0), torch.cat(output_scores, dim = 0)
def base_record_copy(raw_record):
  """Copy annotation metadata shared by every inference variant.
  Args:
    raw_record: Baseline raw prediction and annotation record.
  Returns:
    dict: Annotation metadata without prediction tensors.
  """
  return {
    'image_id': raw_record['image_id'],
    'file_name': raw_record['file_name'],
    'gt_boxes': raw_record['gt_boxes'],
    'gt_category_ids': raw_record['gt_category_ids']
  }
def collect_full_scale_records(
  model,
  device,
  baseline_records,
  image_root,
  short_side,
  batch_size
):
  """Collect raw predictions at one full-frame inference resolution.
  Args:
    model: Evaluation-mode detector.
    device: Torch inference device.
    baseline_records: Records supplying filenames and annotations.
    image_root: Root containing original validation images.
    short_side: Full-frame resize target.
    batch_size: View batch size.
  Returns:
    tuple[list[dict], dict]: Raw records and processing statistics.
  """
  records = []
  for index, baseline_record in enumerate(baseline_records):
    image_path = resolve_image_path(image_root, baseline_record['file_name'])
    with Image.open(image_path) as source_image:
      image = source_image.convert('RGB')
      region = [(0, 0, image.width, image.height)]
      boxes, scores = infer_views(
        model,
        device,
        image,
        region,
        short_side,
        batch_size
      )
    records.append({
      **base_record_copy(baseline_record),
      'boxes': boxes,
      'scores': scores
    })
    if (index + 1) % 50 == 0:
      print(
        f'full {short_side} inference images: {index + 1}',
        flush = True
      )
  return records, {
    'images': len(records),
    'full_views': len(records),
    'tile_views': 0,
    'grid_counts': {'1': len(records)}
  }
def collect_tile_fusion_records(
  model,
  device,
  baseline_records,
  image_root,
  mode,
  tile_short_side,
  overlap,
  minimum_tile_side,
  tile_class_indices,
  batch_size
):
  """Fuse baseline full-frame raw predictions with localized-class tiles.
  Args:
    model: Evaluation-mode detector.
    device: Torch inference device.
    baseline_records: Full-frame 800 raw predictions and annotations.
    image_root: Root containing original validation images.
    mode: Fixed or adaptive tiling mode.
    tile_short_side: Resize target for each tile.
    overlap: Fractional adjacent-tile overlap.
    minimum_tile_side: Smallest native tile dimension.
    tile_class_indices: Raw class indices retained from tile predictions.
    batch_size: Number of tile views processed together.
  Returns:
    tuple[list[dict], dict]: Fused raw records and tile statistics.
  """
  records = []
  tile_view_count = 0
  tiled_image_count = 0
  grid_counts = {'1': 0, '2': 0, '3': 0}
  for index, baseline_record in enumerate(baseline_records):
    image_path = resolve_image_path(image_root, baseline_record['file_name'])
    with Image.open(image_path) as source_image:
      image = source_image.convert('RGB')
      regions, grid_size = select_tile_regions(
        image.width,
        image.height,
        mode,
        overlap,
        minimum_tile_side
      )
      if regions:
        tile_boxes, tile_scores = infer_views(
          model,
          device,
          image,
          regions,
          tile_short_side,
          batch_size
        )
        class_mask = torch.zeros_like(tile_scores)
        class_mask[:, tile_class_indices] = 1.0
        tile_scores *= class_mask
        boxes = torch.cat([baseline_record['boxes'], tile_boxes], dim = 0)
        scores = torch.cat([baseline_record['scores'], tile_scores], dim = 0)
        tile_view_count += len(regions)
        tiled_image_count += 1
      else:
        boxes = baseline_record['boxes']
        scores = baseline_record['scores']
      grid_counts[str(grid_size)] += 1
    records.append({
      **base_record_copy(baseline_record),
      'boxes': boxes.contiguous(),
      'scores': scores.contiguous()
    })
    if (index + 1) % 25 == 0:
      print(
        f'{mode} fusion images: {index + 1}',
        flush = True
      )
  return records, {
    'images': len(records),
    'full_views': len(records),
    'tiled_images': tiled_image_count,
    'tile_views': tile_view_count,
    'grid_counts': grid_counts
  }
def load_or_collect_variant(
  variant_dir,
  collector,
  reuse_cache
):
  """Load cached raw predictions or execute one variant collector.
  Args:
    variant_dir: Variant artifact directory.
    collector: Zero-argument function returning records and statistics.
    reuse_cache: Whether an existing cache may be reused.
  Returns:
    tuple[list[dict], dict]: Raw records and processing statistics.
  """
  variant_dir.mkdir(parents = True, exist_ok = True)
  cache_path = variant_dir / 'raw_model_outputs.pt'
  if reuse_cache and cache_path.is_file():
    payload = torch.load(
      cache_path,
      map_location = 'cpu',
      weights_only = False
    )
    print(f'loaded variant cache: {cache_path}', flush = True)
    return payload['raw_records'], payload['statistics']
  raw_records, statistics = collector()
  torch.save(
    {'raw_records': raw_records, 'statistics': statistics},
    cache_path
  )
  print(f'wrote variant cache: {cache_path}', flush = True)
  return raw_records, statistics
def evaluate_raw_variant(
  variant_name,
  raw_records,
  category_ids,
  category_name_by_id,
  hard_grid,
  cli_args,
  variant_dir
):
  """Jointly tune per-class hard NMS and confidence for one variant.
  Args:
    variant_name: Comparison identifier.
    raw_records: Unsuppressed variant predictions.
    category_ids: COCO category IDs in raw-logit order.
    category_name_by_id: Category names keyed by ID.
    hard_grid: Candidate per-class NMS IoU thresholds.
    cli_args: Threshold search and evaluation configuration.
    variant_dir: Destination for final metrics and search artifacts.
  Returns:
    dict: Evaluated strategy result with selected NMS thresholds.
  """
  print(f'preparing NMS search: {variant_name}', flush = True)
  hard_cache = precompute_hard_candidates(
    raw_records,
    hard_grid,
    cli_args.min_confidence_threshold,
    cli_args.top_k
  )
  selected_nms, coordinate_df = select_per_class_hard_nms(
    raw_records,
    hard_cache,
    hard_grid,
    category_ids,
    category_name_by_id,
    cli_args
  )
  sources = hard_sources_for_config(
    hard_cache,
    selected_nms,
    len(category_ids)
  )
  image_records = build_image_records(
    raw_records,
    sources,
    category_ids,
    cli_args.top_k
  )
  final_dir = variant_dir / 'optimized'
  result = evaluate_strategy(
    variant_name,
    image_records,
    category_ids,
    category_name_by_id,
    cli_args,
    final_dir
  )
  coordinate_df.to_csv(
    variant_dir / 'per_class_nms_coordinate_search.csv',
    index = False
  )
  parameter_rows = [
    {
      'category_id': category_id,
      'class': category_name_by_id[category_id],
      'nms_iou': selected_nms[class_index],
      'confidence': result['thresholds'][category_id]
    }
    for class_index, category_id in enumerate(category_ids)
  ]
  pd.DataFrame(parameter_rows).to_csv(
    variant_dir / 'selected_parameters.csv',
    index = False
  )
  result['selected_nms'] = selected_nms
  del hard_cache
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return result
def load_baseline_result(baseline_results_dir):
  """Load the established per-class-hard-NMS baseline artifacts.
  Args:
    baseline_results_dir: Prior NMS strategy comparison directory.
  Returns:
    tuple[dict, pd.DataFrame]: Aggregate summary and class metric table.
  """
  comparison = pd.read_csv(
    baseline_results_dir / 'strategy_comparison.csv'
  )
  row = comparison.loc[
    comparison['strategy'] == 'per_class_hard_nms'
  ]
  if len(row) != 1:
    raise RuntimeError('Unable to locate per_class_hard_nms baseline row.')
  summary = row.iloc[0].to_dict()
  metrics = pd.read_csv(
    baseline_results_dir /
    'per_class_hard_nms' /
    'class_metrics.csv'
  )
  return summary, metrics
def create_class_comparison(
  baseline_metrics,
  variant_results,
  category_ids,
  category_name_by_id
):
  """Create a wide per-class comparison against the frozen baseline.
  Args:
    baseline_metrics: Established baseline class metrics.
    variant_results: Evaluated scale/tile strategy results.
    category_ids: Ordered category IDs.
    category_name_by_id: Category names keyed by ID.
  Returns:
    pd.DataFrame: Per-class F1, precision, recall, and count comparison.
  """
  metric_columns = [
    'threshold',
    'true_positive',
    'false_positive',
    'false_negative',
    'precision',
    'recall',
    'f1'
  ]
  baseline_by_id = baseline_metrics.set_index('category_id')
  rows = []
  for category_id in category_ids:
    row = {
      'category_id': category_id,
      'class': category_name_by_id[category_id]
    }
    for column in metric_columns:
      row[f'baseline_{column}'] = baseline_by_id.loc[category_id, column]
    for result in variant_results:
      result_by_id = result['metrics_df'].set_index('category_id')
      for column in metric_columns:
        row[f'{result["name"]}_{column}'] = result_by_id.loc[
          category_id,
          column
        ]
    rows.append(row)
  return pd.DataFrame(rows)
def main():
  """Collect scale/tile predictions, optimize postprocessing, and compare."""
  cli_args = parse_args()
  if not 0 <= cli_args.tile_overlap < 1:
    raise ValueError('--tile-overlap must be in the range [0, 1).')
  if cli_args.view_batch_size <= 0:
    raise ValueError('--view-batch-size must be positive.')
  full_scales = parse_integer_grid(cli_args.full_scales, '--full-scales')
  hard_grid = parse_float_grid(cli_args.hard_nms_grid, '--hard-nms-grid')
  if cli_args.baseline_nms_iou not in hard_grid:
    hard_grid.append(cli_args.baseline_nms_iou)
    hard_grid.sort()
  output_dir = Path(cli_args.output_dir).expanduser()
  output_dir.mkdir(parents = True, exist_ok = True)
  image_root = Path(cli_args.image_root).expanduser()
  baseline_payload = torch.load(
    Path(cli_args.baseline_raw_cache).expanduser(),
    map_location = 'cpu',
    weights_only = False
  )
  baseline_records = baseline_payload['raw_records']
  category_ids = [int(item) for item in baseline_payload['category_ids']]
  category_name_by_id = {
    int(category_id): class_name
    for category_id, class_name in baseline_payload[
      'category_name_by_id'
    ].items()
  }
  category_id_by_name = {
    class_name: category_id
    for category_id, class_name in category_name_by_id.items()
  }
  unknown_tile_classes = sorted(
    set(cli_args.tile_classes) - set(category_id_by_name)
  )
  if unknown_tile_classes:
    raise ValueError(f'Unknown tile classes: {unknown_tile_classes}')
  tile_class_indices = [
    category_ids.index(category_id_by_name[class_name])
    for class_name in cli_args.tile_classes
  ]
  device = torch.device(cli_args.device)
  model = load_model(cli_args)
  variants = []
  variant_statistics = {}
  for short_side in full_scales:
    variant_name = f'full_{short_side}'
    variant_dir = output_dir / variant_name
    raw_records, statistics = load_or_collect_variant(
      variant_dir,
      lambda short_side = short_side: collect_full_scale_records(
        model,
        device,
        baseline_records,
        image_root,
        short_side,
        cli_args.view_batch_size
      ),
      cli_args.reuse_variant_caches
    )
    result = evaluate_raw_variant(
      variant_name,
      raw_records,
      category_ids,
      category_name_by_id,
      hard_grid,
      cli_args,
      variant_dir
    )
    variants.append(result)
    variant_statistics[variant_name] = statistics
  tile_modes = [
    ('full_800_plus_2x2', 'two_by_two'),
    ('full_800_plus_adaptive_3x3', 'adaptive_three_by_three')
  ]
  for variant_name, mode in tile_modes:
    variant_dir = output_dir / variant_name
    raw_records, statistics = load_or_collect_variant(
      variant_dir,
      lambda mode = mode: collect_tile_fusion_records(
        model,
        device,
        baseline_records,
        image_root,
        mode,
        cli_args.tile_short_side,
        cli_args.tile_overlap,
        cli_args.minimum_tile_side,
        tile_class_indices,
        cli_args.view_batch_size
      ),
      cli_args.reuse_variant_caches
    )
    result = evaluate_raw_variant(
      variant_name,
      raw_records,
      category_ids,
      category_name_by_id,
      hard_grid,
      cli_args,
      variant_dir
    )
    variants.append(result)
    variant_statistics[variant_name] = statistics
  baseline_dir = Path(cli_args.baseline_results_dir).expanduser()
  baseline_summary, baseline_metrics = load_baseline_result(baseline_dir)
  baseline_micro_f1 = float(baseline_summary['micro_f1'])
  baseline_macro_f1 = float(baseline_summary['macro_f1'])
  comparison_rows = [{
    'strategy': 'v4_per_class_nms_baseline',
    **{
      key: baseline_summary[key]
      for key in (
        'ground_truth',
        'predicted',
        'true_positive',
        'false_positive',
        'false_negative',
        'precision',
        'recall',
        'micro_f1',
        'macro_f1'
      )
    },
    'delta_micro_f1_vs_baseline': 0.0,
    'delta_macro_f1_vs_baseline': 0.0
  }]
  for result in variants:
    comparison_rows.append({
      'strategy': result['name'],
      **{
        key: result['summary'][key]
        for key in (
          'ground_truth',
          'predicted',
          'true_positive',
          'false_positive',
          'false_negative',
          'precision',
          'recall',
          'micro_f1',
          'macro_f1'
        )
      },
      'delta_micro_f1_vs_baseline': (
        result['summary']['micro_f1'] - baseline_micro_f1
      ),
      'delta_macro_f1_vs_baseline': (
        result['summary']['macro_f1'] - baseline_macro_f1
      )
    })
  comparison_df = pd.DataFrame(comparison_rows)
  comparison_df.to_csv(output_dir / 'strategy_comparison.csv', index = False)
  create_class_comparison(
    baseline_metrics,
    variants,
    category_ids,
    category_name_by_id
  ).to_csv(output_dir / 'class_strategy_comparison.csv', index = False)
  report = {
    'settings': {
      'full_scales': full_scales,
      'tile_short_side': cli_args.tile_short_side,
      'tile_overlap': cli_args.tile_overlap,
      'minimum_tile_side': cli_args.minimum_tile_side,
      'tile_classes': cli_args.tile_classes,
      'evaluation_iou': cli_args.iou_threshold,
      'top_k': cli_args.top_k,
      'hard_nms_grid': hard_grid,
      'confidence_grid': {
        'minimum': cli_args.min_confidence_threshold,
        'maximum': cli_args.max_confidence_threshold,
        'step': cli_args.confidence_threshold_step
      }
    },
    'variant_statistics': variant_statistics,
    'comparison': comparison_rows
  }
  (output_dir / 'analysis_report.json').write_text(
    json.dumps(report, indent = 2),
    encoding = 'utf-8'
  )
  print('', flush = True)
  print('scale and tile comparison:', flush = True)
  print(comparison_df.to_string(index = False), flush = True)
  print('', flush = True)
  print(f'outputs written to: {output_dir}', flush = True)
if __name__ == '__main__':
  main()
