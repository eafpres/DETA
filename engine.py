# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified by EAF LLC for custom training
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import numpy as np
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from util import box_ops
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

def log_cuda_memory(prefix):
  """Print current CUDA memory usage."""
  if not torch.cuda.is_available():
    return
  allocated_gb = torch.cuda.memory_allocated() / 1024 ** 3
  reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
  max_allocated_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
  print(
    f"{prefix} "
    f"cuda_allocated_gb={allocated_gb:.3f} "
    f"cuda_reserved_gb={reserved_gb:.3f} "
    f"cuda_max_allocated_gb={max_allocated_gb:.3f}"
  )

def summarize_training_batch(targets):
  """Create a compact summary of the current training batch.

  Args:
    targets: Batch target dictionaries.

  Returns:
    str: Human-readable batch summary.
  """
  parts = []
  for target in targets:
    image_id = target.get("image_id", None)
    orig_size = target.get("orig_size", None)
    labels = target.get("labels", None)
    boxes = target.get("boxes", None)
    if image_id is not None:
      image_id = int(image_id.detach().cpu().item())
    if orig_size is not None:
      orig_size = orig_size.detach().cpu().tolist()
    num_labels = 0 if labels is None else int(labels.numel())
    num_boxes = 0 if boxes is None else int(boxes.shape[0])
    parts.append(
      f"image_id={image_id} "
      f"orig_size={orig_size} "
      f"labels={num_labels} "
      f"boxes={num_boxes}"
    )
  return " | ".join(parts)

def build_detection_f1_image_records(results, targets):
  """Build per-image detection records for threshold/F1 evaluation.

  Args:
    results: Postprocessed predictions for one validation batch.
    targets: Ground-truth targets for one validation batch.

  Returns:
    dict[int, dict]: Per-image prediction and ground-truth records.
  """
  image_records = {}

  for result, target in zip(results, targets):
    image_id = int(target["image_id"].item())

    pred_scores = result["scores"].detach().cpu().to(torch.float32)
    pred_labels = result["labels"].detach().cpu().to(torch.int64)
    pred_boxes = result["boxes"].detach().cpu().to(torch.float32)

    gt_labels = target["labels"].detach().cpu().to(torch.int64)
    gt_boxes = box_ops.box_cxcywh_to_xyxy(
      target["boxes"].detach().cpu().to(torch.float32)
    )

    orig_height, orig_width = target["orig_size"].detach().cpu()
    scale = torch.tensor(
      [
        float(orig_width),
        float(orig_height),
        float(orig_width),
        float(orig_height)
      ],
      dtype = gt_boxes.dtype
    )
    gt_boxes = gt_boxes * scale

    image_records[image_id] = {
      "gt_boxes": gt_boxes,
      "gt_labels": gt_labels,
      "pred_boxes": pred_boxes,
      "pred_labels": pred_labels,
      "pred_scores": pred_scores,
      "total_gt": int(gt_boxes.shape[0])
    }

  return image_records

def calculate_detection_prf(tp, fp, fn):
  """Calculate precision, recall, and F1."""
  precision = tp / (tp + fp) if tp + fp > 0 else 0.0
  recall = tp / (tp + fn) if tp + fn > 0 else 0.0
  f1 = (
    2.0 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0.0
  )
  return precision, recall, f1

def normalize_detection_boxes(boxes):
  if boxes is None:
    return torch.empty((0, 4), dtype = torch.float32)
  if isinstance(boxes, torch.Tensor):
    boxes = boxes.detach().cpu().float()
    if boxes.numel() == 0:
      return torch.empty((0, 4), dtype = torch.float32)
    return boxes.reshape(-1, 4)
  if isinstance(boxes, np.ndarray):
    boxes = torch.from_numpy(boxes).float()
    if boxes.numel() == 0:
      return torch.empty((0, 4), dtype = torch.float32)
    return boxes.reshape(-1, 4)
  if isinstance(boxes, (list, tuple)):
    if len(boxes) == 0:
      return torch.empty((0, 4), dtype = torch.float32)
    tensor_boxes = []
    for box in boxes:
      if isinstance(box, torch.Tensor):
        box = box.detach().cpu().float().reshape(-1, 4)
      elif isinstance(box, np.ndarray):
        box = torch.from_numpy(box).float().reshape(-1, 4)
      else:
        box = torch.tensor(box, dtype = torch.float32).reshape(-1, 4)
      tensor_boxes.append(box)
    return torch.cat(tensor_boxes, dim = 0)
  boxes = torch.tensor(boxes, dtype = torch.float32)
  if boxes.numel() == 0:
    return torch.empty((0, 4), dtype = torch.float32)
  return boxes.reshape(-1, 4)

def greedy_detection_matches(
  gt_boxes,
  gt_labels,
  pred_boxes,
  pred_labels,
  iou_threshold
):
  """Greedily match predictions to ground truth boxes by IoU and class."""
  if len(gt_boxes) == 0 or len(pred_boxes) == 0:
    return [], set(), set()

  gt_boxes = normalize_detection_boxes(gt_boxes)
  pred_boxes = normalize_detection_boxes(pred_boxes)
  gt_labels = torch.as_tensor(gt_labels, dtype = torch.int64)
  pred_labels = torch.as_tensor(pred_labels, dtype = torch.int64)

  ious, _ = box_ops.box_iou(gt_boxes, pred_boxes)
  candidates = []

  for gt_idx in range(ious.shape[0]):
    for pred_idx in range(ious.shape[1]):
      if int(gt_labels[gt_idx].item()) != int(pred_labels[pred_idx].item()):
        continue
      iou = float(ious[gt_idx, pred_idx].item())
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

def count_detection_f1_for_thresholds(
  image_records,
  threshold_by_category_id,
  default_threshold,
  iou_threshold,
  category_id = None
):
  """Count TP, FP, FN using class-specific confidence thresholds."""
  total_tp = 0
  total_fp = 0
  total_fn = 0

  for image_record in image_records.values():
    gt_boxes = image_record["gt_boxes"]
    gt_labels = image_record["gt_labels"]

    pred_boxes = []
    pred_labels = []
    pred_scores = []

    for box, label, score in zip(
      image_record["pred_boxes"],
      image_record["pred_labels"],
      image_record["pred_scores"]
    ):
      label = int(label)
      score = float(score)

      if category_id is not None and label != category_id:
        continue

      threshold = threshold_by_category_id.get(
        label,
        default_threshold
      )
      if score < threshold:
        continue

      pred_boxes.append(box)
      pred_labels.append(label)
      pred_scores.append(score)

    if category_id is not None:
      filtered_gt_boxes = []
      filtered_gt_labels = []
      for box, label in zip(gt_boxes, gt_labels):
        if int(label) == category_id:
          filtered_gt_boxes.append(box)
          filtered_gt_labels.append(label)
      gt_boxes = filtered_gt_boxes
      gt_labels = filtered_gt_labels

    matches, matched_gt, matched_pred = greedy_detection_matches(
      gt_boxes,
      gt_labels,
      pred_boxes,
      pred_labels,
      iou_threshold
    )

    total_tp += len(matches)
    total_fp += len(pred_boxes) - len(matched_pred)
    total_fn += len(gt_boxes) - len(matched_gt)

  return total_tp, total_fp, total_fn

def summarize_best_detection_f1_per_class(
  image_records,
  valid_category_ids = None,
  iou_threshold = 0.5,
  threshold_min = 0.05,
  threshold_max = 0.95,
  threshold_step = 0.01
):
  """Optimize one confidence threshold per class and summarize macro F1.

  Args:
    image_records: Validation records created by build_detection_f1_image_records.
    threshold_min: Minimum confidence threshold.
    threshold_max: Maximum confidence threshold.
    threshold_step: Threshold step size.

  Returns:
    dict: Per-class threshold metrics, macro metrics, and micro counts.
  """
  gathered_image_records = utils.all_gather(image_records)
  merged_image_records = {}
  for rank_image_records in gathered_image_records:
    merged_image_records.update(rank_image_records)
  image_records = merged_image_records
  gt_category_ids = sorted({
    int(label)
    for image_record in image_records.values()
    for label in image_record.get("gt_labels", [])
  })

  pred_category_ids = sorted({
    int(label)
    for image_record in image_records.values()
    for label in image_record.get("pred_labels", [])
  })

  if valid_category_ids is None:
    category_ids = gt_category_ids
  else:
    category_ids = sorted([
      int(category_id)
      for category_id in valid_category_ids
    ])

  extra_pred_category_ids = [
    category_id
    for category_id in pred_category_ids
    if category_id not in category_ids
  ]

  thresholds = np.arange(
    threshold_min,
    threshold_max + 0.5 * threshold_step,
    threshold_step
  )
  threshold_by_category_id = {}
  class_metrics = {}

  for category_id in category_ids:
    best_metrics = None
    best_threshold = None

    for threshold in thresholds:
      tp, fp, fn = count_detection_f1_for_thresholds(
        image_records = image_records,
        threshold_by_category_id = {category_id: float(threshold)},
        default_threshold = float("inf"),
        iou_threshold = iou_threshold,
        category_id = category_id
      )
      precision, recall, f1 = calculate_detection_prf(tp, fp, fn)

      rank = (
        f1,
        precision,
        recall,
        -abs(float(threshold) - 0.5)
      )
      if best_metrics is None or rank > best_metrics["rank"]:
        best_metrics = {
          "rank": rank,
          "tp": tp,
          "fp": fp,
          "fn": fn,
          "precision": precision,
          "recall": recall,
          "f1": f1
        }
        best_threshold = float(threshold)

    threshold_by_category_id[category_id] = best_threshold
    class_metrics[category_id] = {
      key: value
      for key, value in best_metrics.items()
      if key != "rank"
    }
    class_metrics[category_id]["threshold"] = best_threshold

  tp, fp, fn = count_detection_f1_for_thresholds(
    image_records = image_records,
    threshold_by_category_id = threshold_by_category_id,
    default_threshold = float("inf"),
    iou_threshold = iou_threshold,
    category_id = None
  )
  micro_precision, micro_recall, micro_f1 = calculate_detection_prf(
    tp,
    fp,
    fn
  )

  if class_metrics:
    precision = float(np.mean([
      metrics["precision"]
      for metrics in class_metrics.values()
    ]))
    recall = float(np.mean([
      metrics["recall"]
      for metrics in class_metrics.values()
    ]))
    f1 = float(np.mean([
      metrics["f1"]
      for metrics in class_metrics.values()
    ]))
  else:
    precision = 0.0
    recall = 0.0
    f1 = 0.0

  return {
    "threshold_by_category_id": threshold_by_category_id,
    "class_metrics": class_metrics,
    "gt_category_ids": gt_category_ids,
    "pred_category_ids": pred_category_ids,
    "extra_pred_category_ids": extra_pred_category_ids,
    "num_gt_categories": len(gt_category_ids),
    "num_pred_categories": len(pred_category_ids),
    "num_valid_categories": len(category_ids),
    "num_extra_pred_categories": len(extra_pred_category_ids),
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "micro_precision": micro_precision,
    "micro_recall": micro_recall,
    "micro_f1": micro_f1,
    "tp": tp,
    "fp": fp,
    "fn": fn
  }

def train_one_epoch(
  model: torch.nn.Module,
  criterion: torch.nn.Module,
  data_loader: Iterable,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
  epoch: int,
  max_norm: float = 0,
  accumulation_steps: int = 1,
  amp: bool = False,
  scaler = None
):
  """Train the model for one epoch.

  Args:
    model: Model to train.
    criterion: Loss function.
    data_loader: Training data loader.
    optimizer: Optimizer.
    device: Torch device.
    epoch: Zero-based epoch number.
    max_norm: Maximum gradient norm for clipping.
    accumulation_steps: Number of physical batches accumulated before
      each optimizer update.

  Returns:
    Dictionary containing averaged training metrics.
  """
  if accumulation_steps < 1:
    raise ValueError('accumulation_steps must be at least 1')
  if amp and scaler is None:
    raise ValueError("AMP requires a GradScaler")
  model.train()
  criterion.train()
  metric_logger = utils.MetricLogger(delimiter = " ")
  metric_logger.add_meter(
    'lr',
    utils.SmoothedValue(window_size = 1, fmt = '{value:.6f}')
  )
  metric_logger.add_meter(
    'class_error',
    utils.SmoothedValue(window_size = 1, fmt = '{value:.2f}')
  )
  metric_logger.add_meter(
    'grad_norm',
    utils.SmoothedValue(window_size = 1, fmt = '{value:.2f}')
  )
  metric_logger.add_meter(
    'optimizer_step',
    utils.SmoothedValue(window_size = 1, fmt = '{value:.0f}')
  )
  header = 'Epoch: [{}]'.format(epoch)
  print_freq = 10
  prefetcher = data_prefetcher(data_loader, device, prefetch = False)
  samples, targets = prefetcher.next()
#
# Clear gradients before beginning the first accumulation group.
#
  optimizer.zero_grad()
  for iteration in metric_logger.log_every(
    range(len(data_loader)),
    print_freq,
    header
  ):
    if torch.cuda.is_available():
      allocated_gb = torch.cuda.memory_allocated() / 1024 ** 3
      reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
      if reserved_gb > 28.0:
        print(
          f"TRAIN_BATCH_MEMORY_PRE "
          f"epoch={epoch} "
          f"iteration={iteration} "
          f"cuda_allocated_gb={allocated_gb:.3f} "
          f"cuda_reserved_gb={reserved_gb:.3f} "
          f"{summarize_training_batch(targets)}"
        )
    with torch.amp.autocast(
      "cuda",
      enabled = amp
    ):
      outputs = model(samples)
      loss_dict = criterion(outputs, targets)
      weight_dict = criterion.weight_dict
      losses = sum(
        loss_dict[k] * weight_dict[k]
        for k in loss_dict.keys()
        if k in weight_dict
      )
#
# Reduce losses over all GPUs for logging purposes.
#
    loss_dict_reduced = utils.reduce_dict(loss_dict)
    loss_dict_reduced_unscaled = {
      f'{k}_unscaled': v
      for k, v in loss_dict_reduced.items()
    }
    loss_dict_reduced_scaled = {
      k: v * weight_dict[k]
      for k, v in loss_dict_reduced.items()
      if k in weight_dict
    }
    losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    loss_value = losses_reduced_scaled.item()
    if not math.isfinite(loss_value):
      print("Loss is {}, stopping training".format(loss_value))
      print(loss_dict_reduced)
      sys.exit(1)
#
# Divide by the number of microbatches in the current accumulation
# group so that gradient magnitude remains comparable to a larger
# physical batch.
#
    group_start = (
      iteration // accumulation_steps
    ) * accumulation_steps
    group_size = min(
      accumulation_steps,
      len(data_loader) - group_start
    )
    scaled_losses = losses / group_size
    if amp:
      scaler.scale(scaled_losses).backward()
    else:
      scaled_losses.backward()
    should_step = (
      ((iteration + 1) % accumulation_steps == 0) or
      (iteration == len(data_loader) - 1)
    )
#
# Temporarily verify gradient-accumulation boundaries.
#
    if iteration < 12:
      print(
        f'accumulation check: '
        f'iteration={iteration:02d}, '
        f'microbatch={(iteration % accumulation_steps) + 1}/'
        f'{accumulation_steps}, '
        f'optimizer_step={should_step}'
      )
#
    if should_step:
#
# Clip only after the accumulated gradient has been formed.
# With AMP, gradients must be unscaled before clipping.
#
      if amp:
        scaler.unscale_(optimizer)

      if max_norm > 0:
        grad_total_norm = torch.nn.utils.clip_grad_norm_(
          model.parameters(),
          max_norm
        )
      else:
        grad_total_norm = utils.get_total_grad_norm(
          model.parameters(),
          max_norm
        )

      if amp:
        scaler.step(optimizer)
        scaler.update()
      else:
        optimizer.step()

      optimizer.zero_grad()
    else:
      grad_total_norm = 0.0
    metric_logger.update(
      loss = loss_value,
      **loss_dict_reduced_scaled,
      **loss_dict_reduced_unscaled
    )
    metric_logger.update(
      class_error = loss_dict_reduced['class_error']
    )
    metric_logger.update(
      lr = optimizer.param_groups[0]["lr"]
    )
    metric_logger.update(
      grad_norm = grad_total_norm
    )
    metric_logger.update(
      optimizer_step = float(should_step)
    )
    del outputs
    del loss_dict
    del loss_dict_reduced
    del loss_dict_reduced_unscaled
    del loss_dict_reduced_scaled
    del losses
    del scaled_losses
    samples, targets = prefetcher.next()
#
# Gather the stats from all processes.
#
  metric_logger.synchronize_between_processes()
  print("Averaged stats:", metric_logger)
#
# Report end-of-epoch CUDA usage.
#
  log_cuda_memory(f"TRAIN_CUDA_MEMORY epoch={epoch}")
  return {
    k: meter.global_avg
    for k, meter in metric_logger.meters.items()
  }

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    val_detection_image_records = {}
    for val_iteration, (samples, targets) in enumerate(
      metric_logger.log_every(data_loader, 10, header)
      ):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        val_detection_image_records.update(
          build_detection_f1_image_records(
            results = results,
            targets = targets
          )
        )
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
#
# Release per-batch GPU references before the next validation batch.
#
        del outputs
        del loss_dict
        del loss_dict_reduced
        del loss_dict_reduced_scaled
        del loss_dict_reduced_unscaled
        del results
        del res
        del samples
        del targets
        del orig_target_sizes
        if "target_sizes" in locals():
          del target_sizes
        if "res_pano" in locals():
          del res_pano
        if (
          torch.cuda.is_available() and
          (val_iteration + 1) % 100 == 0
          ):
          torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    val_detection_iou_threshold = 0.5
    valid_category_ids = sorted([
      int(category_id)
      for category_id in base_ds.cats.keys()
    ])

    val_detection_metrics_per_class = summarize_best_detection_f1_per_class(
      val_detection_image_records,
      valid_category_ids = valid_category_ids,
      iou_threshold = val_detection_iou_threshold
    )
    print(
      "VAL_F1_PER_CLASS "
      f"iou={val_detection_iou_threshold:.4f} "
      f"precision={val_detection_metrics_per_class['precision']:.6f} "
      f"recall={val_detection_metrics_per_class['recall']:.6f} "
      f"f1={val_detection_metrics_per_class['f1']:.6f} "
      f"tp={val_detection_metrics_per_class['tp']} "
      f"fp={val_detection_metrics_per_class['fp']} "
      f"fn={val_detection_metrics_per_class['fn']} "
      f"micro_precision="
      f"{val_detection_metrics_per_class['micro_precision']:.6f} "
      f"micro_recall="
      f"{val_detection_metrics_per_class['micro_recall']:.6f} "
      f"micro_f1="
      f"{val_detection_metrics_per_class['micro_f1']:.6f}"
      f" valid_classes="
      f"{val_detection_metrics_per_class['num_valid_categories']} "
      f"pred_classes="
      f"{val_detection_metrics_per_class['num_pred_categories']} "
      f"extra_pred_classes="
      f"{val_detection_metrics_per_class['num_extra_pred_categories']}"
    )
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["val_f1_per_class_iou_0_50_precision"] = (
      val_detection_metrics_per_class["precision"]
    )
    stats["val_f1_per_class_iou_0_50_recall"] = (
      val_detection_metrics_per_class["recall"]
    )
    stats["val_f1_per_class_iou_0_50"] = (
      val_detection_metrics_per_class["f1"]
    )
    stats["val_f1_per_class_iou_0_50_thresholds"] = (
      val_detection_metrics_per_class["threshold_by_category_id"]
    )
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
#
# Report end-of-validation CUDA usage.
#
    log_cuda_memory("VAL_CUDA_MEMORY")
    return stats, coco_evaluator
