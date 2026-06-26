# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

  gt_boxes = torch.as_tensor(gt_boxes, dtype = torch.float32)
  pred_boxes = torch.as_tensor(pred_boxes, dtype = torch.float32)
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
  iou_threshold = 0.5,
  threshold_min = 0.05,
  threshold_max = 0.95,
  threshold_step = 0.01
):
  """Optimize one confidence threshold per class and summarize micro F1.

  Args:
    image_records: Validation records created by build_detection_f1_image_records.
    threshold_min: Minimum confidence threshold.
    threshold_max: Maximum confidence threshold.
    threshold_step: Threshold step size.

  Returns:
    dict: Per-class threshold metrics and aggregate micro F1.
  """
  gathered_image_records = utils.all_gather(image_records)
  merged_image_records = {}
  for rank_image_records in gathered_image_records:
    merged_image_records.update(rank_image_records)
  image_records = merged_image_records
  category_ids = sorted({
    int(label)
    for image_record in image_records.values()
    for label in (
      list(image_record.get("gt_labels", [])) +
      list(image_record.get("pred_labels", []))
    )
  })
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
  precision, recall, f1 = calculate_detection_prf(tp, fp, fn)

  return {
    "threshold_by_category_id": threshold_by_category_id,
    "class_metrics": class_metrics,
    "precision": precision,
    "recall": recall,
    "f1": f1,
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
  accumulation_steps: int = 1
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
  prefetcher = data_prefetcher(data_loader, device, prefetch = True)
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
#
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
    samples, targets = prefetcher.next()
#
# Gather the stats from all processes.
#
  metric_logger.synchronize_between_processes()
  print("Averaged stats:", metric_logger)
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
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
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
    val_detection_metrics_per_class = summarize_best_detection_f1_per_class(
      val_detection_image_records,
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
      f"fn={val_detection_metrics_per_class['fn']}"
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
    return stats, coco_evaluator
