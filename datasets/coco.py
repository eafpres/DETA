# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import random

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
#
# Routing annotation-centered draws through the eligible-image pool preserves
# the requested 35% branch rate instead of silently falling back on images that
# contain no localized small-damage anchor.
#
        self._mixed_crop = self._find_mixed_crop(transforms)
        self._mixed_anchor_indices = []
        self._mixed_anchor_index_set = set()
        if self._mixed_crop is not None:
            self._mixed_anchor_indices = [
                dataset_index
                for dataset_index, image_id in enumerate(self.ids)
                if any(
                    self._mixed_crop.annotation_is_anchor(annotation)
                    for annotation in self.coco.imgToAnns.get(image_id, [])
                )
            ]
            self._mixed_anchor_index_set = set(self._mixed_anchor_indices)
            if (
                self._mixed_crop.annotation_probability > 0 and
                not self._mixed_anchor_indices
            ):
                raise ValueError(
                    'Annotation-centered augmentation requires at least one '
                    'eligible anchor image.'
                )
    @staticmethod
    def _find_mixed_crop(transforms):
        """Locate the optional mixed-crop transform in the training pipeline.
        Args:
            transforms: Dataset transform or composed transform sequence.
        Returns:
            MixedDamageCrop or None: Configured crop policy when present.
        """
        if isinstance(transforms, T.MixedDamageCrop):
            return transforms
        for transform in getattr(transforms, 'transforms', []):
            mixed_crop = CocoDetection._find_mixed_crop(transform)
            if mixed_crop is not None:
                return mixed_crop
        return None
#
# A draw remains one logical sample even when its source image is rerouted.
# Tensor metadata makes this behavior auditable and remains compatible with the
# existing prefetcher, which moves every target field to the selected device.
#
    def _route_mixed_crop(self, idx):
        """Choose the branch and an eligible source index for one sample.
        Args:
            idx: Dataset index selected by the sampler.
        Returns:
            tuple: Source index, sampled branch code, and reroute indicator.
        """
        if self._mixed_crop is None:
            return idx, None, False
        branch = self._mixed_crop.sample_branch()
        rerouted = False
        if (
            branch == self._mixed_crop.ANNOTATION_CENTERED and
            idx not in self._mixed_anchor_index_set
        ):
            idx = random.choice(self._mixed_anchor_indices)
            rerouted = True
        self._mixed_crop.force_next_branch(branch)
        return idx, branch, rerouted

    def __getitem__(self, idx):
        source_idx, branch, rerouted = self._route_mixed_crop(idx)
        img, target = super(CocoDetection, self).__getitem__(source_idx)
        image_id = self.ids[source_idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if branch is not None:
            target['augmentation_rerouted'] = torch.tensor(
                int(rerouted),
                dtype = torch.int64
            )
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, bigger=False, mixed_crop_config=None):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if 'train' in image_set:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if 'val' in image_set or 'test' in image_set:
        scales = [800]

    max_size = 1333
    if bigger:
        scales = [int(1.5*s) for s in scales]
        max_size = 2000
    if image_set == 'train' and mixed_crop_config is not None:
        scales = list(mixed_crop_config['train_scales'])
        max_size = int(mixed_crop_config['max_size'])
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.MixedDamageCrop(
                full_probability = mixed_crop_config['full_probability'],
                annotation_probability = mixed_crop_config[
                    'annotation_probability'
                ],
                background_probability = mixed_crop_config[
                    'background_probability'
                ],
                anchor_class_ids = mixed_crop_config['anchor_class_ids'],
                anchor_max_area = mixed_crop_config['anchor_max_area'],
                context_multiplier = mixed_crop_config['context_multiplier'],
                minimum_crop_size = mixed_crop_config['minimum_crop_size'],
                maximum_crop_size = mixed_crop_config['maximum_crop_size'],
                center_jitter = mixed_crop_config['center_jitter'],
                minimum_visible_fraction = mixed_crop_config[
                    'minimum_visible_fraction'
                ],
                background_attempts = mixed_crop_config[
                    'background_attempts'
                ]
            ),
            T.RandomResize(scales, max_size=max_size),
            normalize,
        ])
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize(scales, max_size=max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
def build_mixed_crop_config(args):
    """Build and validate the optional damage-specific augmentation config.
    Args:
        args: Parsed training arguments.
    Returns:
        dict or None: Mixed-crop settings when enabled.
    """
    if not getattr(args, 'damage_mixed_crop', False):
        return None
    train_scales = [
        int(value) for value in args.damage_aug_train_scales
    ]
    if not train_scales or any(value <= 0 for value in train_scales):
        raise ValueError('Damage augmentation scales must be positive.')
    if args.damage_aug_max_size < max(train_scales):
        raise ValueError(
            'Damage augmentation max size cannot be smaller than its scales.'
        )
    anchor_class_ids = [
        int(value) for value in args.damage_aug_anchor_class_ids
    ]
    if not anchor_class_ids:
        raise ValueError('At least one damage anchor class ID is required.')
    return {
        'train_scales': sorted(set(train_scales)),
        'max_size': int(args.damage_aug_max_size),
        'full_probability': float(args.damage_aug_full_probability),
        'annotation_probability': float(
            args.damage_aug_annotation_probability
        ),
        'background_probability': float(
            args.damage_aug_background_probability
        ),
        'anchor_class_ids': sorted(set(anchor_class_ids)),
        'anchor_max_area': float(args.damage_aug_anchor_max_area),
        'context_multiplier': float(args.damage_aug_context_multiplier),
        'minimum_crop_size': int(args.damage_aug_min_crop_size),
        'maximum_crop_size': int(args.damage_aug_max_crop_size),
        'center_jitter': float(args.damage_aug_center_jitter),
        'minimum_visible_fraction': float(
            args.damage_aug_min_visible_fraction
        ),
        'background_attempts': int(args.damage_aug_background_attempts)
    }
def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'

    train_img = Path(args.coco_train_images) if getattr(args, "coco_train_images", "") else (root / "train2017")
    train_ann = Path(args.coco_train_ann) if getattr(args, "coco_train_ann", "") else (root / "annotations" / f"{mode}_train2017.json")
    val_img = Path(args.coco_val_images) if getattr(args, "coco_val_images", "") else (root / "val2017")
    val_ann = Path(args.coco_val_ann) if getattr(args, "coco_val_ann", "") else (root / "annotations" / f"{mode}_val2017.json")

    PATHS = {
        "train": (train_img, train_ann),
        "val": (val_img, val_ann),
    }

    img_folder, ann_file = PATHS[image_set]
    mixed_crop_config = build_mixed_crop_config(args)
    if image_set == 'train' and mixed_crop_config is not None:
        print(f'Damage mixed-crop augmentation: {mixed_crop_config}')
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms = make_coco_transforms(
            image_set,
            args.bigger,
            mixed_crop_config = mixed_crop_config
        ),
        return_masks = args.masks,
        cache_mode = args.cache_mode,
        local_rank = get_local_rank(),
        local_size = get_local_size()
    )
    return dataset
