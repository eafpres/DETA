# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


def crop_with_visibility(
    image,
    target,
    region,
    minimum_visible_fraction,
    anchor_index = None
):
    """Crop a detection sample while rejecting misleading box fragments.
    Args:
        image: Source PIL image.
        target: Detection target with XYXY boxes.
        region: Crop tuple in top, left, height, width order.
        minimum_visible_fraction: Required retained fraction of non-anchor boxes.
        anchor_index: Optional box index that must be retained.
    Returns:
        tuple: Cropped image and consistently filtered target.
    """
    cropped_image = F.crop(image, *region)
    cropped_target = target.copy()
    top, left, height, width = region
    cropped_target["size"] = torch.tensor([height, width])
    boxes = target.get("boxes")
    if boxes is None:
        return cropped_image, cropped_target
    offset = torch.as_tensor(
        [left, top, left, top],
        dtype = boxes.dtype,
        device = boxes.device
    )
    bounds = torch.as_tensor(
        [width, height],
        dtype = boxes.dtype,
        device = boxes.device
    )
    shifted_boxes = boxes - offset
    clipped_boxes = torch.min(
        shifted_boxes.reshape(-1, 2, 2),
        bounds
    ).clamp(min = 0).reshape(-1, 4)
    original_widths = (boxes[:, 2] - boxes[:, 0]).clamp(min = 0)
    original_heights = (boxes[:, 3] - boxes[:, 1]).clamp(min = 0)
    original_areas = original_widths * original_heights
    visible_widths = (clipped_boxes[:, 2] - clipped_boxes[:, 0]).clamp(min = 0)
    visible_heights = (clipped_boxes[:, 3] - clipped_boxes[:, 1]).clamp(min = 0)
    visible_areas = visible_widths * visible_heights
    visible_fractions = torch.where(
        original_areas > 0,
        visible_areas / original_areas,
        torch.zeros_like(original_areas)
    )
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    centers_inside = (
        (center_x >= left) &
        (center_x <= left + width) &
        (center_y >= top) &
        (center_y <= top + height)
    )
    keep = (
        centers_inside &
        (visible_fractions >= minimum_visible_fraction) &
        (visible_areas > 0)
    )
    if anchor_index is not None:
        keep[int(anchor_index)] = True
    cropped_target["boxes"] = clipped_boxes[keep]
    cropped_target["area"] = visible_areas[keep]
    for field in ("labels", "iscrowd"):
        if field in target:
            cropped_target[field] = target[field][keep]
    if "masks" in target:
        cropped_target["masks"] = target["masks"][
            keep,
            top:top + height,
            left:left + width
        ]
    if "keypoints" in target:
        keypoints = target["keypoints"].clone()
        keypoints[..., 0] -= left
        keypoints[..., 1] -= top
        inside = (
            (keypoints[..., 0] >= 0) &
            (keypoints[..., 0] <= width) &
            (keypoints[..., 1] >= 0) &
            (keypoints[..., 1] <= height)
        )
        keypoints[..., 2] *= inside.to(keypoints.dtype)
        keypoints[..., 0].clamp_(min = 0, max = width)
        keypoints[..., 1].clamp_(min = 0, max = height)
        cropped_target["keypoints"] = keypoints[keep]
    return cropped_image, cropped_target
#
# The mixed policy keeps most samples contextual, magnifies selected localized
# damage without pre-rendering images, and still supplies realistic background.
#
class MixedDamageCrop(object):
    """Select full-image, annotation-centered, or random crop augmentation."""
    FULL_IMAGE = 0
    ANNOTATION_CENTERED = 1
    RANDOM_BACKGROUND = 2
    def __init__(
        self,
        full_probability,
        annotation_probability,
        background_probability,
        anchor_class_ids,
        anchor_max_area,
        context_multiplier,
        minimum_crop_size,
        maximum_crop_size,
        center_jitter,
        minimum_visible_fraction,
        background_attempts
    ):
        """Configure the damage-specific crop mixture.
        Args:
            full_probability: Probability of retaining the full image.
            annotation_probability: Probability of requesting an anchor crop.
            background_probability: Probability of requesting a random crop.
            anchor_class_ids: Localized category IDs eligible as anchors.
            anchor_max_area: Largest native box area eligible as a small anchor.
            context_multiplier: Crop side relative to the anchor dimensions.
            minimum_crop_size: Minimum native crop side.
            maximum_crop_size: Maximum native crop side.
            center_jitter: Maximum fractional crop-side center displacement.
            minimum_visible_fraction: Visibility required for non-anchor boxes.
            background_attempts: Random regions considered for background crops.
        """
        probabilities = (
            full_probability,
            annotation_probability,
            background_probability
        )
        if any(value < 0 for value in probabilities):
            raise ValueError("Mixed crop probabilities cannot be negative.")
        if abs(sum(probabilities) - 1.0) > 1e-9:
            raise ValueError("Mixed crop probabilities must sum to 1.0.")
        if minimum_crop_size <= 0 or maximum_crop_size < minimum_crop_size:
            raise ValueError("Mixed crop size bounds are invalid.")
        if not 0 <= center_jitter <= 1:
            raise ValueError("Mixed crop center jitter must be within [0, 1].")
        if not 0 <= minimum_visible_fraction <= 1:
            raise ValueError("Minimum visible fraction must be within [0, 1].")
        if background_attempts < 1:
            raise ValueError("Background attempts must be positive.")
        self.full_probability = full_probability
        self.annotation_probability = annotation_probability
        self.background_probability = background_probability
        self.anchor_class_ids = set(int(value) for value in anchor_class_ids)
        self.anchor_max_area = float(anchor_max_area)
        self.context_multiplier = float(context_multiplier)
        self.minimum_crop_size = int(minimum_crop_size)
        self.maximum_crop_size = int(maximum_crop_size)
        self.center_jitter = float(center_jitter)
        self.minimum_visible_fraction = float(minimum_visible_fraction)
        self.background_attempts = int(background_attempts)
        self._forced_branch = None
    def sample_branch(self):
        """Draw one branch code from the configured mixture."""
        draw = random.random()
        if draw < self.full_probability:
            return self.FULL_IMAGE
        if draw < self.full_probability + self.annotation_probability:
            return self.ANNOTATION_CENTERED
        return self.RANDOM_BACKGROUND
    def force_next_branch(self, branch):
        """Force one dataset-routed branch without changing later draws.
        Args:
            branch: One of the integer branch constants.
        """
        valid = {
            self.FULL_IMAGE,
            self.ANNOTATION_CENTERED,
            self.RANDOM_BACKGROUND
        }
        if branch not in valid:
            raise ValueError(f"Unknown mixed crop branch: {branch}")
        self._forced_branch = branch
    def annotation_is_anchor(self, annotation):
        """Check anchor eligibility directly from one raw COCO annotation.
        Args:
            annotation: Raw COCO annotation dictionary.
        Returns:
            bool: Whether the annotation can anchor a centered crop.
        """
        if int(annotation.get("iscrowd", 0)) != 0:
            return False
        if int(annotation["category_id"]) not in self.anchor_class_ids:
            return False
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            return False
        area = max(float(bbox[2]), 0.0) * max(float(bbox[3]), 0.0)
        return (
            area > 0 and
            (self.anchor_max_area <= 0 or area <= self.anchor_max_area)
        )
    def _record_metadata(
        self,
        target,
        requested_branch,
        applied_branch,
        crop_size,
        fallback,
        anchor_label = -1,
        anchor_box_size = (0.0, 0.0)
    ):
        """Attach tensor-only audit fields that survive device prefetching.
        Args:
            target: Detection target to annotate.
            requested_branch: Initially sampled branch code.
            applied_branch: Branch code actually applied.
            crop_size: Native crop height and width.
            fallback: Whether a requested anchor crop fell back to full image.
            anchor_label: Selected anchor category or -1.
            anchor_box_size: Native anchor width and height.
        Returns:
            dict: Copied target with augmentation audit tensors.
        """
        target = target.copy()
        target["augmentation_requested_branch"] = torch.tensor(
            requested_branch,
            dtype = torch.int64
        )
        target["augmentation_branch"] = torch.tensor(
            applied_branch,
            dtype = torch.int64
        )
        target["augmentation_fallback"] = torch.tensor(
            int(fallback),
            dtype = torch.int64
        )
        target["augmentation_crop_size"] = torch.tensor(
            crop_size,
            dtype = torch.int64
        )
        target["augmentation_anchor_label"] = torch.tensor(
            anchor_label,
            dtype = torch.int64
        )
        target["augmentation_anchor_box_size"] = torch.tensor(
            anchor_box_size,
            dtype = torch.float32
        )
        return target
    def _full_image(self, image, target, requested_branch, fallback = False):
        """Return an unchanged image while recording branch fallback behavior."""
        target = self._record_metadata(
            target,
            requested_branch = requested_branch,
            applied_branch = self.FULL_IMAGE,
            crop_size = (image.height, image.width),
            fallback = fallback
        )
        return image, target
    def _anchor_candidates(self, target):
        """Return localized, native-small annotation indices."""
        boxes = target.get("boxes")
        labels = target.get("labels")
        if boxes is None or labels is None or boxes.shape[0] == 0:
            return []
        widths = (boxes[:, 2] - boxes[:, 0]).clamp(min = 0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min = 0)
        areas = widths * heights
        return [
            index
            for index, (label, area) in enumerate(zip(labels, areas))
            if int(label.item()) in self.anchor_class_ids and
            (
                self.anchor_max_area <= 0 or
                float(area.item()) <= self.anchor_max_area
            )
        ]
    def _anchor_region(self, image, box):
        """Create a jittered square region that fully contains its anchor."""
        box_left, box_top, box_right, box_bottom = [
            float(value) for value in box.tolist()
        ]
        box_width = box_right - box_left
        box_height = box_bottom - box_top
        requested_side = max(
            self.context_multiplier * box_width,
            self.context_multiplier * box_height,
            self.minimum_crop_size
        )
        requested_side = min(requested_side, self.maximum_crop_size)
        crop_width = min(int(round(requested_side)), image.width)
        crop_height = min(int(round(requested_side)), image.height)
        center_x = (box_left + box_right) / 2
        center_y = (box_top + box_bottom) / 2
        desired_left = (
            center_x - crop_width / 2 +
            random.uniform(-self.center_jitter, self.center_jitter) * crop_width
        )
        desired_top = (
            center_y - crop_height / 2 +
            random.uniform(-self.center_jitter, self.center_jitter) * crop_height
        )
        minimum_left = max(0.0, box_right - crop_width)
        maximum_left = min(box_left, image.width - crop_width)
        minimum_top = max(0.0, box_bottom - crop_height)
        maximum_top = min(box_top, image.height - crop_height)
        if minimum_left > maximum_left:
            minimum_left = maximum_left = max(
                0.0,
                min(center_x - crop_width / 2, image.width - crop_width)
            )
        if minimum_top > maximum_top:
            minimum_top = maximum_top = max(
                0.0,
                min(center_y - crop_height / 2, image.height - crop_height)
            )
        left = int(round(min(max(desired_left, minimum_left), maximum_left)))
        top = int(round(min(max(desired_top, minimum_top), maximum_top)))
        return top, left, crop_height, crop_width
    def _random_region(self, image, target):
        """Choose the cleanest of several random native-resolution crops."""
        boxes = target.get("boxes")
        best_region = None
        best_score = None
        for _ in range(self.background_attempts):
            side = random.randint(
                self.minimum_crop_size,
                self.maximum_crop_size
            )
            crop_width = min(side, image.width)
            crop_height = min(side, image.height)
            left = random.randint(0, max(image.width - crop_width, 0))
            top = random.randint(0, max(image.height - crop_height, 0))
            region = (top, left, crop_height, crop_width)
            if boxes is None or boxes.shape[0] == 0:
                return region
            center_x = (boxes[:, 0] + boxes[:, 2]) / 2
            center_y = (boxes[:, 1] + boxes[:, 3]) / 2
            center_count = int((
                (center_x >= left) &
                (center_x <= left + crop_width) &
                (center_y >= top) &
                (center_y <= top + crop_height)
            ).sum().item())
            intersection_left = boxes[:, 0].clamp(min = left, max = left + crop_width)
            intersection_right = boxes[:, 2].clamp(min = left, max = left + crop_width)
            intersection_top = boxes[:, 1].clamp(min = top, max = top + crop_height)
            intersection_bottom = boxes[:, 3].clamp(min = top, max = top + crop_height)
            intersection = (
                (intersection_right - intersection_left).clamp(min = 0) *
                (intersection_bottom - intersection_top).clamp(min = 0)
            )
            overlap_fraction = float(
                intersection.sum().item() /
                max(crop_width * crop_height, 1)
            )
            score = (center_count, overlap_fraction)
            if best_score is None or score < best_score:
                best_score = score
                best_region = region
            if score == (0, 0.0):
                break
        return best_region
    def __call__(self, image, target):
        requested_branch = self._forced_branch
        self._forced_branch = None
        if requested_branch is None:
            requested_branch = self.sample_branch()
        if requested_branch == self.FULL_IMAGE:
            return self._full_image(
                image,
                target,
                requested_branch = self.FULL_IMAGE
            )
        if requested_branch == self.ANNOTATION_CENTERED:
            candidates = self._anchor_candidates(target)
            if not candidates:
                return self._full_image(
                    image,
                    target,
                    requested_branch = self.ANNOTATION_CENTERED,
                    fallback = True
                )
            anchor_index = random.choice(candidates)
            anchor_box = target["boxes"][anchor_index]
            anchor_label = int(target["labels"][anchor_index].item())
            anchor_box_size = (
                float((anchor_box[2] - anchor_box[0]).item()),
                float((anchor_box[3] - anchor_box[1]).item())
            )
            region = self._anchor_region(image, anchor_box)
            cropped_image, cropped_target = crop_with_visibility(
                image,
                target,
                region,
                self.minimum_visible_fraction,
                anchor_index = anchor_index
            )
            cropped_target = self._record_metadata(
                cropped_target,
                requested_branch = self.ANNOTATION_CENTERED,
                applied_branch = self.ANNOTATION_CENTERED,
                crop_size = (region[2], region[3]),
                fallback = False,
                anchor_label = anchor_label,
                anchor_box_size = anchor_box_size
            )
            return cropped_image, cropped_target
        region = self._random_region(image, target)
        cropped_image, cropped_target = crop_with_visibility(
            image,
            target,
            region,
            self.minimum_visible_fraction
        )
        cropped_target = self._record_metadata(
            cropped_target,
            requested_branch = self.RANDOM_BACKGROUND,
            applied_branch = self.RANDOM_BACKGROUND,
            crop_size = (region[2], region[3]),
            fallback = False
        )
        return cropped_image, cropped_target
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
