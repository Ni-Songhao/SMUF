# Copyright (c) OpenMMLab. All rights reserved.
"""Random branch exposure transforms for dual-frequency segmentation.

This file is intentionally self-contained so it can be copied into an mmseg
project and imported with:

custom_imports = dict(
    imports=['mmseg.datasets.transforms.random_band_mask'],
    allow_failed_imports=False)

The training policy is the original random 1:1:1 exposure scheme:
- mode 0: keep branch 0, mask branch 1
- mode 1: keep branch 1, mask branch 0
- mode 2: keep both branches

It is useful when training on a fixed dual-frequency pair such as LS or CX and
the model should remain robust to both single-frequency and dual-frequency
inputs.

Do not combine this data-level masking loader with backbones that already
sample the exposure mode internally and perform prompt completion, such as the
legacy ``LSKNet_Base`` with ``probs=[1, 1, 1]``.  For that recipe, load the full
dual-frequency input and let the backbone choose the exposure mode.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from mmcv.transforms.base import BaseTransform

from mmseg.registry import TRANSFORMS


def _as_hwc_array(array: np.ndarray) -> np.ndarray:
    """Normalize npy arrays to HWC layout."""
    if array.ndim == 3 and array.shape[0] <= 16 and array.shape[1] > 16 and \
            array.shape[2] > 16:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f'Expected 2D or 3D ndarray, got shape {array.shape}')
    return array


def _load_npy_hwc(path: str) -> np.ndarray:
    return _as_hwc_array(np.load(path))


def _validate_probabilities(probabilities: Sequence[float]) -> Tuple[float,
                                                                     float,
                                                                     float]:
    if len(probabilities) != 3:
        raise ValueError('probabilities must contain three values for '
                         'branch0, branch1, and dual modes.')
    probs = tuple(float(p) for p in probabilities)
    if any(p < 0 for p in probs) or sum(probs) <= 0:
        raise ValueError('probabilities must be non-negative and have '
                         'positive sum.')
    return probs


def _sample_mode(probabilities: Tuple[float, float, float]) -> int:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(3, p=probs))


def _apply_branch_exposure(branch0: np.ndarray,
                           branch1: np.ndarray,
                           mode: int,
                           missing_value: float = 0.0) -> Tuple[np.ndarray,
                                                                int,
                                                                np.ndarray]:
    """Apply branch exposure and return concatenated image, modality, mask."""
    if branch0.shape != branch1.shape:
        raise ValueError('The two branches must have the same shape, got '
                         f'{branch0.shape} and {branch1.shape}.')

    if mode == 0:
        fill = np.full_like(branch1, missing_value)
        img = np.concatenate([branch0, fill], axis=-1)
        modality = 0
        mask = np.array([0, 1], dtype=np.int64)
    elif mode == 1:
        fill = np.full_like(branch0, missing_value)
        img = np.concatenate([fill, branch1], axis=-1)
        modality = 1
        mask = np.array([1, 0], dtype=np.int64)
    elif mode == 2:
        img = np.concatenate([branch0, branch1], axis=-1)
        modality = 2
        mask = np.array([0, 0], dtype=np.int64)
    else:
        raise ValueError(f'Unsupported mode: {mode}')
    return img, modality, mask


@TRANSFORMS.register_module()
class LoadDualNpyWithRandomBandMask(BaseTransform):
    """Load two npy branches and randomly expose one or both branches.

    Required keys:
        - ``img_path``: path to branch 0 npy file
        - ``img_path2``: path to branch 1 npy file

    Added / modified keys:
        - ``img``: concatenated HWC array
        - ``img_shape`` / ``ori_shape``
        - ``modality``: 0, 1, or 2
        - ``mask``: length-2 array, where 1 means the branch is missing
        - optionally ``source_id`` and ``source_name``

    Example:
        train_pipeline = [
            dict(
                type='LoadDualNpyWithRandomBandMask',
                probabilities=(1, 1, 1),
                source_id=0,
                source_name='LS'),
            dict(type='LoadAnnotations'),
            ...
        ]
    """

    def __init__(self,
                 probabilities: Sequence[float] = (1, 1, 1),
                 to_float32: bool = True,
                 missing_value: float = 0.0,
                 source_id: Optional[int] = None,
                 source_name: Optional[str] = None):
        self.probabilities = _validate_probabilities(probabilities)
        self.to_float32 = bool(to_float32)
        self.missing_value = float(missing_value)
        self.source_id = None if source_id is None else int(source_id)
        self.source_name = source_name

    def transform(self, results: dict) -> dict:
        branch0 = _load_npy_hwc(results['img_path'])
        branch1 = _load_npy_hwc(results['img_path2'])
        mode = _sample_mode(self.probabilities)
        img, modality, mask = _apply_branch_exposure(
            branch0, branch1, mode, self.missing_value)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['modality'] = modality
        results['mask'] = mask
        if self.source_id is not None:
            results['source_id'] = np.array([self.source_id], dtype=np.int64)
        if self.source_name is not None:
            results['source_name'] = self.source_name
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'probabilities={self.probabilities}, '
                f'to_float32={self.to_float32}, '
                f'missing_value={self.missing_value}, '
                f'source_id={self.source_id}, '
                f'source_name={self.source_name})')


@TRANSFORMS.register_module()
class RandomBandMask(BaseTransform):
    """Randomly mask one branch of an already-loaded dual-branch image.

    This transform is useful when another loader has already produced
    ``results['img']`` as a concatenated dual-branch HWC array.

    Args:
        branch_channels: Number of channels per branch. If None, the channel
            dimension is split in half.
        probabilities: Sampling weights for branch0-only, branch1-only, dual.
        missing_value: Fill value for the masked branch.
    """

    def __init__(self,
                 branch_channels: Optional[int] = None,
                 probabilities: Sequence[float] = (1, 1, 1),
                 missing_value: float = 0.0):
        self.branch_channels = branch_channels
        self.probabilities = _validate_probabilities(probabilities)
        self.missing_value = float(missing_value)

    def transform(self, results: dict) -> dict:
        img = _as_hwc_array(results['img'])
        channels = img.shape[-1]
        branch_channels = self.branch_channels
        if branch_channels is None:
            if channels % 2 != 0:
                raise ValueError('Cannot infer branch_channels from odd '
                                 f'channel count: {channels}')
            branch_channels = channels // 2
        if branch_channels <= 0 or branch_channels * 2 != channels:
            raise ValueError('branch_channels must split img channels into '
                             f'two equal branches, got {branch_channels} '
                             f'for {channels} channels.')

        branch0 = img[..., :branch_channels]
        branch1 = img[..., branch_channels:]
        mode = _sample_mode(self.probabilities)
        out, modality, mask = _apply_branch_exposure(
            branch0, branch1, mode, self.missing_value)

        results['img'] = out.astype(img.dtype, copy=False)
        results['img_shape'] = out.shape[:2]
        results.setdefault('ori_shape', out.shape[:2])
        results['modality'] = modality
        results['mask'] = mask
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'branch_channels={self.branch_channels}, '
                f'probabilities={self.probabilities}, '
                f'missing_value={self.missing_value})')


@TRANSFORMS.register_module()
class LoadSingleNpyAsDualBranch(BaseTransform):
    """Load one npy branch and add a zero dummy branch.

    This is provided for single-frequency samples when the model expects the
    same two-branch input interface used by dual-frequency data.
    """

    def __init__(self,
                 dummy_position: str = 'second',
                 modality: int = 0,
                 to_float32: bool = True,
                 missing_value: float = 0.0,
                 source_id: Optional[int] = None,
                 source_name: Optional[str] = None):
        if dummy_position not in ('first', 'second'):
            raise ValueError("dummy_position must be 'first' or 'second'.")
        self.dummy_position = dummy_position
        self.modality = int(modality)
        self.to_float32 = bool(to_float32)
        self.missing_value = float(missing_value)
        self.source_id = None if source_id is None else int(source_id)
        self.source_name = source_name

    def transform(self, results: dict) -> dict:
        branch = _load_npy_hwc(results['img_path'])
        dummy = np.full_like(branch, self.missing_value)
        if self.dummy_position == 'first':
            img = np.concatenate([dummy, branch], axis=-1)
            mask = np.array([1, 0], dtype=np.int64)
        else:
            img = np.concatenate([branch, dummy], axis=-1)
            mask = np.array([0, 1], dtype=np.int64)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['modality'] = self.modality
        results['mask'] = mask
        if self.source_id is not None:
            results['source_id'] = np.array([self.source_id], dtype=np.int64)
        if self.source_name is not None:
            results['source_name'] = self.source_name
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'dummy_position={self.dummy_position}, '
                f'modality={self.modality}, '
                f'to_float32={self.to_float32}, '
                f'missing_value={self.missing_value}, '
                f'source_id={self.source_id}, '
                f'source_name={self.source_name})')
