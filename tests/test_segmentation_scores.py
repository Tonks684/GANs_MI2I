"""
Unit tests for segmentation_scores.py.
These tests use synthetic masks so no real data is needed.
"""
import math
import os
import sys

import numpy as np
import pytest

# Allow import from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from segmentation_scores import (
    intersection_over_union,
    measures_at,
    psnr_scores,
)

# ---------------------------------------------------------------------------
# intersection_over_union
# ---------------------------------------------------------------------------

class TestIntersectionOverUnion:
    def test_perfect_match(self):
        """Identical masks should give IoU = 1 for each object."""
        gt = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
        pred = gt.copy()
        iou = intersection_over_union(gt, pred)
        assert iou.shape[0] > 0
        np.testing.assert_allclose(np.max(iou, axis=1), 1.0, atol=1e-6)

    def test_no_overlap(self):
        """Non-overlapping objects should give IoU close to 0."""
        gt = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int32)
        pred = np.array([[0, 0, 2], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        iou = intersection_over_union(gt, pred)
        assert np.max(iou) < 1e-6

    def test_partial_overlap(self):
        """50% overlap should give IoU = 1/3 (intersection/union)."""
        gt = np.zeros((4, 4), dtype=np.int32)
        gt[:, :2] = 1
        pred = np.zeros((4, 4), dtype=np.int32)
        pred[:, 1:3] = 1
        iou = intersection_over_union(gt, pred)
        # intersection = 4 pixels, union = 12 pixels → IoU = 1/3
        np.testing.assert_allclose(iou[0, 0], 1 / 3, atol=1e-4)

    def test_empty_prediction(self):
        """Background-only prediction should return empty or zero IoU matrix."""
        gt = np.array([[1, 1], [0, 0]], dtype=np.int32)
        pred = np.zeros((2, 2), dtype=np.int32)
        iou = intersection_over_union(gt, pred)
        # pred has only background → no foreground objects
        assert iou.shape[1] == 0 or np.all(iou == 0)


# ---------------------------------------------------------------------------
# measures_at
# ---------------------------------------------------------------------------

class TestMeasuresAt:
    def _perfect_iou(self, n=3):
        """Return a perfect IoU matrix (identity-like) for n objects."""
        iou = np.eye(n)
        return iou

    def test_perfect_predictions(self):
        iou = self._perfect_iou(3)
        f1, tp, fp, fn = measures_at(0.5, iou)
        assert tp == 3
        assert fp == 0
        assert fn == 0
        np.testing.assert_allclose(f1, 1.0, atol=1e-6)

    def test_all_false_positives(self):
        """Model predicts objects where none exist in GT."""
        iou = np.zeros((0, 3))   # 0 GT objects, 3 predicted
        f1, tp, fp, fn = measures_at(0.5, iou)
        assert tp == 0
        assert fn == 0
        assert fp == 3

    def test_all_false_negatives(self):
        """Model predicts nothing."""
        iou = np.zeros((3, 0))   # 3 GT objects, 0 predicted
        f1, tp, fp, fn = measures_at(0.5, iou)
        assert tp == 0
        assert fp == 0
        assert fn == 3

    def test_raises_on_invalid_iou(self):
        """measures_at should raise ValueError when binary conditions fail."""
        # Simulate a degenerate IoU > 1 that causes a single GT object to match
        # multiple predictions at the same threshold — shouldn't happen with a
        # real IoU matrix, but guards the ValueError path.
        iou = np.array([[1.0, 1.0]])  # one GT object "matches" two predictions
        with pytest.raises(ValueError):
            measures_at(0.5, iou)


# ---------------------------------------------------------------------------
# psnr_scores
# ---------------------------------------------------------------------------

class TestPsnrScores:
    def _write_tiff(self, path, arr):
        from tifffile import imwrite
        imwrite(path, arr)

    def test_identical_images_returns_100(self, tmp_path):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        p1 = str(tmp_path / "a.tiff")
        p2 = str(tmp_path / "b.tiff")
        self._write_tiff(p1, img)
        self._write_tiff(p2, img)
        scores = psnr_scores([p1], [p2], bit=8)
        assert scores == [100.0]

    def test_known_psnr_8bit(self, tmp_path):
        """A constant-offset image has a calculable PSNR."""
        rng = np.random.default_rng(0)
        gt = rng.integers(50, 200, (64, 64), dtype=np.uint8)
        pred = np.clip(gt.astype(np.int32) + 10, 0, 255).astype(np.uint8)
        p_gt = str(tmp_path / "gt.tiff")
        p_pred = str(tmp_path / "pred.tiff")
        self._write_tiff(p_gt, gt)
        self._write_tiff(p_pred, pred)
        scores = psnr_scores([p_pred], [p_gt], bit=8)
        mse = np.mean((pred.astype(float) - gt.astype(float)) ** 2)
        expected = 20 * math.log10(255.0 / math.sqrt(mse))
        np.testing.assert_allclose(scores[0], expected, rtol=1e-4)

    def test_16bit_psnr(self, tmp_path):
        rng = np.random.default_rng(1)
        gt = rng.integers(0, 65535, (64, 64), dtype=np.uint16)
        pred = np.clip(gt.astype(np.int32) + 100, 0, 65535).astype(np.uint16)
        p_gt = str(tmp_path / "gt16.tiff")
        p_pred = str(tmp_path / "pred16.tiff")
        self._write_tiff(p_gt, gt)
        self._write_tiff(p_pred, pred)
        scores = psnr_scores([p_pred], [p_gt], bit=16)
        assert len(scores) == 1
        assert scores[0] > 0
