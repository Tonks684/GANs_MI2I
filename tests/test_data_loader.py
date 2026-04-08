"""
Smoke tests for the aligned TIFF dataset.
Creates a tiny synthetic dataset on disk and verifies the loader returns
correctly shaped tensors without touching remote data.
"""
import sys
import os
import tempfile
import types

import numpy as np
import pytest

# Allow import from repo root and pix2pixHD sub-package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pix2pixHD"))

try:
    import torch
    from tifffile import imwrite
    from data.aligned_dataset_dlmbl import AlignedDatasetDLMBL
except ImportError as e:
    pytest.skip(f"Skipping data-loader tests: {e}", allow_module_level=True)


def _make_synthetic_dataset(root: str, n: int = 4, size: int = 64):
    """Write n paired 16-bit TIFF patches into root/input/train and root/nuclei/train."""
    rng = np.random.default_rng(42)
    for split in ("train", "val"):
        for channel in ("input", "nuclei"):
            os.makedirs(os.path.join(root, channel, split), exist_ok=True)
        for i in range(n):
            phase = rng.integers(0, 65535, (size, size), dtype=np.uint16)
            nucl = rng.integers(0, 65535, (size, size), dtype=np.uint16)
            imwrite(os.path.join(root, "input", split, f"{i:04d}.tiff"), phase)
            imwrite(os.path.join(root, "nuclei", split, f"{i:04d}.tiff"), nucl)


def _make_opt(dataroot: str, phase: str = "train", load_size: int = 64, fine_size: int = 64):
    opt = types.SimpleNamespace(
        dataroot=dataroot,
        target="nuclei",
        phase=phase,
        loadSize=load_size,
        fineSize=fine_size,
        input_nc=1,
        output_nc=1,
        label_nc=0,
        no_instance=True,
        no_flip=True,
        resize_or_crop="none",
        data_type=16,
        isTrain=True,
        max_dataset_size=float("inf"),
        serial_batches=True,
    )
    return opt


class TestAlignedDatasetDLMBL:
    @pytest.fixture(autouse=True)
    def synthetic_data(self, tmp_path):
        _make_synthetic_dataset(str(tmp_path), n=4, size=64)
        self.root = str(tmp_path)

    def test_len(self):
        opt = _make_opt(self.root, phase="train")
        ds = AlignedDatasetDLMBL()
        ds.initialize(opt)
        assert len(ds) == 4

    def test_item_shapes(self):
        opt = _make_opt(self.root, phase="train")
        ds = AlignedDatasetDLMBL()
        ds.initialize(opt)
        item = ds[0]
        assert 'label' in item
        assert 'image' in item
        # Both tensors should have a channel dim
        assert item['label'].dim() == 3  # (C, H, W)
        assert item['image'].dim() == 3

    def test_item_dtype(self):
        opt = _make_opt(self.root, phase="train")
        ds = AlignedDatasetDLMBL()
        ds.initialize(opt)
        item = ds[0]
        assert item['label'].dtype == torch.float32
        assert item['image'].dtype == torch.float32

    def test_val_split(self):
        opt = _make_opt(self.root, phase="val")
        ds = AlignedDatasetDLMBL()
        ds.initialize(opt)
        assert len(ds) == 4
