"""
Smoke test: verify the generator and discriminator forward passes produce
correctly shaped outputs on random input. Runs on CPU so no GPU is required.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pix2pixHD"))

try:
    import torch
    from models import networks
except ImportError as e:
    pytest.skip(f"Skipping model tests: {e}", allow_module_level=True)


def _make_generator(input_nc=1, output_nc=1, ngf=16, n_downsampling=2, n_blocks=4):
    """Create a small GlobalGenerator for CPU smoke-testing."""
    norm_layer = networks.get_norm_layer("instance")
    return networks.GlobalGenerator(
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        n_downsampling=n_downsampling,
        n_blocks=n_blocks,
        norm_layer=norm_layer,
    ).eval()


def _make_discriminator(input_nc=2, ndf=16, n_layers=2):
    norm_layer = networks.get_norm_layer("instance")
    return networks.NLayerDiscriminator(
        input_nc=input_nc,
        ndf=ndf,
        n_layers=n_layers,
        norm_layer=norm_layer,
    ).eval()


class TestGeneratorForward:
    @pytest.mark.parametrize("spatial", [64, 128])
    def test_output_shape(self, spatial):
        net = _make_generator()
        x = torch.randn(1, 1, spatial, spatial)
        with torch.no_grad():
            out = net(x)
        assert out.shape == (1, 1, spatial, spatial), f"Expected (1,1,{spatial},{spatial}), got {out.shape}"

    def test_output_range(self):
        """Generator output should be in [-1, 1] (tanh activation)."""
        net = _make_generator()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_no_nan_in_output(self):
        net = _make_generator()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert not torch.isnan(out).any()


class TestDiscriminatorForward:
    def test_output_shape(self):
        """Discriminator returns a spatial map, not a scalar."""
        net = _make_discriminator(input_nc=2)
        x = torch.randn(1, 2, 64, 64)  # concatenated real/fake + condition
        with torch.no_grad():
            out = net(x)
        # Output should be a 4-D tensor with spatial dims < input
        assert out.dim() == 4
        assert out.shape[0] == 1

    def test_no_nan_in_output(self):
        net = _make_discriminator()
        x = torch.randn(1, 2, 64, 64)
        with torch.no_grad():
            out = net(x)
        assert not torch.isnan(out).any()


class TestCommandGen:
    """Verify command_gen.py correctly converts config/train.yaml to CLI args."""

    def test_bool_flag_without_value(self, tmp_path):
        import yaml as _yaml
        cfg = {"name": "test_run", "no_vgg_loss": True, "fp16": False, "batchSize": 4}
        p = tmp_path / "train.yaml"
        p.write_text(_yaml.dump(cfg))

        # Import the helper from command_gen
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from command_gen import config_to_args
        args = config_to_args(cfg)

        assert "--no_vgg_loss" in args        # True bool → flag present, no value
        assert "--fp16" not in args           # False bool → flag absent
        assert "--batchSize" in args
        idx = args.index("--batchSize")
        assert args[idx + 1] == "4"
