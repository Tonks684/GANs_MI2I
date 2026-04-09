"""
Generate the training CLI command from config/train.yaml.

Usage:
    python command_gen.py                       # prints command
    python command_gen.py | xargs python pix2pixHD/train_dlmbl.py
    python command_gen.py --config config/train.yaml
"""
import argparse

try:
    import yaml
except ImportError:
    import json as yaml  # fallback: rename train.yaml → train.json if pyyaml unavailable

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def config_to_args(cfg: dict) -> list[str]:
    """Convert a flat config dict to a list of CLI flag strings."""
    args = []
    for key, value in cfg.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml",
                        help="Path to YAML config file")
    opts = parser.parse_args()

    cfg = load_config(opts.config)
    cli_args = config_to_args(cfg)
    print(" ".join(cli_args))
