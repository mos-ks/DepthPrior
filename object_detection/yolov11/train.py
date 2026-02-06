import argparse
import contextlib

import yaml
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT


def main(config_file: str, cli_overrides: dict):
    config = load_config(config_file)
    config.update(cli_overrides)
    log_config(config)
    inject_custom_config_params()

    from depth.trainer import CustomDetectionTrainer

    trainer = CustomDetectionTrainer(overrides=config)
    trainer.train()


def inject_custom_config_params():
    """
    Injects custom configuration parameters into the default configuration objects:
        - far_weight (float): Weight parameter for 'far' objects.
        - close_weight (float): Weight parameter for 'close' objects.
        - depth_aware ("step", "linear", False):
            - "step": Applies a step function for depth-based weighting.
            - "linear": Applies a linear function for depth-based weighting.
            - False: Disables depth-aware processing.
        - alpha (float): Scaling factor for depth-based weighting.
        - depth_inverse (bool): If true, inverts normalized depth values for weighting.
        - depth_threshold (float): Depth cutoff used by the `dls` strategy.
    """
    DEFAULT_CFG.far_weight = 1.0
    DEFAULT_CFG_DICT["far_weight"] = 1.0
    DEFAULT_CFG.close_weight = 1.0
    DEFAULT_CFG_DICT["close_weight"] = 1.0
    DEFAULT_CFG.depth_aware = False
    DEFAULT_CFG_DICT["depth_aware"] = False
    DEFAULT_CFG.alpha = 1
    DEFAULT_CFG_DICT["alpha"] = 1
    DEFAULT_CFG.depth_inverse = False
    DEFAULT_CFG_DICT["depth_inverse"] = False
    DEFAULT_CFG.depth_threshold = 0.5
    DEFAULT_CFG_DICT["depth_threshold"] = 0.5


def load_config(config_file: str) -> dict:
    with open(config_file) as f:
        return yaml.safe_load(f)


def log_config(config: dict):
    print("### Config:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("###" * 4)


def parse_unknown_args(unknown_args):
    result = {}
    key = None
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg.lstrip("-")
            result[key] = True  # Default to True for flags
        else:
            if key:
                # Try to interpret as int/float/bool, else keep as string
                val = arg
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        with contextlib.suppress(ValueError):
                            val = float(val)
                result[key] = val
                key = None
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, unknown = parser.parse_known_args()
    cli_overrides = parse_unknown_args(unknown)
    main(args.config, cli_overrides)
