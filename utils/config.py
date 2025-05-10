import argparse
import json
import yaml


def load_config(file_path: str) -> dict:
    """
    Load JSON or YAML config file.
    """
    with open(file_path, 'r') as f:
        if file_path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        elif file_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Config file must be .json or .yaml/.yml")


def parse_args():
    parser = argparse.ArgumentParser(description="gnnsoz training and evaluation")
    parser.add_argument('--config', type=str, help='Path to JSON/YAML config file', required=True)
    return parser.parse_args()