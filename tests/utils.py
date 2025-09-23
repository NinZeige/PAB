import yaml
from pathlib import Path


def load_config():
    CONFIG_FILE = Path('./config/mobile-clip2.yaml')
    with open(CONFIG_FILE) as f:
        cfg: dict[str, str | list[str] | int | dict] = yaml.load(f, Loader=yaml.Loader)
    return cfg
