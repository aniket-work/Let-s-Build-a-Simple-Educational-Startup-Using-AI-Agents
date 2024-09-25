import json
import logging
from constants import CONFIG_PATH


def load_config():
    with open(CONFIG_PATH, 'r') as config_file:
        return json.load(config_file)


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)