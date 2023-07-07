import importlib

import yaml
import os


def parse_config(filepath):
    assert os.path.exists(filepath), "file not exist"
    with open(filepath) as fp:
        config = yaml.load(fp, yaml.FullLoader)
    return config


print(parse_config("/home/yeep/project/py/ALSph2d/config/strategy.yml")["URPCMGQuery"])
