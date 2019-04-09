import os

import yaml

from sys_config import DATA_DIR


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-cfg", "--config",
                        dest="cfg",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    return parser


def make_paths(cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            if cfg[key] is not None:
                cfg[key] = os.path.join(DATA_DIR, cfg[key])
                cfg[key] = os.path.abspath(cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths(cfg[key])
    return cfg


def load_config(file):
    with open(file, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths(cfg)

    return cfg
