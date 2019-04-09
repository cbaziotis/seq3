import argparse
import os
import signal
import subprocess
import sys

import torch

from sys_config import BASE_DIR
from utils.config import load_config


def spawn_visdom():
    try:
        subprocess.run(["visdom  > visdom.txt 2>&1 &"], shell=True)
    except:
        print("Visdom is already running...")

    def signal_handler(signal, frame):
        subprocess.run(["pkill visdom"], shell=True)
        print("Killing Visdom server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name')
    parser.add_argument('--desc')
    parser.add_argument('--resume')
    parser.add_argument('--transfer')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--vocab')
    parser.add_argument('--cp-vocab')
    parser.add_argument('--device', default="auto")
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--source', nargs='*',
                        default=["models", "modules", "utils"])

    args = parser.parse_args()
    config = load_config(args.config)

    if args.name is None:
        config_filename = os.path.basename(args.config)
        args.name = os.path.splitext(config_filename)[0]

    config["name"] = args.name
    config["desc"] = args.desc

    if args.device == "auto":
        args.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

    if args.source is None:
        args.source = []

    args.source = [os.path.join(BASE_DIR, dir) for dir in args.source]

    if args.visdom:
        spawn_visdom()

    for arg in vars(args):
        print("{}:{}".format(arg, getattr(args, arg)))
    print()

    return args, config


def seq2seq2seq_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name')
    parser.add_argument('--desc')
    parser.add_argument('--resume')
    parser.add_argument('--visdom', action='store_true')
    parser.add_argument('--transfer-lm')
    parser.add_argument('--device', default="auto")
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--source', nargs='*',
                        default=["models", "modules", "utils"])

    args = parser.parse_args()
    config = load_config(args.config)

    if args.name is None:
        config_filename = os.path.basename(args.config)
        args.name = os.path.splitext(config_filename)[0]

    config["name"] = args.name
    config["desc"] = args.desc

    if args.device == "auto":
        args.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

    if args.source is None:
        args.source = []

    args.source = [os.path.join(BASE_DIR, dir) for dir in args.source]

    if args.visdom:
        spawn_visdom()

    for arg in vars(args):
        print("{}:{}".format(arg, getattr(args, arg)))
    print()

    return args, config
