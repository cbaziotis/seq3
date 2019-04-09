import os

import torch

print("torch:", torch.__version__)
print("Cuda:", torch.backends.cudnn.cuda)
print("CuDNN:", torch.backends.cudnn.version())

CPU_CORES = 4
RANDOM_SEED = 1618

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CNF_DIR = os.path.join(BASE_DIR, "model_configs")

TRAINED_PATH = os.path.join(BASE_DIR, "checkpoints")

EMBS_PATH = os.path.join(BASE_DIR, "embeddings")

DATA_DIR = os.path.join(BASE_DIR, 'datasets')

EXP_DIR = os.path.join(BASE_DIR, 'experiments')

MODEL_DIRS = ["models", "modules", "utils"]

VIS = {
    "server": "http://localhost",
    "enabled": False,
    "port": 8097,
    "base_url": "/",
    "http_proxy_host": None,
    "http_proxy_port": None,
    "log_to_filename": os.path.join(BASE_DIR, "vis_logger.json")
}
