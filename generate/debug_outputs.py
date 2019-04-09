import os

import torch

from sys_config import DATA_DIR, BASE_DIR
from generate.utils import compress_seq3
from utils.viz import seq3_attentions

checkpoint = "seq3.full"
seed = 1
device = "cpu"
verbose = True
out_file = ""
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
#
# src_file = os.path.join(DATA_DIR, "gigaword/test_1951/input.txt")
# out_file = os.path.join(DATA_DIR, "gigaword/test_1951/preds.txt")

# src_file = os.path.join(DATA_DIR, "gigaword/test_1951/input_min8.txt")
src_file = os.path.join(DATA_DIR, "gigaword/small/valid.article.filter.4K.txt")
# src_file = os.path.join(DATA_DIR, "gigaword/dev/valid.src.small.txt")
# src_file = os.path.join(BASE_DIR, "evaluation/DUC2003/input.txt")
# src_file = os.path.join(BASE_DIR, "evaluation/DUC2004/input.txt")

out_file = os.path.join(BASE_DIR, f"evaluation/{checkpoint}_preds.txt")
results = compress_seq3(checkpoint, src_file, out_file, device, True,
                        mode="debug")

# seq3_attentions(results[:15], file=checkpoint + ".pdf")
