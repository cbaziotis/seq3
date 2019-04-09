import math
from itertools import groupby

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from utils.training import load_checkpoint


def compress_seq3(checkpoint, src_file, out_file,
                  device, verbose=False, mode="attention"):
    checkpoint = load_checkpoint(checkpoint)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    def giga_tokenizer(x):
        return x.strip().lower().split()

    dataset = AEDataset(src_file,
                        preprocess=giga_tokenizer,
                        vocab=checkpoint["vocab"],
                        seq_len=config["data"]["seq_len"],
                        return_oov=True,
                        oovs=config["data"]["oovs"])

    data_loader = DataLoader(dataset, batch_size=config["batch_size"],
                             num_workers=0, collate_fn=Seq2SeqOOVCollate())
    n_tokens = len(dataset.vocab)
    model = Seq2Seq2Seq(n_tokens, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ##############################################

    n_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)

    if verbose:
        iterator = tqdm(enumerate(data_loader, 1), total=n_batches)
    else:
        iterator = enumerate(data_loader, 1)

    def devect(ids, oov, strip_eos, pp):
        return devectorize(ids.tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                           strip_eos=strip_eos, oov_map=oov, pp=pp)

    def id2txt(ids, oov=None, lengths=None, strip_eos=True):
        if lengths:
            return [" ".join(x[:l]) for l, x in
                    zip(lengths, devect(ids, oov, strip_eos, pp=True))]
        else:
            return [" ".join(x) for x in devect(ids, oov, strip_eos, pp=True)]

    results = []
    with open(out_file, "w") as f:
        with torch.no_grad():
            for i, batch in iterator:
                batch_oov_map = batch[-1]
                batch = batch[:-1]

                batch = list(map(lambda x: x.to(device), batch))
                (inp_src, out_src, inp_trg, out_trg,
                 src_lengths, trg_lengths) = batch

                trg_lengths = torch.clamp(src_lengths / 2, min=5, max=30) + 1

                #############################################################
                # Debug
                #############################################################
                if mode in ["attention", "debug"]:

                    outputs = model(inp_src, inp_trg, src_lengths, trg_lengths,
                                    sampling=0)
                    enc1, dec1, enc2, dec2 = outputs

                    if mode == "debug":

                        src = id2txt(inp_src)
                        latent = id2txt(dec1[3].max(-1)[1])
                        rec = id2txt(dec2[0].max(-1)[1])

                        _results = list(zip(src, latent, rec))

                        for sample in _results:
                            f.write("\n".join(sample) + "\n\n")

                    elif mode == "attention":
                        src = devect(inp_src, None, strip_eos=False, pp=False)
                        latent = devect(dec1[3].max(-1)[1],
                                        None, strip_eos=False, pp=False)
                        rec = devect(dec2[0].max(-1)[1],
                                     None, strip_eos=False, pp=False)

                        _results = [src, latent, dec1[4], rec, dec2[4]]

                        results += list(zip(*_results))

                        break

                    else:
                        raise ValueError
                else:
                    enc1, dec1 = model.generate(inp_src, src_lengths,
                                                trg_lengths)

                    preds = id2txt(dec1[0].max(-1)[1],
                                   batch_oov_map, trg_lengths.tolist())

                    for sample in preds:
                        f.write(sample + "\n")
    return results


def devectorize(data, id2tok, eos, strip_eos=True, oov_map=None, pp=True):
    if strip_eos:
        for i in range(len(data)):
            try:
                data[i] = data[i][:list(data[i]).index(eos)]
            except:
                continue

    # ids to words
    data = [[id2tok.get(x, "<unk>") for x in seq] for seq in data]

    if oov_map is not None:
        data = [[m.get(x, x) for x in seq] for seq, m in zip(data, oov_map)]

    if pp:
        rules = {f"<oov-{i}>": "UNK" for i in range(10)}
        rules["unk"] = "UNK"
        rules["<unk>"] = "UNK"
        rules["<sos>"] = ""
        rules["<eos>"] = ""
        rules["<pad>"] = ""

        data = [[rules.get(x, x) for x in seq] for seq in data]

        # remove repetitions
        data = [[x[0] for x in groupby(seq)] for seq in data]

    return data
