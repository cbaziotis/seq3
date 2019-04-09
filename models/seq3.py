import itertools
import math
import os
import warnings

import numpy
import torch
from tabulate import tabulate
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from generate.utils import devectorize
from models.seq3_trainer import Seq3Trainer
from models.seq3_utils import compute_dataset_idf
from modules.data.collates import Seq2SeqCollate, Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.data.samplers import BucketBatchSampler
from modules.models import Seq2Seq2Seq
from modules.modules import SeqReader
from mylogger.attention import samples2html
from mylogger.experiment import Experiment
from sys_config import EXP_DIR, EMBS_PATH, MODEL_CNF_DIR
from utils.eval import rouge_file_list, pprint_rouge_scores
from utils.generic import number_h
from utils.opts import seq2seq2seq_options
from utils.training import load_checkpoint
from utils.transfer import freeze_module

####################################################################
# Settings
####################################################################
opts, config = seq2seq2seq_options()

####################################################################
#
# Weight Transfer
#
####################################################################
vocab = None

if config["model"]["prior_loss"] and config["prior"] is not None:
    print("Loading Oracle LM ...")
    oracle_cp = load_checkpoint(config["prior"])
    vocab = oracle_cp["vocab"]

    oracle = SeqReader(len(vocab), **oracle_cp["config"]["model"])
    oracle.load_state_dict(oracle_cp["model"])
    oracle.to(opts.device)
    freeze_module(oracle)
else:
    oracle = None


####################################################################
#
# Data Loading and Preprocessing
#
####################################################################
def giga_tokenizer(x):
    return x.strip().lower().split()


print("Building training dataset...")
train_data = AEDataset(config["data"]["train_path"],
                       preprocess=giga_tokenizer,
                       vocab=vocab,
                       vocab_size=config["vocab"]["size"],
                       seq_len=config["data"]["seq_len"],
                       oovs=config["data"]["oovs"],
                       swaps=config["data"]["swaps"])

print("Building validation dataset...")
val_data = AEDataset(config["data"]["val_path"],
                     preprocess=giga_tokenizer,
                     vocab=vocab,
                     vocab_size=config["vocab"]["size"],
                     seq_len=config["data"]["seq_len"],
                     return_oov=True,
                     oovs=config["data"]["oovs"])

val_data.vocab = train_data.vocab
vocab = train_data.vocab

# define a dataloader, which handles the way a dataset will be loaded,
# like batching, shuffling and so on ...
train_lengths = [len(x) for x in train_data.data]

train_sampler = BucketBatchSampler(train_lengths, config["batch_size"])
train_loader = DataLoader(train_data, batch_sampler=train_sampler,
                          num_workers=config["num_workers"],
                          collate_fn=Seq2SeqCollate())
val_loader = DataLoader(val_data, batch_size=config["batch_size"],
                        num_workers=config["num_workers"], shuffle=False,
                        collate_fn=Seq2SeqOOVCollate())

####################################################################
#
# Model Definition
# - additional layer initializations
# - weight / layer tying
#
####################################################################

# Define the model
n_tokens = len(train_data.vocab)
model = Seq2Seq2Seq(n_tokens, **config["model"])
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Load Pretrained Word Embeddings
if "embeddings" in config["vocab"] and config["vocab"]["embeddings"]:
    emb_file = os.path.join(EMBS_PATH, config["vocab"]["embeddings"])
    dims = config["vocab"]["embeddings_dim"]

    embs, emb_mask, missing = train_data.vocab.read_embeddings(emb_file, dims)
    model.initialize_embeddings(embs, config["model"]["embed_trainable"])

    # initialize the output layers with the pretrained embeddings,
    # regardless of whether they will be tied
    try:
        model.compressor.Wo.weight.data.copy_(torch.from_numpy(embs))
        model.decompressor.Wo.weight.data.copy_(torch.from_numpy(embs))
    except:
        print("Can't init outputs from embeddings. Dim mismatch!")

    if config["model"]["embed_masked"] and config["model"]["embed_trainable"]:
        model.set_embedding_gradient_mask(emb_mask)

if config["model"]["topic_loss"] and config["model"]["topic_idf"]:
    print("Computing IDF values...")
    idf = compute_dataset_idf(train_data, train_data.vocab.tok2id)
    # idf[vocab.tok2id[vocab.SOS]] = 1  # neutralize padding token
    # idf[vocab.tok2id[vocab.EOS]] = 1  # neutralize padding token
    idf[vocab.tok2id[vocab.PAD]] = 1  # neutralize padding token
    model.initialize_embeddings_idf(idf)

####################################################################
#
# Tie Models
#
####################################################################
# tie the embedding layers
if config["model"]["tie_embedding"]:
    model.cmp_encoder.embed = model.inp_encoder.embed
    model.compressor.embed = model.inp_encoder.embed
    model.decompressor.embed = model.inp_encoder.embed

# tie the output layers of the decoders
if config["model"]["tie_decoder_outputs"]:
    model.compressor.Wo = model.decompressor.Wo

# tie the embedding to the output layers
if config["model"]["tie_embedding_outputs"]:
    emb_size = model.compressor.embed.embedding.weight.size(1)
    rnn_size = model.compressor.Wo.weight.size(1)

    if emb_size != rnn_size:
        warnings.warn("Can't tie outputs, since emb_size != rnn_size.")
    else:
        model.compressor.Wo.weight = model.inp_encoder.embed.embedding.weight
        model.decompressor.Wo.weight = model.inp_encoder.embed.embedding.weight

if config["model"]["tie_decoders"]:
    model.compressor = model.decompressor

if config["model"]["tie_encoders"]:
    model.cmp_encoder = model.inp_encoder

# then we need only one bridge
if config["model"]["tie_encoders"] and config["model"]["tie_decoders"]:
    model.src_bridge = model.trg_bridge

####################################################################
#
# Experiment Logging and Visualization
#
####################################################################

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters,
                             lr=config["lr"],
                             weight_decay=config["weight_decay"])

model.to(opts.device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters()
                             if p.requires_grad)

print("Total Params:", number_h(total_params))
print("Total Trainable Params:", number_h(total_trainable_params))
trainable_params = sorted([[n] for n, p in model.named_parameters()
                           if p.requires_grad])

####################################################################
#
# Experiment Logging and Visualization
#
####################################################################
if config["prior"] is not None:
    opts.name += "_" + config["prior"]

exp = Experiment(opts.name, config, src_dirs=opts.source, output_dir=EXP_DIR)

step_tags = []
step_tags.append("REC")

if config["model"]["prior_loss"] and config["prior"] is not None:
    step_tags.append("PRIOR")
if config["model"]["topic_loss"]:
    step_tags.append("TOPIC")
if config["model"]["length_loss"]:
    step_tags.append("LENGTH")

exp.add_metric("loss", "line", tags=step_tags)
exp.add_metric("ppl", "line", title="perplexity", tags=step_tags)
exp.add_metric("rouge", "line", title="ROUGE (F1)", tags=["R-1", "R-2", "R-L"])
exp.add_value("grads", "text", title="gradients")

exp.add_metric("c_norm", "line", title="Compressor Grad Norms",
               tags=step_tags[:len(set(step_tags) & {"PRIOR", "TOPIC"}) + 1])
exp.add_value("progress", "text", title="training progress")
exp.add_value("epoch", "text", title="epoch summary")
exp.add_value("samples", "text", title="Samples")
exp.get_value("samples").pre = False
exp.add_value("weights", "text")
exp.add_value("rouge-stats", "text")
exp.add_value("states", "scatter")
exp.add_metric("lr", "line", "Learning Rate")
exp.add_value("rouge-stats", "text")


####################################################################
#
# Training Pipeline
# - batch/epoch callbacks for logging, checkpoints, visualization...
# - initialize trainer
# - initialize training loop
#
####################################################################
def stats_callback(batch, losses, loss_list, batch_outputs):
    if trainer.step % config["log_interval"] == 0:

        # log gradient norms
        grads = sorted(trainer.grads(), key=lambda tup: tup[1], reverse=True)
        grads_table = tabulate(grads, numalign="right", floatfmt=".5f",
                               headers=['Parameter', 'Grad(Norm)'])
        exp.update_value("grads", grads_table)

        _losses = losses[-config["log_interval"]:]
        mono_losses = numpy.array([x[:len(step_tags)]
                                   for x in _losses]).mean(0)
        for loss, tag in zip(mono_losses, step_tags):
            exp.update_metric("loss", loss, tag)
            exp.update_metric("ppl", math.exp(loss), tag)

        ################################################

        losses_log = exp.log_metrics(["loss", "ppl"])
        exp.update_value("progress", trainer.progress_log + "\n" + losses_log)

        # clean lines and move cursor back up N lines
        print("\n\033[K" + losses_log)
        print("\033[F" * (len(losses_log.split("\n")) + 2))


def samples_to_text(tensor):
    return devectorize(tensor.tolist(), train_data.vocab.id2tok,
                       train_data.vocab.tok2id[vocab.EOS],
                       strip_eos=False, pp=False)


def outs_callback(batch, losses, loss_list, batch_outputs):
    if trainer.step % config["log_interval"] == 0:
        enc1, dec1, enc2, dec2 = batch_outputs['model_outputs']

        if config["plot_norms"]:
            norms = batch_outputs['grad_norm']
            exp.update_metric("c_norm", norms[0], "REC")

            if "TOPIC" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["topic"]], "TOPIC")

            if "PRIOR" in step_tags:
                exp.update_metric("c_norm", norms[loss_ids["prior"]], "PRIOR")

        if len(batch) == 2:
            inp = batch[0][0]
        else:
            inp = batch[0]
        src = samples_to_text(inp)
        hyp = samples_to_text(dec1[3].max(dim=2)[1])
        rec = samples_to_text(dec2[0].max(dim=2)[1])

        # prior outputs
        if "prior" in batch_outputs:
            prior_loss = batch_outputs['prior'][0].squeeze().tolist()
            prior_logits = batch_outputs['prior'][1]

            prior_argmax = prior_logits.max(dim=2)[1]
            prior_entropy = Categorical(logits=prior_logits).entropy().tolist()

            prior = samples_to_text(prior_argmax)

        if "attention" in batch_outputs:
            att_scores = batch_outputs['attention'][0].squeeze().tolist()
        else:
            att_scores = None

        if config["model"]["learn_tau"]:
            temps = dec1[5].cpu().data.numpy().round(2)
        else:
            temps = None

        rec_losses = batch_outputs['reconstruction'].tolist()

        samples = []
        for i in range(len(src)):
            sample = []

            if att_scores is not None:
                _src = 'SRC', (src[i], att_scores[i]), "255, 0, 0"
            else:
                _src = 'SRC', src[i], "0, 0, 0"
            sample.append(_src)

            if "prior" in batch_outputs:
                _hyp = 'HYP', (hyp[i], prior_loss[i]), "0, 0, 255"
                _pri = 'LM ', (prior[i], prior_entropy[i]), "0, 255, 0"
                sample.append(_hyp)
                sample.append(_pri)
            else:
                _hyp = 'HYP', hyp[i], "0, 0, 255"
                sample.append(_hyp)

            if temps is not None:
                _tmp = 'TMP', (list(map(str, temps[i])), temps[i]), "255, 0, 0"
                sample.append(_tmp)

            _rec = 'REC', (rec[i], rec_losses[i]), "255, 0, 0"
            sample.append(_rec)

            samples.append(sample)

        html_samples = samples2html(samples)
        exp.update_value("samples", html_samples)
        with open(os.path.join(EXP_DIR, f"{opts.name}.samples.html"),
                  'w') as f:
            f.write(html_samples)


def eval_callback(batch, losses, loss_list, batch_outputs):
    if trainer.step % config["checkpoint_interval"] == 0:
        tags = [trainer.epoch, trainer.step]
        trainer.checkpoint(name=opts.name, tags=tags)
        exp.save()

    if trainer.step % config["eval_interval"] == 0:
        preds, oov_maps = trainer.eval_epoch()
        preds = [x.tolist() for x in preds]
        preds = list(itertools.chain.from_iterable(preds))
        oov_maps = list(itertools.chain.from_iterable(oov_maps))

        v = train_data.vocab
        tokens = devectorize(preds, v.id2tok, v.tok2id[v.EOS], True, oov_maps)
        hyps = [" ".join(x) for x in tokens]
        scores = rouge_file_list(config["data"]["ref_path"], hyps)

        rouge_table = pprint_rouge_scores(scores)
        exp.update_value("rouge-stats", rouge_table)
        exp.update_metric("rouge", scores['rouge-1']['f'], "R-1")
        exp.update_metric("rouge", scores['rouge-2']['f'], "R-2")
        exp.update_metric("rouge", scores['rouge-l']['f'], "R-L")

        save_best()


####################################################################
# Loss Weight: order matters!
####################################################################
loss_ids = {}

loss_weights = [config["model"]["loss_weight_reconstruction"]]
loss_ids["reconstruction"] = len(loss_weights) - 1
if config["model"]["prior_loss"] and config["prior"] is not None:
    loss_weights.append(config["model"]["loss_weight_prior"])
    loss_ids["prior"] = len(loss_weights) - 1
if config["model"]["topic_loss"]:
    loss_weights.append(config["model"]["loss_weight_topic"])
    loss_ids["topic"] = len(loss_weights) - 1
if config["model"]["length_loss"]:
    loss_weights.append(config["model"]["loss_weight_length"])
    loss_ids["length"] = len(loss_weights) - 1

trainer = Seq3Trainer(model, train_loader, val_loader,
                      criterion, optimizer, config, opts.device,
                      batch_end_callbacks=[stats_callback,
                                           outs_callback,
                                           eval_callback],
                      loss_weights=loss_weights, oracle=oracle)

####################################################################
# Training Loop
####################################################################

assert not train_data.vocab.is_corrupt()
assert not val_data.vocab.is_corrupt()

best_score = None


def save_best():
    global best_score
    _score = exp.get_metric("rouge").values["R-2"][-1]
    if not best_score or _score > best_score:
        best_score = _score
        trainer.checkpoint()
    exp.save()


for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch()

    # Save the model if the validation loss is the best we've seen so far.
    save_best()
