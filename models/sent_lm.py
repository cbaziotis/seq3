import math
import os

import numpy
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, \
    StepLR
from torch.utils.data import DataLoader

from models.sent_lm_trainer import LMTrainer
from modules.data.collates import LMCollate
from modules.data.datasets import SentenceLMDataset
from modules.data.samplers import SortedSampler, BucketBatchSampler
from modules.data.vocab import Vocab
from modules.modules import SeqReader
from mylogger.experiment import Experiment
from sys_config import EXP_DIR, MODEL_CNF_DIR
from utils.generic import number_h
from utils.opts import train_options
from utils.training import load_checkpoint

####################################################################
# SETTINGS
####################################################################
opts, config = train_options()

####################################################################
# Data Loading and Preprocessing
####################################################################

vocab = None

if config["vocab"]["vocab_path"] is not None:
    vocab_path = config["vocab"]["vocab_path"]
    print(f"Loading vocab from '{vocab_path}'...")
    vocab = Vocab()
    vocab.from_file(vocab_path)

if opts.cp_vocab is not None:
    print(f"Loading vocab from checkpoint '{opts.cp_vocab}'...")
    vcp = load_checkpoint(opts.cp_vocab)
    vocab = vcp["vocab"]

if opts.resume:
    checkpoint = load_checkpoint(opts.resume)
    config["vocab"].update(checkpoint["config"]["vocab"])
    if not config["vocab"]["subword"]:
        vocab = checkpoint["vocab"]


def giga_tokenizer(x):
    return x.strip().lower().split()


print("Building training dataset...")
train_set = SentenceLMDataset(config["data"]["train_path"],
                              preprocess=giga_tokenizer,
                              subword=config["vocab"]["subword"],
                              subword_path=config["vocab"]["subword_path"],
                              vocab=vocab,
                              vocab_size=config["vocab"]["size"],
                              seq_len=config["data"]["seq_len"],
                              sos=config["data"]["sos"],
                              oovs=config["data"].get("oovs", 0))

print("Building validation dataset...")
val_set = SentenceLMDataset(config["data"]["val_path"],
                            preprocess=giga_tokenizer,
                            subword=config["vocab"]["subword"],
                            subword_path=config["vocab"]["subword_path"],
                            vocab=train_set.vocab,
                            seq_len=config["data"]["seq_len"],
                            sos=config["data"]["sos"],
                            oovs=config["data"].get("oovs", 0))

src_lengths = [len(x) for x in train_set.data]
val_lengths = [len(x) for x in val_set.data]

train_sampler = BucketBatchSampler(src_lengths, config["batch_size"],
                                   shuffle=True)
val_sampler = SortedSampler(val_lengths)

train_loader = DataLoader(train_set, batch_sampler=train_sampler,
                          num_workers=0, collate_fn=LMCollate())
val_loader = DataLoader(val_set, sampler=val_sampler,
                        batch_size=config["batch_size"],
                        num_workers=0, collate_fn=LMCollate())

####################################################################
# Model
####################################################################
ntokens = len(train_set.vocab)
model = SeqReader(ntokens, **config["model"])

model.to(opts.device)

print(model)

loss_function = nn.CrossEntropyLoss(ignore_index=0)

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters,
                       lr=config["lr"], weight_decay=config["weight_decay"])

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters()
                             if p.requires_grad)

print("Total Params:", number_h(total_params))
print("Total Trainable Params:", number_h(total_trainable_params))


####################################################################
# Training Pipeline
####################################################################

def batch_callback(batch, epoch_losses, batch_losses, outputs):
    if trainer.step % config["log_interval"] == 0 and trainer.step > 100:
        losses = numpy.array(epoch_losses[-config["log_interval"]:]).mean(0)
        exp.update_metric("loss", losses)
        exp.update_metric("ppl", math.exp(losses))

        losses_log = exp.log_metrics(["loss", "ppl"])
        exp.update_value("progress", trainer.progress_log + "\n" + losses_log)

        # clean lines and move cursor back up N lines
        print("\n\033[K" + losses_log)
        print("\033[F" * (len(losses_log.split("\n")) + 2))


# Trainer: responsible for managing the training process
trainer = LMTrainer(model, train_loader, val_loader, loss_function,
                    [optimizer], config, opts.device,
                    batch_end_callbacks=[batch_callback])

if config["scheduler"] == "plateau":
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  patience=5,
                                  factor=config["gamma"],
                                  threshold=0.1,
                                  verbose=True)

elif config["scheduler"] == "cosine":
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=config["epochs"],
                                  eta_min=config["eta_min"])
elif config["scheduler"] == "step":
    scheduler = StepLR(optimizer,
                       step_size=config["step_size"],
                       gamma=config["gamma"])
else:
    scheduler = None

####################################################################
# Experiment: logging and visualizing the training process
####################################################################
exp = Experiment(opts.name, config, src_dirs=opts.source, output_dir=EXP_DIR)
exp.add_metric("loss", "line")
exp.add_metric("ppl", "line", "perplexity")
exp.add_metric("ep_loss", "line", "epoch loss", ["TRAIN", "VAL"])
exp.add_metric("ep_ppl", "line", "epoch perplexity", ["TRAIN", "VAL"])
exp.add_metric("lr", "line", "Learning Rate")
exp.add_value("progress", "text", title="training progress")
exp.add_value("epoch", "text", title="epoch summary")

####################################################################
# Resume Training from a previous checkpoint
####################################################################
if opts.resume:
    print("Resuming training ...")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizers"][0])
    trainer.epoch = checkpoint["epoch"]
    trainer.step = checkpoint["step"]

####################################################################
# Training Loop
####################################################################
best_loss = None
for epoch in range(config["epochs"]):
    train_loss = trainer.train_epoch()
    val_loss = trainer.eval_epoch()

    if config["scheduler"] == "plateau":
        scheduler.step(val_loss)

    elif config["scheduler"] == "cosine":
        scheduler.step()
    elif config["scheduler"] == "step":
        scheduler.step()

    exp.update_metric("lr", optimizer.param_groups[0]['lr'])

    exp.update_metric("ep_loss", train_loss, "TRAIN")
    exp.update_metric("ep_loss", val_loss, "VAL")
    exp.update_metric("ep_ppl", math.exp(train_loss), "TRAIN")
    exp.update_metric("ep_ppl", math.exp(val_loss), "VAL")

    print()
    epoch_log = exp.log_metrics(["ep_loss", "ep_ppl"])
    print(epoch_log)
    exp.update_value("epoch", epoch_log)

    # Save the model if the validation loss is the best we've seen so far.
    if not best_loss or val_loss < best_loss:
        best_loss = val_loss
        trainer.checkpoint()

    print("\n" * 2)

    exp.save()
