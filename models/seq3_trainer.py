import torch
from torch.nn import functional as F

from models.seq3_losses import _kl_div, kl_length, pairwise_loss
from models.seq3_utils import sample_lengths
from modules.helpers import sequence_mask, avg_vectors, module_grad_wrt_loss
from modules.training.trainer import Trainer


class Seq3Trainer(Trainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.oracle = kwargs.get("oracle", None)
        self.top = self.config["model"]["top"]
        self.hard = self.config["model"]["hard"]
        self.sampling = self.anneal_init(self.config["model"]["sampling"])
        self.tau = self.anneal_init(self.config["model"]["tau"])
        self.len_min_rt = self.anneal_init(self.config["model"]["min_ratio"])
        self.len_max_rt = self.anneal_init(self.config["model"]["max_ratio"])
        self.len_min = self.anneal_init(self.config["model"]["min_length"])
        self.len_max = self.anneal_init(self.config["model"]["max_length"])

    def _debug_grads(self):
        return list(sorted([(n, p.grad) for n, p in
                            self.model.named_parameters() if p.requires_grad]))

    def _debug_grad_norms(self, reconstruct_loss, prior_loss, topic_loss):
        c_grad_norm = []
        c_grad_norm.append(
            module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                 reconstruct_loss,
                                 "rnn"))

        if self.config["model"]["topic_loss"]:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     topic_loss,
                                     "rnn"))

        if self.config["model"]["prior_loss"] and self.oracle is not None:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     prior_loss,
                                     "rnn"))
        return c_grad_norm

    def _topic_loss(self, inp, dec1, src_lengths, trg_lengths):
        """
        Compute the pairwise distance of various outputs of the seq^3 architecture.
        Args:
            enc1: the outputs of the first encoder (input sequence)
            dec1: the outputs of the first decoder (latent sequence)
            src_lengths: the lengths of the input sequence
            trg_lengths: the lengths of the targer sequence (summary)

        """

        enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
        dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()

        enc_embs = self.model.inp_encoder.embed(inp)
        dec_embs = self.model.compressor.embed.expectation(dec1[3])

        if self.config["model"]["topic_idf"]:
            enc1_energies = self.model.idf(inp)
            # dec1_energies = expected_vecs(dec1[3], self.model.idf.weight)

            x_emb, att_x = avg_vectors(enc_embs, enc_mask, enc1_energies)
            # y_emb, att_y = avg_vectors(dec_reps, dec_mask, dec1_energies)
            y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        else:
            x_emb, att_x = avg_vectors(enc_embs, enc_mask)
            y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        distance = self.config["model"]["topic_distance"]
        loss = pairwise_loss(x_emb, y_emb, distance)

        return loss, (att_x, att_y)

    def _prior_loss(self, outputs, latent_lengths):
        """
        Prior Loss
        Args:
            outputs:
            latent_lengths:

        Returns:

        """
        enc1, dec1, enc2, dec2 = outputs
        _vocab = self._get_vocab()

        logits_dec1, outs_dec1, hn_dec1, dists_dec1, _, _ = dec1

        # dists_dec1 contain the distributions from which
        # the samples were taken. It contains one less element than the logits
        # because the last logit is only used for computing the NLL of EOS.
        words_dec1 = dists_dec1.max(-1)[1]

        # sos + the sampled sentence
        sos_id = _vocab.tok2id[_vocab.SOS]
        sos = torch.zeros_like(words_dec1[:, :1]).fill_(sos_id)
        oracle_inp = torch.cat([sos, words_dec1], -1)

        logits_oracle, _, _ = self.oracle(oracle_inp, None,
                                          latent_lengths)

        prior_loss, prior_loss_time = _kl_div(logits_dec1,
                                              logits_oracle,
                                              latent_lengths)

        return prior_loss, prior_loss_time, logits_oracle

    def _process_batch(self, inp_x, out_x, inp_xhat, out_xhat,
                       x_lengths, xhat_lengths):

        self.model.train()

        tau = self.anneal_step(self.tau)
        sampling = self.anneal_step(self.sampling)
        len_min_rt = self.anneal_step(self.len_min_rt)
        len_max_rt = self.anneal_step(self.len_max_rt)
        len_min = self.anneal_step(self.len_min)
        len_max = self.anneal_step(self.len_max)

        latent_lengths = sample_lengths(x_lengths,
                                        len_min_rt, len_max_rt,
                                        len_min, len_max)

        outputs = self.model(inp_x, inp_xhat,
                             x_lengths, latent_lengths, sampling, tau)

        enc1, dec1, enc2, dec2 = outputs

        batch_outputs = {"model_outputs": outputs}

        # --------------------------------------------------------------
        # 1 - RECONSTRUCTION
        # --------------------------------------------------------------
        # reconstruct_loss = self._seq_loss(dec2[0], out_xhat)

        _dec2_logits = dec2[0].contiguous().view(-1, dec2[0].size(-1))
        _x_labels = out_xhat.contiguous().view(-1)
        reconstruct_loss = F.cross_entropy(_dec2_logits, _x_labels,
                                           ignore_index=0,
                                           reduction='none')

        reconstruct_loss_token = reconstruct_loss.view(out_xhat.size())
        batch_outputs["reconstruction"] = reconstruct_loss_token
        mean_rec_loss = reconstruct_loss.sum() / xhat_lengths.float().sum()
        losses = [mean_rec_loss]

        # --------------------------------------------------------------
        # 2 - PRIOR
        # --------------------------------------------------------------
        if self.config["model"]["prior_loss"] and self.oracle is not None:
            prior_loss, p_loss_i, p_logits = self._prior_loss(outputs,
                                                              latent_lengths)
            batch_outputs["prior"] = p_loss_i, p_logits
            losses.append(prior_loss)
        else:
            prior_loss = None

        # --------------------------------------------------------------
        # 3 - TOPIC
        # --------------------------------------------------------------
        if self.config["model"]["topic_loss"]:
            topic_loss, attentions = self._topic_loss(inp_x, dec1,
                                                      x_lengths,
                                                      latent_lengths)
            batch_outputs["attention"] = attentions
            losses.append(topic_loss)
        else:
            topic_loss = None

        # --------------------------------------------------------------
        # 4 - LENGTH
        # --------------------------------------------------------------
        if self.config["model"]["length_loss"]:
            _vocab = self._get_vocab()
            eos_id = _vocab.tok2id[_vocab.EOS]
            length_loss = kl_length(dec1[0], latent_lengths, eos_id)
            losses.append(length_loss)

        # --------------------------------------------------------------
        # Plot Norms of loss gradient wrt to the compressor
        # --------------------------------------------------------------
        if self.config["plot_norms"] and self.step % self.config[
            "log_interval"] == 0:
            batch_outputs["grad_norm"] = self._debug_grad_norms(
                mean_rec_loss,
                prior_loss,
                topic_loss)

        return losses, batch_outputs

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()

        results = []
        oov_maps = []

        self.len_min_rt = self.anneal_init(
            self.config["model"]["test_min_ratio"])
        self.len_max_rt = self.anneal_init(
            self.config["model"]["test_max_ratio"])
        self.len_min = self.anneal_init(
            self.config["model"]["test_min_length"])
        self.len_max = self.anneal_init(
            self.config["model"]["test_max_length"])

        iterator = self.valid_loader
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                batch_oov_map = batch[-1]
                batch = batch[:-1]

                batch = list(map(lambda x: x.to(self.device), batch))
                (inp_src, out_src, inp_trg, out_trg,
                 src_lengths, trg_lengths) = batch

                latent_lengths = sample_lengths(src_lengths,
                                                self.len_min_rt,
                                                self.len_max_rt, self.len_min,
                                                self.len_max)

                enc, dec = self.model.generate(inp_src, src_lengths,
                                               latent_lengths)

                if dec[3] is not None:
                    results.append(dec[3].max(dim=2)[1])
                else:
                    results.append(dec[0].max(dim=2)[1])

                oov_maps.append(batch_oov_map)

        return results, oov_maps

    def _get_vocab(self):
        if isinstance(self.train_loader, (list, tuple)):
            dataset = self.train_loader[0].dataset
        else:
            dataset = self.train_loader.dataset

        if dataset.subword:
            _vocab = dataset.subword_path
        else:
            _vocab = dataset.vocab

        return _vocab

    def get_state(self):

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "vocab": self._get_vocab(),
        }

        return state
