import torch
from numpy import mean
from torch.nn import functional as F
from torch.nn.functional import _gumbel_softmax_sample


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def masked_normalization(logits, mask):
    scores = F.softmax(logits, dim=-1)

    # apply the mask - zero out masked timesteps
    masked_scores = scores * mask.float()

    # re-normalize the masked scores
    normed_scores = masked_scores.div(masked_scores.sum(-1, keepdim=True))

    return normed_scores


def masked_mean(vecs, mask):
    masked_vecs = vecs * mask.float()

    mean = masked_vecs.sum(1) / mask.sum(1)

    return mean


def masked_normalization_inf(logits, mask):
    logits.masked_fill_(1 - mask, float('-inf'))
    # energies.masked_fill_(1 - mask, -1e18)

    scores = F.softmax(logits, dim=-1)

    return scores


def expected_vecs(dists, vecs):
    flat_probs = dists.contiguous().view(dists.size(0) * dists.size(1),
                                         dists.size(2))
    flat_embs = flat_probs.mm(vecs)
    embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))
    return embs


def straight_softmax(logits, tau=1, hard=False, target_mask=None):
    y_soft = F.softmax(logits.squeeze() / tau, dim=1)

    if target_mask is not None:
        y_soft = y_soft * target_mask.float()
        y_soft.div(y_soft.sum(-1, keepdim=True))

    if hard:
        shape = logits.size()
        _, k = y_soft.max(-1)
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        y = y_hard - y_soft.detach() + y_soft
        return y
    else:
        return y_soft


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, target_mask=None):
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd

    Returns:
      Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features

    Constraints:

    - Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x num_features``

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)

    if target_mask is not None:
        y_soft = y_soft * target_mask.float()
        y_soft.div(y_soft.sum(-1, keepdim=True))

    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


def avg_vectors(vectors, mask, energies=None):
    if energies is None:
        centroid = masked_mean(vectors, mask)
        return centroid, None

    else:
        masked_scores = energies * mask.float()
        normed_scores = masked_scores.div(masked_scores.sum(1, keepdim=True))
        centroid = (vectors * normed_scores).sum(1)
    return centroid, normed_scores


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def module_grad_wrt_loss(optimizers, module, loss, prefix=None):
    loss.backward(retain_graph=True)

    grad_norms = [(n, p.grad.norm().item())
                  for n, p in module.named_parameters()]

    if prefix is not None:
        grad_norms = [g for g in grad_norms if g[0].startswith(prefix)]

    mean_norm = mean([gn for n, gn in grad_norms])

    for optimizer in optimizers:
        optimizer.zero_grad()

    return mean_norm
