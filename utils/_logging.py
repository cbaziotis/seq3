import math
import sys
import time

from tabulate import tabulate


def erase_line():
    sys.stdout.write("\033[K")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return asMinutes(s), asMinutes(rs)


def log_seq3_losses(L1_LM, L1_AE, L2_LM, L2_AE,
                    L1_LMD, L1_TRANSD, L2_LMD, L2_TRANSD,
                    L1_TRANSG, L2_TRANSG):
    losses = []
    losses.append(["L1", L1_LM, L1_AE, math.exp(L1_LM), math.exp(L1_AE),
                   L1_LMD, L1_TRANSD, L1_TRANSG])
    losses.append(["L2", L2_LM, L2_AE, math.exp(L2_LM), math.exp(L2_AE),
                   L2_LMD, L2_TRANSD, L2_TRANSG])
    return tabulate(losses,
                    headers=['Lang', 'LM Loss', 'AE Loss', 'LM PPL', 'AE PPL',
                             'LM-D Loss', 'TRANS-D Loss', 'TRANS-G Loss'],
                    floatfmt=".4f")


def progress_bar(percentage, bar_len=20):
    filled_len = int(round(bar_len * percentage))
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    return "[{}]".format(bar)


def epoch_progress(epoch, batch, batch_size, dataset_size, start):
    n_batches = math.ceil(float(dataset_size) / batch_size)
    percentage = batch / n_batches

    # stats = 'Epoch:{}, Batch:{}/{} ({0:.2f}%)'.format(epoch, batch, n_batches,
    #                                                   percentage)
    stats = f'Epoch:{epoch}, Batch:{batch}/{n_batches} ' \
            f'({100* percentage:.0f}%)'
    # stats = f'Epoch:{epoch}, Batch:{batch} ({100* percentage:.0f}%)'

    elapsed, eta = timeSince(start, batch / n_batches)
    time_info = 'Time: {} (-{})'.format(elapsed, eta)

    # clean every line and then add the text output
    # log_output = stats + " " + progress_bar + ", " + time_info

    # log_output = " ".join([stats, time_info])
    log_output = " ".join([stats, progress_bar(percentage), time_info])

    sys.stdout.write("\r \r\033[K" + log_output)
    sys.stdout.flush()
    return log_output
