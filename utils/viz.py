from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from torch.autograd import Variable


def make_dot_2(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])

    add_nodes(var.creator)
    return dot


def attention_heatmap_subplot(src, trg, attention, ax=None):
    g = sns.heatmap(attention,
                    # cmap="Greys_r",
                    cmap="viridis",
                    cbar=False,
                    # annot=True,
                    vmin=0, vmax=1,
                    robust=False,
                    fmt=".2f",
                    annot_kws={'size': 12},
                    xticklabels=trg,
                    yticklabels=src,
                    # square=True,
                    ax=ax)
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=12)
    g.set_xticklabels(g.get_xticklabels(), rotation=60, fontsize=12)

    # g.set_xticks(numpy.arange(len(src)), src, rotation=0)
    # g.set_yticks(numpy.arange(len(trg)), trg, rotation=60)


def visualize_translations(lang, prefix_trg2src=False):
    for s1, s2, a12, s3, a23 in lang:
        # attention_heatmap(i, o, a[:len(o), :len(i)].t().cpu().numpy())
        if prefix_trg2src:
            s2_enc = ["<sos>"] + s2[:-1]
        else:
            s2_enc = s2
        attention_heatmap_pair(s1, s2, s2_enc, s3,
                               a12.t()[:len(s1), :len(s2)].cpu().numpy(),
                               a23.t()[:len(s2_enc), :len(s3)].cpu().numpy())


def visualize_compression(lang, prefix_trg2src=False):
    for s1, s2, a12, s3, a23 in lang:
        # attention_heatmap(i, o, a[:len(o), :len(i)].t().cpu().numpy())
        if prefix_trg2src:
            s2_enc = ["<sos>"] + s2[:-1]
        else:
            s2_enc = s2
        attention_heatmap_pair(s1, s2, s2_enc, s3,
                               a12.t()[:len(s1), :len(s2)].cpu().numpy(),
                               a23.t()[:len(s2_enc), :len(s3)].cpu().numpy())


def seq3_attentions(sent, file='foo.pdf'):
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
    # rc('text', usetex=True)
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # rc('text', usetex=True)

    with PdfPages(file) as pdf:
        for s1, s2, a12, s3, a23 in sent:
            s1 = s1[:s1.index(".") + 1]
            s12 = s2[:s2.index("<eos>") + 1]
            s23 = s2[:s2.index("<eos>")]
            s3 = s3[:len(s1)]

            att12 = a12.t()[:len(s1), :len(s12)].cpu().numpy()
            att23 = a23.t()[:len(s23), :len(s3)].cpu().numpy()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            attention_heatmap_subplot(s1, s12, att12, ax=ax1)
            attention_heatmap_subplot(s23, s3, att23, ax=ax2)
            ax1.set_title("Source to Compression")
            ax2.set_title("Compression to Reconstruction")
            fig.tight_layout()

            pdf.savefig(fig)


def attention_heatmap(src, trg, attention):
    fig, ax = plt.subplots(figsize=(11, 5))
    attention_heatmap_subplot(src, trg, attention)
    fig.tight_layout()
    plt.show()


def attention_heatmap_pair(s1, s2, s3, s4, att12, att23):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    attention_heatmap_subplot(s1, s2, att12, ax=ax1)
    attention_heatmap_subplot(s3, s4, att23, ax=ax2)
    ax1.set_title("RNN1 -> RNN2")
    ax2.set_title("RNN2 -> RNN3")
    fig.tight_layout()
    plt.show()
