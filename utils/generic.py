from itertools import zip_longest

import numpy
import umap
from sklearn.decomposition import PCA


def merge_dicts(a, b):
    a.update({k: v for k, v in b.items() if k in a})
    return a


def number_h(num):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, 'Yi')


def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]

    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.

    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)])


def pairwise(iterable):
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


def concat_multiline_strings(a, b):
    str = []
    for line1, line2 in zip_longest(a.split("\n"), b.split("\n"),
                                    fillvalue=''):
        str.append("\t".join([line1, line2]))

    return "\n".join(str)


def dim_reduce(data_sets, n_components=2, method="PCA"):
    data = numpy.vstack(data_sets)
    splits = numpy.cumsum([0] + [len(x) for x in data_sets])
    if method == "PCA":
        reducer = PCA(random_state=20, n_components=n_components)
        embedding = reducer.fit_transform(data)
    elif method == "UMAP":
        reducer = umap.UMAP(random_state=20,
                            n_components=n_components,
                            min_dist=0.5)
        embedding = reducer.fit_transform(data)
    else:
        reducer_linear = PCA(random_state=20, n_components=50)
        linear_embedding = reducer_linear.fit_transform(data)
        reducer_nonlinear = umap.UMAP(random_state=20,
                                      n_components=n_components,
                                      min_dist=0.5)
        embedding = reducer_nonlinear.fit_transform(linear_embedding)

    return [embedding[start:stop] for start, stop in pairwise(splits)]
