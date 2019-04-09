import numpy
from visdom import Visdom
import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:

    def __init__(self, env="main",
                 server="http://localhost",
                 port=8097,
                 base_url="/",
                 http_proxy_host=None,
                 http_proxy_port=None,
                 log_to_filename=None):
        self._viz = Visdom(env=env,
                           server=server,
                           port=port,
                           http_proxy_host=http_proxy_host,
                           http_proxy_port=http_proxy_port,
                           log_to_filename=log_to_filename,
                           use_incoming_socket=False)
        self._viz.close(env=env)

    def plot_line(self, values, steps, name, legend=None):
        if legend is None:
            opts = dict(title=name)
        else:
            opts = dict(title=name, legend=legend)

        self._viz.line(
            X=numpy.column_stack(steps),
            Y=numpy.column_stack(values),
            win=name,
            update='append',
            opts=opts
        )

    def plot_text(self, text, title, pre=True):
        _width = max([len(x) for x in text.split("\n")]) * 10
        _heigth = len(text.split("\n")) * 20
        _heigth = max(_heigth, 120)
        if pre:
            text = "<pre>{}</pre>".format(text)

        self._viz.text(text, win=title, opts=dict(title=title,
                                                  width=min(_width, 400),
                                                  height=min(_heigth, 400)))

    def plot_bar(self, data, labels, title):
        self._viz.bar(win=title, X=data,
                      opts=dict(legend=labels, stacked=False, title=title))

    def plot_scatter(self, data, labels, title):
        X = numpy.concatenate(data, axis=0)
        Y = numpy.concatenate([numpy.full(len(d), i)
                               for i, d in enumerate(data, 1)], axis=0)
        self._viz.scatter(win=title, X=X, Y=Y,
                          opts=dict(legend=labels, title=title,
                                    markersize=5,
                                    webgl=True,
                                    width=400,
                                    height=400,
                                    markeropacity=0.5))

    def plot_heatmap(self, data, labels, title):
        self._viz.heatmap(win=title,
                          X=data,
                          opts=dict(
                              title=title,
                              columnnames=labels[1],
                              rownames=labels[0],
                              width=700,
                              height=700,
                              layoutopts={'plotly': {
                                  'xaxis': {
                                      'side': 'top',
                                      'tickangle': -60,
                                      # 'autorange': "reversed"
                                  },
                                  'yaxis': {
                                      'autorange': "reversed"
                                  },
                              }
                              }
                          ))
