import numpy
from visdom import Visdom

from mylogger.plotting import plot_line


class Inspector(object):
    """
    Class for inspecting the internals of neural networks
    """

    def __init__(self, model, stats):
        """

        Args:
            model (torch.nn.Module): the PyTorch model
            stats (list): list of stats names. e.g. ["std", "mean"]
        """

        # watch only trainable layers
        self.watched_layers = {}
        for name, module in self.get_watched_modules(model):
            self.watched_layers[name] = {stat: [] for stat in stats}

        self.viz = Visdom()
        self.update_state(model)

    def get_watched_modules(self, model):
        all_modules = []
        for name, module in model.named_modules():
            if len(list(module.parameters())) > 0 and all(
                    param.requires_grad for param in module.parameters()):
                all_modules.append((name, module))

        # filter parent nodes
        fitered_modules = []
        for name, module in all_modules:
            if not any(
                    [(name in n and name is not n) for n, m in all_modules]):
                fitered_modules.append((name, module))

        return fitered_modules

    def plot_layer(self, name, weights):
        self.viz.histogram(X=weights,
                           win=name,
                           opts=dict(title="{} weights dist".format(name),
                                     numbins=40))
        for stat_name, stat_val in self.watched_layers[name].items():
            stat_val.append(getattr(numpy, stat_name)(weights))

            plot_name = "{}-{}".format(name, stat_name)
            plot_line(self.viz, numpy.array(stat_val), plot_name, [plot_name])

    def update_state(self, model):
        gen = (child for child in model.named_modules()
               if child[0] in self.watched_layers)
        for name, layer in gen:
            weights = [param.data.cpu().numpy() for param in
                       layer.parameters()]
            if len(weights) > 0:
                weights = numpy.concatenate([w.ravel() for w in weights])
                self.plot_layer(name, weights)
