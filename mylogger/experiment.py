import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime

from pymongo import MongoClient
from tabulate import tabulate

from mylogger.helpers import dict_to_html, files_to_dict
from mylogger.plotting import Visualizer
from sys_config import VIS, BASE_DIR


class Experiment(object):
    """
    Experiment class
    """

    def __init__(self, name, config, desc=None,
                 output_dir=None,
                 src_dirs=None,
                 use_db=True,
                 db_host="localhost",
                 db_port=27017,
                 db_uri=None,
                 db_name="experiments"):
        """

        Metrics = history of values
        Values = state of values
        Args:
            name:
            config:
            desc:
            output_dir:
            src_dirs:
            use_db:
            db_host:
            db_port:
            db_uri: mongodb://[username:password@]host1[:port1]
            db_name:
        """
        self.name = name
        self.desc = desc
        self.config = config
        self.metrics = defaultdict(Metric)
        self.values = defaultdict(Value)

        self.use_db = use_db
        self.db_host = db_host
        self.db_port = db_port
        self.db_uri = db_uri
        self.db_name = db_name

        # the src files (dirs) to backup
        if src_dirs is not None:
            self.src = files_to_dict(src_dirs)
        else:
            self.src = None

        # the currently running script
        self.src_main = sys.argv[0]

        self.timestamp_start = datetime.now()
        self.timestamp_update = datetime.now()
        self.last_update = time.time()

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = BASE_DIR

        server = VIS["server"]
        port = VIS["port"]
        base_url = VIS["base_url"]
        http_proxy_host = VIS["http_proxy_host"]
        http_proxy_port = VIS["http_proxy_port"]
        self.enabled = VIS["enabled"]
        vis_log_file = os.path.join(self.output_dir, f"{self.name}.vis")

        if self.enabled:
            self.viz = Visualizer(env=name,
                                  server=server,
                                  port=port,
                                  base_url=base_url,
                                  http_proxy_host=http_proxy_host,
                                  http_proxy_port=http_proxy_port,
                                  log_to_filename=vis_log_file)

            self.add_value("config", "text")
            self.update_value("config", dict_to_html(self.config))

        # connect to MongoDB
        if self.use_db and self.enabled:
            if self.db_uri:
                self.db_client = MongoClient(self.db_uri)
            else:
                self.db_client = MongoClient(self.db_host, self.db_port)

            self.db = self.db_client[self.db_name]
            self.db_collection = self.db.experiments
            self.db_record = None

    #############################################################
    # Metric
    #############################################################
    def add_metric(self, key, vis_type, title=None, tags=None):
        """
        Add a new metric to the experiment.
        Metrics hold a history of all the inserted values.
        The last value(s) will be used for presentation (plotting and console)
        Args:
            key (str): the name of the value. This will be used for getting
                a handle of the metric
            vis_type (str): the visualization type
            tags (list): list of tags e.g. ["train_set", "val_set"]
            title (str): used for presentation purposes (figure, console...)

        Returns:

        """
        self.metrics[key] = Metric(key, vis_type, tags, title)

    def get_metric(self, key):
        """
        Returns a handle to the metric with the given key
        Args:
            key:

        Returns:

        """
        return self.metrics[key]

    def update_metric(self, key, value, tag=None):
        """
        Add new value to the given metric
        Args:
            key:
            value:
            tag:

        Returns:

        """
        self.get_metric(key).add(value, tag)

        try:
            if self.enabled:
                self.__plot_metric(key)

        except IndexError as e:
            pass

        except Exception as e:
            print(f"An error occurred while trying to plot metric:{key}")

    def __plot_metric(self, key):

        metric = self.get_metric(key)

        if metric.vis_type == "line":

            if metric.tags is not None:
                x = [[len(metric.values[tag])] for tag in metric.tags]
                y = [[metric.values[tag][-1]] for tag in metric.tags]
            else:
                x = [len(metric.values)]
                y = [metric.values[-1]]
            self.viz.plot_line(y, x, metric.title, metric.tags)

        elif metric.vis_type == "scatter":
            raise NotImplementedError
        elif metric.vis_type == "bar":
            raise NotImplementedError
        else:
            raise NotImplementedError

    #############################################################
    # Value
    #############################################################
    def add_value(self, key, vis_type, title=None, tags=None, init=None):
        self.values[key] = Value(key, vis_type, tags, title)

    def get_value(self, key):
        return self.values[key]

    def update_value(self, key, value, tag=None):
        """
        Update the state of the given value
        Args:
            key:
            value:
            tag:

        Returns:

        """
        self.get_value(key).update(value, tag)

        try:
            if self.enabled:
                self.__plot_value(key)

        except IndexError as e:
            pass

        except Exception as e:
            print(f"An error occurred while trying to plot value:{key}")

    def __plot_value(self, key):
        value = self.get_value(key)

        if value.vis_type == "text":
            self.viz.plot_text(value.value, value.title, pre=value.pre)
        elif value.vis_type == "scatter":
            if value.tags is not None:
                raise NotImplementedError
            else:
                data = value.value

            self.viz.plot_scatter(data[0], data[1], value.title)
        elif value.vis_type == "heatmap":
            if value.tags is not None:
                raise NotImplementedError
            else:
                data = value.value

            self.viz.plot_heatmap(data[0], data[1], value.title)
        elif value.vis_type == "bar":
            if value.tags is not None:
                raise NotImplementedError
            else:
                data = value.value

            self.viz.plot_bar(data[0], data[1], value.title)
        else:
            raise NotImplementedError

    #############################################################
    # Persistence
    #############################################################
    def _state_dict(self):
        omit = ["db", "db_client", "db_collection"]
        state = {k: v for k, v in self.__dict__.items() if k not in omit}

        return state

    def to_db(self):
        self.timestamp_update = datetime.now()
        # record = self._state_dict()

        # todo: avoid this workaround
        record = json.loads(self._serialize())

        if self.db_record is None:
            self.db_record = self.db_collection.insert(record)
        else:
            self.db_collection.replace_one({"_id": self.db_record}, record)

    def _serialize(self):

        data = json.dumps(self._state_dict(),
                          default=lambda o: getattr(o, '__dict__', str(o)))
        return data

    def to_json(self):
        self.timestamp_update = datetime.now()
        name = self.name + "_{}.json".format(self.get_timestamp())
        filename = os.path.join(self.output_dir, name)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self._serialize())

    def get_timestamp(self):
        return self.timestamp_start.strftime("%y-%m-%d_%H:%M:%S")

    def to_pickle(self):
        self.timestamp_update = datetime.now()
        name = self.name + "_{}.pickle".format(self.get_timestamp())
        filename = os.path.join(self.output_dir, name)
        with open(filename, 'wb') as f:
            pickle.dump(self._state_dict(), f)

    def save(self):
        try:
            self.to_pickle()
        except:
            print("Failed to save to pickle...")

        # try:
        #     self.to_json()
        # except:
        #     print("Failed to save to json...")

        # try:
        #     self.to_db()
        # except:
        #     print("Failed to save to db...")

    def log_metrics(self, keys):

        _metrics = [self.metrics[key] for key in keys]
        _tags = _metrics[0].tags
        if _tags is not None:
            values = [[tag] + [metric.values[tag][-1] for metric in _metrics]
                      for tag in _tags]
            headers = ["TAG"] + [metric.title.upper() for metric in _metrics]
        else:
            values = [[metric.values[-1] for metric in _metrics]]
            headers = [metric.title.upper() for metric in _metrics]

        log_output = tabulate(values, headers, floatfmt=".4f")

        return log_output


class Metric(object):
    """
    Metric hold the data of a value of the model that is being monitored

    A Metric object has to have a name,
    a vis_type which defines how it will be visualized
    and a dataset on which it will be attached to.
    """

    def __init__(self, key, vis_type, tags=None, title=None):
        """

        Args:
            key (str): the name of the metric
            vis_type (str): the visualization type
            tags (list): list of tags
            title (str): used for presentation purposes (figure, console...)
        """
        self.key = key
        self.title = title
        self.vis_type = vis_type
        self.tags = tags

        assert vis_type in ["line"]

        if tags is not None:
            self.values = {tag: [] for tag in tags}
        else:
            self.values = []

        if title is None:
            self.title = key

    def add(self, value, tag=None):
        """
        Add a value to the list of values of this metric
        Args:
            value (int, float):
            tag (str):

        Returns:

        """
        if self.tags is not None:
            self.values[tag].append(value)
        else:
            self.values.append(value)


class Value(object):
    """

    """

    def __init__(self, key, vis_type, tags=None, title=None, pre=True):
        """

        Args:
            key (str): the name of the value
            vis_type (str): the visualization type
            tags (list): list of tags
            title (str): used for presentation purposes (figure, console...)
        """
        self.key = key
        self.title = title
        self.vis_type = vis_type
        self.tags = tags
        self.pre = pre

        assert vis_type in ["text", "scatter", "bar", "heatmap"]

        if tags is not None:
            self.value = {tag: [] for tag in tags}
        else:
            self.value = []

        if title is None:
            self.title = key

    def update(self, value, tag=None):
        """
        Update the value
        Args:
            value (int, float):
            tag (str):

        Returns:

        """
        if self.tags is not None:
            self.value[tag] = value
        else:
            self.value = value
