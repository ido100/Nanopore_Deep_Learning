from config import *


class TrainLogger(object):

    def __init__(self):
        self._data = dict()

    def log_metric(self, metric, datapoint):
        if metric not in self._data:
            self.init_metric(metric)
        self._data[metric].append(datapoint)

    def init_metric(self, metric):
        self._data[metric] = []

    def __getitem__(self, metric):
        return self._data[metric]

