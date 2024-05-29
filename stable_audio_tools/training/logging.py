"""
Copyright (C) 2024 Yukara Ikemiya

Convenient modules for logging metrics.
"""

import typing as tp
import time

import torch


class MetricsLogger:
    def __init__(self):
        self.counts = {}
        self.metrics = {}

    def add(self, metrics: tp.Dict[str, torch.Tensor]) -> None:
        for k, v in metrics.items():
            if k in self.counts.keys():
                self.counts[k] += 1
                self.metrics[k] += v
            else:
                self.counts[k] = 1
                self.metrics[k] = v

    def pop(self, mean: bool = True) -> tp.Dict[str, torch.Tensor]:
        metrics = {}
        for k, v in self.metrics.items():
            metrics[k] = v / self.counts[k] if mean else v

        # reset
        self.counts = {}
        self.metrics = {}

        return metrics


class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed * 1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep
