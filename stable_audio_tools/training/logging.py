"""
Copyright (C) 2024 Yukara Ikemiya

Convenient modules for logging metrics.
"""

import typing as tp

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
