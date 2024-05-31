import math
import random
import typing as tp

import torch
from torch import nn


# Padding

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output


class PadCrop_Normalized_T(nn.Module):

    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        super().__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> tp.Tuple[torch.Tensor, float, float, int, int, torch.Tensor]:

        n_channels, n_samples = source.shape
        # maximum offset
        max_ofs = max(0, n_samples - self.n_samples)
        # full-track length
        full_length = max_ofs + self.n_samples
        # set random offset if self.randomize is true
        offset = random.randint(0, max_ofs) if (self.randomize and max_ofs) else 0

        # the start and end positions in full-track (0.0 -- 1.0)
        t_start = offset / full_length
        t_end = (offset + self.n_samples) / full_length

        # new chunk
        chunk = source.new_zeros([n_channels, self.n_samples])
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]

        # offset and full length of the original track in seconds (int)
        # NOTE (by Yukara): I'm not sure why these values are rounded to Int,
        #                   but one assumption is that developers might want to reduce complexity of
        #                   embeddings of NumberConditioner by discretizing input values.
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples], device=source.device)
        padding_mask[:min(n_samples, self.n_samples)] = 1

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )


# Channels

class Mono(nn.Module):
    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return torch.mean(x, dim=0, keepdims=True) if len(x.shape) > 1 else x


class Stereo(nn.Module):
    def __call__(self, x: torch.Tensor):
        x_shape = x.shape
        assert len(x_shape) <= 2
        # Check if it's mono
        if len(x_shape) == 1:  # s -> 2, s
            x = x.unsqueeze(0).repeat(2, 1)
        elif len(x_shape) == 2:
            if x_shape[0] == 1:  # 1, s -> 2, s
                x = x.repeat(2, 1)
            elif x_shape[0] > 2:  # ?, s -> 2,s
                x = x[:2, :]

        return x


# Augumentation

class PhaseFlipper(nn.Module):
    """Randomly invert the phase of a signal"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return -x if (random.random() < self.p) else x
