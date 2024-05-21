
import torch

EPS = 1e-8


def is_silence(audio: torch.Tensor, thresh: float = -60.):
    """checks if entire clip is 'silence' below some dB threshold

    Args:
        audio (Tensor): torch tensor of (multichannel) audio
        thresh (float): threshold in dB below which we declare to be silence
    """

    def get_dbmax(audio: torch.Tensor):
        """finds the loudest value in the entire clip and puts that into dB (full scale)"""
        return 20 * torch.log10(torch.flatten(audio.abs()).max() + EPS).item()

    return get_dbmax(audio) < thresh
