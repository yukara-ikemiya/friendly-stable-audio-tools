"""
Copyright (C) 2024 Yukara Ikemiya

Reconstruct and save audios using pretrained AutoEncoder models.
This script supports Multi-GPU processing.
"""

import argparse
from pathlib import Path
import json

import torchaudio
from torchaudio import transforms as T
from accelerate import Accelerator

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.autoencoders import AudioAutoencoder
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.utils.torch_common import print_once, get_world_size, get_rank, count_parameters, copy_state_dict
from stable_audio_tools.data.dataset import get_audio_filenames
from stable_audio_tools.data.modification import Mono, Stereo

# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def load_audio(filename: str, sr: int):
    # NOTE: can't load mp3
    ext = filename.split(".")[-1]
    audio, in_sr = torchaudio.load(filename, format=ext)

    if in_sr != sr:
        resample_tf = T.Resample(in_sr, sr)
        audio = resample_tf(audio)

    return audio, in_sr


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model-config', type=str)
    args.add_argument('--ckpt-path', type=str)
    args.add_argument('--audio-dir', type=str)
    args.add_argument('--output-dir', type=str)
    args.add_argument('--frame-duration', type=float)
    args.add_argument('--overlap-rate', type=float)
    args.add_argument('--batch-size', type=int)
    args = args.parse_args()
    return args


def reconstruct_audio():

    args = get_args()

    # config
    model_config = args.model_config
    ckpt_path: str = args.ckpt_path
    audio_dir: str = args.audio_dir
    output_dir: str = args.output_dir
    frame_duration: float = args.frame_duration  # [sec]
    overlap_rate: float = args.overlap_rate  # e.g. 0.01
    batch_size: int = args.batch_size

    # get JSON config from args.model_config and create a model
    with open(model_config) as f:
        model_config = json.load(f)

    # create a model and load pretrained checkpoints
    model: AudioAutoencoder = create_model_from_config(model_config)
    copy_state_dict(model, load_ckpt_state_dict(ckpt_path))

    accel = Accelerator()
    rank = get_rank()
    world_size = get_world_size()
    model = model.to(accel.device)

    # model info
    sr: int = model.sample_rate
    compress_ratio: int = model.downsampling_ratio
    latent_dim: int = model.latent_dim
    in_ch: int = model.in_channels
    out_ch: int = model.out_channels

    chunk_size = int((frame_duration * sr) / compress_ratio)
    overlap = max(int((frame_duration * sr * overlap_rate) / compress_ratio), 1)

    # count parameters
    model.train()
    if accel.is_main_process:
        params_enc = count_parameters(model.encoder)
        params_dec = count_parameters(model.decoder)
        params_bottle = count_parameters(model.bottleneck)
        params_total = params_enc + params_dec + params_bottle

        print("=== Model Info ===")
        print(f"\tSample rate:\t{sr}")
        print(f"\tIn/Out ch:\t{in_ch} / {out_ch}")
        print(f"\tCompression:\t{compress_ratio}")
        print(f"\tLatent dim:\t{latent_dim}")
        print("=== Model parameters ===")
        print(f'Total : {params_total / (10**6)} [million]')
        print(f'\tEncoder :\t{params_enc / (10**6)} [million]')
        print(f'\tDecoder :\t{params_dec / (10**6)} [million]')
        print(f'\tBottleneck :\t{params_bottle / (10**6)} [million]')
        print('')

    # get all audio file paths
    audio_files = get_audio_filenames(audio_dir)

    # make output directory
    output_dir = Path(output_dir)
    if accel.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # split files
    audio_files = audio_files[rank:: world_size]

    # audio pretransform
    preprocess = Mono() if in_ch == 1 else Stereo()

    kwargs = {
        'chunked': True,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'max_batch_size': batch_size
    }

    # execution

    model.eval()
    for i in range(len(audio_files)):
        filepath = audio_files[i]
        # load audio
        audio, in_sr = load_audio(filepath, sr)
        audio = audio.to(accel.device)
        print(f"rank {rank} : {filepath} ({in_sr} -> {sr})")

        audio = preprocess(audio)
        audio = audio.unsqueeze(0)
        rec = model.reconstruct_audio(audio, **kwargs)

        # save audio
        rec = rec.squeeze(0).cpu()

        filename = filepath.split('/')[-1]
        torchaudio.save(f"{output_dir}/{filename}", rec, sample_rate=sr, format='wav')
        torchaudio.save(f"{output_dir}/../original/{filename}", audio.squeeze(0).cpu(), sample_rate=sr, format='wav')

    print(f'[Finished : rank-{get_rank()}]')


if __name__ == "__main__":
    reconstruct_audio()
