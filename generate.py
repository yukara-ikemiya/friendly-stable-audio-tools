"""
Copyright (C) 2024 Yukara Ikemiya

Generate audio samples using a pretrained generative model.
This script supports Multi-GPU processing.
"""

import argparse
import math
import yaml
from pathlib import Path

import torchaudio
from accelerate import Accelerator

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.utils.torch_common import count_parameters, get_rank, get_world_size
from stable_audio_tools.utils.audio_utils import float_to_int16_audio


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--output-dir', type=str, required=True, help="A directory for saving generated audio samples.")
    args.add_argument('--cond-yaml-path', type=str, required=True, help="A path to a YAML file of sample conditions (e.g. text prompt).")
    args.add_argument('--model-name', type=str, default="stabilityai/stable-audio-open-1.0", help="A name of the pretrained model.")
    args.add_argument('--sampler-type', type=str, default="dpmpp-3m-sde", help="A sampler type used for diffusion sampling.")
    args.add_argument('--sample-steps', type=int, default=100, help="Number of steps for diffusion sampling.")
    args.add_argument('--cfg-scale', type=float, default=7.0, help="Classifier-free guidance scale for sampling.")
    args.add_argument('--n-sample-per-cond', type=int, default=1, help="Number of samples per condition.")
    args.add_argument('--batch-size', type=int, default=10, help="Batch size of sampling.")
    args.add_argument('--clip-length', action='store_true', help="Whether to clip generated audio to the specified 'seconds_total'.")
    args = args.parse_args()
    return args


def flatten_dict(d, parent_key='', separator='/', depth=0):
    items = {}
    for k, v in d.items():
        if depth == 0:
            assert isinstance(v, dict) and all([isinstance(v_, dict) for v_ in v.values()])
        new_key = f"{parent_key}{separator}{k}" if parent_key else k
        if isinstance(list(v.values())[0], dict):
            items.update(flatten_dict(v, new_key, separator=separator, depth=depth + 1))
        else:
            assert all([not isinstance(v_, dict) for v_ in v.values()])
            items[new_key] = {k_: v_ for k_, v_ in v.items()}

    return items


def parse_cond_yaml(yaml_path):
    with open(yaml_path, 'r') as yml:
        conds: dict = yaml.safe_load(yml)

    conds: dict = flatten_dict(conds)
    return conds


def main():
    args = get_args()

    # config
    output_dir: str = args.output_dir
    cond_yaml_path: str = args.cond_yaml_path
    model_name: str = args.model_name
    sampler_type: str = args.sampler_type
    sample_steps: int = args.sample_steps
    cfg_scale: float = args.cfg_scale
    n_sample_per_cond: int = args.n_sample_per_cond
    batch_size: int = args.batch_size
    clip_length: bool = args.clip_length

    batch_sample: int = max(batch_size // 2, 1) if cfg_scale != 1.0 else batch_size

    # multi-GPU setting
    accel = Accelerator()
    rank = get_rank()
    world_size = get_world_size()

    # download model
    model, model_config = get_pretrained_model(model_name)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model: ConditionedDiffusionModelWrapper = model.to(accel.device)

    # prepare conditions
    conds = parse_cond_yaml(cond_yaml_path)
    path_full = []
    conds_full = []
    for p, cond in conds.items():
        for idx in range(n_sample_per_cond):
            path_full.append(f"{p}_item-{idx + 1}")
            conds_full.append(cond)

    # print info
    if accel.is_main_process:
        model.train()
        params_model = count_parameters(model.model)
        params_cond = count_parameters(model.conditioner)

        print("=== Model Info ===")
        print(f"\tSample rate:\t{sample_rate}")
        print(f"\tOut channels:\t{model.pretransform.io_channels}")
        print(f"\tSample size:\t{sample_size} ({sample_size / sample_rate:.3f} [sec])")
        print("=== Model parameters ===")
        print(f'\tDiffusion :\t\t{params_model / (10**6):.3f} [million]')
        print(f'\tConditioner :\t{params_cond / (10**6):.3f} [million]')
        print('')

    path_rank = path_full[rank:: world_size]
    conds_rank = conds_full[rank:: world_size]

    # generation
    model.eval()
    n_iter = int(math.ceil(len(conds_rank) / batch_sample))
    for idx in range(n_iter):
        path_i = path_rank[idx * batch_sample: (idx + 1) * batch_sample]
        conds_i = conds_rank[idx * batch_sample: (idx + 1) * batch_sample]

        samples_i = generate_diffusion_cond(
            model,
            steps=sample_steps,
            cfg_scale=cfg_scale,
            conditioning=conds_i,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type=sampler_type,
            device=accel.device,
            disable_tqdm=(rank != 0)
        )

        for idx_n in range(samples_i.shape[0]):
            audio = float_to_int16_audio(samples_i[idx_n])
            if clip_length:
                L = int(conds_i[idx_n]['seconds_total'] * sample_rate)
                audio = audio[:, :L]
            save_path = f"{output_dir}/{path_i[idx_n]}.wav"
            # mkdir
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            # save
            torchaudio.save(save_path, audio, sample_rate)

    print(f"->->-> Rank-{rank}: Finished.")


if __name__ == "__main__":
    main()
