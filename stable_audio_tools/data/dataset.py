import importlib
import io
import os
import posixpath
import random
import re
import subprocess
import time
import typing as tp

import torch
from torchaudio import transforms as T
import torchaudio
import numpy as np
from pedalboard.io import AudioFile
import webdataset as wds

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from .audio_utils import is_silence
from stable_audio_tools.utils.torch_common import print_once

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")


def fast_scandir(dir: str, ext: tp.List[str], keywords: tp.Optional[tp.List[str]] = None):
    """ Very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243

    fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

    Args:
        dir (str): top-level directory at which to begin scanning.
        ext (tp.List[str]): list of allowed file extensions.
        keywords (tp.List[str]): list of keywords to search for in the file name
    """
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords] if keywords else None
    banned_words = ["paxheader", "__macosx"]
    # add starting period to extensions if needed
    ext = ['.' + x if x[0] != '.' else x for x in ext]

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = os.path.basename(f.path).startswith(".")
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    has_keyword = True
                    has_banned = False

                    if keywords:
                        name_lower = f.name.lower()
                        has_keyword = any([keyword in name_lower for keyword in keywords])
                        has_banned = any([banned_word in name_lower for banned_word in banned_words])

                    if has_ext and (not is_hidden) and has_keyword and (not has_banned):
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


def get_audio_filenames(
    paths: tp.List[str],  # directories in which to search
    keywords: tp.Optional[tp.List[str]] = None,
    exts: tp.List[str] = ['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    """recursively get a list of audio filenames"""
    filenames = []
    if isinstance(paths, str):
        paths = [paths]

    # get a list of relevant filenames
    for p in paths:
        _, files = fast_scandir(p, exts, keywords)
        filenames.extend(files)

    return filenames


class AudioFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        sample_size=65536,
        sample_rate=48000,
        keywords=None,
        relpath=None,
        random_crop=True,
        force_channels="stereo",
        custom_metadata_fn: tp.Optional[tp.Callable[[str], str]] = None,
        # augmentation
        augment_phase: bool = True
    ):
        assert force_channels in ['mono', 'stereo']

        super().__init__()
        self.filenames = []
        self.relpath = relpath
        self.sr = sample_rate
        self.force_channels = force_channels
        self.custom_metadata_fn = custom_metadata_fn

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper() if augment_phase else torch.nn.Identity()
        )

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        print_once('->-> searching audio files...')
        self.filenames = get_audio_filenames(paths, keywords)
        print_once(f'->-> Found {len(self.filenames)} files.')

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Encode the file to assist in prediction
            audio = self.encoding(audio)

            # Audio augmentations
            audio = self.augs(audio)
            audio = audio.clamp(-1, 1)

            info = {"path": audio_filename}
            if self.relpath:
                info["relpath"] = os.path.relpath(audio_filename, self.relpath)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask

            end_time = time.time()

            info["load_time"] = end_time - start_time

            if self.custom_metadata_fn is not None:
                custom_metadata = self.custom_metadata_fn(info, audio)
                info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]


def group_by_keys(data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if wds.tariterators.trace:
            print_once(prefix, suffix, current_sample.keys() if isinstance(current_sample, dict) else None)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            print_once(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample


wds.tariterators.group_by_keys = group_by_keys

# S3 code and WDS preprocessing code based on implementation
# by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py


def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    cmd = ['aws', 's3', 'ls', bucket_path]

    if profile is not None:
        cmd.extend(['--profile', profile])

    if recursive:
        # Add the --recursive flag if requested
        cmd.append('--recursive')

    # Run the `aws s3 ls` command and capture the output
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    contents = [posixpath.join(s3_url_prefix or '', x)
                for x in contents if not x.endswith('/')]

    # Apply the filter, if specified
    if filter:
        contents = [x for x in contents if filter in x]

    # Remove redundant directory names in the S3 URL
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace(
            '//', '/') for x in contents]

    # Print debugging information, if requested
    if debug:
        print_once("contents = \n", contents)

    return contents


def get_all_s3_urls(
    names=[],           # list of all valid [LAION AudioDataset] dataset names
    # list of subsets you want from those datasets, e.g. ['train','valid']
    subsets=[''],
    s3_url_prefix=None,  # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    # print debugging info -- note: info displayed likely to change at dev's whims
    debug=False,
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            contents_str = posixpath.join(s3_url_prefix, name)

        if debug:
            print_once(f"get_all_s3_urls: {contents_str}:")

        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print_once(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                tar = tar.replace(" ", "\ ").replace(
                    "(", "\(").replace(")", "\)")
                # Construct the S3 path to the current tar file
                s3_path = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"

                if debug:
                    print_once("request_str = ", request_str)

                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    print_once(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_silent = is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]

    return has_json and has_audio and not is_silent and not is_rejected


class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: tp.Optional[tp.Callable[[str], str]] = None,
        profile: tp.Optional[str] = None,
    ):
        self.id = id
        self.s3_path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.s3_path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.s3_path: self.profile} if self.profile else {},
        )

        return self.urls


def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None


def collation_fn(samples):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        else:
            b = b
        result.append(b)
    return result


class S3WebDataLoader():
    def __init__(
        self,
        datasets: tp.List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        **data_loader_kwargs
    ):
        assert force_channels in ['mono', 'stereo']

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels

        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        urls = [url for dataset_urls in urls for url in dataset_urls]

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        ).with_epoch(epoch_steps // num_workers if num_workers > 0 else epoch_steps)

        # Force channel size and augumentation
        self.augs = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
            PhaseFlipper() if augment_phase else torch.nn.Identity()
        )

        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):
        found_key, rewrite_key = '', ''
        # print the all entries in dict
        for k, v in sample.items():
            for akey in AUDIO_KEYS:
                if k.endswith(akey):
                    # to rename long/weird key with its simpler counterpart
                    found_key, rewrite_key = k, akey
                    break
            if '' != found_key:
                break

        # if no audio, None is returned to tell WebDataset to skip this one
        if '' == found_key:
            return None

        audio, in_sr = sample[found_key]
        if in_sr != self.sample_rate:
            resample_tf = T.Resample(in_sr, self.sample_rate)
            audio = resample_tf(audio)

        if self.sample_size is not None:
            # Pad/crop and get the relative timestamp
            pad_crop = PadCrop_Normalized_T(
                self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                audio)
            sample["json"]["seconds_start"] = seconds_start
            sample["json"]["seconds_total"] = seconds_total
            sample["json"]["padding_mask"] = padding_mask
        else:
            t_start, t_end = 0, 1

        # Check if audio is length zero, initialize to a single zero if so
        if audio.shape[-1] == 0:
            audio = torch.zeros(1, 1)

        audio = self.augs(audio)

        sample["json"]["timestamps"] = (t_start, t_end)

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue

            if dataset.s3_path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
            del sample[found_key]

        sample["audio"] = audio

        # Add audio to the metadata as well for conditioning
        sample["json"]["audio"] = audio

        return sample


def create_dataloader_from_config(
        dataset_config,
        batch_size: int,
        sample_size: int,
        sample_rate: int,
        audio_channels: int,
        num_workers: int = 4
):
    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type, "Dataset type must be specified in dataset config"
    assert audio_channels in [1, 2], f"Audio channel must be 1 or 2 -> found {audio_channels}."

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        training_dirs = []

        custom_metadata_fn = None
        custom_metadata_module_path = dataset_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)

            custom_metadata_fn = metadata_module.get_custom_metadata

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            training_dirs.append(audio_dir_path)

        train_set = AudioFolderDataset(
            training_dirs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
            custom_metadata_fn=custom_metadata_fn,
            relpath=training_dirs[0]  # TODO: Make relpath relative to each training dir
        )

        return torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                           num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=True, collate_fn=collation_fn)

    elif dataset_type == "s3":
        dataset_configs = []

        for s3_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = s3_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)

                custom_metadata_fn = metadata_module.get_custom_metadata

            dataset_configs.append(
                S3DatasetConfig(
                    id=s3_config["id"],
                    s3_path=s3_config["s3_path"],
                    custom_metadata_fn=custom_metadata_fn,
                    profile=s3_config.get("profile", None),
                )
            )

        return S3WebDataLoader(
            dataset_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers,
            persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000),
        ).data_loader
