from contextlib import nullcontext
import math
from typing import Literal, Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio import transforms as T
from alias_free_torch import Activation1d
from dac.nn.layers import WNConv1d, WNConvTranspose1d
from einops import rearrange


from ..inference.sampling import sample
from ..inference.utils import prepare_audio
from .blocks import SnakeBeta
from .bottleneck import Bottleneck, DiscreteBottleneck
from .diffusion import ConditionedDiffusionModel, DAU1DCondWrapper, UNet1DCondWrapper, DiTWrapper
from .factory import create_pretransform_from_config, create_bottleneck_from_config
from .pretransforms import Pretransform


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = Activation1d(act)

    return act


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()

        self.dilation = dilation

        act = get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels)

        padding = (dilation * (7 - 1)) // 2

        self.layers = nn.Sequential(
            act,
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=7, dilation=dilation, padding=padding),
            act,
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                     kernel_size=1)
        )

    def forward(self, x):
        res = x

        # Disable checkpoint until tensor mismatch is fixed
        # x = checkpoint(self.layers, x)
        x = self.layers(x)

        return x + res


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()

        act = get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels)

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            act,
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=2 * stride,
                         stride=1,
                         bias=False,
                         padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))

        act = get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels)

        self.layers = nn.Sequential(
            act,
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckEncoder(nn.Module):
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults=[1, 2, 4, 8],
                 strides=[2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
                 ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]

        for i in range(self.depth - 1):
            layers += [EncoderBlock(in_channels=c_mults[i] * channels, out_channels=c_mults[i + 1]
                                    * channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1] * channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self,
                 out_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults=[1, 2, 4, 8],
                 strides=[2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1] * channels, kernel_size=7, padding=3),
        ]

        for i in range(self.depth - 1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i] * channels,
                out_channels=c_mults[i - 1] * channels,
                stride=strides[i - 1],
                use_snake=use_snake,
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
            )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DACEncoderWrapper(nn.Module):
    def __init__(self, in_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Encoder as DACEncoder

        latent_dim = kwargs.pop("latent_dim", None)

        encoder_out_dim = kwargs["d_model"] * (2 ** len(kwargs["strides"]))
        self.encoder = DACEncoder(d_latent=encoder_out_dim, **kwargs)
        self.latent_dim = latent_dim

        # Latent-dim support was added to DAC after this was first written, and implemented differently, so this is for backwards compatibility
        self.proj_out = nn.Conv1d(self.encoder.enc_dim, latent_dim, kernel_size=1) if latent_dim is not None else nn.Identity()

        if in_channels != 1:
            self.encoder.block[0] = WNConv1d(in_channels, kwargs.get("d_model", 64), kernel_size=7, padding=3)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj_out(x)
        return x


class DACDecoderWrapper(nn.Module):
    def __init__(self, latent_dim, out_channels=1, **kwargs):
        super().__init__()

        from dac.model.dac import Decoder as DACDecoder

        self.decoder = DACDecoder(**kwargs, input_channel=latent_dim, d_out=out_channels)

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.decoder(x)


class AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim: int,
        downsampling_ratio: int,
        sample_rate: int,
        io_channels: int = 2,
        bottleneck: Bottleneck = None,
        pretransform: Pretransform = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        soft_clip: bool = False
    ):
        super().__init__()

        self.downsampling_ratio = downsampling_ratio
        self.min_length = self.downsampling_ratio
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = io_channels if (in_channels is None) else in_channels
        self.out_channels = io_channels if (out_channels is None) else out_channels

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.pretransform = pretransform

        self.soft_clip = soft_clip
        self.is_discrete = self.bottleneck is not None and self.bottleneck.is_discrete

    def encode(self, audio, return_info=False, skip_pretransform=False, iterate_batch=False, **kwargs):

        info = {}

        if self.pretransform is not None and not skip_pretransform:
            with nullcontext() if self.pretransform.enable_grad else torch.no_grad():
                if iterate_batch:
                    audios = []
                    for i in range(audio.shape[0]):
                        audios.append(self.pretransform.encode(audio[i:i + 1]))
                    audio = torch.cat(audios, dim=0)
                else:
                    audio = self.pretransform.encode(audio)

        if self.encoder is not None:
            if iterate_batch:
                latents = []
                for i in range(audio.shape[0]):
                    latents.append(self.encoder(audio[i:i + 1]))
                latents = torch.cat(latents, dim=0)
            else:
                latents = self.encoder(audio)
        else:
            latents = audio

        if self.bottleneck is not None:
            # TODO: Add iterate batch logic, needs to merge the info dicts
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)

            info.update(bottleneck_info)

        if return_info:
            return latents, info

        return latents

    def decode(self, latents, iterate_batch=False, **kwargs):

        if self.bottleneck is not None:
            if iterate_batch:
                decoded = []
                for i in range(latents.shape[0]):
                    decoded.append(self.bottleneck.decode(latents[i:i + 1]))
                decoded = torch.cat(decoded, dim=0)
            else:
                latents = self.bottleneck.decode(latents)

        if iterate_batch:
            decoded = []
            for i in range(latents.shape[0]):
                decoded.append(self.decoder(latents[i:i + 1]))
            decoded = torch.cat(decoded, dim=0)
        else:
            decoded = self.decoder(latents, **kwargs)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                if iterate_batch:
                    decodeds = []
                    for i in range(decoded.shape[0]):
                        decodeds.append(self.pretransform.decode(decoded[i:i + 1]))
                    decoded = torch.cat(decodeds, dim=0)
                else:
                    decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    if iterate_batch:
                        decodeds = []
                        for i in range(latents.shape[0]):
                            decodeds.append(self.pretransform.decode(decoded[i:i + 1]))
                        decoded = torch.cat(decodeds, dim=0)
                    else:
                        decoded = self.pretransform.decode(decoded)

        if self.soft_clip:
            decoded = torch.tanh(decoded)

        return decoded

    def decode_tokens(self, tokens, **kwargs):
        '''
        Decode discrete tokens to audio
        Only works with discrete autoencoders
        '''

        assert isinstance(self.bottleneck, DiscreteBottleneck), "decode_tokens only works with discrete autoencoders"

        latents = self.bottleneck.decode_tokens(tokens, **kwargs)

        return self.decode(latents, **kwargs)

    def preprocess_audio_for_encoder(self, audio, in_sr):
        '''
        Preprocess single audio tensor (Channels x Length) to be compatible with the encoder.
        If the model is mono, stereo audio will be converted to mono.
        Audio will be silence-padded to be a multiple of the model's downsampling ratio.
        Audio will be resampled to the model's sample rate. 
        The output will have batch size 1 and be shape (1 x Channels x Length)
        '''
        return self.preprocess_audio_list_for_encoder([audio], [in_sr])

    def preprocess_audio_list_for_encoder(self, audio_list, in_sr_list):
        '''
        Preprocess a [list] of audio (Channels x Length) into a batch tensor to be compatable with the encoder. 
        The audio in that list can be of different lengths and channels. 
        in_sr can be an integer or list. If it's an integer it will be assumed it is the input sample_rate for every audio.
        All audio will be resampled to the model's sample rate. 
        Audio will be silence-padded to the longest length, and further padded to be a multiple of the model's downsampling ratio. 
        If the model is mono, all audio will be converted to mono. 
        The output will be a tensor of shape (Batch x Channels x Length)
        '''
        batch_size = len(audio_list)
        if isinstance(in_sr_list, int):
            in_sr_list = [in_sr_list] * batch_size
        assert len(in_sr_list) == batch_size, "list of sample rates must be the same length of audio_list"
        new_audio = []
        max_length = 0
        # resample & find the max length
        for i in range(batch_size):
            audio = audio_list[i]
            in_sr = in_sr_list[i]
            if len(audio.shape) == 3 and audio.shape[0] == 1:
                # batchsize 1 was given by accident. Just squeeze it.
                audio = audio.squeeze(0)
            elif len(audio.shape) == 1:
                # Mono signal, channel dimension is missing, unsqueeze it in
                audio = audio.unsqueeze(0)
            assert len(audio.shape) == 2, "Audio should be shape (Channels x Length) with no batch dimension"
            # Resample audio
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate).to(audio.device)
                audio = resample_tf(audio)
            new_audio.append(audio)
            if audio.shape[-1] > max_length:
                max_length = audio.shape[-1]
        # Pad every audio to the same length, multiple of model's downsampling ratio
        padded_audio_length = max_length + (self.min_length - (max_length % self.min_length)) % self.min_length
        for i in range(batch_size):
            # Pad it & if necessary, mixdown/duplicate stereo/mono channels to support model
            new_audio[i] = prepare_audio(new_audio[i], in_sr=in_sr, target_sr=in_sr, target_length=padded_audio_length,
                                         target_channels=self.in_channels, device=new_audio[i].device).squeeze(0)
        # convert to tensor
        return torch.stack(new_audio)

    def encode_audio(
        self,
        audio,
        chunked: bool = False,
        chunk_size: int = 128,
        overlap: int = 4,
        max_batch_size: int = 1,
        **kwargs
    ):
        '''
        Encode audios into latents. Audios should already be preprocesed by preprocess_audio_for_encoder.
        If chunked is True, split the audio into chunks of a given maximum size chunk_size, with given overlap.
        Overlap and chunk_size params are both measured in number of latents (not audio samples) 
        # and therefore you likely could use the same values with decode_audio. 
        A overlap of zero will cause discontinuity artefacts. Overlap should be => receptive field size. 
        Every autoencoder will have a different receptive field size, and thus ideal overlap.
        You can determine it empirically by diffing unchunked vs chunked output and looking at maximum diff.
        The final chunk may have a longer overlap in order to keep chunk_size consistent for all chunks.
        Smaller chunk_size uses less memory, but more compute.
        The chunk_size vs memory tradeoff isn't linear, and possibly depends on the GPU and CUDA version
        For example, on a A6000 chunk_size 128 is overall faster than 256 and 512 even though it has more chunks
        '''
        bs, n_ch, sample_length = audio.shape
        compress_ratio = self.downsampling_ratio
        assert n_ch == self.in_channels
        assert sample_length % compress_ratio == 0, 'The audio length must be a multiple of compression ratio.'

        latent_length = sample_length // compress_ratio
        chunk_size_l = chunk_size
        overlap_l = overlap
        hopsize_l = chunk_size - overlap

        # window for cross-fade of latent vectors
        win = torch.bartlett_window(overlap * 2, device=audio.device)

        if not chunked:
            # encode the entire audio in parallel
            return self.encode(audio, **kwargs)
        else:
            # chunked encoding for lower memory consumption

            # converting a unit from latents to samples
            chunk_size *= compress_ratio
            overlap *= compress_ratio
            hopsize = chunk_size - overlap

            # zero padding
            n_chunk = int(math.ceil(sample_length - chunk_size) / hopsize) + 1
            pad_len = chunk_size + hopsize * (n_chunk - 1) - sample_length
            audio = F.pad(audio, (0, pad_len))

            chunks = []
            for i in range(n_chunk):
                head = i * hopsize
                chunk = audio[..., head:head + chunk_size]
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)  # (bs, n_chunk, n_ch, chunk_size)
            chunks = rearrange(chunks, "b n c l -> (b n) c l")

            # batched encoding
            n_iter = int(math.ceil(chunks.shape[0] / max_batch_size))
            zs = []
            for i in range(n_iter):
                head = i * max_batch_size
                chunks_ = chunks[head: head + max_batch_size]
                z_ = self.encode(chunks_)
                zs.append(z_)

            zs = torch.cat(zs, dim=0)
            zs = rearrange(zs, "(b n) c l -> b n c l", b=bs)  # (bs, n_chunk, latent_dim, chank_size_l)

            # cross-fade of latent vectors
            latents = torch.zeros((bs, self.latent_dim, audio.shape[-1] // compress_ratio), device=audio.device)
            for i in range(n_chunk):
                z_ = zs[:, i]
                if i != 0:
                    z_[:, :, :overlap_l] *= win[None, None, :overlap_l]
                if i != n_chunk - 1:
                    z_[:, :, -overlap_l:] *= win[None, None, -overlap_l:]

                head = i * hopsize_l
                latents[head: head + chunk_size_l] += z_

            # fix size
            latents = latents[..., :latent_length]  # (bs, latent_dim, latent_length)

            return latents

    def decode_audio(
        self,
        latents,
        chunked=False,
        chunk_size=128,
        overlap=4,
        max_batch_size: int = 1,
        **kwargs
    ):
        '''
        Decode latents to audio.
        '''
        bs, latent_dim, latent_length = latents.shape
        compress_ratio = self.downsampling_ratio
        assert latent_dim == self.latent_dim

        hopsize = chunk_size - overlap
        chunk_size_s = chunk_size * compress_ratio
        overlap_s = overlap * compress_ratio
        hopsize_s = hopsize * compress_ratio
        sample_length = latent_length * compress_ratio

        # window for cross-fade of audio samples
        win = torch.bartlett_window(overlap_s * 2, device=latents.device)

        if not chunked:
            # decode the entire latent in parallel
            return self.decode(latents, **kwargs)
        else:
            # chunked decoding

            # reflect padding
            n_chunk = int(math.ceil((latent_length - chunk_size) / hopsize)) + 1
            pad_len = chunk_size + hopsize * (n_chunk - 1) - latent_length
            latents = F.pad(latents, (0, pad_len), mode='reflect')

            chunks = []
            for i in range(n_chunk):
                head = i * hopsize
                chunk = latents[..., head: head + chunk_size]
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)
            chunks = rearrange(chunks, "b n c l -> (b n) c l")

            # batched decoding
            n_iter = int(math.ceil(chunks.shape[0] / max_batch_size))
            xs = []
            for i in range(n_iter):
                head = i * max_batch_size
                chunks_ = chunks[head: head + max_batch_size]
                x_ = self.decode(chunks_)
                xs.append(x_)

            xs = torch.cat(xs, dim=0)
            xs = rearrange(xs, "(b n) c l -> b n c l", b=bs)  # (bs, n_chunk, n_ch, chank_size_sample)

            # cross-fade of audio samples
            audios = torch.zeros((bs, xs.shape[2], latents.shape[-1] * compress_ratio), device=latents.device)
            for i in range(n_chunk):
                x_ = xs[:, i]
                if i != 0:
                    x_[:, :, :overlap_s] *= win[None, None, :overlap_s]
                if i != n_chunk - 1:
                    x_[:, :, -overlap_s:] *= win[None, None, -overlap_s:]

                head = i * hopsize_s
                audios[head: head + chunk_size_s] += x_

            # fix size
            audios = audios[..., :sample_length]  # (bs, n_ch, sample_length)

            return audios

    @torch.no_grad()
    def reconstruct_audio(
        self,
        audio,
        chunked: bool = True,
        chunk_size: int = 128,
        overlap: int = 4,
        max_batch_size: int = 1,
        **kwargs
    ):
        '''
        Encode and decode audios at once.
        '''
        bs, n_ch, sample_length = audio.shape
        compress_ratio = self.downsampling_ratio
        assert n_ch == self.in_channels

        # window for cross-fade of audio samples
        overlap_s = overlap * compress_ratio
        win = torch.bartlett_window(overlap_s * 2, device=audio.device)

        if not chunked:
            return self.decode(self.encode(audio, **kwargs), **kwargs)
        else:
            # chunked encoding for lower memory consumption

            # converting a unit from latents to samples
            chunk_size *= compress_ratio
            overlap *= compress_ratio
            hopsize = chunk_size - overlap

            # zero padding
            n_chunk = int(math.ceil(sample_length - chunk_size) / hopsize) + 1
            pad_len = chunk_size + hopsize * n_chunk - sample_length
            audio = F.pad(audio, (0, pad_len))

            chunks = []
            for i in range(n_chunk):
                head = i * hopsize
                chunk = audio[..., head:head + chunk_size]
                chunks.append(chunk)

            chunks = torch.stack(chunks, dim=1)  # (bs, n_chunk, n_ch, chunk_size)
            chunks = rearrange(chunks, "b n c l -> (b n) c l")

            # batched reconstruction
            n_iter = int(math.ceil(chunks.shape[0] / max_batch_size))
            xs = []
            for i in range(n_iter):
                head = i * max_batch_size
                chunks_ = chunks[head: head + max_batch_size]
                x_ = self.decode(self.encode(chunks_))
                xs.append(x_)

            xs = torch.cat(xs, dim=0)
            xs = rearrange(xs, "(b n) c l -> b n c l", b=bs)  # (bs, n_chunk, n_ch, chank_size_sample)

            # cross-fade of audio samples
            audio_rec = torch.zeros((bs, xs.shape[2], audio.shape[-1]), device=audio.device)
            for i in range(n_chunk):
                x_ = xs[:, i]
                if i != 0:
                    x_[:, :, :overlap_s] *= win[None, None, :overlap_s]
                if i != n_chunk - 1:
                    x_[:, :, -overlap_s:] *= win[None, None, -overlap_s:]

                # print(audio_rec.shape, head, chunk_size, x_.shape)
                head = i * hopsize
                audio_rec[:, :, head: head + chunk_size] += x_

            # fix size
            audio_rec = audio_rec[..., :sample_length]  # (bs, n_ch, sample_length)

            return audio_rec


class DiffusionAutoencoder(AudioAutoencoder):
    def __init__(
        self,
        diffusion: ConditionedDiffusionModel,
        diffusion_downsampling_ratio,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.diffusion = diffusion

        self.min_length = self.downsampling_ratio * diffusion_downsampling_ratio

        if self.encoder is not None:
            # Shrink the initial encoder parameters to avoid saturated latents
            with torch.no_grad():
                for param in self.encoder.parameters():
                    param *= 0.5

    def decode(self, latents, steps=100):

        upsampled_length = latents.shape[2] * self.downsampling_ratio

        if self.bottleneck is not None:
            latents = self.bottleneck.decode(latents)

        if self.decoder is not None:
            latents = self.decode(latents)

        # Upsample latents to match diffusion length
        if latents.shape[2] != upsampled_length:
            latents = F.interpolate(latents, size=upsampled_length, mode='nearest')

        noise = torch.randn(latents.shape[0], self.io_channels, upsampled_length, device=latents.device)
        decoded = sample(self.diffusion, noise, steps, 0, input_concat_cond=latents)

        if self.pretransform is not None:
            if self.pretransform.enable_grad:
                decoded = self.pretransform.decode(decoded)
            else:
                with torch.no_grad():
                    decoded = self.pretransform.decode(decoded)

        return decoded

# AE factories


def create_encoder_from_config(encoder_config: Dict[str, Any]):
    encoder_type = encoder_config.get("type", None)
    assert encoder_type is not None, "Encoder type must be specified"

    if encoder_type == "oobleck":
        encoder = OobleckEncoder(**encoder_config["config"])
    elif encoder_type == "seanet":
        from encodec.modules import SEANetEncoder
        seanet_encoder_config = encoder_config["config"]
        # SEANet encoder expects strides in reverse order
        seanet_encoder_config["ratios"] = list(reversed(seanet_encoder_config.get("ratios", [2, 2, 2, 2, 2])))
        encoder = SEANetEncoder(
            **seanet_encoder_config
        )
    elif encoder_type == "dac":
        dac_config = encoder_config["config"]
        encoder = DACEncoderWrapper(**dac_config)
    elif encoder_type == "local_attn":
        from .local_attention import TransformerEncoder1D
        local_attn_config = encoder_config["config"]
        encoder = TransformerEncoder1D(**local_attn_config)
    else:
        raise ValueError(f"Unknown encoder type {encoder_type}")

    requires_grad = encoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder


def create_decoder_from_config(decoder_config: Dict[str, Any]):
    decoder_type = decoder_config.get("type", None)
    assert decoder_type is not None, "Decoder type must be specified"

    if decoder_type == "oobleck":
        decoder = OobleckDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "seanet":
        from encodec.modules import SEANetDecoder

        decoder = SEANetDecoder(
            **decoder_config["config"]
        )
    elif decoder_type == "dac":
        dac_config = decoder_config["config"]

        decoder = DACDecoderWrapper(**dac_config)
    elif decoder_type == "local_attn":
        from .local_attention import TransformerDecoder1D

        local_attn_config = decoder_config["config"]

        decoder = TransformerDecoder1D(
            **local_attn_config
        )
    else:
        raise ValueError(f"Unknown decoder type {decoder_type}")

    requires_grad = decoder_config.get("requires_grad", True)
    if not requires_grad:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder


def create_autoencoder_from_config(config: Dict[str, Any]):

    ae_config = config["model"]

    encoder = create_encoder_from_config(ae_config["encoder"])
    decoder = create_decoder_from_config(ae_config["decoder"])
    bottleneck = ae_config.get("bottleneck", None)

    latent_dim = ae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = ae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = ae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    in_channels = ae_config.get("in_channels", None)
    out_channels = ae_config.get("out_channels", None)

    pretransform = ae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    soft_clip = ae_config["decoder"].get("soft_clip", False)

    return AudioAutoencoder(
        encoder,
        decoder,
        io_channels=io_channels,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        sample_rate=sample_rate,
        bottleneck=bottleneck,
        pretransform=pretransform,
        in_channels=in_channels,
        out_channels=out_channels,
        soft_clip=soft_clip
    )


def create_diffAE_from_config(config: Dict[str, Any]):

    diffae_config = config["model"]

    if "encoder" in diffae_config:
        encoder = create_encoder_from_config(diffae_config["encoder"])
    else:
        encoder = None

    if "decoder" in diffae_config:
        decoder = create_decoder_from_config(diffae_config["decoder"])
    else:
        decoder = None

    diffusion_model_type = diffae_config["diffusion"]["type"]

    if diffusion_model_type == "DAU1d":
        diffusion = DAU1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "adp_1d":
        diffusion = UNet1DCondWrapper(**diffae_config["diffusion"]["config"])
    elif diffusion_model_type == "dit":
        diffusion = DiTWrapper(**diffae_config["diffusion"]["config"])

    latent_dim = diffae_config.get("latent_dim", None)
    assert latent_dim is not None, "latent_dim must be specified in model config"
    downsampling_ratio = diffae_config.get("downsampling_ratio", None)
    assert downsampling_ratio is not None, "downsampling_ratio must be specified in model config"
    io_channels = diffae_config.get("io_channels", None)
    assert io_channels is not None, "io_channels must be specified in model config"
    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "sample_rate must be specified in model config"

    bottleneck = diffae_config.get("bottleneck", None)

    pretransform = diffae_config.get("pretransform", None)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)

    if bottleneck is not None:
        bottleneck = create_bottleneck_from_config(bottleneck)

    diffusion_downsampling_ratio = None,

    if diffusion_model_type == "DAU1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["strides"])
    elif diffusion_model_type == "adp_1d":
        diffusion_downsampling_ratio = np.prod(diffae_config["diffusion"]["config"]["factors"])
    elif diffusion_model_type == "dit":
        diffusion_downsampling_ratio = 1

    return DiffusionAutoencoder(
        encoder=encoder,
        decoder=decoder,
        diffusion=diffusion,
        io_channels=io_channels,
        sample_rate=sample_rate,
        latent_dim=latent_dim,
        downsampling_ratio=downsampling_ratio,
        diffusion_downsampling_ratio=diffusion_downsampling_ratio,
        bottleneck=bottleneck,
        pretransform=pretransform
    )
