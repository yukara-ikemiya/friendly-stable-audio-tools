
import os
import typing as tp
from contextlib import nullcontext

import torch
import torchaudio
import wandb
from safetensors.torch import save_model
from ema_pytorch import EMA
import auraloss
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..models.autoencoders import AudioAutoencoder
from ..models.discriminators import EncodecDiscriminator, OobleckDiscriminator, DACGANLoss
from ..models.bottleneck import (VAEBottleneck, RVQBottleneck, DACRVQBottleneck, DACRVQVAEBottleneck,
                                 RVQVAEBottleneck, WassersteinBottleneck)
from .losses import MultiLoss, AuralossLoss, ValueLoss, L1Loss
from .scheduler import create_optimizer_from_config, create_scheduler_from_config
from .logging import MetricsLogger
from .viz import audio_spectrogram_image, tokens_spectrogram_image, pca_point_cloud


class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        autoencoder: AudioAutoencoder,
        loss_config: dict,
        optimizer_configs: dict,
        lr: float = 1e-4,
        warmup_steps: int = 0,
        encoder_freeze_on_warmup: bool = False,
        sample_rate: int = 48000,
        use_ema: bool = True,
        ema_copy=None,
        force_input_mono: bool = False,
        latent_mask_ratio: float = 0.0,
        teacher_model: tp.Optional[AudioAutoencoder] = None,
        logging_config: dict = {}
    ):
        super().__init__()

        self.automatic_optimization = False

        self.autoencoder = autoencoder
        self.teacher_model = teacher_model

        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.lr = lr
        self.force_input_mono = force_input_mono
        self.latent_mask_ratio = latent_mask_ratio

        self.optimizer_configs = optimizer_configs
        self.loss_config = loss_config

        self.log_every = logging_config.get("log_every", 1)
        self.metrics_logger = MetricsLogger()

        # Spectral reconstruction loss

        stft_loss_args = loss_config['spectral']['config']

        if self.autoencoder.out_channels == 2:
            self.sdstft = auraloss.freq.SumAndDifferenceSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
            self.lrstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        else:
            self.sdstft = auraloss.freq.MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)

        # Discriminator

        if loss_config['discriminator']['type'] == 'oobleck':
            self.discriminator = OobleckDiscriminator(**loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'encodec':
            self.discriminator = EncodecDiscriminator(in_channels=self.autoencoder.out_channels, **loss_config['discriminator']['config'])
        elif loss_config['discriminator']['type'] == 'dac':
            self.discriminator = DACGANLoss(channels=self.autoencoder.out_channels, sample_rate=sample_rate, **loss_config['discriminator']['config'])

        self.gen_loss_modules = []

        # Adversarial and feature matching losses

        self.gen_loss_modules += [
            ValueLoss(key='loss_adv', weight=self.loss_config['discriminator']['weights']['adversarial'], name='loss_adv'),
            ValueLoss(key='feature_matching_distance', weight=self.loss_config['discriminator']
                      ['weights']['feature_matching'], name='feature_matching'),
        ]

        if self.teacher_model:
            # Distillation losses
            stft_loss_weight = self.loss_config['spectral']['weights']['mrstft'] * 0.25
            self.gen_loss_modules += [
                # Reconstruction loss
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss', weight=stft_loss_weight),
                # Distilled model's decoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'decoded', 'teacher_decoded', name='mrstft_loss_distill', weight=stft_loss_weight),
                # Distilled model's encoder is compatible with teacher's decoder
                AuralossLoss(self.sdstft, 'reals', 'own_latents_teacher_decoded',
                             name='mrstft_loss_own_latents_teacher', weight=stft_loss_weight),
                # Teacher's encoder is compatible with distilled model's decoder
                AuralossLoss(self.sdstft, 'reals', 'teacher_latents_own_decoded',
                             name='mrstft_loss_teacher_latents_own', weight=stft_loss_weight)
            ]
        else:
            # Reconstruction loss
            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss',
                             weight=self.loss_config['spectral']['weights']['mrstft']),
            ]

            if self.autoencoder.out_channels == 2:
                # Add left and right channel reconstruction losses in addition to the sum and difference
                self.gen_loss_modules += [
                    AuralossLoss(self.lrstft, 'reals_left', 'decoded_left', name='stft_loss_left',
                                 weight=self.loss_config['spectral']['weights']['mrstft'] / 2),
                    AuralossLoss(self.lrstft, 'reals_right', 'decoded_right', name='stft_loss_right',
                                 weight=self.loss_config['spectral']['weights']['mrstft'] / 2),
                ]

            self.gen_loss_modules += [
                AuralossLoss(self.sdstft, 'reals', 'decoded', name='mrstft_loss',
                             weight=self.loss_config['spectral']['weights']['mrstft']),
            ]

        if self.loss_config['time']['weights']['l1'] > 0.0:
            self.gen_loss_modules.append(
                L1Loss(key_a='reals', key_b='decoded',
                       weight=self.loss_config['time']['weights']['l1'], name='l1_time_loss')
            )

        if self.autoencoder.bottleneck:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(self.autoencoder.bottleneck, self.loss_config)

        self.losses_gen = MultiLoss(self.gen_loss_modules)

        self.disc_loss_modules = [
            ValueLoss(key='loss_dis', weight=1.0, name='discriminator_loss'),
        ]

        self.losses_disc = MultiLoss(self.disc_loss_modules)

        # Set up EMA for model weights
        self.use_ema = use_ema
        self.autoencoder_ema = EMA(
            self.autoencoder,
            ema_model=ema_copy,
            beta=0.9999,
            power=3 / 4,
            update_every=1,
            update_after_step=1
        ) if use_ema else None

    def configure_optimizers(self):
        opt_gen = create_optimizer_from_config(self.optimizer_configs['autoencoder']['optimizer'], self.autoencoder.parameters())
        opt_disc = create_optimizer_from_config(self.optimizer_configs['discriminator']['optimizer'], self.discriminator.parameters())

        if "scheduler" in self.optimizer_configs['autoencoder'] and "scheduler" in self.optimizer_configs['discriminator']:
            sched_gen = create_scheduler_from_config(self.optimizer_configs['autoencoder']['scheduler'], opt_gen)
            sched_disc = create_scheduler_from_config(self.optimizer_configs['discriminator']['scheduler'], opt_disc)
            return [opt_gen, opt_disc], [sched_gen, sched_disc]

        return [opt_gen, opt_disc]

    def training_step(self, batch, batch_idx):
        reals, _ = batch
        warmed_up: bool = self.warmed_up
        freeze_encoder: bool = warmed_up and self.encoder_freeze_on_warmup
        distilled: bool = self.teacher_model is not None

        # Remove extra dimension added by WebDataset
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        encoder_input = reals.mean(dim=1, keepdim=True) if self.force_input_mono else reals

        with torch.no_grad() if freeze_encoder else nullcontext():
            latents, encoder_info = self.autoencoder.encode(encoder_input, return_info=True)

        loss_info = {"reals": reals, "encoder_input": encoder_input, "latents": latents}
        loss_info.update(encoder_info)

        # Encode with teacher model for distillation
        if distilled:
            with torch.no_grad():
                teacher_latents = self.teacher_model.encode(encoder_input, return_info=False)
                loss_info['teacher_latents'] = teacher_latents

        # Optionally mask out some latents for noise resistance
        if self.latent_mask_ratio > 0.0:
            mask = torch.rand_like(latents) < self.latent_mask_ratio
            latents = torch.where(mask, torch.zeros_like(latents), latents)

        decoded = self.autoencoder.decode(latents)

        loss_info["decoded"] = decoded
        if self.autoencoder.out_channels == 2:
            loss_info["decoded_left"] = decoded[:, 0:1, :]
            loss_info["decoded_right"] = decoded[:, 1:2, :]
            loss_info["reals_left"] = reals[:, 0:1, :]
            loss_info["reals_right"] = reals[:, 1:2, :]

        # Distillation
        if distilled:
            with torch.no_grad():
                teacher_decoded = self.teacher_model.decode(teacher_latents)
                own_latents_teacher_decoded = self.teacher_model.decode(latents)  # Distilled model's latents decoded by teacher
                teacher_latents_own_decoded = self.autoencoder.decode(teacher_latents)  # Teacher's latents decoded by distilled model

            loss_info['teacher_decoded'] = teacher_decoded
            loss_info['own_latents_teacher_decoded'] = own_latents_teacher_decoded
            loss_info['teacher_latents_own_decoded'] = teacher_latents_own_decoded

        if warmed_up:
            loss_dis, loss_adv, feature_matching_distance = self.discriminator.loss(reals, decoded)
        else:
            loss_dis = torch.tensor(0.).to(reals)
            loss_adv = torch.tensor(0.).to(reals)
            feature_matching_distance = torch.tensor(0.).to(reals)

        loss_info["loss_dis"] = loss_dis
        loss_info["loss_adv"] = loss_adv
        loss_info["feature_matching_distance"] = feature_matching_distance

        opt_gen, opt_disc = self.optimizers()
        lr_schedulers = self.lr_schedulers()
        sched_gen, sched_disc = lr_schedulers if lr_schedulers else (None, None)

        # Training step

        training_disc: bool = self.global_step % 2 and warmed_up  # weird behaviour of PyTorch Lightning

        if training_disc:
            # Train the discriminator
            loss, losses = self.losses_disc(loss_info)

            log_dict = {
                'train/disc_lr': opt_disc.param_groups[0]['lr']
            }

            opt_disc.zero_grad()
            self.manual_backward(loss)
            opt_disc.step()

            if sched_disc:
                # scheduler step
                sched_disc.step()
        else:
            # Train the generator
            loss, losses = self.losses_gen(loss_info)

            if self.use_ema:
                self.autoencoder_ema.update()

            opt_gen.zero_grad()
            self.manual_backward(loss)
            opt_gen.step()

            if sched_gen:
                # scheduler step
                sched_gen.step()

            log_dict = {
                'train/loss': loss.detach(),
                'train/latent_std': latents.std().detach(),
                'train/data_std': encoder_input.std().detach(),
                'train/gen_lr': opt_gen.param_groups[0]['lr']
            }

        for loss_name, loss_value in losses.items():
            log_dict[f'train/{loss_name}'] = loss_value.detach()

        self.metrics_logger.add(log_dict)
        if (self.global_step - 1) % self.log_every == 0:
            log_dict = self.metrics_logger.pop()
            self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def export_model(self, path, use_safetensors=False):
        model = self.autoencoder_ema.ema_model if self.autoencoder_ema else self.autoencoder

        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

    @property
    def warmed_up(self) -> bool:
        return self.global_step >= self.warmup_steps


class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        max_num_sample=4,
        sample_size=65536,
        sample_rate=48000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.max_num_sample = max_num_sample
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_reals, _ = next(self.demo_dl)

            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]

            max_num_sample = min(demo_reals.shape[0], self.max_num_sample)
            demo_reals = demo_reals[: max_num_sample]

            demo_reals = demo_reals.to(module.device)
            encoder_input = demo_reals

            if module.force_input_mono:
                encoder_input = encoder_input.mean(dim=1, keepdim=True)

            with torch.no_grad():
                if module.use_ema:
                    latents = module.autoencoder_ema.ema_model.encode(encoder_input)
                    fakes = module.autoencoder_ema.ema_model.decode(latents)
                else:
                    latents = module.autoencoder.encode(encoder_input)
                    fakes = module.autoencoder.decode(latents)

            sample_dir = os.path.join(trainer.default_root_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)

            # Create audio table
            table_recon = wandb.Table(columns=['id', 'target', 'target (spec)', 'recon', 'recon (spec)'])
            for idx in range(max_num_sample):
                target_audio = demo_reals[idx].to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                recon_audio = fakes[idx].to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

                # add audio row
                table_recon.add_data(
                    f'sample_{idx}',
                    wandb.Audio(target_audio.numpy().T, sample_rate=self.sample_rate),
                    wandb.Image(audio_spectrogram_image(target_audio)),
                    wandb.Audio(recon_audio.numpy().T, sample_rate=self.sample_rate),
                    wandb.Image(audio_spectrogram_image(recon_audio))
                )

                # save only the first sample as an audio file
                if idx == 0:
                    torchaudio.save(f'{sample_dir}/{trainer.global_step:08}_target.wav', target_audio, self.sample_rate)
                    torchaudio.save(f'{sample_dir}/{trainer.global_step:08}_recon.wav', recon_audio, self.sample_rate)

            log_dict = {}
            log_dict['reconstruction'] = table_recon
            log_dict['embeddings_3dpca'] = pca_point_cloud(latents)
            log_dict['embeddings_spec'] = wandb.Image(tokens_spectrogram_image(latents))

            trainer.logger.experiment.log(log_dict)
        except Exception as e:
            raise e
        finally:
            module.train()


def create_loss_modules_from_bottleneck(bottleneck, loss_config):
    losses = []

    if isinstance(bottleneck, VAEBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        kl_weight = loss_config['bottleneck']['weights']['kl']

        kl_loss = ValueLoss(key='kl', weight=kl_weight, name='kl_loss')
        losses.append(kl_loss)

    if isinstance(bottleneck, RVQBottleneck) or isinstance(bottleneck, RVQVAEBottleneck):
        quantizer_loss = ValueLoss(key='quantizer_loss', weight=1.0, name='quantizer_loss')
        losses.append(quantizer_loss)

    if isinstance(bottleneck, DACRVQBottleneck) or isinstance(bottleneck, DACRVQVAEBottleneck):
        codebook_loss = ValueLoss(key='vq/codebook_loss', weight=1.0, name='codebook_loss')
        commitment_loss = ValueLoss(key='vq/commitment_loss', weight=0.25, name='commitment_loss')
        losses.append(codebook_loss)
        losses.append(commitment_loss)

    if isinstance(bottleneck, WassersteinBottleneck):
        mmd_weight = loss_config['bottleneck']['weights']['mmd']

        mmd_loss = ValueLoss(key='mmd', weight=mmd_weight, name='mmd_loss')
        losses.append(mmd_loss)

    return losses
