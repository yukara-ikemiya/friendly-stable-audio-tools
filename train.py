
import os
import json

import pytorch_lightning as pl
from prefigure.prefigure import get_all_args, push_wandb_config

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config, create_tqdm_callback_from_config
from stable_audio_tools.utils.torch_common import set_seed, copy_state_dict


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def main():

    args = get_all_args()

    # Set a different seed for each process if using SLURM
    seed = args.seed
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    set_seed(seed)

    # Get JSON config from args.model_config
    with open(args.model_config) as f:
        model_config: dict = json.load(f)

    with open(args.dataset_config) as f:
        dataset_config: dict = json.load(f)

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config["audio_channels"]
    )

    model = create_model_from_config(model_config)

    if args.pretrained_ckpt_path:
        print(f"->->-> Loading a pretrained checkpoint from {args.pretrained_ckpt_path}...")
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(model.pretransform)

    if args.pretransform_ckpt_path:
        print(f"->->-> Loading a pretransform checkpoint from {args.pretransform_ckpt_path}...")
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(training_wrapper)

    exc_callback = ExceptionCallback()

    if args.save_dir and isinstance(wandb_logger.experiment.id, str):
        save_dir = os.path.join(args.save_dir, wandb_logger.experiment.project, wandb_logger.experiment.id)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
    else:
        save_dir = None
        checkpoint_dir = None

    ckpt_config = model_config["training"].get("checkpoint", {"every_n_train_steps": 10000, "save_top_k": 1, "save_last": True})
    ckpt_callback = pl.callbacks.ModelCheckpoint(**ckpt_config, dirpath=checkpoint_dir)
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)
    tqdm_callback = create_tqdm_callback_from_config(model_config)

    # Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    push_wandb_config(wandb_logger, args_dict)

    # Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5e8,
                allgather_bucket_size=5e8,
                load_full_weights=True
            )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[ckpt_callback, demo_callback, tqdm_callback, exc_callback, save_model_config_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        default_root_dir=save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0
    )

    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)


if __name__ == '__main__':
    main()
