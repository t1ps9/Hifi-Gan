import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
import itertools
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    dataloaders, batch_transforms = get_dataloaders(config, device)

    model = instantiate(config.model).to(device)
    logger.info(model)

    loss_function = instantiate(config.loss_function).to(device)

    trainable_params_g = filter(lambda p: p.requires_grad, model.generator.parameters())
    optimizer_gen = instantiate(config.optimizer_gen, params=trainable_params_g)
    lr_scheduler_gen = instantiate(config.lr_scheduler_gen, optimizer=optimizer_gen)

    trainable_params_d = filter(lambda p: p.requires_grad, itertools.chain(
        model.mpd.parameters(), model.msd.parameters())
    )
    optimizer_disc = instantiate(config.optimizer_disc, params=trainable_params_d)
    lr_scheduler_disc = instantiate(config.lr_scheduler_disc, optimizer=optimizer_disc)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    metrics = instantiate(config.metrics)

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer_gen=optimizer_gen,
        optimizer_disc=optimizer_disc,
        lr_scheduler_gen=lr_scheduler_gen,
        lr_scheduler_disc=lr_scheduler_disc,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
