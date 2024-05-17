import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.quantization import (
    QuantizerConfig,
    get_model_quantized,
    get_model_size_mb,
    prepare_for_QAT,
    quantize_fuse_bn,
    quantize_PTQ,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _quantize(
    module: LightningModule, cfg: DictConfig, datamodule: LightningDataModule, trainer: Trainer
) -> Tuple[LightningModule, Path]:
    """Performs quantization based on chosen methods."""
    log.info("Starting quantization...")
    quant_path = Path(cfg.paths.output_dir) / "quant"
    if not os.path.exists(quant_path):
        os.makedirs(quant_path, exist_ok=True)
    quantization_methods = cfg.quantization.methods
    quantizer = None
    ckpt_path = None
    quant_config = QuantizerConfig(**cfg.quantization.quant_config)
    model = module.net
    if "fuse_bn" in quantization_methods:
        log.info("FuseBatchNorm Quantization...")
        model = quantize_fuse_bn(model)
    if "ptq" in quantization_methods:
        log.info("Performing PTQ...")
        device = "cuda" if cfg.trainer.accelerator == "gpu" else "cpu"
        datamodule.setup()
        model, quantizer = quantize_PTQ(
            model=model,
            work_dir=quant_path,
            quant_config=quant_config,
            n_ptq_examples=cfg.quantization.ptq.examples_limit,
            device=device,
            dataloader=datamodule.train_dataloader(),
            quantizer=quantizer,
        )
        log.info("PTQ Finished...")
    if "qat" in quantization_methods:
        log.info("Performing QAT...")
        model, quantizer = prepare_for_QAT(
            model=model, work_dir=quant_path, quant_config=quant_config, quantizer=quantizer
        )
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path:
            model.load_state_dict(ckpt_path)
        log.info("QAT Finished...")

    if quantizer:
        quantized_model = get_model_quantized(model=model)
    else:
        quantized_model = model
    model_path = quant_path / f"{'_'.join(quantization_methods)}_qmodel.pth"
    torch.save(quantized_model.state_dict(), model_path)
    module.net = quantized_model
    return module, model_path


@task_wrapper
def quantize(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Quantize the model. After all, it evaluates the model.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects. # TODO
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.trainer.accelerator == "gpu":
        torch.set_float32_matmul_precision("medium")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.get("ckpt_path"))["state_dict"])

    log.info(f"Model size before quantization: {get_model_size_mb(model):.3f} MB")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    qmodel, path = _quantize(module=model, cfg=cfg, datamodule=datamodule, trainer=trainer)
    log.info(f"Model size after quantization: {get_model_size_mb(model):.3f} MB")

    if cfg.get("test"):
        log.info("Starting testing!")
        datamodule.batch_size_per_device = 1
        trainer.test(model=qmodel, datamodule=datamodule, ckpt_path=None)
        log.info(f"Best ckpt path: {path}")

    if cfg.get("inference_speed"):
        log.info("Calculating mean inference speed...")
        model.calc_inference_speed()

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="quantize.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)
    metric_dict, _ = quantize(cfg)
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))
    return metric_value


if __name__ == "__main__":
    main()
