import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from pydantic import BaseModel
from tinynn.graph.quantization.quantizer import QATQuantizer
from torch.utils.data import DataLoader


class QuantizerConfig(BaseModel):
    """Config for PTQ/QAT quantizers."""

    dummy_input_shape: List[int]
    is_per_tensor: bool
    is_asymmetric: bool = True
    backend: str = "qnnpack"
    disable_requantization_for_cat: bool = True


def _ptq_calibration(
    model: torch.nn.Module, examples_limit: int, dataloader: DataLoader, device: torch.device
) -> torch.nn.Module:
    """Performs simple ptq calibration pipeline.

    Args:
        model (torch.nn.Module): torch model to quantize (with enabled ptq observer)
        examples_limit (int): how many samples from training data we will use to observe data vals
        dataloader(Dataloader): traininig data loader for calibration
        device (torch.device): on what device (cuda/cpu) we will do calibration

    Returns:
        torch.nn.Module: ptq calibrated model
    """
    for idx, data in enumerate(dataloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        model(images)
        if idx >= examples_limit:
            break
    return model


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Returns size (on disk) of specific pytorch model.

    Args:
        model (torch.nn.Module): torch model

    Returns:
        float: model disk size in MB
    """
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove("temp.p")
    return size


def quantize_fuse_bn(model: torch.nn.Module) -> torch.nn.Module:
    """For specific model, it fuses its modules like conv+bn, conv+bn+relu."""
    return quantize_fx.fuse_fx(model.eval())


def create_quantizer(model: torch.nn.Module, quant_config: QuantizerConfig, work_dir: Path) -> QATQuantizer:
    """Creates quantizer (tiny neural network).

    Args:
        model (torch.nn.Module): torch model to quantize

    Returns:
        torch.nn.Module: model with performed bn fusing
    """
    model = model.eval()
    quantizer = QATQuantizer(
        model=model,
        dummy_input=torch.randn(*quant_config.dummy_input_shape),
        work_dir=work_dir,
        config={
            "asymmetric": quant_config.is_asymmetric,
            "backend": quant_config.backend,
            "disable_requantization_for_cat": quant_config.disable_requantization_for_cat,
            "per_tensor": quant_config.is_per_tensor,
        },
    )
    return quantizer


def quantize_PTQ(
    model: torch.nn.Module,
    work_dir: Path,
    quant_config: QuantizerConfig,
    n_ptq_examples: int,
    device: torch.device,
    dataloader: DataLoader,
    quantizer: Optional[QATQuantizer] = None,
) -> Tuple[torch.nn.Module, QATQuantizer]:
    """Performs Post Training Quantization (PTQ).

    Args:
        model (torch.nn.Module): torch model to quantize
        work_dir (str): where output of quantization should be stored
        quant_config (QuantizerConfig): config of quantizer
        n_ptq_examples (int): how many samples from training data we will use to observe data vals
        device (torch.device): on what device (cuda/cpu) we will do calibration
        dataloader(Dataloader): traininig data loader for calibration
        quantizer (Optional[QATQuantizer]): if passed, then we don't reinit quantizer. Defaults to None.

    Returns:
        torch.nn.Module: model after ptq
        QATQuantizer: tinynn quantizer class
    """
    if quantizer is None:
        quantizer = create_quantizer(model=model, quant_config=quant_config, work_dir=work_dir)
        qmodel = quantizer.quantize()
    else:
        qmodel = model
    qmodel = qmodel.eval()
    qmodel = _ptq_calibration(model=qmodel, examples_limit=n_ptq_examples, dataloader=dataloader, device=device)
    return qmodel, quantizer


def prepare_for_QAT(
    model: torch.nn.Module,
    work_dir: str,
    quant_config: QuantizerConfig,
    quantizer: Optional[QATQuantizer] = None,
) -> Tuple[torch.nn.Module, QATQuantizer]:
    """Prepares model for QAT (Quantization aware training).

    Args:
        model (torch.nn.Module): torch model to quantize
        work_dir (str): where output of quantization should be stored
        quant_config (QuantizerConfig): config of quantizer
        quantizer (Optional[QATQuantizer]): if passed, then we don't reinit quantizer. Defaults to None.

    Returns:
        torch.nn.Module: model prepared for qat
        QATQuantizer: tinynn quantizer class
    """
    if quantizer is None:
        quantizer = create_quantizer(model=model, quant_config=quant_config, work_dir=work_dir)
        qmodel = quantizer.quantize()
    else:
        qmodel = model
    qmodel = qmodel.eval()
    qmodel = qmodel.apply(torch.quantization.disable_observer)
    return qmodel, quantizer


def get_model_quantized(model: torch.nn.Module) -> torch.nn.Module:
    model = model.cpu()
    model = model.eval()
    return torch.quantization.convert(model.cpu(), inplace=False)
