import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.ao.quantization.quantize_fx as quantize_fx
from pydantic import BaseModel
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.graph.quantization.quantizer import QATQuantizer
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm


class QuantizerConfig(BaseModel):
    """Config for PTQ/QAT quantizers."""

    dummy_input_shape: List[int]
    is_per_tensor: bool
    is_asymmetric: bool = True
    backend: str = "qnnpack"
    disable_requantization_for_cat: bool = True
    use_cle: bool = False
    overwrite_set_ptq: bool = False


def _ptq_calibration(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device, batches_limit: int | None = None
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
    total_examples = len(dataloader) if batches_limit is None else min(len(dataloader), batches_limit)
    with torch.no_grad():
        for idx, data in tqdm(enumerate(dataloader), total=total_examples, desc="Calibration"):
            if batches_limit is not None and idx >= batches_limit:
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            model(images)
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


def create_quantizer(
    model: torch.nn.Module, quant_config: QuantizerConfig, work_dir: Path, device: torch.device
) -> QATQuantizer:
    """Creates quantizer (tiny neural network)."""
    dummy_input = torch.randn(*quant_config.dummy_input_shape)
    config = {
        "asymmetric": quant_config.is_asymmetric,
        "backend": quant_config.backend,
        "disable_requantization_for_cat": quant_config.disable_requantization_for_cat,
        "per_tensor": quant_config.is_per_tensor,
    }
    if quant_config.overwrite_set_ptq:
        config["override_qconfig_func"] = set_ptq_fake_quantize

    model = model.eval()
    if quant_config.use_cle:
        model = cross_layer_equalize(model=model, dummy_input=dummy_input, device=device)

    quantizer = QATQuantizer(
        model=model,
        dummy_input=dummy_input,
        work_dir=work_dir,
        config=config,
    )
    return quantizer


def quantize_PTQ(
    model: torch.nn.Module,
    work_dir: Path,
    quant_config: QuantizerConfig,
    n_ptq_batches_limit: int | None,
    device: torch.device,
    dataloader: DataLoader,
    quantizer: Optional[QATQuantizer] = None,
) -> Tuple[torch.nn.Module, QATQuantizer]:
    """Performs Post Training Quantization (PTQ).

    Args:
        model (torch.nn.Module): torch model to quantize
        work_dir (str): where output of quantization should be stored
        quant_config (QuantizerConfig): config of quantizer
        n_ptq_batches_limit (int): how many samples from training data we will use to observe data vals
        device (torch.device): on what device (cuda/cpu) we will do calibration
        dataloader(Dataloader): traininig data loader for calibration
        quantizer (Optional[QATQuantizer]): if passed, then we don't reinit quantizer. Defaults to None.

    Returns:
        torch.nn.Module: model after ptq
        QATQuantizer: tinynn quantizer class
    """
    if quantizer is None:
        quantizer = create_quantizer(model=model, quant_config=quant_config, work_dir=work_dir, device=device)
        qmodel = quantizer.quantize()
    else:
        qmodel = model
    qmodel.eval()
    qmodel.to(device)
    qmodel.apply(torch.quantization.disable_fake_quant)
    qmodel.apply(torch.quantization.enable_observer)
    qmodel = _ptq_calibration(model=qmodel, dataloader=dataloader, device=device, batches_limit=n_ptq_batches_limit)
    qmodel.apply(torch.quantization.disable_observer)
    qmodel.apply(torch.quantization.enable_fake_quant)
    return qmodel, quantizer


def prepare_for_QAT(
    model: torch.nn.Module,
    work_dir: str,
    quant_config: QuantizerConfig,
    device: torch.device,
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
        quantizer = create_quantizer(model=model, quant_config=quant_config, work_dir=work_dir, device=device)
        qmodel = quantizer.quantize()
    else:
        qmodel = model
    qmodel = qmodel.eval()
    qmodel = qmodel.apply(torch.quantization.disable_observer)
    return qmodel, quantizer


def get_model_quantized(model: torch.nn.Module) -> torch.nn.Module:
    model = model.cpu()
    model = model.eval()
    qmodel = torch.quantization.convert(model)
    return qmodel


def calc_inference_speed(model: torch.nn.Module, dataloader: DataLoader, steps_limit: int = 1000):
    """Simulates inference speed."""

    def _trace_handler(prof):
        print(prof.key_averages().table(sort_by="cpu_time", row_limit=1))

    model = model.eval()
    model = model.cpu()

    batch = next(iter(dataloader))
    batch_sample = batch[0][:1]
    with profile(
        activities=[ProfilerActivity.CPU],
        schedule=schedule(wait=10, warmup=10, active=10, repeat=1),
        on_trace_ready=_trace_handler,
    ) as p:
        for _ in range(steps_limit):
            with torch.no_grad():
                model(batch_sample)
            p.step()
