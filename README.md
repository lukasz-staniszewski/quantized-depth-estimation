<div align="center">

# Quantized Depth Estimation

### *Lukasz Staniszewski*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

## Description

This repository serves as codebase for the project focused on applying effective neural network architectures for Depth Estimation along with researching best quantization methods to reduce their size. Project documentation can be found [here](https://github.com/lukasz-staniszewski/quantized-depth-estimation/blob/main/docs/project_results.md).

## Installation

### Pip

```bash
# clone project
git clone https://github.com/lukasz-staniszewski/quantized-depth-estimation
cd quantized-depth-estimation

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10.13
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda

```bash
# clone project
git clone https://github.com/lukasz-staniszewski/quantized-depth-estimation
cd quantized-depth-estimation

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

### Start by setting `PYTHONPATH` (inside project's root dir)

```sh
export PYTHONPATH=$PWD
```

### Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

### Eval your model

```bash
python src/eval.py ckpt_path=<PATH>

```

### Quantize your model

Configure quantization settings inside [quantization config file](configs/quantize.yaml) and run:

```bash
python src/quantize.py ckpt_path=<PATH>
```

You can set up there your quantization scenario easily:

```yaml
...
inference_speed: True       # set to True if you want to check quantized model inference speed

quantization:
  methods:
    # - "fuse_bn"           # fuse batch norm
    - "ptq"                 # post training quantization
    - "qat"                 # quantization aware training
  ptq:
    batches_limit: 250      # max number of batches for ptq calibration
  qat:
    max_epochs: 10          # max number of epochs for qat
  quant_config:
    dummy_input_shape: [1, 3, 224, 224]
    is_per_tensor: False    # True if per-tensor quantization, False if you prefer per-channel
    is_asymmetric: True
    backend: "qnnpack"      # 'qnnpack' for mobile devices or 'fbgemm' for servers
    disable_requantization_for_cat: True
    use_cle: True           # wether to use Cross Layer Normalization before doing PTQ/QAT
    overwrite_set_ptq: True # if you don't use ptq, set it to False
```

## Reproducing results

### Training

```bash
python src/train.py experiment=nyu_efffnet
```

### Quantization

Download `quantize.yaml` file from [GitHub repository](https://github.com/lukasz-staniszewski/quantized-depth-estimation/blob/main/configs/quantize.yaml) and put to `configs/` directory.

Run:

```bash
python src/quantize.py ckpt_path=<CORRECT RUN PATH>/checkpoints/epoch_023.ckpt
```

## TODO

Check those links:

- <https://github.com/alibaba/TinyNeuralNetwork/blob/b6c78946d09b853071f55fb9b481ff632ea9568c/examples/quantization/specific/vit/vit_post.py>
- <https://github.com/alibaba/TinyNeuralNetwork/blob/b6c78946d09b853071f55fb9b481ff632ea9568c/examples/quantization/specific/mobileone/post.py>
- <https://github.com/alibaba/TinyNeuralNetwork/blob/b6c78946d09b853071f55fb9b481ff632ea9568c/examples/quantization/specific/mobileone/qat.py>
