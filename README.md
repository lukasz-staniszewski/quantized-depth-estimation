<div align="center">

# Neural Networks Compression

### *Lukasz Staniszewski*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

</div>

## Description

ENNCA exercises + final project

## Installation

### Pip

```bash
# clone project
git clone https://github.com/lukasz-staniszewski/neural-networks-compression
cd neural-networks-compression

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda

```bash
# clone project
git clone https://github.com/lukasz-staniszewski/neural-networks-compression
cd neural-networks-compression

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

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

## Quantization

No quantization:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │    0.00825385469943285    │
│         test/mse          │    0.00825385469943285    │
│         test/ssim         │    0.8089691996574402     │
└───────────────────────────┴───────────────────────────┘
[2024-05-17 11:54:30,412][__main__][INFO] - [rank: 0] Calculating mean inference speed
STAGE:2024-05-17 11:54:31 1038662:1038662 ActivityProfilerController.cpp:314] Completed Stage: Warm Up
STAGE:2024-05-17 11:54:31 1038662:1038662 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-05-17 11:54:31 1038662:1038662 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    ProfilerStep*        11.42%      31.256ms       100.00%     273.578ms      27.358ms            10
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

FuseBN:
```

Model size after quantization: 2.594 MB

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │    0.00825385469943285    │
│         test/mse          │   0.008253855630755424    │
│         test/ssim         │    0.8089691996574402     │
└───────────────────────────┴───────────────────────────┘

---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    ProfilerStep*        10.17%      30.225ms       100.00%     297.107ms      29.711ms            10
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```
