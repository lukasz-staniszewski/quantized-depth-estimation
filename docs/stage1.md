<h1 align="center">Neural Networks Compression with Applications</h1>
<h3 align="center"><i>Łukasz Staniszewski</i></h4>
<h3 align="center">Project - Depth Estimation</h4>

<h2>1. Model architecture</h2>
Used model architecture is U-Net type architecture consisting of:

1. Encoder: **EfficientNet-B0** (first five layers), taking output from 3rd, 4th and 5th layer.

2. Decoder: 3x **BiFPN** (middle three layers).

Final architecture code can be found [here](https://github.com/lukasz-staniszewski/neural-networks-compression/blob/main/src/models/components/effnet_bifpn.py).

<h2> 2. Training results </h2>

*Setup*: model was trained for 25 epochs on [NYUv2 Depth Dataset](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2). I used AdamW optimizer (with $0.001$ learning rate and $0.01$ weight decay) and ReduceLROnPlateau scheduler. I used gradient clipping to 1. Images have been resized to $[224, 224]$ and outputed masks were of size $[56, 56]$.

Final metrics:
| Metric | Train | Validation |
|----------|-------|------------|
| SSIM     | $0.8726$    |     $0.8769$     |
| MSE      | $0.0028$    |     $0.0025$     |

Example validation set generations:

![samples](https://github.com/lukasz-staniszewski/neural-networks-compression/assets/59453698/5747052f-da83-402a-bd9b-f01c6f441574)

Whole training run metrics can be found [here](https://api.wandb.ai/links/lukasz-staniszewski/nlh2c64w).
