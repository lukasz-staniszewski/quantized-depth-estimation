<h1 align="center">Neural Networks Compression with Applications</h1>
<h3 align="center"><i>Łukasz Staniszewski</i></h4>
<h3 align="center">Project - Depth Estimation</h4>

<h2>1. Model architecture</h2>
Used model architecture is U-Net type architecture consisting of:

1. Encoder: **EfficientNet-B0** (first six layers), taking output from 3rd, 4th, 5th and 6th layer.

2. Decoder: **Fuse-Free Network** (Type-C version with ).

Final architecture code can be found [here](https://github.com/lukasz-staniszewski/neural-networks-compression/blob/main/src/models/components/efficient_ffnet.py).

<h2> 2. Training results </h2>

*Setup*: model was trained for 25 epochs on [NYUv2 Depth Dataset](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2). I used AdamW optimizer (with $0.001$ learning rate and $0.01$ weight decay) and ReduceLROnPlateau scheduler. I used gradient clipping to 1. Input images have been resized to $[224, 224]$ and outputed masks were of size $[56, 56]$.

Final metrics:
| Metric | Train | Validation | Test |
|----------|-------|------------|-------|
| SSIM     | $0.8883$    |     $0.8967$     | $0.8175$ |
| MSE      | $0.0026$    |     $0.0025$     | $0.0081$ |

Example validation set generations:

![samples](https://github.com/lukasz-staniszewski/neural-networks-compression/assets/59453698/5747052f-da83-402a-bd9b-f01c6f441574)

Whole training run metrics can be found [here](https://api.wandb.ai/links/lukasz-staniszewski/nlh2c64w).

<h2> Quantization </h2>

<h3> Original model </h2>

Metrics:

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │    0.00814119540154934    │
│         test/mse          │   0.008141196332871914    │
│         test/ssim         │     0.817513644695282     │
└───────────────────────────┴───────────────────────────┘
```

Inference speed:

```sh
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                   ProfilerStep*        11.58%      25.276ms       100.00%     218.294ms      21.829ms            10
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

Model size:

```sh
Model size before quantization: 5.045 MB
```

<h3> FuseBN </h3>

Metrics:

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │   0.008141200058162212    │
│         test/mse          │   0.008141197264194489    │
│         test/ssim         │    0.8175134658813477     │
└───────────────────────────┴───────────────────────────┘
```

Speed:

```sh
------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                          Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                 ProfilerStep*        13.82%      29.077ms       100.00%     210.362ms      21.036ms            10
------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 253.266ms
```

Size:

```sh
Model size after quantization: 4.891 MB
```

<h3> Per Channel PTQ (QNNPACK)</h3>

- Calibration done for 250 batches
- We used `qnnpack` as quantization backend
- Applied Cross Layer Equalization

Metrics:

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │   0.010917793959379196    │
│         test/mse          │   0.010917791165411472    │
│         test/ssim         │     0.752718985080719     │
└───────────────────────────┴───────────────────────────┘
```

Speed:

```sh
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    ProfilerStep*         3.52%      37.933ms       100.00%        1.079s     107.908ms            10
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

Size:

```sh
Model size after quantization: 1.607 MB
```

<h3> FuseBN + Per Channel PTQ (QNNPACK) </h3>

- Calibration done for 250 batches
- We used `qnnpack` as quantization backend
- Applied Cross Layer Equalization

Metrics:

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │   0.008759920485317707    │
│         test/mse          │   0.008759919553995132    │
│         test/ssim         │    0.7441965937614441     │
└───────────────────────────┴───────────────────────────┘
```

Speed:

```sh
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    ProfilerStep*         3.34%      35.999ms       100.00%        1.077s     107.673ms            10
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

Size:

```sh
Model size after quantization: 1.605 MB
```

<h3> Per Tensor PTQ (QNNPACK)</h3>

- Calibration done for 250 batches
- We used `qnnpack` as quantization backend
- Applied Cross Layer Equalization

Metrics:

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │    0.03335525840520859    │
│         test/mse          │    0.03335526958107948    │
│         test/ssim         │    0.5828493237495422     │
└───────────────────────────┴───────────────────────────┘
```

Speed:

```sh
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    ProfilerStep*         3.49%      37.777ms       100.00%        1.084s     108.376ms            10
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

Size:

```sh
Model size after quantization: 1.379 MB
```

<h3> FuseBN + Per Tensor PTQ (QNNPACK) </h3>

- Calibration done for 250 batches
- We used `qnnpack` as quantization backend
- Applied Cross Layer Equalization

Metrics:

```sh
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/loss         │    0.03469409421086311    │
│         test/mse          │   0.034694064408540726    │
│         test/ssim         │    0.5722488760948181     │
└───────────────────────────┴───────────────────────────┘
```

Speed:

```sh
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                    ProfilerStep*         3.60%      38.666ms       100.00%        1.075s     107.496ms            10
---------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

Size:

```sh
Model size after quantization: 1.377 MB
```
