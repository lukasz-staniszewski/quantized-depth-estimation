<h1 align="center">Neural Networks Compression with Applications</h1>
<h3 align="center"><i>Łukasz Staniszewski</i></h4>
<h3 align="center">Project - Depth Estimation with Quantization</h4>

<h2>1. Model architecture</h2>
Used model architecture is U-Net type architecture that consists of:

1. Encoder: **[EfficientNet-B0](https://arxiv.org/pdf/1905.11946v5)** (first six layers), taking output from 3rd, 4th, 5th and 6th layer.

2. Decoder: **[FFNet](https://arxiv.org/pdf/2206.08236)** (Up-C version with additional depth task head).

Final architecture code can be found [here](https://github.com/lukasz-staniszewski/quantized-depth-estimation/blob/main/src/models/components/efficient_ffnet.py).

<h2> 2. Depth estimation training results </h2>

*Setup*: model was trained for 25 epochs on [NYUv2 Depth Dataset](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2). AdamW optimizer (with $0.001$ learning rate and $0.01$ weight decay) was used along with ReduceLROnPlateau scheduler. I applied gradient clipping to 1. Input images have been resized to $[224, 224]$ and outputed masks were of size $[56, 56]$.

Final metrics:
| Metric | Train | Validation | Test |
|----------|-------|------------|-------|
| SSIM     | $0.8883$    |     $0.8967$     | $0.8175$ |
| MSE      | $0.0026$    |     $0.0025$     | $0.0081$ |

Example validation set generations:

![samples](https://github.com/lukasz-staniszewski/quantized-depth-estimation/assets/59453698/2672365f-31cf-4bb6-b170-4759aca3c794)

Whole training/validation/test metrics along with loss curves and generated samples can be found [here](https://wandb.ai/lukasz-staniszewski/neural-networks-compression/runs/bmx6o01o/workspace?nw=nwuserlukaszstaniszewski).

<h2> 3. Quantization </h2>

<h3>Experiment setup</h3>

1. Each quantization experiment starts with the same pre-trained float32 neural network model obtained from the training process described before.
2. For Post Training Quantization (PTQ) and Quantization Aware Training (QAT), the `qnnpack` quantization backend was used.
3. In each PTQ/QAT experiment, the quantization process has been started with [Cross Layer Equalization](https://github.com/alibaba/TinyNeuralNetwork/blob/main/tinynn/graph/quantization/algorithm/cross_layer_equalization.py#L165)
4. When doing Post Training Quantization (PTQ), calibration has been done on the NYU Depth training dataset strictly for 250 batches.
5. When doing Quantization Aware Training (QAT), additional training has been performed in the same setup as for task pretraining, but for ten epochs.
6. The order of the quantization methods used is fixed, and the same for each experiment, i.e., the FuseBN operation is always applied first (directly on the pre-trained model), while the QAT operation is always applied last (directly followed by conversion to a smaller model). This leads to the fact that PTQ is always applied after FuseBN and before QAT.
7. Measuring the model's size is done by saving it to a file and checking its disk size, while measuring the model's inference speed is done on CPU and is a mean speed of 10 forward method calls done on tensor with the batch size equal to 1.

<h3> Results </h3>

| Quantization Combination            | Type        | Test MSE     | Test SSIM     | Size (MB)     | Mean Inference Speed (ms) |
|-------------------------------------|-------------|--------------|---------------|---------------|---------------------------|
| Original model                      |    -----    | 0.008141     | 0.817514      | 5.045         | 21.829                    |
| FuseBN                              |    -----    | 0.008141     | 0.817513      | 4.891         | 21.036                    |
| ***Per Channel Quantization***      |             |              |               |               |                           |
| PTQ                                 | Per Channel | 0.010918     | 0.752719      | 1.607         | 107.908                   |
| FuseBN + PTQ                        | Per Channel | 0.008760     | 0.744197      | **1.605**     | **107.673**               |
| QAT                                 | Per Channel | 0.011250     | 0.769749      | 1.607         | 107.894                   |
| FuseBN + QAT                        | Per Channel | 0.010503     | 0.756125      | **1.605**     | 112.216                   |
| PTQ + QAT                           | Per Channel | **0.008601** | **0.812770**  | 1.607         | 110.375                   |
| FuseBN + PTQ + QAT                  | Per Channel | 0.009217     | 0.769732      | **1.605**     | 111.207                   |
| ***Per Tensor Quantization***       |             |              |               |               |                           |
| PTQ                                 | Per Tensor  | 0.033355     | 0.582849      | 1.379         | 108.376                   |
| FuseBN + PTQ                        | Per Tensor  | 0.034694     | 0.572249      | **1.377**     | **107.496**               |
| QAT                                 | Per Tensor  | 0.016056     | 0.699009      | 1.379         | 110.057                   |
| FuseBN + QAT                        | Per Tensor  | 0.020541     | 0.692664      | **1.377**     | 110.852                   |
| PTQ + QAT                           | Per Tensor  | **0.011958** | **0.738777**  | 1.379         | 109.874                   |
| FuseBN + PTQ + QAT                  | Per Tensor  | 0.014009     | 0.677437      | **1.377**     | 110.009                   |

<h3> Conclusions </h3>

1. **Impact of Quantization on Model Size and Speed:**
   - All quantization methods significantly reduced the model size compared to the original model. While FuseBN reduces model size only by a little percent, PTQ and QAT allows us to obtain four times smaller models. Type of PTQ/QAT quantization is also important here since Per Tensor quantization gives smaller models than in Per Channel scenario.
   - Quantized models generally have much slower inference speed, with a notable increase in mean inference time from around 21 ms to over 100 ms, despite being more than four times smaller. Main reasoning for that is usage of `qnnpack` as quantization backend, which generates models suited for mobile devices, while inference speed tests were performed on server with AMD processors. When `fbgemm` was used, each resulting model obtained higher inference speed than the pre-trained model, but as it lacks Per Tensor quantization capabilities, this backend was not used for experiments.

2. **Post Training Quantization (PTQ) Performance:**
   - PTQ, both per-channel and per-tensor, resulted in a increase in test MSE and a decrease in SSIM compared to the original model, indicating a drop in performance.
   - FuseBN + PTQ combination applied in Per Channel scenario resulted in slightly better performance in MSE metric while being worse in SSIM. Applying FuseBN to PTQ lowered model size a bit and increased model inference speed.

3. **Quantization Aware Training (QAT) Performance:**
   - QAT showed improvements over PTQ in both MSE and SSIM in both per-channel and per-tensor settings but did not reach the original model's performance levels.
   - FuseBN + QAT combination, as same as in PTQ, did not improve over QAT alone in both MSE and SSIM metrics, but again lowered model size.

4. **Combining PTQ and QAT:**
   - The combination of PTQ and QAT provided the best results among all the quantized models, in both per channel and per tensor scenarios, particularly in the per-channel setting where it achieved the closest test MSE (0.008601) and SSIM (0.812770) to the original model, with a four times reduced size (1.607 MB).
   - The combination of FuseBN + PTQ + QAT also showed significant improvement but did not surpass PTQ + QAT alone in both metrics in both scenarios.

5. **Per-Channel vs. Per-Tensor Quantization:**
   - Per-channel quantization outperformed per-tensor quantization in terms of both test MSE and SSIM, indicating better preservation of model accuracy.
   - The smallest model sizes (1.377 MB) were observed with per-tensor quantization methods combined with FuseBN, showing a slight advantage in model size reduction.
