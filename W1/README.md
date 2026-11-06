# W1: K2VAE reproduction checkpoints

This folder provides K2VAE checkpoints reproduced in the K2VAE / ProbTS framework for the Electricity and Traffic forecasting settings in the VolDy-VAE experiments. Each file contains inference weights, the model configuration, training-set standardization statistics, and feature names. The original K2VAE architecture and sampling procedure are retained.

## Checkpoints

The following metrics are the recorded test results for the corresponding checkpoint, using 100 forecast samples and 20 evaluation quantiles. ND is the framework's normalized absolute error. Lower is better.

| Dataset | Context | Horizon | Seed | CRPS | ND | Weights |
|---|---:|---:|---:|---:|---:|---|
| Electricity | 96 | 336 | 3 | 0.084208 | 0.106830 | [k2vae_electricity_h336.pt](k2vae_electricity_h336.pt) |
| Electricity | 96 | 720 | 0 | 0.102190 | 0.127641 | [k2vae_electricity_h720.pt](k2vae_electricity_h720.pt) |
| Traffic | 96 | 96 | 2 | 0.185506 | 0.231092 | [k2vae_traffic_h96.pt](k2vae_traffic_h96.pt) |
| Traffic | 96 | 192 | 2 | 0.191213 | 0.238244 | [k2vae_traffic_h192.pt](k2vae_traffic_h192.pt) |

[metadata.json](metadata.json) contains all four recorded metrics, checkpoint epochs, feature order, and batch sizes. [results_per_seed.csv](results_per_seed.csv) contains the complete 20-run reproduction with seeds 0–4; [results_mean_std.csv](results_mean_std.csv) reports the mean and sample standard deviation across those five seeds. The checkpoint table above reports individual runs; the CSV summary reports five-seed statistics.

Training used standard scaling fitted on the training split, a separate validation split, a maximum of 20 epochs and 100 training batches per epoch, and validation-CRPS early stopping with patience 5. The weights are from the best validation-CRPS epoch of their respective run. The bundles contain model tensors and preprocessing statistics, without optimizer states, and load with `torch.load(..., weights_only=True)`.

## Inference

From the repository root:

```bash
python -m pip install -r W1/requirements.txt

python W1/infer.py \
  --weights W1/k2vae_electricity_h336.pt \
  --input /path/to/electricity/electricity.csv \
  --output forecasts/electricity_h336.npz \
  --device cpu
```

Use the standard LTSF benchmark CSV, with a `date` column and all 321 Electricity or 862 Traffic feature columns. The saved feature names determine column order. The script forecasts from the final 96 observed rows. To forecast from a historical origin, pass `--origin ROW`, where `ROW` is the zero-based index of the first unobserved row; rows `[ROW-96:ROW]` supply the context. It also accepts a NumPy `.npy` array of raw observations shaped `[time, channels]` or `[batch, 96, channels]`.

```bash
python W1/infer.py \
  --weights W1/k2vae_electricity_h720.pt \
  --input /path/to/electricity/electricity.csv \
  --output forecasts/electricity_h720.npz --device cuda

python W1/infer.py \
  --weights W1/k2vae_traffic_h96.pt \
  --input /path/to/traffic/traffic.csv \
  --output forecasts/traffic_h96.npz --device cuda

python W1/infer.py \
  --weights W1/k2vae_traffic_h192.pt \
  --input /path/to/traffic/traffic.csv \
  --output forecasts/traffic_h192.npz --device cuda
```

Input values use the original data units. The script applies the stored training scaler before prediction and reverses it afterward. Each output contains `samples` shaped `[batch, 100, horizon, channels]`, `mean` and `median` shaped `[batch, horizon, channels]`, and `quantiles` shaped `[batch, 3, horizon, channels]` for levels 0.05, 0.50, and 0.95. All forecasts are in the original data units. `--num-samples`, `--seed`, and `--batch-size` control forecast sampling and inference batching; the default sampling seed is 0 and the default batch size is 1.

```python
import sys
import numpy as np
import torch

sys.path.insert(0, "W1")
from infer import Predictor

predictor = Predictor("W1/k2vae_traffic_h96.pt", device="cpu")
history = np.load("traffic_history.npy")  # [batch, 96, 862], raw values
torch.manual_seed(0)
samples = predictor.predict(history, num_samples=100)  # [batch, 100, 96, 862]
```

Forecast sampling is stochastic. The recorded test metrics come from the original complete test evaluation, including its random-number stream and batching; an individual forecast window has its own output samples.

## Verification and source

All four bundles were strictly reloaded and compared with the original K2VAE implementation on real benchmark contexts. With identical sampling seeds, their 100-sample predictions matched exactly on CPU: maximum absolute difference **0.0** for every bundle. Training scalers and CSV feature order were also checked. See [verification.json](verification.json), which records the environment and output shapes.

[k2vae_runtime/](k2vae_runtime/) contains the numerical model dependencies used for inference. Their module imports are local; unused alternative model classes and optional imports are omitted. Tested with PyTorch 2.4.1. The bundled code retains its source comments and is distributed with [LICENSE](LICENSE).
