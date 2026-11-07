# Vanilla VAE (MSE): Electricity, seed 1

This archive provides the original training records and recovered inference weights for **Vanilla VAE (MSE), Electricity, context 96, horizon 720, training seed 1**. It contains one historical run, not a multi-seed aggregate or a newly trained replication.

Original run identifier:

```text
electricity_ltsf_GaussianVAE_MSE_Model_CTX96_PRED720_seed1_log_20251229_022121_1766946081859
```

## Training records

The saved configuration specifies Adam with learning rate 0.001, batch size 64, 100 training batches per epoch, a maximum of 50 epochs, gradient clipping at 0.5, training-split standard scaling, and a separate validation split. The TensorBoard log contains **5,000 training steps and 50 validation evaluations**, covering zero-based epochs 0–49.

| Artifact | Contents |
|---|---|
| [seed_1/config.yaml](seed_1/config.yaml) | Original resolved training configuration, including seed, model and data settings |
| [seed_1/hparams.yaml](seed_1/hparams.yaml) | Original fit-stage hyperparameter record |
| [seed_1/events.out.tfevents.1766946082.user-H8230.4113710.0](seed_1/events.out.tfevents.1766946082.user-H8230.4113710.0) | Unmodified TensorBoard event file |
| [seed_1/scalars.csv](seed_1/scalars.csv) | All 11,351 logged scalar events, with tag, step, epoch, value and UTC timestamp |
| [seed_1/training_by_epoch.csv](seed_1/training_by_epoch.csv) | Arithmetic mean of the 100 logged training-batch losses per epoch |
| [seed_1/validation_by_epoch.csv](seed_1/validation_by_epoch.csv) | All validation metrics for each of the 50 epochs |
| [seed_1/horizons_results.csv](seed_1/horizons_results.csv) | Original recorded test results |
| [seed_1/test_metrics.csv](seed_1/test_metrics.csv) | Original test logger output |
| [seed_1/summary.json](seed_1/summary.json) | Original parameter, memory and timing summary |

The minimum logged validation CRPS is **0.319050**, at zero-based epoch **1**. The original recorded test scores are **CRPS 0.369027** and **ND 0.370216**. These are individual-run records. Test-set scores have not been recomputed during archive export.

View the original training curves with TensorBoard:

```bash
tensorboard --logdir W6/vanilla_vae_mse/electricity_L96_H720/seed_1
```

## Weights and loading

Download [vanilla_vae_mse_electricity_L96_H720_seed1.pt](seed_1/vanilla_vae_mse_electricity_L96_H720_seed1.pt). The bundle contains the model state dictionary, configuration, original scaler statistics, feature names and run identifier. It contains **6,153,370 model parameters**.

The original `.ckpt` files were no longer present in this run's checkpoint directory. The weights were recovered from the complete model object serialized in the original **test-stage `hparams.yaml`**. This source is recorded in the bundle and [verification.json](seed_1/verification.json). The bundle is an inference state dictionary rather than an optimizer checkpoint; its checkpoint epoch is not inferred from the validation log.

[seed_1/runtime/](seed_1/runtime/) contains the native model and its MLP/RevIN dependencies, with dependency imports localized. Load from the repository root:

```python
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, "W6/vanilla_vae_mse/electricity_L96_H720/seed_1")
from runtime import GaussianVAE_MSE

bundle = torch.load(
    "W6/vanilla_vae_mse/electricity_L96_H720/seed_1/"
    "vanilla_vae_mse_electricity_L96_H720_seed1.pt",
    map_location="cpu",
    weights_only=True,
)
model = GaussianVAE_MSE(SimpleNamespace(**bundle["model_config"])).eval()
model.load_state_dict(bundle["model_state_dict"], strict=True)

# history_raw: float tensor [batch, 96, 321], in bundle['feature_names'] order.
def forecast(history_raw, num_samples=100):
    scaler = bundle["scaler"]
    mean = scaler["mean"]
    scale = scaler["scale"] + scaler["epsilon"]
    normalized = ((history_raw.cpu() - mean) / scale).float()
    with torch.no_grad():
        samples = model.sample(normalized, num_samples)
    return samples * scale + mean  # [batch, num_samples, 720, 321]
```

The bundle was loaded with `weights_only=True` and strictly reloaded into a fresh model. All state tensors matched the archived model. With the same sampling seed and an actual 96-step Electricity context, its 100 forecast samples matched the archived model exactly: **maximum absolute difference 0.0**. The localized runtime also matched the native model exactly. See [seed_1/verification.json](seed_1/verification.json). Verification used PyTorch 2.4.1 on CPU.

The copied numerical code is distributed under [seed_1/LICENSE](seed_1/LICENSE).
