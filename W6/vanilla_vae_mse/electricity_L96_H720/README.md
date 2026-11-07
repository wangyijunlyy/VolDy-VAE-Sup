# Vanilla VAE (MSE): Electricity, seed 1

This archive provides **Vanilla VAE (MSE), Electricity, context 96, horizon 720, training seed 1** records. The original historical training logs are preserved. The downloadable trained weights come from a separately labeled controlled rerun on 2026-11-07 and its actual validation-selected checkpoint. These are individual-run records, not a multi-seed aggregate.

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

## Trained weights and loading

Download [vanilla_vae_mse_electricity_L96_H720_seed1.pt](seed_1/vanilla_vae_mse_electricity_L96_H720_seed1.pt). The bundle contains **6,153,370 trained model parameters**, configuration, training-split scaler statistics, feature names and the source checkpoint path. The tensors are taken directly from the checkpoint's `state_dict` entries prefixed by `forecaster.model.`.

**Export correction:** The earlier file at this download path was extracted from the model object in test-stage `hparams.yaml`. That logging mechanism serializes Lightning's `hparams_initial` snapshot, captured before trained checkpoint weights are loaded. It cannot establish the trained model state. The earlier file has therefore been replaced with actual trained checkpoint tensors from the controlled rerun. The original historical `.ckpt` files remain unavailable; the new weights are not presented as a recovery of that historical checkpoint. [logger_snapshot_check.json](seed_1/logger_snapshot_check.json) reproduces the distinction: the live model matches its checkpoint, while all 16 model state tensors in the logger snapshot differ.

The controlled rerun uses seed 1, reconstruction coefficient 0.2, prediction coefficient 1 and KL coefficient 0.01, Adam lr 0.001, batch 64 and 100 training batches/epoch. It uses a maximum of 50 epochs with validation-CRPS early stopping patience 10. Validation/test batch 4 and workers 0 evaluate each held-out window once. The selected checkpoint is zero-based epoch 9; training stops after epoch 19. Its validation CRPS is **0.321209**, test CRPS **0.363658**, and test ND **0.364984**. These values belong to this rerun, separately from the historical scores above.

The matching rerun records are [rerun_config.yaml](seed_1/rerun_config.yaml), [rerun_training_metrics.csv](seed_1/rerun_training_metrics.csv), [rerun_test_results.csv](seed_1/rerun_test_results.csv), and [rerun_scores_per_case.csv](seed_1/rerun_scores_per_case.csv). The original historical training files listed above are unchanged.

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

The bundle was loaded with `weights_only=True` and strictly reloaded into a fresh model. All state tensors matched the actual trained checkpoint. With the same sampling seed and a real 96-step Electricity context, 100 forecast samples from the localized runtime matched the native model loaded with the checkpoint tensors exactly: **maximum absolute difference 0.0**. See [seed_1/verification.json](seed_1/verification.json). Verification used PyTorch 2.4.1 on CPU.

The copied numerical code is distributed under [seed_1/LICENSE](seed_1/LICENSE).
