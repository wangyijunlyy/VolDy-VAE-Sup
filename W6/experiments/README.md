# W6: Vanilla MSE control, KL intervention, and Table 2 extension

This directory releases the training and scoring implementation used for the W6 experiments. The original Vanilla VAE (MSE) column in Table 2 is retained; the supplementary configuration is an additional **Vanilla (KL = 0)** column.

## Contents

- `mse_weight/`: the actual full model, MSE wrapper, training entry point, held-out data loaders, and marginal/joint scoring code used in the completed experiments. These files are copied without changing their computations. Only the native MLP/Koopman and RevIN dependency files imported by this model are included.
- `configs/table2/`: eight configurations for ETTh1/Electricity, history 96, horizons 96/192/336/720, reconstruction weight 1, prediction weight 1, and KL coefficient 0.
- `configs/kl_pair/`: Electricity H720 configurations with reconstruction weight 0.2 and KL coefficient 0.01 or 0. The fixed-reconstruction pair tests the effect of KL; it is separate from the reconstruction-weight-1 Table 2 extension.
- `run.py`: portable sequential launcher that resolves data/output paths, trains, evaluates the validation-selected checkpoint, and exports its actual trained `state_dict`.
- `diagnose.py`: read-only posterior, ReLU activation, history-reversal, and optional initial-gradient probes. It accepts trained tensor bundles, not a model object serialized in logger hyperparameters.
- `recorded_results/`: completed numerical records. `table2.csv` retains the measured values; machine-specific checkpoint paths are omitted. Training seeds remain in machine-readable configurations and records.

## Environment and framework

Use Python 3.10 and the K²VAE/ProbTS framework checkout containing `probts/`. The training server used [ProbTSF-NLL](https://github.com/wangyijunlyy/ProbTSF-NLL), based at commit `fc8da55ea4df5ddd83c0fa28fe91704ca1c637d0`, with the experiment-specific implementation frozen here. Install that framework's dependencies in its environment before running these scripts. `environment.json` records the principal installed versions used for the experiments; it is not a complete dependency lockfile. The portable commands below were checked in the existing experiment environment.

Data are supplied separately. `--data-root` is the benchmark dataset directory passed to `probts.data.data_manager.DataManager`, with the same layout as the framework's `datasets/` directory. The dataset keys are `etth1` and `electricity_ltsf`.

## Run the eight Table 2 additions

From this directory, activate the framework environment and run:

```bash
python run.py --suite table2 \
  --framework-root /path/to/ProbTSF-NLL \
  --data-root /path/to/datasets \
  --output-root /path/to/new/table2_results --gpu 0
```

This runs the eight settings sequentially on the selected GPU. For one setting, add `--only electricity_ltsf_h720_s1` or another configuration filename stem. Add `--prepare-only` to write resolved configurations and print commands without training. Existing per-setting output directories are not overwritten.

Training uses Adam at 0.001, batch 64, 100 training batches per epoch, at most 50 epochs, gradient clipping 0.5, and validation-CRPS early stopping with patience 10. Held-out loaders evaluate every selected window once with batch 4 and zero workers. Scaling is fitted on the training split. Forecast origins have stride 96, and evaluation uses 100 latent forecast samples.

For each run, the output includes resolved `config.yaml`, `train.log`, Lightning fit/test metric logs, `ckpt/epoch=...ckpt`, `ckpt/last.ckpt`, `weights/initial_state.pt`, exported `weights/best_model_state.pt`, `evaluation_artifacts/scores_per_case.csv`, and `completed.json`. Model weights are exported from checkpoint `state_dict` entries prefixed `forecaster.model.`, strictly reloaded, and checked tensor by tensor. Logger `hparams_initial` snapshots are not used to recover trained weights.

## Reproduce the fixed-reconstruction KL intervention

```bash
python run.py --suite kl_pair \
  --framework-root /path/to/ProbTSF-NLL \
  --data-root /path/to/datasets \
  --output-root /path/to/new/kl_pair_results --gpu 0
```

The two configurations differ only in the KL coefficient. They share initialization and data settings. The recorded validation-selected results were CRPS 0.363658 / ND 0.364984 for KL 0.01 and CRPS 0.112205 / ND 0.112897 for KL 0. Exact aggregate results can vary with platform and stochastic sampling.

## Inspect posterior collapse

```bash
python diagnose.py \
  --framework-root /path/to/ProbTSF-NLL \
  --data-root /path/to/datasets \
  --config configs/kl_pair/electricity_h720_kl_001.yaml \
  --weights /path/to/kl_pair_results/electricity_h720_kl_001/weights/best_model_state.pt \
  --initial-state /path/to/kl_pair_results/electricity_h720_kl_001/weights/initial_state.pt \
  --output /path/to/diagnostics.json --device cuda:0
```

`--initial-state` is optional. It enables a separate encoder-gradient probe on one fixed four-example training batch. The main diagnostics use all held-out windows and report posterior KL/std, posterior-mean variance, ReLU zero fractions, deterministic forecast response to reversing history, and forecast/target temporal standard deviations in training-standardized coordinates. These probes do not select hyperparameters.

The existing [Electricity training archive](../vanilla_vae_mse/electricity_L96_H720/README.md) provides separately labeled historical logs and an actual trained KL-regularized rerun checkpoint. This code release does not relabel those earlier artifacts or replace their files.
