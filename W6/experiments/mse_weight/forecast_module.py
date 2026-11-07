import csv
import json
from pathlib import Path

import numpy as np
import torch

from probts.data import ProbTSBatchData
from probts.model.forecast_module import ProbTSForecastModule
from probts.model.forecaster import Forecaster
from probts.data.data_utils.data_scaler import Scaler
from probts.utils.save_utils import update_metrics, get_hor_str

from .metrics import marginal_crps, score_batch


class JointForecastModule(ProbTSForecastModule):
    def __init__(self, forecaster: Forecaster, scaler: Scaler = None,
                 train_pred_len_list: list = None, num_samples: int = 100,
                 learning_rate: float = 1e-3, quantiles_num: int = 20,
                 load_from_ckpt: str = None, sampling_weight_scheme: str = 'none',
                 optimizer_config=None, lr_scheduler_config=None, **kwargs):
        super().__init__(forecaster=forecaster, scaler=scaler,
                         train_pred_len_list=train_pred_len_list, num_samples=num_samples,
                         learning_rate=learning_rate, quantiles_num=quantiles_num,
                         load_from_ckpt=load_from_ckpt, sampling_weight_scheme=sampling_weight_scheme,
                         optimizer_config=optimizer_config, lr_scheduler_config=lr_scheduler_config,
                         **kwargs)

    def on_fit_start(self):
        directory = Path(self.trainer.default_root_dir) / 'weights'
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu().clone() for k, v in self.forecaster.model.state_dict().items()},
                   directory / 'initial_state.pt')

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        for key, value in self.forecaster.last_loss_components.items():
            self.log('train_' + key, value, on_step=True, on_epoch=True, batch_size=int(batch['past_target_cdf'].shape[0]))
        return result

    def on_save_checkpoint(self, checkpoint):
        checkpoint['joint_model_config'] = vars(self.forecaster.model.config)
        checkpoint['joint_model_class'] = type(self.forecaster.model).__module__ + '.' + type(self.forecaster.model).__name__

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.case_records = []
        self.batch_index = 0

    def evaluate(self, batch, stage='', dataloader_idx=None):
        data = ProbTSBatchData(batch, self.device)
        raw_history = data.past_target_cdf
        raw_target = data.future_target_cdf
        norm_history = self.scaler.transform(raw_history)
        norm_target = self.scaler.transform(raw_target)
        data.past_target_cdf = norm_history
        self.batch_size.append(len(raw_target))
        if stage == 'test':
            # Reset only test sampling, independently of the completed training RNG.
            # The launch environment records the corresponding training seed.
            import os
            seed = int(os.environ['MSE_TRAIN_SEED'])
            sample_seed = 100000 + 1000 * seed + self.batch_index
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
        forecasts = self.forecaster.forecast(data, self.num_samples)
        expected = (len(raw_target), self.num_samples, raw_target.shape[1], raw_target.shape[2])
        if tuple(forecasts.shape) != expected:
            raise ValueError('Incorrect native forecast shape: {} != {}'.format(tuple(forecasts.shape), expected))
        raw_forecasts = self.scaler.inverse_transform(forecasts)
        if not torch.isfinite(forecasts).all():
            raise ValueError('Non-finite forecast samples.')
        if stage != 'test':
            values = {'CRPS': float(marginal_crps(raw_target, raw_forecasts).mean())}
            self.metrics_dict = update_metrics(values, stage, target_dict=self.metrics_dict)
            return values

        scores = score_batch(raw_target, raw_forecasts, norm_target, forecasts)
        for key, score in scores.items():
            if not torch.isfinite(score).all():
                raise ValueError('Non-finite metric: ' + key)
        values = {key: float(value.mean()) for key, value in scores.items()}
        horizon = get_hor_str(self.forecaster.prediction_length, dataloader_idx)
        self.hor_metrics.setdefault(horizon, {})
        self.hor_metrics[horizon] = update_metrics(values, stage, target_dict=self.hor_metrics[horizon])
        start = len(self.case_records)
        arrays = {key: value.detach().cpu().tolist() for key, value in scores.items()}
        for i in range(len(raw_target)):
            self.case_records.append(dict(case_index=start + i, batch_index=self.batch_index,
                                          sampling_seed=sample_seed,
                                          **{key: value[i] for key, value in arrays.items()}))
        if self.batch_index == 0:
            directory = Path(self.trainer.default_root_dir) / 'evaluation_artifacts'
            directory.mkdir(exist_ok=True, parents=True)
            np.savez_compressed(directory / 'first_case_samples.npz',
                                normalized_past_batch=norm_history.detach().cpu().numpy(),
                                normalized_target_first=norm_target[:1].cpu().numpy(),
                                normalized_draws_first=forecasts[:1].cpu().numpy(),
                                raw_target_first=raw_target[:1].cpu().numpy(),
                                sampling_seed=np.array(sample_seed))
        self.batch_index += 1
        return values

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        directory = Path(self.trainer.default_root_dir) / 'evaluation_artifacts'
        directory.mkdir(exist_ok=True, parents=True)
        with (directory / 'scores_per_case.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.case_records[0]))
            writer.writeheader()
            writer.writerows(self.case_records)
        (directory / 'evaluation_metadata.json').write_text(json.dumps(dict(
            n_cases=len(self.case_records), samples=self.num_samples,
            context_length=self.forecaster.context_length,
            horizon=self.forecaster.prediction_length, variables=self.forecaster.target_dim,
            energy_dimensions='entire standardized H*C vector',
            energy_pair_term='all S*S pairs, including zero diagonal',
            energy_normalization='sqrt(H*C)', scaler='fitted using training split only',
            test_workers=0, quantiles=[i / 20 for i in range(1, 20)],
        ), indent=2))
