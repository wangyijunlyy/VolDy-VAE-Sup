"""Read-only posterior, activation, history-response and gradient diagnostics."""
import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--framework-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--initial-state', type=Path, help='Optional saved initial_state.pt for initial gradient probe.')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    sys.path.insert(0, str(args.framework_root.resolve()))
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    import yaml
    from mse_weight.core import GaussianVAE_MSE
    from probts.data.data_manager import DataManager

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    config = yaml.safe_load(args.config.read_text())
    data = config['data']['data_manager']['init_args']
    data['path'] = str(args.data_root.resolve())
    dm = DataManager(**data)
    bundle = torch.load(args.weights, map_location='cpu', weights_only=True)
    model_config = bundle.get('config', bundle.get('model_config'))
    state = bundle.get('state_dict', bundle.get('model_state_dict'))
    model = GaussianVAE_MSE(SimpleNamespace(**model_config)).to(args.device).eval()
    model.load_state_dict(state, strict=True)
    metrics, latents, activation = {}, [], {}

    def add(name, value):
        metrics.setdefault(name, []).extend(value.cpu().double().tolist())

    def rms(value):
        return value.square().mean((1, 2)).sqrt()

    with torch.no_grad():
        for index, batch in enumerate(DataLoader(dm.test_iter_dataset, batch_size=4, num_workers=0)):
            x = dm.scaler.transform(batch['past_target_cdf'][:, -model.input_len:].to(args.device))
            y = dm.scaler.transform(batch['future_target_cdf'].to(args.device))
            torch.manual_seed(100001 + index)
            samples = model.sample(x, 100)
            median = torch.quantile(samples, .5, dim=1)
            # Native MLP reuses one ReLU object at multiple layer positions.
            # Inspect sequential outputs by position instead of module hooks.
            normalized = model.revin(x, 'norm')
            length = normalized.shape[1]
            padded = torch.cat([normalized[:, length-model.padding_len:], normalized], dim=1)
            patches = torch.stack(padded.chunk(model.freq, dim=1), dim=1)
            activation_value = patches.reshape(len(x), model.freq, -1)
            for position, layer in enumerate(model.encoder.layers):
                activation_value = layer(activation_value)
                if isinstance(layer, torch.nn.ReLU):
                    row = activation.setdefault(str(position), dict(zero=0, n=0))
                    row['zero'] += int((activation_value == 0).sum())
                    row['n'] += activation_value.numel()
            mu, logvar, _ = model.encode(x)
            _, forecast = model.decode(mu)
            reverse_mu, _, _ = model.encode(x.flip(1))
            _, reverse_forecast = model.decode(reverse_mu)
            kl = -.5 * (1 + logvar.double() - mu.double().square()
                         - logvar.double().exp()).sum(-1).mean(1)
            add('kl', kl)
            add('posterior_std', (.5 * logvar).exp().mean((1, 2)))
            add('reverse_history_rmse', rms(forecast - reverse_forecast))
            add('forecast_temporal_std', median.std(1, unbiased=False).mean(1))
            add('target_temporal_std', y.std(1, unbiased=False).mean(1))
            latents.append(mu.cpu().reshape(-1, mu.shape[-1]))
    joined = torch.cat(latents).double()
    report = dict(n_cases=len(metrics['kl']),
                  summaries={k:float(np.mean(v)) for k,v in metrics.items()},
                  latent_mean_variance=float(joined.var(0, unbiased=False).mean()),
                  activation={k:dict(zero_fraction=v['zero']/v['n']) for k,v in activation.items()},
                  units='Training-split standardized coordinates; no hyperparameter selection.')
    if args.initial_state:
        initial = torch.load(args.initial_state, map_location='cpu', weights_only=True)
        model.load_state_dict(initial, strict=True)
        model.train()
        torch.manual_seed(1)
        np.random.seed(1)
        batch = next(iter(DataLoader(dm.train_iter_dataset, batch_size=4, num_workers=0)))
        x = dm.scaler.transform(batch['past_target_cdf'][:, -model.input_len:].to(args.device))
        y = dm.scaler.transform(batch['future_target_cdf'].to(args.device))
        torch.manual_seed(77)
        rec, pred, mu, logvar = model(x)
        beta = config['model']['forecaster']['init_args']['weight_beta']
        terms = dict(prediction_mse=F.mse_loss(pred, y),
                     reconstruction_mse=F.mse_loss(rec, x),
                     weighted_kl=beta * model.kl_divergence(mu, logvar))
        gradients = {}
        for name, value in terms.items():
            grad = torch.autograd.grad(value, list(model.encoder.parameters()),
                                       retain_graph=True, allow_unused=True)
            gradients[name] = dict(value=float(value), encoder_gradient_norm=math.sqrt(
                sum(float(g.detach().double().square().sum()) for g in grad if g is not None)))
        report['initial_gradient_probe'] = gradients
        report['initial_gradient_probe_scope'] = 'One fixed four-example training batch.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
