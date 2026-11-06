"""Run the released K2VAE inference bundles on raw multivariate observations."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from k2vae_runtime import k2VAE


class Predictor:
    def __init__(self, weights, device='cpu'):
        self.device = torch.device(device)
        bundle = torch.load(weights, map_location='cpu', weights_only=True)
        if bundle['format_version'] != 1:
            raise ValueError('Unsupported weight format')
        self.config = SimpleNamespace(**bundle['config'])
        self.metadata = bundle['metadata']
        self.model = k2VAE(self.config).to(self.device).eval()
        self.model.load_state_dict(bundle['state_dict'], strict=True)
        self.mean = bundle['scaler']['mean'].to(self.device)
        self.scale = bundle['scaler']['scale'].to(self.device)
        self.epsilon = bundle['scaler']['epsilon']

    @torch.no_grad()
    def predict(self, history, num_samples=100):
        """Raw [B,96,C] or [96,C] -> original-unit samples [B,S,H,C]."""
        x = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        expected = (self.config.seq_len, self.config.n_vars)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f'Expected [batch,{expected[0]},{expected[1]}], got {tuple(x.shape)}')
        if num_samples < 1:
            raise ValueError('num_samples must be positive')
        if not torch.isfinite(x).all():
            raise ValueError('Input history contains nonfinite observations')
        # Same float32 observations and fitted training scaler as ProbTS.
        norm = ((x-self.mean)/(self.scale+self.epsilon)).to(torch.float32)
        samples = self.model.sample(norm,num_samples).permute(1,0,2,3)
        return samples*(self.scale+self.epsilon)+self.mean


def read_history(path, predictor, origin=None):
    """CSV uses stored feature order; origin is the first unobserved row."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        frame = pd.read_csv(path)
        names = predictor.metadata['feature_names']
        missing = set(names)-set(frame.columns)
        if missing:
            raise ValueError(f'CSV is missing {len(missing)} trained feature columns')
        values = frame.loc[:,names].fillna(0).to_numpy(dtype=np.float32)
    else:
        values = np.load(path,allow_pickle=False)
    if values.ndim == 2:
        end = len(values) if origin is None else origin
        if end < predictor.config.seq_len or end > len(values):
            raise ValueError('origin must leave a full observed context inside the input')
        values = values[end-predictor.config.seq_len:end]
    elif origin is not None:
        raise ValueError('--origin applies to a single [time,channels] sequence')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True,help='Benchmark CSV or raw-value .npy')
    parser.add_argument('--output',type=Path,required=True,help='Output .npz')
    parser.add_argument('--origin',type=int,help='First unobserved row; defaults to the end of input')
    parser.add_argument('--num-samples',type=int,default=100)
    parser.add_argument('--seed',type=int,default=0,help='Forecast sampling seed')
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    torch.set_float32_matmul_precision('high')
    predictor = Predictor(args.weights,args.device)
    values = read_history(args.input,predictor,args.origin)
    if values.ndim == 2:
        values = values[None]
    torch.manual_seed(args.seed)
    chunks = [predictor.predict(values[i:i+args.batch_size],args.num_samples).cpu().numpy()
              for i in range(0,len(values),args.batch_size)]
    samples = np.concatenate(chunks,axis=0)
    quantile_levels = np.array([0.05,0.5,0.95])
    quantiles = np.quantile(samples,quantile_levels,axis=1).transpose(1,0,2,3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as handle:
        np.savez_compressed(handle,samples=samples,mean=samples.mean(axis=1),
                            median=quantiles[:,1],quantiles=quantiles,
                            quantile_levels=quantile_levels,
                            feature_names=np.asarray(predictor.metadata['feature_names']))
    print(f'Saved samples {samples.shape} in original data units to {args.output}')


if __name__=='__main__':
    main()
