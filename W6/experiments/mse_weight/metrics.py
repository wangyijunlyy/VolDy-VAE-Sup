"""Per-case marginal, aggregate and full-trajectory scores."""
import math

import torch


def _check(targets, forecasts):
    if targets.ndim != 3 or forecasts.ndim != 4:
        raise ValueError('Expected targets[B,H,C] and forecasts[B,S,H,C].')
    if forecasts.shape[0] != targets.shape[0] or forecasts.shape[2:] != targets.shape[1:]:
        raise ValueError('Target and forecast dimensions differ.')
    if forecasts.shape[1] < 1:
        raise ValueError('At least one forecast sample is required.')


def marginal_crps(targets, forecasts, quantiles_num=20):
    """Benchmark normalized quantile CRPS, returned separately for each case."""
    _check(targets, forecasts)
    # Sorting once reproduces torch.quantile's linear interpolation at all levels.
    ordered = forecasts.sort(dim=1).values
    levels = torch.arange(1, quantiles_num, device=forecasts.device,
                          dtype=forecasts.dtype) / quantiles_num
    positions = levels * (forecasts.shape[1] - 1)
    lower, upper = positions.floor().long(), positions.ceil().long()
    fraction = (positions - lower).view(1, -1, 1, 1)
    q = ordered.index_select(1, lower)
    q = q + fraction * (ordered.index_select(1, upper) - q)
    residual = q - targets[:, None]
    pinball = 2 * (residual * ((targets[:, None] <= q).to(q.dtype)
                              - levels.view(1, -1, 1, 1))).abs()
    denom = targets.abs().sum((1, 2)).clamp_min(1e-8)
    return pinball.sum((2, 3)).mean(1) / denom


def crps_sum(targets, forecasts, quantiles_num=20):
    """Sum variables within each sample BEFORE scoring the aggregate forecast."""
    _check(targets, forecasts)
    return marginal_crps(targets.sum(-1, keepdim=True),
                         forecasts.sum(-1, keepdim=True), quantiles_num)


def energy_score(targets, forecasts):
    """Empirical energy score on full H*C vectors, divided by sqrt(H*C).

    Uses all S*S pairs (including zero diagonal). Distances and accumulation use
    float64, and the diagonal is explicitly zeroed after the cdist GEMM path.
    Inputs must already be transformed by the TRAINING-split variable scaler.
    """
    _check(targets, forecasts)
    samples = forecasts.flatten(2).double()
    truth = targets.flatten(1).double()
    first = torch.linalg.vector_norm(samples - truth[:, None], dim=-1).mean(1)
    distances = torch.cdist(samples, samples, p=2,
                            compute_mode='use_mm_for_euclid_dist')
    distances.diagonal(dim1=1, dim2=2).zero_()
    second = distances.mean((1, 2))
    return (first - second / 2) / math.sqrt(truth.shape[1])


def score_batch(raw_targets, raw_forecasts, standardized_targets, standardized_forecasts):
    # For an even sample count, torch.median differs from the benchmark median.
    median = torch.quantile(raw_forecasts, .5, dim=1)
    denom = raw_targets.abs().sum((1, 2)).clamp_min(1e-8)
    return {
        'CRPS': marginal_crps(raw_targets, raw_forecasts),
        'ND': (raw_targets - median).abs().sum((1, 2)) / denom,
        'CRPS_sum': crps_sum(raw_targets, raw_forecasts),
        'CRPS_sum_standardized': crps_sum(standardized_targets, standardized_forecasts),
        'EnergyScore_joint': energy_score(standardized_targets, standardized_forecasts),
    }
