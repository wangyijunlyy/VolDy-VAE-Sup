"""Portable launcher for the recorded W6 configurations; no model changes."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parent


def export_best(run_dir):
    import torch
    from types import SimpleNamespace
    from mse_weight.core import GaussianVAE_MSE
    paths = list(run_dir.glob('*/ckpt/epoch=*.ckpt'))
    if len(paths) != 1:
        raise RuntimeError('Expected one validation-selected best checkpoint: ' + str(paths))
    checkpoint = torch.load(paths[0], map_location='cpu', weights_only=False)
    prefix = 'forecaster.model.'
    state = {k[len(prefix):]: v for k, v in checkpoint['state_dict'].items()
             if k.startswith(prefix)}
    config = checkpoint['joint_model_config']
    model = GaussianVAE_MSE(SimpleNamespace(**config))
    model.load_state_dict(state, strict=True)
    destination = run_dir / 'weights/best_model_state.pt'
    torch.save(dict(state_dict=state, config=config, best_epoch=int(checkpoint['epoch']),
                    source_checkpoint=str(paths[0])), destination)
    exported = torch.load(destination, map_location='cpu', weights_only=True)
    model.load_state_dict(exported['state_dict'], strict=True)
    assert all(torch.equal(v, exported['state_dict'][k]) for k, v in state.items())
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['table2', 'kl_pair'], default='table2')
    parser.add_argument('--framework-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--only', help='One config stem; omit to run the full suite sequentially.')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES for the child process.')
    parser.add_argument('--prepare-only', action='store_true', help='Resolve configs without training.')
    args = parser.parse_args()
    framework = args.framework_root.resolve()
    if not (framework / 'probts').is_dir():
        parser.error('--framework-root must contain the probts package.')
    configs = sorted((ROOT / 'configs' / args.suite).glob('*.yaml'))
    if args.only:
        configs = [p for p in configs if p.stem == args.only]
    if not configs:
        parser.error('No matching configuration.')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for template in configs:
        run_dir = output / template.stem
        if run_dir.exists():
            raise FileExistsError('Use a new output directory: ' + str(run_dir))
        run_dir.mkdir()
        config = yaml.safe_load(template.read_text())
        config['data']['data_manager']['init_args']['path'] = str(args.data_root.resolve())
        config['trainer']['default_root_dir'] = str(run_dir)
        resolved = run_dir / 'config.yaml'
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, '-m', 'mse_weight.entry', '--config', str(resolved),
                   '--trainer.default_root_dir', str(run_dir)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                   MSE_TRAIN_SEED=str(config['seed_everything']),
                   PYTHONPATH=os.pathsep.join([str(ROOT), str(framework)]),
                   PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   TF_CPP_MIN_LOG_LEVEL='3')
        print(' '.join(command), flush=True)
        if args.prepare_only:
            continue
        with (run_dir / 'train.log').open('w') as log:
            subprocess.run(command, cwd=framework, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        weights = export_best(run_dir)
        result_path = next(run_dir.glob('*/horizons_results.csv'))
        result = next(csv.DictReader(result_path.open()))
        record = dict(name=template.stem, weights=weights, results=result)
        (run_dir / 'completed.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
        (output / 'completed_runs.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
