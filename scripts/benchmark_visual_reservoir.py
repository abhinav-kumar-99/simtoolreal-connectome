#!/usr/bin/env python3
"""YAML-owned correctness, CUDA timing, and real PPO batch-size experiments."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'rl_games'))
import yaml


def kernel_bench(cfg, output):
    import numpy as np
    import torch
    from rl_games.algos_torch.connectome_ops import Graph
    from rl_games.algos_torch.connectome_triton import fused_step, FrozenTanhRunner
    from simtoolreal_shared.vision_ops import rgba_luminance
    from simtoolreal_shared.retina import RetinalPopulationEncoder
    torch.manual_seed(42)
    device = 'cuda:0'
    with np.load(cfg['artifact_path']) as data:
        crow = torch.tensor(data['crow_indices'], device=device, dtype=torch.long)
        col = torch.tensor(data['col_indices'], device=device, dtype=torch.long)
        values = torch.tensor(data['values'], device=device)
        si = torch.tensor(data['sensory_indices'], device=device, dtype=torch.long)
        di = torch.tensor(data['descending_indices'], device=device, dtype=torch.long)
    graph = Graph(crow, col)
    n = len(crow) - 1
    gi = go = torch.ones(n, device=device)
    leak = torch.full((n,), 1. - .5 ** (1./9.), device=device)
    bias = torch.zeros(n, device=device)
    rows = []

    def timed(fn):
        for _ in range(cfg['warmup']):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(cfg['iterations']):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return statistics.median(samples)

    with torch.no_grad():
        for batch in cfg['batches']:
            h = torch.randn(batch, n, device=device) * .01
            s = torch.randn(batch, len(si), device=device) * .1
            d = torch.randn(batch, len(di), device=device) * .1
            def baseline():
                state = h
                for _ in range(9):
                    state = fused_step(graph, values, state, gi, go, leak, bias, s, d, si, di, .9)
                return state
            reference = baseline()
            rows.append(dict(batch=batch, variant='baseline', milliseconds=timed(baseline)))
            for capture in (False, True):
                runner = FrozenTanhRunner(graph, values, h, gi, go, leak, bias, s, d, si, di, .9, 9, capture)
                actual = runner(h, s, d)
                torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
                saved = actual.clone()
                # Verify evolving inputs, resets and returned-state ownership.
                h[:min(2, batch)].zero_()
                s.mul_(.9)
                changed = baseline()
                torch.testing.assert_close(runner(h, s, d), changed, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual, saved, atol=0, rtol=0)
                rows.append(dict(batch=batch, variant='cuda_graph' if capture else 'frozen',
                                 milliseconds=timed(lambda: runner(h, s, d)),
                                 max_error=float((runner(h, s, d) - changed).abs().max())))
                reference = changed
                del runner, actual, saved, changed
                gc.collect()
                torch.cuda.empty_cache()
            del h, s, d, reference
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps(rows[-3:]), flush=True)

        rgba = torch.randint(0, 256, (384, 36, 64, 4), dtype=torch.uint8, device=device)
        def gray_reference():
            rgb = rgba[..., :3].float() / 255.
            return (rgb * rgb.new_tensor([.299, .587, .114])).sum(-1)
        gray = gray_reference()
        torch.testing.assert_close(rgba_luminance(rgba), gray, atol=2e-7, rtol=2e-6)
        rows.append(dict(variant='luminance_reference', milliseconds=timed(gray_reference)))
        rows.append(dict(variant='luminance_fused', milliseconds=timed(lambda: rgba_luminance(rgba))))
        class Empty(torch.nn.Module):
            def forward(self, obs):
                return obs.new_zeros((len(obs), 3600))
        positions = torch.arange(3534, device=device)
        grid = torch.rand(3534, 2, device=device) * 2 - 1
        grid[:4] = torch.tensor([[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]], device=device)
        obs = torch.rand(384, 2404, device=device)
        regular = RetinalPopulationEncoder(Empty(), positions, grid, image_start=99, height=36, width=64).to(device)
        fused = RetinalPopulationEncoder(Empty(), positions, grid, image_start=99, height=36, width=64, fused=True).to(device)
        torch.testing.assert_close(fused(obs), regular(obs), atol=1e-5, rtol=2e-5)
        rows.append(dict(variant='retina_reference', milliseconds=timed(lambda: regular(obs))))
        rows.append(dict(variant='retina_fused', milliseconds=timed(lambda: fused(obs))))
    (output / 'kernel_results.json').write_text(json.dumps(rows, indent=2))


def environment_bench(cfg, output):
    from isaacgym import gymapi  # import before torch
    import torch
    from deployment.isaac.isaac_env import create_env
    import isaacgymenvs.tasks.simtoolreal.env as env_module
    render_times = []
    original = env_module.render_camera_sensors_for_current_step
    def render(*args):
        torch.cuda.synchronize()
        start = time.perf_counter()
        original(*args)
        torch.cuda.synchronize()
        render_times.append((time.perf_counter() - start) * 1000)
    env_module.render_camera_sensors_for_current_step = render
    env = create_env(cfg['policy_config_path'], device='cuda:0', headless=True,
        overrides={'task.env.numEnvs': cfg['batch'], 'task.env.capture_video': False})
    env.vision_config['fusedLuminance'] = cfg.get('fused', False)
    actions = torch.zeros((cfg['batch'], env.num_acts), device='cuda:0')
    rows = []
    for i in range(cfg['steps'] + 8):
        torch.cuda.synchronize()
        start = time.perf_counter()
        before = len(render_times)
        env.step(actions)
        torch.cuda.synchronize()
        if i >= 8:
            rows.append(dict(milliseconds=(time.perf_counter()-start)*1000,
                             rendered=len(render_times) > before,
                             render_ms=render_times[-1] if len(render_times)>before else 0))
    result = dict(batch=cfg['batch'], fused=cfg.get('fused', False), samples=rows)
    (output / 'environment_results.json').write_text(json.dumps(result, indent=2))
    env.gym.destroy_sim(env.sim)


def sweep(cfg, output):
    for case in cfg['cases']:
        child = dict(cfg)
        child.update(case)
        child.pop('cases')
        directory = output / case['name']
        completed = (directory / 'kernel_results.json' if child['mode'] == 'kernel' else
                     directory / 'environment_results.json' if child['mode'] == 'environment' else
                     directory / 'training' / 'suite_results.json')
        if completed.exists() and cfg.get('resume', False):
            print('Already complete', case['name'], flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=cfg.get('resume', False))
        child['output_directory'] = str(directory)
        path = directory / 'benchmark.yaml'
        path.write_text(yaml.safe_dump(child))
        log_path = directory / ('process_' + str(time.time_ns()) + '.log')
        with log_path.open('w') as log:
            if child['mode'] == 'train':
                suite = yaml.safe_load(Path(cfg['base_suite']).read_text())
                suite['name'] = 'vision_perf_' + case['name']
                suite['output_directory'] = str(directory / 'training')
                train = suite['training']
                b = child['batch']
                train.update(num_envs=b, sapg_block_size=b//6, epochs=cfg['epochs'],
                             max_frames=b*16*cfg['epochs'], minibatch_size=b*4,
                             central_critic_minibatch_size=b*4, actor_microbatch_size=b*4,
                             central_critic_microbatch_size=b*4, save_frequency=cfg['epochs'],
                             inference_checkpoint_interval_frames=b*16*cfg['epochs'])
                train['overrides'].update({
                    '++train.params.network.connectome.backend_options.frozen_inference': child.get('frozen', False),
                    '++train.params.network.connectome.backend_options.cuda_graph': child.get('capture', False),
                    '++train.params.network.connectome.fixed_input_encoder.retina.fused': child.get('fused', False),
                    '++task.env.policyVision.fusedLuminance': child.get('fused', False),
                })
                suite_path = directory / 'suite.yaml'
                suite_path.write_text(yaml.safe_dump(suite, sort_keys=False))
                command = [sys.executable, 'scripts/run_connectome_suite.py', '--config', str(suite_path)]
            else:
                command = [sys.executable, __file__, '--config', str(path)]
            print('Starting', case['name'], flush=True)
            environment = os.environ.copy()
            if child['mode'] == 'train':
                # The suite assigns physical devices itself and verifies on cuda:gpu.
                environment.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                environment['CUDA_VISIBLE_DEVICES'] = str(cfg.get('gpu', 1))
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment)
            print('Finished', case['name'], 'exit', result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError('Benchmark failed; inspect ' + str(log_path))
    summarize(cfg, output)


def summarize(cfg, output):
    """Summarize warmed epochs; keep raw logs for timing provenance."""
    rows = []
    for case in cfg['cases']:
        directory = output / case['name']
        if case['mode'] == 'train':
            logs = sorted(directory.glob('process*.log'))
            if not logs:
                continue
            content = logs[-1].read_text()
            row = dict(case=case['name'], batch=case['batch'])
            for label, key in [('fps step', 'step_fps'),
                               ('fps step + policy inference', 'rollout_fps'),
                               ('fps total', 'total_fps')]:
                values = [float(v.replace(',', '')) for v in re.findall(
                    re.escape(label) + r'\s+:\s+([\d,]+)', content)]
                values = values[cfg.get('discard_epochs', 2):]
                row[key] = statistics.median(values) if values else None
                row['measured_epochs'] = len(values)
            rows.append(row)
        elif case['mode'] == 'environment':
            result = json.loads((directory / 'environment_results.json').read_text())
            row = dict(case=case['name'], batch=case['batch'])
            for rendered in (False, True):
                samples = [v['milliseconds'] for v in result['samples'] if v['rendered'] == rendered]
                row['rendered_ms' if rendered else 'cached_ms'] = statistics.mean(samples)
            rows.append(row)
    (output / 'summary.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    output = Path(cfg['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    {'kernel': kernel_bench, 'environment': environment_bench, 'sweep': sweep}[cfg['mode']](cfg, output)
