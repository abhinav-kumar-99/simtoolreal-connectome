"""Capacity-run orchestration must survive telemetry and child failures."""
import json
import subprocess

from scripts import benchmark_visual_reservoir as benchmark


def test_telemetry_timeout_is_recorded_without_abandoning_child(tmp_path, monkeypatch):
    class Child:
        def __init__(self):
            self.calls = 0
            self.waited = False

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else 1

        def wait(self):
            self.waited = True
            return 1

    child = Child()
    monkeypatch.setattr(benchmark.subprocess, 'Popen', lambda *a, **k: child)

    def fail_query(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 10)

    monkeypatch.setattr(benchmark.subprocess, 'run', fail_query)
    monkeypatch.setattr(benchmark.time, 'sleep', lambda _: None)
    monkeypatch.setattr(benchmark, 'summarize', lambda *a: None)
    benchmark.sweep(dict(cases=[dict(name='probe', mode='environment')],
                         monitor_memory=True, continue_on_failure=True), tmp_path)
    result = json.loads((tmp_path / 'probe/process_result.json').read_text())
    assert child.waited
    assert result['returncode'] == 1
    assert len(result['monitor_errors']) == 1
    assert result['peak_gpu_used_mib'] is None


def test_timeout_terminates_only_owned_process_group(tmp_path, monkeypatch):
    class Child:
        pid = 77777

        def poll(self):
            return None

        def wait(self, timeout=None):
            return -15

    monkeypatch.setattr(benchmark.subprocess, 'Popen', lambda *a, **k: Child())
    clock = iter([0., 2., 3.])
    monkeypatch.setattr(benchmark.time, 'monotonic', lambda: next(clock))
    signals = []
    monkeypatch.setattr(benchmark.os, 'killpg', lambda *args: signals.append(args))
    monkeypatch.setattr(benchmark, 'summarize', lambda *a: None)
    benchmark.sweep(dict(cases=[dict(name='probe', mode='environment')],
                         timeout_seconds=1, continue_on_failure=True), tmp_path)
    result = json.loads((tmp_path / 'probe/process_result.json').read_text())
    assert result['timed_out']
    assert signals == [(77777, benchmark.signal.SIGTERM)]
