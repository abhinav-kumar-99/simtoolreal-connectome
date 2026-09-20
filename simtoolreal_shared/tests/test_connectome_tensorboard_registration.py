from pathlib import Path

import pytest


def test_register_tensorboard_run_creates_and_reuses_expected_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_connectome_suite

    monkeypatch.setattr(run_connectome_suite, "REPOSITORY_ROOT", tmp_path)
    summaries = tmp_path / "runs" / "case" / "summaries"
    registration = run_connectome_suite._register_tensorboard_run(
        {}, "00_suite_case_seed42", summaries
    )

    assert registration.is_symlink()
    assert registration.resolve(strict=False) == summaries.resolve(strict=False)
    assert run_connectome_suite._register_tensorboard_run(
        {}, "00_suite_case_seed42", summaries
    ) == registration


def test_register_tensorboard_run_rejects_a_name_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_connectome_suite

    monkeypatch.setattr(run_connectome_suite, "REPOSITORY_ROOT", tmp_path)
    root = tmp_path / run_connectome_suite.TENSORBOARD_6008_LOG_ROOT
    root.mkdir(parents=True)
    (root / "00_suite_case_seed42").symlink_to(tmp_path / "different")

    with pytest.raises(RuntimeError, match="registration collision"):
        run_connectome_suite._register_tensorboard_run(
            {}, "00_suite_case_seed42", tmp_path / "runs" / "case" / "summaries"
        )
