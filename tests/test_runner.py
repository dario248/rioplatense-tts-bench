"""Runner behaviour, exercised end to end against the white-noise adapter.

These are the tests that had to exist before writing a real adapter: discovering an idempotency
bug after an hour of GPU time is exactly what they prevent.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from rtb.config import BenchConfig
from rtb.generate import dummy
from rtb.generate.runner import GridFilter, build_grid, filter_grid, main, run
from rtb.manifest import index_by_generation_id, read_rows
from rtb.schema import GenerationRow


def manifest_rows(config: BenchConfig) -> list[GenerationRow]:
    return read_rows(config.resolve(config.model.manifest_path))


def test_grid_is_the_cartesian_product(mini_config: BenchConfig) -> None:
    grid = build_grid(mini_config)
    # 1 hablante x 2 frases x 1 clip de referencia x 2 seeds
    assert len(grid) == 4
    assert len({item.generation_id for item in grid}) == 4
    assert len({item.wav_rel_path for item in grid}) == 4
    assert grid[0].generation_id == "dummy__slr61_f_01__yei_001__s11__ref0"
    assert grid[0].wav_rel_path == Path("audio/dummy/slr61_f_01/yei_001_s11_ref0.wav")


def test_grid_order_is_stable(mini_config: BenchConfig) -> None:
    # Un orden estable es lo que hace que --limit signifique siempre el mismo prefijo de trabajo
    # y que una sesion interrumpida retome donde quedo.
    assert [item.generation_id for item in build_grid(mini_config)] == [
        item.generation_id for item in build_grid(mini_config)
    ]


def test_dry_run_touches_nothing(mini_config: BenchConfig) -> None:
    stats = run(mini_config, dry_run=True)
    assert stats.selected == 4
    assert stats.generated == 0
    assert not mini_config.resolve(mini_config.model.manifest_path).exists()
    assert not (mini_config.root / "audio").exists()


def test_full_run_writes_one_row_per_attempt(mini_config: BenchConfig) -> None:
    stats = run(mini_config)
    assert (stats.generated, stats.failed, stats.skipped) == (4, 0, 0)
    assert stats.warmup == 1

    rows = manifest_rows(mini_config)
    assert len(rows) == 5  # 4 del grid + 1 de warm-up
    assert sum(1 for row in rows if row.warmup) == 1

    for row in rows:
        assert row.error is None
        assert row.wav_path is not None
        assert mini_config.resolve(row.wav_path).exists()
        assert row.sample_rate == 24000
        assert row.rtf is not None and row.rtf > 0
        assert row.ref_seconds_used == pytest.approx(4.0)  # 5 s de clip, max_ref_seconds 4


def test_a_voice_by_id_model_reports_no_reference_seconds(
    mini_config: BenchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El fallback que estima desde el clip declarado vale solo para los que clonan. Sin esta
    # excepcion, Sintetica declararia 6 s de referencia que nunca escucho (SPEC 6.5).
    cloning = dummy.DummyAdapter.synthesize

    def no_reference(self, *args, **kwargs):
        result = cloning(self, *args, **kwargs)
        result.ref_seconds_used = None
        return result

    monkeypatch.setattr(dummy.DummyAdapter, "selects_voice_by_id", True)
    monkeypatch.setattr(dummy.DummyAdapter, "synthesize", no_reference)
    run(mini_config, selection=GridFilter(limit=1), warmup=False)
    assert manifest_rows(mini_config)[0].ref_seconds_used is None


def test_a_cloning_model_still_gets_the_estimated_reference_seconds(
    mini_config: BenchConfig,
) -> None:
    run(mini_config, selection=GridFilter(limit=1), warmup=False)
    assert manifest_rows(mini_config)[0].ref_seconds_used == pytest.approx(4.0)


def test_generated_audio_is_a_readable_wav(mini_config: BenchConfig) -> None:
    run(mini_config, selection=GridFilter(limit=1), warmup=False)
    row = manifest_rows(mini_config)[0]
    assert row.wav_path is not None
    with wave.open(str(mini_config.resolve(row.wav_path)), "rb") as fh:
        assert fh.getframerate() == 24000
        assert fh.getnchannels() == 1
        # La duracion escrita en el manifest tiene que coincidir con el audio real.
        assert fh.getnframes() / fh.getframerate() == pytest.approx(row.duration_s, abs=0.01)


def test_resume_skips_finished_work(mini_config: BenchConfig) -> None:
    run(mini_config)
    second = run(mini_config)
    assert (second.generated, second.skipped, second.warmup) == (0, 4, 0)
    assert len(manifest_rows(mini_config)) == 5  # no se agrego nada


def test_deleted_wav_is_regenerated(mini_config: BenchConfig) -> None:
    run(mini_config, warmup=False)
    rows = manifest_rows(mini_config)
    victim = rows[0]
    assert victim.wav_path is not None
    mini_config.resolve(victim.wav_path).unlink()

    stats = run(mini_config, warmup=False)
    assert (stats.generated, stats.skipped) == (1, 3)

    # El manifest es append-only: queda una segunda fila con el mismo id y gana la ultima.
    after = manifest_rows(mini_config)
    assert len(after) == 5
    assert sum(1 for row in after if row.generation_id == victim.generation_id) == 2
    latest = index_by_generation_id(after)[victim.generation_id]
    assert latest.created_at >= victim.created_at
    assert mini_config.resolve(latest.wav_path or "").exists()


def test_soft_failure_is_recorded_and_the_run_continues(mini_config: BenchConfig) -> None:
    mini_config.model.params["fail_if_text_contains"] = ["Vos tenés"]
    stats = run(mini_config, warmup=False)
    assert (stats.generated, stats.failed) == (2, 2)

    failed = [row for row in manifest_rows(mini_config) if row.error is not None]
    assert len(failed) == 2
    for row in failed:
        assert row.phrase_id == "vos_001"
        assert row.wav_path is None
        assert "simulated failure" in (row.error or "")


def test_failed_cells_are_retried_only_when_asked(mini_config: BenchConfig) -> None:
    mini_config.model.params["fail_if_text_contains"] = ["Vos tenés"]
    run(mini_config, warmup=False)

    again = run(mini_config, warmup=False)
    assert (again.generated, again.failed, again.skipped) == (0, 0, 4)

    mini_config.model.params["fail_if_text_contains"] = []
    retried = run(mini_config, warmup=False, retry_errors=True)
    assert (retried.generated, retried.failed) == (2, 0)


def test_adapter_exception_does_not_abort_the_run(
    mini_config: BenchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(dummy.DummyAdapter, "synthesize", explode)
    stats = run(mini_config, warmup=False)
    assert (stats.generated, stats.failed) == (0, 4)
    rows = manifest_rows(mini_config)
    assert all(row.wav_path is None for row in rows)
    assert all("CUDA out of memory" in (row.error or "") for row in rows)


def test_partial_wav_is_removed_so_the_cell_stays_pending(
    mini_config: BenchConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Un wav a medio escribir haria que la proxima corrida crea que la celda ya esta hecha.
    original = dummy.DummyAdapter.synthesize

    def write_then_fail(self: dummy.DummyAdapter, text: str, ref: Path, seed: int, out: Path):
        original(self, text, ref, seed, out)
        raise RuntimeError("died after writing")

    monkeypatch.setattr(dummy.DummyAdapter, "synthesize", write_then_fail)
    run(mini_config, selection=GridFilter(limit=1), warmup=False)

    row = manifest_rows(mini_config)[0]
    assert row.wav_path is None
    assert list((mini_config.root / "audio" / "dummy" / "slr61_f_01").glob("*.wav")) == []


def test_warmup_indices_continue_across_sessions(mini_config: BenchConfig) -> None:
    run(mini_config, selection=GridFilter(limit=2))
    run(mini_config)
    warmups = [row.generation_id for row in manifest_rows(mini_config) if row.warmup]
    assert warmups == ["dummy__warmup0", "dummy__warmup1"]


def test_warmup_text_stays_out_of_the_corpus(mini_config: BenchConfig) -> None:
    run(mini_config, selection=GridFilter(limit=1))
    warmup = next(row for row in manifest_rows(mini_config) if row.warmup)
    # El wav de warm-up no pisa ninguna celda del grid.
    assert warmup.wav_path == "audio/dummy/_warmup/warmup0.wav"


def test_filters_subset_the_grid(mini_config: BenchConfig) -> None:
    grid = build_grid(mini_config)
    assert len(filter_grid(grid, GridFilter(phrase_ids={"vos_001"}))) == 2
    assert len(filter_grid(grid, GridFilter(categories={"yeismo_rehilado"}))) == 2
    assert len(filter_grid(grid, GridFilter(seeds={11}))) == 2
    assert len(filter_grid(grid, GridFilter(limit=3))) == 3
    assert len(filter_grid(grid, GridFilter(speaker_ids={"nobody"}))) == 0


def test_missing_reference_clip_index_is_skipped_not_fatal(mini_config: BenchConfig) -> None:
    mini_config.grid.ref_clip_indices = [0, 5]
    grid = build_grid(mini_config)
    assert {item.ref_clip_idx for item in grid} == {0}


def test_cli_dry_run(mini_repo: Path) -> None:
    code = main(
        [
            "--config",
            "configs/models/dummy.yaml",
            "--root",
            str(mini_repo),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (mini_repo / "data" / "manifests").exists()


def test_cli_model_name_mismatch_is_an_error(mini_repo: Path) -> None:
    # Protege contra el copy-paste de un config con el nombre de otro modelo.
    code = main(
        [
            "--config",
            "configs/models/dummy.yaml",
            "--root",
            str(mini_repo),
            "--model",
            "xtts_v2",
            "--dry-run",
        ]
    )
    assert code == 1


def test_cli_full_run_and_resume(mini_repo: Path) -> None:
    argv = ["--config", "configs/models/dummy.yaml", "--root", str(mini_repo), "--no-warmup"]
    assert main(argv) == 0
    rows = read_rows(mini_repo / "data" / "manifests" / "dummy.jsonl")
    assert len(rows) == 4
    assert main(argv) == 0
    assert len(read_rows(mini_repo / "data" / "manifests" / "dummy.jsonl")) == 4
