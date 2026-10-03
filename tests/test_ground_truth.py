"""Ground-truth rows: the real audio that gives every other number a scale (SPEC 5.5)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from rtb.config import Speaker, load_speakers
from rtb.manifest import read_rows
from rtb.metrics import ground_truth
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow, make_ground_truth_id

from .conftest import write_wav


def speaker_with_clips(root: Path) -> Speaker:
    write_wav(root / "data" / "refs" / "ctrl_00.wav", seconds=3.0, sample_rate=22050)
    write_wav(root / "data" / "refs" / "ctrl_01.wav", seconds=2.0, sample_rate=22050)
    write_wav(root / "data" / "refs" / "calib_00.wav", seconds=0.9, sample_rate=22050)
    return Speaker.model_validate(
        {
            "id": "slr61_f_01",
            "gender": "F",
            "ref_clips": [{"path": "data/refs/ref_00.wav"}],
            "control_clips": [
                {"path": "data/refs/ctrl_00.wav", "duration_s": 3.0, "transcript": "una dos tres"},
                {"path": "data/refs/ctrl_01.wav", "duration_s": 2.0, "transcript": "cuatro cinco"},
            ],
            "calibration_clips": [
                {"path": "data/refs/calib_00.wav", "duration_s": 0.9, "transcript": "seis"}
            ],
        }
    )


def test_ids_are_prefixed_gt_because_the_model_name_is_not_a_token() -> None:
    # "__ground_truth__" arranca con el separador y lo contiene: no puede ser un token de id.
    assert make_ground_truth_id("slr61_f_01", "ctrl_00") == "gt__slr61_f_01__ctrl_00"


def test_rows_cover_controls_and_calibration_but_not_references(tmp_path: Path) -> None:
    # Los ref_clips son el prompt del modelo: transcribirlos mediria el prompt, no una referencia.
    rows = ground_truth.build_rows([speaker_with_clips(tmp_path)], tmp_path)
    assert [row.phrase_id for row in rows] == ["ctrl_00", "ctrl_01", "calib_00"]


def test_rows_carry_the_transcript_in_the_row(tmp_path: Path) -> None:
    # No salen de phrases.yaml, asi que no hay phrase_id que consultar.
    row = ground_truth.build_rows([speaker_with_clips(tmp_path)], tmp_path)[0]
    assert row.reference_text == "una dos tres"
    assert row.model == GROUND_TRUTH_MODEL


def test_rows_declare_no_stack_and_no_reference_clip(tmp_path: Path) -> None:
    # Ningun modelo corrio y nada se condiciono: poner ceros ahi seria dato inventado.
    row = ground_truth.build_rows([speaker_with_clips(tmp_path)], tmp_path)[0]
    assert (row.model_revision, row.lib_version, row.device, row.gpu) == (None,) * 4
    assert (row.ref_clip_idx, row.seed, row.ttfb_ms, row.rtf) == (None,) * 4


def test_duration_is_read_from_the_file(tmp_path: Path) -> None:
    row = ground_truth.build_rows([speaker_with_clips(tmp_path)], tmp_path)[0]
    assert row.duration_s == pytest.approx(3.0, abs=0.01)
    assert row.sample_rate == 22050


def test_a_yaml_duration_that_disagrees_with_the_file_is_fatal(tmp_path: Path) -> None:
    # Si el YAML y el disco no coinciden, la tasa de habla que sale del YAML tampoco vale, y eso
    # mueve en silencio toda la deteccion de duracion.
    speaker = speaker_with_clips(tmp_path)
    speaker.control_clips[0].duration_s = 9.0
    with pytest.raises(ground_truth.GroundTruthError, match="not the ones the config describes"):
        ground_truth.build_rows([speaker], tmp_path)


def test_a_missing_clip_says_what_to_run(tmp_path: Path) -> None:
    speaker = speaker_with_clips(tmp_path)
    (tmp_path / "data" / "refs" / "ctrl_00.wav").unlink()
    with pytest.raises(ground_truth.GroundTruthError, match="build_reference_clips"):
        ground_truth.build_rows([speaker], tmp_path)


def test_a_clip_without_transcript_is_rejected(tmp_path: Path) -> None:
    speaker = speaker_with_clips(tmp_path)
    speaker.control_clips[0].transcript = None
    with pytest.raises(ground_truth.GroundTruthError, match="cannot be a reference point"):
        ground_truth.build_rows([speaker], tmp_path)


def test_an_empty_wav_is_rejected(tmp_path: Path) -> None:
    speaker = speaker_with_clips(tmp_path)
    empty = tmp_path / "data" / "refs" / "ctrl_00.wav"
    with wave.open(str(empty), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(22050)
    with pytest.raises(ground_truth.GroundTruthError, match="no audio"):
        ground_truth.build_rows([speaker], tmp_path)


def test_cli_writes_a_manifest(tmp_path: Path) -> None:
    speaker = speaker_with_clips(tmp_path)
    speakers_yaml = tmp_path / "speakers.yaml"
    speakers_yaml.write_text(
        "\n".join(
            [
                "- id: slr61_f_01",
                "  gender: F",
                "  ref_clips:",
                "    - path: data/refs/ref_00.wav",
                "  control_clips:",
                '    - {path: data/refs/ctrl_00.wav, duration_s: 3.0, transcript: "una dos tres"}',
                '    - {path: data/refs/ctrl_01.wav, duration_s: 2.0, transcript: "cuatro cinco"}',
                "  calibration_clips:",
                '    - {path: data/refs/calib_00.wav, duration_s: 0.9, transcript: "seis"}',
            ]
        ),
        encoding="utf-8",
    )
    assert speaker.id == "slr61_f_01"

    out = tmp_path / "out" / "ground_truth.jsonl"
    code = ground_truth.main(
        ["--root", str(tmp_path), "--speakers", str(speakers_yaml), "--out", str(out)]
    )
    assert code == 0
    rows: list[GenerationRow] = read_rows(out)
    assert len(rows) == 3
    assert all(row.is_ground_truth for row in rows)


def test_cli_reports_a_missing_config(tmp_path: Path) -> None:
    argv = ["--root", str(tmp_path), "--speakers", str(tmp_path / "no.yaml")]
    assert ground_truth.main(argv) == 1


# ── el manifest commiteado ───────────────────────────────────────────────────────────────────


def test_committed_ground_truth_matches_the_speakers(repo_root: Path) -> None:
    path = repo_root / "data" / "manifests" / "ground_truth.jsonl"
    if not path.exists():
        pytest.skip("ground truth not built in this checkout")
    rows: list[GenerationRow] = read_rows(path)
    speakers = load_speakers(repo_root / "configs" / "speakers.yaml")

    expected = {
        make_ground_truth_id(speaker.id, clip.path.stem)
        for speaker in speakers
        for clip in [*speaker.control_clips, *speaker.calibration_clips]
    }
    assert {row.generation_id for row in rows} == expected
    for row in rows:
        assert row.reference_text, row.generation_id
        assert row.duration_s and row.duration_s > 0