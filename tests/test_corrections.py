"""The human-transcript worklist: that it never loses a transcript, and never leaks the answer.

Like the listening lists, a person's work here is the one thing that cannot be recomputed. Unlike
them, this one is also a trap: the expected verbalization must stay out of sight, because a
transcriber who can see it writes what the model should have said.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rtb.metrics import corrections
from rtb.metrics.corrections import DONE_COLUMN, SAID_COLUMN, Clip, Worklist, read_done

from .conftest import write_wav

PHRASES = {"nor_001": ("El total es $15.750.", "normalizacion"), "vos_001": ("Vos sabés.", "voseo")}


def clip(audio_sha: str = "abc123", **overrides: object) -> Clip:
    base: dict[str, object] = {
        "generation_id": "m__s__nor_001__s1__ref0",
        "audio_sha": audio_sha,
        "model": "m",
        "speaker": "slr61_f_01",
        "phrase_id": "nor_001",
        "phrase": "El total es $15.750.",
        "n": 3,
        "wav": "audio/m/slr61_f_01/nor_001_s1_ref0.wav",
        "asr": "El total es 15.750.",
    }
    base.update(overrides)
    return Clip(**base)  # type: ignore[arg-type]


def a_repo(root: Path, audios: dict[str, str], category: str = "normalizacion") -> Path:
    """A manifest plus its wavs, where `audios` maps a generation_id to the clip it shares."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data" / "metrics").mkdir(parents=True, exist_ok=True)
    masters: dict[str, Path] = {}
    rows, metrics = [], []
    for generation_id, master in audios.items():
        model, speaker, phrase_id, seed, _ = generation_id.split("__")
        wav = root / "audio" / f"{generation_id}.wav"
        if master in masters:
            shutil.copyfile(masters[master], wav)
        else:
            masters[master] = write_wav(wav, seconds=0.2)
        rows.append(
            {
                "generation_id": generation_id,
                "model": model,
                "speaker_id": speaker,
                "phrase_id": phrase_id,
                "seed": int(seed[1:]),
                "ref_clip_idx": 0,
                "wav_path": str(wav.relative_to(root)),
                "duration_s": 0.2,
                "sample_rate": 16000,
                "model_revision": "a",
                "lib_version": "x==1",
                "device": "cpu",
            }
        )
        metrics.append(
            {
                "generation_id": generation_id,
                "metric": "asr",
                "metric_version": "1",
                "values": {"text": f"lo que oyo el ASR de {generation_id}"},
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    (root / "data" / "metrics" / "asr.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in metrics), encoding="utf-8"
    )
    return root


def build(root: Path, **overrides: object) -> Worklist:
    kwargs: dict[str, object] = {
        "manifests": [Path("manifest.jsonl")],
        "metrics_dir": root / "data" / "metrics",
        "category": "normalizacion",
        "models": [],
        "phrases_by_id": PHRASES,
        "existing": {},
        "root": root,
    }
    kwargs.update(overrides)
    return corrections.build(**kwargs)  # type: ignore[arg-type]


# ── la respuesta no se filtra ────────────────────────────────────────────────────────────────────


def test_the_expected_verbalization_is_never_a_column() -> None:
    # La condicion de diseño de toda la tanda: a la vista, se transcribe lo que deberia haber
    # dicho en vez de lo que dijo, y el conjunto queda arruinado sin que se note despues.
    assert "verbalizacion" not in corrections.COLUMNS
    assert "alternativas" not in corrections.COLUMNS


def test_the_written_phrase_is_a_column_because_it_carries_no_answer() -> None:
    # El texto escrito es lo que el modelo recibio, con los digitos sin expandir: ubica el clip
    # sin decir como habia que leerlo.
    assert "frase" in corrections.COLUMNS


# ── lo transcripto se preserva ───────────────────────────────────────────────────────────────────


def test_transcripts_are_read_back_by_audio_sha(tmp_path: Path) -> None:
    path = tmp_path / "c.tsv"
    path.write_text(
        Worklist(
            rows=[
                clip("aaa", said="el total es quince mil", done="x"),
                clip("bbb"),
            ]
        ).to_tsv(),
        encoding="utf-8",
    )
    assert read_done(path) == {"aaa": ("el total es quince mil", "x")}


def test_an_unmarked_row_does_not_count_as_transcribed(tmp_path: Path) -> None:
    # `dicho` viene pre-cargado con el borrador del ASR: sin `revisado`, nadie lo escucho.
    path = tmp_path / "c.tsv"
    path.write_text(
        Worklist(rows=[clip("aaa", said="el borrador del ASR")]).to_tsv(), encoding="utf-8"
    )
    assert read_done(path) == {}


def test_a_transcript_survives_a_rebuild(tmp_path: Path) -> None:
    root = a_repo(tmp_path / "r", {"m__slr61_f_01__nor_001__s1__ref0": "a"})
    out = root / "c.tsv"
    out.write_text(build(root).to_tsv(), encoding="utf-8")
    sha = build(root).rows[0].audio_sha
    out.write_text(
        Worklist(rows=[clip(sha, said="lo que dije yo", done="x")]).to_tsv(), encoding="utf-8"
    )
    rebuilt = build(root, existing=read_done(out))
    assert rebuilt.rows[0].said == "lo que dije yo"
    assert rebuilt.carried == 1


# ── una fila por audio distinto ──────────────────────────────────────────────────────────────────


def test_seeds_with_identical_audio_share_one_row(tmp_path: Path) -> None:
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s1__ref0": "a",
            "m__slr61_f_01__nor_001__s2__ref0": "a",
            "m__slr61_f_01__nor_001__s3__ref0": "a",
        },
    )
    worklist = build(root)
    assert len(worklist.rows) == 1
    assert worklist.rows[0].n == 3


def test_seeds_with_different_audio_are_separate_rows(tmp_path: Path) -> None:
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s1__ref0": "a",
            "m__slr61_f_01__nor_001__s2__ref0": "b",
        },
    )
    assert len(build(root).rows) == 2


def test_another_category_stays_out(tmp_path: Path) -> None:
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s1__ref0": "a",
            "m__slr61_f_01__vos_001__s1__ref0": "b",
        },
    )
    assert [row.phrase_id for row in build(root).rows] == ["nor_001"]


# ── el guard de los digitos ──────────────────────────────────────────────────────────────────────


def test_a_reviewed_row_still_holding_digits_is_flagged() -> None:
    # Marcar `revisado` sin reescribir el numero publica la opinion de Whisper como referencia.
    assert clip(said="El total es 15.750 pesos.", done="x").unrewritten_digits()


def test_a_reviewed_row_written_out_in_words_is_not_flagged() -> None:
    assert not clip(
        said="el total es quince mil setecientos cincuenta pesos.", done="x"
    ).unrewritten_digits()


def test_an_unreviewed_row_is_never_flagged() -> None:
    # El borrador del ASR trae digitos por definicion: marcarlo seria ruido en cada fila.
    assert not clip(said="El total es 15.750 pesos.").unrewritten_digits()


# ── el TSV es legible por una planilla ───────────────────────────────────────────────────────────


def test_the_header_matches_the_cells() -> None:
    assert len(clip().cells()) == len(corrections.COLUMNS)


def test_newlines_and_tabs_never_reach_a_cell() -> None:
    dirty = clip(phrase="una\nfrase\tcon\tbasura", asr="otra\nlinea", said="y\totra")
    for cell in dirty.cells():
        assert "\n" not in cell and "\t" not in cell


# ── el archivo real ──────────────────────────────────────────────────────────────────────────────


def test_every_committed_transcript_wrote_the_numbers_out(repo_root: Path) -> None:
    path = repo_root / "data" / "corrections" / "normalizacion.tsv"
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    said_at, done_at = header.index(SAID_COLUMN), header.index(DONE_COLUMN)
    for line in lines[1:]:
        cells = line.split("\t")
        if len(cells) > max(said_at, done_at) and cells[done_at].strip():
            assert not any(c.isdigit() for c in cells[said_at]), (
                f"{cells[0]}: marcada como revisada pero el numero sigue en digitos"
            )


# ── la sonda: un audio por celda ─────────────────────────────────────────────────────────────────


def test_one_seed_keeps_one_audio_per_cell(tmp_path: Path) -> None:
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s1__ref0": "a",
            "m__slr61_f_01__nor_001__s2__ref0": "b",
            "m__slr61_f_01__nor_001__s3__ref0": "c",
        },
    )
    assert len(build(root, one_seed=["m"]).rows) == 1


def test_one_seed_does_not_shrink_a_cell_whose_seeds_share_audio(tmp_path: Path) -> None:
    # El descarte va despues de agrupar por hash: con tres semillas identicas la celda ya es una
    # fila sola, y `n` tiene que seguir diciendo 3 o se pierden dos filas del grid.
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s1__ref0": "a",
            "m__slr61_f_01__nor_001__s2__ref0": "a",
            "m__slr61_f_01__nor_001__s3__ref0": "a",
        },
    )
    rows = build(root, one_seed=["m"]).rows
    assert len(rows) == 1
    assert rows[0].n == 3


def test_one_seed_keeps_every_cell(tmp_path: Path) -> None:
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s1__ref0": "a",
            "m__slr61_f_01__nor_001__s2__ref0": "b",
            "m__slr61_f_02__nor_001__s1__ref0": "c",
        },
    )
    rows = build(root, one_seed=["m"]).rows
    assert {row.speaker for row in rows} == {"slr61_f_01", "slr61_f_02"}


def test_one_seed_picks_the_same_audio_every_time(tmp_path: Path) -> None:
    root = a_repo(
        tmp_path / "r",
        {
            "m__slr61_f_01__nor_001__s2__ref0": "b",
            "m__slr61_f_01__nor_001__s1__ref0": "a",
        },
    )
    assert build(root, one_seed=["m"]).rows[0].generation_id.endswith("s1__ref0")


# ── no se pisa trabajo hecho ─────────────────────────────────────────────────────────────────────


def test_a_rebuild_that_would_drop_a_transcript_raises(tmp_path: Path) -> None:
    # Reconstruir con menos `--model` de los que tiene el archivo borraria transcripciones, y son
    # lo unico del repo que no se puede recomputar.
    root = a_repo(tmp_path / "r", {"m__slr61_f_01__nor_001__s1__ref0": "a"})
    with pytest.raises(corrections.Dropped):
        build(root, existing={"una_que_no_esta": ("lo que dije", "x")})


def test_an_untranscribed_row_may_disappear_without_complaint(tmp_path: Path) -> None:
    root = a_repo(tmp_path / "r", {"m__slr61_f_01__nor_001__s1__ref0": "a"})
    assert len(build(root, models=["otro_modelo"]).rows) == 0
