"""The listening worklist, and above all: that regenerating it never loses an annotation.

Everything else in this project is recomputable. A human listening to 268 clips is not, so the
carry-over of existing verdicts gets more tests than the rest of the module put together.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rtb.metrics import failures, listening
from rtb.metrics.listening import VERDICT_COLUMN, VERDICTS, TailRow, Worklist, read_verdicts


def row(generation_id: str = "m__s__p__s1__ref0", **overrides: object) -> TailRow:
    base: dict[str, object] = {
        "generation_id": generation_id,
        "model": "m",
        "speaker": "slr61_f_01",
        "phrase_id": "cor_001",
        "category": "enunciados_cortos",
        "wav": "audio/m/slr61_f_01/cor_001_s1_ref0.wav",
        "tail_s": 1.5,
        "level_db": -12.0,
        "speech_s": 2.0,
        "padding_s": 0.4,
        "total_s": 2.4,
        "wer": 0.0,
        "phrase": "Dale.",
        "transcript": "Dale.",
    }
    base.update(overrides)
    return TailRow(**base)  # type: ignore[arg-type]


# ── los veredictos ya escritos se preservan ──────────────────────────────────────────────────────


def test_verdicts_are_read_back_by_generation_id(tmp_path: Path) -> None:
    path = tmp_path / "w.tsv"
    path.write_text(
        Worklist(rows=[row("a", verdict="ruido"), row("b"), row("c", verdict="ambas")]).to_tsv(),
        encoding="utf-8",
    )
    assert read_verdicts(path) == {"a": "ruido", "c": "ambas"}


def test_a_missing_file_has_no_verdicts(tmp_path: Path) -> None:
    assert read_verdicts(tmp_path / "nope.tsv") == {}


def test_a_reordered_header_still_finds_the_verdicts(tmp_path: Path) -> None:
    # El archivo lo edita una persona en una planilla: una columna movida no puede tirar el trabajo.
    path = tmp_path / "w.tsv"
    path.write_text(
        f"{VERDICT_COLUMN}\tnota\tgeneration_id\nruido\tse escucha un click\ta\n",
        encoding="utf-8",
    )
    assert read_verdicts(path) == {"a": "ruido"}


def test_an_extra_column_does_not_break_the_read(tmp_path: Path) -> None:
    path = tmp_path / "w.tsv"
    header = "\t".join([*listening.COLUMNS, "comentario"])
    cells = "\t".join([*row("a", verdict="habla").cells(), "un comentario mio"])
    path.write_text(f"{header}\n{cells}\n", encoding="utf-8")
    assert read_verdicts(path) == {"a": "habla"}


def test_a_file_without_the_columns_is_ignored_not_trusted(tmp_path: Path) -> None:
    path = tmp_path / "w.tsv"
    path.write_text("otra cosa\nvalores\n", encoding="utf-8")
    assert read_verdicts(path) == {}


def test_whitespace_only_verdicts_do_not_count_as_written(tmp_path: Path) -> None:
    path = tmp_path / "w.tsv"
    path.write_text(f"generation_id\t{VERDICT_COLUMN}\na\t   \nb\truido\n", encoding="utf-8")
    assert read_verdicts(path) == {"b": "ruido"}


def test_a_verdict_survives_a_full_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "w.tsv"
    path.write_text(Worklist(rows=[row("a", verdict="ambas")]).to_tsv(), encoding="utf-8")
    carried = read_verdicts(path)
    rebuilt = Worklist(rows=[row("a", verdict=carried.get("a", "")), row("b")])
    path.write_text(rebuilt.to_tsv(), encoding="utf-8")
    assert read_verdicts(path) == {"a": "ambas"}


# ── el TSV es legible por una planilla ───────────────────────────────────────────────────────────


def test_the_header_matches_the_cells() -> None:
    assert len(row().cells()) == len(listening.COLUMNS)


def test_newlines_and_tabs_never_reach_a_cell() -> None:
    # Un salto de linea en una celda corre en silencio todas las columnas siguientes.
    dirty = row(phrase="una\nfrase\tcon\tbasura", transcript="otra\nlinea")
    for cell in dirty.cells():
        assert "\n" not in cell and "\t" not in cell


def test_an_absent_wer_is_an_empty_cell_not_a_zero() -> None:
    # Un 0.000 inventado se lee como "transcribio perfecto", que es lo contrario de "no se sabe".
    cells = row(wer=None).cells()
    assert cells[listening.COLUMNS.index("wer")] == ""


def test_the_verdict_column_is_last_and_empty_by_default() -> None:
    assert listening.COLUMNS[-1] == VERDICT_COLUMN
    assert row().cells()[-1] == ""


# ── el vocabulario ───────────────────────────────────────────────────────────────────────────────


def test_the_verdict_vocabulary_is_closed_and_documented() -> None:
    # Sin vocabulario cerrado, la mitad de las filas dice "ruido" y la otra "noise, poco".
    assert set(VERDICTS["trailing_audio"]) == {"habla", "ruido", "ambas", "nada", "corte"}
    assert set(VERDICTS["boundary_disagreement"]) == {
        "dijo_bien",
        "dijo_mal",
        "dialectal",
        "nada",
    }
    assert set(VERDICTS["trailing_disagreement"]) == {
        "dijo_bien",
        "dijo_mal",
        "no_lo_dijo",
        "nada",
    }
    for vocabulary in VERDICTS.values():
        for meaning in vocabulary.values():
            assert meaning and meaning[0].islower()


def test_every_vocabulary_names_a_field_the_taxonomy_writes() -> None:
    # Una lista de escucha se pide por el nombre del campo de failures.jsonl: un vocabulario
    # colgado de un nombre que la taxonomia no escribe no lo alcanza ninguna fila.
    assert set(VERDICTS) <= set(vars(failures.Verdict()))


# ── el nivel de la cola ──────────────────────────────────────────────────────────────────────────


def test_the_level_is_relative_to_the_clip_peak() -> None:
    assert listening.db_relative(1.0, 1.0) == pytest.approx(0.0)
    assert listening.db_relative(0.1, 1.0) == pytest.approx(-20.0)
    assert listening.db_relative(0.5, 1.0) == pytest.approx(-6.02, abs=0.01)


def test_silence_reports_a_floor_not_negative_infinity() -> None:
    # -inf no se puede escribir en un TSV ni ordenar.
    assert listening.db_relative(0.0, 1.0) == -120.0
    assert listening.db_relative(0.5, 0.0) == -120.0


# ── la lista real ────────────────────────────────────────────────────────────────────────────────


def test_the_committed_worklist_is_sorted_worst_first(repo_root: Path) -> None:
    path = repo_root / "data" / "listening" / "trailing_audio.tsv"
    if not path.exists():
        pytest.skip("worklist not built in this checkout")
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    at = header.index("cola_s")
    tails = [float(line.split("\t")[at]) for line in lines[1:] if line.split("\t")[at]]
    # Si la escucha se corta a la mitad, lo escuchado tiene que ser lo que mas informa.
    assert tails == sorted(tails, reverse=True)


def test_every_committed_verdict_uses_the_vocabulary(repo_root: Path) -> None:
    # Las etiquetas son el unico dato no recomputable del repo: una escrita fuera del vocabulario
    # no se puede contar y no hay de donde recuperarla.
    listened = False
    for name, vocabulary in VERDICTS.items():
        path = repo_root / "data" / "listening" / f"{name}.tsv"
        if not path.exists():
            continue
        for generation_id, verdict in read_verdicts(path).items():
            listened = True
            assert verdict in vocabulary, (
                f"{name}.tsv, {generation_id}: veredicto {verdict!r} fuera del vocabulario"
            )
    if not listened:
        pytest.skip("no worklist with verdicts in this checkout")
