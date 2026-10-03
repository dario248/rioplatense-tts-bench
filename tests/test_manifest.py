"""Manifest durability. The manifest IS the result; losing or misreading it loses the run."""

from __future__ import annotations

from pathlib import Path

import pytest

from rtb.manifest import (
    JsonlWriter,
    ManifestError,
    append_row,
    index_by_generation_id,
    read_generation_ids,
    read_manifest_dir,
    read_rows,
)
from rtb.schema import GenerationRow

from .test_schema import ok_row


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    append_row(path, ok_row())
    append_row(path, ok_row(generation_id="dummy__slr61_f_01__vos_001__s1234__ref0"))
    rows = read_rows(path)
    assert [row.generation_id for row in rows] == [
        "dummy__slr61_f_01__yei_003__s1234__ref0",
        "dummy__slr61_f_01__vos_001__s1234__ref0",
    ]


def test_rows_are_flushed_as_they_are_written(tmp_path: Path) -> None:
    # Una sesion interrumpida tiene que conservar todo lo que ya se sintetizo.
    path = tmp_path / "m.jsonl"
    with JsonlWriter(path) as writer:
        writer.write(ok_row())
        assert read_rows(path)  # legible antes de cerrar el writer


def test_strict_read_points_at_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    append_row(path, ok_row())
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"generation_id": "broken"}\n')
    with pytest.raises(ManifestError, match=r"m\.jsonl:2"):
        read_rows(path)


def test_lenient_read_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "m.jsonl"
    append_row(path, ok_row())
    with path.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
    assert len(read_rows(path, strict=False)) == 1


def test_torn_tail_is_repaired_not_concatenated(tmp_path: Path) -> None:
    # Si el proceso murio a mitad de una escritura, la ultima linea queda cortada y sin newline.
    # Sin reparar, la fila siguiente se pegaria a la basura y se perderian las dos.
    path = tmp_path / "m.jsonl"
    append_row(path, ok_row())
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"generation_id": "half-writ')

    with JsonlWriter(path) as writer:
        writer.write(ok_row(generation_id="dummy__slr61_f_01__vos_001__s1234__ref0"))

    ids = read_generation_ids(path)
    assert ids == {
        "dummy__slr61_f_01__yei_003__s1234__ref0",
        "dummy__slr61_f_01__vos_001__s1234__ref0",
    }


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    # Ninguna etapa asume que otra ya corrio.
    assert read_rows(tmp_path / "nope.jsonl") == []
    assert read_generation_ids(tmp_path / "nope.jsonl") == set()


def test_last_row_wins(tmp_path: Path) -> None:
    # El manifest es append-only: regenerar una celda agrega una fila nueva con el mismo id.
    path = tmp_path / "m.jsonl"
    append_row(path, ok_row(duration_s=1.0))
    append_row(path, ok_row(duration_s=2.0))
    latest = index_by_generation_id(read_rows(path))
    assert len(latest) == 1
    assert next(iter(latest.values())).duration_s == 2.0


def test_read_manifest_dir_concatenates_in_filename_order(tmp_path: Path) -> None:
    append_row(tmp_path / "b_model.jsonl", ok_row(model="b_model"))
    append_row(tmp_path / "a_model.jsonl", ok_row(model="a_model"))
    rows: list[GenerationRow] = read_manifest_dir(tmp_path)
    assert [row.model for row in rows] == ["a_model", "b_model"]
