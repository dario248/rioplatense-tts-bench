"""Append-only JSONL manifests: the boundary between stages.

Installed in every environment, so nothing beyond the standard library and pydantic (SPEC 3).
Rows are flushed as they are written, and a torn last line — the process died mid-write — is
skipped on read and repaired by the next writer instead of merging two records into one.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import TextIO, TypeVar

from pydantic import BaseModel, ValidationError

from rtb.schema import GenerationRow

logger = logging.getLogger(__name__)

RowT = TypeVar("RowT", bound=BaseModel)


class ManifestError(Exception):
    """Raised when a JSONL file cannot be parsed in strict mode."""


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        try:
            fh.seek(-1, os.SEEK_END)
        except OSError:  # archivo vacio
            return True
        return fh.read(1) == b"\n"


def repair_trailing_newline(path: Path) -> bool:
    """Append the newline a crashed writer may have left out. Returns True if it repaired."""
    if not path.exists() or path.stat().st_size == 0 or _ends_with_newline(path):
        return False
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.flush()
    logger.warning("repaired missing trailing newline in %s", path)
    return True


class JsonlWriter:
    """Append rows to a JSONL file, flushing after each one.

    The handle stays open across a generation loop, but no row waits in a buffer for the next
    synthesis.
    """

    def __init__(self, path: Path | str, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._fh: TextIO | None = None

    def __enter__(self) -> JsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        repair_trailing_newline(self.path)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def write(self, row: BaseModel) -> None:
        if self._fh is None:
            raise ManifestError("JsonlWriter is not open; use it as a context manager")
        self._fh.write(row.model_dump_json() + "\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())


def append_row(path: Path | str, row: BaseModel) -> None:
    """One-shot append. Prefer `JsonlWriter` inside a loop."""
    with JsonlWriter(path) as writer:
        writer.write(row)


def append_rows(path: Path | str, rows: Iterable[BaseModel]) -> int:
    count = 0
    with JsonlWriter(path) as writer:
        for row in rows:
            writer.write(row)
            count += 1
    return count


def iter_rows(
    path: Path | str,
    model: type[RowT] = GenerationRow,  # type: ignore[assignment]
    *,
    strict: bool = True,
) -> Iterator[RowT]:
    """Parse a JSONL file row by row.

    `strict=True` raises on a malformed line, naming file and line, which is what CI validates
    committed data with. `strict=False` logs and skips, so a torn tail cannot block a resumed run.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except (ValidationError, ValueError) as err:
                if strict:
                    raise ManifestError(f"{path}:{lineno}: {err}") from err
                logger.warning("skipping unparseable line %s:%s (%s)", path, lineno, err)


def read_rows(
    path: Path | str,
    model: type[RowT] = GenerationRow,  # type: ignore[assignment]
    *,
    strict: bool = True,
) -> list[RowT]:
    return list(iter_rows(path, model, strict=strict))


def read_manifest_dir(
    directory: Path | str,
    model: type[RowT] = GenerationRow,  # type: ignore[assignment]
    *,
    strict: bool = True,
    pattern: str = "*.jsonl",
) -> list[RowT]:
    """Read every JSONL file in a directory, in sorted filename order."""
    rows: list[RowT] = []
    for path in sorted(Path(directory).glob(pattern)):
        rows.extend(iter_rows(path, model, strict=strict))
    return rows


def read_generation_ids(path: Path | str) -> set[str]:
    """Ids already present in a manifest, for the runner's idempotency check.

    Deliberately lenient: a resumed run must not be blocked by one bad line.
    """
    return {row.generation_id for row in iter_rows(path, GenerationRow, strict=False)}


def index_by_generation_id(rows: Iterable[GenerationRow]) -> dict[str, GenerationRow]:
    """Last row wins: the manifest is append-only and a regenerated synthesis appends a second
    row with the same id."""
    latest: dict[str, GenerationRow] = {}
    for row in rows:
        latest[row.generation_id] = row
    return latest


def dedupe_latest(rows: Iterable[GenerationRow]) -> list[GenerationRow]:
    return list(index_by_generation_id(rows).values())
