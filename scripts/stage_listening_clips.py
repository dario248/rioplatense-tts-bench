# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Copy the clips of a listening worklist into one folder, numbered in the worklist's order.

    uv run scripts/stage_listening_clips.py --verdict trailing_audio

The clips live one folder per model and per speaker. The number prefix is the point: any player
that sorts by name plays the round in the row order of the TSV, which is what makes a listening
cut in half still worth something. The destination lives under `audio/`, which is not committed.

It works on any TSV with `generation_id` and `wav` columns — `--listening`, `--dest` and
`--column` point it at another round, such as the transcription one in `data/corrections/`.

It imports no `rtb` and depends on nothing: it has to run in any environment.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_LISTENING = Path("data/listening")
DEFAULT_DEST = Path("audio/listening")
VERDICT_COLUMN = "veredicto"


def read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"{path} está vacío")
    return lines[0].split("\t"), [line.split("\t") for line in lines[1:] if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stage_listening_clips.py", description=__doc__)
    parser.add_argument("--verdict", default="trailing_audio", help="name of the TSV, without .tsv")
    parser.add_argument(
        "--column",
        default=VERDICT_COLUMN,
        help="the column that marks a row as done, for --pending-only",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--listening", type=Path, default=DEFAULT_LISTENING)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="only the rows with nothing written in --column yet",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    tsv = root / args.listening / f"{args.verdict}.tsv"
    if not tsv.exists():
        raise SystemExit(f"no existe {tsv}; generalo con `python -m rtb.metrics.listening`")

    header, rows = read_rows(tsv)
    try:
        id_at, wav_at = header.index("generation_id"), header.index("wav")
    except ValueError:
        raise SystemExit(f"{tsv} no tiene columnas generation_id/wav") from None
    verdict_at = header.index(args.column) if args.column in header else None

    dest = root / args.dest / args.verdict
    # Se rehace entero: si el TSV se regeneró, un sobrante de la tanda anterior se reproduce en
    # medio de la nueva secuencia y desordena justo lo que este script viene a ordenar.
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    width = len(str(len(rows)))
    copied = skipped = missing = 0
    for position, cells in enumerate(rows, 1):
        if len(cells) <= max(id_at, wav_at):
            continue
        if (
            args.pending_only
            and verdict_at is not None
            and len(cells) > verdict_at
            and cells[verdict_at].strip()
        ):
            skipped += 1
            continue
        source = root / cells[wav_at]
        if not source.exists():
            print(f"  falta el wav de {cells[id_at]}: {source}", file=sys.stderr)
            missing += 1
            continue
        # La numeracion es la POSICION EN EL TSV, no la del archivo copiado: con --pending-only
        # quedan huecos, y ese hueco es informacion — dice cuantas ya escuchaste.
        shutil.copy2(source, dest / f"{position:0{width}d}__{cells[id_at]}.wav")
        copied += 1

    print(f"{copied} clips en {dest}")
    if skipped:
        print(f"{skipped} salteados por tener veredicto")
    if missing:
        print(f"{missing} sin wav en disco")
    print("El numero del nombre es la fila del TSV: ordenando por nombre escuchás en ese orden.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
