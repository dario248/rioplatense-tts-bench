# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Build the yes/no worklist of the normalizer ablation: did the model pronounce the number?

    .venv/bin/python scripts/build_verbalization_sheet.py

One row per distinct audio of the treated arm, with the numeric spans spelled out and an empty
`dijo_bien`. Rebuilding preserves the answers already written.

The ASR column is deliberately absent: the ASR writes conventional orthography whatever it hears,
so its text would answer the question before the ear does. The expected spoken form is shown
instead, because a yes/no has no meaning without the target.

Audio byte-identical to the base arm comes pre-answered from the transcript already made by ear.

Depends on pyyaml alone and does not import `rtb`: it has to run in a bare environment.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

COLUMNS = (
    "generation_id",
    "audio_sha",
    "modelo",
    "hablante",
    "phrase_id",
    "texto",
    "tramos",
    "wav",
    "dijo_bien",
    "nota",
)
# Agregar o mover una columna no puede cambiar por cual se ordena.
AT = {name: i for i, name in enumerate(COLUMNS)}

ANSWER_COLUMN = "dijo_bien"
NOTE_COLUMN = "nota"
CARRIED = "byte-identico al brazo base; veredicto de la transcripcion a oido"

# Separador de formas aceptadas de un mismo tramo, y entre tramos distintos.
OR = " | "
AND = "  ///  "


def norm(text: str) -> list[str]:
    """Lowercase, unaccented, symbol-free words — the form every comparison here runs on."""
    folded = unicodedata.normalize("NFD", text.lower())
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return re.sub(r"[^\w\s]", " ", folded).split()


def load_yaml_phrases(path: Path) -> dict[str, dict[str, object]]:
    """`configs/phrases*.yaml` reduced to what a yes/no sheet needs, by id.

    Only the phrases that declare a verbalization: the rest of the grid has no number to judge.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        entry["id"]: {
            "text": entry["text"],
            "expected_verbalization": entry["expected_verbalization"],
            "alternatives": entry.get("accepted_alternatives") or [],
        }
        for entry in raw
        if entry.get("expected_verbalization")
    }


def spans(written: str, spoken: str, reference: list[str]) -> list[tuple[int, int]]:
    """Offsets in `reference` of what changed between the two arms: the numeric spans.

    The diff runs on RAW tokens: normalized, an e-mail has already lost its symbols and the
    span breaks apart.
    """
    raw = spoken.split()
    at, n = [], 0
    for token in raw:
        at.append(n)
        n += len(norm(token))
    at.append(n)
    if n != len(reference):
        raise SystemExit(f"la verbalizacion no cierra: {n} != {len(reference)}")
    matcher = difflib.SequenceMatcher(a=written.split(), b=raw, autojunk=False)
    return [(at[op[3]], at[op[4]]) for op in matcher.get_opcodes() if op[0] != "equal"]


def project(reference: list[str], alternative: list[str], lo: int, hi: int) -> tuple[str, ...]:
    """The same span, read off an accepted alternative."""
    start = end = None
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=reference, b=alternative, autojunk=False
    ).get_opcodes():
        if start is None and i1 <= lo < max(i2, i1 + 1):
            start = j1 + (lo - i1) if tag == "equal" else j1
        if i1 < hi <= i2:
            end = j1 + (hi - i1) if tag == "equal" else j2
    if start is None or end is None:
        # El corte saldria `[None:None]`, que es la lista entera: la alternativa completa pasaria
        # por tramo y la planilla ofreceria una oracion donde va un numero.
        raise SystemExit(f"no ubico el tramo [{lo}:{hi}] en: {' '.join(alternative)}")
    return tuple(alternative[start:end])


def accepted(base: dict[str, object], treated: dict[str, object]) -> list[list[tuple[str, ...]]]:
    """Every accepted spoken form of every numeric span of one phrase."""
    reference = norm(str(base["expected_verbalization"]))
    out = []
    for lo, hi in spans(str(base["text"]), str(treated["text"]), reference):
        forms = {tuple(reference[lo:hi])}
        alts = base["alternatives"]
        assert isinstance(alts, list)
        for alt in alts:
            forms.add(project(reference, norm(str(alt)), lo, hi))
        out.append(sorted(forms))
    return out


def said_ok(forms: list[list[tuple[str, ...]]], said: str) -> bool:
    """Every span present, contiguous, in one of its accepted forms."""
    words = norm(said)
    return all(
        any(
            tuple(words[i : i + len(form)]) == form
            for form in group
            for i in range(len(words) - len(form) + 1)
        )
        for group in forms
    )


def read_answers(path: Path) -> dict[str, tuple[str, str]]:
    """Answers already written, by `audio_sha`."""
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    return {
        row["audio_sha"]: (row[ANSWER_COLUMN], row.get(NOTE_COLUMN, ""))
        for row in rows
        if row.get(ANSWER_COLUMN, "").strip()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--phrases", type=Path, default=Path("configs/phrases.yaml"))
    parser.add_argument("--treated", type=Path, default=Path("configs/phrases_ablation.yaml"))
    parser.add_argument("--corrections", type=Path, default=Path("data/corrections"))
    parser.add_argument("--out", type=Path, default=Path("data/corrections/verbalizacion.tsv"))
    args = parser.parse_args(argv)
    root = args.root.resolve()

    base = load_yaml_phrases(root / args.phrases)
    treated = load_yaml_phrases(root / args.treated)
    forms = {pid[:-1]: accepted(base[pid[:-1]], treated[pid]) for pid in treated}

    # El brazo base ya esta transcripto a oido: de ahi sale el veredicto de los clips que el
    # tratamiento no cambio, y el sha que permite reconocerlos.
    heard: dict[tuple[str, str, str], tuple[str, str]] = {}
    with (root / args.corrections / "normalizacion.tsv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["generation_id"].endswith("__s1234__ref0"):
                heard[(row["modelo"], row["phrase_id"], row["hablante"])] = (
                    row["audio_sha"],
                    row["dicho"],
                )

    answers = read_answers(root / args.out)
    rows, carried, pending = [], 0, 0
    for manifest in sorted((root / "data/manifests").glob("*_norm.jsonl")):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            parts = row["generation_id"].split("__")
            if len(parts) != 5 or row.get("error") or not row.get("wav_path"):
                continue
            model, speaker, phrase_id = parts[0], parts[1], parts[2][:-1]
            wav = root / row["wav_path"]
            sha = hashlib.sha256(wav.read_bytes()).hexdigest()[:12]
            base_sha, base_said = heard.get((model, phrase_id, speaker), ("", ""))
            if sha in answers:
                answer, note = answers[sha]
                carried += 1
            elif base_sha and sha == base_sha:
                answer, note = ("si" if said_ok(forms[phrase_id], base_said) else "no"), CARRIED
                carried += 1
            else:
                answer, note = "", ""
                pending += 1
            rows.append(
                [
                    row["generation_id"],
                    sha,
                    model,
                    speaker,
                    phrase_id,
                    str(treated[phrase_id + "n"]["text"]),
                    AND.join(OR.join(" ".join(f) for f in group) for group in forms[phrase_id]),
                    row["wav_path"],
                    answer,
                    note,
                ]
            )

    rows.sort(key=lambda r: (r[AT["modelo"]], r[AT["phrase_id"]], r[AT["hablante"]]))
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(["\t".join(COLUMNS)] + ["\t".join(r) for r in rows]) + "\n", encoding="utf-8"
    )
    print(f"{out}: {len(rows)} clips, {carried} ya contestados, {pending} para escuchar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
