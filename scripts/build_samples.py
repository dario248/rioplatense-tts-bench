# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Curate the committed listening set: one folder per claim, with its numbers.

    uv run scripts/build_samples.py

The selection is the data structure below: every clip is there because it is the audible evidence
of something the metrics already measured, and it carries its `generation_id`, so anyone can
regenerate it from the manifest and check that this is the clip the number came from.

The table in `data/samples/README.md` is written from `data/metrics/`, never typed by hand: a
curated clip whose number drifts would otherwise keep claiming the old one.

No importa `rtb`: tiene que correr en cualquier entorno.
"""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DEST = Path("data/samples")
MANIFESTS = ["xtts_v2", "qwen3_tts", "chatterbox_v3", "sintetica"]
# El clip de referencia real va adentro del set: sin el original, "se parece" no significa nada.
# Es SLR61, CC BY-SA 4.0, atribuido en el README.
REFERENCES = {
    "slr61_f_02": Path("data/refs/slr61/slr61_f_02/ref_00.wav"),
    "slr61_f_01": Path("data/refs/slr61/slr61_f_01/ref_00.wav"),
}


@dataclass(frozen=True)
class Claim:
    """One conclusion, and the clips that let someone hear it."""

    slug: str
    title: str
    listen_for: str
    clips: list[str]
    references: list[str] = field(default_factory=list)


CLAIMS = [
    Claim(
        slug="01_the-text-is-perfect-the-audio-is-not",
        title="A perfect transcript can hide a broken clip",
        listen_for=(
            "Both score WER 0.000. In the first, invented speech runs for almost eight seconds "
            "after the sentence ends and the loudest moment of the clip is in that tail; in the "
            "second, half the clip is vocoder noise. No text metric sees either one."
        ),
        clips=[
            "xtts_v2__slr61_f_02__lex_001__s5678__ref0",
            "chatterbox_v3__slr61_f_02__pro_003__s5678__ref0",
        ],
        references=["slr61_f_02"],
    ),
    Claim(
        slug="02_a-catastrophic-failure",
        title="Twenty-four seconds for one word",
        listen_for=(
            "The prompt is '¿Cómo?'. One model answers with 24 seconds of word salad, the other "
            "with two clean seconds. An aggregate median shows neither."
        ),
        clips=[
            "chatterbox_v3__slr61_f_02__cor_002__s5678__ref0",
            "qwen3_tts__slr61_f_02__cor_002__s5678__ref0",
        ],
    ),
    Claim(
        slug="03_one-label-two-defects",
        title="The tail is two different defects under one name",
        listen_for=(
            "The first invents words — the ASR transcribes the '¡Oh, no!' nobody asked for. The "
            "second is a vocoder that will not stop: noise, no speech. For an agent, one is a "
            "wrong answer and the other is a turn that never closes."
        ),
        clips=[
            "xtts_v2__slr61_f_02__lex_005__s1234__ref0",
            "chatterbox_v3__slr61_f_02__lex_002__s9012__ref0",
        ],
    ),
    Claim(
        slug="04_short-utterances-break-cloning",
        title="Under a second, identity collapses",
        listen_for=(
            "Two words, understood perfectly, no failure flag — and the speaker embedding scores "
            "them at half the model's own average. The second one lands below the floor of the "
            "scale: for a single word, the instrument says it is a different person."
        ),
        clips=[
            "xtts_v2__slr61_f_02__cor_004__s9012__ref0",
            "sintetica__slr61_f_01__cor_002__s1234__ref0",
        ],
        references=["slr61_f_01"],
    ),
    Claim(
        slug="05_what-the-metric-cannot-hear",
        title="Eight pronunciations, eight perfect scores",
        listen_for=(
            "'Bajate en Callao y caminá dos cuadras' is the canonical test of the Rioplatense "
            "[ʃ]: the phrase set marks it as unmeasurable by alignment because the phoneme is "
            "intervocalic. 'Fuimos de Ushuaia a Neuquén en auto' is measured automatically and "
            "every model scores WER 0.000 on it. Four models, two phrases, eight pronunciations, "
            "eight perfect scores. This is where the dialect question lives in v0."
        ),
        clips=[
            f"{model}__slr61_f_02__{phrase}__s1234__ref0"
            for phrase in ("top_001", "top_003")
            for model in MANIFESTS
        ],
        references=["slr61_f_02"],
    ),
    Claim(
        slug="06_identity-has-a-scale",
        title="The best and the worst clone, against the original",
        listen_for=(
            "Same sentence, same speaker, both understood perfectly and neither flagged. The gap "
            "between them is the widest in the corpus. Listen to the reference first: without the "
            "original, 'sounds similar' means nothing."
        ),
        clips=[
            "qwen3_tts__slr61_f_02__vos_001__s9012__ref0",
            "xtts_v2__slr61_f_02__vos_001__s9012__ref0",
        ],
        references=["slr61_f_02"],
    ),
    Claim(
        slug="07_what-identity-costs",
        title="The model that does not clone is the one that runs on a CPU",
        listen_for=(
            "The same long sentence from the best zero-shot cloner and from a fine-tune that "
            "picks its voice by id. The second is 115× cheaper on CPU and needs 16× less VRAM, "
            "and it is the only one of the four that runs faster than real time without a GPU."
        ),
        clips=[
            "qwen3_tts__slr61_f_02__lar_002__s1234__ref0",
            "sintetica__slr61_f_02__lar_002__s1234__ref0",
        ],
    ),
    Claim(
        slug="08_same-input-same-output",
        title="Three seeds: one model varies, the other does not",
        listen_for=(
            "The same cell rendered with three seeds. The zero-shot model gives three different "
            "clips — cosine 0.613, 0.541, 0.458 — and the fine-tune gives the same audio three "
            "times, identical to the fifth decimal. That is what decides whether you can cache."
        ),
        clips=[
            f"{model}__slr61_f_02__yei_002__{seed}__ref0"
            for model in ("xtts_v2", "sintetica")
            for seed in ("s1234", "s5678", "s9012")
        ],
    ),
]


def load_rows(root: Path, path: Path, key: str = "generation_id") -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (root / path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row[key]] = row
    return rows


def metric_values(root: Path, name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for gid, row in load_rows(root, Path("data/metrics") / f"{name}.jsonl").items():
        if not row.get("error"):
            out[gid] = row.get("values") or {}
    return out


def fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_samples.py", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--dry-run", action="store_true", help="say what would be copied")
    args = parser.parse_args(argv)
    root: Path = args.root.resolve()
    dest = root / args.dest

    manifest: dict[str, dict] = {}
    for name in MANIFESTS:
        manifest.update(load_rows(root, Path("data/manifests") / f"{name}.jsonl"))
    failures = metric_values(root, "failures")
    sims = metric_values(root, "speaker_sim")
    transcripts = metric_values(root, "transcripts")

    missing = [gid for claim in CLAIMS for gid in claim.clips if gid not in manifest]
    if missing:
        raise SystemExit("no están en ningún manifest: " + ", ".join(missing))

    lines = [
        "# Listening set",
        "",
        "Every clip here is the audible evidence of something measured in the tables of the",
        "[main README](../../README.md), and carries the `generation_id` that produced it, so it",
        "can be regenerated from the manifest and checked against its number. The numbers in this",
        "file are read from `data/metrics/`; they are not typed by hand.",
        "",
        "`ref_XX.wav` clips are real recordings from [OpenSLR SLR61](https://openslr.org/61/)",
        "(CC BY-SA 4.0), included because a cloned voice cannot be judged without the original.",
        "",
    ]

    copied = 0
    for claim in CLAIMS:
        folder = dest / claim.slug
        # La carpeta ES la seleccion: un clip que sale de CLAIMS no puede quedar en disco
        # afirmando algo que este archivo ya no dice.
        expected = {f"{gid}.wav" for gid in claim.clips}
        expected |= {f"{name}__reference.wav" for name in claim.references}
        for stale in sorted(folder.glob("*.wav")) if folder.exists() else []:
            if stale.name not in expected:
                print(f"sobra, se borra: {stale.relative_to(dest)}")
                if not args.dry_run:
                    stale.unlink()
        # Envuelto a mano como el resto de los documentos del repo: un parrafo de una sola
        # linea hace ilegible el diff cuando cambia una palabra.
        lines += [f"## {claim.title}", "", *textwrap.wrap(claim.listen_for, width=98), ""]
        lines += ["| file | WER | tail | cosine | what the ASR heard |", "|---|---|---|---|---|"]
        for reference in claim.references:
            source = root / REFERENCES[reference]
            target = folder / f"{reference}__reference.wav"
            if not args.dry_run:
                folder.mkdir(parents=True, exist_ok=True)
                if source.exists():
                    shutil.copy2(source, target)
                    copied += 1
            lines.append(
                f"| `{target.name}` | — | — | — | the real speaker, for comparison |"
            )
        for gid in claim.clips:
            row = manifest[gid]
            source = root / str(row["wav_path"])
            target = folder / f"{gid}.wav"
            if not args.dry_run:
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                copied += 1
            values = failures.get(gid, {})
            tail = values.get("trailing_audio_s")
            heard = (transcripts.get(gid, {}).get("text") or "").strip()
            lines.append(
                f"| `{target.name}` | {fmt(values.get('wer'))} | "
                f"{'—' if tail is None else f'{tail:.2f} s'} | "
                f"{fmt(sims.get(gid, {}).get('cosine'))} | {heard[:64] or '—'} |"
            )
        lines.append("")

    if args.dry_run:
        print("\n".join(lines))
        return 0

    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{copied} clips en {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
