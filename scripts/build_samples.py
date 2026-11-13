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

It imports no `rtb`: it has to run in any environment.
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


PAGE_CSS = """
:root {
  color-scheme: light dark;
  --bg: #fcfcfb; --card: #ffffff; --ink: #0b0b0b; --ink2: #3f3e3b; --muted: #7a7973;
  --rule: #e6e5e0; --accent: #2a78d6; --ref: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141413; --card: #1c1c1a; --ink: #f0efec; --ink2: #c3c2b7; --muted: #8e8d85;
    --rule: #2e2e2b; --accent: #3987e5; --ref: #d95926;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink); line-height: 1.5;
  font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 56rem; margin: 0 auto; padding: 2.5rem 1rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 .4rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 0 0 .5rem; letter-spacing: -.01em; }
p { color: var(--ink2); margin: 0 0 1rem; }
a { color: var(--accent); }
.lede { font-size: .95rem; }
.meta { color: var(--muted); font-size: .85rem; }
section { border-top: 1px solid var(--rule); padding-top: 1.5rem; margin-top: 2rem; }
.clip {
  background: var(--card); border: 1px solid var(--rule); border-radius: 10px;
  padding: .75rem .9rem; margin-bottom: .6rem;
}
.clip.ref { border-left: 3px solid var(--ref); }
.clip-head { display: flex; flex-wrap: wrap; gap: .5rem .75rem; align-items: baseline; }
.model { font-weight: 700; font-size: .95rem; }
.nums { color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.heard { color: var(--ink2); font-size: .88rem; margin: .35rem 0 .5rem; }
.heard span { color: var(--muted); }
audio { width: 100%; height: 34px; display: block; }
code { font-size: .78rem; color: var(--muted); word-break: break-all; }
@media (max-width: 480px) { .wrap { padding: 1.5rem 1rem 3rem; } }
"""

PAGE_INTRO = (
    "Every clip here is the audible evidence of something measured in the "
    '<a href="https://github.com/dario248/rioplatense-tts-bench">benchmark</a>, and carries '
    "the <code>generation_id</code> that produced it, so it can be regenerated from the manifest "
    "and checked against its number. Clips marked <em>real speaker</em> are recordings from "
    '<a href="https://openslr.org/61/">OpenSLR SLR61</a> (CC BY-SA 4.0), included because a '
    "cloned voice cannot be judged without the original."
)

MODEL_NAMES = {
    "xtts_v2": "XTTS-v2",
    "qwen3_tts": "Qwen3-TTS",
    "chatterbox_v3": "Chatterbox v3",
    "sintetica": "Sintética",
}


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_page(path: Path, page_claims: list[tuple[Claim, list[dict]]]) -> None:
    """La página de escucha, desde los mismos datos que el README del set.

    Las rutas a los wav son relativas, así que GitHub Pages tiene que servir la raíz del
    repositorio y no `docs/`.
    """
    out = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>rioplatense-tts-bench — listening set</title>",
        f"<style>{PAGE_CSS}</style>",
        "</head>",
        "<body><div class=wrap>",
        "<h1>Zero-shot TTS in Rioplatense Spanish — the listening set</h1>",
        f"<p class=lede>{PAGE_INTRO}</p>",
    ]
    for claim, rows in page_claims:
        out += [
            f'<section id="{claim.slug}">',
            f"<h2>{esc(claim.title)}</h2>",
            f"<p>{esc(claim.listen_for)}</p>",
        ]
        for row in rows:
            model = MODEL_NAMES.get(row["model"]) or str(row["model"])
            nums = (
                "real speaker"
                if row["is_ref"]
                else f"WER {row['wer']} · tail {row['tail']} · cosine {row['cosine']}"
            )
            heard = (
                ""
                if row["is_ref"]
                else f"<p class=heard><span>heard:</span> “{esc(row['heard'] or '—')}”</p>"
            )
            out += [
                f'<div class="clip{" ref" if row["is_ref"] else ""}">',
                "<div class=clip-head>",
                f"<span class=model>{esc(model)}</span>",
                f"<span class=nums>{esc(nums)}</span>",
                "</div>",
                heard,
                f'<audio controls preload=none src="{row["src"]}"></audio>',
                f"<code>{esc(Path(row['src']).name)}</code>",
                "</div>",
            ]
        out.append("</section>")
    out += ["</div></body></html>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


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
        "`*__reference.wav` clips are excerpts of real recordings from",
        "[OpenSLR SLR61](https://openslr.org/61/), included because a cloned voice cannot be "
        "judged without",
        "the original. They are redistributed under CC BY-SA 4.0, the corpus licence, and so is "
        "anything",
        "derived from them; the code and the measurements are MIT.",
        "",
    ]

    copied = 0
    page_claims: list[tuple[Claim, list[dict]]] = []
    for claim in CLAIMS:
        page_rows: list[dict] = []
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
            page_rows.append(
                {
                    "model": reference,
                    "is_ref": True,
                    "src": f"{args.dest}/{claim.slug}/{target.name}",
                }
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
            page_rows.append(
                {
                    "model": gid.split("__")[0],
                    "is_ref": False,
                    "wer": fmt(values.get("wer")),
                    "tail": "—" if tail is None else f"{tail:.2f} s",
                    "cosine": fmt(sims.get(gid, {}).get("cosine")),
                    "heard": heard,
                    "src": f"{args.dest}/{claim.slug}/{target.name}",
                }
            )
        page_claims.append((claim, page_rows))
        lines.append("")

    if args.dry_run:
        print("\n".join(lines))
        return 0

    (dest / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_page(root / "index.html", page_claims)
    print(f"{copied} clips en {dest}, y la página en index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
