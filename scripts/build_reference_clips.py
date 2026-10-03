# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Build the reference, control and calibration clips declared in `configs/speakers.yaml`.

    uv run scripts/build_reference_clips.py --dataset /path/to/slr61-by-speaker
    uv run scripts/build_reference_clips.py --dataset ... --check

Concatenates whole utterances separated by a short silence, because no single SLR61 utterance
covers the 6 s reference budget. Levels are never touched and the audio is never resampled. With
`source_utterances` in the YAML it rebuilds exactly those ids; empty, it selects them by the
deterministic criteria in SPEC 5.2 and prints a fragment to paste back.

Depends on pyyaml alone and does not import `rtb`: it has to run in a bare environment.
"""

from __future__ import annotations

import argparse
import array
import sys
import wave
from pathlib import Path
from typing import Any

import yaml

GAP_S = 0.25
REF_TARGET_S = 6.5
CONTROL_TARGET_S = 5.5
MIN_UTTERANCE_S = 1.2
CALIBRATION_MIN_S = 0.7
CALIBRATION_MAX_WORDS = 4
CLIPPING_THRESHOLD = 32700
CLIPPED_SAMPLES_ALLOWED = 5
DURATION_TOLERANCE_S = 0.01


class BuildError(Exception):
    pass


def read_wav(path: Path) -> tuple[array.array, int]:
    with wave.open(str(path), "rb") as fh:
        if fh.getnchannels() != 1 or fh.getsampwidth() != 2:
            raise BuildError(f"{path}: expected 16-bit mono")
        rate = fh.getframerate()
        samples = array.array("h")
        samples.frombytes(fh.readframes(fh.getnframes()))
    return samples, rate


def write_wav(path: Path, samples: array.array, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(samples.tobytes())


def duration_of(path: Path) -> float:
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes() / fh.getframerate()


def load_utterances(speaker_dir: Path) -> dict[str, dict[str, Any]]:
    """Every usable utterance of one speaker, keyed by corpus id."""
    if not speaker_dir.is_dir():
        raise BuildError(f"speaker directory not found: {speaker_dir}")
    usable: dict[str, dict[str, Any]] = {}
    for wav in sorted(speaker_dir.glob("*.wav")):
        transcript_path = wav.with_suffix(".txt")
        if not transcript_path.exists():
            # Sin transcripcion no sirve para un control (no hay referencia de WER) y tampoco se
            # puede verificar que dice, asi que queda afuera tambien como referencia.
            continue
        text = " ".join(transcript_path.read_text(encoding="utf-8").split())
        if any(char.isdigit() for char in text):
            # Un digito en la transcripcion de un control obligaria a verbalizarlo a mano.
            continue
        samples, rate = read_wav(wav)
        clipped = sum(1 for s in samples if abs(s) >= CLIPPING_THRESHOLD)
        if clipped > CLIPPED_SAMPLES_ALLOWED:
            continue
        usable[wav.stem] = {
            "id": wav.stem,
            "path": wav,
            "duration_s": len(samples) / rate,
            "rate": rate,
            "text": text,
            "key": text.lower(),
        }
    if not usable:
        raise BuildError(f"no usable utterances in {speaker_dir}")
    return usable


def shared_sentences(pools: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Normalized sentences that every speaker read, in a fixed order."""
    keys = [{u["key"] for u in pool.values()} for pool in pools.values()]
    return sorted(set.intersection(*keys)) if keys else []


def concatenate(utterances: list[dict[str, Any]]) -> tuple[array.array, int]:
    rates = {u["rate"] for u in utterances}
    if len(rates) != 1:
        raise BuildError(f"mixed sample rates in one clip: {sorted(rates)}")
    rate = rates.pop()
    gap = array.array("h", [0] * int(GAP_S * rate))
    out = array.array("h")
    for index, utterance in enumerate(utterances):
        if index:
            out.extend(gap)
        samples, _ = read_wav(utterance["path"])
        out.extend(samples)
    return out, rate


def plan_speaker(
    pool: dict[str, dict[str, Any]],
    shared: list[str],
    clip_names: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Choose the utterances for every clip of one speaker, without overlap."""
    used: set[str] = set()
    plan: dict[str, list[dict[str, Any]]] = {}

    controls = clip_names["control"]
    if controls:
        picked = []
        for key in shared:
            candidates = [u for u in pool.values() if u["key"] == key]
            first = min(candidates, key=lambda u: u["id"])
            picked.append(first)
            used.add(first["id"])
        plan[controls[0]] = picked

    for name in clip_names["calibration"]:
        available = [
            u for u in pool.values() if u["id"] not in used and u["duration_s"] >= CALIBRATION_MIN_S
        ]
        preferred = [
            u for u in available if len(u["text"].split()) <= CALIBRATION_MAX_WORDS
        ] or available
        if not preferred:
            raise BuildError("no utterance short enough for the calibration clip")
        chosen = min(preferred, key=lambda u: (u["duration_s"], u["id"]))
        used.add(chosen["id"])
        plan[name] = [chosen]

    remaining = sorted(
        (u for u in pool.values() if u["id"] not in used and u["duration_s"] >= MIN_UTTERANCE_S),
        key=lambda u: (-u["duration_s"], u["id"]),
    )

    def take(target: float) -> list[dict[str, Any]]:
        picked: list[dict[str, Any]] = []
        while remaining and total_duration(picked) < target:
            picked.append(remaining.pop(0))
        if not picked:
            raise BuildError("ran out of usable utterances")
        return picked

    for name in controls[1:]:
        plan[name] = take(CONTROL_TARGET_S)
    for name in clip_names["ref"]:
        plan[name] = take(REF_TARGET_S)
    return plan


def total_duration(utterances: list[dict[str, Any]]) -> float:
    if not utterances:
        return 0.0
    return sum(u["duration_s"] for u in utterances) + GAP_S * (len(utterances) - 1)


def clip_entries(speaker: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(kind, clip name, raw clip mapping) for every clip declared for a speaker."""
    entries = []
    for kind, field in [
        ("ref", "ref_clips"),
        ("control", "control_clips"),
        ("calibration", "calibration_clips"),
    ]:
        for clip in speaker.get(field) or []:
            entries.append((kind, Path(clip["path"]).stem, clip))
    return entries


def build(args: argparse.Namespace) -> int:
    speakers = yaml.safe_load(Path(args.speakers).read_text(encoding="utf-8"))
    dataset = Path(args.dataset)
    pools = {
        speaker["id"]: load_utterances(dataset / speaker["source_speaker"] / args.chapter)
        for speaker in speakers
    }
    shared = shared_sentences(pools)
    print(f"sentences read by all {len(pools)} speakers: {len(shared)}", file=sys.stderr)

    report: list[dict[str, Any]] = []
    for speaker in speakers:
        pool = pools[speaker["id"]]
        entries = clip_entries(speaker)
        declared = {
            name: clip["source_utterances"]
            for _, name, clip in entries
            if clip.get("source_utterances")
        }
        if declared and len(declared) != len(entries):
            raise BuildError(
                f"{speaker['id']}: some clips declare source_utterances and others do not; "
                "fill them all in or none, otherwise the two modes would fight over utterances"
            )

        if declared:
            plan = {}
            for name, ids in declared.items():
                missing = [i for i in ids if i not in pool]
                if missing:
                    raise BuildError(
                        f"{speaker['id']}/{name}: utterances not found in the corpus: "
                        f"{', '.join(missing)}"
                    )
                plan[name] = [pool[i] for i in ids]
            mode = "rebuilt"
        else:
            clip_names: dict[str, list[str]] = {"ref": [], "control": [], "calibration": []}
            for kind, name, _ in entries:
                clip_names[kind].append(name)
            plan = plan_speaker(pool, shared, clip_names)
            mode = "selected"

        speaker_report: dict[str, Any] = {"id": speaker["id"], "mode": mode, "clips": {}}
        for kind, name, clip in entries:
            utterances = plan[name]
            target = Path(args.root) / clip["path"]
            samples, rate = concatenate(utterances)
            transcript = " ".join(u["text"] for u in utterances)
            if args.check:
                if not target.exists():
                    raise BuildError(f"{target} does not exist; run without --check first")
                actual = duration_of(target)
                expected = len(samples) / rate
                if abs(actual - expected) > DURATION_TOLERANCE_S:
                    raise BuildError(
                        f"{target}: on disk {actual:.3f}s but the recorded utterances rebuild to "
                        f"{expected:.3f}s — the clip was built with different settings"
                    )
                declared_text = clip.get("transcript")
                if declared_text is not None and " ".join(declared_text.split()) != transcript:
                    # Un texto que no coincide con el audio degrada el clonado SIN error
                    # (SPEC 6.4): este chequeo es la unica defensa contra eso.
                    raise BuildError(
                        f"{clip['path']}: the transcript in speakers.yaml does not match what the "
                        f"recorded utterances say.\n  declared: {declared_text!r}\n  rebuilt:  "
                        f"{transcript!r}"
                    )
            else:
                write_wav(target, samples, rate)
            speaker_report["clips"][name] = {
                "kind": kind,
                "path": clip["path"],
                "duration_s": round(len(samples) / rate, 2),
                "transcript": transcript,
                "source_utterances": [u["id"] for u in utterances],
            }
        report.append(speaker_report)

    verb = "checked" if args.check else "wrote"
    for speaker_report in report:
        print(f"\n# {speaker_report['id']} ({speaker_report['mode']})", file=sys.stderr)
        for name, clip in speaker_report["clips"].items():
            print(
                f"#   {verb} {name:11s} {clip['duration_s']:5.2f}s  "
                f"n={len(clip['source_utterances'])}",
                file=sys.stderr,
            )

    if not args.check or args.emit_yaml:
        # Fragmento para pegar en configs/speakers.yaml: el YAML commiteado sigue siendo la
        # fuente autoral, con sus comentarios, y no lo sobreescribe una maquina.
        print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False, width=100))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="corpus root, holding one directory per source speaker",
    )
    parser.add_argument("--speakers", type=Path, default=Path("configs/speakers.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--chapter", default="0", help="chapter subdirectory inside each speaker")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the clips on disk rebuild from the recorded utterances, and write nothing; "
        "also verify that any declared transcript matches what the utterances say",
    )
    parser.add_argument(
        "--emit-yaml",
        action="store_true",
        help="print the YAML fragment even with --check, to refresh transcripts in speakers.yaml",
    )
    args = parser.parse_args(argv)
    try:
        return build(args)
    except BuildError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
