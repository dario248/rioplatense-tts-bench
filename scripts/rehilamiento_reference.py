"""Build the reference distribution of rehilamiento from real speech.

Two stages in one pass, per utterance: a forced aligner places the target grapheme, and a
high-over-low band ratio measures how fricative the realization is.

Runs in `.venv-simil`, which is the environment that carries torchaudio:

    .venv-simil/bin/python scripts/rehilamiento_reference.py \
        --corpus <dir> --out data/accent/reference.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtb.metrics.audio_utils import load_resampled  # noqa: E402

SAMPLE_RATE = 16_000
WINDOW_S = 0.025
HOP_S = 0.005
# Tramo que se mide desde la marca del alineador. Cubre cualquier realizacion de la consonante
# sin necesitar donde termina, que es lo que el alineador no da.
SEGMENT_S = 0.100
LOW_HZ = 1_000.0
HIGH_HZ = 2_500.0
# Rango de F0 donde buscar periodicidad. Cubre voz grave masculina y aguda femenina.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 350.0

# Prestamos y nombres propios donde <ll>/<y> no es /ʝ/: "ballet" es [ba'le]. Caen 20 a 37 dB por
# debajo de su hablante, asi que entran como no rehilamiento si no se los saca.
_NOT_THE_PHONEME = frozenset({"ballet", "fellini", "rolling", "you", "youtube", "hayek"})

# El alfabeto de MMS_FA son 29 simbolos romanizados: sin tildes, sin ñ, sin dígitos.
_ALIGNABLE = re.compile(r"^[a-z']+$")
_WORD_RE = re.compile(r"[^\W\d_]+")
_INTERVOCALIC = re.compile(r"[aeiou](ll|y)[aeiou]")

ONSET = "inicial"
INTERVOCALIC = "intervocalica"


VOWELS = frozenset("aeiou")


@dataclass(frozen=True)
class Target:
    """One grapheme to measure inside one word of one utterance."""

    word: str
    word_index: int
    char_index: int
    position: str
    vowel_index: int


def strip_accents(word: str) -> str:
    """Lowercase and drop diacritics, preserving character positions."""
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def find_targets(words: list[str]) -> list[Target]:
    """Locate every measurable <ll>/<y> in a tokenized utterance.

    The bare conjunction `y` is the vowel [i] and never a target; a word-final `y` is a glide.
    """
    targets: list[Target] = []
    for index, word in enumerate(words):
        plain = strip_accents(word)
        if plain == "y" or plain in _NOT_THE_PHONEME or not _ALIGNABLE.match(plain):
            continue
        if plain.startswith("ll") or plain.startswith("y"):
            start = 0
        else:
            match = _INTERVOCALIC.search(plain)
            if not match:
                continue
            start = match.start() + 1
        after = start + (2 if plain.startswith("ll", start) else 1)
        vowel = next((i for i in range(after, len(plain)) if plain[i] in VOWELS), -1)
        if vowel < 0:
            continue
        position = ONSET if start == 0 else INTERVOCALIC
        targets.append(Target(word, index, start, position, vowel))
    return targets


def periodicity(frame: torch.Tensor) -> float:
    """Normalized autocorrelation peak in the F0 range: 1.0 is fully periodic, 0.0 is noise.

    A proxy for voicing, not HNR. Enough to compare two conditions measured the same way.
    """
    energy = float(frame.pow(2).sum())
    if energy <= 0.0:
        return 0.0
    size = 1 << (2 * frame.numel() - 1).bit_length()
    spectrum = torch.fft.rfft(frame, n=size)
    auto = torch.fft.irfft(spectrum.abs().pow(2), n=size)[: frame.numel()]
    lo = int(SAMPLE_RATE / F0_MAX_HZ)
    hi = min(int(SAMPLE_RATE / F0_MIN_HZ), frame.numel() - 1)
    if hi <= lo:
        return 0.0
    return max(0.0, float(auto[lo:hi].max()) / energy)


def frame_energies(wave: torch.Tensor, offset: int, window: int) -> tuple[float, float, float]:
    """Low-band, high-band and total energy of one frame."""
    freqs = torch.fft.rfftfreq(window, d=1.0 / SAMPLE_RATE)
    power = torch.fft.rfft(wave[offset : offset + window] * torch.hann_window(window)).abs().pow(2)
    return (
        float(power[freqs < LOW_HZ].sum()),
        float(power[freqs > HIGH_HZ].sum()),
        float(power.sum()),
    )


def vowel_reference(wave: torch.Tensor, start_s: float) -> float | None:
    """Loudest total energy in the 40 ms that open at the vowel's mark.

    Serves as the denominator that voicing does not contaminate: it says how loud this speaker
    is right here, so the numerator reads as frication and not as the absence of a voice bar.
    """
    window = int(WINDOW_S * SAMPLE_RATE)
    hop = int(HOP_S * SAMPLE_RATE)
    first = int(start_s * SAMPLE_RATE)
    last = first + int(0.040 * SAMPLE_RATE)
    if first < 0 or last + window > wave.numel():
        return None
    best = max(frame_energies(wave, o, window)[2] for o in range(first, last, hop))
    return best if best > 0.0 else None


def measure_segment(
    wave: torch.Tensor, start_s: float, segment_s: float
) -> tuple[float, float, float] | None:
    """Peak band ratio, the voicing of the frame that peaked, and the loudest high-band energy."""
    window = int(WINDOW_S * SAMPLE_RATE)
    hop = int(HOP_S * SAMPLE_RATE)
    first = int(start_s * SAMPLE_RATE)
    last = first + int(segment_s * SAMPLE_RATE)
    if first < 0 or last + window > wave.numel():
        return None
    peak = -math.inf
    voicing = 0.0
    loudest_high = 0.0
    for offset in range(first, last, hop):
        low, high, _ = frame_energies(wave, offset, window)
        if low <= 0.0 or high <= 0.0:
            continue
        loudest_high = max(loudest_high, high)
        ratio = 10.0 * math.log10(high / low)
        if ratio > peak:
            peak, voicing = ratio, periodicity(wave[offset : offset + window])
    return None if peak == -math.inf else (peak, voicing, loudest_high)


class Aligner:
    """MMS_FA forced alignment, loaded once and reused across utterances."""

    def __init__(self) -> None:
        bundle = torchaudio.pipelines.MMS_FA
        self._model = bundle.get_model()
        self._tokenizer = bundle.get_tokenizer()
        self._aligner = bundle.get_aligner()

    def char_starts(self, wave: torch.Tensor, words: list[str]) -> list[list[float]] | None:
        """Seconds at which the aligner emits each character, per word."""
        plain = [strip_accents(w) for w in words]
        with torch.inference_mode():
            emission, _ = self._model(wave.unsqueeze(0))
            try:
                spans = self._aligner(emission[0], self._tokenizer(plain))
            except (RuntimeError, IndexError, KeyError):
                return None
        ratio = wave.numel() / emission.size(1) / SAMPLE_RATE
        return [[span.start * ratio for span in word_spans] for word_spans in spans]


def measure(path: Path, aligner: Aligner, segment_s: float) -> list[dict[str, object]]:
    """Every target in one utterance, with its measured peak."""
    text = path.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
    words = _WORD_RE.findall(text)
    targets = find_targets(words)
    if not targets:
        return []
    if not all(_ALIGNABLE.match(strip_accents(w)) for w in words):
        # Una sola palabra fuera del alfabeto aborta el alineado de la frase entera.
        return []
    wave = load_resampled(path, SAMPLE_RATE)
    starts = aligner.char_starts(wave, words)
    if starts is None:
        return []
    rows: list[dict[str, object]] = []
    for target in targets:
        word_starts = starts[target.word_index]
        if target.char_index >= len(word_starts):
            continue
        measured = measure_segment(wave, word_starts[target.char_index], segment_s)
        if measured is None or target.vowel_index >= len(word_starts):
            continue
        peak, voicing, loudest_high = measured
        vowel = vowel_reference(wave, word_starts[target.vowel_index])
        over_vowel = None if vowel is None else round(10.0 * math.log10(loudest_high / vowel), 2)
        rows.append(
            {
                "speaker": path.parent.parent.name,
                "utterance": path.stem,
                "word": target.word,
                "position": target.position,
                "peak_db": round(peak, 2),
                "voicing": round(voicing, 3),
                "over_vowel_db": over_vowel,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="root of the speaker folders")
    parser.add_argument("--out", type=Path, required=True, help="JSONL destination")
    parser.add_argument(
        "--segment-ms", type=float, default=SEGMENT_S * 1000, help="segment length to measure"
    )
    parser.add_argument(
        "--speaker-prefix",
        default="arf_,arm_",
        help="comma-separated speaker prefixes to keep (default: the Argentinian ones)",
    )
    args = parser.parse_args()

    prefixes = tuple(p for p in args.speaker_prefix.split(",") if p)
    paths = sorted(
        p for p in args.corpus.rglob("*.wav") if p.parent.parent.name.startswith(prefixes)
    )
    if not paths:
        raise SystemExit(f"ningun wav bajo {args.corpus} con prefijo {prefixes}")
    print(f"{len(paths)} enunciados de {len({p.parent.parent.name for p in paths})} hablantes")

    aligner = Aligner()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for seen, path in enumerate(paths, start=1):
            for row in measure(path, aligner, args.segment_ms / 1000):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if seen % 500 == 0:
                print(f"  {seen}/{len(paths)} · {written} instancias", flush=True)
    print(f"{written} instancias en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
