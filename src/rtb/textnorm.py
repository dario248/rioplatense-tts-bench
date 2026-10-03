"""The one text normalization function, shared by config validation and the metrics stage.

Versioned: `NORMALIZER_VERSION` travels in `metric_version`, and a bump means the numbers are not
comparable with the ones published before (SPEC 7.1). Lives in the core because it must stay free
of heavy dependencies and CI validates `configs/phrases.yaml` against it.
"""

from __future__ import annotations

import re
import unicodedata

# "2" desde que se agrego expand_small_cardinals, que cambia el WER de toda frase donde el ASR
# escribio un numero chico en digitos.
NORMALIZER_VERSION = "2"

# Cardinales del 0 al 100 en forma YA normalizada (sin tildes), para poder canonicalizar un
# "3" que el ASR escribio donde el hablante dijo "tres".
# fmt: off
_UNITS = [
    "cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve",
    "diez", "once", "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete",
    "dieciocho", "diecinueve", "veinte", "veintiuno", "veintidos", "veintitres",
    "veinticuatro", "veinticinco", "veintiseis", "veintisiete", "veintiocho", "veintinueve",
]
# fmt: on
_TENS = {
    30: "treinta",
    40: "cuarenta",
    50: "cincuenta",
    60: "sesenta",
    70: "setenta",
    80: "ochenta",
    90: "noventa",
}


def _cardinal(number: int) -> str | None:
    if 0 <= number < len(_UNITS):
        return _UNITS[number]
    if number == 100:
        return "cien"
    if 30 <= number < 100:
        tens, unit = divmod(number, 10)
        word = _TENS[tens * 10]
        return word if unit == 0 else f"{word} y {_UNITS[unit]}"
    return None


SMALL_CARDINALS = {str(number): word for number in range(101) if (word := _cardinal(number))}

# Marcadores privados para proteger la ñ mientras se quitan los diacriticos: en español la
# tilde de la ñ no es un acento, es parte de la letra ("año" no es "ano").
_NTILDE_LOWER = "\ue000"
_NTILDE_UPPER = "\ue001"

_NON_WORD_RE = re.compile(r"[^\w\s]|_", re.UNICODE)


def strip_accents(text: str) -> str:
    """Remove diacritics, keeping ñ intact."""
    protected = text.replace("ñ", _NTILDE_LOWER).replace("Ñ", _NTILDE_UPPER)
    decomposed = unicodedata.normalize("NFD", protected)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    restored = unicodedata.normalize("NFC", without_marks)
    return restored.replace(_NTILDE_LOWER, "ñ").replace(_NTILDE_UPPER, "Ñ")


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics, drop punctuation, collapse whitespace.

    Applied to BOTH sides of every comparison: the reference text and the Whisper transcript.
    """
    lowered = strip_accents(text.lower())
    without_punctuation = _NON_WORD_RE.sub(" ", lowered)
    return " ".join(without_punctuation.split())


def normalized_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def is_normalized(text: str) -> bool:
    """Whether the text is already in normalized form — used to validate the configs."""
    return normalize_text(text) == text


def expand_small_cardinals(text: str) -> str:
    """Rewrite standalone digits from 0 to 100 as words, on already-normalized text.

    Only for phrases where a number is incidental to what is tested. Capped at 100 and at bare
    digit tokens, so it never touches the cases where the number IS the test (SPEC 7.6).
    """
    return " ".join(SMALL_CARDINALS.get(token, token) for token in text.split())
