"""The normalization function. Changing it changes every WER and CER in the project."""

from __future__ import annotations

import pytest

from rtb.textnorm import (
    NORMALIZER_VERSION,
    is_normalized,
    normalize_text,
    normalized_words,
    strip_accents,
)


def test_version_is_declared() -> None:
    # Va a MetricRow.metric_version: sin version, un recalculo silencioso deja resultados
    # incomparables sin que nada lo diga.
    assert NORMALIZER_VERSION


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("¿Sabés si podés venir mañana?", "sabes si podes venir mañana"),
        ("¡No lo puedo creer!", "no lo puedo creer"),
        ("Bueno... a ver... ¿cómo te explico?", "bueno a ver como te explico"),
        ("info@ejemplo.com.ar", "info ejemplo com ar"),
        ("El trámite   se hace  en ARCA", "el tramite se hace en arca"),
        ("Ibas a 120 km/h", "ibas a 120 km h"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_enie_survives_accent_stripping() -> None:
    # La tilde de la ñ no es un acento: es parte de la letra. "año" no es "ano".
    assert strip_accents("añoranza") == "añoranza"
    assert normalize_text("El AÑO que viene, señor.") == "el año que viene señor"


def test_normalization_is_idempotent() -> None:
    once = normalize_text("La inflación de julio fue del 3,5%.")
    assert normalize_text(once) == once
    assert is_normalized(once)


def test_is_normalized_rejects_authored_forms() -> None:
    assert not is_normalized("La inflación fue del tres coma cinco por ciento.")
    assert is_normalized("la inflacion fue del tres coma cinco por ciento")


def test_normalized_words_of_empty_text() -> None:
    assert normalized_words("...") == []
    assert normalized_words("") == []
