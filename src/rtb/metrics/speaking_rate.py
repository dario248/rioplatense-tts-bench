"""The yardstick: seconds per spoken word, per speaker, measured on the real clips (SPEC 7.5).

Taken from the control and calibration clips, never from synthesized audio — that would make the
yardstick depend on what it measures.
"""

from __future__ import annotations

from collections.abc import Iterable

from rtb.config import CONCATENATION_GAP_S, Speaker
from rtb.schema import count_words


def speaking_rate(speaker: Speaker) -> float | None:
    """Seconds per spoken word for one speaker, or None when their clips cannot say.

    Discounts the silence concatenation inserted between utterances: counted as speech it inflates
    the rate, and then every synthesis looks too long.
    """
    seconds = 0.0
    words = 0
    for clip in [*speaker.control_clips, *speaker.calibration_clips]:
        if not clip.transcript or not clip.duration_s:
            continue
        gaps = CONCATENATION_GAP_S * max(0, len(clip.source_utterances) - 1)
        seconds += clip.duration_s - gaps
        words += count_words(clip.transcript)
    if words == 0 or seconds <= 0:
        return None
    return seconds / words


def speaking_rates(speakers: Iterable[Speaker]) -> dict[str, float]:
    """Rate per speaker id, skipping speakers whose clips cannot provide one."""
    rates = {}
    for speaker in speakers:
        rate = speaking_rate(speaker)
        if rate is not None:
            rates[speaker.id] = rate
    return rates
