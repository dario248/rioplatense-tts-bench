"""One point of a threshold sweep, shared by the rates that have a threshold in them (SPEC 7.5)."""

from __future__ import annotations

from dataclasses import dataclass

IN_USE_SUFFIX = " (actual)"


@dataclass
class SensitivityPoint:
    rule: str
    knob: str
    value: float
    flagged: int
    total: int

    @property
    def percent(self) -> float:
        return 100.0 * self.flagged / self.total if self.total else 0.0

    @property
    def in_use(self) -> bool:
        return self.knob.endswith(IN_USE_SUFFIX)


def knob_label(name: str, value: float, in_use: float) -> str:
    """The knob's name, marked when the swept value is the one actually in use.

    The sweep is unreadable without it: a column of percentages says nothing until you can see
    which row produced the published number.
    """
    return name + (IN_USE_SUFFIX if value == in_use else "")
