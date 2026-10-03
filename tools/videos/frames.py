"""Frames for a 1080×1350 audiogram, built to be read muted on a phone.

Two rules shape everything here. The spectrogram is drawn ONCE per segment and the playhead is
composited onto a copy of that array per frame — 900 matplotlib renders per video would take
minutes and look identical. And a frame that changes its text (the ✓ that becomes a ?, the word
being spoken) is a separate pre-rendered base, cached by state, not a redraw.

Colors: the spectrogram is a single cool hue, dark to light, so magnitude reads as one ramp. The
only other color is a warm accent, as far from that ramp in hue as it gets, and it means "look
here". Marks that sit on top of the spectrogram carry a dark casing, because the accent alone
only clears 2.3:1 against the brightest end of the ramp.
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib import mlab
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

logger = logging.getLogger("tools.videos.frames")

WIDTH, HEIGHT, DPI, FPS = 1080, 1350, 100, 30
MARGIN = 88

BG = "#0E1117"
INK = "#ECEFF4"
MUTED = "#9AA3B2"
ACCENT = "#FF7A45"
RAMP = ["#0E1117", "#16304A", "#1E5B86", "#3E95C8", "#8FCDEA", "#E4F2FB"]
SPECTRO = LinearSegmentedColormap.from_list("rtb_cool", RAMP)

TITLE_PX = 72
LABEL_PX = 60
BODY_PX = 46
SMALL_PX = 40
TICK_PX = 28
PROVENANCE_PX = 24

FONT = "DejaVu Sans"

SPEC_NFFT = 1024
SPEC_OVERLAP = 768
SPEC_FMAX = 10_000
SPEC_RANGE_DB = 55.0

STRINGS = {
    "en": {
        "ends_here": "the sentence ends here",
        "after": "+{tail} s after the last word",
        "seconds": "seconds",
        "listen_here": "listen here",
    },
    "es": {
        "ends_here": "acá termina la oración",
        "after": "+{tail} s después de la última palabra",
        "seconds": "segundos",
        "listen_here": "escuchá acá",
    },
}


def number(value: float, digits: int, lang: str) -> str:
    """Decimal comma in Spanish: a video that says "0.88" in Spanish reads as a typo."""
    text = f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "es" else text


def pt(pixels: float) -> float:
    return pixels * 72.0 / DPI


def fx(pixels: float) -> float:
    return pixels / WIDTH


def fy(pixels: float) -> float:
    """Figure fraction for a y measured in pixels from the TOP."""
    return 1.0 - pixels / HEIGHT


def new_figure() -> Figure:
    fig = Figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(BG)
    return fig


def canvas_of(fig: Figure) -> FigureCanvasAgg:
    canvas = fig.canvas
    assert isinstance(canvas, FigureCanvasAgg)
    return canvas


def to_rgb(fig: Figure) -> np.ndarray:
    canvas = canvas_of(fig)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    if image.shape[:2] != (HEIGHT, WIDTH):
        raise RuntimeError(f"figure rendered at {image.shape[:2]}, expected {(HEIGHT, WIDTH)}")
    return image


def _text_width(fig: Figure, text: str, size_px: float, weight: str) -> float:
    artist = fig.text(0, 0, text, fontsize=pt(size_px), fontweight=weight, fontfamily=FONT)
    width = artist.get_window_extent(canvas_of(fig).get_renderer()).width
    artist.remove()
    return float(width)


def wrap(
    fig: Figure, text: str, size_px: float, max_px: float, weight: str = "normal"
) -> list[str]:
    """Greedy wrap measured against the real font, not an estimated character width."""
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width(fig, candidate, size_px, weight) <= max_px:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_block(
    fig: Figure,
    lines: list[str],
    *,
    top_px: float,
    size_px: float,
    color: str,
    weight: str = "normal",
    leading: float = 1.35,
    left_px: float = MARGIN,
) -> float:
    """Draw stacked lines from a top edge; return the y of the next free pixel."""
    y = top_px
    for line in lines:
        fig.text(
            fx(left_px),
            fy(y + size_px * 0.82),
            line,
            fontsize=pt(size_px),
            color=color,
            fontweight=weight,
            fontfamily=FONT,
            va="baseline",
            ha="left",
        )
        y += size_px * leading
    return y


def draw_words(
    fig: Figure,
    words: list[str],
    *,
    top_px: float,
    size_px: float,
    color: str,
    highlight: int | None,
    max_px: float,
    focus: int | None = None,
    left_px: float = MARGIN,
    leading: float = 1.35,
) -> float:
    """Same as `draw_block`, one `text` artist per word, so one of them can wear the accent.

    `highlight` follows the playhead; `focus` does not move — it is the word the viewer was told
    to listen for, and it is already coloured before the audio reaches it.
    """
    space = _text_width(fig, "n n", size_px, "normal") - _text_width(fig, "nn", size_px, "normal")
    space = max(space, size_px * 0.28)
    x, y = left_px, top_px
    for index, word in enumerate(words):
        marked = index in (highlight, focus)
        # El ancho se mide con el peso con el que se dibuja: la palabra resaltada va en bold y
        # midiendola en regular el renglon se come el espacio que sigue.
        weight = "bold" if marked else "normal"
        width = _text_width(fig, word, size_px, weight)
        if x > left_px and x + width > left_px + max_px:
            x, y = left_px, y + size_px * leading
        fig.text(
            fx(x),
            fy(y + size_px * 0.82),
            word,
            fontsize=pt(size_px),
            color=ACCENT if marked else color,
            fontweight=weight,
            fontfamily=FONT,
            va="baseline",
            ha="left",
        )
        x += width + space
    return y + size_px * leading


# ── composición por frame ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlotRect:
    """Where the spectrogram landed, in image pixels, so the playhead can be drawn with numpy."""

    left: int
    right: int
    top: int
    bottom: int


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)


ACCENT_RGB = _rgb(ACCENT)
BG_RGB = _rgb(BG)


def draw_playhead(image: np.ndarray, rect: PlotRect, position: float) -> None:
    """A 4 px accent bar with a 2 px dark casing: the ramp's bright end is too close in contrast."""
    x = int(round(rect.left + position * (rect.right - rect.left)))
    x = max(rect.left, min(rect.right - 1, x))
    for offset, color in ((-3, BG_RGB), (-2, BG_RGB), (3, BG_RGB), (2, BG_RGB)):
        column = x + offset
        if rect.left <= column < rect.right:
            image[rect.top : rect.bottom, column] = color
    for offset in (-1, 0, 1):
        column = x + offset
        if rect.left <= column < rect.right:
            image[rect.top : rect.bottom, column] = ACCENT_RGB


# ── segmentos ────────────────────────────────────────────────────────────────────────────────────


class Rendered:
    """A segment that knows its own duration, its audio, and the frame at any time in it."""

    def __init__(self, duration: float, audio: np.ndarray, frame: Callable[[float], np.ndarray]):
        self.duration = duration
        self.audio = audio
        self.frame = frame


def render_card(
    *,
    title: str,
    lines: list[str],
    stats: list[str],
    note: str | None,
) -> np.ndarray:
    fig = new_figure()
    usable = WIDTH - 2 * MARGIN

    title_lines = wrap(fig, title, TITLE_PX, usable, "bold")
    body_lines = [wrap(fig, line, BODY_PX, usable) for line in lines]

    gap = BODY_PX * 0.5
    height = len(title_lines) * TITLE_PX * 1.22
    if body_lines:
        height += BODY_PX * 1.1 + sum(len(b) for b in body_lines) * BODY_PX * 1.35
        height += gap * (len(body_lines) - 1)
    if stats:
        height += BODY_PX * 1.2 + len(stats) * BODY_PX * 1.6

    top = max(MARGIN * 1.6, (HEIGHT - height) / 2 - (SMALL_PX * 2 if note else 0))
    y = draw_block(fig, title_lines, top_px=top, size_px=TITLE_PX, color=INK, weight="bold",
                   leading=1.22)

    if body_lines:
        y += BODY_PX * 1.1
        for position, block in enumerate(body_lines):
            if position:
                y += gap
            y = draw_block(fig, block, top_px=y, size_px=BODY_PX, color=MUTED)

    if stats:
        y += BODY_PX * 1.2
        for stat in stats:
            fig.patches.append(
                _bullet(fig, x_px=MARGIN + 6, y_px=y + BODY_PX * 0.45, size_px=BODY_PX * 0.26)
            )
            draw_block(
                fig, [stat], top_px=y, size_px=BODY_PX, color=INK, weight="bold",
                left_px=MARGIN + BODY_PX * 0.9,
            )
            y += BODY_PX * 1.6

    if note:
        note_lines = wrap(fig, note, SMALL_PX, usable)
        bottom = HEIGHT - MARGIN - len(note_lines) * SMALL_PX * 1.35
        draw_block(fig, note_lines, top_px=bottom, size_px=SMALL_PX, color=MUTED)

    return to_rgb(fig)


def _bullet(fig: Figure, *, x_px: float, y_px: float, size_px: float):
    from matplotlib.patches import Circle

    return Circle(
        (fx(x_px), fy(y_px)),
        radius=size_px / WIDTH,
        transform=fig.transFigure,
        facecolor=ACCENT,
        edgecolor="none",
    )


def spectrogram_db(mono: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """Log-magnitude spectrogram, and the duration its columns span."""
    if len(mono) < SPEC_NFFT:
        mono = np.pad(mono, (0, SPEC_NFFT - len(mono)))
    power, freqs, _ = mlab.specgram(
        mono, NFFT=SPEC_NFFT, Fs=rate, noverlap=SPEC_OVERLAP, window=mlab.window_hanning
    )
    keep = np.asarray(freqs) <= SPEC_FMAX
    return 10.0 * np.log10(np.maximum(np.asarray(power)[keep], 1e-12)), len(mono) / rate


@dataclass
class ClipVisual:
    """Everything the clip frame needs that does not depend on time."""

    label: str
    sublabel: str | None
    caption: str
    duration: float
    db: np.ndarray
    view: str
    envelope: np.ndarray | None
    provenance: str
    caption_words: list[str] | None
    word_spans: list[tuple[float, float]] | None
    boundary: float | None
    tail_s: float | None
    badge: str | None
    lang: str
    focus_index: int | None = None
    focus_span: tuple[float, float] | None = None


def render_clip_base(
    visual: ClipVisual, *, vmin: float, vmax: float, word: int | None, after_boundary: bool
) -> tuple[np.ndarray, PlotRect]:
    fig = new_figure()
    usable = WIDTH - 2 * MARGIN
    strings = STRINGS[visual.lang]

    draw_block(fig, [visual.label], top_px=108, size_px=LABEL_PX, color=INK, weight="bold")
    if visual.sublabel:
        # El sublabel tiene que caber entre el label y el grafico: se achica antes de pasar a
        # dos lineas, y nunca pasa de dos.
        sub_px = SMALL_PX
        sub_lines = wrap(fig, visual.sublabel, sub_px, usable)
        while len(sub_lines) > 1 and sub_px > 32:
            sub_px -= 2
            sub_lines = wrap(fig, visual.sublabel, sub_px, usable)
        draw_block(fig, sub_lines[:2], top_px=108 + LABEL_PX * 1.3, size_px=sub_px, color=MUTED)

    if visual.badge:
        fig.text(
            fx(WIDTH - MARGIN),
            fy(108 + LABEL_PX * 0.82),
            visual.badge + (" ?" if after_boundary else " ✓"),
            fontsize=pt(SMALL_PX + 4),
            color=ACCENT if after_boundary else INK,
            fontweight="bold",
            fontfamily=FONT,
            va="baseline",
            ha="right",
        )

    top_px, bottom_px = 296.0, 892.0
    axes = fig.add_axes(
        (fx(MARGIN), fy(bottom_px), usable / WIDTH, (bottom_px - top_px) / HEIGHT)
    )
    axes.set_facecolor(BG)
    if visual.view == "waveform" and visual.envelope is not None:
        times = np.linspace(0, visual.duration, len(visual.envelope))
        axes.fill_between(times, -visual.envelope, visual.envelope, color=RAMP[3], linewidth=0)
        axes.set_xlim(0, visual.duration)
        axes.set_ylim(-1.05, 1.05)
        axes.set_yticks([])
    else:
        axes.imshow(
            visual.db,
            origin="lower",
            aspect="auto",
            cmap=SPECTRO,
            vmin=vmin,
            vmax=vmax,
            extent=(0.0, visual.duration, 0.0, SPEC_FMAX / 1000.0),
            interpolation="nearest",
        )
        axes.set_yticks([0, 5, 10])
        axes.set_yticklabels(["0", "5", "10"])
        axes.set_ylabel("kHz", color=MUTED, fontsize=pt(TICK_PX), fontfamily=FONT, labelpad=6)

    axes.set_xlim(0, visual.duration)
    step = 1.0 if visual.duration > 2.5 else 0.5
    axes.set_xticks(np.arange(0, visual.duration + 1e-6, step))
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
    axes.set_xlabel(strings["seconds"], color=MUTED, fontsize=pt(TICK_PX), fontfamily=FONT,
                    loc="right", labelpad=2)
    axes.tick_params(colors=MUTED, labelsize=pt(TICK_PX), length=4, width=1)
    for name, spine in axes.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color("#2A3140")

    if visual.focus_span is not None:
        # Un corchete, no un relleno: la banda marca la region mas interesante del cuadro y
        # teñirla es taparla. El borde la delimita y la barra de arriba la hace legible de lejos.
        start, end = visual.focus_span
        top_khz = SPEC_FMAX / 1000.0
        # Sin bordes verticales: la unica linea vertical en acento tiene que ser el playhead, o
        # adentro de la banda deja de distinguirse.
        axes.axvspan(start, end, color=ACCENT, alpha=0.09, linewidth=0, zorder=2)
        axes.fill_between(
            [start, end], top_khz * 0.965, top_khz, color=ACCENT, linewidth=0, zorder=3
        )
        axes.annotate(
            strings["listen_here"],
            xy=((start + end) / 2, top_khz * 0.975),
            xytext=(0, -8),
            textcoords="offset points",
            color=ACCENT,
            fontsize=pt(TICK_PX + 4),
            fontweight="bold",
            fontfamily=FONT,
            va="top",
            ha="center",
            zorder=4,
        )

    if visual.boundary is not None and 0 < visual.boundary < visual.duration:
        axes.axvline(visual.boundary, color=BG, linewidth=5)
        axes.axvline(visual.boundary, color=ACCENT, linewidth=2, linestyle=(0, (6, 4)))
        axes.annotate(
            strings["ends_here"],
            xy=(visual.boundary, SPEC_FMAX / 1000.0),
            xytext=(6, -10),
            textcoords="offset points",
            color=ACCENT,
            fontsize=pt(TICK_PX + 4),
            fontweight="bold",
            fontfamily=FONT,
            va="top",
            ha="left" if visual.boundary < visual.duration * 0.55 else "right",
        )

    # El caption y la lectura de la cola tienen posicion FIJA: si el caption empujara a la cola
    # hacia abajo, una frase de tres lineas la tiraria encima del pie.
    caption_top, tail_top = 980.0, 1136.0
    max_lines = 2 if visual.boundary is not None else 4
    # Un caption de una o dos palabras es el contenido del cuadro, no un pie: va en grande.
    caption_px = TITLE_PX if len(visual.caption.split()) <= 2 else BODY_PX
    lines = wrap(fig, visual.caption, caption_px, usable)
    while len(lines) > max_lines and caption_px > 32:
        caption_px -= 4
        lines = wrap(fig, visual.caption, caption_px, usable)
    if visual.caption_words and (word is not None or visual.focus_index is not None):
        draw_words(
            fig, visual.caption_words, top_px=caption_top, size_px=caption_px, color=INK,
            highlight=word, focus=visual.focus_index, max_px=usable,
        )
    else:
        draw_block(fig, lines, top_px=caption_top, size_px=caption_px, color=INK)

    # La cola no entra adentro del grafico sin pisar el playhead: va abajo, grande, que es donde
    # se lee en un celular sin sonido.
    if after_boundary and visual.tail_s is not None:
        tail_text = strings["after"].format(tail=number(visual.tail_s, 2, visual.lang))
        tail_px = BODY_PX + 6
        while _text_width(fig, tail_text, tail_px, "bold") > usable and tail_px > 34:
            tail_px -= 2
        draw_block(
            fig, [tail_text], top_px=tail_top, size_px=tail_px, color=ACCENT, weight="bold"
        )

    draw_block(
        fig,
        textwrap.wrap(visual.provenance, 96)[:1],
        top_px=HEIGHT - MARGIN + 6,
        size_px=PROVENANCE_PX,
        color="#5C6574",
    )

    image = to_rgb(fig)
    position = axes.get_position()
    rect = PlotRect(
        left=int(round(position.x0 * WIDTH)),
        right=int(round(position.x1 * WIDTH)),
        top=int(round((1 - position.y1) * HEIGHT)),
        bottom=int(round((1 - position.y0) * HEIGHT)),
    )
    return image, rect
