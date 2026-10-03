"""Render one storyboard into one video.

    .venv-video/bin/python -m tools.videos.render tools/videos/storyboards/post_02.yaml

Writes `media/linkedin/<name>.mp4`, its first frame as `<name>_thumb.png`, and
`<name>_sources.txt`: every `generation_id` the video used, what normalizing it did, and the
commit the repo was on. That file is the answer to "how was this made".
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import string
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tools.videos import audio as av
from tools.videos import frames as fr
from tools.videos import storyboard as sb
from tools.videos.results import Clip, Reference, Results, ResultsError, fold

logger = logging.getLogger("tools.videos.render")

DEFAULT_OUT = Path("media/linkedin")


class RenderError(Exception):
    """The storyboard is valid but this repo cannot render it."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=12"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rel(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


class _Formatter(string.Formatter):
    """Formats each placeholder on its own, so Spanish gets its decimal comma."""

    def __init__(self, lang: str) -> None:
        self.lang = lang

    def format_field(self, value: object, format_spec: str) -> str:
        text = format(value, format_spec)
        if self.lang == "es" and isinstance(value, float):
            text = text.replace(".", ",")
        return text


def fill(template: str, values: dict[str, object], where: str, lang: str = "en") -> str:
    try:
        return _Formatter(lang).vformat(template, (), values)
    except KeyError as error:
        raise RenderError(
            f"{where}: `{template}` asks for {error} — available: {', '.join(sorted(values))}"
        ) from error
    except (ValueError, IndexError) as error:
        raise RenderError(
            f"{where}: `{template}` is not a valid format string ({error})"
        ) from error


@dataclass
class Prepared:
    """A clip segment after its audio is normalized and its numbers resolved."""

    segment: sb.ClipSegment
    visual: fr.ClipVisual
    samples: np.ndarray
    loudness: av.Loudness
    generation_id: str | None
    offset: float  # where in the ORIGINAL clip this audio starts, after a trim


def _mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=1) if samples.ndim == 2 else samples


def subject_of(segment: sb.ClipSegment, results: Results) -> Clip | Reference:
    if segment.source == "generation":
        assert segment.generation_id is not None
        return results.clip(segment.generation_id)
    assert segment.speaker is not None
    return results.reference(segment.speaker, segment.clip)


def prepare_clip(
    segment: sb.ClipSegment,
    subject: Clip | Reference,
    measurement: av.Measurement,
    target: float,
    lang: str,
    workdir: Path,
    index: int,
) -> Prepared:
    where = f"segment {index}"
    values = subject.placeholders()
    normalized = workdir / f"{index:02d}_{segment.source}.wav"
    loudness = av.apply_gain(measurement, normalized, target)
    samples = av.read(normalized)

    offset = 0.0
    caption_words: list[str] | None = None
    word_spans: list[tuple[float, float]] | None = None
    boundary: float | None = None
    tail_s: float | None = None

    if segment.trim is not None:
        if not isinstance(subject, Clip):
            raise RenderError(f"{where}: `trim` needs word timestamps, so it needs a generation")
        word = subject.find_word(segment.trim.word)
        if word is None:
            logger.warning(
                "%s: %r is not in the transcript of %s — rendering the whole clip",
                where, segment.trim.word, subject.generation_id,
            )
        else:
            start = max(0.0, word.start - segment.trim.pad)
            end = word.end + segment.trim.pad
            first = int(start * av.SAMPLE_RATE)
            last = min(len(samples), int(end * av.SAMPLE_RATE))
            if last - first < av.SAMPLE_RATE // 10:
                raise RenderError(f"{where}: the trim around {word.word!r} is under 0.1 s")
            samples = samples[first:last]
            offset = first / av.SAMPLE_RATE

    duration = len(samples) / av.SAMPLE_RATE

    if isinstance(subject, Clip):
        if segment.mark_last_word:
            position = subject.last_word_end_s - offset
            if 0 < position < duration:
                boundary = position
                tail_s = subject.tail_s
            else:
                logger.warning(
                    "%s: the last word of %s ends outside the rendered window",
                    where, subject.generation_id,
                )
        if segment.highlight_words:
            caption_words, word_spans = _align_caption(subject, offset, where)

    caption = (
        segment.caption.get(lang)
        if segment.caption
        else (segment.trim.word if segment.trim else str(values["text"]))
    )
    if caption_words is not None and segment.caption is None and segment.trim is None:
        caption = " ".join(caption_words)

    focus_index, focus_span = None, None
    if segment.focus:
        if not isinstance(subject, Clip):
            raise RenderError(f"{where}: `focus` needs word timestamps, so it needs a generation")
        focus_index, focus_span = _resolve_focus(subject, segment.focus, caption, offset, where)
        if focus_index is not None and caption_words is None:
            caption_words = caption.split()

    db, _ = fr.spectrogram_db(_mono(samples), av.SAMPLE_RATE)
    envelope = None
    if segment.view == "waveform":
        envelope = _envelope(_mono(samples))

    provenance = (
        subject.generation_id
        if isinstance(subject, Clip)
        else f"{subject.speaker_id}/{subject.clip_id} · OpenSLR SLR61, CC BY-SA 4.0"
    )

    visual = fr.ClipVisual(
        label=fill(segment.label.get(lang), values, where, lang),
        sublabel=(
            fill(segment.sublabel.get(lang), values, where, lang) if segment.sublabel else None
        ),
        caption=caption,
        duration=duration,
        db=db,
        view=segment.view,
        envelope=envelope,
        provenance=provenance,
        caption_words=caption_words,
        word_spans=word_spans,
        focus_index=focus_index,
        focus_span=focus_span,
        boundary=boundary,
        tail_s=tail_s,
        badge=(f"WER {subject.wer:.3f}" if segment.mark_last_word else None)
        if isinstance(subject, Clip)
        else None,
        lang=lang,
    )
    return Prepared(
        segment=segment,
        visual=visual,
        samples=samples,
        loudness=loudness,
        generation_id=subject.generation_id if isinstance(subject, Clip) else None,
        offset=offset,
    )


def _align_caption(
    clip: Clip, offset: float, where: str
) -> tuple[list[str] | None, list[tuple[float, float]] | None]:
    """Highlight the caption only when the ASR's words and the phrase's words line up.

    They usually do, but the clips this benchmark curates are the ones where they do not, so a
    silent index-by-index mapping would colour the wrong word exactly when it matters.
    """
    written = clip.text.split()
    heard = [w.word for w in clip.words]
    if len(written) != len(heard) or [fold(w) for w in written] != [fold(w) for w in heard]:
        logger.warning(
            "%s: %s was transcribed differently from its phrase — word highlighting is off",
            where, clip.generation_id,
        )
        return None, None
    return written, [(w.start - offset, w.end - offset) for w in clip.words]


def _resolve_focus(
    clip: Clip, target: str, caption: str, offset: float, where: str
) -> tuple[int | None, tuple[float, float] | None]:
    """Where the word to listen for sits, in the caption and on the time axis.

    The two are resolved separately — the caption comes from the phrase set, the span from the
    ASR — so a model whose transcript drifted still gets its band marked, and the caption just
    does not colour anything. Failing to find it at all is loud, because a video that says
    "listen here" and marks nothing is worse than one that marks the wrong place.
    """
    word = clip.find_word(target)
    if word is None:
        raise RenderError(
            f"{where}: {target!r} is not in the transcript of {clip.generation_id} — "
            "`focus` cannot mark a word the ASR did not hear"
        )
    span = (word.start - offset, word.end - offset)

    wanted = fold(target)
    index = next((i for i, w in enumerate(caption.split()) if fold(w) == wanted), None)
    if index is None:
        logger.warning("%s: %r is not in the caption — only the band is marked", where, target)
    return index, span


def _envelope(mono: np.ndarray, buckets: int = 1200) -> np.ndarray:
    if len(mono) < buckets:
        buckets = max(1, len(mono))
    edges = np.linspace(0, len(mono), buckets + 1).astype(int)
    peak = np.array(
        [np.abs(mono[a:b]).max(initial=0.0) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    )
    high = peak.max()
    return peak / high if high > 0 else peak


def build(
    story: sb.Storyboard,
    results: Results,
    workdir: Path,
    target: float,
    measured: dict[Path, av.Measurement],
) -> tuple[list[fr.Rendered], list[Prepared]]:
    prepared: list[Prepared] = []
    for index, segment in enumerate(story.segments):
        if isinstance(segment, sb.ClipSegment):
            subject = subject_of(segment, results)
            prepared.append(
                prepare_clip(
                    segment, subject, measured[subject.wav], target, story.lang, workdir, index
                )
            )

    # Una sola escala de dB para todo el video: si cada clip se autoescala, el que tiene menos
    # energia se dibuja igual de brillante y la comparacion visual miente.
    if prepared:
        vmax = max(float(np.percentile(p.visual.db, 99.9)) for p in prepared)
        vmin = vmax - fr.SPEC_RANGE_DB
    else:
        vmax, vmin = 0.0, -fr.SPEC_RANGE_DB

    rendered: list[fr.Rendered] = []
    queue = list(prepared)
    for segment in story.segments:
        if isinstance(segment, sb.Card):
            image = fr.render_card(
                title=segment.title.get(story.lang),
                lines=[t.get(story.lang) for t in segment.lines],
                stats=[t.get(story.lang) for t in segment.stats],
                note=segment.note.get(story.lang) if segment.note else None,
            )
            rendered.append(fr.Rendered(segment.duration, av.silence(segment.duration),
                                        lambda _, image=image: image))
        else:
            rendered.append(_clip_renderer(queue.pop(0), vmin, vmax))
    return rendered, prepared


def _clip_renderer(item: Prepared, vmin: float, vmax: float) -> fr.Rendered:
    visual = item.visual
    cache: dict[tuple[int | None, bool], tuple[np.ndarray, fr.PlotRect]] = {}

    def base(word: int | None, after: bool) -> tuple[np.ndarray, fr.PlotRect]:
        key = (word, after)
        if key not in cache:
            cache[key] = fr.render_clip_base(
                visual, vmin=vmin, vmax=vmax, word=word, after_boundary=after
            )
        return cache[key]

    def word_at(t: float) -> int | None:
        if not visual.word_spans:
            return None
        for index, (start, end) in enumerate(visual.word_spans):
            if start <= t < end:
                return index
        return None

    def frame(t: float) -> np.ndarray:
        after = visual.boundary is not None and t >= visual.boundary
        image, rect = base(word_at(t), after)
        position = min(1.0, t / visual.duration) if visual.duration > 0 else 0.0
        out = image.copy()
        fr.draw_playhead(out, rect, position)
        return out

    hold = max(0.0, item.segment.hold)
    held = visual.duration

    def frame_with_hold(t: float) -> np.ndarray:
        return frame(min(t, held))

    audio = np.concatenate([item.samples, av.silence(hold)]) if hold else item.samples
    return fr.Rendered(held + hold, audio, frame_with_hold)


def render(
    story: sb.Storyboard,
    results: Results,
    out_dir: Path,
    root: Path,
    target: float,
    measured: dict[Path, av.Measurement],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / f".{story.name}.work"
    workdir.mkdir(exist_ok=True)

    rendered, prepared = build(story, results, workdir, target, measured)

    blocks: list[np.ndarray] = []
    timeline: list[tuple[fr.Rendered, float]] = []
    clock = 0.0
    for index, item in enumerate(rendered):
        timeline.append((item, clock))
        blocks.append(item.audio)
        clock += len(item.audio) / av.SAMPLE_RATE
        if index < len(rendered) - 1:
            blocks.append(av.silence(story.gap))
            clock += story.gap

    audio_path = workdir / "timeline.wav"
    total = av.write(audio_path, blocks)

    video = out_dir / f"{story.name}.mp4"
    thumb = out_dir / f"{story.name}_thumb.png"
    frame_count = int(round(total * fr.FPS))
    logger.info("%s: %d segments, %.2f s, %d frames", story.name, len(rendered), total, frame_count)

    log = workdir / "ffmpeg.log"
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{fr.WIDTH}x{fr.HEIGHT}", "-r", str(fr.FPS), "-i", "-",
            "-i", str(audio_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
            "-movflags", "+faststart", "-r", str(fr.FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", str(av.SAMPLE_RATE), "-ac", str(av.CHANNELS),
            "-shortest", str(video),
        ],
        stdin=subprocess.PIPE,
        stderr=handle,
    )
    assert process.stdin is not None

    first: np.ndarray | None = None
    cursor = 0
    try:
        for number in range(frame_count):
            t = number / fr.FPS
            while cursor + 1 < len(timeline) and t >= timeline[cursor + 1][1]:
                cursor += 1
            item, start = timeline[cursor]
            local = min(t - start, max(item.duration - 1e-6, 0.0))
            image = item.frame(local)
            if first is None:
                first = image
            process.stdin.write(image.tobytes())
    finally:
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()
        handle.close()
    if process.wait() != 0:
        raise RenderError(f"ffmpeg failed:\n{log.read_text(encoding='utf-8')[-2000:]}")
    log.unlink(missing_ok=True)

    if first is not None:
        from PIL import Image

        Image.fromarray(first).save(thumb)

    _write_sources(out_dir / f"{story.name}_sources.txt", story, prepared, root, total, target)
    for leftover in workdir.glob("*.wav"):
        leftover.unlink()
    workdir.rmdir()
    return video


def _write_sources(
    path: Path,
    story: sb.Storyboard,
    prepared: list[Prepared],
    root: Path,
    total: float,
    target: float,
) -> None:
    lines = [
        f"{story.name} — rioplatense-tts-bench",
        f"repo commit : {git_commit(root)}",
        f"storyboard  : {rel(story.path, root)} (lang {story.lang})",
        f"rendered    : {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"duration    : {total:.2f} s",
        "",
        f"Loudness: every clip linearly gained to {target:.1f} LUFS integrated, measured with "
        f"ffmpeg ebur128, true peak kept under {av.TARGET_TP} dBTP.",
        f"The target is the loudest level all of these clips reach under that ceiling; "
        f"{av.TARGET_I} LUFS was not reachable without clipping one of them.",
        "No compression, no limiting, no EQ, no trimming.",
        "",
        f"{'source':<50} {'in LUFS':>8} {'in dBTP':>8} {'gain dB':>8} {'out LUFS':>9} "
        f"{'out dBTP':>9}  file",
    ]
    for item in prepared:
        name = item.generation_id or item.visual.provenance.split(" ·")[0]
        loud = item.loudness
        lines.append(
            f"{name:<50} {loud.input_i:>8.1f} {loud.input_tp:>8.1f} {loud.gain_db:>+8.2f} "
            f"{loud.output_i:>9.1f} {loud.output_tp:>9.1f}  {rel(loud.source, root)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.videos.render", description=__doc__)
    parser.add_argument("storyboard", type=Path, nargs="+")
    parser.add_argument("--lang", choices=sb.LANGUAGES, help="override the storyboard's language")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--target-lufs",
        type=float,
        help="force the integrated loudness instead of deriving it from the clips",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = repo_root()
    results = Results(root)

    try:
        av.require_ffmpeg()
        stories = [sb.load(path, args.lang) for path in args.storyboard]

        # Los storyboards que entran en una misma corrida comparten el nivel: si cada video
        # eligiera el suyo, los tres posts de la serie sonarian a distinto volumen.
        measured: dict[Path, av.Measurement] = {}
        for story in stories:
            for segment in story.segments:
                if isinstance(segment, sb.ClipSegment):
                    wav = subject_of(segment, results).wav
                    if wav not in measured:
                        measured[wav] = av.measure(wav)
        target = args.target_lufs or av.common_target(list(measured.values()))
        logger.info("loudness target: %.1f LUFS over %d distinct clips", target, len(measured))

        for story in stories:
            video = render(story, results, root / args.out, root, target, measured)
            print(rel(video, root))
    except (sb.StoryboardError, ResultsError, RenderError, av.AudioError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
