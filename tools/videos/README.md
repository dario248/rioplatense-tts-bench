# Audiogram renderer

Renders the short videos of the LinkedIn series from the benchmark's own data. No metric is typed
into a storyboard: the renderer looks every value up by `generation_id` and stops if one is
missing.

```bash
uv venv --python 3.11 .venv-video
uv pip install --python .venv-video -e ".[video]"

# los tres juntos: comparten el nivel de audio
.venv-video/bin/python -m tools.videos.render tools/videos/storyboards/post_0{1,2,3}.yaml

# la version en español del mismo storyboard
.venv-video/bin/python -m tools.videos.render --lang es tools/videos/storyboards/post_02.yaml
```

Output goes to `media/linkedin/` (gitignored): `<name>.mp4`, `<name>_thumb.png` (first frame) and
`<name>_sources.txt`, which lists every clip, what normalizing it did and the repo commit.

`ffmpeg` is a system dependency, not a wheel. Fonts: DejaVu Sans, which matplotlib bundles, so the
output does not depend on what is installed.

## Where the numbers come from

| on screen | source |
|---|---|
| `{model}`, `{phrase_id}`, `{seed}` | `data/manifests/<model>.jsonl` |
| `{text}` | `configs/phrases.yaml`, by `phrase_id` |
| `{wer}`, `{cer}`, `{tail_s}` | `data/metrics/failures.jsonl` |
| `{cosine}` | `data/metrics/speaker_sim.jsonl` |
| `{sim_scaled}` | the cosine placed between the floor and ceiling of `data/metrics/speaker_scale.jsonl` |
| the last-word line, the word highlight, the `focus` band | `data/metrics/transcripts.jsonl` |
| where the tail ends | `data/metrics/audio_shape.jsonl` (`speech_end_s`) |

Audio is read from `data/samples/`, which is committed, so a clean clone renders without
regenerating anything. The numbers a card states by hand — the three on the end card of post 1 —
are checked against the README tables, not resolved.

## Storyboard format

A list of segments, rendered in order with 0.4 s of silence between them. Every text field takes
an `_es` twin for the Spanish render; unknown keys are an error.

```yaml
lang: en
segments:
  - type: card
    duration: 4.5
    title: "..."          # plus optional lines, stats, note
  - type: clip
    source: reference      # or: generation
    speaker: slr61_f_02    # references take a speaker; generations a generation_id
    label: "{model}"
    sublabel: "identity {sim_scaled:.2f}"
    view: spectrogram      # spectrogram | waveform
    mark_last_word: false  # draws the end of the sentence and the seconds after it
    highlight_words: false # colours the word being spoken, if the ASR matched the phrase
    focus: "Callao"        # the word to listen for: coloured in the caption, its span
                           # bracketed on the spectrogram, from its own timestamp
    trim: {word: "Callao", pad: 0.3}   # cuts the clip around a word; post 3 does not use it
    hold: 0.6
```
