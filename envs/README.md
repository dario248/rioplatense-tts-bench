# Environments

Five environments, never one. The three TTS models pull in mutually incompatible versions of
`transformers`, `torch` and `numpy`, so trying to make them share a virtualenv is a fight that
ends with a benchmark nobody can reproduce. The boundary between stages is files on disk — wavs
and JSONL — not imports (SPEC section 3).

| Environment | File | Holds | Writes |
|---|---|---|---|
| generation, one **per model** | `xtts_v2.txt`, `qwen3_tts.txt`, `chatterbox_v3.txt`, `sintetica.txt` | that model's TTS stack | `audio/`, `data/manifests/*.jsonl` |
| ASR | `eval.txt` | faster-whisper on ctranslate2, GPU | `transcripts.jsonl`, `audio_shape.jsonl` |
| speaker similarity | `speaker_sim.txt` | speechbrain + **CPU** torch | `speaker_sim.jsonl`, `speaker_scale.jsonl` |

`sintetica.txt` is the one that cannot use 3.11: its vendored coqui fork pins `torch<2.0`, which
publishes no 3.11 wheels, and the core needs `>=3.10`. The intersection is a single version.

The evaluation stage started as one environment and split into two. Not for tidiness: speechbrain
resolves cleanly against the ASR environment but brings torch with CUDA 13 wheels, and
`ctranslate2` does not declare its CUDA libraries as dependencies — it links cudnn from outside.
Two CUDA runtimes in one environment is a shared-library conflict the resolver cannot see, and it
would surface on the next re-transcription rather than at install time. `envs/speaker_sim.txt` has
the measurements. Splitting costs nothing precisely because the stages talk through files.

Everything that does not touch audio — latency, the duration screen, WER, the failure taxonomy, and
every table — runs in the core environment with no torch at all, which is why the report can be
regenerated anywhere, CI included.

The core package (`pydantic`, `pyyaml`) is installed into all of them; it is the only thing they
share. It never imports torch, and `tests/test_core_isolation.py` fails if that changes.

## Generation

```bash
uv venv --python 3.11 .venv-xtts
uv pip install --python .venv-xtts -r envs/xtts_v2.txt
uv pip install --python .venv-xtts -e .
.venv-xtts/bin/python -m rtb.generate --config configs/models/xtts_v2.yaml --dry-run
```

`uv` everywhere, and `--python <venv>` rather than activating: a shell that has one model's
virtualenv activated is the single most common way to run a model against the wrong stack. Note
that `uv venv` deliberately creates the environment *without* pip, so `python -m pip` will not
exist — use `uv pip install --python <venv>`, or `uv venv --seed` if something genuinely needs pip
inside.

Reference clips have to exist first — `scripts/build_reference_clips.py`, see
`configs/speakers.yaml`. A run aborts before synthesizing anything if a clip is missing, rather
than generating hundreds of wavs against the wrong voice.

Runs are idempotent: stop with Ctrl-C and start again, and it picks up where it left off. Nothing
is regenerated when both its manifest row and its wav are on disk.

## Metrics

Two audio passes, then everything else.

```bash
# ASR — needs the GPU, about five minutes for 552 clips
uv venv --python 3.11 .venv-eval
uv pip install --python .venv-eval -r envs/eval.txt
uv pip install --python .venv-eval -e .
.venv-eval/bin/python -m rtb.metrics.transcribe \
    --manifest data/manifests/xtts_v2.jsonl \
    --manifest data/manifests/ground_truth.jsonl

# speaker similarity — CPU, under a minute
uv venv --python 3.11 .venv-simil
uv pip install --python .venv-simil -r envs/speaker_sim.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --python .venv-simil -e .
.venv-simil/bin/python -m rtb.metrics.speaker_sim \
    --manifest data/manifests/xtts_v2.jsonl --phrases configs/phrases.yaml

# tables — no torch, no audio, runs anywhere
python -m rtb.metrics --manifest data/manifests/xtts_v2.jsonl \
                      --manifest data/manifests/ground_truth.jsonl
```

Both passes are idempotent and keyed on `metric_version`, which carries the model identity: a row
already measured with the same ASR model and quantization, or the same ECAPA checkpoint, is skipped;
one measured with a different model is redone. That is what makes it safe to add a fourth model and
re-run everything.

The `--extra-index-url` on the similarity install is not optional. Without it uv resolves torch to
the CUDA 13 wheels, which on a CUDA 12 driver leaves the GPU quietly unavailable, and pulls about
3 GB of nvidia packages this metric never uses.

## Colab

One notebook per model, which is the cheapest isolation there is — each runtime installs its own
stack and then dies. The notebooks mount Drive or write to a bucket so `audio/` and
`data/manifests/` survive the runtime.

## Pinning

Only `envs/*.txt` files that say they are frozen are frozen. Until a model has had its first real
run, the exact versions are unknown and the file says so. The manifests are the fallback source of
truth: every row carries the `lib_version`, `torch_version`, `device` and `gpu` that produced it,
so a published number can always be traced to a stack even when the txt lags behind.

**No Docker in v0/v1.** It buys nothing on Colab and nothing here yet.
