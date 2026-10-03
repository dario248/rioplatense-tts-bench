# rioplatense-tts-bench

A reproducible benchmark of **zero-shot text-to-speech in Rioplatense Spanish**.

Multilingual TTS models advertise "Spanish support", but none of them report what happens to the
features that make the River Plate dialect recognizable: *yeísmo rehilado*, *voseo*, aspirated
syllable-final /s/, local vocabulary. This project measures what can be measured objectively —
speaker identity, intelligibility, failure rate and cost — publishes the thresholds that decide
each number, and says plainly which questions it does **not** answer.

**v0, measured between 2026-08-18 and 2026-09-19.** 45 phrases × 4 speakers × 3 seeds × 3 zero-shot
models = 1620 syntheses, plus a fourth model that runs out of competition. Every number below comes
from JSONL committed to this repository, and every threshold from
[`configs/thresholds.yaml`](configs/thresholds.yaml).

## Results

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética † |
|---|---|---|---|---|
| **speaker identity** (scaled, floor 0 → ceiling 1) | 0.642 | **0.848** | 0.815 | 0.601 ‡ |
| raw cosine, p50 | 0.586 | 0.725 | 0.703 | 0.559 |
| **WER**, mean / p95 | 0.106 / 0.500 | **0.012** / 0.100 | 0.108 / 0.417 | 0.060 / 0.400 |
| CER, mean | 0.078 | **0.005** | 0.120 | 0.031 |
| **failure rate** | 14.6 % | **0.0 %** | 47.9 % | 0.0 % |
| the same, without the trailing-audio rule | 7.9 % | **0.0 %** | 3.4 % | 0.0 % |
| of that, hard errors | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| **RTF on GPU**, p50 | **0.252** | 0.910 | 0.617 § | 0.009 |
| **RTF on CPU**, 4 threads, p50 | 2.486 | 5.396 | 5.597 | **0.047** |
| **peak VRAM** (reserved) | 2750 MiB | 5166 MiB | 3880 MiB | **318 MiB** |
| peak RSS on CPU, 4 threads | 4596 MiB | 7093 MiB | 7031 MiB | 2271 MiB |
| reference audio consumed | 6.0 s | 6.5–11.6 s | 6.0 s | none |

† **Sintética is not a zero-shot model and does not compete.** It is a fine-tune that picks a voice
by id, and it is in the table because it shares the four speakers, the scale and the instrument.
‡ Its similarity is measured against audio that was in its training set, so it is an **upper
bound**, not a comparable score. § With a process restart every 27 syntheses; the same grid in one
process drifts to 0.936. Both numbers are real and the difference is explained below.

Hardware: NVIDIA RTX 4050 Laptop (6141 MiB) for GPU, 4 threads fp32 for CPU. Everything ran
in-process on the same machine in the same session.

### How to read them

**Similarity needs its scale, or 0.72 means nothing.** The scale is measured on real recordings of
the same four speakers:

| point | what it compares | pairs | mean | range |
|---|---|---|---|---|
| **ceiling** | same speaker, different content | 4 | 0.827 | 0.806–0.852 |
| **floor** | different speakers, identical content | 6 | 0.155 | 0.085–0.221 |
| short ceiling | same speaker, one clip under 1.2 s | 4 | 0.671 | 0.588–0.715 |

The scaled column puts the raw cosine between that floor and that ceiling. It is not clipped at 1:
a synthesis can sit above the ceiling, and 51 of Qwen's 540 do — 38 of which stop exceeding it when
measured against a clip the model never heard, which is the next caveat at its extreme.

**Part of that similarity is the prompt, not the speaker.** Measured against the *other* reference
clip of the same speaker, every model scores lower: 0.027 in XTTS-v2, 0.032 in Qwen3-TTS and 0.035
in Chatterbox v3 of raw cosine — 0.040 to 0.053 of scale — with the sign holding in 92 to 96 % of
each model's 540 cells. It cannot be speaker identity: the two reference clips of one speaker score
0.853–0.927 against each other, above the ceiling, so a model copying only the voice would score
the same against both. It is **declared and not subtracted**, because cloning from that prompt is
the question this benchmark asks. The three carry it almost equally, so the order between them does
not depend on it; the absolute number does.

**WER is published next to a real-audio floor.** The same ASR over 12 real recordings of these
speakers gives WER 0.000, so the transcription is not adding error of its own on this material.

**The failure rate has two columns on purpose.** One rule — audible sound more than 0.5 s after the
last word — dominates it and inverts the ranking, so the rate is published with and without that
rule, and with the sweep of its threshold. Details below.

## Listen

Thirty curated clips in [`data/samples/`](data/samples/), one folder per claim, each with the
`generation_id` that produced it and the numbers it is evidence of. The shortest path through
them:

- [A perfect transcript can hide a broken clip](data/samples/01_the-text-is-perfect-the-audio-is-not/)
  — WER 0.000, and almost eight seconds of invented speech after the sentence ends.
- [Twenty-four seconds for one word](data/samples/02_a-catastrophic-failure/) — the prompt was
  "¿Cómo?".
- [Eight pronunciations, eight perfect scores](data/samples/05_what-the-metric-cannot-hear/) —
  "Callao" and "Ushuaia" in the four models. The first is the canonical test of the Rioplatense
  [ʃ] and the phrase set declares it unmeasurable by alignment; the second is measured
  automatically and every model scores WER 0.000. This is where the dialect question lives in v0.
- [The best and the worst clone](data/samples/06_identity-has-a-scale/), with the real recording
  first, because without the original "sounds similar" means nothing.

## What v0 does not claim

**Nothing about dialect realization.** The accent metric is v1 (SPEC 7.4), and without it no
category measures whether the output *sounds* Rioplatense — only whether it was understood. Two
categories are blind to their own phenomenon: Spanish spelling is yeísta, so 97.2 % of yeísmo
onsets come back as `ll…`, which covers [ʃ], [ʒ] and [ʝ] alike, and a model can produce a Castilian
[ʝ] through the whole phrase and score WER 0.000. `lexico_local` turned out the same way for
stress. What v0 publishes is identity, intelligibility, failures and cost.

**Not every category is measured the same way, and that is declared:**

| level | categories | method |
|---|---|---|
| **automatic** | yeísmo, rhotics, short utterances, long sentence, vocabulary, place names, loanwords | WER/CER against the transcript, plus a failure taxonomy from the alignment |
| **listening only** | text normalization, naturalness, and three phrases by name: `top_001` (Callao, [ka'ʃao] vs [ka'ʎao]), `rot_005` (the *pero*/*perro* minimal pair), `rot_004` (the /tɾ/ cluster) | blind binary labelling |
| **semi-automatic** (v1) | voseo, prosody | vowel-nucleus duration, final F0 slope |

Those three phrases are scored automatically for intelligibility like the rest of their
category; what needs an ear is the phenomenon they were written to expose, which no alignment can
reach — an intervocalic phoneme, an opposition between two taps, a consonant cluster.

`normalizacion` is unmeasurable with this instrument, and that is a finding, not an omission:
Whisper writes numbers back as digits, so a synthesis that correctly says *"quince mil setecientos
cincuenta"* returns as `15 750` and scores 5 errors out of 11 without the model having made one.
Its 60 rows per model are marked `unscorable`. Forcing words with `initial_prompt` fixes plain
numbers, leaves dates in digits and degrades other content ("AFIP" → "Zafip"), so the instrument
stays uniform and the category is declared out of scope for v0.

## Failures

`hard` (no audio), `trunc` (missing words at the end), `loop` (repetition **with** extra material),
`halluc` (extra words), `trailing` (audible sound after the last word). Percentages are over
classifiable rows; `normalizacion` is excluded as unscorable.

| | n | hard | trunc | loop | halluc | trailing | **any** |
|---|---|---|---|---|---|---|---|
| XTTS-v2 | 480 | 0.0 | 0.2 | 0.2 | 7.7 | 7.7 | **14.6** |
| Qwen3-TTS | 482 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| Chatterbox v3 | 501 | 0.0 | 0.2 | 1.4 | 3.2 | 45.7 | **47.9** |
| Sintética | 492 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| *real audio* | 12 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |

**Why `n` differs when every model synthesized the same 540 cells.** `n` is the classifiable rows:
540 minus the ones the ASR makes impossible to score, and all of those are `normalizacion`. The
rule is per row and detected — one side has digits, the other does not — rather than a category
switched off, so a model whose output broke the digit pattern keeps those rows. Chatterbox keeps 21
of them ("El total es 15 con su asiento"), and 9 of the 21 count as failures. Per row that is
right: garbling an amount that way *is* a failure. Across models it means the four rates are not
over the same set, so here is the same table on a uniform denominator — `normalizacion` dropped for
everyone, 480 rows each: Chatterbox goes to **48.1 %**, and to **2.7 %** without the trailing-audio
rule instead of 3.4 %. The other three do not move a decimal.

### The trailing audio, which is the number that moves

The rule fires on audible sound after the last transcribed word. It is not a false-positive
machine: the loudest frame of the tail clears −30 dB of the clip's peak in **100 %** of the 268
flagged rows, all 266 on the original list were listened to, and 265 had something in them — 99.6 %
precision. But two things about it have to be published with the rate.

**It is two different defects under one label.** Listening to every flagged row:

| | n | invented speech | noise | both | nothing |
|---|---|---|---|---|---|
| XTTS-v2 | 37 | **21 (56.8 %)** | 15 (40.5 %) | 1 | 0 |
| Chatterbox v3 | 230 | 3 (1.3 %) | **223 (97.0 %)** | 2 | 2 |
| Qwen3-TTS | 1 | 0 | 0 | 0 | 1 |

XTTS's tail is the model inventing words; Chatterbox's is a vocoder that does not stop. For a
conversational agent the first is a wrong answer and the second is a turn that will not close.

**The absolute figure depends on where the threshold is cut, and the order does not.** Percentage
of classifiable rows flagged, sweeping it:

| `tail_after_last_word_s` | XTTS-v2 | Qwen3-TTS | Chatterbox v3 |
|---|---|---|---|
| 0.2 s | 35 % | 24 % | 60 % |
| 0.3 s | 15 % | 5 % | 51 % |
| **0.5 s (in use)** | **8 %** | **0 %** | **46 %** |
| 1.0 s | 5 % | 0 % | 38 % |
| 1.5 s | 2 % | 0 % | 23 % |
| 2.0 s | 1 % | 0 % | 12 % |

XTTS moves from 35 % to 1 % across the range, so no single figure is the phenomenon. Chatterbox
stays above 12 % even demanding two full seconds of tail: that is the result, and it does not
depend on a chosen number.

**The tail is really a quantity of seconds**, and as such it is latency the RTF never sees:

| seconds after the last word | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética |
|---|---|---|---|---|
| p50 | 0.18 | 0.16 | 0.30 | 0.16 |
| p95 | 1.10 | 0.34 | **2.40** | 0.32 |
| rows over 0.5 s | 9.4 % | 0.7 % | **43.3 %** | 1.1 % |

## Cost

RTF is seconds of compute per second of audio produced; below 1.0 is faster than real time. GPU numbers come
from the full grid, CPU numbers from a stratified subset of 8 phrases × 2 speakers × 3 seeds.

| | GPU p50 | GPU p95 | CPU 1 thread | CPU 4 threads | threads buy ‖ |
|---|---|---|---|---|---|
| XTTS-v2 | 0.252 | 0.272 | 2.937 | 2.486 | 1.20× |
| Qwen3-TTS | 0.910 | 1.043 | 10.535 | 5.396 | 1.84× |
| Chatterbox v3 | 0.617 | 0.723 | 10.019 | 5.597 | 1.82× |
| Sintética | 0.009 | 0.027 | 0.103 | **0.047** | 2.10× |

‖ Median of the per-cell ratio between one and four threads, over the same 48 cells.
**How much threads buy is a property of the model, not of the machine**, and assuming it from one
model would have given the wrong number for the others.

**A difference inside the bands is not an order.** On CPU at 4 threads Qwen and Chatterbox are the
same number (5.40 and 5.60, p95 7.19 and 7.18) and are reported tied. That the order *changes*
between platforms is itself a result: on GPU Chatterbox is ahead of Qwen, on CPU they are
indistinguishable.

**Two asymmetries that inflate the GPU side by a few percent**, measured and declared rather than
corrected: the GPU figure comes from the 540-cell grid and the CPU one from a 48-cell subset, and
the same cells run 2–6 % slower in a short run than inside the grid (paired, same cells: XTTS
+0.005, Qwen +0.055). And Chatterbox's published GPU RTF uses a process restart every 27
syntheses, because in one long process it degrades monotonically from 0.65 to 1.5 — restarting 20
times is cheaper than carrying the drift, and the control is that the restarted run returned audio
of identical duration in 540 of 540 cells.

**Only one of the four reaches real time on CPU**, and it is the one that does not clone.

## Protocol asymmetries

These are differences between what each model was given, and they are declared because a benchmark
that hides them is not comparable:

- **Qwen3-TTS receives audio *and* its transcript** (its ICL mode), the other two receive audio
  only. Disabling it would degrade cloning quality by its own documentation, so measuring it that
  way would answer nothing.
- **Qwen consumes 6.5–11.6 s of reference and the others exactly 6.0 s.** Its reference cannot be
  cut to 6 s without desynchronizing the (audio, text) pair, and one speaker has no cut point at
  all. The seconds used are recorded per row.
- **Timing excludes the reference conditioning and the wav write.** Conditioning is computed once
  per clip and cached, which is what a deployment does; charging it to the first of 135 syntheses
  would turn that row into an outlier and leave the other 134 measuring something else.
- **Sintética receives no reference at all**: it selects a voice by id. Its similarity answers a
  different question from the zero-shot rows, and its 0.601 is an upper bound because the clips it
  is measured against were in its training data.

## Two honesty notes

**"0.0 % failures" means "no failure the automatic taxonomy counts", not "no errors were found".**
In the one 0.0 % model that was also listened to end to end, listening found a real error: in
`nor_005`, one speaker pronounces a single "1" where two belong. It lands in
`boundary_disagreement`, which leaves the ranking by design because the instrument cannot decide
whose segmentation is right. The instrument's own noise floor is measured: 2 rows out of 540
misattributed in the model with fewest failures, and 99.6 % precision on the tail rule, so
**differences below half a percentage point are not published as differences**. This bounds false
positives only — nobody listened to the 538 rows that passed clean.

**Three taxonomy rules changed between 2026-09-14 and 2026-09-16, and they moved only the
in-house model.** Stated plainly because it is the objection any reader should make:

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética |
|---|---|---|---|---|
| boundary cut 0.10 → 0.12 | 7.9 → 7.7 % | 0.0 % | 3.4 → 3.2 % | **1.9 → 0.0 %** |
| third loop condition | 14.6 % | 0.0 % | 47.9 % | **2.4 → 0.6 %** |
| `trailing_disagreement` precedence | 14.6 % | 0.0 % | 47.9 % | **0.6 → 0.0 %** |

The three zero-shot models move 0.2 points in total; Sintética moves 2.5. Two structural reasons,
both verifiable: it is the only deterministic model, so one bad cell appears three times with
bit-identical audio, and its failures were concentrated in a single phrase, so any rule touching
that phrase moved its whole rate. What each change rests on is a named mechanism and a listening
session, not the result: the boundary cut because the models do say "bondi" and were losing by one
character, the loop condition because it was written in the spec and not implemented and its sweep
has a wide plateau, the precedence because a transcript missing ten words at the end cannot be
charged with extra material.

## The phrase set

45 phrases in 10 categories, each written to expose a named failure mode: *yeísmo rehilado*, the
multiple trill, *voseo* stress patterns, very short utterances, local vocabulary, place names, text
normalization, English loanwords, prosody, and long sentences. 28 of them carry word-initial
targets (20 yeísmo, 8 rhotic) per speaker and seed — 1008 accent measurements already in the
corpus, waiting for the v1 metric.

Reference speech is [OpenSLR SLR61](https://openslr.org/61/) (Argentinian Spanish, CC BY-SA 4.0):
four speakers, two male and two female, the same four for every model. Reference clips are whole
utterances concatenated with 250 ms of silence, and the concatenation was verified not to damage
identity in the embedding — the weakest concatenated reference scores 0.823 against a real control
of the same speaker, above the weakest real same-speaker pair.

## Design in one paragraph

The boundary between stages is **files on disk, not imports**. Generation runs in one dirty,
disposable environment *per model* (their dependencies are mutually incompatible) and writes wavs
plus a JSONL manifest. Evaluation runs in a clean environment that reads those files and never
imports a TTS library. Analysis reads the metrics JSONL. Manifests are append-only and every row
carries the checkpoint commit, the library version and the device that produced it, so any stage
can run weeks after the previous one and a number can always be traced to what computed it.

## Reproducing

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
pytest

# Print the generation grid without synthesizing anything
python -m rtb.generate --config configs/models/dummy.yaml --dry-run

# The whole pipeline against a white-noise dummy model, no GPU
python -m rtb.generate --config configs/models/dummy.yaml
```

Each model needs its own environment; the recipes, with the reason for every pin, are in
[`envs/`](envs/). Generation and evaluation are separate commands:

```bash
# one model, its own venv
.venv-xtts/bin/python -m rtb.generate --config configs/models/xtts_v2.yaml

# every metric that needs no audio, plus the tables
python -m rtb.metrics --manifest data/manifests/xtts_v2.jsonl --sensitivity

# speaker similarity, its own venv, CPU
.venv-simil/bin/python -m rtb.metrics.speaker_sim --manifest data/manifests/xtts_v2.jsonl
```

Checkpoints are pinned to a commit, not a tag:

| model | repo | revision |
|---|---|---|
| XTTS-v2 | `coqui/XTTS-v2` | `6c2b0d75eae4b7047358e3b6bd9325f857d43f77` |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `fd4b254389122332181a7c3db7f27e918eec64e3` |
| Chatterbox v3 | `ResembleAI/chatterbox` | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |

## What is in `data/`

Manifests (one row per synthesis), metrics (one file per metric, so recomputing one never
invalidates the others), the listening TSVs — the only thing here that cannot be recomputed — and
the resource cost of each run. Audio is not committed; it regenerates from the manifests.

Open questions about these results, each with its cost and what every possible answer would mean,
live in [`ANALYSIS.md`](ANALYSIS.md). The design and the reasoning behind every threshold live in
[`SPEC.md`](SPEC.md), in Spanish.

## License

MIT for the code. The reference speech comes from [OpenSLR SLR61](https://openslr.org/61/)
(CC BY-SA 4.0).
