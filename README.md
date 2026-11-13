# rioplatense-tts-bench

A reproducible benchmark of **zero-shot text-to-speech in Rioplatense Spanish**.

Multilingual TTS models advertise "Spanish support", but none of them report what happens to the
features that make the River Plate dialect recognizable: *yeísmo rehilado*, *voseo*, aspirated
syllable-final /s/, local vocabulary. This project measures what can be measured objectively —
speaker identity, intelligibility, failure rate and cost — publishes the thresholds that decide
each number, and says plainly which questions it does **not** answer.

**v0, measured between 2026-08-18 and 2026-09-19**, plus the first v1 model, measured on
2026-10-30. 45 phrases × 4 speakers × 3 seeds × 5 zero-shot models = 2700 syntheses, and a sixth
model that runs out of competition. Every number below comes from JSONL committed to this
repository, and every threshold from [`configs/thresholds.yaml`](configs/thresholds.yaml).

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

## Results

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética † | Pocket TTS 6L ‖ | Pocket TTS 24L ‖ |
|---|---|---|---|---|---|---|
| **speaker identity** (scaled, floor 0 → ceiling 1) | 0.642 | **0.848** | 0.815 | 0.601 ‡ | 0.340 | 0.389 |
| the same, against the model's own decoder ceiling | — ¤ | — ¤ | — ¤ | **0.727** | **0.460** | **0.525** |
| raw cosine, p50 | 0.586 | 0.725 | 0.703 | 0.559 | 0.384 | 0.417 |
| **WER**, mean / p95 | 0.106 / 0.500 | **0.011** / 0.100 | 0.093 / 0.333 | 0.056 / 0.400 | 0.056 / 0.286 | 0.063 / 0.222 |
| CER, mean | 0.078 | **0.004** | 0.111 | 0.027 | 0.038 | 0.037 |
| **failure rate** | 14.6 % | **0.0 %** | 48.1 % | 0.0 % | 3.5 % | 4.4 % |
| the same, without the trailing-audio rule | 7.9 % | **0.0 %** | 2.7 % | 0.0 % | 3.5 % | 3.5 % |
| of that, hard errors | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| **RTF on GPU**, p50 | **0.252** | 0.910 | 0.617 § | 0.009 | unsupported path ‖ | unsupported path ‖ |
| **RTF on CPU**, p50 | 2.486 | 5.396 | 5.597 | **0.047** | 0.419 | 1.378 |
| threads behind that CPU figure | 4 | 4 | 4 | 4 | 1 ¶ | 1 ¶ |
| **peak VRAM** (reserved) | 2750 MiB | 5166 MiB | 3880 MiB | **318 MiB** | — | — |
| peak RSS on CPU | 4596 MiB | 7093 MiB | 7031 MiB | 2271 MiB | **1507 MiB** | 2537 MiB |
| reference audio consumed | 6.0 s | 6.5–11.6 s | 6.0 s | none | 6.0 s | 6.0 s |

† **Sintética is not a zero-shot model and does not compete.** It is a fine-tune that picks a voice
by id, and it is in the table because it shares the four speakers, the scale and the instrument.
‡ Its similarity is measured against audio that was in its training set, so it is an **upper
bound**, not a comparable score. That pushes the figure up and the decoder ceiling below pushes it
down; neither is folded into it. § With a process restart every 27 syntheses; the same grid in one
process drifts to 0.936. Both numbers are real and the difference is explained below.

‖ **Pocket TTS is a v1 model, and CPU is the device its authors support.** `load_model` takes no
device and returns on CPU; the GPU is reached by moving the module by hand, so the quality grid ran
on CPU and the GPU column is left without a published figure. It was measured all the same, as
verification rather than as a row, under *Cost*. The two variants differ only in transformer depth:
6 layers and 109.5 M parameters against 24 layers and 336.1 M.

¤ **What each model's own decoder costs, where it could be measured.** Passing the *real* control
clips through the decoder alone — Pocket's Mimi codec, Sintética's HiFiGAN fed the mel of the clip,
no synthesis either way — drops the ceiling from 0.827 to **0.655** for Mimi and to **0.722** for
HiFiGAN. By ear the speaker stays recognizable through both. The second row rescales each measured
model against its own ceiling, which is the best a perfect clone could score through it, so **read
that row vertically against the figure above it, never across**. The em-dashes are the caveat, not
a zero: XTTS, Qwen and Chatterbox decode through a vocoder too, but theirs is not separable from
the acoustic model and went unmeasured. The one reordering the correction produces — Sintética
above XTTS — is for that reason not a result: XTTS would rise too if its own were measured.

¶ The library executes `torch.set_num_threads(1)` at import, which overrides `OMP_NUM_THREADS`.
Raising it afterwards changes nothing — measured at 1, 4 and 8 threads — so the figure is published
at one thread and compared against four.

> Hardware: NVIDIA RTX 4050 Laptop (6141 MiB) for GPU, fp32 for CPU. Everything ran in-process on the same machine, each model in its own session.

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

**Part of that similarity is the prompt, not the speaker.** Scored against the *other* reference
clip of the same speaker, each v0 zero-shot model drops 0.027 to 0.035 of raw cosine — 0.040 to
0.053 of scale — and the sign holds in 92 to 96 % of its 540 cells. It is **declared and not
subtracted**, because cloning from that prompt is the question this benchmark asks. The three carry
it almost equally, so the order between them does not depend on it; the absolute number does. The
second clip was not run for Pocket TTS, so its 0.340 and 0.389 carry an unmeasured share of the
same effect — which would make them lower, not higher. The codec ceiling below pulls those same two
numbers the other way, and neither correction is folded into the published figure. Why it cannot be speaker identity, and how
the effect splits between prompt and target, is in [`ANALYSIS.md`](ANALYSIS.md) (A10).

**WER is published next to a real-audio floor.** The same ASR over 12 real recordings of these
speakers gives WER 0.000, so the transcription is not adding error of its own on this material.
That floor holds for the text in the aggregate and not for every kind: on `normalizacion` the same
ASR scores Sintética at 0.242 where the ear scores 0.007, because it writes numbers back as digits
whatever it hears. That category is measured by ear and published apart, under *Text
normalization*.

**The failure rate has two columns on purpose.** One rule — audible sound more than 0.5 s after the
last word — dominates it and inverts the ranking, so the rate is published with and without that
rule, and with the sweep of its threshold. Details below.

## Listen

Thirty curated clips in [`data/samples/`](data/samples/), one folder per claim, each with the
`generation_id` that produced it and the numbers it is evidence of. They play in the browser at
**[dario248.github.io/rioplatense-tts-bench](https://dario248.github.io/rioplatense-tts-bench/)**.
The shortest path through them:

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

## Text normalization

**The ASR cannot score this category, so 240 clips were transcribed by ear.** Whisper writes
numbers back as digits whatever it hears, so a synthesis that correctly says *"quince mil
setecientos cincuenta"* returns as `15 750` and scores 5 errors out of 11 without the model having
made one. Forcing words with `initial_prompt` fixes plain numbers, leaves dates in digits and
degrades other content ("AFIP" → "Zafip"). The figures below are WER against what each model
actually said, and they are **not averaged into the aggregate**: two instruments do not mix.

| | WER by ASR | **WER by ear** | rows |
|---|---|---|---|
| Sintética † | 0.242 | **0.007** | 20 ‡ |
| XTTS-v2 | 0.399 | **0.177** | 60 |
| Qwen3-TTS | 0.387 | **0.230** | 60 |
| Chatterbox v3 | 0.467 | **0.395** | 60 |
| Pocket TTS 6L | 0.606 | **0.582** | 20 § |
| Pocket TTS 24L | 0.631 | **0.592** | 20 § |

‡ Deterministic: its three seeds are bit-identical, so 20 clips cover its 60 rows. § One seed per
cell, a probe rather than the full 60.

**The instrument had the ranking wrong.** By ASR, Qwen beat XTTS; by ear XTTS wins by five points.
The same pass recovered 24 real failures that the exclusion was swallowing.

### A normalizer in front fixes it

The same grid with the numbers already written out in words — what a text front-end hands the TTS —
judged by ear, yes or no, on the numeric span alone:

| | without | with |
|---|---|---|
| Pocket TTS 6L | 0/20 | **20/20** |
| Pocket TTS 24L | 0/20 | **20/20** |
| Chatterbox v3 | 0/20 | **20/20** |
| XTTS-v2 | 4/20 | 19/20 |
| Qwen3-TTS | 4/20 | 16/20 |
| Sintética † | 19/20 | 20/20 |

27/120 to 115/120 over the six models. **Sintética is the control**: it already
has a normalizer in production, and 16 of its 20 clips come out bit-identical between the two arms,
so putting another one in front is a no-op. That is what validates the method.

**Scaling the model is not the alternative.** The two Pocket variants differ 4× in depth — 336.1 M
parameters against 109.5 M — and both go 0/20 without a normalizer and 20/20 with one. What they
lack is the text front-end, not capacity.

**Two defects it does not fix.** Qwen regroups what is already in words: fed *"ochenta y nueve cero
cero"* it says *"ochenta y nueve mil"*, 0 of 4 on the phone number. Where the spoken form is itself
readable as a quantity, writing it out does not protect. And Pocket does not say acronyms — it
drops "IVA" in all four and says *"afipí"* where AFIP belongs, in both variants.

## What v0 does not claim

**Nothing about dialect realization.** The accent metric is v1 (SPEC 7.4), and without it no
category measures whether the output *sounds* Rioplatense — only whether it was understood. Two
categories are blind to their own phenomenon: Spanish spelling is yeísta, so 97.2 % of yeísmo
onsets come back as `ll…`, which covers [ʃ], [ʒ] and [ʝ] alike, and a model can produce a Castilian
[ʝ] through the whole phrase and score WER 0.000. `lexico_local` is blind the same way to stress:
a model can say *bondí* for *bóndi*, every phoneme right and the accent on the wrong syllable, and
as long as the ASR writes the word the row scores 0.000. What v0 publishes is identity,
intelligibility, failures and cost.

**Not every category is measured the same way, and that is declared:**

| level | categories | method |
|---|---|---|
| **automatic** | yeísmo, rhotics, short utterances, long sentence, vocabulary, place names, loanwords | WER/CER against the transcript, plus a failure taxonomy from the alignment |
| **human transcript** | text normalization | the same WER/CER, but against what the model actually said, written down by ear |
| **listening only** | naturalness, and three phrases by name: `top_001` (Callao, [ka'ʃao] vs [ka'ʎao]), `rot_005` (the *pero*/*perro* minimal pair), `rot_004` (the /tɾ/ cluster) | blind binary labelling |
| **semi-automatic** (v1) | voseo, prosody | vowel-nucleus duration, final F0 slope |

Those three phrases are scored automatically for intelligibility like the rest of their
category; what needs an ear is the phenomenon they were written to expose, which no alignment can
reach — an intervocalic phoneme, an opposition between two taps, a consonant cluster.

**`normalizacion` is out of the aggregate and has its own table above**, measured by ear because
this instrument cannot score it. It is a finding, not an omission, and it is published rather than
dropped.

**Its 60 rows per model are excluded by category, and until 2026-10-30 they were excluded row by
row** — when one side had digits and the other did not. That heuristic assumed a model would at
least attempt the number. Pocket TTS does not: it emits invented words where the digits belong, so
`$15.750 con IVA` comes back as *"la de Ampran, Bocún, Rua de New Conicuerto incluido"* and
`25/05/2026` as *"el vacío a Breden y Sioux Falls"*. No row of its tripped the rule, and its 60
would have been averaged in against models whose 60 were already out. **Every model now publishes
over the same 480 rows**, and what the change cost is under *Two honesty notes* below. That a model
has no number front-end at all is worth knowing before deploying one, and a table says so instead
of a paragraph.

## Failures

`hard` (no audio), `trunc` (missing words at the end), `loop` (repetition **with** extra material),
`halluc` (extra words), `trailing` (audible sound after the last word). Percentages are over
classifiable rows: 540 minus the 60 of `normalizacion`, which is out of scope for every model and
carries its own table above.

| | n | hard | trunc | loop | halluc | trailing | **any** |
|---|---|---|---|---|---|---|---|
| XTTS-v2 | 480 | 0.0 | 0.2 | 0.2 | 7.7 | 7.7 | **14.6** |
| Qwen3-TTS | 480 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| Chatterbox v3 | 480 | 0.0 | 0.0 | 1.5 | 2.7 | 46.2 | **48.1** |
| Sintética | 480 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |
| Pocket TTS 6L | 480 | 0.0 | 0.0 | 0.8 | 3.5 | 0.0 | **3.5** |
| Pocket TTS 24L | 480 | 0.0 | 0.0 | 0.0 | 3.5 | 0.8 | **4.4** |
| *real audio* | 12 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** |

### The trailing audio, which is the number that moves

The rule fires on audible sound after the last transcribed word. It is not a false-positive
machine: the loudest frame of the tail clears −30 dB of the clip's peak in **100 %** of the 268
flagged rows, all 266 on the original list were listened to, and 265 had something in them — 99.6 %
precision. That verification covers the four v0 models; the 4 rows Pocket TTS 24L contributes were
not listened to. But two things about it have to be published with the rate.

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

| seconds after the last word | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética | Pocket 6L | Pocket 24L |
|---|---|---|---|---|---|---|
| p50 | 0.18 | 0.16 | 0.30 | 0.16 | **0.10** | **0.10** |
| p95 | 1.10 | 0.34 | **2.40** | 0.32 | **0.24** | 0.26 |
| rows over 0.5 s | 9.4 % | 0.7 % | **43.3 %** | 1.1 % | 1.1 % | 1.3 % |

The two Pocket variants close a turn faster than anything else here, real audio included, and that
is the one axis where they lead.

## Cost

RTF is seconds of compute per second of audio produced; below 1.0 is faster than real time. GPU numbers come
from the full grid, CPU numbers from a stratified subset of 8 phrases × 2 speakers × 3 seeds.

| | GPU p50 | GPU p95 | CPU 1 thread | CPU 4 threads | threads buy ‖ |
|---|---|---|---|---|---|
| XTTS-v2 | 0.252 | 0.272 | 2.937 | 2.486 | 1.20× |
| Qwen3-TTS | 0.910 | 1.043 | 10.535 | 5.396 | 1.84× |
| Chatterbox v3 | 0.617 | 0.723 | 10.019 | 5.597 | 1.82× |
| Sintética | 0.009 | 0.027 | 0.103 | **0.047** | 2.10× |
| Pocket TTS 6L | *0.093* | *0.105* | **0.419** | n/a | 1.00× |
| Pocket TTS 24L | *0.251* | *0.282* | 1.378 | n/a | 1.00× |

‖ Median of the per-cell ratio between one and four threads, over the same 48 cells.
Pocket's GPU figures are in italics because the GPU is not its supported path: they verify the
speed-up and are not published as a row. They are worth stating because they are larger than the
~2.6× its authors report on a T4 — 4.2× for the 6-layer variant and 5.0× for the 24-layer one,
never below 3.2× on any single cell — and because the 24-layer variant only crosses into real time
there. Its `n/a` for four threads is not a gap: the library pins one thread at import, and forcing
four or eight moves the figure by less than the noise between repetitions (6L: 0.416, 0.425, 0.439;
24L: 1.255, 1.243, 1.336).
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

**Two of the six reach real time on CPU**: the one that does not clone, and the 6-layer Pocket,
which does it on a single core with the smallest memory footprint in the table. That is the family
v0 was missing — the other three zero-shot models ask for 4.6 to 7.1 GB of RSS and none of them
gets near real time there.

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
- **Pocket TTS splits the text into chunks of at most 50 tokens**, the other models receive the
  sentence whole. Nothing in this phrase set reaches that length, so no row was actually split, but
  the path differs and a longer corpus would exercise it.
- **Pocket TTS runs on CPU and the rest on GPU.** The grid definition does not include the device,
  so the quality comparison holds; what changes is that its RTF column is the CPU one, which is
  also the shared column where every model has a figure.

## Two honesty notes

**"0.0 % failures" means "no failure the automatic taxonomy counts", not "no errors were found".**
In the one 0.0 % model that was also listened to end to end, listening found a real error: in
`nor_005`, one speaker pronounces a single "1" where two belong. It lands in
`boundary_disagreement`, which leaves the ranking by design because the instrument cannot decide
whose segmentation is right. The instrument's own noise floor is measured: 2 rows out of 540
misattributed in the model with fewest failures, and 99.6 % precision on the tail rule, so
**differences below half a percentage point are not published as differences**. This bounds false
positives only — nobody listened to the 538 rows that passed clean.

**Four taxonomy rules changed, and the first three moved only the in-house model.** Stated plainly
because it is the objection any reader should make:

| | XTTS-v2 | Qwen3-TTS | Chatterbox v3 | Sintética |
|---|---|---|---|---|
| boundary cut 0.10 → 0.12 | 7.9 → 7.7 % | 0.0 % | 3.4 → 3.2 % | **1.9 → 0.0 %** |
| third loop condition | 14.6 % | 0.0 % | 47.9 % | **2.4 → 0.6 %** |
| `trailing_disagreement` precedence | 14.6 % | 0.0 % | 47.9 % | **0.6 → 0.0 %** |
| `normalizacion` out by category | 14.6 % | 0.0 % | **47.9 → 48.1 %** | 0.0 % |

The three zero-shot models move 0.2 points in total; Sintética moves 2.5. Two structural reasons,
both verifiable: it is the only deterministic model, so one bad cell appears three times with
bit-identical audio, and its failures were concentrated in a single phrase, so any rule touching
that phrase moved its whole rate. What each change rests on is a named mechanism and a listening
session, not the result: the boundary cut because the models do say "bondi" and were losing by one
character, the loop condition because it was written in the spec and not implemented and its sweep
has a wide plateau, the precedence because a transcript missing ten words at the end cannot be
charged with extra material.

**The fourth change is from 2026-10-30 and breaks that pattern: it moves Chatterbox and leaves
Sintética alone.** A model that never verbalizes a number showed that excluding `normalizacion` by
detection was not excluding it at all. On the failure rate the change is worth 0.2 points, but on
intelligibility it is the largest of the four — Chatterbox's mean WER goes from 0.108 to 0.093 and
its CER from 0.120 to 0.111, Sintética's WER from 0.060 to 0.056, Qwen's from 0.012 to 0.011, and
XTTS does not move because its 60 rows were already out. **Every figure in this README is the
post-change one**, so the v0 numbers published on 2026-10-03 differ.

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
