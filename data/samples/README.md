# Listening set

Every clip here is the audible evidence of something measured in the tables of the
[main README](../../README.md), and carries the `generation_id` that produced it, so it
can be regenerated from the manifest and checked against its number. The numbers in this
file are read from `data/metrics/`; they are not typed by hand.

`*__reference.wav` clips are excerpts of real recordings from
[OpenSLR SLR61](https://openslr.org/61/), included because a cloned voice cannot be judged without
the original. They are redistributed under CC BY-SA 4.0, the corpus licence, and so is anything
derived from them; the code and the measurements are MIT.

## A perfect transcript can hide a broken clip

Both score WER 0.000. In the first, invented speech runs for almost eight seconds after the
sentence ends and the loudest moment of the clip is in that tail; in the second, half the clip is
vocoder noise. No text metric sees either one.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `slr61_f_02__reference.wav` | — | — | — | the real speaker, for comparison |
| `xtts_v2__slr61_f_02__lex_001__s5678__ref0.wav` | 0.000 | 7.88 s | 0.591 | se armó un quilombo bárbaro en la esquina. |
| `chatterbox_v3__slr61_f_02__pro_003__s5678__ref0.wav` | 0.000 | 3.14 s | 0.740 | Mi hermano, el que vive en Córdoba, viene el sábado. |

## Twenty-four seconds for one word

The prompt is '¿Cómo?'. One model answers with 24 seconds of word salad, the other with two clean
seconds. An aggregate median shows neither.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `chatterbox_v3__slr61_f_02__cor_002__s5678__ref0.wav` | 18.000 | 6.86 s | 0.780 | como alablosación y quesadestanción y quesación de las acciones  |
| `qwen3_tts__slr61_f_02__cor_002__s5678__ref0.wav` | 0.000 | 0.14 s | 0.305 | ¿Cómo? |

## The tail is two different defects under one name

The first invents words — the ASR transcribes the '¡Oh, no!' nobody asked for. The second is a
vocoder that will not stop: noise, no speech. For an agent, one is a wrong answer and the other is
a turn that never closes.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `xtts_v2__slr61_f_02__lex_005__s1234__ref0.wav` | 0.300 | 8.02 s | 0.582 | Dejate de chamullar y poné la guita sobre la mesa. ¡Oh, no! |
| `chatterbox_v3__slr61_f_02__lex_002__s9012__ref0.wav` | 0.000 | 2.72 s | 0.632 | Me tomé el bondí hasta el laburo. |

## Under a second, identity collapses

Two words, understood perfectly, no failure flag — and the speaker embedding scores them at half
the model's own average. The second one lands below the floor of the scale: for a single word, the
instrument says it is a different person.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `slr61_f_01__reference.wav` | — | — | — | the real speaker, for comparison |
| `xtts_v2__slr61_f_02__cor_004__s9012__ref0.wav` | 0.000 | 0.22 s | 0.322 | Ya está. |
| `sintetica__slr61_f_01__cor_002__s1234__ref0.wav` | 0.000 | 0.14 s | -0.015 | ¿Cómo? |

## Eight pronunciations, eight perfect scores

'Bajate en Callao y caminá dos cuadras' is the canonical test of the Rioplatense [ʃ]: the phrase
set marks it as unmeasurable by alignment because the phoneme is intervocalic. 'Fuimos de Ushuaia
a Neuquén en auto' is measured automatically and every model scores WER 0.000 on it. Four models,
two phrases, eight pronunciations, eight perfect scores. This is where the dialect question lives
in v0.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `slr61_f_02__reference.wav` | — | — | — | the real speaker, for comparison |
| `xtts_v2__slr61_f_02__top_001__s1234__ref0.wav` | 0.000 | 0.06 s | 0.515 | Bájate en callao y camina dos cuadras. |
| `qwen3_tts__slr61_f_02__top_001__s1234__ref0.wav` | 0.000 | 0.04 s | 0.656 | Bajate en callao y camina dos cuadras. |
| `chatterbox_v3__slr61_f_02__top_001__s1234__ref0.wav` | 0.000 | 0.06 s | 0.682 | Bajate en callao y camina dos cuadras. |
| `sintetica__slr61_f_02__top_001__s1234__ref0.wav` | 0.000 | 0.06 s | 0.452 | Bájate en callao y camina dos cuadras. |
| `xtts_v2__slr61_f_02__top_003__s1234__ref0.wav` | 0.000 | 0.30 s | 0.480 | Fuimos de Ushuaia a Neuquén, en auto. |
| `qwen3_tts__slr61_f_02__top_003__s1234__ref0.wav` | 0.000 | 0.16 s | 0.653 | Fuimos de Ushuaia a Neuquén en auto. |
| `chatterbox_v3__slr61_f_02__top_003__s1234__ref0.wav` | 0.000 | 1.10 s | 0.615 | Fuimos de Ushuaia a Neuquén en auto. |
| `sintetica__slr61_f_02__top_003__s1234__ref0.wav` | 0.000 | 0.20 s | 0.490 | Fuimos de Ushuaia a Neuquén en auto. |

## The best and the worst clone, against the original

Same sentence, same speaker, both understood perfectly and neither flagged. The gap between them
is the widest in the corpus. Listen to the reference first: without the original, 'sounds similar'
means nothing.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `slr61_f_02__reference.wav` | — | — | — | the real speaker, for comparison |
| `qwen3_tts__slr61_f_02__vos_001__s9012__ref0.wav` | 0.000 | 0.14 s | 0.745 | Vos tenés razón, pero no me lo dijiste antes. |
| `xtts_v2__slr61_f_02__vos_001__s9012__ref0.wav` | 0.000 | 0.18 s | 0.362 | Vos tenés razón, pero no me lo dijiste antes. |

## The model that does not clone is the one that runs on a CPU

The same long sentence from the best zero-shot cloner and from a fine-tune that picks its voice by
id. The second is 115× cheaper on CPU and needs 16× less VRAM, and it is the only one of the four
that runs faster than real time without a GPU.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `qwen3_tts__slr61_f_02__lar_002__s1234__ref0.wav` | 0.000 | 0.32 s | 0.827 | El sistema toma el audio del usuario, lo transcribe con un model |
| `sintetica__slr61_f_02__lar_002__s1234__ref0.wav` | 0.000 | 0.28 s | 0.744 | El sistema toma el audio del usuario, lo transcribe con un model |

## Three seeds: one model varies, the other does not

The same cell rendered with three seeds. The zero-shot model gives three different clips — cosine
0.613, 0.541, 0.458 — and the fine-tune gives the same audio three times, identical to the fifth
decimal. That is what decides whether you can cache.

| file | WER | tail | cosine | what the ASR heard |
|---|---|---|---|---|
| `xtts_v2__slr61_f_02__yei_002__s1234__ref0.wav` | 0.000 | 0.12 s | 0.613 | Llámalo ya, que la llave está en el llavero. |
| `xtts_v2__slr61_f_02__yei_002__s5678__ref0.wav` | 0.000 | 0.10 s | 0.541 | Llámalo ya, que la llave está en el llavero. |
| `xtts_v2__slr61_f_02__yei_002__s9012__ref0.wav` | 0.000 | 0.12 s | 0.458 | Llamalo ya, que la llave está en el llavero. |
| `sintetica__slr61_f_02__yei_002__s1234__ref0.wav` | 0.000 | 0.18 s | 0.425 | Llámalo ya, que la llave está en el llavero. |
| `sintetica__slr61_f_02__yei_002__s5678__ref0.wav` | 0.000 | 0.18 s | 0.425 | Llámalo ya, que la llave está en el llavero. |
| `sintetica__slr61_f_02__yei_002__s9012__ref0.wav` | 0.000 | 0.18 s | 0.425 | Llámalo ya, que la llave está en el llavero. |

