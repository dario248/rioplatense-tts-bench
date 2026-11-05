# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""What Pocket's neural codec costs the speaker-similarity metric, measured on real audio.

    .venv-pocket/bin/python scripts/codec_ceiling.py encode
    .venv-simil/bin/python  scripts/codec_ceiling.py measure

Two stages because they need different environments, with wavs on disk between them, like every
other pair of stages here. `encode` passes the control clips through Mimi — encode to latent and
decode back, no synthesis — and `measure` recomputes the ceiling and the floor with one side of
each pair carrying the codec, which is the geometry of the published metric: the reference clip is
real and never goes through it, the synthesis always does.

The clean side is the clip already resampled to 24 kHz, not the original: resampling scores cosine
1.0000 against itself, so what the second column loses is the codec and nothing else.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
from pathlib import Path
from typing import Any

from rtb.config import load_speakers

DEFAULT_OUT = Path("audio/codec_ceiling")
DEFAULT_SPEAKERS = Path("configs/speakers.yaml")
# Sufijos de archivo: cambiarlos deja huérfanos los wavs ya generados.
CLEAN_FILE, CODEC_FILE = "clean", "codec"
CLEAN, CODEC = "limpio", "por Mimi"


def control_clips(root: Path, speakers_path: Path) -> list[tuple[str, str, Path]]:
    """(hablante, nombre del clip, ruta) de cada clip de control."""
    return [
        (speaker.id, Path(clip.path).stem, root / clip.path)
        for speaker in load_speakers(root / speakers_path, root=root)
        for clip in speaker.control_clips
    ]


def encode(root: Path, out: Path, speakers_path: Path) -> int:
    """Round-trip every control clip through Mimi. Needs the Pocket environment."""
    import torch  # noqa: PLC0415 — import pesado: va acá, no a nivel de módulo
    from pocket_tts import TTSModel  # noqa: PLC0415
    from pocket_tts.data.audio import audio_read  # noqa: PLC0415
    from pocket_tts.data.audio_utils import convert_audio  # noqa: PLC0415
    from pocket_tts.modules.stateful_module import init_states  # noqa: PLC0415

    from rtb.generate.audio_io import write_wav  # noqa: PLC0415

    model: Any = TTSModel.load_model(
        language="spanish", temp=0.3, sampler_decode_steps=1, quantize=False
    )
    model.to("cpu")
    rate = int(model.sample_rate)
    out.mkdir(parents=True, exist_ok=True)

    for speaker, name, path in control_clips(root, speakers_path):
        audio, source_rate = audio_read(str(path))
        audio = convert_audio(audio, source_rate, rate, 1)
        batched = audio[None] if audio.dim() == 2 else audio
        with torch.no_grad():
            latent = model.mimi.encode_to_latent(batched)
            state = init_states(model.mimi, batch_size=1, sequence_length=10000)
            restored = model.mimi.decode_from_latent(latent, state)
        write_wav(out / f"{speaker}__{name}__{CLEAN_FILE}.wav", batched.squeeze().tolist(), rate)
        write_wav(out / f"{speaker}__{name}__{CODEC_FILE}.wav", restored.squeeze().tolist(), rate)
        print(f"  {speaker} {name}: {source_rate} Hz -> {rate} Hz")
    return 0


def measure(root: Path, out: Path, speakers_path: Path) -> int:
    """Ceiling and floor, clean and with one side through the codec. Needs the eval environment."""
    from rtb.metrics.speaker_sim import EcapaConfig, Embedder, cosine  # noqa: PLC0415

    clips = control_clips(root, speakers_path)
    # La clave lleva el clip y no sólo el hablante: el techo compara los dos clips de una persona,
    # así que tienen que seguir siendo distinguibles.
    by_speaker: dict[str, list[tuple[str, Path]]] = {}
    for speaker, name, path in clips:
        by_speaker.setdefault(speaker, []).append((name, path))
    coded = {
        (speaker, name): out / f"{speaker}__{name}__{CODEC_FILE}.wav" for speaker, name, _ in clips
    }

    embedder = Embedder(EcapaConfig())
    embedder.load()
    embed = embedder.embed
    print(f"ECAPA {embedder.metric_version}\n")

    # Techo: mismo hablante, texto distinto. Con codec van las dos direcciones, porque cualquiera
    # de los dos clips puede hacer de referencia limpia.
    ceiling_clean = [cosine(embed(c[0][1]), embed(c[1][1])) for c in by_speaker.values()]
    ceiling_codec = [
        cosine(embed(paths[index][1]), embed(coded[speaker, paths[1 - index][0]]))
        for speaker, paths in by_speaker.items()
        for index in (0, 1)
    ]
    # Piso: hablantes distintos sobre el primer clip, que es el del texto compartido.
    floor_clean, floor_codec = [], []
    for left, right in itertools.combinations(by_speaker, 2):
        shared_left, shared_right = by_speaker[left][0], by_speaker[right][0]
        floor_clean.append(cosine(embed(shared_left[1]), embed(shared_right[1])))
        floor_codec.append(cosine(embed(shared_left[1]), embed(coded[right, shared_right[0]])))
        floor_codec.append(cosine(embed(shared_right[1]), embed(coded[left, shared_left[0]])))

    mean = statistics.mean
    print(f"{'':<36} {CLEAN:>8} {CODEC:>10} {'cae':>7}")
    for label, clean, codec in (
        ("TECHO  mismo hablante, otro texto", ceiling_clean, ceiling_codec),
        ("PISO   otro hablante, mismo texto", floor_clean, floor_codec),
    ):
        print(
            f"{label:<36} {mean(clean):>8.3f} {mean(codec):>10.3f} "
            f"{mean(codec) - mean(clean):>7.3f}"
        )

    top, bottom = mean(ceiling_clean), mean(floor_clean)
    top_codec, bottom_codec = mean(ceiling_codec), mean(floor_codec)
    print("\nLa fila de un modelo que decodifica con el códec se lee contra el techo de abajo:")
    # Medianas publicadas del coseno crudo: regenerar el grid las mueve y hay que actualizarlas.
    for name, raw in (("Pocket 6c", 0.384), ("Pocket 24c", 0.417)):
        published = (raw - bottom) / (top - bottom)
        corrected = (raw - bottom_codec) / (top_codec - bottom_codec)
        print(f"  {name:<11} escala publicada {published:.3f}  ->  propio techo {corrected:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what the neural codec costs ECAPA.")
    parser.add_argument("stage", choices=("encode", "measure"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    runner = encode if args.stage == "encode" else measure
    return runner(root, out, args.speakers)


if __name__ == "__main__":
    sys.exit(main())
