# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""What a model's own signal chain costs the speaker-similarity metric, measured on real audio.

    .venv-pocket/bin/python    scripts/codec_ceiling.py encode --model pocket
    .venv-sintetica/bin/python scripts/codec_ceiling.py encode --model sintetica
    .venv-simil/bin/python     scripts/codec_ceiling.py measure --out <dir> --row Name=0.559

Two stages because they need different environments, with wavs on disk between them, like every
other pair of stages here. `encode` passes the control clips through the model's decoder alone —
Pocket's Mimi codec, or Sintética's HiFiGAN from the mel of the real audio — with no synthesis and
no text. `measure` then recomputes the ceiling and the floor with one side of each pair carrying
that chain, which is the geometry of the published metric: the reference clip is real and never
goes through it, the synthesis always does.

The clean side is the clip as the chain received it, resampled but not decoded, so what the second
column loses is the chain and nothing else.
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
CLEAN, CODEC = "limpio", "por el modelo"
# Pesos de Sintética: no están en el repo y el adapter los toma de la misma variable.
SINTETICA_ROOT = "SINTETICA_TTS_ROOT"


def control_clips(root: Path, speakers_path: Path) -> list[tuple[str, str, Path]]:
    """(hablante, nombre del clip, ruta) de cada clip de control."""
    return [
        (speaker.id, Path(clip.path).stem, root / clip.path)
        for speaker in load_speakers(root / speakers_path, root=root)
        for clip in speaker.control_clips
    ]


def encode_pocket(root: Path, out: Path, speakers_path: Path) -> int:
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


def encode_sintetica(root: Path, out: Path, speakers_path: Path) -> int:
    """Copy-synthesis through HiFiGAN alone. Needs the Sintética environment and its weights.

    The mel comes from the vocoder's own `AudioProcessor`, so extraction and consumption share one
    config. Both audio configs are identical and `signal_norm` is off, so `normalize` is a no-op
    and nothing is scaled twice. A checkpoint per gender, as the adapter builds it.
    """
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    from TTS.api import TTS  # noqa: PLC0415

    from rtb.generate.audio_io import write_wav  # noqa: PLC0415

    model = yaml.safe_load((root / "configs/models/sintetica.yaml").read_text(encoding="utf-8"))
    params = model["params"]
    weights = Path(os.path.expandvars(str(params["weights_root"])))
    if "$" in str(weights):
        raise SystemExit(f"{SINTETICA_ROOT} sin definir: los pesos no están en el repo")
    voices = params["voices"]

    def patched(path: Path, speakers: Path | None, stats: Path | None) -> str:
        """Los configs entrenados traen paths del layout original; se reapuntan acá."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if speakers is not None:
            data["speakers_file"] = str(speakers.resolve())
            if isinstance(data.get("model_args"), dict):
                data["model_args"]["speakers_file"] = str(speakers.resolve())
        if isinstance(data.get("audio"), dict) and data["audio"].get("stats_path"):
            data["audio"]["stats_path"] = str(stats.resolve()) if stats else None
        handle, tmp = tempfile.mkstemp(suffix=".json", prefix="rtb_a18_")
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return tmp

    clips: dict[str, list[tuple[str, Path]]] = {}
    for speaker, name, path in control_clips(root, speakers_path):
        clips.setdefault(speaker, []).append((name, path))
    out.mkdir(parents=True, exist_ok=True)

    for checkpoint in params["checkpoints"]:
        mine = [s for s, voice in voices.items() if voice in checkpoint["voices"]]
        if not mine:
            continue
        stats = weights / checkpoint["stats"]
        synth_config = patched(
            weights / checkpoint["config"], weights / checkpoint["speakers"], stats
        )
        vocoder_config = patched(weights / params["vocoder"]["config"], None, stats)
        try:
            tts: Any = TTS(
                model_path=str(weights / checkpoint["synth"]),
                config_path=synth_config,
                vocoder_path=str(weights / params["vocoder"]["weights"]),
                vocoder_config_path=vocoder_config,
                gpu=False,
            )
        finally:
            for tmp in (synth_config, vocoder_config):
                Path(tmp).unlink(missing_ok=True)
        ap = tts.synthesizer.vocoder_ap
        vocoder = tts.synthesizer.vocoder_model

        for speaker in mine:
            for name, path in clips[speaker]:
                wav = ap.load_wav(str(path), sr=ap.sample_rate)
                mel = ap.melspectrogram(wav)
                with torch.no_grad():
                    restored = vocoder.inference(torch.tensor(ap.normalize(mel)).unsqueeze(0))
                write_wav(
                    out / f"{speaker}__{name}__{CLEAN_FILE}.wav", wav.tolist(), ap.sample_rate
                )
                write_wav(
                    out / f"{speaker}__{name}__{CODEC_FILE}.wav",
                    restored.squeeze().cpu().numpy().tolist(),
                    ap.sample_rate,
                )
                print(f"  {speaker} {name}: mel {mel.shape}")
    return 0


def measure(root: Path, out: Path, speakers_path: Path, rows: list[str]) -> int:
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
    if rows:
        print("\nLa fila del modelo se lee contra el techo de abajo:")
    for row in rows:
        name, _, value = row.partition("=")
        raw = float(value)
        published = (raw - bottom) / (top - bottom)
        corrected = (raw - bottom_codec) / (top_codec - bottom_codec)
        print(f"  {name:<12} escala publicada {published:.3f}  ->  propio techo {corrected:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what the neural codec costs ECAPA.")
    parser.add_argument("stage", choices=("encode", "measure"))
    parser.add_argument("--model", choices=("pocket", "sintetica"), help="required by encode")
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        metavar="NAME=RAW_COSINE",
        help="rescale this published median against the measured ceiling; repeat",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--speakers", type=Path, default=DEFAULT_SPEAKERS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    if args.stage == "measure":
        return measure(root, out, args.speakers, args.row)
    if args.model is None:
        parser.error("encode needs --model")
    encoder = encode_pocket if args.model == "pocket" else encode_sintetica
    return encoder(root, out, args.speakers)


if __name__ == "__main__":
    sys.exit(main())
