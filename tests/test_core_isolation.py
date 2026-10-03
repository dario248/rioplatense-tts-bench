"""The test that keeps SPEC section 3 from eroding.

The core and the runner must be importable in a process where torch, numpy and every TTS
library are unavailable. If someone adds a top-level `import torch` to an adapter module, or
lets a heavy dependency leak into `rtb.schema`, this fails — which is the whole point: stage 2
must never import a TTS library, not even indirectly.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BANNED = [
    "torch",
    "numpy",
    "TTS",
    "transformers",
    "librosa",
    "soundfile",
    "speechbrain",
    "whisper",
    "pandas",
    "qwen_tts",
    "chatterbox",
]

MUST_IMPORT = [
    "rtb.schema",
    "rtb.manifest",
    "rtb.config",
    "rtb.logconf",
    "rtb.generate.base",
    "rtb.generate.dummy",
    "rtb.generate.seeding",
    "rtb.generate.runner",
    # El adapter de XTTS entra en esta lista a proposito: es la prueba de que sus imports pesados
    # estan dentro de load() y no a nivel de modulo. Si alguien mueve `import torch` arriba, esto
    # falla, y con eso se rompe todo el diseño de la seccion 3.
    "rtb.generate.xtts",
    "rtb.generate.qwen3",
    "rtb.generate.chatterbox",
    "rtb.generate.audio_io",
    # Y la etapa 2 al reves: estas metricas corren sobre el manifest y no deben arrastrar NADA
    # pesado, ni siquiera numpy. Es lo que las hace corribles en CI.
    "rtb.metrics",
    "rtb.metrics.latency",
    "rtb.metrics.duration_screen",
    "rtb.metrics.speaking_rate",
    "rtb.metrics.ground_truth",
    "rtb.metrics.audio_shape",
    "rtb.metrics.__main__",
    "rtb.report.tables",
    # El pase de ASR sigue la misma regla que los adapters: faster_whisper se importa dentro de
    # load(), asi que estos modulos son importables donde no esta instalado.
    "rtb.metrics.asr",
    "rtb.metrics.transcribe",
    # Y la similitud igual, aunque su entorno traiga torch: el modulo se importa para REPORTAR la
    # escala desde el JSONL, que es lo que permite regenerar las tablas sin GPU ni speechbrain.
    "rtb.metrics.speaker_sim",
    "rtb.metrics.audio_utils",
]

PROBE = textwrap.dedent(
    """
    import importlib
    import sys

    BANNED = {banned!r}

    class Blocker:
        def find_spec(self, fullname, path=None, target=None):
            root = fullname.split(".")[0]
            if root in BANNED:
                raise AssertionError(
                    f"{{fullname}} was imported at module level; heavy imports belong "
                    "inside load()"
                )
            return None

    # Bloquea tambien lo que ya estuviera cargado en el intérprete padre.
    for name in list(sys.modules):
        if name.split(".")[0] in BANNED:
            del sys.modules[name]

    sys.meta_path.insert(0, Blocker())

    for module in {modules!r}:
        importlib.import_module(module)

    print("ok")
    """
)


def test_core_imports_without_heavy_dependencies() -> None:
    code = PROBE.format(banned=BANNED, modules=MUST_IMPORT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_seeding_reports_only_the_sources_it_actually_seeded() -> None:
    # "Seedeado" con torch ausente no es el mismo experimento: el runner loguea la lista.
    from rtb.generate.seeding import seed_everything

    seeded = seed_everything(1234)
    assert "python" in seeded
