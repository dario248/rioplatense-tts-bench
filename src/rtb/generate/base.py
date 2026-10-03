"""The adapter contract every TTS model must satisfy.

An adapter translates between this ABC and one model's API and does nothing else: anything that
could change the experiment lives in the runner or in a config (SPEC 12).
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from rtb.config import ModelConfig


@dataclass(slots=True)
class SynthesisResult:
    """What one synthesis reports back. The audio itself went to disk, not through here.

    The core never touches audio arrays: that is what keeps `rtb.schema`/`rtb.manifest` free of
    numpy and lets the evaluation stage run in a different environment weeks later.
    """

    sample_rate: int | None = None
    duration_s: float | None = None
    ttfb_ms: float | None = None
    total_ms: float | None = None
    # Segundos de referencia realmente consumidos tras recortar a max_ref_seconds. Lo sabe el
    # adapter, no el runner: comparar modelos con referencias distintas invalida la similitud.
    ref_seconds_used: float | None = None
    # Si esta sintesis recibio la TRANSCRIPCION de la referencia ademas del audio: es una
    # condicion experimental distinta y va por fila (SPEC 6.4).
    ref_text_provided: bool | None = None
    # Si esta sintesis corrio en modo streaming. Es lo que hizo, no lo que el modelo soporta: un
    # adapter que puede streamear pero corrio entero tiene que decir False, porque de eso depende
    # como se lee su ttfb_ms. None deja que el runner use `supports_streaming`.
    streaming: bool | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class AdapterError(Exception):
    """Raised when an adapter cannot be resolved or configured."""


class TTSAdapter(ABC):
    """One TTS model, driven by the runner.

    Class attributes are declared defaults; the values actually used come from the model's
    config file, which is the single source of truth for the experiment.
    """

    name: str = "unnamed"
    max_ref_seconds: float = 10.0
    supports_streaming: bool = False
    default_sample_rate: int = 24000
    # Si el modelo necesita la transcripcion de la referencia para clonar. Declarativo y no
    # inferido del adapter, porque de esto depende una validacion que corre ANTES de sintetizar:
    # ver `requires_ref_text` en ModelConfig.
    requires_ref_text: bool = False
    # Si el modelo elige la voz por id en vez de clonarla de un clip. Son dos clases distintas de
    # TTS y el benchmark mide las dos: ver SPEC 6.5.
    selects_voice_by_id: bool = False

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.name = config.name
        self.max_ref_seconds = config.max_ref_seconds
        self.supports_streaming = config.supports_streaming
        self.default_sample_rate = config.default_sample_rate
        self.requires_ref_text = config.requires_ref_text
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory.

        Every heavy import goes HERE, so the module stays importable with no GPU and no TTS stack.
        """

    @abstractmethod
    def synthesize(
        self,
        text: str,
        ref_wav: Path,
        seed: int,
        out_path: Path,
        ref_text: str | None = None,
        speaker_id: str | None = None,
    ) -> SynthesisResult:
        """Synthesize `text` in the voice of `ref_wav` and write a wav to `out_path`.

        Trim the reference from the START of the clip to `max_ref_seconds`, always from the same
        clip, and report the seconds used. Return timings and metadata — never the audio.

        `ref_text` is the transcript of the WHOLE clip. An adapter that uses it must not trim the
        audio, and must set `requires_ref_text` (SPEC 6.4). Breaking either degrades cloning
        without raising anything.

        `speaker_id` is the benchmark's speaker, which a model that selects a baked-in voice needs
        instead of `ref_wav`. Such an adapter sets `selects_voice_by_id` and maps the id itself;
        the ones that clone from audio ignore it (SPEC 6.5).
        """

    def unload(self) -> None:
        """Release the model. Default is a no-op; override when it frees VRAM."""
        self._loaded = False

    @abstractmethod
    def describe(self) -> dict[str, str | None]:
        """Identity of what actually ran: it goes verbatim into every manifest row.

        Must include `model_revision` (a commit hash, not a repo name), `lib_version`,
        `torch_version` and `device` (SPEC 5.3).
        """


def resolve_adapter(spec: str) -> type[TTSAdapter]:
    """Import an adapter class from a `"module.path:ClassName"` string.

    Deferred to call time on purpose: this is the seam that keeps a TTS library out of every
    process that does not need it.
    """
    module_path, _, class_name = spec.partition(":")
    if not module_path or not class_name:
        raise AdapterError(f"adapter spec must be 'module.path:ClassName', got {spec!r}")
    try:
        module = importlib.import_module(module_path)
    except ImportError as err:
        raise AdapterError(f"cannot import adapter module {module_path!r}: {err}") from err
    try:
        candidate = getattr(module, class_name)
    except AttributeError as err:
        raise AdapterError(f"{module_path!r} has no attribute {class_name!r}") from err
    if not (isinstance(candidate, type) and issubclass(candidate, TTSAdapter)):
        raise AdapterError(f"{spec} is not a TTSAdapter subclass")
    return candidate


def build_adapter(config: ModelConfig) -> TTSAdapter:
    return resolve_adapter(config.adapter)(config)
