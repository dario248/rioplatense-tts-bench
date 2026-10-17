"""The committed manifests have to agree with the committed environment locks.

A frozen `envs/<model>.lock.txt` is a claim: *these numbers came from this stack*. Nothing stops
that claim from rotting — someone upgrades a package, reruns half the grid, and the lock now
describes an environment that produced none of the published rows. These tests turn the claim into
something CI checks, using only files that are in git: the manifests and the locks.

They skip cleanly for any model that has not run yet, so they stay green through v0's staggered
generation (SPEC section 3: stages are asynchronous in time).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rtb.config import ModelConfig, load_model_config
from rtb.manifest import index_by_generation_id, read_rows
from rtb.schema import GROUND_TRUTH_MODEL, GenerationRow

MANIFEST_DIR = Path("data/manifests")
MODEL_CONFIG_DIR = Path("configs/models")
ENV_DIR = Path("envs")

# El dummy es un doble de prueba: no tiene entorno propio ni pretende reproducibilidad. Y el
# ground truth es audio real: no tiene stack ni checkpoint que declarar, asi que exigirle un
# commit hash lo haria fallar por existir. Se filtra por el nombre del ARCHIVO, que es
# `ground_truth.jsonl` y no el nombre del modelo.
NOT_A_REAL_MODEL = {"dummy", "ground_truth", GROUND_TRUTH_MODEL}

_PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)==(.+)$")

# Un wheel local, que es como entra el fork de Sintetica: su nombre de archivo es lo unico que
# declara nombre y version, porque ese pin no existe en ningun indice.
_WHEEL_RE = re.compile(r"([A-Za-z0-9._]+)-([^-]+)-[^-]+-[^-]+-[^-]+\.whl$")


def committed_manifests(repo_root: Path) -> list[Path]:
    return sorted(
        path
        for path in (repo_root / MANIFEST_DIR).glob("*.jsonl")
        if path.stem not in NOT_A_REAL_MODEL
    )


def parse_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        match = _PIN_RE.match(line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
            continue
        wheel = _WHEEL_RE.search(line)
        if wheel:
            pins[wheel.group(1).replace("_", "-").lower()] = wheel.group(2)
    return pins


def same_build(recorded: str | None, pinned: str | None) -> bool:
    """El lock pinea la version de distribucion; la fila guarda `torch.__version__`.

    Los wheels del indice de PyTorch llevan el sufijo de CUDA en la version misma y los dos lados
    coinciden. Los de PyPI no: torch 1.13.1 se instala como 1.13.1 y en runtime se presenta como
    1.13.1+cu117. Aceptar ese sufijo de mas no afloja el chequeo: otra version sigue fallando.
    """
    if recorded is None or pinned is None:
        return False
    return recorded == pinned or recorded.startswith(pinned + "+")


def manifests(repo_root: Path) -> list[tuple[str, list[GenerationRow]]]:
    """Cada manifest con sus filas EFECTIVAS: la ultima de cada id, que es lo que leen las metricas.

    El formato es append-only, asi que un reintento o una corrida repetida dejan la fila vieja al
    lado de la nueva. Mirar todas las filas confundiria esa historia con media grilla corrida sobre
    otro stack, que es lo que estos tests existen para atrapar.

    El warm-up queda afuera por lo mismo que en el resto del proyecto: se genera y se descarta, y
    su id no se reusa entre corridas, asi que arrastraria el stack de la primera para siempre.
    """
    return [
        (
            path.stem,
            [
                row
                for row in index_by_generation_id(read_rows(path)).values()
                if not row.warmup
            ],
        )
        for path in committed_manifests(repo_root)
    ]


def models_that_already_ran(repo_root: Path) -> list[tuple[ModelConfig, list[GenerationRow]]]:
    """Every model config that has a manifest next to it, paired with its rows."""
    paired = []
    for config_path in sorted((repo_root / MODEL_CONFIG_DIR).glob("*.yaml")):
        model = load_model_config(config_path)
        manifest = repo_root / MANIFEST_DIR / f"{model.name}.jsonl"
        if manifest.exists():
            paired.append((model, read_rows(manifest)))
    return paired


def test_every_row_of_a_model_shares_one_stack(repo_root: Path) -> None:
    """Timings from two different stacks in one table are not comparable."""
    found = manifests(repo_root)
    if not found:
        pytest.skip("no model manifests committed yet")
    for model, rows in found:
        stacks = {(row.lib_version, row.torch_version, row.device, row.gpu) for row in rows}
        assert len(stacks) == 1, f"{model} was generated across several stacks: {stacks}"


def test_every_row_of_a_model_shares_one_checkpoint(repo_root: Path) -> None:
    """Published weights change without notice; half a grid on new weights is not a benchmark."""
    found = manifests(repo_root)
    if not found:
        pytest.skip("no model manifests committed yet")
    for model, rows in found:
        revisions = {row.model_revision for row in rows}
        assert len(revisions) == 1, f"{model} mixes checkpoints: {revisions}"


# Un commit de HuggingFace, o —para un modelo local sin repo que citar— el sha256 del contenido
# de cada archivo de pesos, como `nombre=<hash>` separados por coma (SPEC 5.3, 6.5).
COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
CONTENT_HASHES = re.compile(r"[A-Za-z0-9_]+=[0-9a-f]{12,64}(,[A-Za-z0-9_]+=[0-9a-f]{12,64})*")


def test_the_recorded_revision_pins_the_weights(repo_root: Path) -> None:
    """A repo name is not a revision: it does not pin anything (SPEC 5.3)."""
    found = manifests(repo_root)
    if not found:
        pytest.skip("no model manifests committed yet")
    for model, rows in found:
        revision = rows[0].model_revision or ""
        assert COMMIT_HASH.fullmatch(revision) or CONTENT_HASHES.fullmatch(revision), (
            f"{model}: model_revision={revision!r} is neither a commit hash nor content hashes"
        )


def test_a_model_that_already_ran_has_its_revision_pinned(repo_root: Path) -> None:
    """`model_revision: null` is fine until the first run, and a liability afterwards.

    Before running there is nothing to pin: the adapter resolves the hash and records it. After
    running, leaving it null means the next run could download different weights without a single
    line of the config changing, and half the grid would be a different model.

    The requirement applies to models whose weights come from HuggingFace. A model with no
    `hf_repo` — the dummy, or Sintética behind HTTP — has no commit hash to pin.
    """
    checked = 0
    for model, _rows in models_that_already_ran(repo_root):
        if not model.hf_repo:
            continue
        assert model.model_revision and re.fullmatch(r"[0-9a-f]{40}", model.model_revision), (
            f"{model.name}: model_revision={model.model_revision!r} is not a commit hash. A repo "
            "name, a branch or a short hash can all point at different weights tomorrow"
        )
        checked += 1
    if not checked:
        pytest.skip("no HuggingFace-backed model has run yet")


def test_the_pinned_revision_is_the_one_that_actually_ran(repo_root: Path) -> None:
    """The hash in the config has to be the hash in the rows.

    Pinning is a copy-paste, and a copy-paste can be wrong. If it is, the config declares one set
    of weights while the published numbers came from another — a claim that is trivially checkable
    and, without this, never checked.
    """
    checked = 0
    for model, rows in models_that_already_ran(repo_root):
        if not model.model_revision:
            continue
        recorded = {row.model_revision for row in rows}
        assert recorded == {model.model_revision}, (
            f"{model.name}: the config pins {model.model_revision} but the manifest was generated "
            f"with {sorted(recorded)}"
        )
        checked += 1
    if not checked:
        pytest.skip("no pinned model has run yet")


def test_manifest_versions_match_the_frozen_lock(repo_root: Path) -> None:
    """What the rows say they ran with has to be what the lock file pins."""
    found = manifests(repo_root)
    if not found:
        pytest.skip("no model manifests committed yet")

    checked = 0
    for model, rows in found:
        lock_path = repo_root / ENV_DIR / f"{model}.lock.txt"
        if not lock_path.exists():
            continue
        pins = parse_lock(lock_path)
        row = rows[0]

        # lib_version viene como "coqui-tts==0.27.5": el mismo formato que un pin.
        assert row.lib_version and "==" in row.lib_version, model
        package, _, declared = row.lib_version.partition("==")
        pinned = pins.get(package.lower())
        assert pinned == declared, (
            f"{model}: rows ran {package}=={declared} but {lock_path.name} pins {pinned}"
        )

        assert same_build(row.torch_version, pins.get("torch")), (
            f"{model}: rows ran torch {row.torch_version} but {lock_path.name} pins "
            f"{pins.get('torch')}"
        )
        checked += 1

    if not checked:
        pytest.skip("no frozen locks for the models that have run")


def test_a_frozen_lock_carries_the_index_for_its_local_wheels(repo_root: Path) -> None:
    """A `+cu126` pin cannot be installed from PyPI, so the lock must name the index itself."""
    for lock_path in sorted((repo_root / ENV_DIR).glob("*.lock.txt")):
        text = lock_path.read_text(encoding="utf-8")
        if "+cu" not in text:
            continue
        assert "--extra-index-url" in text or "--index-url" in text, (
            f"{lock_path.name} pins wheels with a local version but names no index"
        )
