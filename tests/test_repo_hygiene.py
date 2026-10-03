"""Guards against files that the project needs but git silently drops.

`configs/models/*.yaml` was ignored for a while by a `.gitignore` line reading `models/`, meant
for downloaded weights: without a leading slash git applies a directory pattern at every level,
so it also swallowed `configs/models/`. Nothing failed locally — the files were on disk — but a
fresh clone had no model configs at all.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Todo lo que el benchmark necesita para correr desde un clon limpio.
MUST_NOT_BE_IGNORED = [
    "configs/phrases.yaml",
    "configs/speakers.yaml",
    "configs/grid.yaml",
    "configs/models",
    "src/rtb",
    "scripts",
    "tools",
    "tests",
    "pyproject.toml",
    "README.md",
    "SPEC.md",
]


def tracked_paths(repo_root: Path) -> list[Path]:
    """Every file under the paths above, as git would see them."""
    found: list[Path] = []
    for entry in MUST_NOT_BE_IGNORED:
        target = repo_root / entry
        if target.is_dir():
            found.extend(
                path
                for path in sorted(target.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            )
        elif target.is_file():
            found.append(target)
    return found


def test_nothing_the_project_needs_is_gitignored(repo_root: Path) -> None:
    if shutil.which("git") is None or not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")

    candidates = tracked_paths(repo_root)
    assert candidates, "the path list itself is wrong: nothing was found"

    # check-ignore devuelve las rutas que SI estan ignoradas: la lista tiene que salir vacia.
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *[str(p) for p in candidates]],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, "gitignored but needed by a clean clone:\n" + "\n".join(ignored)


# Lo que el repo produce legitimamente sin trackear en un momento dado son unos pocos fuentes
# nuevos y algun JSONL: el manifest mas grande son 360 KB. Los 93 MB de `audio/` o un checkpoint
# bajado dentro del repo pasan este techo por varios ordenes de magnitud.
UNTRACKED_TOTAL_LIMIT_MB = 20


def test_nothing_big_is_sitting_untracked(repo_root: Path) -> None:
    """The mirror of the test above: what must stay out of git, does.

    Measures the TOTAL rather than the largest file, because the realistic accident is many medium
    files: `audio/` is 93 MB spread over hundreds of wavs of a few hundred KB each, so a per-file
    limit would let it through. Virtualenvs happen to be covered twice — `uv venv` writes a
    `.gitignore` containing `*` inside each one — but generated audio, model checkpoints and stray
    wavs have no such protection.

    Asking about size beats enumerating names: the list of things that must not be committed grows
    every time a stage is added, and nobody remembers to update it.
    """
    if shutil.which("git") is None or not (repo_root / ".git").exists():
        pytest.skip("not a git checkout")

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    sizes: list[tuple[float, Path]] = []
    for line in result.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        path = repo_root / line[3:].strip().strip('"')
        if path.is_file():
            sizes.append((path.stat().st_size / 1_000_000, path.relative_to(repo_root)))

    total = sum(megabytes for megabytes, _ in sizes)
    biggest = "\n".join(
        f"  {megabytes:7.2f} MB  {path}" for megabytes, path in sorted(sizes, reverse=True)[:10]
    )
    assert total <= UNTRACKED_TOTAL_LIMIT_MB, (
        f"{total:.0f} MB untracked, over the {UNTRACKED_TOTAL_LIMIT_MB} MB limit — something that "
        f"should be in .gitignore is one `git add -A` from the history:\n{biggest}"
    )


# Un config de variante —`_cpu`, `_gpu`, `_lat`— es una copia del base con otro `name` y otro
# manifest. La copia es el punto: si un parametro diverge, las dos corridas miden modelos distintos
# y la comparacion entre plataformas miente sin avisar.
VARIANT_SUFFIXES = ("_cpu", "_gpu", "_lat", "_lat_ec")

# Lo unico que una variante puede cambiar a proposito, y por que.
ALLOWED_DIFFERENCES = {
    "name": "es lo que separa los generation_id de las dos corridas",
    "manifest_path": "idem, del lado de los archivos",
    "device": "es la plataforma que la variante existe para medir",
}
# Excepciones declaradas, una por una. La rama de A14 que probo el mitigador tiene que diferir.
ALLOWED_PARAM_DIFFERENCES = {"chatterbox_v3_lat_ec": {"empty_cache_between"}}


def variant_configs(repo_root: Path) -> list[tuple[Path, Path]]:
    """Every `<base>_<suffix>.yaml` next to the `<base>.yaml` it copies."""
    models = repo_root / "configs" / "models"
    pairs = []
    for path in sorted(models.glob("*.yaml")):
        for suffix in VARIANT_SUFFIXES:
            if path.stem.endswith(suffix):
                base = models / f"{path.stem[: -len(suffix)]}.yaml"
                if base.exists():
                    pairs.append((path, base))
                break
    return pairs


def test_variant_configs_are_copies_of_their_base(repo_root: Path) -> None:
    yaml = pytest.importorskip("yaml")
    pairs = variant_configs(repo_root)
    assert pairs, "no variant configs found: the suffix list is wrong"

    for variant_path, base_path in pairs:
        variant = yaml.safe_load(variant_path.read_text())
        base = yaml.safe_load(base_path.read_text())
        allowed = ALLOWED_PARAM_DIFFERENCES.get(variant["name"], set())

        for field, value in base.items():
            if field in ALLOWED_DIFFERENCES or field == "params":
                continue
            assert variant.get(field) == value, (
                f"{variant_path.name} has {field}={variant.get(field)!r} where "
                f"{base_path.name} has {value!r}: the variant stopped being the same model"
            )

        keys = set(base["params"]) | set(variant["params"])
        for key in sorted(keys - allowed):
            assert variant["params"].get(key) == base["params"].get(key), (
                f"{variant_path.name} differs from {base_path.name} in params.{key} — either "
                "copy it back or declare it in ALLOWED_PARAM_DIFFERENCES with its reason"
            )
