"""What one run cost the machine: peak memory and the thread count that produced it.

Por corrida y no por fila (SPEC 7.3). Las dos memorias crecen monótonas y las dominan los pesos
del modelo, así que un campo por fila repetiria el mismo numero 540 veces y sugeriria que varia.

Importa torch lazy, igual que `seeding.py`: sin el, el RSS se mide igual y las columnas de GPU
quedan en None.
"""

from __future__ import annotations

import json
import os
import resource
from pathlib import Path
from typing import Any


def snapshot(threads: int | None = None) -> dict[str, Any]:
    """Peak memory so far, from the OS and from the CUDA allocator when there is one."""
    usage: dict[str, Any] = {
        # ru_maxrss es el pico de toda la vida del proceso, en KiB en Linux.
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "threads": threads,
        # Lo pedido por entorno, al lado de lo que torch termino usando: si difieren, el numero
        # de la tabla no es el que se creia estar midiendo.
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "peak_vram_reserved_mib": None,
        "peak_vram_allocated_mib": None,
        "gpu": None,
    }

    try:
        import torch
    except ImportError:
        return usage

    if usage["threads"] is None:
        usage["threads"] = torch.get_num_threads()
    if torch.cuda.is_available():
        # El RESERVADO y no el asignado: es el que decide si el modelo entra en la placa, y es el
        # que mostro la deriva de Chatterbox (ANALYSIS A14).
        usage["peak_vram_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        usage["peak_vram_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        usage["gpu"] = torch.cuda.get_device_name(0)
    return usage


def resources_path(manifest_path: Path) -> Path:
    """Sibling of the manifest: el costo de la corrida viaja con lo que la corrida produjo."""
    return manifest_path.with_name(f"{manifest_path.stem}_resources.json")


def write(manifest_path: Path, device: str, **extra: Any) -> Path:
    """Write the run's resource cost next to its manifest, replacing any earlier one.

    Se sobreescribe en vez de acumular: el pico de la ultima corrida es el que corresponde al
    manifest que quedo en disco.
    """
    path = resources_path(manifest_path)
    payload = {"device": device, **snapshot(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
