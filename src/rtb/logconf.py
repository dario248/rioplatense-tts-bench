"""Structured logging. No `print` anywhere in this project (SPEC.md section 12).

Log records go to stderr as one JSON object per line, so a long generation run can be grepped,
counted, and diffed afterwards. Extra fields travel in `extra={"fields": {...}}`; use the
`event()` helper rather than formatting values into the message.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per record, with `fields` merged into the top level."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        # Campos pasados como extra={...} suelto, sin envolver en "fields".
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "fields":
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str | int = "INFO", *, json_lines: bool = True) -> None:
    """Configure the root logger once, replacing any previous handler.

    El nivel se fija ademas en el arbol `rtb`, y no es redundante: importar una libreria de TTS
    puede reconfigurar el logger raiz —coqui lo sube a WARNING— y con el nivel solo en la raiz eso
    apaga en silencio todo lo que este proyecto loguea DESPUES de cargar el modelo, que es
    justamente el costo de conditioning que SPEC 6.3 pide registrar aparte.
    """
    handler = logging.StreamHandler(sys.stderr)
    if json_lines:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    resolved = level if isinstance(level, int) else level.upper()
    root.setLevel(resolved)
    logging.getLogger("rtb").setLevel(resolved)


def event(logger: logging.Logger, message: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log a structured event: a stable message plus machine-readable fields."""
    logger.log(level, message, extra={"fields": fields})
