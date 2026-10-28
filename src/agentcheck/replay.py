"""Tool record/replay: hold the tool side of an agent fixed between runs.

In `record` mode every tool call goes to the real tool and its result (or ToolError) is
stored in a JSON cassette under the tool name and its arguments. In `replay` mode the
cassette answers and the real tool is never called; a call that was never recorded is an
infrastructure error, not something to guess. The first recorded result for a key wins,
so the cassette is a snapshot: baseline and candidate see identical tool outputs.
"""

import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from inspect_ai.tool import Tool, ToolDef, ToolError

CASSETTE_VERSION = 1
Scalar = str | int | float | bool


class ToolMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class CassetteMissError(RuntimeError):
    """Replay asked for a tool call that was never recorded."""


def call_key(tool_name: str, arguments: dict[str, Any]) -> str:
    return f"{tool_name}({json.dumps(arguments, sort_keys=True, separators=(',', ':'))})"


class Cassette:
    def __init__(self, path: Path, entries: dict[str, dict[str, Scalar]] | None = None) -> None:
        self.path = path
        self.entries: dict[str, dict[str, Scalar]] = entries or {}
        self.live_calls = 0
        self.replayed_calls = 0

    @classmethod
    def load(cls, path: Path) -> "Cassette":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("version") != CASSETTE_VERSION:
            raise ValueError(f"{path}: unsupported cassette version {data.get('version')!r}")
        return cls(path, data["entries"])

    @classmethod
    def open(cls, path: Path, mode: ToolMode) -> "Cassette":
        if mode is ToolMode.REPLAY:
            return cls.load(path)  # a missing cassette in replay mode must fail loudly
        return cls.load(path) if path.exists() else cls(path)

    def save(self) -> None:
        """Atomic write: an interrupted run never leaves a half-written cassette."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": CASSETTE_VERSION, "entries": self.entries}, indent=1, sort_keys=True
        )
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, self.path)


def recorded(tool: Tool, cassette: Cassette, mode: ToolMode) -> Tool:
    """Wrap an Inspect tool so it records to, or replays from, `cassette`."""
    if mode is ToolMode.LIVE:
        return tool
    tdef = ToolDef(tool)
    live = cast(Callable[..., Awaitable[Any]], tool)

    async def execute(**kwargs: Any) -> Scalar:
        key = call_key(tdef.name, kwargs)
        if mode is ToolMode.REPLAY:
            if key not in cassette.entries:
                raise CassetteMissError(f"no recording for {key} in {cassette.path}")
            entry = cassette.entries[key]
            cassette.replayed_calls += 1
        elif key in cassette.entries:
            entry = cassette.entries[key]
        else:
            cassette.live_calls += 1
            try:
                result = await live(**kwargs)
            except ToolError as ex:
                entry = {"error": ex.message}
            else:
                if not isinstance(result, str | int | float | bool):
                    raise TypeError(f"{tdef.name}: only scalar results can be recorded")
                entry = {"result": result}
            cassette.entries[key] = entry
            cassette.save()
        if "error" in entry:
            raise ToolError(str(entry["error"]))
        return entry["result"]

    return ToolDef(
        execute, name=tdef.name, description=tdef.description, parameters=tdef.parameters
    ).as_tool()
