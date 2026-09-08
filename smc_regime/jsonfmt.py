"""JSON serialisation for the dashboard payloads: pretty at the top, one
line per record underneath.

`json.dumps(..., indent=2)` turns the setup-grade payload into ~27k lines
for 415 tickers, because every scalar inside every component gets its own
line. That is 845KB of mostly punctuation, and it makes the published
artifact expensive to read back -- reviewing or merging one costs tens of
thousands of lines of diff for what is really 415 records.

Going the other way, `separators=(',', ':')` with no indent at all, is
worse: the whole payload becomes a single 450KB line, which no line-oriented
tool (diff, grep, a chunked read) can navigate at all. Halving the bytes is
not worth losing the ability to look at it.

So: indent the structure down to `compact_depth`, and emit everything at or
below that depth on one line. The result keeps the shape of the document
browsable while collapsing each leaf record -- a ticker's whole scorecard,
or one strategy's stats -- to a single greppable line.
"""
from __future__ import annotations

import json
from typing import Any

_COMPACT_SEPARATORS = (",", ":")


def dumps(obj: Any, compact_depth: int, indent: int = 2) -> str:
    """Serialise `obj`, indenting containers shallower than `compact_depth`
    and collapsing everything from that depth down onto one line.

    Depth counts containers from the root: the root object is depth 0, its
    values are depth 1, and so on. `compact_depth=2` on {"tickers": [ {...} ]}
    therefore puts each ticker dict on its own line.
    """
    return _render(obj, compact_depth, indent, level=0)


def _render(obj: Any, compact_depth: int, indent: int, level: int) -> str:
    if level >= compact_depth or not isinstance(obj, (dict, list)) or not obj:
        # Leaf, or deep enough to collapse. Empty containers render the same
        # either way, so short-circuit them here rather than emitting "[\n]".
        return json.dumps(obj, separators=_COMPACT_SEPARATORS)

    pad = " " * (indent * (level + 1))
    close_pad = " " * (indent * level)

    if isinstance(obj, dict):
        items = [
            f"{pad}{json.dumps(k)}: {_render(v, compact_depth, indent, level + 1)}"
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + close_pad + "}"

    items = [f"{pad}{_render(v, compact_depth, indent, level + 1)}" for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + close_pad + "]"
