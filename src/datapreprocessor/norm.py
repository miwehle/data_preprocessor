import re
from pathlib import Path
from typing import Protocol, TextIO

_WHITESPACE_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


class NormReporter(Protocol):
    def note_change(self, before: str, after: str, norm_changes: list[str]) -> None: ...


class NormReport:
    def __init__(self, out: TextIO, *, debug: bool = False):
        self.out = out
        self.debug = debug
        self.seq_no = 0

    @classmethod
    def from_path(cls, path: str | Path = "norm_report.txt", *, debug: bool = False) -> "NormReport":
        return cls(open(path, "w", encoding="utf-8"), debug=debug)

    def note_change(self, before: str, after: str, norm_changes: list[str]) -> None:
        self.seq_no += 1
        if not norm_changes:
            return
        record = {
            "seq_no": self.seq_no,
            "norm_changes": norm_changes,
        }
        if self.debug:
            record["before"] = before
            record["after"] = after
        self.out.write(f"{record}\n")

    def flush(self) -> None:
        self.out.flush()

    def close(self) -> None:
        self.out.close()


def norm(s: str, norm_reporter: NormReporter | None = None) -> str:
    """Normalize text by removing control chars and collapsing whitespace."""
    before = str(s)
    after = before
    norm_changes: list[str] = []

    stripped = after.strip()
    if stripped != after:
        norm_changes.append("strip_edges")
    after = stripped

    no_ctrl = _CTRL_RE.sub("", after)
    if no_ctrl != after:
        norm_changes.append("remove_control_chars")
    after = no_ctrl

    collapsed_ws = _WHITESPACE_RE.sub(" ", after)
    if collapsed_ws != after:
        norm_changes.append("collapse_whitespace")
    after = collapsed_ws

    if norm_reporter is not None:
        norm_reporter.note_change(before, after, norm_changes)

    return after
