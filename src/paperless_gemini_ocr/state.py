"""Carries what happened in ``parse()`` over to the moment the document exists.

A parser only returns text and files; it never sees the document it produces.
Paperless sends ``document_consumption_started`` before parsing and
``document_consumption_finished`` once the document is stored, both in the
thread that also calls ``parse()``. The outcome waits here in between.

The same hand-over tells a consumption apart from "reprocess", which calls
``parse()`` without either signal.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from dataclasses import field

_local = threading.local()


@dataclass
class Report:
    """Pages the model could not read, for the tag and note on the document."""

    total_pages: int = 0
    failures: dict[int, str] = field(default_factory=dict)
    # Set when the whole document went to local OCR without trying the model.
    document_error: str | None = None

    def note(self) -> str:
        if self.document_error:
            return (
                "Gemini OCR could not process this document "
                f"({self.document_error}). The text comes from local OCR and "
                "may be poor. Use 'Reprocess' to try again."
            )
        numbers = sorted(self.failures)
        listed = ", ".join(str(n) for n in numbers[:30])
        if len(numbers) > 30:
            listed += f" and {len(numbers) - 30} more"
        reasons = sorted(set(self.failures.values()))
        return (
            f"Gemini OCR could not read page(s) {listed} of {self.total_pages} "
            f"({'; '.join(reasons[:3])}). Their text comes from local OCR and "
            "may be poor or missing. Use 'Reprocess' to try again."
        )


def consume_started(path: object) -> None:
    _local.consume_path = str(path) if path is not None else None
    _local.report = None


def take_consume_flag(document_path: object) -> bool:
    """True when this ``parse()`` belongs to the consumption that was just announced."""
    path = getattr(_local, "consume_path", None)
    _local.consume_path = None
    _local.report = None
    return path is not None and path == str(document_path)


def set_report(report: Report) -> None:
    _local.report = report


def pop_report() -> Report | None:
    report = getattr(_local, "report", None)
    _local.report = None
    return report
