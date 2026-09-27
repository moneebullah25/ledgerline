"""Minimal tracer: spans are buffered in memory and written with the report's transaction,
so a verdict's trace_id always resolves to the spans that produced it (§12)."""

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ledgerline.llm import LLMUnavailable
from ledgerline.models import Span

# Exception types whose message is written for the reader (a missing quarter, an unsupported
# request, an unreachable model: llm keeps model output and CLI stderr out of its messages).
# Anything else, e.g. a DB error quoting SQL or an integrity failure quoting rejected model output,
# is summarised by class name only; the traceback goes to the log.
_READABLE = (LookupError, ValueError, LLMUnavailable)


def error_summary(e: BaseException, limit: int = 300) -> str:
    """Class and a bounded message: what an API reader may see."""
    msg = " ".join(str(e).split()) if isinstance(e, _READABLE) else ""
    return f"{type(e).__name__}: {msg[:limit]}" if msg else type(e).__name__


class Tracer:
    def __init__(self, trace_id: uuid.UUID | None = None):
        self.trace_id = trace_id or uuid.uuid4()
        self.spans: list[Span] = []
        self._stack: list[uuid.UUID] = []

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[dict]:
        """Yields the attrs dict so the body can add results (tokens, cost, stance...)."""
        sid = uuid.uuid4()
        parent = self._stack[-1] if self._stack else None
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        self._stack.append(sid)
        try:
            yield attrs
        except Exception as e:
            attrs["error"] = error_summary(e)
            raise
        finally:
            self._stack.pop()
            attrs["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            self.spans.append(
                Span(
                    id=sid,
                    trace_id=self.trace_id,
                    parent_id=parent,
                    name=name,
                    started_at=started,
                    ended_at=datetime.now(UTC),
                    attrs=attrs,
                )
            )

    def flush(self, s: Session) -> None:
        s.add_all(self.spans)
        self.spans = []
