"""Graph state. Evidence and usages use list-append reducers because the evidence agents
run as parallel branches and each returns its own one-item update."""

import operator
from dataclasses import dataclass
from datetime import date
from threading import Lock
from typing import Annotated, TypedDict

from ledgerline.llm import Usage
from ledgerline.models import Claim
from ledgerline.schemas import AgentName, CriticOutput, EvidenceItem, VerdictDraft


@dataclass(frozen=True)
class Ctx:
    """Per-claim facts looked up once before routing, shared read-only by every node."""

    company_id: int
    cik: str
    ticker: str
    event_date: date | None  # release filed date or call date
    reported_through: tuple[int, str] | None  # latest (fiscal_year, fiscal_period) in a 10-Q/10-K
    # The agents run in threads but share the caller's Session: every DB read happens under
    # this lock (a Session must never be used by two threads at once), model work outside it.
    db_lock: Lock


class GraphState(TypedDict, total=False):
    claim: Claim
    ctx: Ctx
    agents: list[AgentName]
    evidence: Annotated[list[EvidenceItem], operator.add]
    usages: Annotated[list[tuple[str, Usage]], operator.add]
    critic: CriticOutput | None
    verdict: VerdictDraft
    refused: bool
    model: str | None
