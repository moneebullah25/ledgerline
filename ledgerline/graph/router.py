"""FR-40: deterministic dispatch. No model call; a pure function of populated claim fields.

route() takes two facts that live outside the claim row as keyword arguments so it stays
pure: the claim's event date (release filed date / call date) and the latest fiscal period
the company has reported in a 10-Q/10-K. verify_claim looks both up before routing.
"""

from datetime import date

from ledgerline.concepts import CONCEPTS
from ledgerline.models import Claim
from ledgerline.schemas import AgentName

_Q = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}  # FY is reported with Q4, in the 10-K


def period_key(fiscal_year: int, fiscal_period: str) -> tuple[int, int]:
    return fiscal_year, _Q[fiscal_period]


def route(
    claim: Claim,
    *,
    event_date: date | None = None,
    reported_through: tuple[int, str] | None = None,
) -> list[AgentName]:
    numeric = claim.value is not None and claim.metric in CONCEPTS
    agents: list[AgentName] = []
    if claim.claim_type in ("reported_result", "comparison") and numeric:
        agents.append("xbrl_table")
    if claim.claim_type == "guidance" and numeric:
        agents.append("forecast")
        target = (claim.target_fiscal_year, claim.target_fiscal_period)
        if (
            reported_through
            and None not in target
            and period_key(*target) <= period_key(*reported_through)
        ):
            agents.append("xbrl_table")
    agents.append("filing_rag")
    if event_date is not None:
        agents.append("market_reaction")
    return agents
