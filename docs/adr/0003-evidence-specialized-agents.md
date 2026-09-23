# ADR-0003: Agents are specialized by evidence source, not by persona

Status: accepted

## Context

A claim like "revenue grew 8% year over year to $94.0 billion" can be checked against
several independent sources: the XBRL facts, the 10-Q/10-K narrative, a statistical
forecast of what the number should have been, and the market's reaction to the release.
Each source needs different tools, and each fails differently.

## Decision

Four evidence agents, one per source: `xbrl_table` (typed fact queries), `filing_rag`
(hybrid retrieval over filings), `forecast` (baseline ensemble vs. seasonal naive), and
`market_reaction` (event-window abnormal return). Each returns an `EvidenceItem` with a
stance and citations and never a verdict. A separate critic argues the other side and an
adjudicator alone assigns the label.

## Rejected alternative: persona agents ("skeptical analyst", "bullish analyst", "auditor")
debating over the same retrieved context

Cost of that alternative:
- Personas share one context, so their "independent" opinions are correlated: they all
  see the same retrieved chunks and fail on the same retrieval miss.
- A persona's output cannot be evaluated against ground truth per source; "the auditor was
  wrong" is not actionable, "the XBRL agent picked the 6-month value for a quarterly claim"
  is.
- Ablations become meaningless. With source agents, removing one agent measures what that
  source contributes to L2 precision (§11 L2 ablations).
- Debate loops multiply token cost without adding information (§14 budget).

## Consequences

- Each agent has its own tools, prompt and failure mode, and can be tested and ablated
  alone.
- The router decides which agents apply from claim fields, so a qualitative claim never
  pays for a forecast.
- The critic is the only adversarial component, and it is structurally barred from
  emitting a label (`CriticOutput` has no label field).
