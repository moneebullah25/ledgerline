# ADR-0004: Numbers come from typed fact queries, not model-written SQL or model-written numbers

Status: accepted

## Context

The product's credibility rests on one property: every number a reader sees traces to a
filed XBRL fact or a filing span (NFR-8, FR-45). XBRL data has traps a language model
handles badly: the same concept under different tags per company (R8), quarterly vs.
year-to-date durations under the same fiscal period, restated prior-period values, and
Q4 never being filed directly.

## Decision

The model never produces a number or SQL. It (or deterministic code) produces a typed
request `(cik, concept, fiscal_year, fiscal_period)`; `store.xbrl_query.get_xbrl_fact`
resolves the concept through the tag-alias map, picks the right duration, derives Q4 and
ratios in code, and returns a `Fact` with the fact IDs it used, or `None`. `None` becomes
`unverifiable`, never a nearby number. Verdict text is checked by
`schemas.uncited_numbers`: any number in the rationale that is not in a citation fails.

## Rejected alternative: text-to-SQL over `xbrl_fact`

Cost of that alternative:
- The model would have to know, per query, which of four revenue tags this company used,
  that `duration_days ~ 90` separates a quarter from a nine-month figure, and how to
  derive Q4. Each is a silent wrong answer when missed, not an error.
- Generated SQL is an injection and resource risk (arbitrary queries against the
  database) that needs its own sandboxing.
- A wrong-but-plausible number from a wrong query carries a real fact ID, so it passes
  citation checks while being wrong. The typed layer makes the period and tag logic
  reviewed code with unit tests instead.

## Consequences

- Only questions the typed layer supports can be asked; a new metric means a new entry in
  `concepts.py`, not a new prompt.
- Anti-hallucination is enforced by construction plus one validator, and is testable
  offline.
- Coverage is bounded by the concept map: companies that report under an unmapped tag get
  `unverifiable` until the map is extended. That is the intended failure direction.
