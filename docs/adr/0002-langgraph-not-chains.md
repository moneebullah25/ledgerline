# ADR-0002: The verification flow is a LangGraph state graph, not a chain

Status: accepted

## Context

Verifying one claim is not a straight line. A deterministic router picks which evidence
agents apply to the claim; those agents run concurrently; if every agent comes back
`insufficient` the flow must stop and return `unverifiable` without calling the
adjudicator (FR-44); otherwise a critic argues the opposite case and an adjudicator
decides. Every step must be traced, and the router must be testable without a model.

## Decision

Build the flow as a LangGraph `StateGraph`: typed state, one node per step, a conditional
edge for the router fan-out, and a conditional edge for the refusal short-circuit. The
router is a plain function (`graph.route`) that the graph calls, so it is unit-tested with
no LLM and no network.

## Rejected alternative: a linear chain (LangChain LCEL or hand-written sequential calls)

Cost of that alternative:
- Branching (which agents run) and early exit (refusal) end up as `if` statements inside
  prompt-calling code, where they are invisible in traces and hard to test in isolation.
- Concurrency across agents has to be bolted on with ad-hoc threads or asyncio around a
  sequential abstraction.
- The refusal path, one of the never-cut features (§19), becomes a convention rather than
  a structural edge that tests can assert on ("adjudicator node was not visited").

## Consequences

- The graph shape is the spec: nodes map to §10.5 boxes, edges to its arrows.
- Each node writes a trace span, so a bad verdict can be traced to the node that
  produced the bad input.
- Adds a dependency with a fast-moving API; the graph code is kept thin (nodes are plain
  functions over the §22 schemas) so a replacement would be a rewrite of the wiring only.
