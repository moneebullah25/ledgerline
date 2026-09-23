# ADR-0001: One Postgres for facts, text search, vectors, traces and the job queue

Status: accepted

## Context

Ledgerline stores four kinds of data that are usually split across systems: relational
facts (companies, filings, XBRL values), full-text search over filings, dense vectors for
semantic retrieval, and operational state (reports, verdicts, trace spans, the report job
queue). The corpus is small: 15 companies x 6 quarters is on the order of a few hundred
filings and well under a million chunks. The one query that matters most joins all of
them: "chunks of this company's filings for this period, ranked by keyword and vector
similarity, next to the XBRL facts for the same period".

## Decision

Postgres 16 with pgvector holds everything: `tsvector` + GIN for the keyword leg, a
pgvector HNSW index for the dense leg, JSONB for agent payloads and citations, and the
`report` table doubles as the job queue (`SELECT ... FOR UPDATE SKIP LOCKED`).

## Rejected alternative: Postgres + a dedicated vector DB + a search engine + a queue

For example Postgres for facts, Qdrant/Pinecone for vectors, Elasticsearch/OpenSearch for
BM25, Redis/SQS for jobs.

Cost of that alternative:
- Every hybrid query becomes a cross-system fan-out with its own filtering semantics;
  "only this company's 10-Q for FY2025 Q3" must be kept consistent in three places.
- No transaction spans a chunk row and its embedding, so a crash mid-index leaves
  searchable text with no vector (or the reverse), and idempotent re-runs need a
  reconciliation job.
- Four services to run, back up, secure and pay for, for a corpus that fits in RAM.
- A verdict's evidence, citations and trace could not be written atomically with the
  verdict itself.

## Consequences

- Hybrid retrieval, fact lookup and provenance are single SQL statements with foreign
  keys, and a report's rows commit together.
- `docker compose up` needs one stateful service, which is what makes NFR-6 cheap.
- Ceiling: HNSW in Postgres degrades past tens of millions of vectors and competes with
  OLTP load; BM25 via `ts_rank_cd` is weaker than a real BM25 implementation. If the
  corpus grows by orders of magnitude, the dense leg is the first thing to move out.
- The queue has no priorities, delays or dead-letter queue; stale-job recovery is
  application code (`worker.sweep_stale`).
