"""Frozen golden claim set for L2/L3 (PRD §11, R3) and the perturbation set.

    python -m ledgerline.evals.golden sample      # stratified draw -> data/golden_sample.jsonl
    python -m ledgerline.evals.golden worksheet [--ids g001,g002] [--pass2]
    python -m ledgerline.evals.golden perturb     # data/golden_perturbed.jsonl from golden.jsonl
    python -m ledgerline.evals.golden sample-heldout   # -> data/heldout_sample.jsonl

Records are self-contained snapshots of the claim fields, so the set stays frozen when
extraction is re-run and claim ids change; the source filing is located by accession number.

Who labeled, and how (also written into every record's `annotator` field): see ANNOTATOR.
Label guidelines (the gold label judges what the claim TEXT asserts, as said in the release,
against the company's 10-K/10-Q text and XBRL facts in the database, NOT the extractor's
normalized fields; a normalization error is recorded in `extraction_note`):
  supported     a filed figure for the same measure, basis (GAAP / segment / total), and
                period matches within the claim's printed rounding. Q4 figures may be
                derived as FY minus the nine-month year-to-date figure.
  contradicted  a filed figure for the same measure, basis and period differs beyond
                rounding. Guidance: the later filed actual landed outside the guided range.
  unverifiable  no filed counterpart: non-GAAP / adjusted / constant-currency measures
                that no 10-K/10-Q prints (a non-GAAP figure a 10-Q reconciliation prints
                IS verifiable), operating metrics absent from filings, guidance for a period
                not yet reported, subjective statements, or no stored filing has the period.
"""

import argparse
import gzip
import json
import random
import re
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal
from functools import cache
from pathlib import Path, PurePath

from psycopg import sql
from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from ledgerline.concepts import CONCEPTS
from ledgerline.models import Claim, Company, Filing, XbrlFact

DATA = Path(__file__).with_name("data")
GOLDEN = DATA / "golden.jsonl"
SAMPLE = DATA / "golden_sample.jsonl"
PERTURBED = DATA / "golden_perturbed.jsonl"
HELDOUT = DATA / "heldout.jsonl"
HELDOUT_SAMPLE = DATA / "heldout_sample.jsonl"
SEED = 20260923
QUOTA = {"reported_result": 90, "comparison": 45, "guidance": 25, "qualitative": 40}
# Held-out set (M1): a new seed, today's extractor, nothing golden.jsonl already covers.
# "guidance_known" = guidance whose target period has a stored 10-Q/10-K (an outcome to check).
HELDOUT_EXTRACTOR = "s4-llm-1"
HELDOUT_SEED = 20260925
HELDOUT_QUOTA = {
    "reported_result": 40,
    "comparison": 25,
    "guidance_known": 20,
    "guidance": 5,
    "qualitative": 20,
}
LABELS = ("supported", "contradicted", "unverifiable")
FIELDS = (
    "text",
    "claim_type",
    "metric",
    "value",
    "value_high",
    "unit",
    "fiscal_year",
    "fiscal_period",
    "target_fiscal_year",
    "target_fiscal_period",
    "comparison",
    "hedging_score",
    "speaker_role",
)
ANNOTATOR = (
    "Single annotator: the Claude agent that built the D evals (claude-opus-5-5), not a human. "
    "Pass 1 read the release context and the raw xbrl_fact rows / 10-K and 10-Q text lines "
    "printed by `golden worksheet`, and was written before verify_claim was run on any golden "
    "claim. Pass 2 re-labeled a subset with pass-1 labels hidden from a text-only worksheet "
    "(no XBRL rows), by the same annotator: it measures intra-annotator consistency across two "
    "evidence routes, not inter-annotator agreement."
)


def jsonable(v):
    return float(v) if isinstance(v, Decimal) else v


def load(path: Path = GOLDEN) -> list[dict]:
    """Golden records, validated: unique ids, a label from LABELS, evidence recorded."""
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r["id"] for r in recs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate golden ids in {path}")
    for r in recs:
        if r["gold"] not in LABELS or not r.get("evidence"):
            raise ValueError(f"{r['id']}: gold label must be one of {LABELS} with evidence")
        for k in ("pass1", "pass2"):
            if r.get(k) not in (None, *LABELS):
                raise ValueError(f"{r['id']}: bad {k} label {r[k]!r}")
    return recs


def to_claim(s: Session, rec: dict) -> Claim:
    """A transient (never added to the session) Claim for verify_claim."""
    filing = s.scalars(select(Filing).where(Filing.accession_no == rec["accession_no"])).one()
    c = rec["claim"]
    return Claim(
        company_id=filing.company_id,
        filing_id=filing.id,
        span_start=rec["span"][0],
        span_end=rec["span"][1],
        extractor_version=rec["extractor_version"],
        **{k: c[k] for k in FIELDS if k not in ("value", "value_high")},
        value=None if c["value"] is None else Decimal(str(c["value"])),
        value_high=None if c["value_high"] is None else Decimal(str(c["value_high"])),
    )


# ---------------------------------------------------------------- stratified sample


def stratified(claims: list[dict], quota: dict[str, int], seed: int = SEED) -> list[dict]:
    """Per claim type: round-robin over companies; within a company, metrics interleaved
    so one metric can't fill a company's share. Deterministic for a fixed seed."""
    rng = random.Random(seed)
    out = []
    for ctype, n in quota.items():
        by_co: dict[str, list[dict]] = defaultdict(list)
        for c in claims:
            if c["claim_type"] == ctype:
                by_co[c["ticker"]].append(c)
        queues = []
        for ticker in sorted(by_co):
            by_metric: dict[str, list[dict]] = defaultdict(list)
            for c in sorted(by_co[ticker], key=lambda c: c["source_claim_id"]):
                by_metric[str(c["metric"])].append(c)
            for lst in by_metric.values():
                rng.shuffle(lst)
            keys = sorted(by_metric)
            rng.shuffle(keys)
            q = []
            while any(by_metric.values()):
                q += [by_metric[k].pop() for k in keys if by_metric[k]]
            queues.append(q)
        rng.shuffle(queues)
        picked = []
        while len(picked) < n and any(queues):
            for q in queues:
                if q and len(picked) < n:
                    picked.append(q.pop(0))
        out += picked
    return out


def sample(s: Session) -> list[dict]:
    """Draw QUOTA claims from the claims table. golden.jsonl was drawn once, from extractor
    s4-rules-1; re-running this after a re-extraction draws a different set (by design the
    frozen set does not follow the extractor)."""
    rows = s.execute(
        select(Claim, Company.ticker, Filing.accession_no)
        .join(Company, Company.id == Claim.company_id)
        .join(Filing, Filing.id == Claim.filing_id)
    ).all()
    claims = [
        {
            "source_claim_id": c.id,
            "ticker": t,
            "accession_no": acc,
            "extractor_version": c.extractor_version,
            "span": [c.span_start, c.span_end],
            "claim_type": c.claim_type,
            "metric": c.metric,
            "claim": {k: jsonable(getattr(c, k)) for k in FIELDS},
        }
        for c, t, acc in rows
    ]
    picked = stratified(claims, QUOTA)
    return [
        {"id": f"g{i + 1:03d}", **{k: v for k, v in c.items() if k not in ("claim_type", "metric")}}
        for i, c in enumerate(picked)
    ]


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def heldout_candidates(claims: list[dict], golden: list[dict], judged: set[int]) -> list[dict]:
    """Claims that share no filing span with a golden record and that the system has never
    verified (`judged` = claim ids with a verdict row), so labels are blind to its output."""
    spans = defaultdict(list)
    for g in golden:
        spans[g["accession_no"]].append(tuple(g["span"]))
    return [
        c
        for c in claims
        if c["source_claim_id"] not in judged
        and not any(_overlaps(tuple(c["span"]), sp) for sp in spans[c["accession_no"]])
    ]


def sample_heldout(s: Session) -> list[dict]:
    rows = s.execute(
        text("""
        SELECT c.id, co.ticker, f.accession_no, c.claim_type = 'guidance' AND EXISTS (
                 SELECT 1 FROM filing p WHERE p.company_id = c.company_id
                   AND p.form_type IN ('10-Q', '10-K') AND p.fiscal_year = c.target_fiscal_year
                   AND (p.fiscal_period = c.target_fiscal_period
                        OR (c.target_fiscal_period = 'Q4' AND p.fiscal_period = 'FY')))
        FROM claim c JOIN company co ON co.id = c.company_id JOIN filing f ON f.id = c.filing_id
        WHERE c.extractor_version = :v"""),
        {"v": HELDOUT_EXTRACTOR},
    ).all()
    known = {cid for cid, _, _, k in rows if k}
    meta = {cid: (t, acc) for cid, t, acc, _ in rows}
    judged = set(s.scalars(text("SELECT DISTINCT claim_id FROM verdict")))
    claims = []
    for c in s.scalars(select(Claim).where(Claim.id.in_(meta))):
        t, acc = meta[c.id]
        claims.append(
            {
                "source_claim_id": c.id,
                "ticker": t,
                "accession_no": acc,
                "extractor_version": c.extractor_version,
                "span": [c.span_start, c.span_end],
                "claim_type": "guidance_known" if c.id in known else c.claim_type,
                "metric": c.metric,
                "claim": {k: jsonable(getattr(c, k)) for k in FIELDS},
            }
        )
    picked = stratified(
        heldout_candidates(claims, load(GOLDEN), judged), HELDOUT_QUOTA, HELDOUT_SEED
    )
    return [
        {"id": f"h{i + 1:03d}", **{k: v for k, v in c.items() if k not in ("claim_type", "metric")}}
        for i, c in enumerate(picked)
    ]


# ---------------------------------------------------------------- annotation worksheet


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def printed(text: str, value: float) -> tuple[re.Match, float] | None:
    """The first number in `text` that renders `value`, and its scale (1e9 for "94.9 billion")."""
    for m in _NUM.finditer(text):
        x = float(m.group().replace(",", ""))
        for scale in (1e9, 1e6, 1e3, 1.0):
            if x and abs(x * scale - abs(value)) <= 1e-6 * abs(value):
                return m, scale
    return None


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def number_matcher(claim: dict):
    """line -> bool: does the line print the claim's number? Either a number as printed in
    the claim, or (dollar amounts) a millions/thousands figure within the printed rounding."""
    exact = {t for t in _NUM.findall(claim["text"]) if len(t.replace(",", "")) >= 2}
    ranges = []
    v = claim["value"]
    if (
        claim["unit"] == "USD"
        and v is not None
        and abs(v) >= 1e6
        and (hit := printed(claim["text"], v))
    ):
        m, scale = hit
        half = 0.5 * 10 ** -_decimals(m.group()) * scale
        ranges = [((abs(v) - half) / d, (abs(v) + half) / d) for d in (1e6, 1e3)]

    def match(line: str) -> bool:
        for t in _NUM.findall(line):
            if t in exact:
                return True
            if "." not in t and any(lo <= int(t.replace(",", "")) <= hi for lo, hi in ranges):
                return True
        return False

    return match


@cache
def _text(storage_path: str) -> str:
    from ledgerline.ingest.documents import html_to_text
    from ledgerline.ingest.objects import read_object

    return html_to_text(read_object(storage_path), PurePath(storage_path).suffix)


def _periodic_lines(s: Session, company_id: int, claim: dict, match, limit: int = 12) -> list[str]:
    fy = claim["target_fiscal_year"] or claim["fiscal_year"]
    filings = s.scalars(
        select(Filing)
        .where(
            Filing.company_id == company_id,
            Filing.form_type.in_(("10-Q", "10-K")),
            Filing.fiscal_year.between(fy, fy + 1),
        )
        .order_by(Filing.filed_at)
    ).all()
    out = []
    for f in filings:
        hits = [line for line in _text(f.storage_path).splitlines() if match(line)]
        for line in hits[:limit]:
            out.append(f"  [{f.form_type} {f.fiscal_year}{f.fiscal_period}] {line[:260]}")
    return out


def _xbrl_rows(s: Session, company_id: int, claim: dict, event) -> list[str]:
    concept = CONCEPTS.get(claim["metric"] or "")
    if concept is None:
        return []
    names = [concept.name, *(concept.ratio_of or concept.diff_of or ())]
    tags = [t for n in names for t in CONCEPTS[n].tags]
    rows = s.scalars(
        select(XbrlFact)
        .where(
            XbrlFact.company_id == company_id,
            XbrlFact.tag.in_(tags),
            XbrlFact.period_end.between(event - timedelta(days=500), event + timedelta(days=400)),
        )
        .order_by(XbrlFact.tag, XbrlFact.period_end, XbrlFact.duration_days)
    ).all()
    seen, out = set(), []
    for r in rows:
        key = (r.tag, r.period_start, r.period_end, r.value)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            f"  id={r.id} {r.tag.split(':')[1]} {float(r.value):,.2f} {r.unit} "
            f"{r.period_start}..{r.period_end} ({r.duration_days}d) "
            f"fy={r.fiscal_year}{r.fiscal_period} {r.form}{' DERIVED' if r.derived else ''}"
        )
    return out


def worksheet(s: Session, recs: list[dict], hide_labels: bool = True, xbrl: bool = True) -> str:
    """Raw material for labeling: release context, raw XBRL rows, periodic-filing lines that
    print the number. It shows filings and facts only, never system output."""
    out = []
    for r in recs:
        f = s.scalars(select(Filing).where(Filing.accession_no == r["accession_no"])).one()
        c, (a, b) = r["claim"], r["span"]
        text = _text(f.storage_path)
        ctx = text[max(0, a - 300) : b + 200].replace("\n", " / ")
        out += [
            (
                f"=== {r['id']} {r['ticker']} {c['claim_type']} metric={c['metric']} "
                f"value={c['value']} hi={c['value_high']} unit={c['unit']} "
                f"release={c['fiscal_year']}{c['fiscal_period']} "
                f"target={c['target_fiscal_year']}{c['target_fiscal_period']} "
                f"cmp={c['comparison']}"
            ),
            f"CLAIM: {c['text']}",
            f"CONTEXT: ...{ctx}...",
        ]
        if not hide_labels and "gold" in r:
            out.append(f"LABEL: {r['gold']} | {r['evidence']}")
        if xbrl:
            out += ["XBRL:", *(_xbrl_rows(s, f.company_id, c, f.filed_at) or ["  (none)"])]
        if c["value"] is not None:
            out += [
                "FILINGS:",
                *(
                    _periodic_lines(s, f.company_id, c, number_matcher(c))
                    or ["  (no line prints the number)"]
                ),
            ]
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- perturbation set


def _rewrite_number(text: str, value: float, factor: float) -> tuple[str, float] | None:
    """Scale the printed token that renders `value` by `factor`, keeping its format.
    Returns (new text, new value as printed), or None when the printed number would not
    move by at least 10% after rounding (e.g. "1%" x1.15 still prints "1")."""
    if (hit := printed(text, value)) is None:
        return None
    m, scale = hit
    old = float(m.group().replace(",", ""))
    dec = _decimals(m.group())
    new = round(old * factor, dec)
    if abs(new - old) < 0.1 * abs(old):
        return None
    tok = f"{new:,.{dec}f}" if "," in m.group() else f"{new:.{dec}f}"
    return text[: m.start()] + tok + text[m.end() :], (1 if value > 0 else -1) * new * scale


def perturb(recs: list[dict], seed: int = SEED) -> list[dict]:
    """Real claims labeled `supported` whose value is numeric, with an injected error:
    the printed value scaled by 0.8/0.85/1.15/1.25 (text and value rewritten together).
    Gold = contradicted by construction: the unperturbed claim matched the filing within
    its printed rounding, and the perturbed one is at least 10% away from it.

    ponytail: value errors only; period shifts were dropped because a shifted period can
    land on a coincidentally close value, which would need a per-item re-label."""
    rng = random.Random(seed)
    out = []
    for r in recs:
        c = r["claim"]
        if (
            r["gold"] != "supported"
            or not c["value"]
            or c["value_high"] is not None
            or c["claim_type"] not in ("reported_result", "comparison")
        ):
            continue
        factor = rng.choice((0.8, 0.85, 1.15, 1.25))
        if (hit := _rewrite_number(c["text"], c["value"], factor)) is None:
            continue
        text, value = hit
        out.append(
            r
            | {
                "id": f"p{r['id'][1:]}",
                "perturbed_from": r["id"],
                "claim": c | {"text": text, "value": round(value, 6)},
                "gold": "contradicted",
                "pass1": None,
                "pass2": None,
                "adjudication": None,
                "evidence": f"Perturbation of {r['id']} (printed value x{factor}); source evidence: "
                f"{r['evidence']}",
                "annotator": "Constructed by golden.perturb; gold by construction from the source label.",
            }
        )
    return out


# ---------------------------------------------------------------- CI fixture (D5)

FIXTURE = DATA / "golden_fixture"
# table -> (columns, WHERE clause over {t} = the ticker list). Only what verify_claim reads
# for golden claims: 10-K/10-Q chunks (FilingRAG never searches 8-Ks), all XBRL facts of the
# companies, their prices and SPY's for the market-reaction agent. Chunk embeddings are recomputed on load: the model
# is deterministic and the text is ~5x smaller than the vectors.
_FIXTURE_TABLES = {
    "company": ("id, cik, ticker, name, fiscal_year_end", "ticker = ANY({t})"),
    "filing": ("*", "company_id IN (SELECT id FROM company WHERE ticker = ANY({t}))"),
    "xbrl_fact": ("*", "company_id IN (SELECT id FROM company WHERE ticker = ANY({t}))"),
    "chunk": (
        "id, filing_id, chunk_index, section_path, char_start, char_end, text, token_count",
        (
            "filing_id IN (SELECT f.id FROM filing f JOIN company c ON c.id = f.company_id "
            "WHERE c.ticker = ANY({t}) AND f.form_type IN ('10-K', '10-Q', '10-K/A', '10-Q/A'))"
        ),
    ),
    "market_price": ("*", "ticker = ANY({t}) OR ticker = 'SPY'"),
}


def _raw(s: Session):
    return s.connection().connection.driver_connection  # psycopg connection, for COPY


def dump_fixture(s: Session, tickers: list[str], out: Path = FIXTURE) -> dict[str, int]:
    """Write one gzipped CSV per table; returns table -> compressed bytes."""
    out.mkdir(parents=True, exist_ok=True)
    with _raw(s).cursor() as cur:
        for table, (cols, where) in _FIXTURE_TABLES.items():
            query = sql.SQL(
                f"COPY (SELECT {cols} FROM {table} WHERE {where} ORDER BY 1) "
                "TO STDOUT WITH (FORMAT csv, HEADER)"
            ).format(t=sql.Literal(tickers))
            with gzip.open(out / f"{table}.csv.gz", "wb") as f, cur.copy(query) as copy:
                for block in copy:
                    f.write(block)
    return {t: (out / f"{t}.csv.gz").stat().st_size for t in _FIXTURE_TABLES}


def load_fixture(s: Session, src: Path = FIXTURE, batch: int = 256) -> int:
    """Load the fixture into an empty, migrated database; returns chunks embedded.
    Rows keep their ids (citations and chunk ids stay comparable with the source DB), so
    each id sequence is moved past the loaded rows afterwards."""
    from ledgerline.models import Chunk
    from ledgerline.store import embed
    from ledgerline.store.index import passage

    with _raw(s).cursor() as cur:
        for table in _FIXTURE_TABLES:
            with gzip.open(src / f"{table}.csv.gz", "rb") as f:
                header = f.readline().decode().strip()
                with cur.copy(f"COPY {table} ({header}) FROM STDIN WITH (FORMAT csv)") as copy:
                    while block := f.read(1 << 20):
                        copy.write(block)
    for table in _FIXTURE_TABLES:
        s.execute(
            text(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                f"(SELECT coalesce(max(id), 0) + 1 FROM {table}), false)"
            )
        )
    rows = s.execute(
        select(
            Chunk.id,
            Company.name,
            Filing.form_type,
            Filing.fiscal_year,
            Filing.fiscal_period,
            Filing.period_of_report,
            Chunk.section_path,
            Chunk.text,
        )
        .join(Filing, Filing.id == Chunk.filing_id)
        .join(Company, Company.id == Filing.company_id)
        .where(Chunk.embedding.is_(None))
        .order_by(Chunk.id)
    ).all()
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        vecs = embed.embed_passages([passage(*r[1:]) for r in part])
        s.execute(
            Chunk.__table__.update()
            .where(Chunk.id == bindparam("cid"))
            .values(embedding=bindparam("vec")),
            [{"cid": r[0], "vec": v} for r, v in zip(part, vecs)],
        )
    return len(rows)


def _write(path: Path, recs: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))


def main(argv: list[str] | None = None) -> None:
    from ledgerline.db import session

    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.golden")
    p.add_argument(
        "cmd",
        choices=(
            "sample",
            "sample-heldout",
            "worksheet",
            "perturb",
            "dump-fixture",
            "load-fixture",
        ),
    )
    p.add_argument("--ids", help="comma-separated golden ids (worksheet)")
    p.add_argument("--source", type=Path, default=None, help="records file (worksheet)")
    p.add_argument("--pass2", action="store_true", help="worksheet without XBRL rows")
    a = p.parse_args(argv)
    if a.cmd == "perturb":
        recs = perturb(load())
        _write(PERTURBED, recs)
        print(f"wrote {len(recs)} perturbed claims to {PERTURBED}")
        return
    with session() as s:
        if a.cmd == "dump-fixture":
            print(dump_fixture(s, sorted({r["ticker"] for r in load()})))
            return
        if a.cmd == "load-fixture":
            print(f"loaded fixture, embedded {load_fixture(s)} chunks")
            return
        if a.cmd == "sample":
            recs = sample(s)
            _write(SAMPLE, recs)
            print(f"wrote {len(recs)} sampled claims to {SAMPLE}")
            return
        if a.cmd == "sample-heldout":
            recs = sample_heldout(s)
            _write(HELDOUT_SAMPLE, recs)
            print(f"wrote {len(recs)} sampled claims to {HELDOUT_SAMPLE}")
            return
        src = a.source or (GOLDEN if GOLDEN.exists() else SAMPLE)
        recs = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
        if a.ids:
            want = set(a.ids.split(","))
            recs = [r for r in recs if r["id"] in want]
        print(worksheet(s, recs, xbrl=not a.pass2))


if __name__ == "__main__":
    main()
