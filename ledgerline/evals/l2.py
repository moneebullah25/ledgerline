"""L2 claim-level eval (PRD §11): verify_claim on the frozen golden set.

    python -m ledgerline.evals.l2 [--no-write] [--set golden|heldout]

Every golden claim (golden.jsonl, real claims, labeled blind to system output) and every
perturbation (golden_perturbed.jsonl, real supported claims with an injected wrong value)
goes through the real verification graph. Nothing is written to claim/verdict tables: each
claim is a transient Claim built from the frozen snapshot.

Metrics: precision / recall / F1 per label on the REAL set; the headline (what the §12 gate
compares) is `contradiction_precision` on the real set. Real contradictions are rare, so the
perturbation set is reported separately (`perturbed_*`): it measures whether a wrong number
is caught, and `combined_contradiction_precision` shows precision when such errors exist.
The combined figure depends on the arbitrary real:perturbed mix, so it is never the headline.
Contradiction precision, contradiction recall and accuracy also carry exact (Clopper-Pearson)
95% intervals (`*_ci`, [low, high]; null when the denominator is 0): on ~10-25 real
contradictions a point estimate alone overstates what the set can show.

`--set heldout` (eval_run name "l2_heldout") scores data/heldout.jsonl instead: 110 current
(s4-llm-1) claims drawn with a new seed, sharing no filing span with golden.jsonl, and labelled
before the system ran on them. golden.jsonl was used while the graph was being tuned, so the
held-out set is the unbiased estimate. It has no perturbation set.

Ablations (same claims, the full run's evidence replayed for every agent the ablation does
not touch, so only the ablated part changes):
  no_reranker  FilingRAG retrieval without the cross-encoder rerank (top-K of the RRF fusion).
  no_critic    the critic returns no weak point. Offline the critic only lowers confidence,
               so labels cannot change; the ablation is still run so an online run shows it.
  xbrl_only    every agent except the XBRL typed query answers `insufficient`.

A claim whose verification raises is scored as a refused `unverifiable` (what the pipeline
stores, F5) and counted in `errors`; database errors and ImportError propagate.

`--current` (eval_run name "l2_current") re-points every golden record at the claim the
current extractor (extract.pipeline.extractor_version()) produced for the same filing and an
overlapping span
(preferring the same claim type and comparison, then the highest span IoU, then the newest
claim), keeping the frozen gold label.
It measures the graph on today's extraction; records with no overlapping claim are listed as
unmatched and left out. The perturbation set is built from the frozen snapshot, so it is not
re-mapped (n_perturbed = 0). Caveat: the gold label judged the frozen claim's text; a
re-mapped span that covers different words is still scored against it.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.evals.golden import (
    ANNOTATOR,
    FIELDS,
    GOLDEN,
    HELDOUT,
    LABELS,
    PERTURBED,
    jsonable,
    load,
    to_claim,
)
from ledgerline.extract.pipeline import extractor_version
from ledgerline.graph import build, critic
from ledgerline.graph.adjudicator import decide
from ledgerline.graph.agents import filing_rag
from ledgerline.models import Claim, Filing
from ledgerline.schemas import CriticOutput, EvidenceItem
from ledgerline.store.retrieval import search
from ledgerline.tracing import Tracer

SETS = {"golden": GOLDEN, "heldout": HELDOUT}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float] | None:
    """Exact two-sided (1 - alpha) binomial interval for k successes in n trials; None if n == 0."""
    if n == 0:
        return None
    from scipy.stats import beta  # already installed with the [ml] extra L2 needs

    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [low, high]


def prf(pairs: list[tuple[str, str]]) -> dict:
    """pairs: (gold, predicted). Per-label precision/recall/F1 (None when undefined)."""
    out = {
        "n": len(pairs),
        "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None,
    }
    f1s = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in pairs)
        pred = sum(p == lab for _, p in pairs)
        gold = sum(g == lab for g, _ in pairs)
        prec = tp / pred if pred else None
        rec = tp / gold if gold else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if pred and gold else None)
        out[lab] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp,
            "predicted": pred,
            "gold": gold,
        }
        if f1 is not None:
            f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else None
    out["confusion"] = {f"{g}->{p}": n for (g, p), n in sorted(Counter(pairs).items())}
    return out


def _fixed(ev: EvidenceItem):
    return lambda *_: (ev, llm.OFFLINE)


def _insufficient(agent: str) -> EvidenceItem:
    return EvidenceItem(agent=agent, stance="insufficient", summary="disabled for ablation")


def _neutral_critic(claim, evidence):
    stance, _ = decide(claim, evidence)
    counter = {"supports": "contradicts", "contradicts": "supports"}.get(stance, "insufficient")
    return CriticOutput(
        counter_stance=counter, argument="Critic disabled for ablation.", weakest_evidence=None
    ), llm.OFFLINE


def _search_no_rerank(s: Session, query: str, company_id: int, lock):
    """Stand-in for filing_rag._search: same filings, RRF order, no cross-encoder. Holds the
    lock through the query embedding too (serialises agents; fine for an ablation)."""
    with lock:
        fids = list(
            s.scalars(
                select(Filing.id).where(
                    Filing.company_id == company_id, Filing.form_type.in_(filing_rag.PERIODIC)
                )
            )
        )
        return (
            search(s, query, company_id=company_id, k=filing_rag.K, filing_ids=fids, rerank=False)
            if fids
            else []
        )


def _match_key(old: dict, a: int, b: int, c: Claim) -> tuple:
    """Same claim type, same comparison, then span IoU; the newest (highest id) breaks ties."""
    iou = (min(b, c.span_end) - max(a, c.span_start)) / (max(b, c.span_end) - min(a, c.span_start))
    return c.claim_type == old["claim_type"], c.comparison == old["comparison"], iou, c.id


def remap_current(s: Session, recs: list[dict]) -> tuple[list[dict], list[str]]:
    """(records re-pointed at the current claims table, ids with no overlapping claim)."""
    out, unmatched = [], []
    for r in recs:
        (a, b), old = r["span"], r["claim"]
        cands = s.scalars(
            select(Claim)
            .join(Filing, Filing.id == Claim.filing_id)
            .where(
                Filing.accession_no == r["accession_no"],
                Claim.extractor_version == extractor_version(),
                Claim.span_start < b,
                Claim.span_end > a,
            )
        ).all()
        if not cands:
            unmatched.append(r["id"])
            continue
        c = max((_match_key(old, a, b, c), c) for c in cands)[1]  # key ends in the unique id
        out.append(
            r
            | {
                "source_claim_id": c.id,
                "extractor_version": c.extractor_version,
                "span": [c.span_start, c.span_end],
                "claim": {k: jsonable(getattr(c, k)) for k in FIELDS},
            }
        )
    return out, unmatched


def _predict_one(s: Session, r: dict, agents: dict | None = None) -> dict:
    try:
        res = build.verify_claim(s, to_claim(s, r), Tracer(), agents)
        return {
            "label": res.verdict.label,
            "refused": res.refused,
            "evidence": {e.agent: e for e in res.evidence},
        }
    except (SQLAlchemyError, ImportError, llm.LLMUnavailable):
        raise
    except Exception as e:  # noqa: BLE001 - scored as the pipeline stores it
        return {
            "label": "unverifiable",
            "refused": True,
            "evidence": {},
            "error": f"{type(e).__name__}: {e}",
        }


def predict(s: Session, recs: list[dict], override=None, workers: int = 1) -> dict[str, dict]:
    """golden id -> {"label", "refused", "evidence": {agent: EvidenceItem}, "error"?}. A claim
    whose verification raises scores as the pipeline stores it: a refused `unverifiable`
    (F5). Database errors and ImportError propagate: they would fail every claim alike.
    override(rid) -> {agent: fn} replaces agents for that claim only (ablations). workers > 1
    verifies claims in parallel, one Session per claim, for the slow online path."""

    def agents(r):
        return override(r["id"]) if override else None

    if workers <= 1:
        return {r["id"]: _predict_one(s, r, agents(r)) for r in recs}
    from ledgerline.db import engine

    def one(r):
        with Session(engine()) as ws:
            try:
                return r["id"], _predict_one(ws, r, agents(r))
            finally:
                ws.rollback()

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, recs))


def save_evidence(pred: dict[str, dict], path: str) -> None:
    """The per-claim evidence L3's judge reads, so it does not re-run the graph."""
    with open(path, "w") as f:
        json.dump(
            {rid: {a: e.model_dump() for a, e in p["evidence"].items()} for rid, p in pred.items()},
            f,
        )


def ablations(full: dict[str, dict]) -> dict:
    """name -> (agent override per golden id, extra patch context)."""

    def keep(rid, names):
        return {n: _fixed(e) for n, e in full[rid]["evidence"].items() if n in names}

    others = [a for a in build.AGENTS if a != "xbrl_table"]
    return {
        "no_reranker": (
            lambda rid: keep(rid, set(build.AGENTS) - {"filing_rag"}),
            patch.object(filing_rag, "_search", _search_no_rerank),
        ),
        "no_critic": (
            lambda rid: keep(rid, set(build.AGENTS)),
            patch.object(critic, "critique", _neutral_critic),
        ),
        "xbrl_only": (
            lambda rid: keep(rid, {"xbrl_table"}) | {a: _fixed(_insufficient(a)) for a in others},
            nullcontext(),
        ),
    }


def summarize(recs: list[dict], pert: list[dict], pred: dict[str, dict]) -> tuple[dict, dict]:
    """(flat gate metrics, per-set prf detail); the headline is computed on the real set."""
    real = [(r["gold"], pred[r["id"]]["label"]) for r in recs]
    per = [(r["gold"], pred[r["id"]]["label"]) for r in pert]
    m, mp, mc = prf(real), prf(per), prf(real + per)
    c = m["contradicted"]
    return {
        "contradiction_precision": c["precision"],
        "contradiction_precision_ci": clopper_pearson(c["tp"], c["predicted"]),
        "contradiction_recall": c["recall"],
        "contradiction_recall_ci": clopper_pearson(c["tp"], c["gold"]),
        "accuracy": m["accuracy"],
        "accuracy_ci": clopper_pearson(sum(g == p for g, p in real), len(real)),
        "macro_f1": m["macro_f1"],
        **{
            f"{lab}_{k}": m[lab][k]
            for lab in ("supported", "unverifiable")
            for k in ("precision", "recall", "f1")
        },
        "perturbed_contradiction_recall": mp["contradicted"]["recall"],
        "combined_contradiction_precision": mc["contradicted"]["precision"],
        "n_real": len(recs),
        "n_perturbed": len(pert),
        "errors": sum("error" in p for p in pred.values()),
    }, {"real": m, "perturbed": mp, "combined": mc}


def cap_workers(workers: int, pool_size: int) -> int:
    """Parallel claims, each holding a pooled connection while the run's own Session holds
    one more: stay below the pool so no worker waits on (or times out for) a connection."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    capped = min(workers, max(1, pool_size - 1))
    if capped < workers:
        print(f"l2: workers capped at {capped} (DB pool size {pool_size})", file=sys.stderr)
    return capped


def run(
    write: bool = False,
    current: bool = False,
    workers: int = 1,
    ablate: bool = True,
    evidence_out: str | None = None,
    set_name: str = "golden",
) -> dict:
    """The §12 gate contract: {"metrics": {...}, "per_claim": {id: {"gold", "predicted"}}}.
    current=True scores the gold labels against today's claims (see module docstring);
    set_name picks the labelled set (SETS). Perturbations exist for the golden set only."""
    from ledgerline.db import engine

    workers = cap_workers(workers, engine().pool.size())
    recs = load(SETS[set_name])
    pert = load(PERTURBED) if set_name == "golden" and PERTURBED.exists() and not current else []
    unmatched: list[str] = []
    # A plain Session that is never committed: the eval must not write claim/verdict rows.
    with Session(engine()) as s:
        if current:
            recs, unmatched = remap_current(s, recs)
        allr = recs + pert
        full = predict(s, allr, workers=workers)
        if evidence_out:
            save_evidence(full, evidence_out)
        abl = {}
        for name, (override, ctx) in ablations(full).items() if ablate else ():
            with ctx:
                abl[name] = predict(s, allr, override, workers)
        s.rollback()
    metrics, detail = summarize(recs, pert, full)
    ab_metrics = {name: summarize(recs, pert, p)[0] for name, p in abl.items()}
    for name, am in ab_metrics.items():
        metrics[f"ablation_{name}_contradiction_precision"] = am["contradiction_precision"]
        metrics[f"ablation_{name}_accuracy"] = am["accuracy"]
    per_claim = {r["id"]: {"gold": r["gold"], "predicted": full[r["id"]]["label"]} for r in allr}
    if current:
        metrics["n_unmatched"] = len(unmatched)
    result = {"metrics": metrics, "per_claim": per_claim}
    if write:
        from ledgerline.db import session
        from ledgerline.models import EvalRun

        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
        with session() as w:
            run_name = "l2_current" if current else "L2"
            if set_name != "golden":
                run_name = f"l2_{set_name}" + ("_current" if current else "")
            w.add(
                EvalRun(
                    git_sha=sha,
                    layer="L2",
                    name=run_name,
                    metrics=metrics,
                    details={
                        "llm_available": llm.available(),
                        "set": set_name,
                        "annotator": ANNOTATOR if set_name == "golden" else recs[0]["annotator"],
                        **(
                            {
                                "unmatched": unmatched,
                                "claim_ids": {r["id"]: r["source_claim_id"] for r in recs},
                            }
                            if current
                            else {}
                        ),
                        "detail": detail,
                        "ablations": ab_metrics,
                        "per_claim": {
                            rid: {
                                **v,
                                "ablations": {n: abl[n][rid]["label"] for n in abl},
                                **({"error": full[rid]["error"]} if "error" in full[rid] else {}),
                            }
                            for rid, v in per_claim.items()
                        },
                    },
                )
            )
    return result


def positive_int(v: str) -> int:
    """argparse type for worker counts: ThreadPoolExecutor rejects 0 only once it is too late."""
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {n}")
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.l2")
    p.add_argument("--no-write", action="store_true", help="skip the eval_run row")
    p.add_argument(
        "--current",
        action="store_true",
        help="score gold labels against the current extractor's claims",
    )
    p.add_argument(
        "--set",
        dest="set_name",
        choices=sorted(SETS),
        default="golden",
        help="labelled claim set (heldout: blind-labelled, never used for tuning)",
    )
    p.add_argument(
        "--workers",
        type=positive_int,
        default=1,
        help="parallel claims (online path); capped below the DB pool size",
    )
    p.add_argument(
        "--no-ablations",
        action="store_true",
        help="skip the ablation re-runs (each is a full online pass)",
    )
    p.add_argument("--evidence-out", help="save per-claim evidence for `l3 --evidence`")
    a = p.parse_args(argv)
    print(
        json.dumps(
            run(
                write=not a.no_write,
                current=a.current,
                workers=a.workers,
                ablate=not a.no_ablations,
                evidence_out=a.evidence_out,
                set_name=a.set_name,
            )["metrics"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
