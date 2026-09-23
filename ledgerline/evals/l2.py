"""L2 claim-level eval (PRD §11): verify_claim on the frozen golden set.

    python -m ledgerline.evals.l2 [--no-write]

Every golden claim (golden.jsonl, real claims, labeled blind to system output) and every
perturbation (golden_perturbed.jsonl, real supported claims with an injected wrong value)
goes through the real verification graph. Nothing is written to claim/verdict tables: each
claim is a transient Claim built from the frozen snapshot.

Metrics: precision / recall / F1 per label on the REAL set; the headline (what the §12 gate
compares) is `contradiction_precision` on the real set. Real contradictions are rare, so the
perturbation set is reported separately (`perturbed_*`): it measures whether a wrong number
is caught, and `combined_contradiction_precision` shows precision when such errors exist.
The combined figure depends on the arbitrary real:perturbed mix, so it is never the headline.

Ablations (same claims, the full run's evidence replayed for every agent the ablation does
not touch, so only the ablated part changes):
  no_reranker  FilingRAG retrieval without the cross-encoder rerank (top-K of the RRF fusion).
  no_critic    the critic returns no weak point. Offline the critic only lowers confidence,
               so labels cannot change; the ablation is still run so an online run shows it.
  xbrl_only    every agent except the XBRL typed query answers `insufficient`.
"""

import argparse
import json
import subprocess
from collections import Counter
from contextlib import nullcontext
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from ledgerline import llm
from ledgerline.evals.golden import ANNOTATOR, GOLDEN, LABELS, PERTURBED, load, to_claim
from ledgerline.graph import build, critic
from ledgerline.graph.adjudicator import IntegrityError, decide
from ledgerline.graph.agents import filing_rag
from ledgerline.models import Filing
from ledgerline.schemas import CriticOutput, EvidenceItem
from ledgerline.store.retrieval import search
from ledgerline.tracing import Tracer

ERROR = "error"  # verify_claim raised (citation integrity); counts as wrong for every label


def prf(pairs: list[tuple[str, str]]) -> dict:
    """pairs: (gold, predicted). Per-label precision/recall/F1 (None when undefined)."""
    out = {"n": len(pairs), "accuracy": sum(g == p for g, p in pairs) / len(pairs) if pairs else None}
    f1s = []
    for lab in LABELS:
        tp = sum(g == p == lab for g, p in pairs)
        pred = sum(p == lab for _, p in pairs)
        gold = sum(g == lab for g, _ in pairs)
        prec = tp / pred if pred else None
        rec = tp / gold if gold else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if pred and gold else None)
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "predicted": pred,
                    "gold": gold}
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
    return CriticOutput(counter_stance=counter, argument="Critic disabled for ablation.",
                        weakest_evidence=None), llm.OFFLINE


def _search_no_rerank(s: Session, query: str, company_id: int, lock):
    """Stand-in for filing_rag._search: same filings, RRF order, no cross-encoder. Holds the
    lock through the query embedding too (serialises agents; fine for an ablation)."""
    with lock:
        fids = list(s.scalars(select(Filing.id).where(
            Filing.company_id == company_id, Filing.form_type.in_(filing_rag.PERIODIC))))
        return search(s, query, company_id=company_id, k=filing_rag.K, filing_ids=fids,
                      rerank=False) if fids else []


def predict(s: Session, recs: list[dict], override=lambda rid: {}) -> dict[str, dict]:
    """golden id -> {"label", "evidence": {agent: EvidenceItem}, "error"?}. A citation-integrity
    failure is recorded as ERROR (what the pipeline persists per claim, B2); anything else
    propagates, so a broken graph fails the eval instead of scoring as a wrong label."""
    out = {}
    for r in recs:
        with patch.dict(build.AGENTS, override(r["id"])):
            try:
                res = build.verify_claim(s, to_claim(s, r), Tracer())
                out[r["id"]] = {"label": res.verdict.label, "refused": res.refused,
                                "evidence": {e.agent: e for e in res.evidence}}
            except IntegrityError as e:
                out[r["id"]] = {"label": ERROR, "evidence": {}, "error": str(e)}
    return out


def ablations(full: dict[str, dict]) -> dict:
    """name -> (agent override per golden id, extra patch context)."""
    def keep(rid, names):
        return {n: _fixed(e) for n, e in full[rid]["evidence"].items() if n in names}

    others = [a for a in build.AGENTS if a != "xbrl_table"]
    return {
        "no_reranker": (lambda rid: keep(rid, set(build.AGENTS) - {"filing_rag"}),
                        patch.object(filing_rag, "_search", _search_no_rerank)),
        "no_critic": (lambda rid: keep(rid, set(build.AGENTS)),
                      patch.object(critic, "critique", _neutral_critic)),
        "xbrl_only": (lambda rid: keep(rid, {"xbrl_table"})
                      | {a: _fixed(_insufficient(a)) for a in others}, nullcontext()),
    }


def summarize(recs: list[dict], pert: list[dict], pred: dict[str, dict]) -> tuple[dict, dict]:
    """(flat gate metrics, per-set prf detail); the headline is computed on the real set."""
    real = [(r["gold"], pred[r["id"]]["label"]) for r in recs]
    per = [(r["gold"], pred[r["id"]]["label"]) for r in pert]
    m, mp, mc = prf(real), prf(per), prf(real + per)
    return {
        "contradiction_precision": m["contradicted"]["precision"],
        "contradiction_recall": m["contradicted"]["recall"],
        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
        **{f"{lab}_{k}": m[lab][k] for lab in ("supported", "unverifiable")
           for k in ("precision", "recall", "f1")},
        "perturbed_contradiction_recall": mp["contradicted"]["recall"],
        "combined_contradiction_precision": mc["contradicted"]["precision"],
        "n_real": len(recs), "n_perturbed": len(pert),
        "errors": sum(p["label"] == ERROR for p in pred.values()),
    }, {"real": m, "perturbed": mp, "combined": mc}


def run(write: bool = False) -> dict:
    """The §12 gate contract: {"metrics": {...}, "per_claim": {id: {"gold", "predicted"}}}."""
    from ledgerline.db import engine

    recs, pert = load(GOLDEN), (load(PERTURBED) if PERTURBED.exists() else [])
    allr = recs + pert
    # A plain Session that is never committed: the eval must not write claim/verdict rows.
    with Session(engine()) as s:
        full = predict(s, allr)
        abl = {}
        for name, (override, ctx) in ablations(full).items():
            with ctx:
                abl[name] = predict(s, allr, override)
        s.rollback()
    metrics, detail = summarize(recs, pert, full)
    ab_metrics = {name: summarize(recs, pert, p)[0] for name, p in abl.items()}
    for name, am in ab_metrics.items():
        metrics[f"ablation_{name}_contradiction_precision"] = am["contradiction_precision"]
        metrics[f"ablation_{name}_accuracy"] = am["accuracy"]
    per_claim = {r["id"]: {"gold": r["gold"], "predicted": full[r["id"]]["label"]} for r in allr}
    result = {"metrics": metrics, "per_claim": per_claim}
    if write:
        from ledgerline.db import session
        from ledgerline.models import EvalRun

        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip() or None
        with session() as w:
            w.add(EvalRun(git_sha=sha, layer="L2", metrics=metrics, details={
                "llm_available": llm.available(), "annotator": ANNOTATOR,
                "detail": detail, "ablations": ab_metrics, "per_claim": {
                    rid: {**v, "ablations": {n: abl[n][rid]["label"] for n in abl},
                          **({"error": full[rid]["error"]} if "error" in full[rid] else {})}
                    for rid, v in per_claim.items()},
            }))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.l2")
    p.add_argument("--no-write", action="store_true", help="skip the eval_run row")
    a = p.parse_args(argv)
    print(json.dumps(run(write=not a.no_write)["metrics"], indent=2))


if __name__ == "__main__":
    main()
