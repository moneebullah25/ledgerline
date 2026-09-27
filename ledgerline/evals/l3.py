"""L3 judge calibration (PRD §11): does the Opus adjudicator agree with the gold labels?

    python -m ledgerline.evals.l3 [--set golden|heldout] [--workers N] [--evidence PATH] [--no-write]

Requires llm.available(); without credentials it exits 1 (no judge ran, nothing to measure).

For every labelled claim with graph evidence (from `--evidence`, saved by `l2 --evidence-out`,
or a fresh graph run), claims where adjudicator.needs_judgment() is True go to the online
critic + adjudicator (the "judge" tier, Opus); the rest are decided by the deterministic rule
and are skipped (counted as `skipped_deterministic`). The headline is Cohen's kappa between the
adjudicator's labels and gold on the judged claims (target >= 0.60, else gate falls back to L2,
R4), with a 95% percentile bootstrap CI and a gold x predicted confusion matrix.

Note: the online adjudicator's label must equal the rule's (A3) or it falls back to the rule
(`adjudicator_fallbacks`), so this measures the rule-constrained judge, not a free one.

Claims with no evidence are skipped (`skipped_no_evidence`); a claim whose adjudication fails
(ValueError, IntegrityError) is recorded and counted (`judge_errors`). Rate limits abort.
The same-annotator pass1/pass2 agreement is also reported (`annotator_pass_kappa`).
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from ledgerline import llm
from ledgerline.evals.golden import ANNOTATOR, FIELDS, LABELS, load
from ledgerline.graph import adjudicator, critic
from ledgerline.models import Claim
from ledgerline.schemas import EvidenceItem

TARGET_KAPPA = 0.60
NO_LLM = "L3 requires LLM; set ANTHROPIC_API_KEY"


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa for two raters over the same items; None when undefined."""
    if len(a) != len(b):
        raise ValueError("raters must label the same items")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def annotator_agreement(recs: list[dict]) -> dict:
    pairs = [(r["pass1"], r["pass2"]) for r in recs if r.get("pass2") is not None]
    a, b = [p for p, _ in pairs], [q for _, q in pairs]
    return {
        "annotator_pass_kappa": cohen_kappa(a, b),
        "annotator_pass_agreement": (sum(x == y for x, y in pairs) / len(pairs)) if pairs else None,
        "n_pass2": len(pairs),
        "pass2_disagreements": [r["id"] for r in recs
                                if r.get("pass2") is not None and r["pass2"] != r["pass1"]],
    }


def bootstrap_ci(a: list[str], b: list[str], n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap CI for kappa over paired items; resamples with undefined kappa
    are dropped. None when fewer than 2 items or no resample is defined."""
    if len(a) < 2:
        return None
    rng = random.Random(seed)
    ks = []
    for _ in range(n):
        i = rng.choices(range(len(a)), k=len(a))
        if (k := cohen_kappa([a[j] for j in i], [b[j] for j in i])) is not None:
            ks.append(k)
    ks.sort()
    if not ks:
        return None
    return [ks[int(alpha / 2 * (len(ks) - 1))], ks[int((1 - alpha / 2) * (len(ks) - 1))]]


def confusion(gold: list[str], pred: list[str]) -> dict[str, dict[str, int]]:
    """gold label -> predicted label -> count, every LABELS cell present."""
    c = Counter(zip(gold, pred, strict=True))
    return {g: {p: c[g, p] for p in LABELS} for g in LABELS}


def _claim(r: dict) -> Claim:
    """A transient Claim with the fields the critic and adjudicator read (no DB)."""
    c = r["claim"]
    return Claim(**{k: c.get(k) for k in FIELDS if k not in ("value", "value_high")},
                 **{k: None if c.get(k) is None else Decimal(str(c[k]))
                    for k in ("value", "value_high")})


def judge(recs: list[dict], evidence: dict[str, list[EvidenceItem]],
          workers: int = 1) -> dict[str, dict | str]:
    """golden id -> {"label", "model"} from critic + adjudicator, or the error text when the
    adjudication failed. model None means the adjudicator fell back to the rule."""
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    def one(r):
        claim, ev = _claim(r), evidence[r["id"]]
        try:
            crit, _ = critic.critique(claim, ev)
            v, _, model = adjudicator.adjudicate(claim, ev, crit)
        except (ValueError, adjudicator.IntegrityError) as e:  # RateLimited etc. abort
            return r["id"], f"{type(e).__name__}: {e}"
        return r["id"], {"label": v.label, "model": model}
    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, recs))


def _load_evidence(path: str) -> dict[str, list[EvidenceItem]]:
    with open(path) as f:
        saved = json.load(f)
    return {rid: [EvidenceItem.model_validate(e) for e in ev.values()]
            for rid, ev in saved.items()}


def run(evidence_path: str | None = None, workers: int = 1,
        set_name: str = "golden") -> tuple[dict, dict]:
    from ledgerline.evals.l2 import SETS

    if not llm.available():
        raise RuntimeError(NO_LLM)
    recs = load(SETS[set_name])
    if evidence_path:  # saved by `l2 --evidence-out`: same graph run, not a second one
        evidence = _load_evidence(evidence_path)
    else:
        from sqlalchemy.orm import Session

        from ledgerline.db import engine
        from ledgerline.evals.l2 import cap_workers, predict

        with Session(engine()) as s:
            pred = predict(s, recs, workers=cap_workers(workers, engine().pool.size()))
            s.rollback()
        evidence = {rid: list(p["evidence"].values()) for rid, p in pred.items()}
    with_ev = [r for r in recs if evidence.get(r["id"])]
    judged = [r for r in with_ev if adjudicator.needs_judgment(_claim(r), evidence[r["id"]])]
    out = judge(judged, evidence, workers)
    ok = [r for r in judged if isinstance(out[r["id"]], dict)]
    gold, pred_l = [r["gold"] for r in ok], [out[r["id"]]["label"] for r in ok]
    kappa = cohen_kappa(gold, pred_l)
    metrics = {
        "set": set_name, "n_total": len(recs), "judge_n": len(ok),
        "skipped_deterministic": len(with_ev) - len(judged),
        "skipped_no_evidence": len(recs) - len(with_ev),
        "judge_errors": len(judged) - len(ok),
        "adjudicator_fallbacks": sum(out[r["id"]]["model"] is None for r in ok),
        "judge_kappa": kappa, "judge_kappa_ci95": bootstrap_ci(gold, pred_l),
        "target_kappa": TARGET_KAPPA,
        "gate_falls_back_to_l2": (kappa or 0) < TARGET_KAPPA,
        "confusion": confusion(gold, pred_l),
    }
    details: dict = {
        "annotator": ANNOTATOR if set_name == "golden" else recs[0]["annotator"],
        "judge_labels": {k: v for k, v in out.items() if isinstance(v, dict)},
        "judge_errors": {k: v for k, v in out.items() if isinstance(v, str)},
        "skipped_no_evidence": [r["id"] for r in recs if not evidence.get(r["id"])],
    }
    agree = annotator_agreement(recs)
    details["pass2_disagreements"] = agree.pop("pass2_disagreements")
    metrics |= agree
    return metrics, details


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from ledgerline.evals.l2 import SETS, positive_int

    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.l3")
    p.add_argument("--set", dest="set_name", choices=sorted(SETS), default="golden",
                   help="labelled claim set")
    p.add_argument("--no-write", action="store_true", help="skip the eval_run row")
    p.add_argument("--evidence", help="evidence JSON from `l2 --evidence-out` (no graph re-run)")
    p.add_argument("--workers", type=positive_int, default=1, help="parallel judge calls")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    if not llm.available():
        print(NO_LLM, file=sys.stderr)
        sys.exit(1)
    metrics, details = run(a.evidence, a.workers, a.set_name)
    if not a.no_write:
        from ledgerline.db import session
        from ledgerline.models import EvalRun

        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=False).stdout.strip() or None
        with session() as s:
            name = "L3" if a.set_name == "golden" else f"l3_{a.set_name}"
            s.add(EvalRun(git_sha=sha, layer="L3", name=name, metrics=metrics, details=details))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
