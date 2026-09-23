"""L1 extraction eval: entity spans (FR-30) and claim typing (FR-31).

`python -m ledgerline.evals.ner [--dry-run]`

Gold data (ledgerline/evals/data/), single annotator, written from six held-out EX-99.1
releases (CSCO, ADBE, HD, NKE, KO, INTC) that the extraction rules were NOT developed on
(development used NVDA, AAPL, MSFT), and before any model output was produced:

ner_gold.jsonl     {"source", "text", "entities": [[label, surface, occurrence], ...]}
    Labels: metric | money | percent | period. Guidelines:
    - metric: the noun phrase naming the measured quantity, including segment/product
      qualifiers ("Data Center revenue") and "total"; excluding accounting-basis words
      (GAAP, non-GAAP, adjusted, comparable, organic), period words and determiners.
      Retail "comparable sales" is a named metric and keeps "comparable".
    - money: "$" amount with its scale word; no "approximately", no "per share". Range
      endpoints are separate entities.
    - percent: number with %, percent, basis points, percentage points, "-point".
    - period: fiscal/reporting periods and comparison baselines ("third quarter of FY2026",
      "year over year", "prior year"); calendar dates are not periods.
typing_gold.jsonl  {"source", "context", "text", "anchor", "occ", "type"}
    The claim's nature (reported_result | comparison | guidance | qualitative) or
    non_claim, for the claim built from `anchor` (a value) or, anchor null, the sentence.
    `context` is the section heading the sentence sits under, when typing depends on it.
"""

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from ledgerline.extract.ner import entity_spans, regex_spans
from ledgerline.extract.pipeline import extract_candidates

DATA = Path(__file__).with_name("data")
ANNOTATOR_NOTE = (
    "Single annotator (the S4 extraction agent). Gold labels were written from held-out "
    "releases (CSCO, ADBE, HD, NKE, KO, INTC) before any model output was produced and "
    "committed before the extraction rules were written; rules were developed on NVDA, "
    "AAPL and MSFT releases. No inter-annotator agreement is available."
)
# Quarter each held-out release reports, needed to resolve periods during typing.
SOURCE_PERIOD = {"CSCO": (2026, "Q4"), "ADBE": (2026, "Q3"), "HD": (2026, "Q2"),
                 "NKE": (2026, "Q4"), "KO": (2026, "Q2"), "INTC": (2026, "Q2")}


def load_ner_gold() -> list[tuple[str, set[tuple[int, int, str]]]]:
    out = []
    for line in (DATA / "ner_gold.jsonl").read_text().splitlines():
        row = json.loads(line)
        spans = set()
        for label, surface, occ in row["entities"]:
            start = -1
            for _ in range(occ + 1):
                start = row["text"].index(surface, start + 1)
            spans.add((start, start + len(surface), label))
        out.append((row["text"], spans))
    return out


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def _matches(pred: list[tuple], gold: set[tuple], partial: bool) -> int:
    """One-to-one matches with the same label; partial = any character overlap."""
    used, tp = set(), 0
    for p in pred:
        for g in gold:
            if g in used or g[2] != p[2]:
                continue
            if (p == g) if not partial else (p[0] < g[1] and g[0] < p[1]):
                used.add(g)
                tp += 1
                break
    return tp


def score_spans(system, gold) -> dict:
    res = {}
    for label in (None, "metric", "money", "percent", "period"):
        tp_e = tp_p = n_pred = n_gold = 0
        for text, g in gold:
            pred = sorted(set(system(text)))
            if label:
                pred, g = [p for p in pred if p[2] == label], {x for x in g if x[2] == label}
            n_pred, n_gold = n_pred + len(pred), n_gold + len(g)
            tp_e += _matches(pred, g, partial=False)
            tp_p += _matches(pred, g, partial=True)
        res[label or "all"] = {"exact": prf(tp_e, n_pred, n_gold),
                               "partial": prf(tp_p, n_pred, n_gold), "n_gold": n_gold}
    return res


def typing_predictions(use_ner: bool) -> list[tuple[str, str]]:
    """(gold, predicted) per typing_gold row."""
    out = []
    for line in (DATA / "typing_gold.jsonl").read_text().splitlines():
        row = json.loads(line)
        prefix = f"{row['context']}\n" if row["context"] else ""
        text = prefix + row["text"]
        fy, fp = SOURCE_PERIOD[row["source"].split()[0]]
        cands = extract_candidates(text, fiscal_year=fy, fiscal_period=fp, use_ner=use_ner)
        cands = [c for c in cands if c.span_end > len(prefix)]
        if row["anchor"]:
            a = len(prefix) - 1
            for _ in range(row["occ"] + 1):
                a = text.index(row["anchor"], a + 1)
            a_end = a + len(row["anchor"])
            hits = [c for c in cands if c.span_start <= a and c.span_end >= a_end and c.value is not None]
            best = min(hits, key=lambda c: c.span_end - a_end, default=None)
            pred = best.claim_type if best else "non_claim"
        else:
            textual = [c for c in cands if c.value is None]
            pred = textual[0].claim_type if textual else cands[0].claim_type if cands else "non_claim"
        out.append((row["type"], pred))
    return out


def score_typing(pairs: list[tuple[str, str]]) -> dict:
    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    per = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == p == c)
        per[c] = prf(tp, sum(1 for _, p in pairs if p == c), sum(1 for g, _ in pairs if g == c)) | {
            "n_gold": sum(1 for g, _ in pairs if g == c), "n_pred": sum(1 for _, p in pairs if p == c)}
    return {"accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 3), "per_class": per,
            "confusion": {f"{g}->{p}": n for (g, p), n in sorted(Counter(pairs).items()) if g != p}}


def run(write: bool = True) -> dict:
    gold = load_ner_gold()
    t0 = time.perf_counter()
    ner = {"regex_only": score_spans(regex_spans, gold),
           "gliner_regex": score_spans(entity_spans, gold)}
    t_ner = time.perf_counter() - t0
    typing = {"rules": score_typing(typing_predictions(use_ner=False)),
              "rules_gliner": score_typing(typing_predictions(use_ner=True))}
    g = ner["gliner_regex"]["all"]
    metrics = {
        "ner_f1": g["exact"]["f1"], "ner_f1_partial": g["partial"]["f1"],
        "ner_f1_regex_only": ner["regex_only"]["all"]["exact"]["f1"],
        "ner_f1_partial_regex_only": ner["regex_only"]["all"]["partial"]["f1"],
        "typing_accuracy": typing["rules_gliner"]["accuracy"],
        **{f"typing_precision_{c}": v["p"] for c, v in typing["rules_gliner"]["per_class"].items()},
    }
    details = {"annotator_note": ANNOTATOR_NOTE, "n_sentences": len(gold),
               "n_entities": sum(len(x) for _, x in gold), "ner": ner, "typing": typing,
               "ner_seconds": round(t_ner, 1)}
    if write:
        from ledgerline.db import session
        from ledgerline.models import EvalRun

        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        with session() as s:
            s.add(EvalRun(git_sha=sha or None, layer="L1", metrics=metrics, details=details))
    return {"metrics": metrics, **details}


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.ner")
    p.add_argument("--dry-run", action="store_true", help="do not write an eval_run row")
    r = run(write=not p.parse_args().dry_run)
    print(f"sentences: {r['n_sentences']}  gold entities: {r['n_entities']}")
    for name, res in r["ner"].items():
        print(f"\n[{name}]  label     exact P/R/F1          partial P/R/F1     n_gold")
        for label, v in res.items():
            e, pa = v["exact"], v["partial"]
            print(f"  {label:8} {e['p']:.3f} {e['r']:.3f} {e['f1']:.3f}   "
                  f"{pa['p']:.3f} {pa['r']:.3f} {pa['f1']:.3f}   {v['n_gold']}")
    for name, t in r["typing"].items():
        print(f"\n[typing {name}] accuracy {t['accuracy']:.3f}")
        for c, v in t["per_class"].items():
            print(f"  {c:16} P {v['p']:.3f}  R {v['r']:.3f}  (gold {v['n_gold']}, pred {v['n_pred']})")
        print("  errors:", t["confusion"])
    print("\nmetrics:", json.dumps(r["metrics"]))


if __name__ == "__main__":
    main()
