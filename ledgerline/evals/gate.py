"""§12 eval gate: `python -m ledgerline.evals.gate [--baseline PATH] [--write-baseline]`

Runs the L2 claim-level eval on the frozen golden set and fails (exit 1) if contradiction
precision drops more than MAX_DROP below the baseline. The threshold is ABSOLUTE:
0.72 -> 0.699 is a 2.1-point drop and fails. A relative 2% of 0.72 would be only 1.4
points, which on a ~200-claim golden set is about one flipped verdict: noise, not signal.

Contract with ledgerline.evals.l2.run() (and the baseline JSON, same shape):
    {"metrics": {"contradiction_precision": float, ...},
     "per_claim": {"<golden id>": {"gold": label, "predicted": label}, ...}}

Writes eval-gate.md (a per-claim verdict diff) for CI to post as a PR comment.
"""

import argparse
import json
import sys
from pathlib import Path

METRIC = "contradiction_precision"
MAX_DROP = 0.02
BASELINE = Path(__file__).with_name("baseline.json")


def compare(current: dict, baseline: dict) -> tuple[bool, str]:
    cur = current["metrics"].get(METRIC)
    if cur is None:
        return False, f"{METRIC} missing from this run (no contradictions predicted?)."
    base = baseline.get("metrics", {}).get(METRIC)
    if base is None:
        return True, f"{METRIC} = {cur:.3f}; no baseline recorded yet, gate passes."
    drop = base - cur
    ok = drop <= MAX_DROP + 1e-9
    verb = "within" if ok else "exceeds"
    return ok, (f"{METRIC}: baseline {base:.3f}, this PR {cur:.3f} "
                f"(change {-drop * 100:+.1f} pts; {verb} the {MAX_DROP * 100:.0f}-pt limit).")


def verdict_diff(current: dict, baseline: dict) -> list[tuple[str, str | None, str | None, str]]:
    """(golden id, baseline prediction, current prediction, gold) for every changed claim."""
    base, cur = baseline.get("per_claim", {}), current.get("per_claim", {})
    return [
        (k, base.get(k, {}).get("predicted"), cur.get(k, {}).get("predicted"),
         (cur.get(k) or base.get(k))["gold"])
        for k in sorted(base.keys() | cur.keys())
        if base.get(k, {}).get("predicted") != cur.get(k, {}).get("predicted")
    ]


def render(ok: bool, summary: str, diff: list) -> str:
    lines = [f"### Eval gate: {'pass' if ok else 'FAIL'}", "", summary, ""]
    if diff:
        lines += ["| golden id | baseline | this PR | gold |", "|---|---|---|---|"]
        lines += [f"| {k} | {b} | {c} | {g} |" for k, b, c, g in diff]
    else:
        lines.append("No per-claim verdict changes.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ledgerline.evals.gate")
    p.add_argument("--baseline", type=Path, default=BASELINE)
    p.add_argument("--write-baseline", action="store_true",
                   help="record this run as the new baseline (do this on main, not in a PR)")
    p.add_argument("--out", type=Path, default=Path("eval-gate.md"))
    a = p.parse_args(argv)

    from ledgerline.evals import l2

    current = l2.run()
    if a.write_baseline:
        a.baseline.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"wrote {a.baseline}")
        return 0
    baseline = json.loads(a.baseline.read_text()) if a.baseline.exists() else {}
    ok, summary = compare(current, baseline)
    a.out.write_text(render(ok, summary, verdict_diff(current, baseline)))
    print(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
