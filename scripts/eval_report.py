"""Reporting harness for the golden set. Per-intent accuracy is the primary metric.

Stub: the golden set does not exist yet, so this runs against any predictions frame
with columns [tweet_id, intent_true, intent_pred]. `python scripts/eval_report.py`
with no argument runs the built-in self-check.

Why the headline number is per-intent and not one blended figure: decisions.md #14.
The golden set deliberately over-samples ios_version_downgrade (~9x) and
out_of_scope (~10x) so each has enough rows to measure. A blended average over
those rows therefore weights two rare intents as ~18% of the score when they are
~3% of production traffic. It is reported, labeled, and never used as the headline.
"""
import sys

import pandas as pd

INTENTS = [
    "software_feature_defect",
    "general_complaint_nonactionable",
    "billing_account",
    "battery_drain",
    "non_english",
    "out_of_scope",
    "ios_version_downgrade",
]

# Estimated true corpus mix, from the unbiased 1,800-row random pool corrected by the
# hand-check confusion matrix (scripts/derive_prevalence.py, decisions.md #15).
# NOT derived from the golden set: that set deliberately over-samples rare intents, so
# weights taken from it would cancel the correction they exist to make.
PREVALENCE = {
    "software_feature_defect": 0.353,
    "general_complaint_nonactionable": 0.371,
    "billing_account": 0.087,
    "battery_drain": 0.113,
    "non_english": 0.022,
    "out_of_scope": 0.050,
    "ios_version_downgrade": 0.005,
}


def per_intent(df):
    """Primary metric: accuracy within each intent, plus its sampling weight."""
    rows = []
    for intent in INTENTS:
        sub = df[df["intent_true"] == intent]
        if not len(sub):
            rows.append({"intent": intent, "n": 0, "correct": 0,
                         "accuracy": float("nan"), "sample_share": 0.0,
                         "prod_share": PREVALENCE.get(intent, 0.0)})
            continue
        correct = int((sub["intent_pred"] == sub["intent_true"]).sum())
        rows.append({"intent": intent, "n": len(sub), "correct": correct,
                     "accuracy": correct / len(sub),
                     "sample_share": len(sub) / len(df),
                     "prod_share": PREVALENCE.get(intent, 0.0)})
    return pd.DataFrame(rows)


def blended(df):
    """Naive micro-average over golden-set rows. Misleading -- see module docstring."""
    return float((df["intent_pred"] == df["intent_true"]).mean())


def reweighted(tbl):
    """Blended accuracy with each intent reweighted to its production prevalence."""
    t = tbl[tbl["n"] > 0]
    w = t["prod_share"] / t["prod_share"].sum()
    return float((t["accuracy"] * w).sum())


def report(df):
    tbl = per_intent(df)
    print("=== PRIMARY: per-intent accuracy ===")
    print(f"{'intent':<34}{'n':>5}{'acc':>8}{'sample%':>10}{'prod%':>8}{'over':>7}")
    for r in tbl.itertuples():
        acc = "  n/a" if r.n == 0 else f"{100 * r.accuracy:5.1f}%"
        over = "-" if not r.prod_share or not r.n else f"{r.sample_share / r.prod_share:.1f}x"
        print(f"{r.intent:<34}{r.n:>5}{acc:>8}{100 * r.sample_share:>9.1f}%"
              f"{100 * r.prod_share:>7.1f}%{over:>7}")

    worst = tbl[tbl["n"] > 0].nsmallest(1, "accuracy")
    if len(worst):
        w = worst.iloc[0]
        print(f"\nweakest intent: {w['intent']} at {100 * w['accuracy']:.1f}% (n={int(w['n'])})")

    print("\n--- secondary, do NOT headline ---")
    print(f"naive blended accuracy   {100 * blended(df):5.1f}%   "
          f"(over-weights rare intents; see decisions.md #14)")
    print(f"prevalence-reweighted    {100 * reweighted(tbl):5.1f}%   "
          f"(closer to production, inherits bootstrap prevalence error)")
    return tbl


def demo():
    """Self-check: the two aggregates must diverge when a rare intent is weak."""
    rows = []
    # 100 rows of a common intent, all correct
    rows += [{"tweet_id": f"c{i}", "intent_true": "software_feature_defect",
              "intent_pred": "software_feature_defect"} for i in range(100)]
    # 20 rows of a heavily over-sampled rare intent, all wrong
    rows += [{"tweet_id": f"r{i}", "intent_true": "ios_version_downgrade",
              "intent_pred": "general_complaint_nonactionable"} for i in range(20)]
    df = pd.DataFrame(rows)
    tbl = report(df)

    assert tbl.set_index("intent").loc["software_feature_defect", "accuracy"] == 1.0
    assert tbl.set_index("intent").loc["ios_version_downgrade", "accuracy"] == 0.0
    nb, rw = blended(df), reweighted(tbl)
    assert abs(nb - 100 / 120) < 1e-9, nb
    # reweighting must pull the number UP: the failing intent is 17% of the sample
    # but only 1.2% of production. This gap is the whole reason for the rule.
    assert rw > nb, (rw, nb)
    assert rw > 0.98, rw
    print(f"\nself-check ok: naive {100 * nb:.1f}% vs reweighted {100 * rw:.1f}% "
          f"-- a 20-row rare-intent failure costs {100 * (rw - nb):.1f}pp of headline "
          f"it would not cost in production")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report(pd.read_pickle(sys.argv[1]))
    else:
        demo()
