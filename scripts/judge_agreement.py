"""Judge-vs-human agreement, once the scoring sheet is filled in (decisions.md #34).

Usage: python scripts/judge_agreement.py [path-to-filled-sheet.csv]

Reports, per dimension: exact agreement, agreement within 1 point, linear-weighted
Cohen's kappa, and the judge's mean minus the human's mean (its leniency bias).
Computed separately for the `random` rows -- the unbiased estimate -- and for all rows.
"""
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

JUDGE = "data/processed/judge_scores.pkl"
DEFAULT = "data/processed/human_scoring_sheet.csv"
DIMS = ["relevance", "groundedness", "tone", "correctness"]


def block(m, label):
    print(f"\n=== {label} (n={len(m)}) ===")
    print(f"{'dimension':<16}{'exact':>8}{'within1':>9}{'kappa_w':>9}"
          f"{'judge':>8}{'human':>8}{'bias':>7}")
    for d in DIMS:
        sub = m[[f"j_{d}", f"h_{d}"]].dropna()
        if len(sub) < 2:
            print(f"{d:<16}   insufficient scored rows")
            continue
        a, b = sub[f"j_{d}"].astype(int), sub[f"h_{d}"].astype(int)
        exact = (a == b).mean()
        within = ((a - b).abs() <= 1).mean()
        try:
            k = cohen_kappa_score(a, b, weights="linear")
        except ValueError:
            k = float("nan")
        print(f"{d:<16}{100 * exact:>7.0f}%{100 * within:>8.0f}%{k:>9.2f}"
              f"{a.mean():>8.2f}{b.mean():>8.2f}{a.mean() - b.mean():>+7.2f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    h = pd.read_csv(path, dtype={"tweet_id": str})
    j = pd.read_pickle(JUDGE)[["tweet_id"] + DIMS].copy()
    j["tweet_id"] = j.tweet_id.astype(str)
    j = j.rename(columns={d: f"j_{d}" for d in DIMS})
    m = h.merge(j, on="tweet_id", how="left")
    for d in DIMS:
        m[f"h_{d}"] = pd.to_numeric(m[f"h_{d}"], errors="coerce")

    scored = m[[f"h_{d}" for d in DIMS]].notna().any(axis=1).sum()
    print(f"sheet: {len(m)} rows, {scored} with at least one human score")
    if not scored:
        print("nothing scored yet - fill in the h_* columns and re-run")
        return
    block(m[m.sample_type == "random"], "RANDOM rows only (unbiased estimate)")
    block(m, "ALL rows (includes deliberately-hard cases)")
    print("\nkappa_w is linear-weighted Cohen's kappa. bias = judge mean - human mean;")
    print("positive means the judge is more generous than the human.")


if __name__ == "__main__":
    main()
