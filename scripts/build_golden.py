"""Assemble the 200-row golden eval set from hand-checked candidates (decisions.md #14).

Every row here was confirmed by a hand-check pass; the record of accepts and rejects
is data/processed/golden_handcheck.pkl. blind_spot rows carry intent
general_complaint_nonactionable with sub_stratum="blind_spot" -- they are residual
members that nothing in the pipeline detects (taxonomy.md "Still not in scope").
"""
import sys

import pandas as pd

HC = "data/processed/golden_handcheck.pkl"
BACKFILL = "data/processed/golden_residual_backfill.pkl"
OUT = "data/processed/golden_set.pkl"
CSV = "data/processed/golden_set.csv"
SEED = 0
RESID = "general_complaint_nonactionable"

# rows accepted in the residual backfill hand-check (indices into BACKFILL order)
BACKFILL_ACC = [2, 5, 7, 9, 13, 16, 19, 20, 27, 29, 30, 37, 39, 40, 41, 44, 45, 48, 49]

TARGETS = {                                  # intent -> (n, max bug_kw)
    "software_feature_defect": (67, 10),
    RESID: (42, 6),
    "blind_spot": (12, 2),                   # folded into RESID, 54 total, <=8 bug
    "billing_account": (20, None),
    "battery_drain": (16, None),
    "ios_version_downgrade": (18, None),
    "out_of_scope": (18, None),
    "non_english": (7, None),
}
OOS_PER_KIND = 6                             # warranty / phishing / meta; praise is empty


def take(df, n, max_bug):
    """Draw n rows, honouring the bug_kw ceiling from decisions.md #6."""
    if max_bug is None:
        return df.sample(min(n, len(df)), random_state=SEED)
    bug, clean = df[df.bug_kw], df[~df.bug_kw]
    nb = min(max_bug, len(bug), n)
    nc = min(n - nb, len(clean))
    return pd.concat([bug.sample(nb, random_state=SEED),
                      clean.sample(nc, random_state=SEED)])


def main():
    h = pd.read_pickle(HC)
    ok = h[h.accepted].copy()

    bf = pd.read_pickle(BACKFILL).reset_index(drop=True)
    bf = bf.iloc[BACKFILL_ACC].copy()
    bf["stratum"] = RESID
    bf["accepted"] = True
    ok = pd.concat([ok, bf[["tweet_id", "stratum", "bug_kw", "accepted", "customer_text"]]],
                   ignore_index=True)

    picks = []
    defect = ok[ok.stratum.str.startswith("software_feature_defect")]
    picks.append(take(defect, *TARGETS["software_feature_defect"]).assign(
        intent="software_feature_defect", sub_stratum=""))
    picks.append(take(ok[ok.stratum == RESID], *TARGETS[RESID]).assign(
        intent=RESID, sub_stratum=""))
    picks.append(take(ok[ok.stratum == "blind_spot"], *TARGETS["blind_spot"]).assign(
        intent=RESID, sub_stratum="blind_spot"))
    for st in ["billing_account", "battery_drain", "ios_version_downgrade", "non_english"]:
        picks.append(take(ok[ok.stratum == st], *TARGETS[st]).assign(intent=st, sub_stratum=""))
    for kind in ["warranty", "phishing", "meta"]:
        sub = ok[ok.stratum == f"out_of_scope/{kind}"]
        picks.append(sub.sample(min(OOS_PER_KIND, len(sub)), random_state=SEED).assign(
            intent="out_of_scope", sub_stratum=kind))

    g = pd.concat(picks, ignore_index=True)
    g = g[["tweet_id", "intent", "sub_stratum", "bug_kw", "customer_text"]]
    assert g.tweet_id.is_unique, "duplicate tweet_id in golden set"
    g.to_pickle(OUT)
    g.to_csv(CSV, index=False)

    print(f"golden set: {len(g)} rows -> {OUT}")
    print(f"\n{'intent':<34}{'n':>5}{'share':>8}{'bug_kw':>8}")
    for k, n in g.intent.value_counts().items():
        b = int(g[g.intent == k].bug_kw.sum())
        print(f"{k:<34}{n:>5}{100 * n / len(g):>7.1f}%{b:>6} ({100 * b / n:.0f}%)")
    print(f"\ntotal bug_kw: {int(g.bug_kw.sum())} ({100 * g.bug_kw.mean():.1f}%)")
    resid = g[g.intent == RESID]
    print(f"residual block: {len(resid)} ({int((resid.sub_stratum == 'blind_spot').sum())} "
          f"blind_spot), bug_kw {int(resid.bug_kw.sum())} ({100 * resid.bug_kw.mean():.1f}%)")
    print("out_of_scope kinds:", g[g.intent == "out_of_scope"].sub_stratum.value_counts().to_dict())
    return g


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
