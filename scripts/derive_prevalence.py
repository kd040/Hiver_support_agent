"""Estimate true corpus intent prevalence from the unbiased random pool.

Prevalence CANNOT come from the golden set: it deliberately over-samples
ios_version_downgrade ~9x and out_of_scope ~10x (decisions.md #14), so weights
derived from it would cancel the very correction they exist to make.

The 1,800-row random pool IS unbiased w.r.t. the deduped corpus. Within each qwen
label the hand-checked rows are a random subsample, so they estimate
P(true | predicted); prevalence(k) = sum_j share(j) * P(true=k | pred=j).

Precision alone is NOT enough for this and using it alone was an error: a label can
have poor precision while still being under-counted, because rows flow both ways.
"""
import pandas as pd

POOL = "data/processed/golden_random_pool_labeled.pkl"
INTENTS = ["software_feature_defect", "general_complaint_nonactionable",
           "billing_account", "battery_drain", "non_english", "out_of_scope",
           "ios_version_downgrade"]
RESID = "general_complaint_nonactionable"

# Hand-check outcomes as counts of TRUE label per PREDICTED label.
# blind_spot rows are residual members (taxonomy.md "Still not in scope"), so they
# are counted as general_complaint_nonactionable here.
CONFUSION = {
    "software_feature_defect": {          # 109 checked (91 + 18 backfill)
        "software_feature_defect": 76, "battery_drain": 7, RESID: 26},
    RESID: {                              # 107 checked (57 + 50 backfill)
        RESID: 63, "software_feature_defect": 26, "out_of_scope": 8,
        "billing_account": 5, "battery_drain": 5},
    "billing_account": {                  # 27 checked
        "billing_account": 22, "battery_drain": 1,
        "software_feature_defect": 2, RESID: 2},
    "battery_drain": {                    # 22 checked
        "battery_drain": 20, RESID: 2},
}
# Not hand-checked inside the pool; precision measured on their own corpus-wide
# detector pools instead, with the miss mass assigned to residual.
EXTERNAL_PRECISION = {"out_of_scope": 32 / 60, "ios_version_downgrade": 27 / 45}
# non_english is taken from the brand's own language-redirect marker rather than the
# pool's 6 rows: 1,084 marked threads of 48,199 after exact dedup.
NON_ENGLISH_DIRECT = 1084 / 48199


def main():
    p = pd.read_pickle(POOL)
    share = (p["intent_est"].value_counts() / len(p)).to_dict()
    print(f"pool {len(p)} rows; predicted shares:")
    for k in INTENTS:
        print(f"  {k:<34}{100 * share.get(k, 0):>6.1f}%")

    prev = {k: 0.0 for k in INTENTS}
    for pred, s in share.items():
        if pred in CONFUSION:
            tot = sum(CONFUSION[pred].values())
            for true, n in CONFUSION[pred].items():
                prev[true] += s * n / tot
        elif pred in EXTERNAL_PRECISION:
            prec = EXTERNAL_PRECISION[pred]
            prev[pred] += s * prec
            prev[RESID] += s * (1 - prec)
        # pool's non_english prediction is dropped; overridden below
    prev["non_english"] = NON_ENGLISH_DIRECT

    tot = sum(prev.values())
    prev = {k: v / tot for k, v in prev.items()}

    N500 = {"software_feature_defect": .392, RESID: .322, "billing_account": .120,
            "battery_drain": .096, "non_english": .040, "out_of_scope": .018,
            "ios_version_downgrade": .012}
    print(f"\n{'intent':<34}{'n=500':>9}{'corrected':>11}{'delta':>9}")
    for k in INTENTS:
        d = 100 * (prev[k] - N500[k])
        print(f"{k:<34}{100 * N500[k]:>8.1f}%{100 * prev[k]:>10.1f}%{d:>+8.1f}pp")
    print("\nPREVALENCE = {")
    for k in INTENTS:
        print(f'    "{k}": {prev[k]:.3f},')
    print("}")
    return prev


if __name__ == "__main__":
    main()
