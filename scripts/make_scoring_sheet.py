"""Build the human scoring sheet for judge-vs-human agreement (decisions.md #34).

Produces data/processed/human_scoring_sheet.csv: 36 of the 141 auto-handled drafts with
blank score columns for a human to fill in.

Two deliberate design choices:
  1. The judge's scores are NOT in the sheet. Showing them would anchor the human and
     inflate agreement, which is the one thing this measurement must not do.
  2. The sample is 24 random (stratified by intent) + 12 deliberately chosen hard cases
     (drafts the specificity flag caught, drafts that narrate, drafts the judge scored
     low). `sample_type` marks which is which, so agreement can be computed on the
     random subset alone -- the unbiased number -- as well as over all 36.
"""
import sys

import pandas as pd

JUDGE = "data/processed/judge_scores.pkl"
DECISIONS = "data/processed/escalation_decisions.pkl"
RETRIEVAL = "data/processed/retrieval_metric_embed.pkl"
SHEET = "data/processed/human_scoring_sheet.csv"
N_RANDOM, N_HARD, SEED = 24, 12, 0
DIMS = ["relevance", "groundedness", "tone", "correctness"]


def main():
    j = pd.read_pickle(JUDGE)
    dec = pd.read_pickle(DECISIONS)[["tweet_id", "unsupported", "quality_flags"]]
    j = j.merge(dec, on="tweet_id", how="left", suffixes=("", "_d"))
    unsup = j["unsupported_d"] if "unsupported_d" in j.columns else j["unsupported"]
    j["flagged"] = unsup.map(lambda x: bool(x) if isinstance(x, list) else False)
    j["narrates"] = j.quality_flags.map(
        lambda x: "narration" in x if isinstance(x, list) else False)
    j["low_judge"] = j[DIMS].min(axis=1) <= 3

    hard = j[j.flagged | j.narrates | j.low_judge]
    hard = hard.sample(min(N_HARD, len(hard)), random_state=SEED).assign(sample_type="hard")
    rest = j[~j.tweet_id.isin(hard.tweet_id)]
    rand = pd.concat([
        x.sample(max(1, round(N_RANDOM * len(x) / len(rest))), random_state=SEED)
        for _, x in rest.groupby("intent")]).assign(sample_type="random")
    sheet = pd.concat([rand, hard]).sample(frac=1, random_state=SEED)  # shuffle

    r = pd.read_pickle(RETRIEVAL)
    grounding = []
    for x in sheet.itertuples():
        if x.route != "grounded":
            grounding.append("(none - fixed policy statement or canned template)")
            continue
        m = r[(r.tweet_id == x.tweet_id) & (r.n_tweet_id == x.grounded_on)]
        grounding.append(
            f"past customer: {m.iloc[0].n_text}\n---\nApple's reply: {m.iloc[0].n_reply}"
            if len(m) else "(grounding row not found)")
    sheet["grounding_shown_to_drafter"] = grounding

    out = sheet[["tweet_id", "sample_type", "intent", "route",
                 "q_text", "grounding_shown_to_drafter", "final_draft"]].rename(
        columns={"q_text": "customer_message", "final_draft": "draft_reply"})
    for d in DIMS:
        out[f"h_{d}"] = ""          # 1-5, blank for the human to fill
    out["h_notes"] = ""
    out.to_csv(SHEET, index=False)

    print(f"wrote {SHEET}: {len(out)} rows to score")
    print(f"  random (unbiased)  {(out.sample_type == 'random').sum()}")
    print(f"  hard (deliberate)  {(out.sample_type == 'hard').sum()}")
    print(f"\nintent coverage: {out.intent.value_counts().to_dict()}")
    print(f"route coverage:  {out.route.value_counts().to_dict()}")
    print(f"\ncolumns to fill: {', '.join(f'h_{d}' for d in DIMS)}, h_notes")
    print("judge scores deliberately withheld from the sheet (anti-anchoring).")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
