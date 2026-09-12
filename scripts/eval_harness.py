"""End-to-end evaluation harness (deliverable 3, decisions.md #33).

Three components rolled up over the 200-row golden set:
  A  intent classification accuracy      (per-intent primary, blended secondary)
  B  escalation decision accuracy        (routing vs what the taxonomy says should happen)
  C  reply quality on auto-handled rows  (HUMAN scores on a 36-row subset)

Component C uses HUMAN scores only. The llama3.1:8b judge was measured against them and
failed (decisions.md #32, #33): linear-weighted kappa <= 0 on relevance, groundedness
and correctness. Its scores are still printed for transparency and are explicitly
labelled unreliable; they must not appear in any headline number.

Reported through the auto_handle DECOMPOSITION, never as a raw auto_handle rate -- see
decisions.md #31. auto_handle means "replied with no human involved", not "resolved".

TWO LIMITS ON WHAT B MEASURES, stated here because they are easy to miss:

1. golden_set.csv carries no "correct action" column, so the expected action is DERIVED
   from docs/taxonomy.md's per-intent "Good resolution" lines. The router implements the
   same taxonomy. B is therefore largely an IMPLEMENTATION-CONFORMANCE check -- does the
   code do what the doc says -- not a validity check on whether the doc is right. Only
   the disagreements carry information, so they are enumerated individually.
2. The battery physical-symptom exception reuses the router's own OVERRIDES regex, so
   those rows are checked against the rule that produced them and cannot disagree. They
   are counted separately as `circular`.
"""
import json
import os
import sys
import time

import pandas as pd

from draft_replies import OVERRIDES
from eval_report import report
from llm_relabel import TARGET, classify, resolve

GOLDEN = "data/processed/golden_set.pkl"
DECISIONS = "data/processed/escalation_decisions.pkl"
JUDGE = "data/processed/judge_scores.pkl"
HUMAN = "data/processed/human_scoring_sheet.csv"
CLS_CACHE = "data/processed/golden_pipeline_intent.jsonl"
OUT = "data/processed/eval_rollup.pkl"
DIMS = ["relevance", "groundedness", "tone", "correctness"]

# From docs/taxonomy.md "Good resolution" per intent. battery_drain is conditional:
# section 3 escalates on a physical symptom.
EXPECTED_ACTION = {
    "software_feature_defect": "auto_handle",     # self_contained_answer, best grounding
    "ios_version_downgrade": "auto_handle",       # fixed public answer, section 2
    "general_complaint_nonactionable": "auto_handle",  # ask one diagnostic question, section 5
    "battery_drain": "auto_handle",               # unless physical symptom -> escalate
    "billing_account": "escalate",                # section 4, by policy
    "non_english": "escalate",                    # section 6, always
    "out_of_scope": "escalate",                   # section 7 + conservative default, #30
}


def expected_action(intent, text):
    if intent == "battery_drain" and OVERRIDES["physical_or_hardware"].search(str(text)):
        return "escalate", True          # circular: same regex the router used
    return EXPECTED_ACTION[intent], False


def pipeline_intent(texts):
    """The real pipeline's classifier: rule gates + qwen, per decisions #21."""
    done = {}
    if os.path.exists(CLS_CACHE):
        with open(CLS_CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}
    todo = [(t, x) for t, x in texts if t not in done]
    if todo:
        print(f"classifying {len(todo)} golden rows through the pipeline", flush=True)
        t0 = time.time()
        with open(CLS_CACHE, "a") as f:
            for i, (tid, txt) in enumerate(todo, 1):
                lab, _ = classify(txt)
                rec = {"tweet_id": tid, "pred": resolve(txt, lab or TARGET)}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[tid] = rec
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return {t: done[t]["pred"] for t, _ in texts}


def main():
    g = pd.read_pickle(GOLDEN)
    g["tweet_id"] = g["tweet_id"].astype(str)
    dec = pd.read_pickle(DECISIONS)
    d = g.merge(dec[["tweet_id", "route", "action", "reason", "final_draft"]],
                on="tweet_id", how="left")

    # ---------- A: intent classification ----------
    preds = pipeline_intent(list(zip(d.tweet_id, d.customer_text)))
    d["intent_pred"] = d.tweet_id.map(preds)
    print("\n" + "=" * 78 + "\nA. INTENT CLASSIFICATION (rule gates + qwen2.5:3b)\n" + "=" * 78)
    report(d.rename(columns={"intent": "intent_true"})[
        ["tweet_id", "intent_true", "intent_pred"]])
    print("\nCAVEAT: retrieval pools and drafting used the GOLD intent, not this "
          "prediction,\nso components B and C do not inherit these errors. End-to-end "
          "numbers are\noptimistic by however much intent error would have propagated.")

    # ---------- B: escalation decision ----------
    exp = [expected_action(i, t) for i, t in zip(d.intent, d.customer_text)]
    d["expected_action"] = [a for a, _ in exp]
    d["circular"] = [c for _, c in exp]
    d["action_correct"] = d.action == d.expected_action
    print("\n" + "=" * 78 + "\nB. ESCALATION DECISION vs TAXONOMY-DERIVED EXPECTATION\n" + "=" * 78)
    n, ok = len(d), int(d.action_correct.sum())
    print(f"agreement: {ok}/{n} = {100 * ok / n:.1f}%   "
          f"({int(d.circular.sum())} rows circular, see module docstring)")
    print(f"\n{pd.crosstab(d.expected_action, d.action).to_string()}")
    bad = d[~d.action_correct]
    print(f"\nthe {len(bad)} disagreements (the only rows carrying information):")
    for x in bad.itertuples():
        print(f"  [{x.intent}] expected {x.expected_action}, got {x.action}"
              f" via {x.route}")
        print(f"      reason: {x.reason}")

    # ---------- C: reply quality ----------
    print("\n" + "=" * 78 + "\nC. REPLY QUALITY on auto-handled rows (HUMAN scores)\n" + "=" * 78)
    if os.path.exists(HUMAN):
        hs = pd.read_csv(HUMAN, dtype={"tweet_id": str})
        for dim in DIMS:
            hs[f"h_{dim}"] = pd.to_numeric(hs[f"h_{dim}"], errors="coerce")
        rnd = hs[hs.sample_type == "random"]
        print(f"HEADLINE: human-scored, {len(rnd)} randomly sampled auto-handled drafts")
        print(f"{'dimension':<16}{'mean':>7}{'median':>8}{'>=4':>7}{'<=2':>7}{'n':>5}")
        for dim in DIMS:
            v = rnd[f"h_{dim}"].dropna()
            print(f"{dim:<16}{v.mean():>7.2f}{v.median():>8.1f}"
                  f"{100 * (v >= 4).mean():>6.0f}%{100 * (v <= 2).mean():>6.0f}%{len(v):>5}")
        print(f"\nincluding the 12 deliberately-hard rows (n={len(hs)}), for contrast:")
        for dim in DIMS:
            v = hs[f"h_{dim}"].dropna()
            print(f"  {dim:<16}{v.mean():>6.2f}")
        print("\nCAVEAT: 24 random rows of 141 auto-handled, one annotator, so each mean\n"
              "carries roughly +/- 0.2-0.3. It is the only reply-quality number in this\n"
              "harness backed by a human, which is why it is the headline despite the n.")
    if os.path.exists(JUDGE):
        print("\n--- LLM judge scores: RETAINED FOR TRANSPARENCY, NOT RELIABLE ---")
        print("kappa_w <= 0 vs human on relevance/groundedness/correctness (#32).")
        j = pd.read_pickle(JUDGE)
        print(f"{'dimension':<16}{'mean':>7}{'median':>8}{'>=4':>7}{'<=2':>7}")
        for dim in DIMS:
            s = j[dim].dropna()
            print(f"{dim:<16}{s.mean():>7.2f}{s.median():>8.1f}"
                  f"{100 * (s >= 4).mean():>6.0f}%{100 * (s <= 2).mean():>6.0f}%")
        print("\nby route:")
        print(j.groupby("route")[DIMS].mean().round(2).to_string())
        # independent cross-check: judge groundedness vs the deterministic flag
        flags = pd.read_pickle(DECISIONS)[["tweet_id", "unsupported"]].copy()
        flags["flagged"] = flags.unsupported.map(
            lambda x: bool(x) if isinstance(x, list) else False)
        jj = j.merge(flags[["tweet_id", "flagged"]], on="tweet_id", how="left")
        print(f"\ncross-check, judge groundedness vs the deterministic specificity flag:")
        print(f"  flagged rows    mean groundedness {jj[jj.flagged].groundedness.mean():.2f}"
              f"  (n={int(jj.flagged.sum())})")
        print(f"  unflagged rows  mean groundedness {jj[~jj.flagged].groundedness.mean():.2f}"
              f"  (n={int((~jj.flagged).sum())})")
        print(f"  -> the flag finds {int(jj.flagged.sum())} drafts with unsupported "
              f"specifics; if the judge\n     does not score those lower, its "
              f"groundedness dimension is not measuring groundedness.")
        print("\ndistinct scores actually used per dimension (a 5-point scale the judge "
              "collapses\nto 2-3 anchors is not a 5-point scale):")
        for dim in DIMS:
            print(f"  {dim:<16}{sorted(j[dim].dropna().unique().tolist())}")
    else:
        print(f"{JUDGE} not present - run scripts/judge_replies.py first")

    # ---------- framing ----------
    print("\n" + "=" * 78 + "\nAUTO_HANDLE DECOMPOSITION (the framing, not the raw rate)\n" + "=" * 78)
    a = d[d.action == "auto_handle"]
    rows = [
        ("answered from retrieved grounding", (a.route == "grounded") &
         (a.intent == "software_feature_defect")),
        ("asked a diagnostic question, resolves nothing",
         a.intent == "general_complaint_nonactionable"),
        ("answered from a stated policy fact", a.intent == "ios_version_downgrade"),
        ("emitted an identical canned template", a.intent == "battery_drain"),
    ]
    for label, mask in rows:
        print(f"  {label:<48}{int(mask.sum()):>4}")
    answered = int(((a.route == "grounded") & (a.intent == "software_feature_defect")).sum()
                   + (a.intent == "ios_version_downgrade").sum()
                   + (a.intent == "battery_drain").sum())
    print(f"\n  auto_handle total {len(a)}/{n} ({100 * len(a) / n:.1f}%) -- but that is "
          f"'replied without a human',\n  NOT 'resolved'. Rows receiving something "
          f"resembling an answer: {answered}/{n} ({100 * answered / n:.0f}%),\n  of which "
          f"{int((a.intent == 'battery_drain').sum())} are an identical template.")
    d.to_pickle(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
