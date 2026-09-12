"""Step 5: escalation decision layer (decisions.md #30).

Formalization, not new logic. Every signal consolidated here is already produced by
scripts/draft_replies.py; this maps each route to auto_handle vs escalate and attaches
the stated reason the assignment requires. Reason labels are REUSED from the routing
layer rather than reinvented, so a reason here can always be traced to the check that
fired it.

IMPORTANT FRAMING, required wherever the auto_handle rate is quoted: auto_handle means
"a reply went out with no human involved", NOT "the case was resolved". 45 of the 141
auto-handled rows are general_complaint_nonactionable, where the correct reply is a
diagnostic question (taxonomy section 5) that resolves nothing by design. Quoting 70.5%
as a resolution rate would overstate the system by roughly a third of its own output.

Mapping (route -> action):
  escalate_override    escalate    taxonomy sections 3/4/7 hard safety overrides
  no_draft_policy      escalate    non_english (section 6), out_of_scope (section 7)
  no_usable_grounding  escalate    zero of top-5 retrieval candidates passed the gates
  grounded             auto_handle drafted from gated grounding
  policy_fact          auto_handle answered from a stated fact or canned template
"""
import sys

import pandas as pd

from draft_replies import NO_DRAFT, TEMPLATES, draft_quality_flags

DRAFTS = "data/processed/drafts.pkl"
OUT = "data/processed/escalation_decisions.pkl"
CSV = "data/processed/escalation_decisions.csv"

ESCALATE_ROUTES = {"escalate_override", "no_draft_policy", "no_usable_grounding"}


def escalation_decision(row):
    """-> (action, reason). Reason is never empty for an escalated row."""
    route = row["route"]

    if route == "escalate_override":
        # reuse the override labels emitted by draft_replies.check_overrides
        return "escalate", f"safety_override:{row['override']}"
    if route == "no_draft_policy":
        # reuse the policy string already generated per intent
        return "escalate", NO_DRAFT[row["intent"]]
    if route == "no_usable_grounding":
        return "escalate", ("no_usable_grounding: zero of top-5 retrieval candidates "
                            "passed gates G1-G4")
    if route == "policy_fact":
        kind = "canned_template" if row["intent"] in TEMPLATES else "stated_policy_fact"
        return "auto_handle", f"{kind}:{row['intent']}"
    if route == "grounded":
        return "auto_handle", (f"grounded:thread={row['grounded_on']},"
                               f"rank={int(row['grounding_rank'])},"
                               f"gated_score={row['gated_score']:.2f}")
    raise ValueError(f"unmapped route: {route}")


def main():
    d = pd.read_pickle(DRAFTS)
    decisions = [escalation_decision(r) for _, r in d.iterrows()]
    d["action"] = [a for a, _ in decisions]
    d["reason"] = [r for _, r in decisions]
    d["quality_flags"] = d.draft.map(draft_quality_flags)
    # escalated rows carry no customer-facing draft
    d["final_draft"] = d.draft.where(d.action == "auto_handle")

    cols = ["tweet_id", "intent", "route", "action", "reason", "final_draft",
            "unsupported", "quality_flags", "n_usable", "chars"]
    if "sub_stratum" in d.columns:
        cols.insert(2, "sub_stratum")
    out = d[cols].copy()
    out.to_pickle(OUT)
    out.to_csv(CSV, index=False)

    n = len(d)
    auto = (d.action == "auto_handle").sum()
    print(f"=== escalation decisions over {n} rows ===")
    print(f"  auto_handle {auto:>4}  ({100 * auto / n:.1f}%)")
    print(f"  escalate    {n - auto:>4}  ({100 * (n - auto) / n:.1f}%)")
    diag = ((d.action == "auto_handle") & (d.intent == "general_complaint_nonactionable")).sum()
    print(f"\n  NOTE: auto_handle means REPLIED WITHOUT A HUMAN, not RESOLVED.")
    print(f"  {diag} of {auto} auto-handled rows ({100 * diag / auto:.0f}%) are "
          f"general_complaint_nonactionable,")
    print(f"  where the correct reply is a diagnostic question that resolves nothing.")
    print(f"  Do not quote {100 * auto / n:.1f}% as a resolution rate.")

    print(f"\nby route:")
    print(pd.crosstab(d.route, d.action).to_string())
    print(f"\nby intent:")
    print(pd.crosstab(d.intent, d.action).to_string())

    esc = d[d.action == "escalate"]
    missing = esc[esc.reason.isna() | (esc.reason.astype(str).str.strip() == "")]
    print(f"\nescalated rows with a non-empty stated reason: "
          f"{len(esc) - len(missing)}/{len(esc)}")
    assert len(missing) == 0, f"{len(missing)} escalated rows lack a reason"
    print("stated reasons in use:")
    for r, c in esc.reason.str.split(":").str[0].value_counts().items():
        print(f"  {r:<30}{c:>4}")

    print(f"\nauto_handle rows carrying a draft: "
          f"{d[d.action == 'auto_handle'].final_draft.notna().sum()}/{auto}")
    print(f"escalated rows with final_draft null (as required): "
          f"{d[d.action == 'escalate'].final_draft.isna().sum()}/{len(esc)}")
    q = d[d.quality_flags.map(bool)]
    print(f"\ndraft quality flags (length/narration): {len(q)}/{n}")
    for x in q.itertuples():
        print(f"  [{x.action}] {x.intent}: {x.quality_flags}")
    print(f"\nwrote {OUT} and {CSV}")
    return out


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
