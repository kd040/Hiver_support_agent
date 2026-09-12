"""Clean + tag AppleSupport clean 2-turn threads.

Placeholder anonymization, iOS 11.1 incident tagging, resolution_type classification.
Import the functions; run the module to rebuild data/processed/apple_threads.pkl.
"""
import os
import re
import pandas as pd
from profile_brands import load, clean_replies

BRAND, SEED, N = "AppleSupport", 0, 30
OUT = "data/processed/apple_threads.pkl"

MENTION = re.compile(r"@(\w+)")
LINK = re.compile(r"https?://\S+")
DM = re.compile(r"\bDMs?\b|\bDM'?d\b|direct message", re.I)
# Apple's DM deep-link, in both t.co forms seen in the data.
DM_LINKS = {"https://t.co/GDrqU22YpT", "https://t.co/GDrqU2kzhr"}
# Redirects to a support surface that is neither DM nor this account.
REDIRECT = re.compile(
    r"\bour [\w' ]{0,30}?(experts|specialists|engineers|team(?! up)\b"
    r"|support team|store team|sales support|billing support|application support)"
    r"|reach (out to )?(them|their team)|reach them (here|at)"
    r"|connect with our|get you (over )?to (our|the right)|get you in touch with"
    r"|pointed in the right direction|get you to the right place"
    r"|get help in \w+ here"
    r"|apple store|genius bar|make an appointment|give us a call|call us"
    r"|support\.apple\.com|getsupport|apple support (app|community)"
    r"|contact (your|the) (carrier|provider)",
    re.I,
)
# Generic template filler: apology, acknowledgment, call-to-action.
GENERIC = re.compile(
    r"(thanks|thank you) for (reaching out|contacting|bringing this|letting us know|your patience)( to us)?"
    r"|we'?(re| are) (here|happy|glad|sorry) (to help|for you|to hear)?"
    r"|we'?d (like|love) to (help|know more)|we (can|want to|would like to) (help|assist)"
    r"|(please )?(send us a|join us in|shoot us a|could you )?\bDM\b( us)?( via the following link)?"
    r"|send us a direct message|we'?ll (assist|help|continue|go from there|take it from there)"
    r"|(and )?we'?ll (continue|go) (from )?there|let us know|from there|we'?ve got your back"
    r"|sorry (to hear|about|for)|we apologize|our apologies|happy to help|here to help",
    re.I,
)
# Actual instructions, as opposed to acknowledgment.
INSTRUCT = re.compile(
    r"\bsettings\b|\btap\b|\bgo to\b|follow (the |these )?steps|try (this|that|these)"
    r"|restart|back up|turn (off|on)|toggle|reset|forget (those|the|this)"
    r"|update to|now available|workaround|make sure to|here'?s what you can do",
    re.I,
)
# The iOS 11.1 autocorrect incident. The leading alternative MUST stay anchored
# to "I": a bare ️ matches every VS16 emoji. See tests/test_cleaning.py.
BUG = re.compile(
    "I️"                      # the artifact itself: capital I + VS16
    "|autocorrec|keyboard"
    "|letter [\"“'‘]?i\\b"
    "|question mark|11\\.1\\.1",
    re.I,
)
WINDOW = ("2017-11-02", "2017-11-12")


def anonymize(text, brand_alias, brand_handles):
    """@AppleSupport and the brand's numeric alias -> @BRAND; other mentions -> @USER."""
    def sub(m):
        h = m.group(1)
        return "@BRAND" if (h == brand_alias or h in brand_handles) else "@USER"
    return re.sub(r"\s{2,}", " ", MENTION.sub(sub, text)).strip()


def strip_dm_links(text):
    """Drop the boilerplate DM deep-link so it cannot pass as answer content."""
    return re.sub(
        r"\s{2,}", " ", LINK.sub(lambda m: "" if m.group() in DM_LINKS else m.group(), text)
    ).strip()


def classify_resolution_type(reply):
    """Six-way label for how the brand closed the thread. Order matters."""
    txt = strip_dm_links(reply)
    if REDIRECT.search(txt):
        return "other_channel_redirect"
    if any(u not in DM_LINKS for u in LINK.findall(reply)):
        # A real content link survives. If the reply ALSO pushes to DM it is not
        # a self-contained resolution -- eval-labeling distinction only, never
        # a customer intent.
        return "answer_with_dm_followup" if DM.search(txt) else "self_contained_answer"
    if DM.search(txt) or txt != reply:          # says DM, or carried the DM deep-link
        return "dm_handoff"
    if "?" in txt and not INSTRUCT.search(txt):
        return "clarifying_question"
    core = GENERIC.sub("", txt)
    if INSTRUCT.search(txt) or len(re.findall(r"\w+", core)) >= 8:
        return "self_contained_answer"
    return "other"


def has_bug_keyword(text):
    return bool(BUG.search(text))


def in_incident_window(created_at):
    return WINDOW[0] <= str(pd.Timestamp(created_at).date()) <= WINDOW[1]


def tag_incident_window(text, created_at):
    """'ios_11_1_bug' if the message is about the incident OR lands in its window."""
    return "ios_11_1_bug" if has_bug_keyword(text) or in_incident_window(created_at) else None


def build():
    df = load("text", "created_at")
    brand_handles = set(df.loc[~df["inbound"], "author_id"].unique())

    t = clean_replies(df, BRAND).rename(columns={"text": "brand_text"})
    text, when = df.set_index("tweet_id")["text"], df.set_index("tweet_id")["created_at"]
    t["customer_text"] = text.reindex(t["parent_id"]).values
    t["created_at"] = pd.to_datetime(
        when.reindex(t["parent_id"]).values, format="%a %b %d %H:%M:%S %z %Y"
    ).tz_convert("UTC")

    # The brand is mentioned by handle (@AppleSupport) AND by an anonymized numeric
    # alias (@115858). Derive the alias: it is by far the most-mentioned numeric id
    # in messages addressed to this brand.
    nums = t["customer_text"].str.findall(r"@(\d+)").explode().dropna().value_counts()
    alias = nums.index[0]
    print(f"brand numeric alias for {BRAND}: @{alias} "
          f"({nums.iloc[0]:,} mentions; runner-up @{nums.index[1]} at {nums.iloc[1]:,})\n")

    for col in ("brand_text", "customer_text"):
        t[col] = t[col].map(lambda s: anonymize(s, alias, brand_handles))
    t["resolution_type"] = t["brand_text"].map(classify_resolution_type)
    t["bug_kw"] = t["customer_text"].map(has_bug_keyword)
    t["in_window"] = t["created_at"].map(in_incident_window)
    t["incident_window"] = None
    t.loc[t["bug_kw"] | t["in_window"], "incident_window"] = "ios_11_1_bug"
    return t


def main():
    t = build()
    bug_kw, in_win = t["bug_kw"], t["in_window"]
    print(f"clean {BRAND} threads: {len(t):,}")
    print(f"tagged ios_11_1_bug: {int((in_win | bug_kw).sum()):,} "
          f"({100 * (in_win | bug_kw).mean():.1f}%)  "
          f"[date-only {int((in_win & ~bug_kw).sum()):,}, "
          f"keyword-only {int((~in_win & bug_kw).sum()):,}, "
          f"both {int((in_win & bug_kw).sum()):,}]  none: {int((~(in_win | bug_kw)).sum()):,}")

    print("\n=== RESOLUTION TYPE ===")
    c = t["resolution_type"].value_counts()
    for k, n in c.items():
        print(f"  {k:<24} {n:>7,}  {100 * n / len(t):>5.1f}%")
    print("\n  by incident_window:")
    print(pd.crosstab(t["resolution_type"], t["incident_window"].fillna("none")).to_string())

    for cat in c.index:
        print(f"\n\n{'#' * 78}\n### {cat}  (15 random of {c[cat]:,})\n{'#' * 78}")
        for i, r in enumerate(t[t["resolution_type"] == cat].sample(15, random_state=SEED).itertuples(), 1):
            print(f"\n--- {i}  {r.created_at:%Y-%m-%d}  incident={r.incident_window}")
            print(f"  C: {r.customer_text}")
            print(f"  B: {r.brand_text}")

    # Rows whose only DM push is the bare deep-link, with no "DM" in the prose --
    # arguably belong in answer_with_dm_followup too. Surfaced, not reclassified.
    silent = (t["resolution_type"] == "self_contained_answer") & t["brand_text"].str.contains(
        "|".join(re.escape(u) for u in DM_LINKS))
    print(f"\n  self_contained_answer carrying a bare DM deep-link (no 'DM' in prose): {int(silent.sum()):,}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    t.to_pickle(OUT)
    print(f"\nwrote {OUT}  ({len(t):,} rows)")

    print("\n\n" + "=" * 78 + "\nSAME 30-THREAD SAMPLE, PLACEHOLDER ANONYMIZATION\n" + "=" * 78)
    for i, r in enumerate(t.sample(N, random_state=SEED).itertuples(), 1):
        print(f"\n[{i}/{N}] {r.created_at:%Y-%m-%d} [{r.resolution_type}] incident={r.incident_window}")
        print(f"  C: {r.customer_text}")
        print(f"  B: {r.brand_text}")


if __name__ == "__main__":
    main()
