"""Second independent judgment on the general/defect boundary (decisions.md #18).

The golden set is single-annotator. This measures how far a second judgment diverges
on the two intents that carry the boundary: general_complaint_nonactionable (54) and
software_feature_defect (67).

IMPORTANT non-independence: the golden labels for these strata are rows where the
annotator AGREED with a qwen candidate label, so re-running the ORIGINAL few-shot
prompt at temperature 0 would reproduce itself and report ~100% agreement as an
artifact. This pass therefore uses a different procedure -- the intent definitions and
examples lifted from docs/taxonomy.md rather than the hand-picked few-shot block in
llm_relabel.py -- and applies none of the #12/#13 post-rules. Same model, so errors
are still correlated; this is a floor on disagreement, not a human reliability study.
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd

MODEL = "qwen2.5:3b"
URL = "http://localhost:11434/api/chat"
GOLDEN = "data/processed/golden_set.pkl"
CACHE = "data/processed/second_pass_cache.jsonl"
OUT = "data/processed/second_pass.pkl"
DISAGREE_CSV = "data/processed/second_pass_disagreements.csv"
BOUNDARY = ["general_complaint_nonactionable", "software_feature_defect"]

INTENTS = ["software_feature_defect", "ios_version_downgrade", "battery_drain",
           "billing_account", "general_complaint_nonactionable", "non_english",
           "out_of_scope"]

# Definitions transcribed from docs/taxonomy.md, deliberately worded as the doc words
# them rather than as llm_relabel.py's prompt words them.
SYSTEM = """You are labelling customer messages sent to Apple Support with exactly one intent.

1. software_feature_defect
A SPECIFIC NAMED COMPONENT plus a SYMPTOM. The named component is what separates this
from a vague complaint. "my phone is broken" is NOT this intent; "Siri doesn't work" is.
Generic device words -- phone, iPhone, iPad, device, update, iOS -- are NOT components.
Examples: "This time Siri doesn't work"; "my phone deleted some of my contacts & wont
let me add them back"; "my iPhone X keyboard is laggy/unresponsive in cold weather".

2. ios_version_downgrade
An explicit request to revert, roll back, or reinstall a PREVIOUS iOS/macOS version.
Examples: "Can I revert to 10?"; "How do I get iOS 10 back on my 6s?"; "Let me go back
to ios 10 plz". Switching to Android is NOT this. A failed install is NOT this.

3. battery_drain
Battery life, drain rate, charge cycles, or charging.
Examples: "draining my 7 plus battery like crazy. Went to bed at 85%, woke up at 11%";
"any way to check the overall battery health of my iPhone 5s?".

4. billing_account
Purchases, iTunes, Apple Music, subscriptions, refunds, Apple ID, orders, pre-orders,
account access.
Examples: "payment has been taken but Apple Music is saying I'm not subscribed";
"won't let me log in I want to cancel subscriptions".

5. general_complaint_nonactionable
Frustration about an update, a device, or the brand with NO specific component and NO
actionable request. The residual bucket by design.
Examples: "when y'all gonna fix my phone... It don't need to be messin' up"; "I am
absolutely fuming with your technology"; "This has been the worst experience moving to
iPhone X".

6. non_english
The message is written in a language other than English. Slang, dialect, ALL CAPS,
profanity, typos and emoji are STILL ENGLISH.

7. out_of_scope
English and coherent, but not an actionable support request: warranty/policy/pricing
questions, phishing or scam reports, praise with no request, or meta-commentary about
Twitter or the brand.
Examples: "has Apple started offering International Warranty on iPhones?"; "I just got
this scam text message, thought you should know".

Decide between 1 and 5 by asking: is a specific component or feature named? If yes, 1.
If only the device or the update in general is named, 5.
Reply with the intent name alone."""


def classify(text, timeout=180):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 12},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    got = raw.strip().strip(".`\"' \n").lower()
    for i in INTENTS:
        if i in got:
            return i, raw.strip()
    return None, raw.strip()


def main():
    g = pd.read_pickle(GOLDEN)
    rows = g[g.intent.isin(BOUNDARY)].copy()
    print(f"second pass over {len(rows)} boundary rows "
          f"({(rows.intent == BOUNDARY[0]).sum()} residual, "
          f"{(rows.intent == BOUNDARY[1]).sum()} defect)")

    done = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}
    t0 = time.time()
    with open(CACHE, "a") as f:
        for n, r in enumerate(rows.itertuples(), 1):
            if r.tweet_id in done:
                continue
            lab, raw = classify(r.customer_text)
            rec = {"tweet_id": r.tweet_id, "second": lab, "raw": raw}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[r.tweet_id] = rec
            if n % 40 == 0:
                print(f"  {n}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    rows["second"] = rows.tweet_id.map(lambda i: done.get(i, {}).get("second"))
    rows["agree"] = rows.second == rows.intent
    rows.to_pickle(OUT)

    print(f"\nsimple agreement: {rows.agree.sum()}/{len(rows)} = "
          f"{100 * rows.agree.mean():.1f}%")
    print(f"\nby golden label:")
    for k in BOUNDARY:
        s = rows[rows.intent == k]
        print(f"  {k:<34}{s.agree.sum():>3}/{len(s):<4}{100 * s.agree.mean():>6.1f}%")
    print(f"\nwhere the second pass disagreed, what it said instead:")
    dis = rows[~rows.agree]
    print(pd.crosstab(dis.intent, dis.second).to_string())
    print(f"\nblind_spot sub-stratum (independent of qwen, keyword-sourced):")
    bs = rows[rows.sub_stratum == "blind_spot"]
    if len(bs):
        print(f"  {bs.agree.sum()}/{len(bs)} = {100 * bs.agree.mean():.1f}%")
    dis[["tweet_id", "intent", "second", "sub_stratum", "customer_text"]].to_csv(
        DISAGREE_CSV, index=False)
    print(f"\nwrote {len(dis)} disagreements -> {DISAGREE_CSV} (for hand re-judgement)")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
