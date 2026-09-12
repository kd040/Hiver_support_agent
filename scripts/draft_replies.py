"""Step 4: reply drafting, grounded in gated retrieval (decisions.md #26).

Designed around the measurements rather than around the happy path:

- Retrieval hands over the TOP-5 with gate flags, not the argmax. G3 topic mismatch is
  29.3% at rank 0 (#25), so the nearest neighbour is the wrong default.
- Grounding is the candidate with the highest GATED score, which is frequently not
  rank 0.
- If zero of the top-5 pass all gates, no draft is forced. The row is flagged
  no_usable_grounding and becomes an input to the escalation step, which is NOT built
  here.
- llama3.1:8b, not qwen2.5:3b: this is generation, not classification.

Two intents never retrieve (#11, taxonomy section 7):
  ios_version_downgrade  -- answered from a stated policy fact, which is the one case
                            where the pipeline deliberately bypasses retrieval.
  out_of_scope           -- taxonomy section 7 forbids auto-drafting from retrieval.
And non_english is always-escalate with no reply logic (taxonomy section 6), so it is
routed without a generation call.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

MODEL = "llama3.1:8b"
URL = "http://localhost:11434/api/chat"
RETRIEVAL = "data/processed/retrieval_metric_embed.pkl"
GOLDEN = "data/processed/golden_set.pkl"
CACHE = "data/processed/draft_cache.jsonl"
OUT = "data/processed/drafts.pkl"
NO_GROUNDING = "escalate: no grounding found"
ESCALATE_OVERRIDE = "escalate: safety/policy override"
RESID = "general_complaint_nonactionable"

# What a good reply looks like per intent, from docs/taxonomy.md's "Good resolution".
TARGET_REPLY = {
    "software_feature_defect": "a self-contained answer: the known workaround or the "
                               "settings steps for the specific component named",
    "battery_drain": "a self-contained answer: battery-health check steps and the "
                     "standard settings guidance",
    RESID: "ONE good diagnostic question. Do not attempt a fix - the customer has not "
           "said what is actually wrong. Ask what specifically is failing",
    "billing_account": "a polite redirect to a private channel. Never attempt the "
                       "account action itself and never ask for account details in public",
    "ios_version_downgrade": "an honest self-contained answer stating the position",
}

# Intents answered from a stated fact rather than a retrieved neighbour.
# ios_version_downgrade: #11 -- the answer is public, fixed and customer-independent.
# battery_drain: #28 -- its grounding pool is overwhelmingly bare links, and with those
#   stripped the model invented specifics in 86% of drafts, 11 of them naming Battery
#   Health, which shipped in iOS 11.3 beta and does not exist for a corpus centred on
#   11.0-11.2. Every step below is version-checked against that window:
#     Settings > Battery usage list      iOS 8+
#     Low Power Mode                     iOS 9+
#     Background App Refresh             iOS 7+
#     Settings > General > Software Update  long-standing
#   Deliberately excluded: Battery Health (11.3+), Optimised Battery Charging (13+).
# Canned templates: emitted verbatim, NO generation call. Used where the correct reply
# is fixed and customer-independent AND the model proved unable to follow a stated
# procedure (#29). battery_drain is here because three escalating prompt constraints
# all failed to stop it citing Battery Health (11.3+) and, worse, inverting Low Power
# Mode to "turned off" in 5 of 6 drafts that mentioned it.
TEMPLATES = {
    "battery_drain": (
        "Settings > Battery shows which apps have used the most battery - worth starting "
        "there. Turning Low Power Mode on helps, and so does turning off Background App "
        "Refresh for apps that don't need it. It's also worth checking Settings > General "
        "> Software Update for a pending update."
    ),
}

POLICY_FACTS = {
    "ios_version_downgrade": (
        "Downgrading is not supported once Apple stops signing the previous iOS build. "
        "Once signing ends there is no supported way to reinstall an older version, and "
        "the answer does not vary by customer or device."
    ),
}

# Intents routed without any generation call.
NO_DRAFT = {
    "non_english": "escalate: language redirect (taxonomy section 6, always escalate)",
    "out_of_scope": "escalate: out of scope, no retrieval grounding permitted "
                    "(taxonomy section 7)",
}

# --- fix 1: grounding is stripped of URLs before it is ever shown to the model ---
_URL = re.compile(r"https?://\S+|\bt\.co/\S+", re.I)


def strip_urls(text):
    return _URL.sub("[link removed]", str(text)).strip()


def has_link(text):
    return bool(_URL.search(str(text)))


# --- fix 2: hard escalate-anyway overrides, checked on the CUSTOMER MESSAGE and
# applied AHEAD of intent routing. Any hit escalates regardless of what retrieval
# found, overriding the "grounded" route entirely. Sources: taxonomy.md section 3
# (physical battery symptoms are a hardware safety path, not a settings answer),
# section 4 (account and payment actions are not safe to auto-handle), section 7
# (phishing reports).
OVERRIDES = {
    "physical_or_hardware": re.compile(
        r"\b(swell\w*|swollen|bulg\w*|overheat\w*|too hot|burn\w*|smok\w*|melt\w*|"
        r"spark\w*|crack\w*|shatter\w*|smash\w*|broken screen|screen (is )?(broke|crack)|"
        r"water damage|liquid damage|dropped (it|my|the)|went black|gone black|"
        r"(wont|won.t|not|isn.t|doesn.t) charg\w*|refus\w* to charge|"
        r"physical damage|bent|snapped)\b", re.I),
    "account_or_payment_action": re.compile(
        r"\b(refund|charge(d)? me|charging me|unauthoris?zed charge|double charged|"
        r"cancel (my )?(subscription|order)|change (my )?(payment|card|billing)|"
        r"payment method|reset (my )?password|apple id password|hacked|"
        r"locked out|can.t log ?in|cannot log ?in|billed)\b", re.I),
    "phishing_or_scam": re.compile(
        r"\b(scam|phish\w*|fake (email|text|message|website|site|link)|fraud\w*|"
        r"suspicious (email|text|message|link)|is this (real|legit|genuine|from you))\b",
        re.I),
}


def check_overrides(text):
    return [k for k, rx in OVERRIDES.items() if rx.search(str(text))]


# --- fix 3: post-generation specificity check. Flags, never rejects -- the rate is
# reported as a number rather than suppressed.
# Features that postdate the 11.0-11.2 corpus window. Flagged unconditionally, never
# via grounding comparison -- see the note in the era-guard prompt block.
BANNED_ERA_TERMS = ["Battery Health", "Optimised Battery Charging",
                    "Optimized Battery Charging", "Screen Time", "Find My",
                    "Dark Mode", "App Tracking Transparency"]


def era_violations(draft):
    return [t for t in BANNED_ERA_TERMS if t.lower() in str(draft).lower()]


SETTINGS_PATH = re.compile(r"Settings\s*>\s*[A-Za-z][\w\s>&-]*", re.I)
VERSION = re.compile(r"\b\d{1,2}\.\d(\.\d)?\b")
FEATURES = [
    "Battery Health", "Face ID", "Touch ID", "True Tone", "iCloud Drive", "Apple Pay",
    "Screen Time", "Control Center", "Low Power Mode", "AirDrop", "Handoff", "Night Shift",
    "Do Not Disturb", "Find My", "Keychain", "Safe Mode", "DFU", "Recovery Mode",
    "Optimised Battery Charging", "Optimized Battery Charging", "Background App Refresh",
]


def unsupported_specifics(draft, grounding_text):
    """Concrete claims in the draft that its grounding does not contain."""
    g = str(grounding_text).lower()
    out = []
    for m in SETTINGS_PATH.findall(str(draft)):
        frag = m.strip().lower()
        if frag and frag not in g:
            out.append(f"path:{m.strip()}")
    for m in VERSION.finditer(str(draft)):
        if m.group(0) not in g:
            out.append(f"version:{m.group(0)}")
    for f in FEATURES:
        if f.lower() in str(draft).lower() and f.lower() not in g:
            out.append(f"feature:{f}")
    return sorted(set(out))


SYSTEM = """You draft public replies for Apple Support on Twitter.

RULES:
- Reply in Apple Support's voice: warm, direct, plain. No marketing language, no emoji,
  no "we apologise for the inconvenience" filler.
- Under 280 characters. One or two sentences.
- Use ONLY what the grounding material and the stated facts support. Never invent
  settings paths, version numbers, timelines, or causes.
- Never ask for account details, passwords, serial numbers or payment information in a
  public reply.
- Never include a URL or link. The grounding has had its links removed; do not invent
  replacements or refer the customer to "the link below".
- Do not state settings paths, version numbers or feature names unless they appear in
  the grounding material. If the fix needs a specific step you were not given, abstain.
- These customers are on iOS 11.0-11.2. The following do not exist yet and must NEVER
  appear in a reply: Battery Health, Optimised/Optimized Battery Charging, Screen Time,
  Find My, Dark Mode. For battery questions use only the steps you are given.
- If the grounding material does not actually address this customer's problem, do not
  guess. Reply with exactly: escalate: no grounding found
- Output the reply text only. No preamble, no quotes, no explanation."""


def build_prompt(text, intent, grounding):
    parts = [f"CUSTOMER MESSAGE:\n{text}",
             f"\nINTENT: {intent}",
             f"\nWHAT A GOOD REPLY IS HERE: {TARGET_REPLY.get(intent, 'a helpful reply')}"]
    if intent == "ios_version_downgrade":
        parts.append(f"\nSTATED FACTS (this is your grounding; there is no retrieved "
                     f"example for this intent):\n{POLICY_FACTS[intent]}")
    else:
        parts.append(
            f"\nGROUNDING - a past thread Apple Support handled, retrieved as similar:\n"
            f"  past customer: {strip_urls(grounding['n_text'])}\n"
            f"  Apple's reply:  {strip_urls(grounding['n_reply'])}")
    parts.append("\nDraft the reply now.")
    return "\n".join(parts)


def generate(prompt, timeout=300):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"].strip()


def pick_grounding(cands):
    """Best GATED candidate, not best cosine. Returns (row, n_usable) or (None, 0)."""
    ok = cands[cands.usable]
    if not len(ok):
        return None, 0
    return ok.loc[ok.gated_score.idxmax()], len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="0 = all golden rows")
    ap.add_argument("--intent", default=None, help="restrict to one intent")
    a = ap.parse_args()

    g = pd.read_pickle(GOLDEN)
    g["tweet_id"] = g["tweet_id"].astype(str)
    r = pd.read_pickle(RETRIEVAL)

    if a.intent:
        g = g[g.intent == a.intent]
        print(f"restricted to {a.intent}: {len(g)} rows\n")
    if a.sample:
        # span intents, and deliberately include zero-usable rows to exercise the
        # no-grounding path rather than only the happy path
        zero = r.groupby(["tweet_id", "q_intent"]).usable.any().reset_index()
        zero_ids = set(zero[~zero.usable].tweet_id)
        picks = []
        for intent, n in [("software_feature_defect", 4), (RESID, 4), ("battery_drain", 2),
                          ("billing_account", 2), ("ios_version_downgrade", 2),
                          ("out_of_scope", 2), ("non_english", 2)]:
            sub = g[g.intent == intent]
            hard = sub[sub.tweet_id.isin(zero_ids)].head(2)
            easy = sub[~sub.tweet_id.isin(zero_ids)].head(max(0, n - len(hard)))
            picks.append(pd.concat([hard, easy]).head(n))
        g = pd.concat(picks, ignore_index=True)
        print(f"sample of {len(g)} rows "
              f"({len(set(g.tweet_id) & zero_ids)} deliberately zero-grounding)\n")

    done = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}

    out, t0 = [], time.time()
    with open(CACHE, "a") as f:
        for q in g.itertuples():
            cands = r[r.tweet_id == q.tweet_id]
            best, n_usable = pick_grounding(cands)
            rec = {"tweet_id": q.tweet_id, "intent": q.intent, "q_text": q.customer_text,
                   "n_candidates": len(cands), "n_usable": n_usable}

            fired = check_overrides(q.customer_text)
            if fired:
                rec |= {"route": "escalate_override", "draft": ESCALATE_OVERRIDE,
                        "override": ",".join(fired), "grounded_on": None,
                        "grounding_rank": None, "gated_score": None}
            elif q.intent in NO_DRAFT:
                rec |= {"route": "no_draft_policy", "draft": NO_DRAFT[q.intent],
                        "grounded_on": None, "grounding_rank": None, "gated_score": None}
            elif q.intent in TEMPLATES:
                rec |= {"route": "policy_fact", "draft": TEMPLATES[q.intent],
                        "grounded_on": "TEMPLATE", "grounding_rank": None,
                        "gated_score": None}
            elif q.intent in POLICY_FACTS:
                rec |= {"route": "policy_fact", "grounded_on": "POLICY_FACT",
                        "grounding_rank": None, "gated_score": None}
            elif best is None:
                rec |= {"route": "no_usable_grounding", "draft": NO_GROUNDING,
                        "grounded_on": None, "grounding_rank": None, "gated_score": None}
            else:
                rec |= {"route": "grounded", "grounded_on": best["n_tweet_id"],
                        "grounding_rank": int(best["rank"]),
                        "gated_score": float(best["gated_score"])}

            if "draft" not in rec:
                if q.tweet_id in done and done[q.tweet_id].get("route") == rec["route"]:
                    rec["draft"] = done[q.tweet_id]["draft"]
                else:
                    rec["draft"] = generate(
                        build_prompt(q.customer_text, q.intent, best))
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            out.append(rec)

    d = pd.DataFrame(out)
    if "override" not in d.columns:
        d["override"] = None
    d["abstained"] = d.draft.str.lower().str.startswith("escalate")
    d["chars"] = d.draft.str.len()
    d["has_link"] = d.draft.map(has_link)
    specifics = []
    for x in d.itertuples():
        if x.route == "policy_fact":
            # NOTE: for a canned TEMPLATE the draft IS its own grounding, so this check
            # is vacuous by construction and its 0% is not evidence of anything. The
            # real guarantee for templates is the one-time review of the template text
            # plus tests/test_draft_guards.py. Generated policy_fact rows (downgrade)
            # are checked against their stated fact, where the 0% does mean something.
            src = POLICY_FACTS.get(x.intent) or TEMPLATES.get(x.intent, "")
            specifics.append(unsupported_specifics(x.draft, src))
            continue
        if x.route != "grounded":
            specifics.append([])
            continue
        g = r[(r.tweet_id == x.tweet_id) & (r.n_tweet_id == x.grounded_on)].iloc[0]
        specifics.append(unsupported_specifics(
            x.draft, strip_urls(g.n_text) + " " + strip_urls(g.n_reply)))
    d["unsupported"] = specifics
    d["flagged"] = d.unsupported.map(bool)
    d["era_violation"] = d.draft.map(era_violations)
    d["era_flagged"] = d.era_violation.map(bool)
    d.to_pickle(OUT)
    print(f"drafted {len(d)} rows in {time.time() - t0:.0f}s\n")
    print(d.route.value_counts().to_string())
    ov = d[d.route == "escalate_override"]
    print(f"\nescalate overrides: {len(ov)}/{len(d)} "
          f"({100 * len(ov) / len(d):.0f}%)")
    if len(ov):
        print("  by reason:", ov.override.str.split(",").explode().value_counts().to_dict())
    print(f"abstained overall: {d.abstained.sum()}/{len(d)}")
    print(f"era violations (features postdating iOS 11.0-11.2): "
          f"{d.era_flagged.sum()}/{len(d)}")
    if d.era_flagged.any():
        import itertools
        print("  terms:", pd.Series(list(itertools.chain(*d.era_violation))
                                    ).value_counts().to_dict())
    for rt in ["policy_fact"]:
        sub = d[d.route == rt]
        if len(sub):
            print(f"\n{rt}: {len(sub)} rows, flagged {sub.flagged.sum()} "
                  f"({100 * sub.flagged.mean():.0f}%)")
            for k, g2 in sub.groupby("intent"):
                items = [i for xs in g2.unsupported for i in xs]
                print(f"  {k:<34}{g2.flagged.sum():>3}/{len(g2):<4}"
                      f"{100 * g2.flagged.mean():>5.0f}%  {items if items else ''}")
    gr = d[d.route == "grounded"]
    if len(gr):
        print(f"\ngrounded drafts: {len(gr)}")
        print(f"  grounding rank used: {gr.grounding_rank.value_counts().sort_index().to_dict()}")
        print(f"  rank-0 used: {(gr.grounding_rank == 0).sum()}/{len(gr)}")
        print(f"  over 280 chars: {(gr.chars > 280).sum()}")
        print(f"  containing a link (fix 1 backstop): {gr.has_link.sum()}")
        print(f"  FLAGGED unsupported specifics (fix 3): {gr.flagged.sum()}/{len(gr)} "
              f"({100 * gr.flagged.mean():.0f}%)")
        allspec = [x for xs in gr.unsupported for x in xs]
        if allspec:
            print(f"  flagged items: {pd.Series(allspec).value_counts().to_dict()}")
    return d


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
