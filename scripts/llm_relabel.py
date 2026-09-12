"""Re-label the general_complaint_nonactionable bucket with local Qwen2.5 3B.

The regex labeler's residual bucket is ~2/3 recall failure (decisions.md #10).
This re-classifies only those rows against the six locked intents; the other
five buckets keep their regex labels.

Resumable: results append to a jsonl cache keyed by tweet_id, so a crash or
Ctrl-C loses at most one call. The cache is per-model -- switching MODEL starts a
fresh file rather than inheriting the previous model's labels.

Processed in batches of BATCH rows, with a progress line per batch.
"""
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

MODEL = "qwen2.5:3b"
URL = "http://localhost:11434/api/chat"
LABELED = "data/processed/taxonomy_sample_labeled.pkl"
OUT = "data/processed/taxonomy_sample_llm.pkl"
CACHE = f"data/processed/llm_relabel_cache_{MODEL.replace(':', '-')}.jsonl"
TARGET = "general_complaint_nonactionable"
OUT_OF_SCOPE = "out_of_scope"
BATCH = 50
THIN_MAX_WORDS = 6      # <= this many real words -> not classifiable from text

# A message whose content is a link or a screenshot has nothing to classify. The 3B
# model does not abstain on these, it scatters them into the rare labels: they were
# 36% of its non_english and 20% of its ios_version_downgrade output, vs ~9% of the
# two large labels. Route them to residual without spending a call.
_STRIP_RE = re.compile(r"https?://\S+|@BRAND|@USER|#\w+")


def real_words(text):
    """Words left once URLs, @BRAND/@USER placeholders and hashtags are removed."""
    return len([w for w in re.findall(r"[A-Za-z']+", _STRIP_RE.sub(" ", str(text))) if len(w) > 1])


def is_thin(text):
    return real_words(text) <= THIN_MAX_WORDS


# ios_version_downgrade needs an explicit request to move backwards. The model fired
# it on any iOS-version grievance ("please send a new update", "11.0.2 just showed
# up", "IOS11 disabled my device") -- ~7 of 10 were false positives.
DOWNGRADE_RE = re.compile(r"""\b(?:
      down\s?grad\w*
    | revert\w*
    | roll\s?back\w*
    | go(?:ing)?\s+back
    | (?:get|put|have|let|bring)\s+(?:\w+\s+){0,3}back\s+(?:to|on)
    | back\s+(?:to|on)\s+(?:ios|os|mac\s?os|sierra|version|\d)
    | (?:older|previous|earlier|old)\s+(?:\w+\s+){0,2}(?:version|versions|build|ios|os)
    | re-?install\w*\s+(?:ios|os|the\s+(?:old|previous))
    | un-?install\w*\s+(?:ios|the\s+update)
)""", re.I | re.X)


# Letter ranges only. Arabic starts at \u0620 rather than \u0600 because \u061F is
# the Arabic question mark, which turns up inside otherwise-English tweets.
NON_LATIN_RE = re.compile(r"[\u0620-\u06FF\u0750-\u077F"      # Arabic
                          r"\u0400-\u04FF"                      # Cyrillic
                          r"\u0370-\u03FF"                      # Greek
                          r"\u0590-\u05FF"                      # Hebrew
                          r"\u0900-\u097F"                      # Devanagari
                          r"\u0E00-\u0E7F"                      # Thai
                          r"\u3040-\u30FF\u4E00-\u9FFF"        # kana + CJK
                          r"\uAC00-\uD7AF]")                    # Hangul

# Function words common in English and rare in the languages this corpus actually
# contains (es, pt, de, nl, tr, fr, it). Deliberate omissions: "a", "no", "me",
# "to", "in", "is", "was" all occur in those languages too, and "help" appears
# inside Portuguese rows here ("... o iphone 8 @BRAND help me").
ENGLISH_FW = {
    "the", "you", "and", "of", "it", "my", "for", "with", "have", "this", "that",
    "not", "your", "are", "be", "do", "does", "can", "what", "why", "how", "but",
    "so", "if", "just", "all", "has", "dont", "cant", "im", "its", "there", "they",
    "will", "would", "get", "got", "been", "am", "we", "when", "now", "only",
    "please", "very", "from", "about", "after", "since", "still", "again", "more",
    "than", "because", "thanks", "thank", "know", "want", "need", "could",
    "should", "really", "every", "other", "much", "many", "even", "also", "both",
    "who", "which", "where", "into", "youre", "thats",
}
FW_MIN_WORDS = 4        # below this, absence of a function word means nothing


def _tokens(text):
    """Lowercased words, curly apostrophes normalised so don't -> dont matches."""
    t = re.sub(r"[\u2018\u2019\u02BC`]", "'", str(text))
    return [w.lower().replace("'", "")
            for w in re.findall(r"[A-Za-z']+", _STRIP_RE.sub(" ", t)) if len(w) > 1]


def looks_non_english(text):
    """Cheap replacement for the model's non_english label, which scored 0/14.

    Non-Latin script is decisive. Otherwise a message long enough to judge that
    contains no common English function word is treated as non-English.
    Measured on this sample: 18/20 recall against the regex labels, 0 false
    positives across 172 English rows.
    """
    if NON_LATIN_RE.search(str(text)):
        return True                         # script evidence is decisive
    # An explicit downgrade phrase is itself English evidence. Without this,
    # "Let me go back to ios 10 plz" carries no word from the narrow list above and
    # gets misread as foreign.
    if DOWNGRADE_RE.search(str(text)):
        return False
    w = _tokens(text)
    return len(w) >= FW_MIN_WORDS and not (set(w) & ENGLISH_FW)


def resolve(text, llm_label):
    """Apply the two hard rules on top of a raw model label.

    Pure and cache-independent, so both rules can be re-run over an existing cache
    without re-calling the model.
    """
    # Order matters.
    # 1. Script/function-word evidence decides non_english outright.
    if looks_non_english(text):
        return "non_english"
    # 2. An evidenced downgrade request carries explicit classifiable content, so it
    #    survives the thin-row rule -- "iOS 11 sucks can I go back to 10?" is 6 real
    #    words and would otherwise be discarded. Scoped to this label only: a thin
    #    row the model called something else is still thin.
    if llm_label == "ios_version_downgrade":
        if DOWNGRADE_RE.search(str(text)):
            return llm_label
        return TARGET                       # no second opinion available -> residual
    # 3. Nothing classifiable in the text at all.
    if is_thin(text):
        return TARGET
    # 4. The model's non_english is its out-of-taxonomy dumping ground (0/14 correct):
    #    warranty/policy, phishing reports, praise, meta-commentary. Anything it put
    #    there that is demonstrably English is out_of_scope, not non_english.
    if llm_label == "non_english":
        return OUT_OF_SCOPE
    return llm_label

INTENTS = [
    "software_feature_defect",
    "ios_version_downgrade",
    "battery_drain",
    "billing_account",
    "general_complaint_nonactionable",
    "non_english",
    "out_of_scope",
]

SYSTEM = """You classify customer support tweets sent to Apple Support into exactly one intent.

INTENTS:
- software_feature_defect: a SPECIFIC named component or feature (Messages, Siri, WiFi, camera, keyboard, Face ID, notifications, speaker, alarm, App Store, Apple Pay, contacts...) plus a symptom. The named component is what distinguishes this from a vague complaint.
- ios_version_downgrade: explicitly asks to GO BACK to an older iOS/macOS version (revert, roll back, downgrade, reinstall iOS 10). An update that fails to install, or a complaint about a new version, is NOT this intent.
- battery_drain: battery life, drain rate, charging, or battery health.
- billing_account: purchases, iTunes, Apple Music, subscriptions, refunds, payment methods, Apple ID, account access, orders, pre-orders.
- general_complaint_nonactionable: frustration about an update, a device, or the brand with NO specific named component and NO actionable request. Vague "it's slow", "fix my phone", "this update ruined everything", pure rant, or a general statement about the whole device rather than one feature.
- non_english: the message is written in a language other than English (Spanish, Portuguese, German, French, Dutch, Italian, Turkish, etc.). Slang, dialect, ALL CAPS, profanity, abbreviations, typos, missing punctuation and emoji are STILL ENGLISH. Only use this label if the actual words are in another language.

RULES:
- A generic device word alone (phone, iPhone, iPad, device, update, iOS) is NOT a specific component. "My iPhone is slow" is general_complaint_nonactionable. "My iPhone camera won't focus" is software_feature_defect.
- If the message names a specific component AND a symptom, prefer software_feature_defect over general_complaint_nonactionable.
- Decide non_english FIRST, and only if the words are genuinely in another language. If you can read it as English, it is English.
- Complaints that the letter "I" displays as a box, a question mark, or an "A" are the iOS 11 keyboard/autocorrect bug. Label those software_feature_defect. The odd characters are a rendering bug, NOT another language.
- Answer with the intent name only. No explanation."""

# Few-shot drawn from the same corpus (see docs/taxonomy.md).
SHOTS = [
    ("@BRAND my Messages app keeps crashing since the update", "software_feature_defect"),
    ("Hey @BRAND thanks for the last update! This time Siri doesn't work.", "software_feature_defect"),
    ("What a disappointment, 3 hours after unboxing Face ID stops working #iPhoneX @BRAND",
     "software_feature_defect"),
    ("@BRAND iOS11 seems to be a disaster. Can I revert to 10? Fix coming? Help!",
     "ios_version_downgrade"),
    ("@BRAND How do I get iOS 10 back on my 6s?", "ios_version_downgrade"),
    ("iOS update 11.1.2 is draining my 7 plus battery like crazy. Went to bed at 85%, woke up at 11%.",
     "battery_drain"),
    ("@BRAND any way to check the overall battery health of my iPhone 5s?", "battery_drain"),
    ("@BRAND I've just subbed to Apple Music and the payment has been taken but Apple Music is saying I'm not subscribed.",
     "billing_account"),
    ("@BRAND Help!! I need to update the payment method for my iPhone X pre-order. It won't let me do it online",
     "billing_account"),
    ("So @BRAND when y'all gonna fix my phone... I just got this last December... It don't need to be messin' up.",
     "general_complaint_nonactionable"),
    ("@BRAND I am absolutely fuming with your technology. I'd like to make a complaint and try to resolve this problem with some help please",
     "general_complaint_nonactionable"),
    ("This iPhone update has ruined my phone @BRAND", "general_complaint_nonactionable"),
    ("@BRAND @BRAND la actualización al iOS 11.0.3 acabó con mi iPhone 6 díganme ustedes qué hacer.",
     "non_english"),
    ("Meu, to tendo problema com o wifi, n sei se o problema é o iOS 11 ou o iphone 8 @BRAND help me",
     "non_english"),
    # Negative shots: slang-heavy English is still English.
    ("Hello yes @BRAND I jus wanna know why I can't do anything on my phone now YA GIRL WAS JUST ABOUT TO UPDATE WHATS GOING ON",
     "general_complaint_nonactionable"),
    ("@BRAND YALL FINNA OWE ME SOME MONEY IF YALL DONT FIX THIS UPDATE IMMEDIATELY",
     "general_complaint_nonactionable"),
    ("i want answers @BRAND", "general_complaint_nonactionable"),
    ("So I updated my phone and still see question marks when Pple use the letter “I” @BRAND",
     "software_feature_defect"),
    ("I swear @BRAND better get it’s shit together.. now my messages aren’t popping up",
     "software_feature_defect"),
    # A failed install is a defect, not a request to go back a version.
    ("@BRAND the iOS 11.2 update won't install on my iPad, it fails every single time",
     "software_feature_defect"),
]

BASE = [{"role": "system", "content": SYSTEM}]
for text, label in SHOTS:
    BASE += [{"role": "user", "content": text}, {"role": "assistant", "content": label}]


def classify(text, timeout=180):
    body = json.dumps({
        "model": MODEL,
        "messages": BASE + [{"role": "user", "content": text}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 12},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    got = raw.strip().strip(".`\"' \n").lower()
    for i in INTENTS:                      # tolerate the model padding its answer
        if i in got:
            return i, raw.strip()
    return None, raw.strip()


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    s = pd.read_pickle(LABELED)
    todo = s[s["intent"] == TARGET]
    if limit:
        todo = todo.head(limit)

    done = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}
    print(f"{len(todo)} rows to classify, {len(done)} already cached")

    pending = [r for r in todo.itertuples()
               if r.tweet_id not in done and not is_thin(r.customer_text)]
    n_thin = sum(is_thin(r.customer_text) for r in todo.itertuples())
    print(f"{n_thin} thin rows -> {TARGET} with no call; "
          f"{len(pending)} to classify in batches of {BATCH}")

    t0, bad = time.time(), 0
    with open(CACHE, "a") as f:
        for start in range(0, len(pending), BATCH):
            batch, bt = pending[start:start + BATCH], time.time()
            for r in batch:
                label, raw = classify(r.customer_text)
                if label is None:
                    bad += 1
                    label = TARGET                  # unparseable -> leave as-is
                rec = {"tweet_id": r.tweet_id, "intent_llm": label, "raw": raw}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[r.tweet_id] = rec
            print(f"  batch {start // BATCH + 1}: {start + len(batch)}/{len(pending)} rows, "
                  f"{(time.time() - bt) / len(batch):.1f}s/row, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
    print(f"classified {len(pending)} new rows in {time.time() - t0:.0f}s, {bad} unparseable")

    # cache holds the RAW model label; the rules are re-applied over it every run
    s["intent_llm_raw"] = s["tweet_id"].map(lambda i: done.get(i, {}).get("intent_llm"))
    in_bucket = s["intent"] == TARGET
    s["intent_llm"] = s["intent_llm_raw"]
    s.loc[in_bucket, "intent_llm"] = [
        resolve(t, l) if pd.notna(l) else l
        for t, l in zip(s.loc[in_bucket, "customer_text"], s.loc[in_bucket, "intent_llm_raw"])]
    s["intent_final"] = s["intent_llm"].where(s["intent_llm"].notna(), s["intent"])
    s.to_pickle(OUT)

    b = s[in_bucket]
    thin_n = int(b["customer_text"].map(is_thin).sum())
    dg = int(((b["intent_llm_raw"] == "ios_version_downgrade")
              & (b["intent_llm"] == TARGET) & ~b["customer_text"].map(is_thin)).sum())
    ne_rule = int(b["customer_text"].map(looks_non_english).sum())
    oos = int((b["intent_llm"] == OUT_OF_SCOPE).sum())
    print(f"\nrules: {thin_n} thin -> residual, {dg} downgrade false positives -> residual, "
          f"{ne_rule} non_english by rule, {oos} model-non_english -> {OUT_OF_SCOPE}")

    print(f"\n=== {TARGET} bucket, re-labeled (n={len(todo)}) ===")
    v = s.loc[s["intent"] == TARGET, "intent_llm"].value_counts()
    for k in INTENTS:
        n = int(v.get(k, 0))
        print(f"  {k:<34} {n:>4}  {100 * n / len(todo):>5.1f}%")

    print(f"\n=== FINAL DISTRIBUTION (n={len(s)}) ===")
    v = s["intent_final"].value_counts()
    for k in INTENTS:
        n = int(v.get(k, 0))
        print(f"  {k:<34} {n:>4}  {100 * n / len(s):>5.1f}%  {'#' * round(50 * n / v.max())}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
