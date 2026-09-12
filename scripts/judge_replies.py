"""LLM-as-judge for reply quality on auto-handled drafts (decisions.md #32).

Scores the 141 auto_handle drafts on four dimensions, 1-5, with a one-line
justification. llama3.1:8b, structured JSON output, temperature 0.

The judge is kept BLIND to the specificity flag. That flag already measures
"claims not supported by the grounding" deterministically (#26), so leaving it out of
the prompt lets it act as an independent check on the judge's groundedness score --
if the two disagree, at least one is wrong, and that is worth knowing.
"""
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

MODEL = "llama3.1:8b"
URL = "http://localhost:11434/api/chat"
DECISIONS = "data/processed/escalation_decisions.pkl"
DRAFTS = "data/processed/drafts.pkl"
RETRIEVAL = "data/processed/retrieval_metric_embed.pkl"
CACHE = "data/processed/judge_cache.jsonl"
OUT = "data/processed/judge_scores.pkl"
DIMS = ["relevance", "groundedness", "tone", "correctness"]

RUBRIC = """You are evaluating a draft reply written by an automated support agent for
Apple Support on Twitter. Score it on four dimensions, 1 to 5.

relevance - does the reply address what THIS customer actually asked?
  5 directly addresses the specific problem stated
  3 generically on-topic but does not engage the specific problem
  1 addresses something the customer did not ask about

groundedness - is every specific claim supported by the grounding material shown?
  5 every specific claim (settings paths, versions, feature names) appears in the grounding
  3 mostly supported, one unsupported detail
  1 invents specifics the grounding does not contain
  If NO grounding is shown, score whether the reply avoids specifics it cannot support.

tone - does it sound like Apple Support: warm, direct, plain, no filler?
  5 indistinguishable from a real Apple Support reply
  3 serviceable but stiff, or mildly off-voice
  1 robotic, salesy, or narrates its own reasoning

correctness - is the advice factually right for a customer on iOS 11.0-11.2?
  5 correct and appropriate
  3 plausible but unverifiable, or incomplete
  1 factually wrong, or cites a feature that does not exist in this iOS version
  Note: Battery Health did not exist until iOS 11.3. Asking a diagnostic question
  instead of giving a fix is CORRECT for a vague complaint - do not penalise it.

Respond with JSON only, no other text:
{"relevance": n, "groundedness": n, "tone": n, "correctness": n, "justification": "one short sentence"}"""


def build(row, grounding):
    p = [f"CUSTOMER MESSAGE:\n{row.q_text}", f"\nCLASSIFIED INTENT: {row.intent}"]
    if grounding is not None:
        p.append(f"\nGROUNDING SHOWN TO THE DRAFTER:\n  past customer: {grounding[0]}"
                 f"\n  Apple's reply: {grounding[1]}")
    else:
        p.append("\nGROUNDING SHOWN TO THE DRAFTER: none (answered from a fixed policy "
                 "statement or template)")
    p.append(f"\nDRAFT REPLY TO SCORE:\n{row.final_draft}")
    return "\n".join(p)


def ask(prompt, timeout=300):
    body = json.dumps({
        "model": MODEL, "format": "json",
        "messages": [{"role": "system", "content": RUBRIC},
                     {"role": "user", "content": prompt}],
        "stream": False, "options": {"temperature": 0, "num_predict": 200},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        d = json.loads(m.group(0)) if m else {}
    out = {}
    for k in DIMS:
        v = d.get(k)
        try:
            v = int(round(float(v)))
            out[k] = v if 1 <= v <= 5 else None
        except (TypeError, ValueError):
            out[k] = None
    out["justification"] = str(d.get("justification", ""))[:300]
    out["raw"] = raw[:400]
    return out


def main():
    dec = pd.read_pickle(DECISIONS)
    dr = pd.read_pickle(DRAFTS)[["tweet_id", "route", "grounded_on", "q_text"]]
    auto = dec[dec.action == "auto_handle"].merge(dr, on="tweet_id", how="left",
                                                 suffixes=("", "_d"))
    r = pd.read_pickle(RETRIEVAL)
    print(f"judging {len(auto)} auto-handled drafts", flush=True)

    done = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}

    t0 = time.time()
    with open(CACHE, "a") as f:
        for i, row in enumerate(auto.itertuples(), 1):
            if row.tweet_id in done:
                continue
            g = None
            if row.route == "grounded":
                m = r[(r.tweet_id == row.tweet_id) & (r.n_tweet_id == row.grounded_on)]
                if len(m):
                    g = (m.iloc[0].n_text, m.iloc[0].n_reply)
            rec = {"tweet_id": row.tweet_id, **ask(build(row, g))}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[row.tweet_id] = rec
            if i % 20 == 0:
                print(f"  {i}/{len(auto)}  {time.time() - t0:.0f}s", flush=True)

    sc = pd.DataFrame([done[t] for t in auto.tweet_id if t in done])
    out = auto.merge(sc, on="tweet_id", how="left")
    out.to_pickle(OUT)

    print(f"\njudged {len(sc)} in {time.time() - t0:.0f}s")
    print(f"unparseable dimensions: "
          f"{ {d: int(out[d].isna().sum()) for d in DIMS} }")
    print(f"\n{'dimension':<16}{'mean':>7}{'1':>5}{'2':>5}{'3':>5}{'4':>5}{'5':>5}")
    for d in DIMS:
        vc = out[d].value_counts()
        print(f"{d:<16}{out[d].mean():>7.2f}" +
              "".join(f"{int(vc.get(i, 0)):>5}" for i in range(1, 6)))
    print("\nmean score by route:")
    print(out.groupby("route")[DIMS].mean().round(2).to_string())
    print("\nmean score by intent:")
    print(out.groupby("intent")[DIMS].mean().round(2).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
