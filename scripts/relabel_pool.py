"""Relabel the golden-set random pool with qwen2.5:3b + the decisions.md #12/#13 rules.

Same model, prompt, rules and per-row jsonl cache as llm_relabel.py; only the input
differs (an arbitrary pool rather than the n=500 residual bucket).
"""
import json
import os
import time

import pandas as pd

from llm_relabel import BATCH, MODEL, TARGET, classify, is_thin, resolve

POOL = "data/processed/golden_random_pool.pkl"
OUT = "data/processed/golden_random_pool_labeled.pkl"
CACHE = f"data/processed/pool_relabel_cache_{MODEL.replace(':', '-')}.jsonl"


def main():
    p = pd.read_pickle(POOL)
    done = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}
    pending = [r for r in p.itertuples()
               if r.tweet_id not in done and not is_thin(r.customer_text)]
    print(f"pool {len(p)}, cached {len(done)}, "
          f"{sum(is_thin(r.customer_text) for r in p.itertuples())} thin (no call), "
          f"{len(pending)} to classify", flush=True)

    t0 = time.time()
    with open(CACHE, "a") as f:
        for start in range(0, len(pending), BATCH):
            batch = pending[start:start + BATCH]
            for r in batch:
                label, raw = classify(r.customer_text)
                rec = {"tweet_id": r.tweet_id, "intent_llm": label or TARGET, "raw": raw}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[r.tweet_id] = rec
            print(f"  {start + len(batch)}/{len(pending)}  {time.time() - t0:.0f}s", flush=True)

    p["intent_raw"] = p["tweet_id"].map(lambda i: done.get(i, {}).get("intent_llm"))
    p["intent_est"] = [resolve(t, l) if pd.notna(l) else TARGET
                       for t, l in zip(p["customer_text"], p["intent_raw"])]
    p.to_pickle(OUT)
    print("\n" + p["intent_est"].value_counts().to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
