"""Embedding retriever for the gated retrieval metric (decisions.md #23).

Same gates, same per-intent pools, same k as scripts/retrieval_metric.py -- only the
similarity function changes: sentence-transformers all-MiniLM-L6-v2 instead of TF-IDF.
Run over the FULL 200-row golden set.

G1's threshold is NOT transferable between retrievers. TF-IDF cosine on short tweets
sits near zero for unrelated pairs, so TAU=0.20 was meaningful there; MiniLM puts
unrelated short texts around 0.3-0.5, where 0.20 would pass everything. TAU is
therefore recalibrated per retriever as the 95th percentile of RANDOM pair similarity
within each pool -- i.e. "closer than 95% of arbitrary pairs" -- and both the old and
calibrated thresholds are reported so the change is visible rather than buried.
"""
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval_metric import (GOLDEN, K, POLICY, apply_gates, label_neighbors,
                              per_intent_pools, summarize)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = "data/processed/retrieval_metric_embed.pkl"
SEED = 0
RANDOM_PAIRS = 20000
TAU_PCTILE = 95


def calibrate(emb, rng):
    """95th percentile similarity of random pairs -- the 'no better than chance' line."""
    n = len(emb)
    a, b = rng.integers(0, n, RANDOM_PAIRS), rng.integers(0, n, RANDOM_PAIRS)
    keep = a != b
    sims = (emb[a[keep]] * emb[b[keep]]).sum(1)
    return float(np.percentile(sims, TAU_PCTILE)), float(sims.mean())


def main():
    model = SentenceTransformer(EMB_MODEL)
    rng = np.random.default_rng(SEED)
    g = pd.read_pickle(GOLDEN)
    g["tweet_id"] = g["tweet_id"].astype(str)
    pools = per_intent_pools(set(g.tweet_id))

    # defect and battery share the answer-type pool; encode each distinct pool once
    cache, parts, taus = {}, [], {}
    for intent, pool in pools.items():
        sub = g[g.intent == intent]
        if not len(sub):
            continue
        key = tuple(sorted(POLICY[intent][1]))
        if key not in cache:
            print(f"encoding pool for {key}: {len(pool):,} threads", flush=True)
            e = model.encode(pool.customer_text.tolist(), batch_size=256,
                             normalize_embeddings=True, show_progress_bar=False)
            tau, mean = calibrate(e, rng)
            print(f"  random-pair sim: mean {mean:.3f}, p{TAU_PCTILE} {tau:.3f} -> TAU")
            cache[key] = (pool, e, tau)
        pool_, emb, tau = cache[key]
        taus[intent] = tau

        q = model.encode(sub.customer_text.tolist(), batch_size=256,
                         normalize_embeddings=True, show_progress_bar=False)
        sims = q @ emb.T
        idx = np.argsort(-sims, axis=1)[:, :K]
        for qi in range(len(sub)):
            qr = sub.iloc[qi]
            for rank, j in enumerate(idx[qi]):
                n = pool_.iloc[j]
                parts.append({
                    "tweet_id": qr.tweet_id, "q_intent": qr.intent,
                    "q_text": qr.customer_text, "rank": rank,
                    "cos": float(sims[qi, j]), "n_tweet_id": n.tweet_id,
                    "n_text": n.customer_text, "n_reply": n.brand_text,
                    "n_res": n.resolution_type, "tau": tau})

    c = pd.DataFrame(parts)
    print(f"\n{len(c)} candidates for {c.tweet_id.nunique()} queries")
    c = label_neighbors(c)

    gs = g[g.intent.isin(pools)]
    print(f"\n### G1 at the OLD TF-IDF threshold (0.20) -- shown to prove it is vacuous")
    summarize(apply_gates(c), gs, "EMBEDDINGS, TAU=0.20 (uncalibrated)")

    print(f"\n\n### G1 at the calibrated per-pool threshold")
    cc = apply_gates(c)
    cc["g1_lexical"] = cc.cos >= cc.tau
    cc["usable"] = cc.g1_lexical & cc.g2_form & cc.g3_topic & cc.g4_template
    cc["gated_score"] = cc.cos.where(cc.usable, 0.0)
    summarize(cc, gs, f"EMBEDDINGS, calibrated TAU (p{TAU_PCTILE} of random pairs)")
    cc.to_pickle(OUT)

    print("\nper-intent calibrated TAU:")
    for k, v in sorted(taus.items()):
        print(f"  {k:<34}{v:.3f}")
    print(f"\nmean raw cosine {cc.cos.mean():.3f}   mean gated score {cc.gated_score.mean():.3f}")
    print("\nG3 failures by intent (topic mismatch, the gate the swap was meant to fix):")
    for k, v in cc.groupby("q_intent").g3_topic.apply(lambda s: 100 * (~s).mean()).items():
        print(f"  {k:<34}{v:>6.1f}% fail")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
