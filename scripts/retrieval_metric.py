"""Retrieval quality metric that separates "similar thread" from "usable answer".

Cosine similarity alone counts a language-redirect template retrieved for an English
battery question as a hit. This gates it on whether the retrieved reply is the right
KIND of reply for the query's intent, per docs/taxonomy.md's "good resolution" lines.

Gates (all must pass for a candidate to be usable grounding):
  G1 lexical   cos_sim >= TAU
  G2 form      retrieved resolution_type is acceptable for the query's intent
  G3 topic     retrieved thread's own intent matches the query's intent
  G4 template  not the language-redirect template, unless the query IS non_english

Combined score: gated_score = max over top-k of (cos_sim if all gates pass else 0).
hit@k is gated_score > 0.

Retrieval policy per intent is NOT uniform, and averaging over it is what hid the
problem. Three groups:
  required   retrieval must supply the answer's content (defect, battery, residual)
  escalate   route is fixed by policy; retrieval supplies at most redirect wording
  none       retrieval must be bypassed entirely (downgrade per #11, out_of_scope per
             taxonomy.md section 7)
"""
import json
import os
import sys
import time
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import label_intents as LI
from llm_relabel import MODEL, TARGET, classify, resolve

GOLDEN = "data/processed/golden_set.pkl"
CORPUS = "data/processed/apple_threads_clustered.pkl"
CACHE = "data/processed/retrieval_neighbor_cache.jsonl"
OUT = "data/processed/retrieval_metric.pkl"
TAU = 0.20
K = 5
MAX_REPS = 3
SAMPLE_PER_INTENT = 10
SEED = 0

RESID = "general_complaint_nonactionable"
ANSWER = {"self_contained_answer", "answer_with_dm_followup"}

# (policy, acceptable resolution_types) from docs/taxonomy.md "Good resolution" lines.
POLICY = {
    "software_feature_defect":  ("required", ANSWER),
    "battery_drain":            ("required", ANSWER),
    RESID:                      ("required", {"clarifying_question"}),
    "billing_account":          ("escalate", {"other_channel_redirect", "dm_handoff"}),
    "non_english":              ("escalate", {"other_channel_redirect"}),
    "ios_version_downgrade":    ("none", set()),
    "out_of_scope":             ("none", set()),
}


def build_pool(golden_ids, answer_only=False):
    t = pd.read_pickle(CORPUS)
    p = t[~t.tweet_id.isin(golden_ids)]
    if answer_only:
        p = p[p.resolution_type.isin(ANSWER)]
    capped = p.groupby("cluster", group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_REPS), random_state=SEED), include_groups=False)
    return capped.reset_index(drop=True)


def topk(g, pool, k=K):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                          strip_accents="unicode")
    X = vec.fit_transform(pool.customer_text)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(vec.transform(g.customer_text))
    rows = []
    for qi, (ds, ix) in enumerate(zip(dist, idx)):
        q = g.iloc[qi]
        for rank, (d, j) in enumerate(zip(ds, ix)):
            n = pool.iloc[j]
            rows.append({
                "tweet_id": q.tweet_id, "q_intent": q.intent, "q_text": q.customer_text,
                "rank": rank, "cos": 1 - d, "n_tweet_id": n.tweet_id,
                "n_text": n.customer_text, "n_reply": n.brand_text,
                "n_res": n.resolution_type})
    return pd.DataFrame(rows)


def label_neighbors(cands):
    """Intent-label each unique retrieved thread with the same model + rules."""
    done = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            done = {json.loads(l)["tweet_id"]: json.loads(l) for l in f if l.strip()}
    uniq = cands.drop_duplicates("n_tweet_id")[["n_tweet_id", "n_text"]]
    todo = [r for r in uniq.itertuples() if r.n_tweet_id not in done]
    print(f"labelling {len(todo)} unique retrieved threads ({len(done)} cached)")
    t0 = time.time()
    with open(CACHE, "a") as f:
        for i, r in enumerate(todo, 1):
            lab, raw = classify(r.n_text)
            rec = {"tweet_id": r.n_tweet_id, "intent": resolve(r.n_text, lab or TARGET)}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[r.n_tweet_id] = rec
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return cands.assign(n_intent=cands.n_tweet_id.map(lambda i: done[i]["intent"]))


def apply_gates(c):
    c = c.copy()
    c["g1_lexical"] = c.cos >= TAU
    c["g2_form"] = [r in POLICY[i][1] for i, r in zip(c.q_intent, c.n_res)]
    c["g3_topic"] = c.n_intent == c.q_intent
    is_redirect = c.n_reply.str.contains(LI.LANG_REDIRECT, na=False)
    c["g4_template"] = ~is_redirect | (c.q_intent == "non_english")
    c["usable"] = c.g1_lexical & c.g2_form & c.g3_topic & c.g4_template
    c["gated_score"] = c.cos.where(c.usable, 0.0)
    return c


def summarize(c, g, label):
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    old = c[c.g1_lexical].groupby("tweet_id").size().reindex(g.tweet_id).fillna(0) > 0
    new = c.groupby("tweet_id").usable.any().reindex(g.tweet_id).fillna(False)
    print(f"{'intent':<34}{'n':>4}{'policy':>10}{'old hit@5':>11}{'new hit@5':>11}{'gap':>8}")
    for k in POLICY:
        ids = g[g.intent == k].tweet_id
        if not len(ids):
            continue
        o, nw = 100 * old.loc[ids].mean(), 100 * new.loc[ids].mean()
        print(f"{k:<34}{len(ids):>4}{POLICY[k][0]:>10}{o:>10.0f}%{nw:>10.0f}%{nw - o:>+7.0f}pp")
    req = g[g.intent.map(lambda i: POLICY[i][0] == "required")].tweet_id
    print(f"\nALL rows            old {100 * old.mean():.0f}%   new {100 * new.mean():.0f}%")
    print(f"'required' subset   old {100 * old.loc[req].mean():.0f}%   "
          f"new {100 * new.loc[req].mean():.0f}%   <- the honest headline")
    print("\ngate attrition (share of all candidates failing each gate):")
    for gt in ["g1_lexical", "g2_form", "g3_topic", "g4_template"]:
        print(f"  {gt:<14}{100 * (~c[gt]).mean():>6.1f}% fail")
    return old, new


def main():
    g_all = pd.read_pickle(GOLDEN)
    g = pd.concat([x.sample(min(SAMPLE_PER_INTENT, len(x)), random_state=SEED)
                   for _, x in g_all.groupby("intent")]).reset_index(drop=True)
    g["tweet_id"] = g["tweet_id"].astype(str)
    print(f"golden sample: {len(g)} rows, {g.intent.nunique()} intents")
    print(g.intent.value_counts().to_string())

    pool = build_pool(set(g_all.tweet_id))
    print(f"\nfull retrieval pool: {len(pool):,} threads (all resolution_types, "
          f"<={MAX_REPS}/cluster)")
    print(pool.resolution_type.value_counts().to_string())

    c = apply_gates(label_neighbors(topk(g, pool)))
    c.to_pickle(OUT)
    summarize(c, g, "FULL POOL (all resolution_types)")

    apool = build_pool(set(g_all.tweet_id), answer_only=True)
    print(f"\n\nanswer-type-only pool (decisions #7 shape): {len(apool):,} threads")
    ac = apply_gates(label_neighbors(topk(g, apool)))
    summarize(ac, g, "ANSWER-TYPE-ONLY POOL (decisions #7 shape)")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()


# --- per-intent pools -----------------------------------------------------
# One pool cannot serve every intent: the resolution_type an intent needs has to be
# IN the pool it retrieves from. decisions #7's answer-type pool serves defect and
# battery and structurally cannot serve the rest.
def per_intent_pools(golden_ids):
    t = pd.read_pickle(CORPUS)
    t = t[~t.tweet_id.isin(golden_ids)]
    pools = {}
    for intent, (policy, types) in POLICY.items():
        if policy == "none" or not types:
            continue
        p = t[t.resolution_type.isin(types)]
        capped = p.groupby("cluster", group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_REPS), random_state=SEED),
            include_groups=False).reset_index(drop=True)
        pools[intent] = capped
    return pools


def run_per_intent(g, golden_ids):
    pools = per_intent_pools(golden_ids)
    print(f"\n\nper-intent pools (each filtered to the resolution_types that intent needs):")
    for k, v in pools.items():
        print(f"  {k:<34}{len(v):>7,} threads  {sorted(POLICY[k][1])}")
    parts = []
    for intent, pool in pools.items():
        sub = g[g.intent == intent]
        if not len(sub):
            continue
        parts.append(topk(sub, pool))
    c = apply_gates(label_neighbors(pd.concat(parts, ignore_index=True)))
    c.to_pickle("data/processed/retrieval_metric_per_intent.pkl")
    summarize(c, g[g.intent.isin(pools)], "PER-INTENT POOLS")
    return c
