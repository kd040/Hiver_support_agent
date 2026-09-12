"""Dedup the retrieval pool, draw a capped stratified sample, cluster it.

Step 1: near-duplicate clustering of brand replies (exact-after-normalization,
        then >=0.95 cosine merge) -> capped retrieval pool.
Step 2: ~500 customer-first messages, no bug/near-dup cluster over 15%.
Step 3: k-means over those messages, k chosen by silhouette in [6, 12].
"""
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANSWER_TYPES = ["self_contained_answer", "answer_with_dm_followup"]
NEAR_DUP = 0.95
MAX_REPS = 3          # representatives kept per near-duplicate cluster
SAMPLE_N = 500
CAP = 0.15            # no single bug/cluster stratum above this share
SEED = 0

CACHE = "data/processed/apple_threads_clustered.pkl"
SAMPLE_OUT = "data/processed/taxonomy_sample.pkl"
t = pd.read_pickle("data/processed/apple_threads.pkl").reset_index(drop=True)
t["incident_window"] = t["incident_window"].where(t["incident_window"].notna(), None)
model = SentenceTransformer(MODEL)


def normalize(s):
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"@(BRAND|USER)", " ", s)
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", s.lower())).strip()


# --- near-duplicate clusters over every brand reply ------------------------
import os
if os.path.exists(CACHE):
    t = pd.read_pickle(CACHE)
    print(f"loaded cached clustering: {t['cluster'].nunique():,} clusters over {len(t):,} replies")
else:
    t["norm"] = t["brand_text"].map(normalize)
    uniq = t["norm"].drop_duplicates().reset_index(drop=True)
    print(f"brand replies: {len(t):,}   unique after normalization: {len(uniq):,}")

    emb = model.encode(uniq.tolist(), batch_size=256, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=False)
    communities = util.community_detection(emb, threshold=NEAR_DUP, min_community_size=2)
    print(f"near-duplicate communities (>= {NEAR_DUP} cos): {len(communities):,}")

    # cluster id per unique string; singletons get their own id
    cid = np.arange(len(uniq)) + len(communities)
    for i, comm in enumerate(communities):
        for j in comm:
            cid[j] = i
    t["cluster"] = t["norm"].map(dict(zip(uniq, cid)))
    print(f"total clusters incl. singletons: {t['cluster'].nunique():,}")
    t.to_pickle(CACHE)

# --- retrieval pool: cap representatives per cluster -----------------------
pool = t[t["resolution_type"].isin(ANSWER_TYPES)]
capped = pool.groupby("cluster", group_keys=False).apply(
    lambda g: g.sample(min(len(g), MAX_REPS), random_state=SEED), include_groups=False
)
print(f"\n=== RETRIEVAL POOL ===")
print(f"  before dedup: {len(pool):,}")
print(f"  after  dedup: {len(capped):,}  (max {MAX_REPS}/cluster, "
      f"{100 * (1 - len(capped) / len(pool)):.1f}% removed)")

big = pool["cluster"].value_counts().head(5)
print("\n  largest reply clusters, before -> after:")
for c, n in big.items():
    ex = pool.loc[pool["cluster"] == c, "brand_text"].iloc[0]
    print(f"    {n:>6,} -> {min(n, MAX_REPS):>2}   {ex[:96]}")

# --- stratified ~500 customer messages -------------------------------------
cap_n = int(SAMPLE_N * CAP)
rows, n_bug, per_cluster = [], 0, {}
for r in t.sample(frac=1, random_state=SEED).itertuples():
    if len(rows) >= SAMPLE_N:
        break
    is_bug = r.incident_window == "ios_11_1_bug"
    if is_bug and n_bug >= cap_n:
        continue
    if per_cluster.get(r.cluster, 0) >= cap_n:
        continue
    rows.append(r)
    n_bug += is_bug
    per_cluster[r.cluster] = per_cluster.get(r.cluster, 0) + 1

s = pd.DataFrame(rows)
print(f"\n=== TAXONOMY SAMPLE ===")
print(f"  n={len(s)}  bug_kw={n_bug} ({100 * n_bug / len(s):.1f}%, cap {cap_n})"
      f"  largest reply-cluster share={max(per_cluster.values())} "
      f"({100 * max(per_cluster.values()) / len(s):.1f}%)")
print(f"  resolution_type mix: {dict(s['resolution_type'].value_counts())}")
s.to_pickle(SAMPLE_OUT)
print(f"  wrote {SAMPLE_OUT}")

# --- cluster the customer messages -----------------------------------------
X = model.encode(s["customer_text"].tolist(), batch_size=256,
                 normalize_embeddings=True, show_progress_bar=False)
print("\n  k selection (silhouette):")
best = None
for k in range(6, 13):
    fit = KMeans(k, n_init=10, random_state=SEED).fit(X)
    sc = silhouette_score(X, fit.labels_)
    print(f"    k={k:>2}  silhouette={sc:.4f}")
    if best is None or sc > best[1]:
        best = (k, sc, fit)
k, sc, km = best
labels = km.labels_
print(f"  -> k={k} (silhouette {sc:.4f})")

s["cl"] = labels
for c in range(k):
    idx = np.where(labels == c)[0]
    d = X[idx] @ km.cluster_centers_[c]        # cosine to centroid
    print(f"\n{'#' * 78}\n### cluster {c}  (n={len(idx)}, {100 * len(idx) / len(s):.0f}%)\n{'#' * 78}")
    for rank in np.argsort(-d)[:10]:
        row = s.iloc[idx[rank]]
        bug = " [bug]" if row["incident_window"] == "ios_11_1_bug" else ""
        print(f"  ({d[rank]:.2f}){bug} {row['customer_text'][:200]}")
