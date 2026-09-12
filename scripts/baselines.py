"""Two baselines to beat, scored with eval_report.py (decisions.md #19).

1. trivial  -- always predict the majority intent, canned reply, always escalate.
2. tfidf    -- TF-IDF + logistic regression intent classifier, plus nearest-neighbour
               reply retrieval. No LLM at inference time.

The classifier trains on the qwen-relabeled random pool, NEVER on the golden set.
Golden tweet_ids are dropped from the training frame AND from the retrieval pool, or a
golden message would retrieve its own thread.
"""
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

from eval_report import report

GOLDEN = "data/processed/golden_set.pkl"
POOL = "data/processed/golden_random_pool_labeled.pkl"
CORPUS = "data/processed/apple_threads_clustered.pkl"
ANSWER_TYPES = ["self_contained_answer", "answer_with_dm_followup"]
MAX_REPS = 3
SEED = 0
CANNED = "Thanks for reaching out - we'll look into it and get back to you."


def load():
    g = pd.read_pickle(GOLDEN)
    pool = pd.read_pickle(POOL)
    train = pool[~pool.tweet_id.isin(set(g.tweet_id))].copy()
    return g, train


def trivial(g):
    majority = g.intent.value_counts().idxmax()
    print(f"majority intent in golden set: {majority} "
          f"({100 * (g.intent == majority).mean():.1f}%)")
    return pd.DataFrame({"tweet_id": g.tweet_id, "intent_true": g.intent,
                         "intent_pred": majority}), majority


def tfidf_clf(g, train, balanced):
    pipe = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                        strip_accents="unicode", lowercase=True),
        LogisticRegression(max_iter=2000, C=2.0, random_state=SEED,
                           class_weight="balanced" if balanced else None))
    pipe.fit(train.customer_text, train.intent_est)
    pred = pipe.predict(g.customer_text)
    return pd.DataFrame({"tweet_id": g.tweet_id, "intent_true": g.intent,
                         "intent_pred": pred}), pipe


def retrieval_pool(golden_ids):
    t = pd.read_pickle(CORPUS)
    p = t[t.resolution_type.isin(ANSWER_TYPES) & ~t.tweet_id.isin(golden_ids)]
    capped = p.groupby("cluster", group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_REPS), random_state=SEED), include_groups=False)
    print(f"retrieval pool: {len(p):,} answer-type threads -> {len(capped):,} "
          f"after <={MAX_REPS}/cluster")
    return capped.reset_index(drop=True)


def nn_replies(g, pool):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                          strip_accents="unicode")
    X = vec.fit_transform(pool.customer_text)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(vec.transform(g.customer_text))
    return pd.DataFrame({
        "tweet_id": g.tweet_id.values, "intent": g.intent.values,
        "query": g.customer_text.values,
        "retrieved_reply": pool.brand_text.iloc[idx[:, 0]].values,
        "cos_sim": (1 - dist[:, 0])})


def main():
    g, train = load()
    print(f"golden {len(g)} rows; training frame {len(train)} rows "
          f"({len(g) + len(train) - len(train)} golden ids removed from pool)")
    print("training label counts:", train.intent_est.value_counts().to_dict())

    print("\n" + "=" * 78 + "\nBASELINE 1: trivial (majority class + canned reply + always escalate)\n" + "=" * 78)
    t_pred, majority = trivial(g)
    report(t_pred)
    print(f'\ncanned reply for every row: "{CANNED}"\nescalation: always')

    for balanced in (True, False):
        tag = "balanced" if balanced else "unweighted"
        print("\n" + "=" * 78 +
              f"\nBASELINE 2: TF-IDF + logistic regression ({tag} classes)\n" + "=" * 78)
        s_pred, pipe = tfidf_clf(g, train, balanced)
        report(s_pred)
        if balanced:
            s_pred.to_pickle("data/processed/baseline_tfidf_predictions.pkl")

    print("\n" + "=" * 78 + "\nBASELINE 2 reply side: nearest-neighbour retrieval\n" + "=" * 78)
    pool = retrieval_pool(set(g.tweet_id))
    r = nn_replies(g, pool)
    r.to_pickle("data/processed/baseline_retrieval.pkl")
    print(f"median cosine similarity to retrieved neighbour: {r.cos_sim.median():.3f}")
    print(f"rows with sim < 0.20 (essentially no match): "
          f"{(r.cos_sim < 0.20).sum()}/{len(r)} ({100 * (r.cos_sim < 0.20).mean():.0f}%)")
    print("\nby intent, median similarity:")
    for k, v in r.groupby("intent").cos_sim.median().sort_values().items():
        print(f"  {k:<34}{v:.3f}")
    print("\nexamples:")
    for row in r.sort_values("cos_sim", ascending=False).head(2).itertuples():
        print(f"  [sim {row.cos_sim:.2f}] Q: {' '.join(row.query.split())[:90]}")
        print(f"                 A: {' '.join(row.retrieved_reply.split())[:90]}")
    print("\nNOTE: reply quality is NOT scored here. eval_report measures intent accuracy")
    print("only; there is no reply metric yet, so retrieval is reported descriptively.")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
