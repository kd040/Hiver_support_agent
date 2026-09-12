"""Draw candidate pools for the 200-row golden set (decisions.md #14).

Candidates only -- this writes pools for a hand-check pass, NOT the golden set.
Strata targets and sourcing are the approved plan in decisions.md #14.

Near-duplicate capping runs before any pool is drawn, same logic as the retrieval
pool dedup in decisions.md #7: exact-after-normalization dedup on customer_text,
then at most MAX_PER_CLUSTER rows sharing a brand-reply near-dup cluster.
"""
import re
import sys

import pandas as pd

from llm_relabel import DOWNGRADE_RE, TARGET, is_thin, looks_non_english

CORPUS = "data/processed/apple_threads_clustered.pkl"
OUT = "data/processed/golden_candidates.pkl"
POOL_OUT = "data/processed/golden_random_pool.pkl"
MAX_PER_CLUSTER = 2
RANDOM_POOL_N = 1800
SEED = 0

# --- approved targets (decisions.md #14) ---------------------------------
TARGETS = {
    "software_feature_defect": 67,
    "general_complaint_nonactionable": 42,   # + 12 blind-spot = 54
    "billing_account": 20,
    "battery_drain": 16,
    "ios_version_downgrade": 18,
    "out_of_scope": 18,
    "non_english": 7,
    "blind_spot": 12,
}
OVERDRAW = {                                 # candidates to hand-check per stratum
    "ios_version_downgrade": 45,
    "out_of_scope": 60,
    "non_english": 20,
    "blind_spot": 30,
}

_URL = re.compile(r"https?://\S+")


def normalize(s):
    """Same normalization as the retrieval-pool dedup (taxonomy_sample.py)."""
    s = _URL.sub(" ", str(s))
    s = re.sub(r"@(BRAND|USER)", " ", s)
    return re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\s+", " ", s.lower())).strip()


def dedupe(t):
    """Exact-after-normalization dedup, then <=MAX_PER_CLUSTER per reply cluster."""
    n0 = len(t)
    t = t.copy()
    t["cust_norm"] = t["customer_text"].map(normalize)
    t = t[t["cust_norm"].str.len() > 0]
    t = t.drop_duplicates(subset="cust_norm", keep="first")
    n1 = len(t)
    t = (t.sample(frac=1, random_state=SEED)          # shuffle so the cap is unbiased
          .groupby("cluster", group_keys=False).head(MAX_PER_CLUSTER))
    print(f"dedup: {n0:,} -> {n1:,} after exact-normalized dedup "
          f"({n0 - n1:,} dropped) -> {len(t):,} after <={MAX_PER_CLUSTER}/cluster "
          f"({n1 - len(t):,} dropped)")
    return t


# --- out_of_scope detector ------------------------------------------------
# Four kinds per taxonomy.md section 7. Deliberately recall-oriented: everything
# it returns is a candidate for hand-check, not a label.
WARRANTY = re.compile(
    r"\b(warrant(y|ies)|apple\s?care|out of warranty|covered under|"
    r"trade[- ]?in|insurance|eligib(le|ility)|policy|terms and conditions|t&cs?|"
    r"how much (does|is|would)|what.{0,15}(price|cost)|price of|cost of)\b", re.I)
PHISH = re.compile(
    r"\b(scam|phish\w*|fake (email|text|message|website|site|link)|fraud\w*|"
    r"suspicious (email|text|message|link)|is this (real|legit|genuine|from you)|"
    r"did (you|apple) send|spam (email|text))\b", re.I)
PRAISE = re.compile(
    r"\b(thank you|thanks|love (my|the|this|you)|best phone|amazing|awesome|"
    r"well done|congrats|congratulations|treating my|happy with|great job|"
    r"you guys rock|appreciate it)\b", re.I)
META = re.compile(
    r"\b(280 characters?|140 characters?|character limit|characters we can|"
    r"this account|your twitter|twitter support|these paragraphs|"
    r"more characters)\b", re.I)
OOS_KINDS = {"warranty": WARRANTY, "phishing": PHISH, "praise": PRAISE, "meta": META}

# Genius Bar / retail logistics and hardware damage. Warranty deliberately excluded
# -- taxonomy.md sends that to out_of_scope, this stratum is the residual blind spot.
BLIND = re.compile(
    r"\b(genius bar|appointment|apple store|in[- ]store|the store|repair(s|ed)?|"
    r"replac(e|ed|ement)|cracked|shattered|broken screen|screen (is )?(broke|crack)|"
    r"water damage|liquid damage|dropped my)\b", re.I)


def oos_kind(text):
    for kind, rx in OOS_KINDS.items():
        if rx.search(str(text)):
            return kind
    return None


def main():
    t = pd.read_pickle(CORPUS)
    print(f"corpus: {len(t):,} clean threads")
    t = dedupe(t)

    pools = []

    def take(name, rows, n, note=""):
        rows = rows.sample(min(n, len(rows)), random_state=SEED).copy()
        rows["stratum"] = name
        pools.append(rows)
        short = "  SHORT" if len(rows) < n else ""
        print(f"  {name:<34} drew {len(rows):>3} of {n:>3} requested{short} {note}")

    # --- targeted strata (rare intents; a random draw will not find them) ---
    print("\ntargeted pools:")
    dg = t[t["customer_text"].str.contains(DOWNGRADE_RE, na=False)]
    take("ios_version_downgrade", dg, OVERDRAW["ios_version_downgrade"],
         f"(pool {len(dg):,})")

    ne = t[t["customer_text"].map(looks_non_english)]
    take("non_english", ne, OVERDRAW["non_english"], f"(pool {len(ne):,})")

    oos = t[t["customer_text"].map(lambda x: oos_kind(x) is not None) & ~t["customer_text"].map(is_thin)]
    oos = oos.copy()
    oos["oos_kind"] = oos["customer_text"].map(oos_kind)
    print(f"    out_of_scope raw matches by kind: {oos['oos_kind'].value_counts().to_dict()}")
    # spread the over-draw across the four kinds so praise does not swamp it
    per = max(1, OVERDRAW["out_of_scope"] // len(OOS_KINDS))
    bal = oos.groupby("oos_kind", group_keys=False).apply(
        lambda g: g.sample(min(per, len(g)), random_state=SEED))
    take("out_of_scope", bal, OVERDRAW["out_of_scope"], f"(pool {len(oos):,}, balanced by kind)")

    bs = t[t["customer_text"].str.contains(BLIND, na=False)
           & ~t["customer_text"].map(is_thin)
           & ~t["customer_text"].map(lambda x: oos_kind(x) is not None)]
    take("blind_spot", bs, OVERDRAW["blind_spot"], f"(pool {len(bs):,})")

    cand = pd.concat(pools, ignore_index=True)
    cand.to_pickle(OUT)
    print(f"\nwrote {len(cand)} targeted candidates -> {OUT}")

    # --- random pool for the four common intents (needs LLM relabel) --------
    used = set(cand["tweet_id"])
    pool = t[~t["tweet_id"].isin(used)].sample(RANDOM_POOL_N, random_state=SEED)
    pool.to_pickle(POOL_OUT)
    print(f"wrote random pool of {len(pool)} -> {POOL_OUT} (relabel next)")


if __name__ == "__main__":
    sys.path.insert(0, "scripts")
    main()
