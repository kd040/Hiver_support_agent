"""Profile brand reply volumes + clean 2-turn thread counts in twcs.csv."""
import pandas as pd

CSV = "data/raw/twcs.csv"
COLS = ["tweet_id", "author_id", "inbound", "response_tweet_id", "in_response_to_tweet_id"]


def load(*extra):
    """extra: additional column names to read, e.g. "text", "created_at"."""
    df = pd.read_csv(CSV, usecols=COLS + list(extra), dtype=str)
    df["inbound"] = df["inbound"].str.strip().str.lower() == "true"
    return df


def clean_replies(df, brand_name):
    """Brand replies forming a strict terminal 2-turn thread:
       customer opens thread (no parent) -> exactly one reply, this brand's -> nothing follows.
       Returns those brand-reply rows with a `parent_id` column."""
    cust = df[df["inbound"]]
    opener = cust["in_response_to_tweet_id"].isna()
    single = cust["response_tweet_id"].notna() & ~cust["response_tweet_id"].str.contains(",", na=False)
    solo_openers = set(cust.loc[opener & single, "tweet_id"])
    resp_of = df.set_index("tweet_id")["response_tweet_id"]

    r = df[(~df["inbound"]) & (df["author_id"] == brand_name)]
    parent = r["in_response_to_tweet_id"]
    ok = (
        parent.isin(solo_openers)
        & r["response_tweet_id"].isna()
        & (resp_of.reindex(parent).values == r["tweet_id"].values)
    )
    return r[ok].assign(parent_id=parent[ok])


if __name__ == "__main__":
    df = load()
    brand = df[~df["inbound"]]
    volumes = brand["author_id"].value_counts()

    print(f"total rows: {len(df):,}   brand replies: {len(brand):,}   brands: {volumes.size:,}\n")
    print("=== TOP 20 BRANDS BY REPLY VOLUME ===")
    print(f"{'#':>3}  {'brand':<22} {'replies':>10}  {'% of brand replies':>18}")
    for i, (b, n) in enumerate(volumes.head(20).items(), 1):
        print(f"{i:>3}  {b:<22} {n:>10,}  {100*n/len(brand):>17.2f}%")

    print("\n=== CLEAN SINGLE-TURN THREADS (top 10 brands) ===")
    print(f"{'brand':<22} {'replies':>10} {'clean threads':>14} {'% clean':>9}")
    for b, n in volumes.head(10).items():
        c = len(clean_replies(df, b))
        print(f"{b:<22} {n:>10,} {c:>14,} {100*c/n:>8.1f}%")
