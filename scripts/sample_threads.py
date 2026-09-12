"""Dump N random clean 2-turn threads for a brand, full text, for manual skim."""
import sys
from profile_brands import load, clean_replies

brand, n, seed = (sys.argv + ["AppleSupport", "30", "0"])[1:4]
df = load("text")
sample = clean_replies(df, brand).sample(int(n), random_state=int(seed))
text = df.set_index("tweet_id")["text"]

for i, row in enumerate(sample.itertuples(), 1):
    print(f"\n{'='*78}\n[{i}/{n}]  customer {row.parent_id} -> {brand} {row.tweet_id}\n{'='*78}")
    print(f"CUSTOMER: {text[row.parent_id]}\n")
    print(f"{brand.upper()}: {row.text}")
