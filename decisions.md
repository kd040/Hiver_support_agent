# Decision log

Running list of non-obvious decisions and why. Newest last.

---

### 1. Brand: AppleSupport

106,860 brand replies, second-highest volume in the dataset, and 45.8% of them
form strict terminal 2-turn threads — the best volume × cleanliness trade among
the top 10. AmazonHelp has more volume but only 18% clean threads.

### 2. "Clean thread" defined strictly, and up front

A clean thread is: customer tweet opens the thread (no parent), has exactly one
reply, that reply is the brand's, and nothing follows it. 48,968 threads for
AppleSupport. Chose the strict definition so the grounding pool has unambiguous
question→answer pairs; the messier multi-turn threads are still on disk and can
be added later if retrieval is starved.

### 3. Anonymization by placeholder, not deletion

Mentions are replaced with `@BRAND` / `@USER` rather than stripped. Deleting them
left sentence holes ("Dear , how the hell...", "Purchased unlocked from apple
using") that corrupt the input to both the classifier and the LLM. The brand's
numeric alias (`@115858`) is derived from the data rather than hardcoded — it
appears 23,161 times against a runner-up of 683, so the detection is unambiguous.

**Known limitation:** other brands' numeric aliases map to `@USER`, so a
third-party retailer mention reads as a customer. Rare; silent when wrong.

### 4. `resolution_type` is an eval-labeling field, never a customer intent

Six values: `dm_handoff` (59.2%), `self_contained_answer` (26.2%),
`clarifying_question` (5.4%), `answer_with_dm_followup` (5.3%),
`other_channel_redirect` (3.8%), `other` (0.0%). It describes how the brand
closed the thread, which is what "good" gets measured against per intent — it is
not a feature of the incoming message and must not leak into intent
classification.

`answer_with_dm_followup` was split out from `self_contained_answer` because a
reply that gives a real answer *and* pushes to DM is not a self-contained
resolution; collapsing them inflated the apparent grounding pool by 2,609.

### 5. The DM deep-link is stripped before classifying

`t.co/GDrqU22YpT` appears in 61.8% of all brand replies. Any "does the reply
contain a link" heuristic is meaningless until it is removed, since the DM invite
is itself a link.

### 6. The iOS 11.1 incident is tagged, not dropped

19% of customer messages concern one dead autocorrect bug, and Nov 2–12 2017 is
34% of threads from 17% of days. Rather than filter, threads carry
`incident_window = "ios_11_1_bug"` (an OR of date-window and keyword match, with
both components kept separately as `in_window` / `bug_kw`). Capped at 15% of any
sample. Deliberately kept in the eventual golden set: they are the best available
test of whether the agent over-retrieves the dead-bug template for superficially
similar but different issues.

### 7. Retrieval pool deduped to ≤3 representatives per near-duplicate cluster

15,428 → 8,038 (47.9% removed). One canned workaround reply appears 4,192 times;
left in, it would dominate retrieval and inflate any hit-rate metric. Clustering
is exact-match-after-normalization then ≥0.95 cosine merge via
`sentence_transformers.util.community_detection`.

### 8. Taxonomy named by hand, not by k-means

Silhouette was 0.033–0.042 across k=6..12 — no real separation, and the spread
between k values is noise. The clusters sorted by language, register, and surface
keyword, splitting the vague-complaint mass three ways. Used them as evidence,
not as the answer; merged to six intents by hand. Recording this because the
headline "we clustered and found 6 intents" would be misleading.

### 9. Cleaning pipeline refactored to pure functions with pinned regression tests

`scripts/clean_threads.py` originally ran everything at import, so none of it was
testable. Two bugs shipped silently as a result and were caught only by reading
output:

1. The `BUG` regex's leading alternative was a bare U+FE0F variation selector,
   which matches *every* VS16 emoji (☺️, ❤️) rather than the iOS artifact. It
   over-tagged the incident by 635 threads.
2. `None` became `NaN` through a DataFrame roundtrip, and `NaN` is truthy, so
   every clustered example printed as `[bug]`.

Both classes of bug are invisible in aggregate counts — they only surface when
someone reads individual rows. Split into `anonymize` / `classify_resolution_type`
/ `has_bug_keyword` / `in_incident_window` / `tag_incident_window` / `build()`,
with execution behind `if __name__ == "__main__"`, and pinned both bugs plus one
example per `resolution_type` in `tests/test_cleaning.py`. Verified the refactor
is behavior-preserving: all counts identical.

**Cache invalidation is manual.** `taxonomy_sample.py` prefers
`data/processed/apple_threads_clustered.pkl` whenever it exists, so a change to
the cleaning logic will not invalidate it — delete the file by hand after any
cleaning change. A content hash would fix this; not worth it until the cleaning
logic changes often enough to bite.

### 10. Intent labeling of the taxonomy sample is bootstrap-quality on purpose

`scripts/label_intents.py` is keyword/rule heuristics to *size* the intents, not
to label the golden set. A 30-row hand audit of the residual bucket found roughly
two-thirds are labeler recall failures rather than genuinely vague messages, so
the reported 60.8% `general_complaint_nonactionable` is a ceiling, not an
estimate. The golden set gets hand-labelled; this exists to inform its sampling
strategy.

### 11. `ios_version_downgrade` is grounded in objective correctness, not historical behavior

**Deliberate target-vs-history divergence. Flag for the report's problem-framing
section.**

Every `ios_version_downgrade` thread in the taxonomy sample (4/4) was answered
with `dm_handoff` — Apple deflected all of them to DM. But the correct answer is
public, fixed, and identical for every customer: downgrading is not supported
once Apple stops signing the older iOS build. There is nothing account-specific
to look up.

So for this intent the agent is graded against **what the right answer is**, not
**what Apple did**. Retrieval-grounded drafting would otherwise learn to imitate
the deflection, and the eval harness would score that imitation as correct —
the agent would be measured as successful precisely when it withheld an answer
it already had.

Consequences, all of which need stating in the report:

- The retrieval pool contains no usable grounding for this intent. Its reply
  must come from a stated policy fact, not a retrieved neighbor. That makes it
  the one intent where the pipeline deliberately bypasses retrieval.
- Any "matches historical reply" metric will score this intent as a *failure*
  when the agent is behaving correctly. Report it separately rather than letting
  it drag the headline number in either direction.
- This is a live example for the "what is misleading about my headline number?"
  section: agreement-with-history and correctness are different targets, and
  this intent is where they visibly diverge.

Not generalized to other intents. `billing_account` deflections, by contrast,
are *correct* — they need account access the agent does not have — so there
history and target agree. The divergence is specific to intents whose answer is
public, fixed, and customer-independent; `ios_version_downgrade` is currently
the only one that qualifies.

### 12. LLM re-labeling runs on qwen2.5:3b behind two hard rules

The residual-bucket re-label (decisions.md #10) moved from `llama3.1:8b` to
`qwen2.5:3b`: a 6-way classification against fixed definitions does not need the
larger model, and the 3B runs 6x faster on this machine (304 rows in 88s at
0.3s/row warm, vs ~1.9s/row cold for the 8B). The cache is now keyed per model —
it previously keyed on `tweet_id` alone, so switching models would have silently
folded 62 stale `llama3.1:8b` labels into a distribution reported as qwen's.

**The unguarded 3B output was not trustworthy, and only a row-by-row read showed
it.** Aggregate counts looked plausible. Hand-auditing the reassignments found:

- `non_english`: **0 of 14 correct.** Every row was English. It caught messages
  that *mention* a language ("Please change the Arabic font in iOS11"), praise
  ("Wow! Thanks @BRAND!"), phishing reports, and bare URLs.
- `ios_version_downgrade`: **~3 of 10 correct.** It fired on any iOS-version
  grievance — "please send a *new* update", "11.0.2 has just showed up 👍",
  "IOS11 disabled my device" — inflating a ≈1% intent to 2.8%.
- `software_feature_defect`: ~15 of 20 sampled correct. The bulk of the +130 is
  genuine recall recovery (podcast app, speakerphone, alarms, WiFi, letter-"I"
  keyboard bug). This is the load-bearing number and it holds.
- `battery_drain` 4/4, `billing_account` ~5-6/10.

Two rules now sit on top of the model's label, both pure functions in
`scripts/llm_relabel.py` applied at aggregation time so they re-run over an
existing cache without spending calls:

1. **Thin-row shortcut.** A row with ≤6 real words (after stripping URLs,
   `@BRAND`/`@USER` placeholders and hashtags) goes straight to residual with no
   LLM call. 32 of 304 rows (11%) qualify — a link or a screenshot has no text to
   classify, and the 3B does not abstain, it scatters them into the rare labels:
   they were 36% of its `non_english` and 20% of its `ios_version_downgrade`
   output against ~9% of the two large labels.
2. **Downgrade precondition.** `ios_version_downgrade` requires explicit
   backwards-movement language (downgrade / revert / roll back / go back to /
   older-previous version / reinstall iOS). 7 false positives dropped to residual.

Rule precedence is load-bearing and was caught only by a test: an evidenced
downgrade request is exempt from the thin rule, because "iOS 11 sucks can I go
back to 10?" is exactly 6 real words and is unambiguously the intent. The
exemption is scoped to that one label — a thin row the model called something
else is still thin. Pinned in `tests/test_relabel_rules.py` with the real corpus
rows that motivated each rule.

Result, bucket of 304: 38.8% `software_feature_defect`, 53.0% residual, 3.3%
`billing_account`, 3.0% `non_english`, 1.3% `battery_drain`, 0.7%
`ios_version_downgrade`. Full 500-row sample: 39.2 / 32.2 / 12.0 / 5.8 / 9.6 /
1.2%. The residual bucket fell from 60.8% to 32.2%, so roughly half of it was
labeler recall failure — consistent with the ~2/3 ceiling estimated by hand in #10.

**`non_english` is still broken and is deliberately left unfixed.** After the
thin-row rule removed 5 of its 14 rows, all 9 survivors are English with zero
foreign signal — no accented characters, no non-English function words. The 3B
is using `non_english` as a second residual for out-of-taxonomy messages:
warranty/policy questions, phishing and scam reports, praise, and meta-commentary
about Twitter itself. Four of the nine fall under taxonomy.md's own "Not in
scope" list. The LLM contributes **0% precision** on this label, so the final
`non_english` count of 29 should be read as the 20 regex labels plus 9 errors.
Not patched yet because the right fix is probably a cheap script-range/stopword
check rather than another prompt rule, and because it is evidence for the
separate question of whether out-of-taxonomy messages need their own intent.

**Reporting caveat for the golden set:** these labels are still bootstrap
quality. #10's rule stands — the golden set gets hand-labelled; this sizes the
intents and tells the sampler where to over-sample. `non_english` and
`ios_version_downgrade` counts in particular are not safe to stratify on.

### 13. `non_english` moved off the LLM to a script/function-word rule, and `out_of_scope` added as a 7th intent

Follow-up to #12, which left `non_english` knowingly broken. Two changes, both
pure functions in `scripts/llm_relabel.py`, no extra model calls.

**The model no longer decides `non_english` at all.** It scored 0/14 on the label.
A rule replaces it: non-Latin script (Arabic, Cyrillic, Greek, Hebrew, Devanagari,
Thai, kana/CJK, Hangul) is decisive; otherwise a message with ≥4 real words
containing no common English function word is treated as non-English. Measured
against the 20 rows the regex labeler had tagged: **18/20 recall, 0 false
positives across 172 English rows.**

Three calibration details that were not guessable and are pinned in
`tests/test_relabel_rules.py`:

- Arabic must start at U+0620, not U+0600. U+061F is the Arabic question mark and
  appears inside otherwise-English tweets ("they took from 34.97$ ؟؟ what should I
  do"), which the wider range flagged as foreign.
- The function-word list deliberately omits `a`, `no`, `me`, `to`, `in`, `is`,
  `was` — all frequent in Spanish, Portuguese, Dutch and German too — and omits
  `help`, which appears inside Portuguese rows here ("... ou o iphone 8 @BRAND
  help me"). Including any of them costs recall.
- Curly apostrophes must be normalised first, or `don't` tokenises to `don` + `t`,
  matches nothing, and "17 hours to update is a little ridiculous don't ya think?"
  reads as foreign.

The two remaining misses are a bare link (correctly handled by the thin-row rule)
and "very bad service Apple service centre", which is English — the rule is
arguably right and the regex label wrong. Not chased.

**Rule precedence now has three interacting layers, and a test caught each
collision.** Script evidence wins outright. Then an evidenced `ios_version_downgrade`
is exempt from *both* the thin rule and the function-word rule — "Let me go back to
ios 10 plz" has no word from the list and was being read as foreign, so a downgrade
phrase is now treated as English evidence in its own right. Then thin. Then the
`out_of_scope` reroute. Both collisions were invisible in the aggregate counts.

**`out_of_scope` (7th intent).** Applying the rule inside the residual bucket found
**zero** genuine non-English rows — the regex labeler had already caught all 20, so
every one of the model's 9 surviving `non_english` rows was a misfile. They are a
coherent group: warranty and policy questions, phishing/scam reports, praise with no
request, and meta-commentary about Twitter. Derived rather than hardcoded: model said
`non_english` AND the rule says it is English AND it is not thin → `out_of_scope`.
That generalises to new data with the same failure mode instead of pinning 9 ids.
Documented in `docs/taxonomy.md` §7; phishing reports were moved there from
`billing_account`, whose §4 note was updated so the doc does not contradict itself.

`out_of_scope` is a **floor, not an estimate.** These 9 are only the ones the
re-labeler happened to misfile as `non_english`. Retail/Genius Bar logistics and
hardware-damage reports are still sitting in the residual bucket because nothing
looks for them.

Also worth flagging for the report: Apple answered 5 of the 9 with
`self_contained_answer`, so history-agreement will reward the agent for confidently
answering region-specific warranty questions it has no grounding for. Same
target-vs-history divergence as #11, in the opposite direction — there history
under-answers the target, here it over-answers it.

Final distribution, n=500: `software_feature_defect` 196 (39.2%),
`general_complaint_nonactionable` 161 (32.2%), `billing_account` 60 (12.0%),
`battery_drain` 48 (9.6%), `non_english` 20 (4.0%), `out_of_scope` 9 (1.8%),
`ios_version_downgrade` 6 (1.2%). Taxonomy corrections end here; next step is
golden-set sampling design, and #12's caveat still holds — these are bootstrap
labels, the golden set gets hand-labelled.

### 14. Golden-set sampling plan, and why per-intent accuracy is the headline metric

200 rows drawn from the full 48,968-thread cleaned corpus (not the n=500 taxonomy
sample, which existed for discovery). Targets: `software_feature_defect` 67,
`general_complaint_nonactionable` 54 (42 general + 12 blind-spot), `billing_account`
20, `battery_drain` 16, `ios_version_downgrade` 18, `out_of_scope` 18,
`non_english` 7. Roughly proportional to the corrected n=500 mix (#13), except that
`ios_version_downgrade` and `out_of_scope` get floors of 18 regardless of their ~1-2%
share, because a rare intent needs enough rows to measure at all.

**Blended accuracy would misrepresent production performance, so it is never the
headline.** The floors over-sample `ios_version_downgrade` ~9x and `out_of_scope`
~10x. A single average over golden-set rows therefore lets two intents worth ~3% of
real traffic drive ~18% of the score. `scripts/eval_report.py` reports per-intent
accuracy as primary, names the weakest intent, and prints two clearly-labeled
secondary aggregates: a naive blended figure and one reweighted to estimated
production prevalence. Its self-check pins the size of the distortion — a 20-row
failure confined to one over-sampled rare intent costs 13.7pp of naive headline that
it would not cost in production.

The same divergence cuts the other way per intent: `ios_version_downgrade` (#11) and
`out_of_scope` (#13) are both intents where matching Apple's historical reply is the
wrong target, so agreement-with-history must be reported separately from correctness
for those two rows groups.

**Near-duplicate capping before any draw** (as in #7): exact-after-normalization
dedup on `customer_text` (48,968 → 48,199) then ≤2 rows per brand-reply near-dup
cluster (→ 32,295). The cluster cap is aggressive by design; it removed 15,904 rows.

**One conflict it caused, resolved by exempting one stratum.** Apple's
language-redirect reply is the most reliable `non_english` label source (taxonomy.md
§6), but it is a single near-dup cluster — so the ≤2/cluster cap cut it from 1,084
threads to **9**, nearly destroying the label source for that intent. `non_english`
candidates are therefore drawn after exact-dedup but before the cluster cap. Safe
here because the cap exists to stop one canned *brand reply* dominating, while these
customers' messages are all distinct.

**The #13 non_english rule does not survive contact with the full corpus.** Inside
the 304-row residual bucket it flagged nothing and measured 18/20 recall at 0/172
false positives. Applied to 32,295 rows it returned a pool that was **1/20 correct**
on hand-check. Two causes: terse English tweets carry no word from the deliberately
narrow function-word list ("mail app on crashes consistently due to new updated
restart off on etc no use" contains none of them), and `NON_LATIN_RE` runs on raw
text, so an Arabic *hashtag* on an English message trips it. Absence-of-English is a
weak signal at scale. Replaced for sampling by positive evidence — the brand's
language-redirect reply OR `label_intents.FOREIGN` — which hand-checked **20/20**.
The rule is left in place for the residual bucket, where it is applied to few rows
and its errors are visible; it should not be used as a corpus-scale detector.

Detector precision measured against hand-check (370 candidates, 244 confirmed):

| source | precision |
|---|---|
| `non_english` positive-evidence | 20/20 (100%) |
| `battery_drain` (qwen) | 20/22 (91%) |
| `billing_account` (qwen) | 22/27 (82%) |
| `blind_spot` keyword pass | 23/30 (77%) |
| `software_feature_defect` (qwen) | 76/109 (70%) |
| `out_of_scope` keyword pass | 32/60 (53%) |
| `ios_version_downgrade` regex | 27/45 (60%) |
| `general_complaint_nonactionable` (qwen) | 24/57 (42%) |
| `non_english` absence-of-English rule | 1/20 (5%) |

**`out_of_scope`'s praise sub-kind does not exist in this corpus: 0/15.** Every
"thanks"/"love my iPhone" match was sarcasm ("Thanks for destroying my wife's
iPhone"), a polite sign-off on a real complaint, or praise bundled with a request.
Genuine unsolicited praise with no ask is real but very rare — one row in the n=500.
taxonomy.md §7 still lists it as a kind; the golden set will be composed of the other
three (warranty, phishing, meta) unless someone hand-finds praise rows.

**`general_complaint_nonactionable` is the one stratum that came back short: 24
confirmed against a target of 42.** At 42% precision the model's residual bucket is
mostly other intents — overwhelmingly the letter-I keyboard bug (which is
`software_feature_defect`) and battery complaints. This is the same recall problem as
#10 seen from the other side, and it means the corrected n=500 figure of 32.2%
residual is itself likely an overestimate. Backfill is available without new model
calls (745 unused residual-labeled rows in the relabeled pool), but at 42% precision
it costs ~45 more hand-checks per 18 rows.

### 15. The residual bucket was NOT inflated: precision-only reasoning was the error

After #14 I reported that the corrected n=500 residual share of 32.2% was "likely an
overestimate", reasoning from the 42% hand-check precision on qwen's residual
predictions. **That inference was invalid and the conclusion was wrong.**

Precision tells you how many rows leave a bucket. It says nothing about how many
arrive. Rows flow both ways, and the inbound flow here is large: 23.9% of everything
qwen labelled `software_feature_defect` is truly residual, and that label is 35.2% of
the corpus. Estimating prevalence needs the full confusion matrix, not a diagonal.

`scripts/derive_prevalence.py` builds it from the unbiased 1,800-row random pool.
Within each predicted label the hand-checked rows are a random subsample, so they
estimate P(true | predicted), and prevalence(k) = Σ_j share(j)·P(true=k | pred=j).
Result, against the n=500 figures:

| intent | n=500 | corrected | delta |
|---|---|---|---|
| `general_complaint_nonactionable` | 32.2% | **37.1%** | **+4.9pp** |
| `software_feature_defect` | 39.2% | 35.3% | −3.9pp |
| `battery_drain` | 9.6% | 11.3% | +1.7pp |
| `billing_account` | 12.0% | 8.7% | −3.3pp |
| `out_of_scope` | 1.8% | **5.0%** | **+3.2pp** |
| `non_english` | 4.0% | 2.2% | −1.8pp |
| `ios_version_downgrade` | 1.2% | **0.5%** | −0.7pp |

So residual was *under*-stated, not inflated. The real limitation of the bootstrap
correction is not that it is biased in one direction — it is that a labeler's error
rate gives no directional information at all about prevalence, and every number in
#10 through #13 derived from bucket precision inherits that.

**Two consequences worth stating in the report's misleading-number section.**
`out_of_scope` is ~5% of traffic, nearly 3x its n=500 estimate, so the bucket added in
#13 is not the rounding error it looked like. And `ios_version_downgrade` at 0.5%
means the golden set's 18-row floor over-samples it **18x**, not the ~9x planned —
the strongest single argument for never headlining a blended number.

`eval_report.PREVALENCE` now carries these figures. They are still bootstrap
quality: the confusion matrix rests on one annotator (me) over 265 pool rows, and
`out_of_scope` / `ios_version_downgrade` use precision measured on their own detector
pools rather than on the random pool, with the miss mass assigned to residual.

### 16. `non_english` is exempt from the ≤2-per-cluster near-duplicate cap (final)

Flagged in #14, now settled as a deliberate rule rather than a deviation. The cap
exists to stop one canned **brand reply** from dominating a sample. Apple's
language-redirect template is exactly such a reply — and it is also the most reliable
`non_english` label source in the corpus, so the cap cut that source from 1,084
threads to 9. The customers' own messages behind those replies are all distinct, which
is the thing the golden set actually samples, so the cap protects nothing here while
destroying the stratum. `non_english` candidates are therefore drawn after
exact-after-normalization dedup but before the cluster cap. The exemption is specific
to this intent and its justification is the identical-reply/distinct-message split; it
does not generalise to intents whose *customer messages* cluster.

### 17. Golden eval set finalized at 200 rows

`data/processed/golden_set.pkl` (+ `.csv`), built by `scripts/build_golden.py` from
370 hand-checked candidates (244 confirmed, record in
`data/processed/golden_handcheck.pkl`).

| intent | n | share | bug_kw |
|---|---|---|---|
| `software_feature_defect` | 67 | 33.5% | 10 (15%) |
| `general_complaint_nonactionable` | 54 | 27.0% | 2 (4%) |
| `billing_account` | 20 | 10.0% | 0 |
| `ios_version_downgrade` | 18 | 9.0% | 0 |
| `out_of_scope` | 18 | 9.0% | 0 |
| `battery_drain` | 16 | 8.0% | 0 |
| `non_english` | 7 | 3.5% | 0 |

The 54-row residual block is 42 general + 12 `blind_spot` (retail/Genius Bar logistics
and hardware damage), carried as `sub_stratum` so the known gap stays visible instead
of dissolving into the bucket. `out_of_scope` is 6 warranty + 6 phishing + 6 meta;
praise contributes 0 (see taxonomy.md §7). Incident rows are 6.0% overall, 15% within
`software_feature_defect` — at the #6 ceiling, under it everywhere else.

Every row is one annotator's judgement. Before these numbers are used to make a
claim about the agent, a second pass on a sample of them is worth the time,
particularly for `general_complaint_nonactionable`, where the accept rate across two
draws was 40% and the boundary against `software_feature_defect` is where nearly all
the disagreement lives.
