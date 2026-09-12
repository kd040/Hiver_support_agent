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

### 18. Boundary reliability spot-check: 57.9% concordance, and the boundary is where it lives

The golden set is **one annotator's judgement** (mine) on 370 candidates. That is the
single largest limitation on every number derived from it, and it deserves a measured
figure rather than a disclaimer.

**Scope.** The 121 golden rows on the known-weak boundary — 54
`general_complaint_nonactionable` + 67 `software_feature_defect`. A second judgment was
produced by `scripts/second_pass.py`: qwen2.5:3b prompted with the intent definitions
and examples transcribed from `docs/taxonomy.md`, blind to the golden label.

**Why not the original prompt.** The golden labels for these strata are rows where the
annotator *agreed* with a qwen candidate label — mismatches were rejected during
hand-check. Re-running the `llm_relabel.py` few-shot prompt at temperature 0 would
therefore reproduce its own earlier answer and report ~100% agreement as a pure
artifact. Hence a different procedure: doc-derived definitions, no few-shot block, and
none of the #12/#13 post-rules.

**Result: 53.7% raw, 57.9% with the #12/#13 rules applied** (70/121). Per label, rules
applied: `general_complaint_nonactionable` 63.0%, `software_feature_defect` 53.7%.

**Disagreement does concentrate where expected — 45 of the 51 disagreements (88%) are
residual↔defect** (16 residual→defect, 29 defect→residual). Two smaller clusters are
worth naming: 4 residual rows were called `out_of_scope`, consistent with #15's finding
that `out_of_scope` is ~5% of traffic and under-detected; and 1 defect→`billing_account`.
Before the post-rules were applied, 18 rows spilled into `ios_version_downgrade` —
the unguarded over-firing documented in #12, which is evidence the precondition earns
its place rather than evidence about the boundary.

**The worst subset is the one genuinely independent of qwen: `blind_spot` at 5/12
(41.7%).** Those rows were sourced by keyword, not by a qwen label, so no agreement is
baked in — and they are retail/Genius Bar and hardware-damage messages whose golden
label is residual only because the taxonomy has nowhere else to put them
(taxonomy.md "Still not in scope"). Low concordance there is the documented gap
showing up as measured noise.

**How to read 57.9% honestly.** It is concordance between two procedures, one of which
is a 3B model reading definitions — a weak annotator. It does **not** establish that
58% of the golden labels are correct; a weak second annotator drags the number down on
its own. What it does establish is a floor on reliability and a clear location for the
problem: the residual/defect line is not a minor edge, it is where a second procedure
diverges from the first on roughly a third of rows. Treat the 67/54 split as
provisional, and report `software_feature_defect` and
`general_complaint_nonactionable` accuracy with this caveat attached.

The 51 disagreements are written to `data/processed/second_pass_disagreements.csv` for
a human second pass, which is the measurement that would actually settle it.

### 19. Baselines to beat: trivial majority-class and TF-IDF + logistic regression

`scripts/baselines.py`, scored by `scripts/eval_report.py` against the 200-row golden
set. Neither uses an LLM at inference time. The classifier trains on the
qwen-relabeled random pool with all 200 golden `tweet_id`s removed (1,655 rows), and
golden ids are removed from the retrieval pool too — otherwise a golden message
retrieves its own thread.

**Baseline 1, trivial.** Majority intent (`software_feature_defect`, 33.5% of the
golden set) for every row, the canned reply "we'll look into it", always escalate.
Per-intent: 100% on `software_feature_defect`, **0% on the other six**. Naive blended
33.5%, prevalence-reweighted 35.3%. Worth noting the majority class differs by source:
the *training pool's* majority is `general_complaint_nonactionable`, so the same
trivial strategy fitted on the pool instead of the eval set would score 0% on defect
and 27% overall — the baseline is sensitive to which frame you read "majority" from.

**Baseline 2, TF-IDF + logistic regression** (word 1-2 grams, sublinear tf, min_df 2,
C=2.0), with nearest-neighbour reply retrieval over the 7,993-row deduped answer pool.

| intent | n | balanced | unweighted |
|---|---|---|---|
| `software_feature_defect` | 67 | 73.1% | 61.2% |
| `general_complaint_nonactionable` | 54 | 70.4% | **88.9%** |
| `billing_account` | 20 | 75.0% | 35.0% |
| `battery_drain` | 16 | 75.0% | 43.8% |
| `ios_version_downgrade` | 18 | 66.7% | **0.0%** |
| `out_of_scope` | 18 | 22.2% | **0.0%** |
| `non_english` | 7 | 14.3% | **0.0%** |
| naive blended | | 65.5% | 51.5% |
| reweighted | | 68.6% | 62.5% |

**`class_weight` is the whole story, and it is a per-intent-metric argument in
miniature.** Unweighted scores its best single number on the largest intent (88.9% on
residual) while returning **zero** on three intents — it has learned to answer
"residual" and coast. Balanced trades 18pp of residual accuracy for non-zero coverage
everywhere. A blended headline rewards the collapsed model on two of the three
aggregate figures; per-intent makes the collapse visible immediately.

Three results worth carrying into the report:

- **The rules beat the learned model on `non_english`: 14.3% here vs ~100% for the
  #13 positive-evidence detector.** With 6 training examples TF-IDF cannot learn a
  language boundary. The baseline deliberately excludes the rules, so this is a
  measure of what the rules are worth, not a defect in them.
- **`out_of_scope` at 22.2%** on 58 training examples is the weakest learned intent,
  consistent with #15's finding that it is ~5% of traffic and under-detected.
- **The classifier scores about as well as the labels it trained on.** qwen's
  hand-checked precision was 70% on defect and 47% on residual (#14), yet the
  classifier hits 73.1% and 70.4% against hand labels. TF-IDF appears to smooth the
  teacher's noise rather than inherit it, which is worth remembering before treating
  noisy-teacher training as disqualifying.

**Retrieval is reported descriptively, not scored — there is no reply metric yet.**
Median cosine similarity to the retrieved neighbour is 0.288, and 7% of rows retrieve
essentially nothing (sim < 0.20). Highest median similarity is `non_english` at 0.443,
which is not a success: those messages match each other lexically and the pool's
replies to them are language redirects or generic acknowledgements, so a high score
there means the retriever has found the deflection template. Any future reply metric
has to separate "retrieved a similar thread" from "retrieved a usable answer".

### 20. `blind_spot`'s 41.7% concordance is the taxonomy gap, not a labeling defect

The 12 `blind_spot` rows scored worst in #18's reliability check (5/12). Accepted as
correct behaviour of the measurement, not a problem with the labels.

These are retail/Genius Bar logistics and hardware-damage messages. Their golden label
is `general_complaint_nonactionable` only because the taxonomy has nowhere else to put
them (taxonomy.md "Still not in scope") — they are not vague complaints, which is the
bucket's actual definition. A second annotator reading the definitions will therefore
disagree, and *should*: "I dropped my iPhone and the screen shattered" is not a
non-actionable complaint by any reading of §5. The 41.7% is the cost of routing rows
through a bucket they do not belong to, surfacing as measured disagreement.

Consequence: the `blind_spot` sub-stratum is the empirical case for eventually giving
these their own intent. It is deliberately carried as `sub_stratum` in the golden set
so the disagreement stays attributable rather than diffusing into the residual number.
Do not "fix" it by relabelling those rows; fix it by adding an intent, if the report's
scope allows.

### 21. `non_english` and `ios_version_downgrade` are rule-gated in the real pipeline, not model-decided

Architecture decision, grounded in measured numbers rather than preference.

`non_english`: the learned classifier scored **14.3%** on it (#19) against **~100%**
for the positive-evidence rule (#13). Six training examples cannot teach a language
boundary, and the unguarded LLM scored **0/14** on it (#13). Three procedures have now
been measured on this label and the cheapest one is the only one that works.

`ios_version_downgrade`: the unguarded LLM produced ~7 false positives per 10
predictions (#12) and re-fired on 18 of 121 boundary rows when the precondition was
removed (#18). The regex precondition is what holds it at 1.2% rather than an inflated
2.8%.

So in the pipeline both labels are decided before any model call: script-range plus
function-word evidence for `non_english`, explicit backwards-movement language for
`ios_version_downgrade`. The model is asked only to choose among the remaining five
intents. This also removes both intents' routes from the LLM's failure surface
entirely — `non_english` is always-escalate with no reply logic, and
`ios_version_downgrade` answers from a stated policy fact rather than retrieval (#11),
so neither needs generation at all.

Cost of the decision, stated plainly: the rules are recall-limited. The `non_english`
rule missed 2 of 20 known rows (#13) and is a poor corpus-scale detector on its
absence-of-English half (#14) — it is used with the positive-evidence source in front
of it. Rule misses land in the residual bucket, which is the safe direction.

### 22. Retrieval metric: cosine gated by resolution-type compatibility, and the 73pp gap it exposed

`scripts/retrieval_metric.py`. Cosine similarity alone counts any lexically-similar
thread as a hit regardless of whether its reply is the right KIND of reply. Four gates,
all of which must pass for a candidate to count as usable grounding:

| gate | test |
|---|---|
| G1 lexical | `cos_sim >= 0.20` |
| G2 form | retrieved `resolution_type` is acceptable for the query's intent |
| G3 topic | retrieved thread's own intent matches the query's intent |
| G4 template | not the language-redirect template, unless the query IS `non_english` |

Combined score: `gated_score = max over top-k of (cos_sim if all gates pass else 0)`;
`hit@k` is `gated_score > 0`. Acceptable types per intent come from taxonomy.md's
"Good resolution" lines, and retrieval policy is **not uniform** — averaging over it is
what hid the problem. Three groups: `required` (retrieval supplies the answer's
content: defect, battery, residual), `escalate` (route fixed by policy, retrieval
supplies at most redirect wording: billing, non_english), `none` (retrieval must be
bypassed: `ios_version_downgrade` per #11, `out_of_scope` per taxonomy §7).

**Measured on a 67-row stratified golden sample, k=5:**

| pool | old hit@5, all | new hit@5, all | old, `required` | new, `required` |
|---|---|---|---|---|
| full corpus, all resolution_types | 91% | **19%** | 93% | **20%** |
| answer-type only (#7 shape) | 90% | 25% | 90% | 57% |
| **per-intent pools (fix)** | 85% | **77%** | 87% | **80%** |

Mean raw cosine is 0.274; mean gated score is 0.020. Cosine-only was reporting
~90% retrieval success on a pipeline that could actually ground a reply for 20% of the
rows that need one.

**The structural cause: one pool cannot serve every intent.** An intent can only
retrieve an appropriate reply if the `resolution_type` it needs is *in* the pool.
decisions #7's answer-type pool is right for `software_feature_defect` (**100%** hit@5,
unchanged by gating) and fine for `battery_drain` (70%), and is structurally incapable
of serving the rest — it scores **0%** for residual, billing and `non_english` because
the types those intents need were filtered out when the pool was built. The corpus has
2,642 `clarifying_question` and 1,874 `other_channel_redirect` threads; the pool
contains none of them.

Splitting into per-intent pools fixes it: residual 0% → 70%, `non_english` 0% → 86%,
billing 0% → 60%, defect unchanged at 100%. G2 then fails 0% of the time by
construction. **This supersedes #7's single-pool design** — #7's dedup logic and its
≤3-per-cluster cap still stand, but the pool is built per intent from that intent's
acceptable resolution types.

**What remains broken is G3, at 54% failure even with per-intent pools.** The retrieved
thread is often about a different problem than the query. That is the retriever's
fault, not the pool's — TF-IDF on short tweets is weak — and it is the thing to improve
before drafting, since a topically-wrong neighbour with the right reply shape is
exactly the input that produces a confident wrong draft.

**G4 fired zero times and two earlier claims need correcting.** I previously wrote that
`non_english`'s high median similarity (0.443, #19) meant the retriever was finding the
deflection template. It is not: those queries retrieve genuine same-language
neighbours (French and Spanish messages), and their failure was the `resolution_type`,
not the template. Relatedly, the language-redirect replies split 611
`self_contained_answer` / 490 `other_channel_redirect`, so they are largely absent from
the answer-type pool and G4 had nothing to catch there. The gate is kept as cheap
insurance for the per-intent `non_english` pool, where those 490 rows now live, but it
has not yet earned its place empirically.

**Caveat on the numbers:** 67 rows, ~10 per intent, so each per-intent figure is ±1-2
rows of noise. Neighbour intents are qwen-labelled with the #12/#13 rules, so G3
inherits that labeller's error rate — the same limitation as #14's confusion matrix.

### 23. Retriever swapped to sentence-transformers embeddings: required hit@5 77% → 94%

`scripts/retrieval_embed.py`. all-MiniLM-L6-v2 replaces TF-IDF; per-intent pools, gates
G1-G4 and k=5 unchanged. Measured on the **full 200-row golden set** — 164 of them, since
`ios_version_downgrade` and `out_of_scope` are policy `none` and retrieve nothing (#11,
taxonomy §7). The TF-IDF arm was re-run on the same 200 rows so the comparison is not
confounded by the earlier 67-row sample.

**G1's threshold is not transferable between retrievers, and reusing it would have
faked the result.** TF-IDF cosine on short tweets sits near zero for unrelated pairs, so
TAU=0.20 was meaningful; MiniLM puts random pairs at mean 0.27-0.35, where 0.20 passes
**everything** — G1 fails 0.0% of candidates at that threshold. TAU is therefore
recalibrated per pool as the 95th percentile of random-pair similarity (0.508-0.591,
"closer than 95% of arbitrary pairs"), where G1 fails 8.0%. Both are reported below; the
calibrated figures are the real ones.

| metric | TF-IDF | embeddings |
|---|---|---|
| **`required` hit@5** | **77%** | **94%** |
| all-rows hit@5 | 77% | 93% |
| G1 lexical fail | 26.7% | 8.0% |
| G2 form fail | 0.0% | 0.0% |
| **G3 topic fail** | **50.1%** | **29.4%** |
| G4 template fail | 0.0% | 0.0% |

Per intent, new hit@5: `software_feature_defect` 75%→96%, `battery_drain` 62%→100%,
`general_complaint_nonactionable` 85%→91%, `billing_account` 70%→95%, `non_english`
86%→71% (7 rows, 2 fall below calibrated TAU — noise at that n).

**G3 improved by 21pp but is still 29.4%, and reporting hit@5 alone would hide what
that costs.** Rank matters:

- hit@5 (any of 5 usable) **93%**
- **hit@1 (top-1 usable) 68%**
- precision@5 65%, mean 3.26 of 5 candidates usable
- G3 fails **29.3% at rank 0** — identically to all ranks, so topic mismatch is *not*
  concentrated in the tail

So a usable neighbour is nearly always present in the top 5, but the single nearest
neighbour is topically wrong about one time in three. **Drafting from top-1 would
ground on the wrong thread ~30% of the time.** Whatever drafting does, it must be
handed several candidates and allowed to reject them, not the argmax.

Per the agreed sequencing, no third retriever was tried. Stating the remaining options
with their real costs rather than picking one:

- **Intent-conditioned retrieval is genuinely untried and is the direct fix for G3.**
  The per-intent pools filter on `resolution_type` only — they still contain threads of
  every intent, which is exactly what G3 catches. Filtering a pool to same-intent
  threads would drive G3 toward zero by construction. Cost: pool sizes collapse
  (indicative, TF-IDF classifier: defect 7,993→~2,800, residual 2,568→~1,076,
  battery 7,993→~447, `non_english` 1,282→~3), it needs intent labels for all ~35k pool
  threads, and it makes retrieval depend on the *predicted* query intent — 73% accurate
  for defect (#19) — so classifier error now compounds on both the query and the pool
  side. `battery_drain` and `non_english` would be starved outright.
- **Hybrid lexical+embedding** would likely help G1 more than G3, and G1 is no longer
  the binding constraint at 8%.
- **Document it as a limitation** and handle it at draft time by passing top-k with the
  gate flags attached, letting the drafter decline to ground. Given hit@5 93% vs hit@1
  68%, this extracts most of the available value without new machinery.

My read: the third option first, because it is free and the measurement says the
information is already there in the top 5. Intent-conditioning is worth trying only for
`software_feature_defect` and `general_complaint_nonactionable`, the two intents whose
same-intent pools stay large enough to be usable.

### 24. Gate thresholds are retriever-specific and must be recalibrated, never reused

Standalone finding, surfaced by #23 but general.

A similarity threshold is a property of the *similarity function*, not of the task. The
G1 lexical gate used `cos_sim >= 0.20` against TF-IDF, where unrelated short tweets
score near zero, so 0.20 meant "meaningfully more similar than nothing". Carried over to
MiniLM embeddings unchanged, the same number **failed 0.0% of candidates** — random
pairs in these pools score mean 0.27-0.35, so every candidate clears 0.20 and the gate
silently stops being a gate. Reusing it would have reported 98% required-hit@5 instead of
the real 94%, and the inflation would have looked like a win from the retriever swap.

The fix is to define the threshold in distribution terms rather than as a constant:
TAU = the 95th percentile of **random-pair** similarity within the pool being searched,
i.e. "closer than 95% of arbitrary pairs". Per-pool values came out 0.508-0.591 for
MiniLM against 0.20 for TF-IDF — a 2.5-3x difference for the identical semantic
criterion. Recalibrated, G1 fails 8.0%.

Generalisation for the rest of this project: **any absolute threshold inherited across a
component swap is a bug until re-derived.** This applies to the near-duplicate 0.95
cosine merge in #7 (calibrated for MiniLM, would need re-deriving for a different
encoder), and to any confidence cut-off the escalation step later introduces. Where a
threshold cannot be avoided, prefer a distributional definition that travels.

### 25. G3 topic mismatch (29.4%) accepted as a documented v1 limitation

Decision: do not chase a third retriever, and do not build intent-conditioned retrieval
in v1. Recorded as a known limitation with its measured cost, and as future work.

After the embedding swap, 29.4% of retrieved candidates are about a different problem
than the query, and the rate is identical at rank 0 (29.3%) — so it is not a tail
effect. The consequence is specific and bounded: **hit@5 93% against hit@1 68%.** A
usable neighbour is almost always in the top 5; the nearest one is wrong about a third
of the time. v1 therefore hands the drafter the top 5 with gate flags rather than the
argmax, which is free and captures most of the available headroom.

**Future work, deliberately not built.** Intent-conditioned retrieval — filtering each
pool to same-intent threads before similarity search — attacks G3 directly and would
drive it toward zero by construction. It was not built because the costs are real and
measured: pools collapse (defect 7,993→~2,800, residual 2,568→~1,076, battery
7,993→~447, `non_english` 1,282→~3, so two intents are starved outright); it requires
intent labels for all ~35k pool threads; and it makes retrieval depend on the *predicted*
query intent at 73% accuracy (#19), compounding classifier error on both the query and
the pool side. If revisited, it is worth trying for `software_feature_defect` and
`general_complaint_nonactionable` only — the two intents whose same-intent pools stay
large enough to search.

Hybrid lexical+embedding retrieval was also considered and dropped: it would mainly
improve G1, which at 8% failure is no longer the binding constraint.

### 26. Drafting (step 4): gated grounding, hard safety overrides, and measured fabrication

`scripts/draft_replies.py`, llama3.1:8b (generation quality matters here; qwen2.5:3b is
the classifier, not the drafter).

**Routing, in priority order.** Safety overrides are checked on the customer message
FIRST and beat everything, including a successfully grounded retrieval:

1. `escalate_override` — physical/hardware symptoms (swelling, heat, won't charge,
   physical damage), account/payment actions, phishing/scam reports. From taxonomy.md
   §3, §4 and §7, which state these escalate regardless of classifier confidence.
2. `no_draft_policy` — `non_english` (§6 always-escalate, no reply logic) and
   `out_of_scope` (§7 forbids auto-drafting from retrieval). No generation call.
3. `policy_fact` — `ios_version_downgrade`, answered from a stated fact, the one place
   the pipeline deliberately bypasses retrieval (#11).
4. `grounded` — ground in the **highest gated-score** candidate of the top 5.
5. `no_usable_grounding` — zero of top-5 pass the gates: no draft is forced, the row is
   flagged for the escalation step (step 5, not built).

**Grounding is the best gated candidate, not the argmax.** At n=200, rank 0 is used
88/122 times (72%) and a non-argmax candidate 34/122 times (28%) — so the choice matters
for a bit over a quarter of grounded drafts, concentrated in the harder ones. The first
18-row sample suggested 2 of 6 (33% rank-0) and was wrong: it deliberately over-sampled
zero-grounding rows. See #28.

**Route distribution over the full 200-row golden set:** `grounded` 122 (61%),
`policy_fact` 32 (16%: `ios_version_downgrade` 18 generated from a stated fact,
`battery_drain` 14 emitted as a canned template), `escalate_override` 20 (10%),
`no_draft_policy` 18 (9%), `no_usable_grounding` 8 (4%). Total abstention 48/200 (24%).
Zero era violations, zero links, mean 100 characters.

**One prompt-adherence failure survives, 1 of 122:** a 558-character draft that narrated
its own reasoning ("Let's try to draft a reply based on the grounding material. Since
the grounding material mentions...") rather than producing a reply. It is counted by the
over-280 check rather than suppressed, on the same principle as the specificity flag. At
temperature 0.2 this class of failure is rare but not zero, and a length check is the
cheapest guard if it matters at volume.

Escalate-override reasons across the 20: account/payment 7, phishing 7,
physical/hardware 6.

**Three defects found in the first 18-row sample and fixed:**

- Drafts copied `t.co` shortlinks out of the grounding (1 of 7). Grounding is now
  URL-stripped before it reaches the prompt, links are forbidden in output, and a regex
  backstop counts any that survive. Post-fix: 0.
- The blackened-charger-contacts row received a settings answer, violating §3's
  hardware-safety rule. The prompt encoded each intent's *target reply* but none of the
  *escalate-anyway* conditions. Now an override; post-fix that row escalates.
- Drafts stated settings paths absent from their grounding (2 of 7), including
  `Settings > Battery > Battery Health` — a path that shipped in iOS 11.3 beta and is
  anachronistic for a corpus centred on 11.0-11.2. Right-looking, era-wrong.

**Fabrication is flagged and counted, never silently rejected.** The specificity check
compares settings paths, version numbers and feature names in the draft against its
grounding text. Rate over the 200-row run: **19 of 122 grounded drafts (16%)**, and it
is confined to one intent — `software_feature_defect` 19/64 (30%),
`general_complaint_nonactionable` 0/45, `billing_account` 0/13. Residual and billing
flag at zero because their target replies are diagnostic questions and redirects, which
need no specifics at all. The check measures "unsupported by the grounding", not
"wrong" — most flagged paths are real — and those are different claims. Keeping it as a
rate rather than a filter is deliberate: the number belongs in the report.

`battery_drain` originally flagged 12/14 (86%) and is now a canned template with no
generation call, for the reasons in #29. Note that the specificity flag is **vacuous for
templates** — the draft is its own grounding, so its 0% is an artifact, not a result.
The real guarantee for the template is `tests/test_draft_guards.py`, which pins the two
defects that were actually found: no post-window features, and Low Power Mode stated
ON rather than inverted.

**Unintended interaction worth recording:** stripping links did not reduce fabrication,
it changed its shape. Previously the model answered the autocorrect question by
parroting a shortlink; with links removed it produced `Settings > General > Keyboard`
instead. That is a net gain in *observability* — an opaque unverifiable link became a
concrete claim the specificity check can catch — but not a gain in groundedness. Fixes
that remove a crutch tend to relocate the failure rather than delete it.

**Near-copying is a documented characteristic, not a bug.** Mean similarity between
draft and grounding is 0.55 (max 0.86). The system is substantially **extractive**:
it retrieves a past Apple reply and rewrites it. That is a reasonable design for support
replies and it is why an 8B model suffices, but it means draft quality is bounded by
retrieval quality, and credit for good drafts belongs mostly to the retriever. No change
made; stated so the report does not overclaim generation.

### 27. Abstention losing recall is G3 evidence, not a new problem

Two `no_usable_grounding` rows in the first sample should have had grounding: "fix this
I️ glitch thing" is the iOS 11.1 keyboard bug, for which the corpus holds 4,192
near-identical canned replies (#7), and a Files/iCloud Drive row was a well-formed
specific defect.

Recorded as **corroborating evidence for #25's G3 limitation, not a separate issue.**
The answer demonstrably exists in the pool; the retriever failed to surface it inside
the top 5 with a matching topic, which is exactly the 29.4% topic-mismatch rate already
accepted as a v1 limitation. No retriever change and no gate loosening: abstaining is
the safe failure direction, and loosening the gates to recover these rows would trade a
measured recall loss for an unmeasured correctness loss.

What this does change is how the number is reported. At n=200 the
`no_usable_grounding` rate is 8/200 (4%), and it is **not** a clean measure of
"questions the corpus cannot answer" — it is that plus G3 misses. Both belong in the report's misleading-number section, since quoting abstention
as a coverage figure would overstate how much of the corpus is genuinely un-groundable.

### 28. Four small-sample findings reversed at scale — a running methodological note

Not a one-off. Four times now, a number that looked settled on a small sample moved
materially, or reversed, when measured properly. Collected here because the pattern
itself belongs in the report's misleading-number section, and because it is the best
argument in this project for reporting n alongside every figure.

| finding | small-sample reading | at scale | direction |
|---|---|---|---|
| residual prevalence (#15) | "32.2% is inflated", from 42% bucket precision | **37.1%** via the full confusion matrix | reversed |
| rank-0 grounding use (#26) | 2 of 6 = 33%, on 18 rows | **99 of 136 = 73%** | 40pp |
| battery policy fact (#29) | sound in design, every step version-checked | **10 of 14 drafts defective** | failed |
| `non_english` absence rule (#14) | 18/20 recall, 0/172 FP on 304 rows | **1/20 correct** on 32,295 rows | collapsed |

The four fail in different ways, which is the useful part:

- **#15 was a reasoning error, not a sampling error.** Precision was measured correctly;
  the inference from precision to prevalence was invalid because rows flow both ways.
  A bigger sample would not have caught it — only the confusion matrix did.
- **#26 was a biased sample.** The 18 rows deliberately over-sampled zero-grounding
  cases to exercise the abstention path, which made argmax look far less useful than it
  is. The bias was introduced on purpose and then forgotten when reading the result.
- **#29 was a design that could not be validated by inspection.** The fact's content was
  verified step by step and was correct; the failure was entirely in whether the model
  would follow it, which only a run could reveal.
- **#14 was a threshold-free rule validated on the wrong base rate.** 0 false positives
  in 172 rows bounds the FP rate near 1%, which is fine at n=172 and catastrophic at
  n=32,295 where it swamps a 3.6% true class.

Practical rules adopted from this: quote n with every rate; never infer a distribution
from a precision; re-measure any number when the sample it came from was stratified or
deliberately skewed; and treat "the design is obviously right" as a hypothesis with a
cheap test, not a conclusion.

### 29. Stated facts work for propositions and fail for procedures; the fix is a template, not a stronger prompt

`ios_version_downgrade` is answered from a stated fact and its drafts are clean — zero
specificity flags, zero era violations. The same mechanism applied to `battery_drain`
failed completely, and the difference is the *shape* of the fact: downgrade's is a single
unambiguous proposition ("not supported once signing ends"), battery's was a four-step
procedure. Handed a procedure, the model treated it as suggestions — it dropped steps
(`Software Update` used 0 of 14 times), substituted a better-known step from its own
prior, and **inverted a polarity**: of 6 drafts mentioning Low Power Mode, 5 advised
turning it **off**, which is the opposite of the guidance and actively worsens battery
life.

Three escalating prompt-level constraints were tried and measured:

| attempt | flag rate | `Battery Health` occurrences |
|---|---|---|
| retrieval grounding (links stripped) | 12/14 (86%) | 11 |
| + stated policy fact listing allowed steps | **14/14 (100%)** | 12 |
| + explicit named prohibition of the term | 12/14 (86%) | 10 |
| **canned template, no generation** | **0/14** | **0** |

The allow-list made it *worse*. The explicit prohibition — naming "Battery Health" and
forbidding it outright — reduced occurrences from 12 to 10 and never reached zero. A
strong model prior ("battery question → Battery Health") survived every instruction
placed against it.

This is #21's lesson in a second place: where the model cannot be trusted on a specific
decision, the reliable fix is deterministic, not a better prompt. `battery_drain` now
emits a fixed, version-checked template with no generation call, justified by the
taxonomy itself — §3's good resolution is standard settings guidance that does not vary
by customer, which is a template by definition.

**The cost, stated plainly.** Battery replies are now identical for every customer, so
they cannot acknowledge a specific symptom, and the retrieval work for this intent
(100% gated hit@5) is discarded. Two of the 14 rows previously had clean grounded
drafts. Physical-symptom rows still escalate ahead of the template via the #26
overrides, which is what protects the safety case. A hybrid — template as the spine plus
retrieved grounding for tone — was not tried, and is the obvious next thing if
per-customer wording matters.

**Generalisation worth carrying:** prefer stating facts to the model when the fact is a
single assertion it can echo; prefer a template when the fact is a sequence whose order
or polarity carries the correctness. Checking whether a prompt constraint held is not
optional — two of the three attempts above looked reasonable and were measurably wrong.
