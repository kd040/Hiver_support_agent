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
