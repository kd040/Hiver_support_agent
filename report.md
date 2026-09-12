# AppleSupport AI Customer-Support Agent — Report

## 1. Problem Framing

### What "good" means for this brand

AppleSupport's Twitter presence is not primarily a resolution channel — it is a
triage layer. Across the cleaned corpus, the single most common brand action
(59.2% of terminal replies) is a handoff to DM, and even within replies that
look self-contained, over half (§ data prep) ultimately point the customer
elsewhere. Given that, "good" for this agent is not defined as "resolves the
customer's problem in 280 characters." It is defined as:

1. **Correctly identify what the customer needs** (intent), even when the
   message is short, angry, or non-actionable.
2. **Answer directly, and only, when a direct answer is both possible and
   safe** — grounded in either historical precedent or a verified policy
   fact, never fabricated.
3. **Escalate deliberately, with a stated reason, whenever a direct answer
   is not possible or not safe** — including cases where the historical data
   shows Apple's agents answered directly but shouldn't have (see the
   `ios_version_downgrade` and prior `battery_drain` findings).

This framing has a direct consequence for how the headline numbers in this
report should be read: **a high auto-handle rate is not, by itself, evidence
of a good system.** A system that auto-handles everything by asking a vague
diagnostic question would score well on "percentage handled" and be useless.
Section 4 quantifies this distinction explicitly.

### What we chose not to build

- **Multi-language support.** ~4% of traffic is non-English. The agent
  detects this reliably (rule-based, not LLM-based — see decisions #13,
  #32–33) and always escalates. No translation or non-English reply
  generation was built.
- **Real-time / streaming interaction.** The system processes one message
  at a time as a batch classification-and-draft pipeline; it does not
  maintain conversational state across a multi-turn thread beyond the
  single customer message being answered.
- **Image/attachment handling.** Several customer messages in the corpus
  reference screenshots or photos (e.g. of a cracked screen); the agent
  reasons only over text.
- **Account or payment actions.** The agent never attempts to perform,
  authorize, or advise on a specific account/payment action — this entire
  intent (`billing_account`) escalates by policy, regardless of what a
  drafted reply might have said (decisions #30).
- **A general-purpose Apple support agent.** This system is scoped
  specifically to the seven intents discovered and validated against this
  corpus (§2, taxonomy.md). It is not designed to handle arbitrary Apple
  support queries outside that taxonomy — out-of-distribution messages are
  expected to fall into `general_complaint_nonactionable` or
  `out_of_scope` and escalate, not receive a best-effort generated answer.

## 2. Results vs. Baselines

All numbers below are measured against the 200-row golden evaluation set
(§3). Per-intent figures are primary; blended figures are shown only where
explicitly labeled, per the framing in §1.

### Baseline 1: Trivial

Majority-class intent (`software_feature_defect`, 33.5% of the golden set)
+ a canned "we'll look into it" reply for every message + always escalate.

- Intent accuracy: 33.5% naive / 35.3% reweighted (correct only on the
  majority class, 0% elsewhere)
- Reply quality: not applicable — every reply is identical and generic
- Escalation: 100% escalate rate — technically "safe," but abdicates the
  auto-handle requirement entirely

### Baseline 2: Simple (TF-IDF + logistic regression)

TF-IDF features + logistic regression intent classifier, trained on the
qwen-relabeled random pool (held out from the golden set); nearest-neighbor
reply retrieval, no LLM involved anywhere.

| Intent | Balanced accuracy | Unweighted accuracy |
|---|---|---|
| software_feature_defect | 73.1% | 61.2% |
| general_complaint_nonactionable | 70.4% | 88.9% |
| billing_account | 75.0% | 35.0% |
| battery_drain | 75.0% | 43.8% |
| ios_version_downgrade | 66.7% | 0.0% |
| out_of_scope | 22.2% | 0.0% |
| non_english | 14.3% | 0.0% |
| **Blended** | **68.6%** | **65.5%** |

The unweighted model's headline number (65.5%, driven by 88.9% on the
majority-adjacent residual class) is a textbook case of the framing in §1:
it scores zero on three of seven intents, having effectively learned to
answer "residual" and coast. This is the same failure mode a naive
auto-handle-rate headline would produce at the system level — one more
reason per-intent reporting is treated as primary throughout this report.

Retrieval-only reply quality was not separately scored at this stage; a
gated-retrieval quality metric (cosine similarity + resolution-type
compatibility) was developed for the full pipeline instead (§3), since raw
cosine similarity was shown to overstate usable-grounding rate by roughly
4x (91% naive vs. 19% gated, corpus-wide — see Failure Mode 1, §4).

### Full pipeline

| Component | Result |
|---|---|
| Intent classification (independently-sourced strata, unbiased) | 70.9% |
| Intent classification (rule-gated: non_english, ios_version_downgrade) | 100% |
| Intent classification (qwen-selected strata — circular, not quotable) | 100.0% |
| Escalation decision accuracy (taxonomy-conformance check) | 94.0% (188/200) |
| Reply quality — relevance (human-scored, n=24) | 4.46 / 5 |
| Reply quality — groundedness (human-scored, n=24) | 4.54 / 5 |
| Reply quality — tone (human-scored, n=24) | 4.88 / 5 |
| Reply quality — correctness (human-scored, n=24) | 4.39 / 5 |

The full pipeline beats both baselines on every intent where a direct
comparison is meaningful, and — more importantly given §1's framing — adds
two capabilities neither baseline has at all: a stated-reason escalation
decision, and a grounding-quality gate that prevents drafting when no
usable historical or policy grounding exists (rather than forcing an answer
from the nearest, possibly irrelevant, neighbor).

Every number in this table carries a caveat addressed directly in §4 —
several of them are the report's central findings, not footnotes.

## 4. Failure Analysis: Top 5 Failure Modes

### 1. Cosine similarity overstates usable grounding (retrieval)

**Example:** A raw cosine-similarity retrieval metric reported 91% hit-rate
for "found a usable grounding candidate in the top 5." Once gated by
resolution-type compatibility and same-topic match, the real figure was
19% corpus-wide (77% once retrieval pools were corrected to be per-intent).
The `non_english` pool's highest median similarity (0.443) turned out to
be near-*language* neighbors, not near-*answer* neighbors.

**Hypothesis:** Lexical/semantic proximity on short text (tweets) tracks
topic and register far more than it tracks "this reply would actually help
this customer." A high-similarity neighbor can share vocabulary, length,
and emotional tone with the query while addressing a completely different
underlying issue. Any retrieval-grounded system built on raw similarity
without a downstream compatibility check will silently overstate its own
groundedness.

### 2. LLMs treat procedural instructions as suggestions, not constraints

**Example:** A stated policy fact for `battery_drain` explicitly specified
"Low Power Mode: **on**." Across three escalating prompt-level constraints
(allow-list, plain instruction, explicit named prohibition), the drafting
model inverted this in 5 of 6 mentions — telling customers to make sure
Low Power Mode was **off**. The same model handled a single-proposition
fact (`ios_version_downgrade`: "not supported once signing ends") with
zero defects.

**Hypothesis:** A single factual assertion competes with the model's prior
on roughly equal footing and usually wins when explicitly stated. A
multi-step procedure invites the model to reorder, substitute, and "improve"
based on its training prior, because it pattern-matches to a *type* of
instruction (battery troubleshooting) rather than treating the specific
steps as fixed. Prompt-level constraints do not reliably suppress this;
only a deterministic template eliminated the defect (verified: 0/14 after
the fix vs. up to 14/14 across three prompt attempts).

### 3. The LLM-as-judge has no measurable relationship to reply quality

**Example:** The judge's groundedness score averaged 4.84/5 on both the 19
drafts a deterministic specificity check flagged as containing unsupported
claims, and the 122 it did not — identical to two decimal places.
Independently, judge-vs-human agreement on a 36-row hand-scored subset
produced Cohen's kappa ≤ 0 on relevance, groundedness, and correctness
(negative kappa means worse than chance). The judge awarded a flat 5/5 on
correctness to every scored row; the human mean was 4.39.

**Hypothesis:** The same model (llama3.1:8b) both drafted and judged the
replies, with no independent check against the grounding text — a
self-preference/self-consistency effect with nothing to break it. Score
collapse (correctness only ever used {3,4,5} across 141 drafts) compounded
this by removing most of the scale's discriminative range.

### 4. The taxonomy has a documented, unclosed coverage gap

**Example:** Retail/hardware-logistics messages (cracked screens, water
damage, Genius Bar appointments) have no home in the seven-intent taxonomy.
They are filed under `general_complaint_nonactionable` "for want of
anywhere better" (decisions #20), where the system correctly asks a
diagnostic question — a technically-correct action per §5 that resolves
nothing for a customer whose actual issue is physical damage requiring an
in-person repair.

**Hypothesis:** The taxonomy was built from clustering + hand-naming on a
finite sample; low-frequency-but-real intents (this one estimated at
roughly 2.7% of the residual bucket, i.e. under 1% of all traffic) are
easy to miss during discovery and only surface once deliberately probed
for. The `out_of_scope` bucket (added in decisions #13) exists precisely
because a symmetric gap was caught in time; this one was caught but
deliberately left unclosed as a documented scope decision, not an oversight.

### 5. Small-sample validation repeatedly looked correct and wasn't

**Example:** Four separate times during this build, a measurement taken on
a small sample (18–304 rows) was later contradicted at full or larger
scale: the corrected residual-intent prevalence (32.2%→37.1%, direction
initially mis-inferred from precision alone); non-argmax retrieval usage
(33%→73%); the `non_english` detection rule (18/20 recall on 304 rows →
1/20 correct at 32,295-row scale); and a battery drafting fix that passed
design review and looked sound before failing verification at n=14.

**Hypothesis — and this is the most important one in the report:** each of
these four had a *different* root cause — a reasoning error (precision
conflated with prevalence), a biased/non-representative sample, a rule
validated against the wrong base rate, and a design that was unverifiable
by inspection alone. **"Use a bigger sample" would only have caught one of
the four.** The actual mitigation that worked every time was building a
cheap, repeatable verification step (a script, a test, a recomputation)
rather than trusting a plausible-looking result — the same lesson as
failure mode 3, generalized.

## 5. What Is Misleading About My Headline Number

The single number most likely to be quoted from this project is the
**auto-handle rate: 70.5% (141/200)**. Read on its own, it implies "the
agent successfully handles 70% of incoming support requests." That reading
is wrong, in a specific and measurable way.

**`auto_handle` means "a reply was sent without a human in the loop." It
does not mean "resolved."** Decomposing the 141 auto-handled rows by what
actually happened:

| What auto_handle actually did | n | % of total |
|---|---|---|
| Answered directly from retrieved historical grounding | 64 | 32% |
| Asked a diagnostic question, resolving nothing | 45 | 22.5% |
| Answered from a verified, fixed policy fact | 18 | 9% |
| Emitted an identical canned template (battery) | 14 | 7% |

At most **96 of 200 rows (48%) receive anything resembling a substantive
answer**, and of those, 14 are the exact same templated text. The other 45
— nearly a third of everything counted in the celebrated 70.5% — are the
system correctly following its own taxonomy by asking a clarifying
question, which is the right behavior for an ambiguous message, but is not
resolution by any reasonable definition. Quoting 70.5% as a success or
resolution rate would be the single most misleading statement this report
could make.

**This was not a one-off naming slip; it is a pattern in this project.**
Every other headline metric produced during the build had a version of the
same problem, each for a different underlying reason:

- **Intent classification accuracy: 100.0%** on the strata drawn from the
  qwen relabeler's own predictions — impossible given the same relabeler's
  independently-measured ~70% precision, because the golden-set rows for
  those strata were sampled *from* the relabeler's output and kept only
  where a human agreed. The honest, non-circular figure — measured on
  independently-sourced rows — is **70.9%**.
- **Escalation decision accuracy: 94.0%.** This is close to a tautology:
  the router is scored against taxonomy rules it directly implements, so
  most of the 94% reflects "the code does what the code says," not
  validated correctness. Only the 12 disagreements (6%) carry real
  information, and half of those are arguably the taxonomy being wrong,
  not the router.
- **Retrieval hit-rate: 91%** under a cosine-similarity-only metric,
  against **19%** once gated by whether the retrieved reply was actually
  usable (§4, failure mode 1) — a 4.8x overstatement from a single
  unstated assumption (similarity implies usefulness).
- **LLM-judge reply-quality scores** (means of 4.84–5.00 across
  dimensions) looked like strong, clean evidence of quality until checked
  against a deterministic ground-truth signal and an independent human
  rater, both of which showed the judge carries no real signal (§4,
  failure mode 3).

**The general lesson, stated once and meant to generalize:** every number
in this pipeline that was constructed from, filtered by, or evaluated
against its own upstream component's output looked better than the
number obtained once measured against an independent source — the golden
set's independently-sourced rows, a deterministic check, or a human rater.
The gap was not small in any instance (30 points on intent accuracy, 4.8x
on retrieval, the entire judge dimension). **Any single metric in this
report that is not explicitly marked as measured against an independent
source should be treated as an upper bound, not an estimate.**

## 6. What I'd Do With One More Week

In priority order — each chosen because it addresses a *measured* gap
from this report, not a hypothetical one:

1. **Second annotator pass on the golden set's weakest boundary.**
   Self-consistency between `general_complaint_nonactionable` and
   `software_feature_defect` measured at 57.9% (§4-adjacent finding, not
   detailed above for space) — low enough that a second labeler, blind to
   the first pass, would meaningfully tighten the golden set's reliability
   on its two largest intents.
2. **Close or formally document the retail/hardware taxonomy gap
   (failure mode 4).** Either add an eighth intent with its own resolution
   policy (likely: always escalate, hardware safety), or keep it inside
   `general_complaint_nonactionable` but add a hardware-symptom override
   so at least the diagnostic-question default doesn't fire on messages
   that clearly need a different response.
3. **Fix G3 (topic-mismatch) in retrieval, not just work around it.**
   The current system handles this by letting the drafter decline to
   ground rather than forcing a bad match — safe, but it costs real
   coverage (§4, failure mode 1). Intent-conditioned retrieval was
   scoped and costed but not built; trying it for the two intents whose
   same-intent pools stay large enough to survive it
   (`software_feature_defect`, `general_complaint_nonactionable`) is the
   most promising untried lever.
4. **Re-run judge-vs-human agreement with a fixed judge design.** The
   current judge fails specifically because it shares a model with the
   drafter and has no explicit grounding-comparison step. A cheap next
   experiment: a different model as judge (even `qwen2.5:3b`, already
   available locally), with the grounding text and draft presented for
   explicit side-by-side comparison rather than holistic scoring.
5. **A second hand-labeled pool draw to re-derive intent prevalence.**
   The current prevalence estimate rests on one ~1,800-row qwen-relabeled
   pool; given how often numbers moved at scale during this build (§4,
   failure mode 5), a second independent pool would either confirm or
   further correct the current 37.1%/35.3%/etc. distribution before it's
   relied on for anything beyond this report.

## 7. Decision Log

The full list of 34 non-obvious decisions made during this build, with
reasoning, is maintained in `decisions.md` in the repository root. It is
referenced throughout this report by number (e.g. "decisions #21") rather
than duplicated here, since several entries are directly load-bearing for
claims made in §4 and §5 and are easier to audit in their original,
chronological form.



