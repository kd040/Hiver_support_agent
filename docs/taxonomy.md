# Intent taxonomy — AppleSupport

Seven intents, locked. Six were derived from k-means over a 500-message stratified sample of
the cleaned AppleSupport corpus (`scripts/taxonomy_sample.py`), then merged and
named by hand — the raw clusters split the vague-complaint mass three ways by
length and tone rather than by intent, so they were collapsed. The seventh,
`out_of_scope`, was added later: LLM re-labeling surfaced a coherent group of
messages that are neither actionable complaints nor non-English (decisions.md #13).

All examples below are real messages from the cleaned corpus, after `@BRAND` /
`@USER` placeholder anonymization.

---

## 1. `software_feature_defect`

A **specific named component** plus a **symptom**. The component is what makes
this distinct from `general_complaint_nonactionable` — "my phone is broken" is
not this intent; "Siri doesn't work" is.

> Hey @BRAND thanks for the last update! This time Siri doesn't work. Any heads up on what you're planning to ruin on our iPhones with the next update?

> my phone deleted some of my contacts & wont let me add them back. Wtf @BRAND

> @BRAND my iPhone X keyboard is laggy/unresponsive outside in cold weather, is this a hardware issue?

**Good resolution:** a `self_contained_answer` pointing at the known workaround or
support article for that component. This is the intent with the most usable
grounding material, and the one where the agent should most often auto-handle.
Falls back to `clarifying_question` when the component is named but the symptom
is underspecified.

---

## 2. `ios_version_downgrade`

An explicit request to revert, roll back, or reinstall a previous iOS version.
Rare (≈1%) but distinctive, and it has a fixed correct answer.

> @BRAND iOS11 seems to be a disaster. Can I revert to 10? Fix coming? Help!

> @BRAND How do I get iOS 10 back on my 6s? My phone was in perfect working condition until I updated to iOS 11, now it barely functions & restarts every 5 mins.

> Hey @BRAND my iphone 5s with ios 11 is draining my battery 70% faster! I feel cheated like never before! Let me go back to ios 10 plz

**Good resolution:** a `self_contained_answer` stating the position honestly —
downgrading is not supported once Apple stops signing the old build. The answer
does not vary by customer, so this should auto-handle at high confidence. In the
corpus Apple in fact answers these with `dm_handoff` (4/4 in the sample), which
is a case where the historical behavior is *not* the target behavior.

---

## 3. `battery_drain`

Battery life, drain rate, or charge-cycle complaints. Kept separate from
`software_feature_defect` because it is high-volume, has its own diagnostic path,
and its symptom vocabulary barely overlaps with the rest.

> iOS update 11.1.2 is draining my 7 plus battery like crazy. Went to bed at 85%, woke up at 11%. @BRAND

> @BRAND y'all need to fix this battery shit. Cause ain't no way I should take a nap with my phone on 100% and wake up 2 hours later to 10% 😑

> @BRAND any way to check the overall battery health of my iPhone 5s?

**Good resolution:** `self_contained_answer` with battery-health check steps and
the standard settings guidance. Escalate when the customer reports a physical
symptom (swelling, heat, refusing to charge) — that is a hardware safety path,
not a settings answer.

---

## 4. `billing_account`

Purchases, iTunes / Apple Music, subscriptions, refunds, Apple ID, orders and
pre-orders, account access.

> @BRAND hi I've just subbed to Apple Music and the payment has been taken but Apple Music is saying I'm not subscribed.

> @BRAND won't let me log in I want to cancel subscriptions. The verifications codes I keep being sent aren't working

> @BRAND Help!! I need to update the payment method for my iPhone X pre-order. It won't let me do it online

**Good resolution:** `other_channel_redirect` or `dm_handoff` — almost never a
public self-contained answer, because resolving it requires account access.
This intent should escalate by policy regardless of classifier confidence:
account and payment actions are not safe to auto-handle. Phishing and scam reports
were previously filed here; they now belong to `out_of_scope` (§7), since the
customer is reporting something rather than asking for account action. A phishing
message that *also* asks about a real charge stays `billing_account`.

---

## 5. `general_complaint_nonactionable`

Frustration about an update or the brand with **no specific component and no
actionable request**. The residual bucket by design.

> So @BRAND when y'all gonna fix my phone... I just got this last December... It don't need to be messin' up. You hear me? #Apple

> @BRAND I am absolutely fuming with your technology. I'd like to make a complaint and try to resolve this problem with some help please

> @BRAND I started updating my phone at 5pm yesterday. This has been the worst experience moving to iPhone X. In 10 years, I've never had this much trouble.

**Good resolution:** `clarifying_question` — the only honest move is to ask what
specifically is wrong. This is the one intent where a `dm_handoff` is arguably
the *correct* behavior rather than a deflection, since there is nothing to ground
a reply in. Auto-handling should mean "ask one good diagnostic question", not
"draft a fix".

---

## 6. `non_english`

Message is not in English. ≈4% of the corpus (Spanish, Portuguese, German,
Dutch, Turkish, French).

> @BRAND @BRAND la actualización al iOS 11.0.3 acabó con mi iPhone 6 díganme ustedes qué hacer. #fuckingupgrades

> Tirar print com o IOS 11 tá uma merda. Demora um século pra fazer 😡😡😡😡😡 Meliore Apple @BRAND

> Meu, to tendo problema com o wifi, n sei se o problema é o iOS 11 ou o iphone 8 @BRAND help me

**Good resolution:** `other_channel_redirect` — always escalate, no reply logic,
no grounding, no intent sub-classification. Apple's own corpus behavior confirms
this is a first-class route: its language-redirect template ("We offer support via
Twitter in English…") is the second-largest reply cluster in the whole dataset at
611 near-identical replies. That template is also a reliable label source, since
the brand itself marks these threads.

---

## 7. `out_of_scope`

Messages that are in English and coherent, but are not a support request the agent
can act on. Distinct from `general_complaint_nonactionable`, which is a genuine
grievance too vague to act on — these are not grievances at all. The bucket exists
because the 3B re-labeler kept filing them under `non_english`, where they were
0/14 correct (decisions.md #13); they needed a home that is not the residual.

Four recognisable kinds, all present in the sample:

- **Warranty, policy and pricing questions** — answerable, but from policy, not
  from a retrieved support thread.
- **Phishing and scam reports** — the customer is reporting, not asking.
- **Praise and compliments** with no request attached. **Empirically near-zero:**
  15 hand-checked keyword candidates yielded 0 genuine hits — every match was sarcasm
  ("Thanks for destroying my wife's iPhone"), a polite sign-off on a real complaint,
  or praise bundled with a request. Retained as a documented kind because it does
  occur (one row in the n=500 design sample), but it is not built for and the golden
  set contains none. Do not spend detector effort here without new evidence.
- **Meta-commentary** about Twitter, the brand, or the product generally.

> @BRAND has @BRAND started offering International Warranty on #iPhones? If yes, then is it applicable in India too? If yes, then can you please share official page stating the same along with T&Cs?

> @BRAND I just got this scam text message, thought you should know, as it's quite convincing: [link]

> Hey @BRAND, IOS 11 is treating my iPhone very well. [link]

> now we have more characters we can add to a tweet. Will it help? feels like 140 was enough. [...] But cool update. What's with the I️ issue tho.

**Good resolution:** varies by kind, which is why this is a routing bucket rather
than a reply-drafting one. Praise takes a polite acknowledgment and nothing else.
Phishing reports take an acknowledgment plus the report-it channel. Warranty and
policy questions take `other_channel_redirect` or `dm_handoff` — the answer is
region-specific policy the retrieval pool does not contain.

**Do not auto-draft from retrieval for this intent.** Apple's own behavior is
misleading here: 5 of the 9 sampled threads were answered `self_contained_answer`,
so a history-agreement metric will reward confident public answers to warranty and
policy questions the agent has no grounding for. Same divergence as
`ios_version_downgrade` (decisions.md #11), opposite direction — there history
under-answers the target, here it over-answers it.

**Expected to be under-counted.** The 9 rows found are only those the re-labeler
misfiled as `non_english`. Others are still sitting in
`general_complaint_nonactionable`, since nothing looks for them directly. Treat
1.8% as a floor, and sample the residual bucket for them by hand when building the
golden set.

---

## Still not in scope

Retail store and Genius Bar logistics, and hardware-damage reports (cracked
screen, liquid damage, anti-reflective coating) have no intent of their own and
still land in `general_complaint_nonactionable`. They are plausible `out_of_scope`
members but were not separated, because nothing in the current pipeline detects
them — see the under-counting note in §7. If any grows past a few percent of the
golden set it earns its own intent; none does today.
