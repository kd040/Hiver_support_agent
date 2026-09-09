# Hiver SDE Intern Assignment — Build Spec

## Goal
Build an AI customer-support agent for **one brand** from the Kaggle
"Customer Support on Twitter" dataset (`thoughtvector/customer-support-on-twitter`).
The agent must:
1. Classify each incoming customer message into a small set of intents (defined from the data).
2. Draft a reply grounded in how the brand has historically resolved similar issues.
3. Decide auto-handle vs. escalate-to-human, with a stated reason.

**What matters most:** the proof, not the system. Golden eval set, evaluation
harness (with judge-vs-human agreement), failure analysis, and an honest
"what's misleading about my headline number" section carry as much weight as
the pipeline itself.

## Open decisions (resolve before/while building)
- [ ] **Brand**: TBD — profile brand volumes in the dataset once CSV is uploaded,
      pick one with dense, resolved threads (candidates: AmazonHelp, SpotifyCares,
      UberSupport).
- [ ] **Local model runtime**: TBD — Ollama (easiest, e.g. Llama 3.1 8B / Qwen2.5 7B),
      llama.cpp, or HF `transformers`. Pick one and use it consistently for
      classification, reply generation, and LLM-as-judge.
- [ ] **Embedding model**: local sentence-transformers model for retrieval (e.g.
      `all-MiniLM-L6-v2` or similar) — no API needed.

## Build order

### 0. Scope lock-in
- Write down explicitly what's *not* being built (e.g. no multi-language, no
  attachments/images, no real-time streaming). Goes in report's problem framing.

### 1. Data pipeline
- Reconstruct conversation threads from raw tweets using
  `in_response_to_tweet_id` / `response_tweet_id`.
- Filter to chosen brand; keep threads with ≥1 customer turn + ≥1 brand-agent turn.
- Clean: strip @handles/links, dedupe boilerplate replies, preserve timestamps for ordering.
- Subsample to a size that reproduces in <15 min (thousands of threads, not 3M rows).

### 2. Intent taxonomy
- Cluster sample of customer-first messages (embeddings + k-means, or manual
  skim of 200–300) to find natural intent buckets for this brand.
- Land on 6–10 intents (e.g. billing, outage, account access, refund,
  delivery/order status, general complaint, feature question, other).
- Write a taxonomy spec doc with 2–3 examples per intent — doubles as labeling
  guide and decision-log entry.

### 3. Golden evaluation set (150–250 examples)
- Stratified sample across intents AND across easy/ambiguous cases — don't
  cherry-pick clean examples.
- Hand-label: intent, ideal grounding source (which past thread it should draw
  from), correct auto-handle/escalate decision.
- Document sampling/labeling method in the report (required deliverable).

### 4. Retrieval-grounded reply drafting
- Embed historical resolved brand-agent replies locally.
- For incoming message: retrieve top-k similar past resolutions → feed as
  context to local LLM → draft reply.

### 5. Escalation policy
- Rule-based + confidence threshold: low classifier confidence, certain intents
  (e.g. account access/legal) always escalate, urgency/sentiment spikes escalate.
- Log a stated reason per decision (required, not optional).

### 6. Baselines
- **Trivial**: majority-class intent + canned reply, always escalate.
- **Simple**: TF-IDF/logistic-regression intent classifier + nearest-neighbor
  reply retrieval, no LLM.
- System must beat both, measurably.

### 7. Evaluation harness
- Automated: intent accuracy/F1 vs. golden set, retrieval hit-rate, escalation
  precision/recall.
- LLM-as-judge: rubric (relevance, groundedness, tone, correctness) scored by
  local model.
- Judge calibration: hand-score ~30–50 examples yourself, compute agreement
  (Cohen's kappa or % agreement) with the judge.

### 8. Report + decision log + repo
- Report sections (max 6 pages or README section):
  - Problem framing — what "good" means for this brand, what was cut
  - Results vs. both baselines
  - Failure analysis — top 5 failure modes with real examples + hypotheses
  - "What is misleading about my headline number?" (mandatory)
  - What you'd do next with one more week
- Decision log: 10–15 non-obvious decisions + why (running list from day one,
  bullets fine).
- README: reproduces headline results in <15 min on a subsample.

## Deliverables checklist
- [ ] Repo with runnable pipeline + README (<15 min reproduction)
- [ ] Golden eval set (150–250 hand-labelled examples) + sampling/labeling note
- [ ] Evaluation harness (automated metrics + LLM-judge rubric + human agreement evidence)
- [ ] Report (≤6 pages or README section, all 5 required subsections)
- [ ] Decision log (10–15 entries)
- [ ] Submit via Notion form: repo link (public or access granted) + report. No email submissions.

## Rules
- AI coding assistants allowed freely; be ready to explain/modify your own code live.
- Cite anything borrowed — borrowing is fine, not knowing what you borrowed is not.
- Full dataset won't be run — a subsample is expected and encouraged.
