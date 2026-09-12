# AppleSupport AI Customer-Support Agent

An intent-classification, retrieval-grounded reply-drafting and escalation pipeline
built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, scoped to `@AppleSupport` (48,968 clean two-turn threads).

Everything runs locally. No API keys, no paid services — two Ollama models do the
classification and generation.

- **[report.md](report.md)** — findings, headline numbers, and what is misleading about them
- **[decisions.md](decisions.md)** — 34 numbered decisions with the evidence behind each
- **[docs/taxonomy.md](docs/taxonomy.md)** — the seven locked intents
- **[data/processed/golden_set.csv](data/processed/golden_set.csv)** — the 200-row hand-labelled eval set

---

## 1. Setup

Python 3.10+ and [Ollama](https://ollama.com) are required.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

ollama serve &                    # if not already running
ollama pull qwen2.5:3b            # ~1.9 GB - classification
ollama pull llama3.1:8b           # ~4.9 GB - reply drafting
```

The two models do different jobs and are not interchangeable: `qwen2.5:3b` classifies
(fast, 0.3 s/row), `llama3.1:8b` drafts (generation quality, ~8 s/reply). See
decisions.md #12 and #26.

**Model download is ~6.8 GB and is not counted in the runtimes below** — it depends
entirely on your connection.

## 2. The dataset

The corpus is **not in this repo** (517 MB). Download `twcs.csv` from
[kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place it at:

```
data/raw/twcs.csv
```

`data/raw/` is gitignored, as are the regenerable `.pkl` / `.jsonl` intermediates under
`data/processed/`.

## 3. Reproducing the headline results

Run from the repo root with the venv active. `PYTHONPATH=scripts` is required because
the scripts import each other.

### Fastest check — no dataset needed (~5 seconds)

```bash
PYTHONPATH=scripts python -m pytest tests/ -q      # 59 tests
python scripts/eval_report.py                      # metric self-check
```

The second prints the per-intent reporting harness against a synthetic case, and
demonstrates the core measurement argument: a failure confined to one over-sampled rare
intent reads as 83.3% naive but 98.6% prevalence-reweighted (decisions.md #14).

### Subsample pipeline (~12 minutes, dataset required)

```bash
# 1. clean + anonymize the corpus                                    ~12 s
PYTHONPATH=scripts python scripts/clean_threads.py

# 2. near-duplicate clustering + the 500-row taxonomy sample          ~3-6 min (cold)
PYTHONPATH=scripts python scripts/taxonomy_sample.py

# 3. bootstrap regex intent labels                                    ~5 s
PYTHONPATH=scripts python scripts/label_intents.py

# 4. LLM re-label of the residual bucket - SUBSAMPLE of 50 rows       ~20 s
#    (omit the 50 to run all 304, ~90 s)
PYTHONPATH=scripts python scripts/llm_relabel.py 50

# 5. the golden set is hand-labelled and ships as CSV - load it       instant
python -c "import pandas as pd; pd.read_csv('data/processed/golden_set.csv', \
dtype={'tweet_id':str}).fillna({'sub_stratum':''}).to_pickle('data/processed/golden_set.pkl')"

# 6. gated retrieval over per-intent pools                            ~4 min
PYTHONPATH=scripts python scripts/retrieval_embed.py

# 7. reply drafting - SUBSAMPLE of 18 rows spanning all routes        ~2.5 min
#    (omit --sample to run all 200, ~19 min)
PYTHONPATH=scripts python scripts/draft_replies.py --sample 18

# 8. escalation decisions                                             ~5 s
PYTHONPATH=scripts python scripts/escalate.py

# 9. end-to-end rollup                                                ~1 min
PYTHONPATH=scripts python scripts/eval_harness.py
```

Step 9 prints all three evaluation components: per-intent classification accuracy,
escalation-decision agreement, and human-scored reply quality — each with its caveat
printed inline rather than left to the reader.

### What the subsample changes

Steps 4 and 7 run on a subset, so their counts will be smaller than the report's.
**Everything else is full-corpus and should match exactly** — the retrieval metric, the
escalation split, and the evaluation rollup are all deterministic given the shipped
golden set.

To reproduce the report's numbers exactly, drop the `50` in step 4 and `--sample 18` in
step 7, and add the optional steps below.

### Optional / slow steps (not needed for the headline numbers)

| script | what it does | runtime |
|---|---|---|
| `relabel_pool.py` | labels the 1,800-row random pool used for prevalence | ~8.5 min |
| `derive_prevalence.py` | corpus prevalence from the hand-check confusion matrix | ~5 s |
| `baselines.py` | trivial + TF-IDF baselines to beat | ~1 min |
| `second_pass.py` | boundary reliability check (decisions.md #18) | ~1 min |
| `judge_replies.py` | LLM-as-judge scoring — **failed validation, see #32** | ~26 min |
| `make_scoring_sheet.py` | builds the human scoring sheet | ~5 s |
| `judge_agreement.py` | judge-vs-human agreement | ~2 s |
| `retrieval_metric.py` | the TF-IDF retrieval arm, for the before/after comparison | ~6 min |

## 4. Runtimes, measured

Timed on an M-series Mac, CPU only, models already pulled.

| step | measured |
|---|---|
| full test suite (59 tests) | 0.2 s |
| `clean_threads.py` | 12 s |
| `taxonomy_sample.py` (cold) | ~3-6 min *(estimate — encoding 35,181 replies measured at ~20 s at 1,742 texts/s; the community-detection pass dominates and was not separately timed)* |
| `llm_relabel.py 50` | ~20 s (0.3 s/row measured over 299 rows) |
| `retrieval_embed.py` | ~4 min (encode ~20 s + 645 neighbour labels) |
| `draft_replies.py --sample 18` | 2.5 min (measured: 153 s) |
| `draft_replies.py` (all 200) | 19 min (measured: 1,140 s) |
| `escalate.py` | ~5 s |
| `eval_harness.py` | ~1 min (200 classification calls) |
| **subsample path, steps 1-9** | **~12 min** |

Every figure above except `taxonomy_sample.py` was measured during the build; that one
row is an extrapolation and is marked as such.

## 5. What cannot be regenerated

**The golden set is hand-labelled and is not reproducible by running code.** 370
candidates were hand-checked to confirm 244, and `build_golden.py` depends on
`golden_handcheck.pkl`, which records those manual accept/reject decisions. It ships as
`data/processed/golden_set.csv` instead. `build_golden.py` and `sample_golden.py` are in
the repo so the sampling design is auditable (decisions.md #14), not so it can be re-run
from scratch.

The same applies to `data/processed/human_scoring_sheet.csv` — 36 drafts scored by hand
against `docs/reply_quality_rubric.md`, which is the only reply-quality measurement in
the project backed by a human (decisions.md #32).

## 6. Layout

```
scripts/       pipeline, one concern per file
tests/         59 tests, pinning regressions found during the build
docs/          taxonomy.md (the 7 intents), reply_quality_rubric.md
data/raw/      twcs.csv goes here (gitignored)
data/processed/ intermediates (gitignored) + the committed CSV deliverables
report.md      findings and their caveats
decisions.md   34 numbered decisions, newest last
```

## 7. Honest notes

Three things a reader should know before quoting any number from this repo:

- **`auto_handle` is 70.5%, and it does not mean "resolved."** 45 of those 141 rows ask
  a diagnostic question that resolves nothing by design. At most 48% of rows receive
  anything resembling an answer (decisions.md #31).
- **The 92% blended classification accuracy is circular.** Golden rows for four intents
  were drawn from the classifier's own predictions, so it scores 100% on them by
  construction. The defensible figure is 70.9% on independently-sourced rows
  (decisions.md #34).
- **The LLM-as-judge failed validation** — weighted kappa ≤ 0 against human scores on
  three of four dimensions. Its scores are retained in the harness output under an
  explicit unreliability banner and appear in no headline number (decisions.md #32, #33).
