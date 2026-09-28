# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Orcya  
**Team Members:** Hamza Anwar (lead), Harsh Dahiya, Afzal, Shahan Ayyubi  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve each Source-1 business against Source-2/3 records with a three-stage pipeline built entirely
from the provided data, with no external lookup of any kind:

1. **Candidate generation as a cascade.** Per country, five sparse TF-IDF channels (name words, name
   character 3-grams, address words, house-number×street combinations, joint) each retrieve 250
   neighbours per S1 entity with multi-threaded sparse matrix products; a reverse channel lets every
   S2/S3 record nominate its best S1 owners. Reciprocal-rank fusion orders the union and caps it at
   150 (US, France) or 200 (India). A **stage-2 pruner** (a small gradient-boosted model over the 16
   retrieval scores and ranks, which cost nothing extra) then scores every retrieved pair; each S1
   entity keeps at most its **20 best-scored** pairs above a per-country threshold (set to keep 99.9 %
   of the retrieved true matches on validation). The survivors — about **20 candidates per S1 entity**,
   down from 150–200 retrieved — are exactly the pairs the matching model scores and the content of
   `candidate_pairs.tsv`.
2. **A LightGBM + CatBoost pair classifier** over 124 noise-aware features, trained on 300 k S1
   entities searched against the full S2/S3 pools of their countries.
3. **An entity-level decision layer** that directly optimises the competition metric (macro F0.5 per
   S1 entity) on a held-out split: per-country and per-address-bin thresholds, a rank ladder for the
   2nd/3rd/4th+ link of an entity, a one-S1-per-S2/S3 conflict rule, an address-empty
   unique-claimant rule, expected-F0.5 set selection as an alternative, and a label-free raise-only
   threshold calibration for France, which never appears in training.

Key innovations: an Indic→Latin **transliteration dictionary learned from the ground truth**
(99.7 % token coverage), **house-number relation coding** that separates truncation noise
(607→60) from the distractor generator's shifted numbers (3338→3351), **learned inserted-word
statistics** that identify distractor vocabulary ("Holdings", "Midtown", "Overseas" never occur in
true matches; "Center", "Services", "Dr" almost always do), and a **density-adjusted validation
metric** that anticipates the test pool's 1.9x higher distractor density instead of tuning to the
training pool.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the full training data (12.5 M rows, see `research/DATA_FINDINGS.md` for all numbers):

* **Ground truth structure.** 5.6 % of S1 entities are singletons; non-singletons have 3.7 matches
  on average (up to 11). Every S2/S3 record belongs to at most one S1 entity (0 violations), and
  26 % of S2/S3 rows are unmatched distractors.
* **Name noise (S2/S3 only; S1 is clean).** 18 % of Indian S2/S3 names are Indic-script
  transliterations (Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Odia,
  Gurmukhi); 3.6 % are domain-collapsed (`cardiologysafecare.com`, `@riddhitraders`); 2.6 % contain
  leetspeak (`Cardi0logy`, `5afe`); 7.9 % carry brackets; plus accent injection, junk prefixes,
  `X d/b/a Y` / `X formerly Y`, word reorder/duplication/deletion, scrambled typos, legal-form
  variants in any position, and ~4 % of true matches with a completely random name at the true
  address.
* **Address noise.** State/region given as full name, code or Indic script (`Telangana` / `TG` /
  `తెలంగాణ`), French region vs. department (`Hauts-de-France` vs. `Nord`), abbreviations
  (`Rd`/`Road`, `R.`/`Rue`, `Saint`/`St`), truncated house numbers (4.5 % of true pairs:
  `607→60`, `2007→007→02007`), garbage number prefixes (`#867 HOUSE NO 121`, 5.8 %), a genuinely
  different first number in 4.4 %, ordinals (`NINTH`/`9th`/`71ND`), hyphen-split numbers
  (`102→1-02`), `N/A`/`<NULL>` tokens, component reordering and dropped components, 3.4 % empty.
* **The distractor generator.** Unmatched S2/S3 records are, in 34 % of cases, the S1 name with an
  inserted word (`Holdings`, `Midtown`, `Lakeside`, `Exports`…) or swapped legal form, on the
  **same street with the house number shifted by +3…+21**; another 10 % keep the house number but
  change a secondary number (floor, unit); 3 % are same-name chains at another address. 20 % of
  singletons have such a near-miss next door — roughly 1.1 % of the total score depends on rejecting
  them.
* **Blocking ceiling.** 14.6 % of true pairs share no name token and 4.5 % share no address token,
  but only 0.02 % share neither: unioning name and address channels loses almost nothing.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier + entity-level decision layer (hybrid, per country).  
**Core Innovation:** noise-model-driven normalisation (learned transliteration, word segmentation of
collapsed names, leet repair), house-number relation coding, learned inserted-word statistics, and
a decision layer that optimises expected macro-F0.5 per entity with the one-owner constraint.

---

## 3. Candidate Generation (Blocking)
Per country, S2/S3 records are vectorised into five L2-normalised TF-IDF matrices and each S1
entity retrieves its top-k neighbours with sparse matrix products (`sparse_dot_topn`, multi-threaded,
20 k S1 rows per chunk):

| Channel | Document | top-k |
|---|---|---|
| `name_tok` | core-name word unigrams (legal forms removed, transliterated, de-leeted, segmented) | 250 |
| `name_chr` | character 3-grams of the space-less core name (typos, leetspeak, collapsed domains) | 250 |
| `addr_tok` | address words + numbers (abbreviations canonicalised, admin unit as one token) | 250 |
| `addr_num` | (number, street word) combinations — the most precise address key | 250 |
| `joint` | name + address tokens in one vector | 250 |

Tokens occurring in more than 3 % of the country's S2/S3 rows are dropped; cosine < 0.08 is never a
candidate. A **reverse channel** additionally lets every S2/S3 record retrieve its top-5 S1 entities on
the joint vector: the S1's rank in that list is a "competition" feature (is this S1 the record's best
owner?) and each record's top-1 S1 is added as an extra candidate.

**Candidate generation is a two-stage cascade**, because the search space is cut in two different ways:

*Stage 1 — retrieval.* The channels are fused by **reciprocal-rank fusion** (sum over channels of
1/(60 + rank); the reverse rank r counts as 2r) and capped per S1 entity at **150 (US, France) / 200
(India)**. Retrieval depth is nearly free (the sparse product, not the top-n selection, dominates the
cost), so every channel retrieves deep and the fusion decides what enters the cap. Measured on a
1.24 M-record India pool (30 % of full scale, 6 000 queries, 22 090 true matches): the old min-rank
fusion at depth 60/40/60/40/80 saturated at ~198 candidates and 0.9928 recall even uncapped
(0.9890 at cap 100); depth 250 with RRF reaches 0.9928 at cap 150 and 0.9939 at cap 200. The residual
tail is empty-address records with destroyed names and same-name chains whose address is truncated
to a city — unreachable by any retrieval.

*Stage 2 — candidate pruner.* Every retrieved pair carries its score and rank in every channel, the
fused rank and the reverse rank/score — 16 numbers that cost nothing extra. A small LightGBM (63 leaves,
300 rounds) trained on these numbers alone scores each retrieved pair, and a per-country threshold —
the largest value that keeps **99.9 % of the retrieved true matches on the validation split** — drops
the hopeless pairs before any string comparison is made. The survivors are the candidate set: the exact
pairs the matching model computes its 124 features for and scores, and the content of
`output/candidate_pairs.tsv`. On the validation split the threshold alone kept 47.1 candidates per S1
entity in the US (of 150 retrieved) and 129.9 in India (of 200), at 99.90 % of the retrieved true
matches. A **per-entity cap** then keeps only the 20 highest pruner scores of each S1. Its cost was
measured end to end on the 75 000 validation entities by removing the selected pairs that fall outside
the cap (`prune_cap_eval.py`):

| cap per S1 | candidates per S1 (US / India) | adjusted macro-F0.5 | change |
|---|---|---|---|
| 10 | 10.0 / 10.0 | 0.98053 | −0.00141 |
| 20 (**shipped**) | 19.9 / 20.0 | 0.98133 | −0.00061 |
| 40 | 36.3 / 40.0 | 0.98170 | −0.00024 |
| 80 | 46.3 / 80.0 | 0.98188 | −0.00006 |
| none (threshold only) | 47.1 / 129.9 | 0.98194 | 0 |

The cap of 20 was chosen as the operating point that keeps the candidate set small (the organizers rank
smaller candidate sets higher) and the full test set within one machine's time budget, at a cost of
0.0006 F0.5. On the test set: 1 732 544 S1 entities × at most 20 = about 34.6 M scored pairs, from
about 280 M retrieved.

- **Candidates per S1 entity (test):** at most 20; about 20 on average (validation: US 19.9, India 20.0;
  France uses the lowest seen threshold and the same cap).
- **Recall on a held-out training split (60 000 entities, final run):** retrieval within the cap
  0.99240 (US, cap 150) and 0.98709 (India, cap 200); the pruner threshold keeps 0.99899 (US) and 0.99900
  (India) of the retrieved true matches.
- **How true matches are not lost:** name and address channels are unioned (only 0.02 % of true
  pairs share neither a name nor an address token), the Indic transliteration and domain-name
  segmentation make otherwise token-less names retrievable, the character-3-gram channel covers
  typos/leetspeak, RRF lifts candidates found by several channels above single-channel same-name
  floods, and the pruner threshold is set from measured recall, not by hand.

---

## 4. Matching Model

**Features used (124):**
- Name: rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on core, strict (filler
  words removed), with-suffix, collapsed and alt (d/b/a) variants; word- and char-3-gram TF-IDF
  cosines; token Jaccard / containment / shared count; max and sum IDF of shared tokens; first/last
  token equality; lengths; learned inserted-word statistics (min / mean P(true | extra token),
  fraction known, counts of extra / missing tokens); script, domain, dba flags.
- Address: ratio / token-set / token-sort / partial on the full address and on the street string
  (numbers and admin removed); TF-IDF cosine; word Jaccard / containment / shared IDF; admin
  (state/region) relation; component and length counts; postal-code relation; empty flags.
- Numbers: first-number relation code (equal / prefix-truncation / suffix-truncation / appears
  elsewhere / same-length one-digit substitution / multi-digit substitution / unrelated / missing),
  absolute difference, digit Hamming distance, shared-number count and Jaccard, S1 numbers ⊆ S2/S3
  numbers, extra / missing numbers, range containment (`628-632`), digit-string containment
  (`1-02` vs `102`).
- Literature comparators: longest-common-subsequence, Indel and postfix similarities, consonant-skeleton
  similarity (vowel/schwa noise), SoftTFIDF-style token coverage (Jaro-Winkler ≥ 0.9 per token),
  acronym match, and edit counts split by token length (a one-letter edit in a ≤4-letter acronym is
  the distractor generator's fingerprint — `YE Agro`→`YM Agro` — while edits in long words are
  ordinary typos).
- Error-analysis features: out-of-vocabulary fraction of each name against the S1 vocabulary (random
  generated names such as `Iriecto` are 100 % OOV and, unlike distractors, sit at the true address),
  number of S1 entities carrying exactly the candidate's core name (chains vs unique names),
  Telangana↔Andhra Pradesh treated as the same admin unit.
- Retrieval: per-channel score and rank, number of channels, fused rank, reverse-channel rank/score.
- Group-relative (within the S1 entity's candidate set): gap to the best candidate on name,
  address, street, joint cosine and retrieval score; rank by combined similarity. Absolute pool-density
  counts (candidate count, number of equal-house-number candidates) were deliberately removed because
  the test pool is denser than the training pool.

**Model type:** LightGBM (255 leaves, learning rate 0.05, up to 4 000 rounds with early stopping on the
validation split) + CatBoost (depth 8, learning rate 0.06, GPU), probabilities averaged 0.5 / 0.5.
Training rows: every positive plus the 25 highest-ranked negatives per entity plus a 35 % random sample
of the remaining negatives (re-weighted by 1/0.35 so probabilities stay calibrated to the full
candidate pool), from 300 000 S1 entities; a further 60 000 entities form a statistics split (inserted-word
statistics, blocking recall) and 75 000 a validation split. Inserted-word features are masked on 25 % of
the training rows that have inserted words, so the model also handles the unseen French insertion
vocabulary. Every split is searched against the FULL S2/S3 pool of its country, so validation sees the
same collision density as the test set within a country.

**Decision layer.** The pair probabilities are turned into per-entity match sets by rules selected on
the validation split with a **density-adjusted** macro-F0.5: false positives on unmatched (distractor)
S2/S3 rows are weighted 1.9x because the test pool contains 5.75 S2/S3 rows per S1 entity versus 4.67
in training with the same 3.46 true matches per entity, i.e. ~1.9x more near-miss distractors per
entity; plain validation therefore overstates test precision. Rules, each adopted only when it improves
that score:
- thresholds per country and per bin (candidate address present / empty), by coordinate ascent over a
  0.30–0.95 grid;
- **rank ladder**: separate acceptance deltas for the 2nd, 3rd and 4th-plus link of an entity (searched
  over all non-decreasing triples of {−0.30 … +0.10}) and an optional cap on links per entity. Under
  F0.5 the break-even probability of the (k+1)-th link rises with k (0.727, 0.759, 0.771, 0.8 …), so
  one delta cannot fit every rank; an entity whose best candidate passed its threshold is a confirmed
  non-singleton, which is why relaxing later ranks can pay;
- **one-owner conflict rule**: each S2/S3 record is kept for its highest-probability S1 only (the ground
  truth has zero records with two owners);
- **address-empty unique-claimant rule**: an address-less record is a noisy copy of *some* S1 almost
  always (unmatched distractors keep their addresses), so the question is ownership: such a record is
  given to its top claimant when no other S1 claims it (or the lead is ≥ 0.30), the claim exceeds a
  per-country threshold and the claimant has fewer than 11 links. Adopted only if the adjusted score
  rises by ≥ 0.0002, with the precision of the rescued set logged;
- expected-F0.5 set selection (E[F] = 1.25·TP/(1.25·TP+0.25·FN+FP) with plug-in expectations,
  empty-set expectation ∏(1−p)), raw and after two-fold isotonic calibration, and a consensus rule
  (minimum of the two models), as alternatives to the threshold family.

The selected configuration is stored in `models/model_config.json` and applied unchanged at inference.
**France (unseen in training)** starts at the mean seen-country threshold plus a precision-first shift
(0.02) and is then raised, never lowered, in steps of 0.005 until its predicted-empty rate reaches the
seen countries' validation rate (floored by the generator's 5.59 % singleton share, identical in US and
India), stopping early if links per entity would fall more than 0.2 below the seen level or after a
raise of 0.25 — a label-free calibration on the rule that is actually applied. Unseen countries use
the strictest seen threshold for the address-empty rule.

Only pairs whose maximum model probability is ≥ 0.02 reach the decision layer, in training and at
inference alike.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), final run (300 000 training entities, 75 000 validation entities searched
  against the full S2/S3 pools):** **0.98271 plain / 0.98194 density-adjusted** (US 0.98461 / 0.98393,
  India 0.97988 / 0.97896) with the threshold-only candidate set; with the shipped cap of 20 candidates
  per entity 0.98210 plain / 0.98133 density-adjusted. The density-adjusted number weights false
  positives on distractor rows 1.9x to mirror the test pool and is our estimate of the leaderboard
  score for US and India; France has no labels.
- **Models (validation pairs, 12.7 M):** LightGBM log-loss 0.00199, AUC 0.999954 (best iteration 1 257);
  CatBoost log-loss 0.00216, AUC 0.999945 (3 998 iterations, GPU); ensemble log-loss 0.00202, AUC
  0.999952. Training matrix: 22.7 M pairs, 1.03 M positives. Most important features by gain: summed
  retrieval score, fused rank, reverse-channel rank, first-house-number relation, learned inserted-word
  statistic, normalised-name ratio, admin-unit relation, address token-set ratio.
- **Selected decision layer:** mean of the two models; thresholds US 0.71 (address present) / 0.74
  (address empty), India 0.75 / 0.69; rank ladder deltas −0.05 / +0.05 / +0.05 for the 2nd / 3rd / 4th+
  link (0.98183 → 0.98194 adjusted); one-owner conflict rule on. Not adopted because they scored lower:
  expected-F0.5 set selection (0.98108), the minimum-of-models consensus (0.98145), and the address-empty
  unique-claimant rule (no threshold improved the score). France starts at 0.75 and is calibrated
  upward on the test set as described in section 4.
- **Blocking recall at the shipped cap, final run (60 000 held-out entities):** US 0.99240 (cap 150),
  India 0.98709 (cap 200). With the previous settings (top-k 60/40/60/40/80, min-rank fusion, cap 100)
  the same measurement gave 0.99185 and 0.97593: the retrieval upgrade recovered 1.1 % of India's true
  matches, which no downstream stage could otherwise reach.
- **Candidate set:** about 20 candidates per S1 entity after the two-stage cascade (150–200 retrieved,
  then pruned and capped), see section 3.
- **Public leaderboard:** as shown on the challenge portal for the submitted `matching_results.tsv`.
- **Reference run with 20 000 training entities (same full-scale pools, previous retrieval settings):**
  0.97560 plain / 0.97391 density-adjusted (US 0.97877 / 0.97742, India 0.97067 / 0.96843); ensemble
  AUC 0.99989. Error decomposition on its 5 000 validation entities (122 entity-equivalents lost): 66 % of
  the loss is missed matches (720 entities with false negatives only), 16 % false links (83 entities),
  15 % singletons wrongly given a link (18 entities), 3 % both. False-positive pairs: 89 on unmatched
  distractor rows, 36 on records owned by another S1. Small-sample runs (40 000-entity subsets of the
  data) scored about 0.987 and are reported here only as a warning: retrieval collisions grow with the
  pool, so only full-pool validation is trustworthy.
- **Common false positives (wrong merges):** same-street distractors with a one-letter acronym edit
  (`YE Agro`→`YM Agro`) or a shifted secondary number and an otherwise identical address; exact-name
  records with an empty address that belong to a same-name chain elsewhere.
- **Common false negatives (missed matches):** 56 % of missed pairs are empty-address candidates whose
  name was altered (typo, appended legal form, inserted word) — inherently ambiguous against
  same-name chains; random generated names (`Iriquo`, `Belonovivio`) whose address was also degraded;
  true pairs whose secondary number (floor, unit) was changed.

---

## 6. Conclusion
A noise-model-driven pipeline — every normalisation and feature traces back to a measured noise
pattern — with a two-stage candidate cascade and a metric-aware decision layer resolves the three
countries, including unseen France, from the provided data alone. The largest wins came from
understanding the generator (learned transliteration, house-number relation coding, inserted-word
statistics), from measuring retrieval at full pool scale rather than on samples (deep top-k with
reciprocal-rank fusion recovered 1.1 % of India's matches), and from tuning the decision layer to the
test pool's density rather than the training pool's.

**Scalability.** Every stage is per country and streams S1 entities in chunks of 10 000: retrieval is
five sparse products per chunk plus one reverse product per country, the pruner runs on numbers the
retrieval already produced, and the 124-feature computation touches only the survivors, in forked
workers that receive just their slice of the data. The full test set (1.73 M S1 × 10 M S2/S3) runs on
one 16-vCPU machine in a few hours with about 30 GB of memory; each component is embarrassingly
parallel across countries and chunks.

**Reproducibility.** `train.py` writes every learned artefact (transliteration table, segmentation
vocabulary, inserted-word statistics, models, pruner, decision configuration with all thresholds) to
`models/`; `inference.py` reads only those. Checkpointing of the training matrices allows the model and
decision stages to be re-run alone (`ER_RESUME=1`, `ER_REUSE_MODELS=1`), and `redecide.py` re-runs
the decision layer on the saved scored pairs in minutes. All settings live in `config.py` with
environment-variable overrides; the whole run is two commands.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `config.py`, `text_tables.py`, `translit.py`,
`preprocess.py`, `blocking.py`, `features.py`, `evaluate.py`, `decision.py`, `france_filter.py`,
`train.py` (entry point 1: trains the models, the stage-2 pruner and selects the decision config;
checkpoints the training matrices so the model stage can be re-run with `ER_RESUME=1`, or only the
pruner and decision layer with `ER_RESUME=1 ER_REUSE_MODELS=1`), `inference.py` (entry point 2: writes
`output/matching_results.tsv`, `output/candidate_pairs.tsv` and the scored pairs, then runs the
validator), `redecide.py` (re-runs only the decision layer on the saved scored pairs, for threshold or
France variants in minutes), `make_sample.py` (stratified sample dataset for smoke tests).
`package_submission.sh` at the repository root assembles and validates the submission zip. `README.md` gives the exact commands;
`requirements.txt` pins the environment; `setup_runpod.sh` (repository root) is the one-command setup
used for the submitted run (1x A40, 9 vCPU, 50 GB RAM).

### B. Additional Results
[feature importance table, threshold sweep, per-country diagnostics]
