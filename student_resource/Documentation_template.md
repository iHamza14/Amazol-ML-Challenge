# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We resolve each Source-1 business against Source-2/3 records with a two-stage pipeline built entirely
from the provided data: (1) multi-channel sparse TF-IDF top-k **blocking** per country (name words,
name character 3-grams, address words, house-number×street combinations, joint), and (2) a
**LightGBM + CatBoost pair classifier** over 109 noise-aware features, followed by an
**entity-level decision layer** that directly optimises the competition metric (macro F0.5 per S1
entity) with per-country thresholds / expected-F0.5 set selection and a one-S1-per-S2/S3 conflict
rule. Key innovations: an Indic→Latin **transliteration dictionary learned from the ground truth**
(99.7 % token coverage), **house-number relation coding** that separates truncation noise
(607→60) from the distractor generator's shifted numbers (3338→3351), and **learned
inserted-word statistics** that identify distractor vocabulary ("Holdings", "Midtown", "Overseas"
never occur in true matches; "Center", "Services", "Dr" almost always do).

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
| `name_tok` | core-name word unigrams (legal forms removed, transliterated, de-leeted, segmented) | 60 |
| `name_chr` | character 3-grams of the space-less core name (typos, leetspeak, collapsed domains) | 40 |
| `addr_tok` | address words + numbers (abbreviations canonicalised, admin unit as one token) | 60 |
| `addr_num` | (number, street word) combinations — the most precise address key | 40 |
| `joint` | name + address tokens in one vector | 80 |

Tokens occurring in more than 3 % of the country's S2/S3 rows are dropped; cosine < 0.08 is never a
candidate. A **reverse channel** additionally lets every S2/S3 record retrieve its top-5 S1 entities on
the joint vector: the S1's rank in that list is a "competition" feature (is this S1 the record's best
owner?) and each record's top-1 S1 is added as an extra candidate. The union is capped at 80
candidates per S1 entity by (best channel rank, summed score); recall@80 equals recall@150 on the
held-out split. Every candidate keeps its score and rank in every channel as model features.

- **Candidate pairs generated:** [total, from inference log]
- **Recall on a held-out training split:** [from train log, per country]
- **How true matches are not lost:** name and address channels are unioned (only 0.02 % of true
  pairs share neither a name nor an address token), the Indic transliteration and domain-name
  segmentation make otherwise token-less names retrievable, and the character-3-gram channel
  covers typos/leetspeak.

---

## 4. Matching Model

**Features used (109):**
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

**Model type:** LightGBM (255 leaves, lr 0.05, early stopping) + CatBoost (depth 8, GPU), averaged.
Training rows: all positives plus the top-20 ranked negatives per entity plus a 20 % sample of
the rest, from 400 k S1 entities; inserted-word features are masked on 15 % of rows so the model
also handles the unseen French insertion vocabulary.  
**Threshold selection method:** entity-level macro-F0.5 sweep on a held-out split of S1 entities
searched against the FULL S2/S3 pool of its country (all candidates kept, no negative sampling), with
thresholds per country and per bin (candidate address present / empty), compared against expected-F0.5
set selection (E[F] = 1.25·TP/(1.25·TP+0.25·FN+FP) with plug-in expectations, empty-set expectation
∏(1−p)) and against a consensus rule (minimum of the two models), each with and without the one-owner
conflict rule. The selection metric is **density-adjusted**: false positives on unmatched (distractor)
S2/S3 rows are weighted 1.9x because the test pool contains 5.75 S2/S3 rows per S1 entity versus 4.67
in training with the same 3.46 true matches per entity, i.e. ~1.9x more near-miss distractors per
entity; plain validation therefore overstates test precision. The selected configuration is stored in
`models/model_config.json`. France (unseen in training) starts at the mean seen-country threshold plus a
precision-first shift and is then raised, never lowered, until its predicted-empty rate reaches the
generator's singleton share (5.59%, identical in US and India) — a label-free calibration.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [validation, from train log] (per country: [..])
- **Public leaderboard:** [..]
- **Common false positives (wrong merges):** same-street distractors whose house number differs
  only by a digit substitution pattern that also occurs in true matches; same-name records at a
  different address when the S2/S3 address is empty.
- **Common false negatives (missed matches):** true matches whose house number was replaced by an
  unrelated number (1.3 % of true pairs) together with a heavily altered name; domain-collapsed
  names with an empty address.

---

## 6. Conclusion
A noise-model-driven pipeline — every normalisation and feature traces back to a measured noise
pattern — with sparse multi-channel blocking and a metric-aware decision layer resolves the three
countries, including unseen France, from the provided data alone. The largest wins came from
understanding the generator: learned transliteration, house-number relation coding and
inserted-word statistics.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/`: `config.py`, `text_tables.py`, `translit.py`,
`preprocess.py`, `blocking.py`, `features.py`, `evaluate.py`, `decision.py`, `france_filter.py`,
`train.py` (entry point 1: trains models and selects the decision config), `inference.py` (entry
point 2: writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the
validator). `README.md` gives the exact commands; `requirements.txt` pins the environment.

### B. Additional Results
[feature importance table, threshold sweep, per-country diagnostics]
