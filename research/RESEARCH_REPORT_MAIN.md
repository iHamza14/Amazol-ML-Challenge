# Research report — Amazon ML Challenge 2026, Business Entity Resolution
*Compiled Sep 27 2026, 03:00-04:00 IST, from a 12-agent live-web sweep (121 findings kept at high/medium
impact) plus direct measurement of the training data (see `DATA_FINDINGS.md`). Everything below that
changed the pipeline is already implemented in `src/`; the "not done" items are listed at the end.*

## 1. New intelligence the previous session did not have

### 1.1 Competition-specific
| Finding | Source | Why it matters |
|---|---|---|
| **Public leaderboard (5,880 teams, scraped 2026-09-27 00:08 IST):** #1 0.990788, #10 0.989780, **#50 0.987560**, #100 0.986469, #200 0.984719, #500 0.980113. Only 9 teams ≥ 0.99. | amazonmlchallengeexplorer.vercel.app API (via r/Btechtards) | Top-50 needs ≥ 0.9876 public; every +0.001 is 15-25 ranks in the crowded 0.987-0.990 band. Ties go to the earlier upload. |
| **Validation overstates the leaderboard by ~0.015 for every team that logged both** (harshgitty58 0.9842→0.969; mayankgoplani 0.9776→0.964 and 0.9806→0.956; rishabhiitj25 OOF 0.9872→0.969; aditya-bheke 0.985→0.981). | their PROGRESS/audit/RESULTS docs | Plain in-country validation is not the metric to select on. |
| **Cause 1 — test distractor density:** test has 5.75 S2/S3 rows per S1 vs 4.67 in train with the same 3.46 true matches per S1 ⇒ ~1.9x more near-miss distractors per entity. Teams whose extra features were density-sensitive counts (candidates with equal house number, posting sizes) gained on val and lost on the board; mayankgoplani measured that 90 % (US) / 61 % (India) / 75 % (France) of their extra test pairs had a different house number than the S1. | mayankgoplani audit.md, rishabhiitj25 'clone09' | Weight distractor false positives ~1.9x when choosing thresholds; drop absolute-count features. |
| **Cause 2 — France:** the only diagnostic upload that blanked France scored identically (France pairs are "invisible" on the public subset for that team), while another team's France-blank probe showed US+India ≈ 0.985 (= validation) and France ≈ 0.948. Leave-one-country-out (train US → score India) drops 0.9739 → 0.9512 with 3x false positives and singleton accuracy 96.6 → 93.8 %: an unseen country **over-matches** (France predicts 3.47 links/S1 and only 4.2 % empties vs the generator's 5.6 %). | aditya-bheke, mayankgoplani PLAN_98 | France needs a higher threshold and a label-free way to set it. |
| The transliteration-dictionary trick is known to several teams (vc940 dataset, adbhargav, rishabhiitj25, Vishaal1409, rheasushanth): exactly 1,534 Indic tokens, identical vocabulary in train and test (0 OOV). | HF vc940, GitHub logs | Confirms our measurement (1,347 tokens ≥2 occurrences, 99.7 % val coverage). |
| "Reverse rank became the single strongest feature" (each S2/S3 record retrieves its top-K S1; recall 96.4→97.5 %, val +0.004) and address-less record assignment (+0.0017). | aditya-bheke (LB 0.981) | Implemented as the reverse channel. |
| "Five of the top six features by gain are relative" (rank within the S1's list, gap to best); relative features worth ~0.2 F0.5 over absolute-only. | TanmayJaiswal | Our group-relative features are the right family. |
| 52 % of S1 entities share an exact name with a guaranteed-wrong record; ~70 % of singletons have a same-name look-alike. | abhaykdas, adbhargav | Address evidence must dominate; a same-name-count feature added. |
| Stacking / cross-encoder gains that show on OOF did not transfer to the board (rishabh: +0.0045 OOF, −0.004 LB; SanskariXD: cross-encoder ceiling ≈ +0.0005). 92 % of one team's errors were false negatives from blocking, not the model. | rishabhiitj25, SanskariXD | Do not spend the remaining hours on a cross-encoder; spend them on blocking recall, decision layer and France. |
| Guardrail that caught a 0.42 disaster: predicted singleton rate must be ~5.5-6 % and mean links ~3.3-3.5 before uploading. | PranavKamate submission log | Implemented as inference guardrails. |
| Rules: 5 uploads/day; two leaderboards; top-100 announced after artefact check; ties by earlier submission; candidate_pairs.tsv audited for recall ceiling and reduction ratio (other teams submit 10-25 candidates per S1). | guidelines PDF, Unstop page | Keep the cap at 80 (recall@80 = recall@150 measured) and upload early. |

### 1.2 Technique-level (what the literature and other competitions say)
* **Sparse beats dense for blocking at this scale**: Sparkly (VLDB 2023) top-k char-3-gram TF-IDF beats deep blockers on 14/15 datasets and scales to 26M rows; UniBlocker/SC-Block/e5 zero-shot trail tuned sparse indexes; multilingual-e5-base top-50 name+addr recall 95 % vs char-3-gram TF-IDF top-10 98.9 %. Dense is a union channel at best. (Our sparse union: 99.7-99.9 % at 80.)
* **Decision layer**: expected-F_β optimum is always a top-k prefix or the empty set (Lewis 1995 / Ye et al. 2012 / Waegeman 2014); with the one-sided many-to-one constraint the exact record-side optimum is the argmax S1 (no Hungarian needed); calibrate boosted trees (isotonic, per country) before any expected-F rule; F-optimal thresholds for calibrated scores sit near 0.8·F* for F0.5. On our validation the tuned per-group thresholds still beat the plug-in expected-F rule by ~0.002, so thresholds remain the default and expected-F is kept as an evaluated alternative.
* **Foursquare Location Matching winners** (the closest analogue): per-id aggregate/ratio features over the candidate list, reverse-side aggregates, size-adaptive greedy post-processing (stricter threshold for the 2nd, 3rd… addition), LCS / common-substring counts, name-frequency count encoding, GroupKFold by entity with a held-out-region fold as the unseen-domain proxy.
* **Splink / dedupe / Magellan / recordlinkage comparator catalogues**: term-frequency-adjusted exact matches (sum of −log rel-freq over shared tokens), SoftTFIDF (JW-secondary, θ=0.9) as the best name comparator in Cohen et al. 2003, Damerau-Levenshtein ≤1/≤2 ladders, double-metaphone token intersection, prefix/postfix normalised similarity, near-integer / prefix-string house-number predicates, address "root" (street types, directionals, unit types removed).
* **Walmart 'ER in practice' (2026)**: LightGBM on similarity features ties DeepMatcher at 25-50x the speed; hard-field vetoes and sparsity-aware (per-bin) thresholds gave the largest purity gains — the origin of our per-bin thresholds.
* **French normalisation** (La Poste / AFNOR abbreviations, BAN normadresse regex stages, libpostal fr dictionaries, INSEE Sirene 61 street-type codes, GLEIF ISO-20275 legal forms, Etalab's Sirene search analyzer): Sirene-shaped names usually omit the legal form, elisions (`l'`, `d'`) must be split, `Saint/St/Ste`, `Bd/Boulevard`, `N°`, CEDEX postal codes differ in the last digits, arrondissement ↔ postcode. All of the abbreviation tables are in `text_tables.py`; the elision/acronym extras are listed under "not done".
* **Indic**: unidecode (GPL!) and anyascii both drop the inherent schwa ('Kolkata'→'klkata'); IndicXlit (MIT, 11M) is the neural alternative; but the GT-learned dictionary makes this moot for 96-99.7 % of tokens.

## 2. Gap analysis — previous pipeline vs. evidence vs. current pipeline

| Area | Previous session (v1) | Best evidence | Now (v2) |
|---|---|---|---|
| Scale | Python dict-of-sets indexes, one dict per pair (>100 GB at 12M rows; hours of loops) | sparse top-k products (Sparkly), rapidfuzz batch scoring | `sparse_dot_topn` multi-channel blocking, `cpdist` features, streaming per country/chunk, forked parallel features |
| Indic names (18 % of Indian S2/S3) | unidecode only (token_set_ratio ≈ 67, 88 % of pairs share no token) | GT-learned token dictionary (several teams) | learned dictionary, 99.7 % val token coverage, similarity ≈ 99 |
| Domain-collapsed names (3.6 %) | nothing | DP word-breaking against the S1 vocabulary (adbhargav, Vishaal) | Viterbi segmentation + collapsed-string features + char-3-gram channel |
| Leetspeak (2.6 %) | nothing | guarded digit→letter repair (Gitanaskhan26, DNB) | vocabulary-guarded de-leet |
| Admin units | none | state/region tables incl. Indic script, dept→region | comma-component parsing, `zzustx`-style admin tokens, relation feature |
| House numbers | first-number match/conflict only | truncation ≠ conflict (18 % of true pairs drop a number), shifted numbers = distractor fingerprint | 10-class relation code, abs-diff, digit hamming, range, digit-string containment, secondary-number agreement |
| Inserted words | nothing | decoy-word score (kshirinshetty +0.012 LB) | learned P(true \| extra token), min/mean/known, masked 15 % for France robustness |
| Legal forms | suffix stripping | legal-form family conflict as decoy signal (aditya-bheke) | family relation feature |
| Relative features | blocking rank only | "five of top six features are relative" | gaps/ranks vs best on name, address, street, joint, retrieval score |
| Reverse side | none | strongest single feature in the 0.981 pipeline | reverse channel: rank/score + top-1 S1 as extra candidate |
| Density-sensitive counts | candidate_count etc. as features | gained val / lost LB | dropped |
| Threshold selection | global sweep, pair-level singleton floor | per-country, per-bin, density-aware, conflicts, expected-F, calibrated | all evaluated on validation; density-adjusted metric selects |
| France | street-similarity filter (hurts: 0.985→0.970 on US) | unseen-country over-matches; set threshold by known singleton share | filter off by default; raise-only calibration to 5.6 % empties; France-vocabulary masking |
| Model | CatBoost only | LightGBM ≈ CatBoost; consensus of two models = cheap precision (+0.003-0.005 LB) | LightGBM + CatBoost mean, `--consensus` = min |
| Validation | 15 % random S1, all candidates | full-pool, entity-disjoint, density-corrected, error decomposition | full-pool val + adjusted metric + error analysis + saved predictions |
| Guardrails | none | singleton rate 5.5-6 %, 3.3-3.5 links | per-country guardrails in inference log |

## 3. Ranked action list
Done (in `src/`): sparse blocking · learned transliteration · segmentation · de-leet · admin tables ·
number relations · extra-token stats · legal-form relation · same-name count · group-relative features ·
reverse channel · density-adjusted per-bin thresholds · consensus option · conflict resolution ·
France calibration · guardrails · error analysis · validator run.

Done in optimization round 1 (commit ec4c0cb and the review-fix commit that follows): LOCO mode
(`ER_LOCO`), per-country isotonic calibration + expected-F0.5 as an evaluated alternative, size-adaptive
acceptance (evaluated per delta), LCS/Indel/postfix/consonant-skeleton/SoftTFIDF-coverage/acronym
comparators, short-vs-long token edit counts, out-of-vocabulary fraction, S1 same-name counts, French
elisions, Telangana↔Andhra Pradesh alias, plus 27 code-review fixes (hyphenated French regions, domain
truncation of single-token names, legal sigles as identity tokens, subsampling reweighting, KEEP_PROB
alignment, persisted blocking settings, validator memory, ...).

Measured on the 40k-entity local sample (20k training entities): baseline 0.98738 plain /
0.98542 density-adjusted; with the error-analysis features 0.98697 / 0.98469 (missed matches
−47 entities, empty-address false positives +22 — a re-balancing inside the inherently ambiguous
empty-address bin, within noise at this size). The 100k SageMaker run decides; `ER_DROP_FEATURES`
allows a clean ablation.

Not done, in expected-value order:
1. **Sibling expansion** (harshgitty58): for low-confidence records, propose the S1 of confidently matched records sharing house-number+street.
2. **Validation faithfulness for cross-split owners**: false positives on rows owned by a non-validation S1 (13-25 of ~100) would be resolved at inference by the owner's own claim; scoring them as errors biases the sweep towards conservatism. Fix: also score the owner's pair for those rows.
3. **Candidate-set reduction for the audit**: a light pruner to ≤ 25 candidates per S1 for `candidate_pairs.tsv` (recall@20 is already 99.2-99.5 %).
4. **Reverse channel on char-3-grams for empty-address records** (3.4 % of S2/S3): the remaining blocking tail is garbled names without an address; a name-only reverse retrieval restricted to those rows is cheap.
5. Dense channel (multilingual-e5-small, MIT) as a union channel — only if a per-stratum recall report shows misses concentrated in Indic/empty-address rows.

## 4. Traps other teams fell into
* Random negatives instead of blocking-generated hard negatives → local 0.988, LB 0.42.
* Exact-name fast path for generic names → LB 0.64.
* Truncating posting lists by position (first 15) instead of by document frequency → LB 0.20.
* Small dev slices (3 %) → +0.04 optimistic.
* Density-sensitive count features, stacking tuned on plain OOF → higher val, lower LB.
* Street-similarity France filter → removes reordered-address true matches (measured −0.015 on US).
* Chasing sub-0.003 public-LB moves (noise on a random subset).

## 5. Key references
Sparkly https://www.vldb.org/pvldb/vol16/p1507-paulsen.pdf · one-to-one ER algorithms https://link.springer.com/article/10.1007/s00778-023-00791-3 ·
F-measure thresholding https://arxiv.org/abs/1402.1892 · exact E[F] https://arxiv.org/abs/1206.4625 · Waegeman 2014 https://jmlr.org/papers/v15/waegeman14a.html ·
Cohen 2003 SoftTFIDF https://www.cs.cmu.edu/~wcohen/postscript/ijcai-ws-2003.pdf · Splink comparisons https://github.com/moj-analytical-services/splink ·
libpostal fr dictionaries https://github.com/openvenues/libpostal/tree/master/resources/dictionaries/fr · La Poste abbreviations https://www.laposte.fr/envoyer/abreviation-adresses-postales ·
BAN normadresse https://github.com/BaseAdresseNationale/normadresse · Foursquare 1st/4th https://www.kaggle.com/competitions/foursquare-location-matching ·
Team logs: https://github.com/aditya-bheke/business-entity-resolution · https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution ·
https://github.com/rishabhiitj25/AmazonML26 · https://github.com/harshgitty58/Amazon_ML_Challenge · https://github.com/SanskariXD/ML-2-C-26 ·
https://github.com/adbhargav/amazon-ml-challenge · https://huggingface.co/datasets/vc940/business-entity-resolution-normalized ·
leaderboard scraper https://amazonmlchallengeexplorer.vercel.app
