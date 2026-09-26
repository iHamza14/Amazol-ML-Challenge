# Data findings — Amazon ML Challenge 2026 (measured on the provided training data)

All numbers below were measured directly on `train_source1/2/3.tsv` and `train_ground_truth.tsv`
(12.5M rows) and on the test sources, in the Sep 27 2026 session. They drive every design
decision in `src/`.

## 1. Ground-truth structure

| Fact | Value | Consequence |
|---|---|---|
| S1 entities | 2,206,821 (US 60% / India 40%) | |
| Singletons (no matches) | 123,247 = 5.59% (identical in US and India) | 5.6% of the macro score is decided by predicting *empty* |
| Matches per non-singleton | mean 3.67, max 11 (S2 up to 5, S3 up to 6) | match sets are large; recall matters too |
| S2/S3 ids matched to >1 S1 | **0** | each S2/S3 record belongs to at most one S1 -> conflict resolution is safe |
| Unmatched S2/S3 rows (distractors) | 2,681,854 = 26.0% | precision risk lives here |
| Exact duplicate (name,address) rows inside S2 / S3 | 25,891 / 18,881 | |
| Empty business_name | 0 everywhere | |
| Empty business_address | S1 0%, S2 3.4%, S3 3.3% (test: same, France 3.0%) | model must match on name alone for ~3.4% |

## 2. Name noise (S2/S3 side; S1 is clean ASCII)

| Pattern | Share of S2/S3 | Example | Handling |
|---|---|---|---|
| Indic-script transliteration | 18.2% of India rows (Devanagari 55%, Telugu, Kannada, Tamil, Gujarati, Bengali, Malayalam, Odia, Gurmukhi) | `फॉर्च्यून फाइनेंस प्राइवेट लिमिटेड` = Fortune Finance Private Limited | **Learned token dictionary** (only 1,347 distinct Indic tokens in train!). Val token coverage 99.66%, test 96.1%. token_set_ratio to S1 name: mean 99 (vs 67 with unidecode) |
| Accent injection | 6.7% US, 4.9% India, 24% France | `FÁRMS`, `Çentre`, `Réncontre` | unidecode |
| Domain-collapsed name | 3.6% (US 4.0%, India 3.0%, France 3.6%) | `cardiologysafecare.com`, `sreetradecom`, `mieuxgroupementsarl.com`, `@riddhitraders`, `#whiteurban` | strip TLD/handle, Viterbi word segmentation with S1 vocabulary, collapsed-string features, char-3gram channel |
| Leetspeak | 2.6% | `Cardi0logy`, `5afe`, `A1len`, `1imited`, `Pi1ates` | digit->letter inside alphabetic tokens, vocabulary-guarded |
| Junk prefix | 2.2% | `*** `, `... `, `-- `, `#`, `@` | stripped |
| Brackets | 7.9% | `[INC]`, `(PC)`, `((LIMITED))`, `(India)` | brackets removed, content kept |
| `X d/b/a Y`, `X formerly Y`, `DBA:` | 0.4% | `Tavotavogild d/b/a Liz & Associates` | Y = primary name, X = alt name feature |
| Trailing `#digits` | 0.35% | `springdalecity.com #39256` | stripped |
| Word reorder / duplication / deletion / typo scramble | frequent | `Limited Perfect Cotnsuhftanst`, `Family Family Bright`, `Sepciialsist` | token_set/sort, char-grams |
| Replacement char U+FFFD | present | `C�re` | removed before unidecode |
| Completely random name at the true address | ~4% of true pairs | `Iriecto` @ 617 Firehouse Rd | address features decide |
| Legal-form variants | ubiquitous | `Pvt Ltd`/`Private Limited`/`प्रा. लि.`, `E.U.R.L.`/`EURL`, `S.A.S.U.`, `[Limited]`, moved to front (`SCI Fer & Fils`) | dotted collapse + canonicalisation + removal anywhere |

### 2.1 Inserted words are the distractor fingerprint
Tokens present in the S2/S3 name but not in the S1 name (1.79M true pairs vs 0.89M one-extra-token
distractor pairs):

| Token | in true | in distractor | P(true) |
|---|---|---|---|
| center | 62,370 | 5,419 | 0.92 |
| services / service | 44,517 / 21,641 | 2,394 / 1,368 | 0.95 / 0.94 |
| sri, shri, smt, dr, mr | ~10k each | ~1.2k each | 0.90 |
| partners | 16,655 | 32,692 | 0.34 |
| llp | 4,020 | 10,211 | 0.28 |
| enterprises | 2,135 | 7,889 | 0.21 |
| holdings, group, public, infratech, exports, overseas, ventures, industries | 0-179 | 8k-44k | **0.00-0.02** |
| greater, southside, downtown, westgate, riverside, eastgate, uptown, northside, lakeside, midtown, central, east, south, north, west, valley, coastal, summit, highland, metro, clinic, global, blue, mumbai, family | 0 | 1.3k-8k | **0.00** |

-> `ExtraTokenStats` feature (smoothed P(true | extra token), min/mean over extras). France uses a
different (French) insertion vocabulary that is unseen, so the feature is neutral there; 15% of
training rows have it masked so the model also learns the "unknown vocabulary" regime.

## 3. Address noise

| Pattern | Share | Example | Handling |
|---|---|---|---|
| State abbreviation vs full name vs Indic script | very common | `TX`/`Texas`, `TG`/`Telangana`/`తెలంగాణ`, `Keralam`, `WB`/`পশ্চিমবঙ্গ` | comma-component parsing + state tables -> `us_tx`, `in_tg` |
| France region vs department | S1 always region; S2/S3 region OR department | `Hauts-de-France` vs `Nord`/`Pas-de-Calais`; `Nouvelle-Aquitaine` vs `Gironde`; `Pays de la Loire` vs `Loire-Atlantique` | department -> region table -> `fr_hauts_de_france` |
| House number truncation | 4.5% of true pairs | `607`->`60`, `2007`->`007`->`02007`, `5506`->`550` | prefix/suffix relation codes 3/4 |
| Garbage number prefix | 5.8% of true pairs have S1's first number elsewhere | `#867 HOUSE NO 121`, `##1174`, `#B3/560 515` | relation codes 5/6, `#` stripped |
| Totally different first number | 4.4% of true pairs | `6503`->`3503`, `1401`->`742`, `1305-A`->`4305-A` | digit hamming / abs diff features; accepted loss for precision |
| Leading zeros | 5.2% S2/S3 (1.1% S1) | `001446`, `045`, `00354` | lstrip |
| Hyphen-split numbers | | `102` -> `1-02`; ranges `628-632`, `55BIS`, `8-10` | joined variant + range containment |
| Ordinals | | `NINTH ST` vs `9th Street`, `71ND AVE`, `1er` | ordinal words/suffixes -> digits |
| Junk tokens | 3.4% | `N/A`, `<NULL>` | removed |
| Abbreviations | ubiquitous | `RD`/`Road`, `ST`/`Street`/`Saint`, `R.`/`Rue`, `AV`, `ALL`, `IMP.`, `N°`, `PMB`, `Unit`, `Fl No`, `H.No` | canonical short forms |
| Component reordering / dropped components | common | `TX, TERRELL, VIRGINIA ST`; `Flat 203, Secunderabad, Hyderabad, TG` | token-set metrics, IDF-weighted overlap |
| City vs county / neighbouring city | common | `Floyd County` vs `Willis`; `Brookhaven` vs `Centereach` | do not rely on city equality |
| Postal code | France 0.4%, rare elsewhere | `59000` | postal relation feature |

## 4. Distractor generator (unmatched S2/S3 rows) vs nearest S1

| Category (name sim, street sim, number relation) | Share |
|---|---|
| name HI, street HI, **disjoint numbers** | **33.6%** |
| name HI, street HI, some shared numbers (secondary number changed, e.g. floor/unit) | 9.8% |
| name LO/MID, street HI, disjoint numbers (incl. Indic-script distractors) | 14.0% |
| name HI, street HI, **first number equal** (secondary number differs: `Door 6 3Rd Floor` vs `Door 6 4Rd Floor`, `2F-CS-38` vs `2F-CS-40`) | 4.2% |
| name HI, street LO (same-name chain at another address) | 3.1% |
| random unrelated businesses | remainder |

Distractor house numbers are S1's number shifted by a small positive delta (+3 ... +21, same digit
count: 3338->3351, 18910->18931, 42->63, 112->133). True-pair perturbations are digit
substitutions (6503->3503) or truncations. Distractor names carry an inserted word from the
vocabulary in 2.1 or a swapped legal form (Inc->Ltd, EURL->SCI/SASU, Private->Public).

**True pairs** for comparison: 54% (HI, HI, equal), 7.8% some shared, 6.9% name MID, 5.8% one side
without numbers, 4.2% S2/S3 address empty, 4.1% name LO + address exact, 1.8% truncation,
**1.3% (HI, HI, disjoint)**.

## 5. Singletons
- 37.5% have no name-similar S2/S3 record at all (easy).
- 19.7% have a same-name record at a different address that belongs to ANOTHER S1 (chains).
- **20%** have a near-miss distractor on the same street with a different/secondary number
  (13.5% disjoint + 4.5% other + 1.2% same first number).
-> about 1.1% of the total macro score depends on rejecting same-street near-misses.

## 6. Blocking recall ceiling (true pairs, sample of 550k)
- 14.6% share no core-name token (88% of Indic-script pairs, 9.4% of Latin pairs).
- 4.5% share no address token (mostly empty S2/S3 addresses).
- **0.02%** share neither -> union of name and address channels loses almost nothing.

## 7. Test set
- 1,732,544 S1 (India 47%, US 38%, **France 15%**); 9,969,589 S2/S3 (France 1.43M).
- France S1 names are templated: `{Word} {Word} {SARL|SAS|EURL|SA|SASU|SCI|EI}` (SARL 28%,
  SAS 20%, EURL 6.5%, SA 4.9%, SASU 4.1%, SCI 3.2%, EI 1.6%; `(France)` inserted often;
  `Ets`/`Etablissements`, `Maison`, `Cie`, `Frères`, `Fils`, `Association`, `Amicale`, `Comité`).
- Only ~15 French cities (Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais,
  Saint-Nazaire, Pessac, La Teste-de-Buch, Mérignac, Lège-Cap-Ferret, Pornic, Saint-Herblain,
  La Baule-Escoublac): city is not discriminative, street + number is.
- France S2/S3 noise mirrors US/India: `R.`/`R`/`AV`/`ALL`/`IMP.`, `N°`/`Nº`/`No.`, `55BIS`,
  `004`, `E.U.R.L.`, `(S.A.S) Rimes Centre`, `formerly`, domain names with legal form glued in
  (`mieuxgroupementsarl.com`), French insertion words (`Participations`, `Distribution`, `Union`,
  `Et Fils`, `Cie`, `Comite`, `Pharmacie`, `Institut`, `Groupe`, `Developpement`).
- India test has Gurmukhi (Punjabi) names too (12.5k rows, also present in train: 10.8k).
