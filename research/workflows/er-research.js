export const meta = {
  name: 'er-competition-research',
  description: 'Exhaustive live-web research sweep for Amazon ML Challenge 2026 entity resolution, deep-read of top sources, and gap analysis vs our pipeline',
  phases: [
    { title: 'Sweep', detail: '12 parallel research agents, one per modality/topic' },
    { title: 'Deep-read', detail: 'Extract code-level details from top URLs' },
    { title: 'Synthesize', detail: 'Consolidated report + gap analysis' },
  ],
}

const CONTEXT = `
CONTEXT (what we already know — do NOT re-report these, find what is NEW):
- Competition: Amazon ML Challenge 2026 (Unstop, Sep 25-27 2026, 72h hackathon). Task: Business Entity Resolution. 3 sources (S1 reference, S2, S3). For each S1 entity, list all S2/S3 matching IDs. Metric: macro-averaged F0.5 per S1 entity (singletons: empty prediction = 1.0, any prediction = 0.0). Train: US + India only (12.5M rows). Test: US + India + FRANCE (unseen country, 15% of S1). 5.6% singletons. Each S2/S3 record matches at most ONE S1. 26% of S2/S3 rows are unmatched distractors.
- Observed noise: Indic-script transliterated names (Devanagari/Tamil/Telugu/Kannada/Malayalam/Bengali/Gujarati/Oriya) for ~15% of Indian S2/S3 records; domain-name-collapsed names ('cardiologysafecare.com', 'sreetradecom'); leetspeak ('Cardi0logy', '5afe', 'A1len'); truncated house numbers (607->60, 2007->007->02007); random garbage prefixes ('#867', '##', '***', 'N/A', '<NULL>'); state abbreviations vs full names (TX/Texas, TG/Telangana, Indic-script state names); word reorder/duplication/insertion ('Dr','Smt','Center','Services'); completely random names matched only by address ('Iriecto'); 'X d/b/a Y' patterns; accented vowel injection ('FÁRMS').
- Our current pipeline: unidecode+lowercase normalization; legal-suffix stripping; 4 inverted indexes for blocking (name tokens IDF-weighted, name token pairs, address tokens, address number+word combos), top-120 candidates per S1, per-country; 40 handcrafted rapidfuzz features (JaroWinkler, ratio, token_sort, token_set, partial, jaccard, containment, number agreement, lengths, blocking meta-features); CatBoost classifier; entity-level threshold sweep; singleton floor; France street-similarity post-filter; optional mdeberta-v3-base cross-encoder as extra feature.
- The #12 team's pipeline (known): same blocking idea, 36 features, CatBoost, threshold 0.625, France filter on street similarity >=75 + house-number non-conflict.
- Constraints: models must be MIT/Apache-2.0 and <=8B params; NO external data lookup (no APIs/geocoding/registries); compute is one SageMaker ml.g5.2xlarge (A10G 24GB, 32GB RAM, 8 vCPU) for ~20 hours.
TOOLS: first call ToolSearch with query "select:WebSearch,WebFetch" to load web tools, then search extensively (at least 8-15 distinct queries, follow links, read actual pages/code). Return concrete, specific, actionable findings with URLs. Prefer primary sources (code, notebooks, papers) over summaries. If a source is paywalled/unavailable, say so. Never fabricate URLs or scores.`

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' },
      url: { type: 'string' },
      source_type: { type: 'string', description: 'kaggle|github|reddit|hf|paper|library|blog|leaderboard|other' },
      summary: { type: 'string', description: '2-6 sentences of concrete content, techniques, numbers' },
      actionable_for_us: { type: 'string', description: 'Exactly what we should change/add in our pipeline because of this' },
      est_impact: { type: 'string', description: 'high|medium|low on macro F0.5' },
      deep_read: { type: 'boolean', description: 'true if this URL contains code/details worth a dedicated deep read' },
    }, required: ['title','url','source_type','summary','actionable_for_us','est_impact','deep_read'] } },
    queries_run: { type: 'array', items: { type: 'string' } },
    dead_ends: { type: 'array', items: { type: 'string' }, description: 'what you searched that returned nothing useful' },
  },
  required: ['findings','queries_run'],
}

const DEEP_SCHEMA = {
  type: 'object',
  properties: {
    url: { type: 'string' },
    accessible: { type: 'boolean' },
    blocking_strategy: { type: 'string' },
    preprocessing: { type: 'string' },
    features: { type: 'string', description: 'exhaustive list of features/comparators used, with exact definitions where visible' },
    model_and_training: { type: 'string' },
    postprocessing_and_thresholds: { type: 'string' },
    reported_scores: { type: 'string' },
    code_snippets: { type: 'string', description: 'verbatim key snippets (<= 80 lines total) most useful to port' },
    novel_ideas_vs_our_pipeline: { type: 'array', items: { type: 'string' } },
  },
  required: ['url','accessible','novel_ideas_vs_our_pipeline'],
}

const TOPICS = [
  { key: 'kaggle', prompt: `Find EVERY public Kaggle notebook and dataset for "Amazon ML Challenge 2026" / "ML Challenge 2026" business entity resolution (search kaggle.com directly and via web search: "amazon ml challenge 2026 kaggle", "entity resolution S1 S2 S3 matching_results.tsv", "train_source1.tsv", "F0.5 entity resolution kaggle notebook"). For each notebook, open it and extract: blocking method, features, model, threshold, reported validation F0.5, France handling, singleton handling. Also look at notebook comments/discussion for tricks.` },
  { key: 'github', prompt: `Find GitHub repositories for Amazon ML Challenge 2026 entity resolution. Use web search AND GitHub search URLs like https://github.com/search?q=%22amazon+ml+challenge+2026%22&type=repositories , https://github.com/search?q=train_source1.tsv&type=code , https://github.com/search?q=%22matching_results.tsv%22&type=code , https://github.com/search?q=%22candidate_pairs.tsv%22&type=code , topics amazon-ml-challenge, business-entity-resolution. Open each repo, read README and core files, extract blocking/features/model/threshold/scores and anything clever (transliteration, address parsing, 1-to-1 assignment).` },
  { key: 'social', prompt: `Find discussions of the Amazon ML Challenge 2026 (Sept 2026, entity resolution, Unstop) on Reddit (r/MachineLearning, r/Btechtards, r/learnmachinelearning, r/developersIndia, r/india), Twitter/X, LinkedIn posts, Discord/Telegram summaries, Medium/Dev.to/Substack blogs, YouTube. Extract: reported leaderboard scores and ranks, what approaches people say work / do not work, complaints (e.g. about France, singletons, memory), any shared code or feature lists, leaderboard plateaus, tips about thresholds.` },
  { key: 'huggingface', prompt: `Investigate HuggingFace for this challenge: datasets akshatbakshi/amazon-ml-challenge-2026 and logicalguy/amazon-ml-challenge-2026 (open their dataset cards, discussion/community tabs, any linked notebooks/spaces), search HF for "amazon ml challenge 2026", "entity resolution business", spaces/models fine-tuned for this. Also identify the best MIT/Apache-2.0 licensed multilingual models <=8B useful for cross-encoding or embedding French+English+Indic business names (e.g. mdeberta-v3-base license? xlm-roberta license? bge-m3 license? multilingual-e5? gte-multilingual? IndicBERT/MuRIL license?) — report license and params explicitly and whether the license satisfies MIT/Apache-2.0.` },
  { key: 'papers', prompt: `Survey the academic state of the art in entity resolution / entity matching 2023-2026 with a bias toward what is practical for 12M records in 20 hours on one A10G: Ditto, Sudowoodo, Unicorn, HierGAT, DADER, ZeroER, MatchGPT/LLM-based EM (Peeters & Bizer), JointBERT, PromptEM, AnyMatch, Jellyfish, KAER, blocking papers (DeepBlocker, Sparkly, SC-Block, Sudowoodo blocking, BM25-based blocking, "blocking with token IDF" ), and any 2025-2026 arXiv papers on address/business/POI matching. For each: what technique, what gains vs GBDT+features, what is implementable quickly, license of released models.` },
  { key: 'libraries', prompt: `Study production entity-resolution libraries for features and blocking tricks we may be missing: splink (comparison levels, term-frequency adjustments, array intersection, name/address comparison templates, phonetic + Jaro-Winkler + Damerau, 'exact match on column with TF', 'cosine on tf-idf'), dedupe.io (string/predicate learning, Affine gap distance, address/name types), recordlinkage (compare.String methods, sortedneighbourhood indexing), py_entitymatching/Magellan (feature auto-generation catalogue), Zingg, PyJedAI, libpostal (address parsing/expansion rules - do NOT propose external lookups, only offline rules), usaddress, Name-matching libraries (nominally, cleanco, company-name-matching, python-Levenshtein, jellyfish phonetics: soundex/metaphone/nysiis/match_rating). Report EXACT comparator definitions worth porting and any published benchmarks of which comparators matter most for business names/addresses.` },
  { key: 'past_amazon', prompt: `Find write-ups and code of top solutions of previous Amazon ML Challenges (2021 browse-node classification, 2023 product length, 2024 entity extraction from images, 2025 if it existed) AND the specific Amazon ML Challenge 2026 problem statement documents/blogs. Goals: (1) how did winning teams structure 72h efforts, what stack/compute, what evaluation hygiene; (2) any pattern in how Amazon builds synthetic noise for these datasets (e.g. their noise generators, distractor construction) that we can exploit; (3) any info on how public vs private leaderboard splits were done historically and shake-up magnitude.` },
  { key: 'other_competitions', prompt: `Extract concrete techniques from winning solutions of highly analogous competitions: Kaggle Foursquare Location Matching 2022 (POI name+address matching; read 1st-5th place write-ups: candidate generation via TF-IDF/KNN, name/address feature sets, LightGBM/XGBoost stage-1, transformer stage-2, per-pair vs per-entity thresholds, 'top-N + relative margin' selection, post-processing via clustering/connected components, cross-validation strategy), SIGMOD Programming Contest 2020/2021/2022 (entity resolution on products/notebooks; blocking + matching tricks), DI2KG, WDC Product matching, Shopee product matching 2021 (thresholding relative to max similarity, "min2" trick), KDD Cup ESCI. For each, give feature definitions and post-processing rules VERBATIM enough to implement.` },
  { key: 'french', prompt: `Compile an exhaustive offline-rules reference for normalizing FRENCH business names and addresses for fuzzy matching (no external lookups): all French legal forms and abbreviations (SARL, SAS, SASU, SA, EURL, SCI, SNC, SCOP, SCP, SELARL, SELAS, SCM, GIE, EARL, EI, EIRL, SEM, SCA, SCS, Ets, Sté, Cie, Fils, Frères, Groupe, Holding, etc.) with variants; French address abbreviations (rue/r., avenue/av./ave, boulevard/bd/bld/boul, place/pl., chemin/ch./chem., route/rte, impasse/imp., allée/all., faubourg/fbg, saint/st/ste/sainte, lieu-dit/ld, zone artisanale ZA/ZI/ZAC, résidence/res, bâtiment/bat, appartement/appt, cedex, bis/ter/quater, BP, arrondissement formats like 'Paris 8e' / '75008'), La Poste's official abbreviation list (AFNOR NF Z 10-011), apostrophe/elision handling (l', d', "L'Atelier" vs "Atelier"), accent/cedilla/ligature (œ, æ) handling, common French stopwords for names. Also find any resources on French business name typo/variation patterns and on matching SIRENE-style denominations to trade names (enseigne vs dénomination). Report as ready-to-use Python dict/list snippets.` },
  { key: 'indic', prompt: `Research how to match INDIC-SCRIPT business names (Devanagari/Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Oriya) against their Latin/romanized forms WITHOUT external lookups: evaluate unidecode's transliteration quality for Indic scripts, alternatives with permissive licenses (indic_transliteration package (MIT?), ai4bharat IndicXlit (MIT), libindic soundex / indic-soundex, indic NLP library, Aksharamukha (license?), ICU transliteration via PyICU), phonetic matching approaches robust to romanization variants (Double Metaphone, Match Rating, Editex, 'Indic phonetic hashing'), and the idea of LEARNING a token-level transliteration dictionary from aligned training pairs. Also compile Indian state names/abbreviations (all states+UTs, 2-letter codes, common Indic-script spellings, 'Keralam', 'Bengaluru/Bangalore', 'Gurgaon/Gurugram', 'Bombay/Mumbai' etc.) and Indian address abbreviation rules (H.No, Fl No, Plot No, Gali, Chowk, Marg, Opp, Nr, Behind, PO, Dist, Tal, Teh, Vill) as Python dicts. Report which solutions are pip-installable, their license, and speed.` },
  { key: 'leaderboard', prompt: `Find live intelligence about the Amazon ML Challenge 2026 leaderboard as of Sept 26-27 2026: top scores, how many teams above 0.98/0.99, what the public leaderboard subset looks like, any Unstop/LinkedIn/X/Reddit posts quoting exact F0.5 scores and describing approaches, any leak/warnings, submission limits, team posts about 'France', 'singletons', 'threshold'. Search Unstop hackathon page (https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604) and social. Also find the official guidelines PDF contents if published online (submission count limits, zip requirements, licensing of models).` },
  { key: 'postprocessing', prompt: `Research decision-theoretic and post-processing techniques for maximizing per-entity macro F-beta (beta=0.5) in set-prediction/entity resolution: optimal thresholding per entity given calibrated probabilities (expected-F-beta maximization, choosing the subset that maximizes E[F0.5] given independent match probabilities — derive or find the algorithm), 1-to-1 / many-to-one assignment constraints (each S2/S3 record belongs to at most one S1 entity: Hungarian, greedy argmax, bipartite matching) and when they help, relative thresholding (keep candidates within delta of the top probability), calibration (isotonic/Platt) before thresholding, per-group thresholds (per country / per candidate-count bucket), stacking/ensembling of LightGBM+XGBoost+CatBoost with different seeds, and handling of an unseen domain (France) via domain-invariant features and adversarial validation. Provide formulas and pseudo-code.` },
]

phase('Sweep')
log(`Launching ${TOPICS.length} research agents`)
const sweep = await parallel(TOPICS.map(t => () =>
  agent(`${CONTEXT}\n\nYOUR TOPIC (${t.key}):\n${t.prompt}\n\nBe exhaustive. Run many varied queries. Open and read the actual pages. Report 5-20 findings.`,
    { label: `sweep:${t.key}`, phase: 'Sweep', schema: FINDINGS_SCHEMA })
))
const sweepResults = sweep.map((r, i) => ({ key: TOPICS[i].key, result: r })).filter(x => x.result)
const allFindings = sweepResults.flatMap(x => (x.result.findings || []).map(f => ({ ...f, topic: x.key })))
log(`Sweep done: ${allFindings.length} findings from ${sweepResults.length}/${TOPICS.length} agents`)

// dedupe URLs and pick deep-read candidates
const seen = new Set()
const deepCands = []
for (const f of allFindings) {
  const u = (f.url || '').replace(/[#?].*$/, '').replace(/\/$/, '')
  if (!u || seen.has(u)) continue
  seen.add(u)
  if (f.deep_read && (f.est_impact === 'high' || f.est_impact === 'medium')) deepCands.push({ ...f, url: u })
}
const MAX_DEEP = 14
const deepList = deepCands.slice(0, MAX_DEEP)
if (deepCands.length > MAX_DEEP) log(`Deep-read capped: ${deepCands.length} candidates, reading ${MAX_DEEP} (dropped: ${deepCands.slice(MAX_DEEP).map(d => d.url).join(' | ')})`)

phase('Deep-read')
const deep = await parallel(deepList.map(f => () =>
  agent(`${CONTEXT}\n\nDEEP READ this source: ${f.url}\nTitle: ${f.title}\nWhy flagged: ${f.summary}\n\nOpen it with WebFetch (and any linked code files / notebook cells / sub-pages / raw GitHub files). Extract EXACT implementation details: blocking, preprocessing/normalization rules, every feature with its definition, model + hyperparameters, thresholds and post-processing rules, reported scores. Quote key code verbatim (<=80 lines). Then list every idea NOT already in our pipeline (see CONTEXT).`,
    { label: `deep:${f.topic}:${f.url.slice(0, 40)}`, phase: 'Deep-read', schema: DEEP_SCHEMA })
))
const deepResults = deep.filter(Boolean)
log(`Deep-read done: ${deepResults.length}/${deepList.length}`)

phase('Synthesize')
const report = await agent(`${CONTEXT}

You are the synthesis lead. Below are (A) all sweep findings and (B) deep-read extractions from a 12-agent live-web research sweep. Also read the file D:\\Downloads\\Amazol-ML-Challenge\\CLAUDE_CONTEXT_TRANSFER.md (sections 6, 7, 11, 15) for our exact pipeline and prior research, so you can identify what is genuinely NEW.

Produce a rigorous research report in Markdown with these sections:
1. NEW intelligence not in our prior research (bullet list, each with URL and one-line why-it-matters), grouped: competition-specific (Kaggle/GitHub/social/leaderboard), technique-level (papers/libraries/other competitions), country-specific (France, India).
2. GAP ANALYSIS table: for each area (preprocessing, blocking, features, model, threshold/decision, post-processing, France, India/Indic, singletons, validation hygiene, engineering/scale), what we do vs what the best evidence says, and the expected direction/magnitude of impact on macro F0.5.
3. RANKED ACTION LIST: top 25 concrete changes ordered by (expected F0.5 gain x confidence / implementation time). Each: what, why (cite source), how (2-5 lines of pseudo-code or exact rule), risk.
4. Things that DON'T work / traps others fell into (with sources).
5. Ready-to-paste reference tables: French legal forms + address abbreviations; Indian states/UTs codes + Indic spellings + address abbreviations; US state abbreviations + street-suffix abbreviations (USPS Pub 28 core list); leetspeak map; domain-name-collapsed-name handling rule.
6. Open questions we should test empirically on our data.
Be specific, cite URLs inline, no filler. Write the report to the file D:\\Downloads\\Amazol-ML-Challenge\\research\\RESEARCH_REPORT.md (create the directory) using the Write tool, AND return the full report text.

(A) SWEEP FINDINGS:
${JSON.stringify(allFindings, null, 1).slice(0, 180000)}

(B) DEEP READS:
${JSON.stringify(deepResults, null, 1).slice(0, 180000)}
`, { label: 'synthesize', phase: 'Synthesize', effort: 'max' })

return { n_findings: allFindings.length, n_deep: deepResults.length, sweep_dead_ends: sweepResults.map(x => ({ key: x.key, dead_ends: x.result.dead_ends || [] })), report }