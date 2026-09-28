export const meta = {
  name: 'er-pipeline-review',
  description: 'Adversarial correctness/accuracy review of the entity-resolution pipeline v2, findings verified by independent skeptics',
  phases: [
    { title: 'Review', detail: 'one reviewer per module group, read-only' },
    { title: 'Verify', detail: '2 skeptics per finding, majority must confirm' },
    { title: 'Synthesize', detail: 'ranked confirmed findings with fixes' },
  ],
}

const SRC = 'D:\\Downloads\\Amazol-ML-Challenge\\student_resource\\code\\business_entity_resolution\\src'
const CONTEXT = `
You are reviewing a Python entity-resolution pipeline for a Kaggle-style competition (Amazon ML Challenge 2026).
Source dir: ${SRC}  (config.py, text_tables.py, translit.py, preprocess.py, blocking.py, features.py, evaluate.py, decision.py, france_filter.py, train.py, inference.py). Design notes: D:\\Downloads\\Amazol-ML-Challenge\\research\\DATA_FINDINGS.md and research\\RESEARCH_REPORT_MAIN.md (read the relevant parts).
Task: given S1 business records (name, address, country) find all matching S2/S3 records. Metric: macro F0.5 per S1 entity; singletons (no true matches) score 1 if predicted empty else 0. Each S2/S3 record belongs to at most one S1. Train: US+India (12.5M rows); test adds FRANCE (unseen; 15% of S1). Production run: SageMaker Linux, 8 vCPU, 32 GB RAM, A10G GPU; train.py then inference.py; multiprocessing uses fork on Linux (spawn on Windows dev laptop).
Data flow: preprocess (per-record normalisation) -> per-country Blocker (sparse TF-IDF top-k over 5 channels + reverse channel) -> candidate chunk DataFrame (s1_pos, s23_pos, meta) -> compute_features (112 features) -> LightGBM/CatBoost -> decision layer (per-country/per-bin thresholds selected on validation with distractor false positives weighted 1.9x; conflict resolution; France calibrated to 5.6% empty rate) -> matching_results.tsv / candidate_pairs.tsv.
Known-verified: a 40k-S1 sample run trains in 11 min and inference passes the official validator; unit tests pass for blocking/decision/evaluate/translit.
READ THE ACTUAL CODE with the Read tool (all of it for your modules; skim the others as needed to check interfaces). Do not run training. You may run tiny python snippets to confirm a suspected bug (python3 on this machine; set PYTHONIOENCODING=utf-8).
Report only REAL defects or concrete accuracy losses, each with: file:line, what happens, a concrete input that triggers it, impact (crash / wrong output / silent accuracy loss / train-inference skew / memory-time at full scale), and a minimal fix. No style comments. Prefer fewer, verified findings.`

const FINDINGS = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  file: { type: 'string' }, line: { type: 'integer' }, title: { type: 'string' },
  category: { type: 'string', description: 'crash|wrong-output|silent-accuracy-loss|train-inference-skew|scale|edge-case' },
  description: { type: 'string' }, trigger: { type: 'string' }, fix: { type: 'string' },
  severity: { type: 'string', description: 'high|medium|low' } },
  required: ['file', 'line', 'title', 'category', 'description', 'trigger', 'fix', 'severity'] } } }, required: ['findings'] }

const VERDICT = { type: 'object', properties: { real: { type: 'boolean' }, confidence: { type: 'number' }, reasoning: { type: 'string' }, corrected_fix: { type: 'string' } }, required: ['real', 'confidence', 'reasoning'] }

const GROUPS = [
  { key: 'preprocess', files: 'preprocess.py, text_tables.py, translit.py', focus: 'normalisation correctness for every noise pattern in DATA_FINDINGS.md (Indic names incl. mixed-script, domains, handles, leetspeak, dotted acronyms, d/b/a, brackets, junk, legal forms, admin components incl. Indic-script state names and French departments, N degree markers, hyphenated numbers, ranges, ordinals, empty strings, unicode edge cases like U+FFFD or emoji), asymmetry between S1 and S23 treatment, multiprocessing initializer correctness (globals _TRANS/_SEG under fork AND spawn), regex catastrophic cases, the _admin_lookup fallback for unknown countries, whether any transformation could differ between train.py and inference.py runs' },
  { key: 'blocking', files: 'blocking.py, config.py', focus: 'SparseVectorizer math (idf, l2 norm, transform with unseen tokens, duplicate tokens in a doc), sp_matmul_topn usage (threshold semantics, sort flag, dtype), rank computation, union/keying arithmetic (int64 overflow: row*n_s23+col and s23_local*n_s1c+s1_local), reverse channel correctness (searchsorted lookups, s1_local mapping when the chunk is a subset of the country S1 set, empty results), cap ordering, cand_count/fused_rank semantics, memory at full scale (6.2M S23 x vocab; char-3gram docs as Python lists), behaviour when a country has zero S23 or zero vocab, iter_candidate_chunks mask semantics' },
  { key: 'features', files: 'features.py (and blocking.BLOCK_META_COLS)', focus: 'every feature definition: NaN/inf risks, dtype issues, division by zero, features that depend on candidate-set density (train pool 4.67 vs test 5.75 S23 per S1) beyond the ones already dropped, features that could leak labels, ExtraTokenStats semantics, group features when a group has one candidate, compute_features_parallel ordering/alignment (inverse permutation), WORKERS global under fork, correctness of _first_num_relation and _num_features, legal_rel/name_dup fallbacks, feature_columns ordering stability between train and inference (LightGBM/CatBoost require identical column order!)' },
  { key: 'training', files: 'train.py, evaluate.py, decision.py', focus: 'split logic and leakage (stats/train/val disjointness; ExtraTokenStats learned on stats split but S23 pool shared), owner array and label correctness, negative subsampling vs group features, France-robustness masking, feature-column order saved vs used, density-adjusted sweep math (fp_weight), per-bin coordinate ascent, expected_f05_select correctness (first-best selection, min_prob), resolve_conflicts, best-config serialisation (json of numpy types!), error-analysis arithmetic, val_predictions saving, unseen-country settings derivation, anything that would crash only at full scale or only with CatBoost GPU' },
  { key: 'inference', files: 'inference.py, france_filter.py, decision.py', focus: 'model loading and feature order, KEEP_PROB filtering interaction with conflict resolution and per-bin thresholds, cand_lists assembly (complete groups per chunk), calibrate_unseen_threshold semantics and the adjusted target arithmetic, pair_groups for unseen countries, threshold_shift application to unseen thresholds, output writing (every S1 exactly once, ordering, dedup, subset of candidates, tab/newline/encoding), validator invocation, memory of P_* arrays at full scale (KEEP_PROB=0.02 over ~140M pairs), guardrail logic, apply_france_filter when enabled' },
]

phase('Review')
const reviews = await parallel(GROUPS.map(g => () =>
  agent(`${CONTEXT}\n\nYOUR MODULES: ${g.files}\nFOCUS: ${g.focus}\n\nRead every line of your modules. Return 3-15 findings.`,
    { label: `review:${g.key}`, phase: 'Review', schema: FINDINGS, effort: 'high' })))
const all = reviews.map((r, i) => ({ g: GROUPS[i].key, r })).filter(x => x.r).flatMap(x => x.r.findings.map(f => ({ ...f, group: x.g })))
log(`Review: ${all.length} candidate findings`)

phase('Verify')
const verified = await parallel(all.map(f => () =>
  parallel([0, 1].map(k => () =>
    agent(`${CONTEXT}\n\nA reviewer claims this defect. Your job is to REFUTE it if you can: read the code at the cited location and around it, trace the actual behaviour on the trigger input (run a tiny snippet if useful), and decide. Default to real=false when the claim is speculative, already handled elsewhere in the code, or has no accuracy/correctness impact.\n\nCLAIM (${f.group}, ${f.severity}): ${f.title}\nfile: ${f.file} line ${f.line}\n${f.description}\nTrigger: ${f.trigger}\nProposed fix: ${f.fix}\n\nLens: ${k === 0 ? 'correctness: does the code really do this?' : 'impact: even if the code does this, does it change outputs/metric/stability materially?'}`,
      { label: `verify:${f.group}:${f.title.slice(0, 30)}`, phase: 'Verify', schema: VERDICT })))
    .then(vs => ({ ...f, votes: vs.filter(Boolean) }))))
const confirmed = verified.filter(Boolean).filter(f => f.votes.length && f.votes.filter(v => v.real).length >= Math.ceil(f.votes.length / 2))
log(`Verify: ${confirmed.length}/${all.length} confirmed`)

phase('Synthesize')
const report = await agent(`${CONTEXT}\n\nHere are the CONFIRMED findings (each survived at least a majority of independent skeptics). Produce a ranked action list (most severe first) in Markdown: for each finding give file:line, one-paragraph explanation, trigger, and the exact code change (a unified diff-style snippet where practical). Merge duplicates. Then list the refuted-but-notable claims briefly. Write the report to D:\\Downloads\\Amazol-ML-Challenge\\research\\CODE_REVIEW.md with the Write tool and return it.\n\nCONFIRMED:\n${JSON.stringify(confirmed.map(f => ({ ...f, votes: f.votes.map(v => ({ real: v.real, conf: v.confidence, why: v.reasoning.slice(0, 400), fix: v.corrected_fix })) })), null, 1).slice(0, 150000)}\n\nREFUTED:\n${JSON.stringify(verified.filter(Boolean).filter(f => !confirmed.includes(f)).map(f => ({ title: f.title, file: f.file, line: f.line, why: f.votes.map(v => v.reasoning.slice(0, 200)) })), null, 1).slice(0, 40000)}`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'high' })
return { n_candidates: all.length, n_confirmed: confirmed.length, confirmed: confirmed.map(f => `${f.severity} ${f.file}:${f.line} ${f.title}`), report }