# Code review: business_entity_resolution/src (ranked action list)

Reviewed tree: `student_resource/code/business_entity_resolution/src` at HEAD `e4f0bbb` (working tree identical; `git diff HEAD` is empty). All line numbers below refer to that tree. Every item was re-checked against the current source; several of the confirmed claims were written against an older revision and are already fixed in HEAD (listed in section C so nobody re-applies them). Trigger snippets were run with `python3 -B` and a SegVocab built from the 40k-S1 local sample.

Verdict summary: no crash or wrong-output bug was found in the current tree. The remaining defects are (1) two full-scale memory peaks that make the 32 GB production box marginal, (2) one silent-accuracy issue in the France (unseen-country) calibration, (3) one measured regression introduced by an earlier fix in the leet/domain normaliser, and (4) two wall-time / observability items.

---

## A. Open findings, most severe first

### A1. `blocking.py:69-97, 192-203` - S2/S3 vocabulary fit materialises ~100 M Python `str` objects per channel (memory, full scale)

`Blocker.__init__` calls `channel_docs(s23_df, ch)` for the whole country (line 193), which for `name_chr` creates a new 3-char `str` per trigram per record (16.7/record on India, p95 34) and for `addr_num` an f-string per (number, word) combo (13.9/record on India). `_flatten` (56-67) copies the pointers into an object array, and `fit_transform` (69-97) runs `pd.factorize` + `np.unique` over it while `docs` is still alive (only deleted at line 203, after the fit). For US train S23 (6.19 M rows) that is ~103 M short strings (~4.5 GB) + list overhead (~1.2 GB) + object array (0.8 GB) + int64 `codes`/`pair`/`np.unique` sort copies (~4 x 0.8 GB): roughly 10-14 GB transient on top of ~15-20 GB resident (raw S2/S3 frame, preprocessed country frame with ~15 object columns, preprocessed S1). `del docs` after `_flatten` alone does not help, because `flat` keeps every string alive until factorize returns.

Trigger: `train.py` on full data (US: 6.19 M S2/S3 rows) or `inference.py` on test India (4.72 M rows) on ml.g5.2xlarge (32 GB). Impact: memory peak during the `name_chr`/`addr_num` fits approaches or exceeds RAM (OOM kill = crash); no effect on outputs.

Fix (keeps tokenisation, vocabulary order and cosine values identical, so blocking output and the exact-cosine features in `features.py` are unchanged): fit block-wise, mapping each block's factorised uniques onto a running global id dict, and build the CSR from concatenated int32 triples.

```diff
--- a/blocking.py
+++ b/blocking.py
@@ class SparseVectorizer
+    def fit_transform_blocks(self, df, channel, block=500_000):
+        """Same result as fit_transform(channel_docs(df, channel)) but never holds more than one block of docs."""
+        n_docs = len(df)
+        gid = {}                      # token -> global id, in first-appearance order (== pd.factorize order)
+        rows_l, cols_l, tf_l = [], [], []
+        for lo in range(0, n_docs, block):
+            docs = channel_docs(df.iloc[lo:lo + block], channel)
+            rows, flat = self._flatten(docs)
+            del docs
+            if len(flat) == 0:
+                continue
+            codes, uniques = pd.factorize(flat, sort=False)
+            del flat
+            gmap = np.fromiter((gid.setdefault(t, len(gid)) for t in uniques), dtype=np.int64, count=len(uniques))
+            g = gmap[codes]
+            pair = (rows + lo) * np.int64(1 << 31) + g          # rows are block-disjoint: per-block unique == global unique
+            pair_u, tf = np.unique(pair, return_counts=True)
+            rows_l.append((pair_u // (1 << 31)).astype(np.int32))
+            cols_l.append((pair_u % (1 << 31)).astype(np.int32))
+            tf_l.append(tf.astype(np.int32))
+        if not gid:
+            self.vocab = {}; self.idf = np.zeros(0, dtype=np.float32)
+            return sp.csr_matrix((n_docs, 0), dtype=np.float32)
+        r_u = np.concatenate(rows_l); c_u = np.concatenate(cols_l); tf = np.concatenate(tf_l)
+        del rows_l, cols_l, tf_l
+        df_ = np.bincount(c_u, minlength=len(gid))
+        max_df = max(self.min_df, int(self.max_df_frac * n_docs), 50)
+        keep = (df_ >= self.min_df) & (df_ <= max_df)
+        new_index = -np.ones(len(gid), dtype=np.int64)
+        new_index[keep] = np.arange(int(keep.sum()))
+        self.vocab = {tok: int(new_index[i]) for tok, i in gid.items() if keep[i]}
+        idf_full = np.log((1.0 + n_docs) / (1.0 + df_)) + 1.0
+        self.idf = idf_full[keep].astype(np.float32)
+        m = keep[c_u]
+        r_u, c_u, tf = r_u[m], new_index[c_u[m]], tf[m]
+        vals = (1.0 + np.log(tf)) if self.sublinear else tf.astype(np.float32)
+        vals = (vals * self.idf[c_u]).astype(np.float32)
+        X = sp.csr_matrix((vals, (r_u, c_u)), shape=(n_docs, len(self.idf)), dtype=np.float32)
+        return self._l2(X)
@@ class Blocker.__init__
         for ch in self.channels:
-            docs = channel_docs(s23_df, ch)
             v = SparseVectorizer(max_df_frac=cfg.BLOCK_MAX_DF_FRAC)
-            X = v.fit_transform(docs)
+            X = v.fit_transform_blocks(s23_df, ch)
             self.vec[ch] = v
@@
-            del docs
```

Peak transient drops from ~13 GB to ~1.5-2 GB per channel (one 500k block of docs plus ~12 B per token-in-doc of int32 triples). `transform()` is untouched (S1 chunks are <= 20k rows; the reverse channel's joint docs are small). Optional further speed-up: for `name_chr` encode trigrams as `int32 = (b0<<16)|(b1<<8)|b2` from the concatenated latin-1 byte buffer (name_collapsed is ASCII after unidecode) and factorize int32; `transform` must then use the same encoder. Also log `psutil.Process().memory_info().rss` after each channel in `Blocker.__init__` so the 32 GB run can be validated from the log.

### A2. `train.py:301-308, 335-357` - full-scale training matrix + CatBoost copy peaks near 32 GB at the default `TRAIN_S1_ENTITIES=400000`

The negative subsampling itself is correct (`train.py:89/277` key on `cand['fused_rank']`, the within-S1 rank; retention is ~51%, ~52 rows/S1 incl. positives, not the 80% claimed against an older revision). The memory arithmetic still does not support the docstring at `train.py:19` ("Fits in 32 GB for the full data with defaults"): at 400k train S1 and 112 float32 features, `X_tr` is ~20.8 M x 112 = ~9.3 GB and `X_va` ~4.5 GB. Line 301 `np.concatenate(train_X)` runs while the whole `train_X` list is alive (another 9.3 GB) and `val_X` is alive too (4.5 GB): peak ~23 GB before `del train_X, val_X` at 308. Later, `cb.fit(X_tr, ...)` at 352 builds CatBoost's own raw Pool (~11 GB) plus its quantized copy, while `X_tr`, `X_va`, and the LightGBM `dtrain`/`dval` (built with `free_raw_data=False`, ~4 GB binned) are all still referenced (`del X_tr` only happens at 357). Rough peak: ~30 GB before `df_s1` and metadata.

Trigger: `python train.py` with defaults (`ER_SAMPLE_S1` unset, CatBoost enabled) on ml.g5.2xlarge. Impact: OOM kill at STEP 8/9 after the whole blocking/feature pass (hours lost). The runbook's `ER_SAMPLE_S1=100000` path (~2.3 GB `X_tr`) is safe.

Fix (no accuracy effect):

```diff
--- a/train.py
+++ b/train.py
@@ STEP 8
-    X_tr = np.concatenate(train_X)
-    y_tr = np.concatenate(train_y)
-    w_sub_all = np.concatenate(train_w).astype(np.float32)
-    meta_tr = pd.concat(train_meta, ignore_index=True)
-    X_va = np.concatenate(val_X)
-    y_va = np.concatenate(val_y)
-    meta_va = pd.concat(val_meta, ignore_index=True)
-    del train_X, val_X
-    gc.collect()
+    X_tr = np.concatenate(train_X); del train_X; gc.collect()      # free the chunk list before touching val
+    y_tr = np.concatenate(train_y)
+    w_sub_all = np.concatenate(train_w).astype(np.float32)
+    meta_tr = pd.concat(train_meta, ignore_index=True)
+    X_va = np.concatenate(val_X); del val_X; gc.collect()
+    y_va = np.concatenate(val_y)
+    meta_va = pd.concat(val_meta, ignore_index=True)
@@ STEP 9
     p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
     probs = {'lgbm': p_va}
+    del dtrain, dval; gc.collect()        # binned copies are no longer needed
 
     if cfg.USE_CATBOOST:
         try:
-            from catboost import CatBoostClassifier
+            from catboost import CatBoostClassifier, Pool
             cb = CatBoostClassifier(**cfg.CATBOOST_PARAMS)
-            cb.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va), use_best_model=True)
+            train_pool = Pool(X_tr, y_tr, weight=w_tr)
+            del X_tr; gc.collect()          # CatBoost now owns the only copy of the train matrix
+            eval_pool = Pool(X_va, y_va)
+            cb.fit(train_pool, eval_set=eval_pool, use_best_model=True)
+            del train_pool
             cb.save_model(os.path.join(cfg.MODEL_DIR, 'catboost.cbm'))
-            probs['catboost'] = cb.predict_proba(X_va)[:, 1]
+            probs['catboost'] = cb.predict_proba(eval_pool)[:, 1]
         except Exception as e:  # noqa
             log.warning(f"CatBoost failed ({e}); continuing with LightGBM only")
-    del X_tr, y_tr
+    X_tr = None; del y_tr
     gc.collect()
```

(Alternatively preallocate `X_tr` from the summed chunk shapes and fill it inside the Pass B+C loop.) Also either lower `config.py:96` default to ~250,000 or correct `train.py:19` and the runbook to state that 400k + CatBoost needs 64 GB (`ml.g5.4xlarge`) or `ER_NO_CATBOOST=1`. Log `psutil` RSS before `lgb.train` and before `cb.fit`.

### A3. `inference.py:256-266` (+ `decision.py:160-171`) - France threshold is calibrated on "any pair >= t", not on the decision rule that is applied

`calibrate_unseen_threshold` counts an entity as empty iff no candidate pair has `P >= t`, over all of the country's pairs. `decide()` (called at `inference.py:268`) then applies (a) `t_c + noaddr_delta` to address-less S2/S3 candidates (`thresholds[c+'|noaddr']`, line 263; delta was +0.13 in the sample-run `model_config.json`), (b) `resolve_conflicts`, and (c) `size_adaptive`/`expected_f` when configured. Each of these can only remove selected pairs, so the true predicted-empty rate after `decide()` is >= the calibrated one; the raise-only loop therefore stops at a `t_c` whose real empty rate overshoots the 5.59% target by an unmeasured amount, i.e. France (15% of test S1, every missed non-singleton costs a full entity score) ends stricter than intended, and the log line at 264-266 reports a rate that is not the final one. Reproduced on synthetic pairs: calibrated 0.061 vs 0.163 after `decide()` (direction is guaranteed; magnitude depends on the model).

Trigger: a France S1 whose only candidates above `t_c` are address-less records with probability in `[t_c, t_c + noaddr_delta)`, or whose best record is claimed with a higher probability by another S1.

Fix: calibrate with the applied rule (conflicts are within-country because S2/S3 are matched per `country_norm`, so restricting to `cm` is exact):

```diff
--- a/inference.py
+++ b/inference.py
@@
         if n_with_pairs:
             no_cand_rate = (n_c - n_with_pairs) / n_c
             target_adj = max(0.0, (target - no_cand_rate) / max(1e-9, n_with_pairs / n_c))
-            t_c, rate_c = calibrate_unseen_threshold(P_p[cm], codes_c, n_with_pairs, start_t, target_adj,
-                                                     max_t=float(unseen.get('max_threshold', cfg.UNSEEN_MAX_THRESHOLD)))
+            delta_na = float(unseen.get('noaddr_delta', 0.0)) if mcfg.get('per_bin_thresholds', True) else 0.0
+            max_t = float(unseen.get('max_threshold', cfg.UNSEEN_MAX_THRESHOLD)); step = 0.005
+            def applied_empty_rate(t):
+                cfg_c = dict(dec, thresholds={c: t, c + '|noaddr': min(0.99, t + delta_na)}, default_threshold=t)
+                m_c = decide(P_s1[cm], s23_code[cm], P_p[cm], P_group[cm], cfg_c)
+                return float((np.bincount(codes_c[m_c], minlength=n_with_pairs) == 0).mean())
+            t_c = float(start_t); rate_c = applied_empty_rate(t_c)
+            while rate_c < target_adj and t_c + step <= max_t:
+                t_c = round(t_c + step, 4); rate_c = applied_empty_rate(t_c)
             overall_empty = no_cand_rate + rate_c * n_with_pairs / n_c
```

Keep lines 261-263 as they are; relabel the log as "applied-rule predicted-empty rate". Cost: one `decide()` over France pairs per 0.005 step (a lexsort of selected pairs), well under a minute at full scale. Apply the same change to the LOCO calibration in `train.py` so the LOCO "calibrated" row measures the same rule. Expected effect: France threshold lower by ~0.005-0.02; sign of the metric change is favourable in expectation but small - do not spend a submission slot on it alone.

### A4. `preprocess.py:281-288` - segment-then-per-token de-leet regresses genuine leet domains (measured)

The bounded `_LEET_TOKEN_RE` (line 61) is correct now, but the domain/handle branch was changed to "segment first, then `_deleet_token` per token". A leet digit inside a collapsed seed breaks the Viterbi segmentation before any repair can happen, so the documented cases regress: `cardi0logysafecare.com -> 'c ar di 0 lo gy safe care'` (S1 `Cardiology Safe Care`), `visi0ntechnologies.com -> 'vi si 0n technologies'`, `so1utionskyros.com -> 'solu tion sky ros'`, `a1lenpi1ates -> 'a1 le npi 1 at es'`. Measured on the 264 ground-truth S2/S3 domain-like names with digits in the sample (0.19% of S2/S3 rows): mean S1-token recall of the resulting `name_core` is 0.514 (current), 0.595 (bounded-regex whole-seed de-leet, the pre-change variant), 0.659 (segmentation-score comparison below). Leetspeak affects 2.6% of names overall (DATA_FINDINGS), but only domain-collapsed names take this path, so the metric impact is small (<1e-4); it is nevertheless a strict regression relative to the intended behaviour and cheap to fix.

Trigger: `normalize_name('cardi0logysafecare.com')['name_core']` should be `'cardiology safe care'`, is `'c ar di 0 lo gy safe care'`; `'4528mountainview.com'` must stay `'4528 mountain view'`.

Fix: choose between the raw seed and its de-leeted form by segmentation score (the one the vocabulary explains better wins; a 4-digit house number never wins because `LEET_MAP` turns it into an unknown letter run):

```diff
--- a/preprocess.py
+++ b/preprocess.py
@@ class SegVocab
+    def score(self, s):
+        """Best Viterbi log-prob of s (same scoring as segment(); 0.0 for empty / over-long input)."""
+        n = len(s)
+        if n == 0 or n > 60:
+            return 0.0
+        best = [0.0] + [-1e18] * n
+        lp, unk = self.logp, self.unk
+        for i in range(1, n + 1):
+            bi = -1e18
+            for j in range(max(0, i - self.maxlen), i):
+                w = s[j:i]
+                sc = lp.get(w)
+                if sc is None:
+                    sc = -6.0 if w.isdigit() else unk * (i - j)
+                if best[j] + sc > bi:
+                    bi = best[j] + sc
+            best[i] = bi
+        return best[n]
@@ normalize_name
     if collapsed_seed is not None:
-        if _SEG is not None:
-            # segment first, then repair leetspeak per token (keeps genuine digit runs such as '4528')
-            seg_toks = _SEG.segment(collapsed_seed).split()
-            s = ' '.join(_deleet_token(t) for t in seg_toks)
-            collapsed_seed = s.replace(' ', '')
-        else:
-            s = collapsed_seed
+        if any(ch.isdigit() for ch in collapsed_seed):
+            fixed = ''.join(T.LEET_MAP.get(ch, ch) if ch.isdigit() else ch for ch in collapsed_seed)
+            if _SEG is None:
+                if _LEET_TOKEN_RE.match(collapsed_seed):        # bounded regex when no vocabulary is available
+                    collapsed_seed = fixed
+            elif _SEG.score(fixed) > _SEG.score(collapsed_seed):  # 'cardi0logysafecare' -> 'cardiologysafecare'; '4528mountainview' stays
+                collapsed_seed = fixed
+        s = _SEG.segment(collapsed_seed) if _SEG is not None else collapsed_seed
```

Add `cardi0logysafecare.com`, `visi0ntechnologies.com`, `4528mountainview.com`, `017madras600.com`, `Team2000 Inc` to the `__main__` tests. Retrain after the change (S1 and S2/S3 are both re-preprocessed through `preprocess_dataframe`, so train and inference stay symmetric).

### A5. `train.py:232, 265` + `blocking.py:409` - the Blocker is still built twice per country

Pass B and Pass C are already merged (one `iter_candidate_chunks` call for train+val at line 265), but Pass A (stats split, line 232) builds its own Blocker for the same `df_s23_c`. `Blocker.__init__` fits all five S2/S3 vectorizers (the A1 transient) and runs the reverse product `X_joint (S23 x V) @ A.T (V x all-S1-of-country)`; neither depends on `s1_mask`, so the second build is identical work. Impact: blocking wall time x2 per country and the A1 memory peak hit twice; outputs unchanged. Pass A must finish over all countries before features are computed (`extra_stats = acc.finish()` at 250), so the fix is to cache, not to reorder.

Fix (minimal, keeps pass structure): add an optional cache to `iter_candidate_chunks` and spill the Blocker to disk between passes so the memory peak is not raised.

```diff
--- a/blocking.py
+++ b/blocking.py
-def iter_candidate_chunks(df_s1, df_s23, chunk_size=None, max_candidates=None, s1_mask=None):
+def iter_candidate_chunks(df_s1, df_s23, chunk_size=None, max_candidates=None, s1_mask=None, blocker_cache=None):
@@
-        blocker = Blocker(df_s23.iloc[s23_pos_all], s23_pos_all,
-                          s1_df_country=df_s1.iloc[s1_pos_country] if rev_k else None,
-                          s1_pos_country=s1_pos_country, reverse_k=rev_k)
+        blocker = None
+        if blocker_cache is not None and os.path.exists(blocker_cache.get(country, '')):
+            with open(blocker_cache[country], 'rb') as f:
+                blocker = pickle.load(f)
+        if blocker is None:
+            blocker = Blocker(df_s23.iloc[s23_pos_all], s23_pos_all,
+                              s1_df_country=df_s1.iloc[s1_pos_country] if rev_k else None,
+                              s1_pos_country=s1_pos_country, reverse_k=rev_k)
+            if blocker_cache is not None:
+                path = os.path.join(cfg.MODEL_DIR, f'blocker_{country}.pkl')
+                with open(path, 'wb') as f:
+                    pickle.dump(blocker, f, protocol=pickle.HIGHEST_PROTOCOL)
+                blocker_cache[country] = path
```

In `train.py` pass the same dict to both calls (`blocker_cache=bcache`) and delete the pickles after Pass B+C. The Blocker holds only numpy/scipy arrays and dicts, so it pickles cleanly (a few GB per country on disk).

### A6. `features.py:609-641` - forked feature workers: refcount-driven copy-on-write growth is unmeasured

`gc.collect(); gc.freeze()` before `ctx.Pool` and `gc.disable()` in `_par_init` are already in place (lines 600, 634-640), so cyclic-GC page dirtying is handled. What remains is that every child fancy-indexes the inherited object columns (`df_s23[c].values[p2]`, `.tolist()`, `set(...)`), which INCREFs each touched string/list and dirties its page: ~200k scattered S2/S3 rows x ~30 objects per child, on the order of 0.5-1 GB per child (4-8 GB for 8 children) on top of a ~12 GB parent for India. This path has never run at full scale (the verified 40k run was serial on Windows). Impact: possible OOM on the 32 GB box during India inference; no output effect.

Fix: (1) measure first - after the first `pool.map` log `resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss` and the parent RSS; (2) if it is tight, hand each part only the rows it needs so children never touch the big frames:

```python
# in compute_features_parallel, replacing the parts/initargs construction
payloads = []
for part in parts:
    u1 = np.unique(part['s1_pos'].values); u2 = np.unique(part['s23_pos'].values)
    sub1 = df_s1.iloc[u1].reset_index(drop=True); sub2 = df_s23.iloc[u2].reset_index(drop=True)
    part = part.assign(s1_pos=np.searchsorted(u1, part['s1_pos'].values),
                       s23_pos=np.searchsorted(u2, part['s23_pos'].values))
    payloads.append((part, sub1, sub2))
# _par_init(vecs, extra_stats); _par_work((part, sub1, sub2)) -> compute_features(part, sub1, sub2, vecs=..., extra_stats=...)
```

`compute_features` output carries no positions (`BLOCK_META_COLS` excludes `s1_pos`/`s23_pos`), so remapped positions do not leak; the caller's `inv` reorder restores candidate order as today. Add an `ER_FEAT_JOBS` env var to cap only the feature pool without shrinking LightGBM / preprocessing threads.

---

## B. Recommended order of work

1. A1 (block-wise vectorizer fit) and A2 (train-matrix / CatBoost memory) - these decide whether the full-data run survives on 32 GB; both are output-neutral and can be verified with the psutil log lines on the 40k sample before the SageMaker run.
2. A4 (leet/domain seed) - small, output-changing, needs a retrain; bundle it with the A1/A2 retrain.
3. A3 (France calibration on the applied rule) - inference-only, no retrain, low risk.
4. A5, A6 - wall time and observability; A6's RSS logging should ship with the first full run so the next one is tuned from real numbers.

---

## C. Confirmed claims that are already fixed in HEAD `e4f0bbb` (do not re-apply)

- **Hyphenated French regions never resolve to an admin token** (`preprocess.py:394-398`). Fixed: the admin lookup key is `_WS_RE.sub(' ', cn.replace('-', ' '))`. Verified: `normalize_address('19 R DE LA LIBERTE, LILLE, Hauts-de-France', 'france')` -> `admin='zzfrhautsdefrance'`, `addr_words=['liberte','lille']`; `Pas-de-Calais`, `Nouvelle-Aquitaine`, `Provence-Alpes-Cote d'Azur`, `Ile-de-France`, `Bourgogne-Franche-Comte` all resolve. The other reviewer measured France S1 "admin found" 0.28 -> 1.00 with this change. Keep an eye on the preprocess log line at full scale.
- **`_DOMAIN_RE` chops a TLD-looking suffix off single-token names** (`preprocess.py:54-55, 272-274`). Fixed: a literal `.tld` is required; the dot-less form goes through `_DOMAIN_NODOT_RE` (>= 6 chars before `com`) with a `_SEG.words` guard. Verified: `Sirius`, `Nexus`, `Robin`, `Costco`, `Zetavio`, `telecom` keep their full core with `is_domain=0`; `sreetradecom`, `GSVISIONCOM`, `atlascbrecom`, `@riddhitraders`, `Amazon.com` are still detected.
- **Leet guard does not bound digits** (`preprocess.py:61`). Fixed: `_LEET_TOKEN_RE = ^(?=(?:[a-z]*\d){1,2}[a-z]*$)(?=(?:.*[a-z]){3})[a-z0-9]+$`; `abc12345`, `team2000` no longer match, `cardi0logy` does. `Team2000 Inc -> 'team2000'`, `Shop24x7 Retail -> 'shop24x7 retail'`. The second half of that fix (domain seeds) is the regression in A4.
- **`gc.freeze()` missing before the fork** (`features.py:634-640`) - present, with `gc.unfreeze()` in `finally` and `gc.disable()` in `_par_init`. Residual concern is A6.
- **Negative subsampling keys on `min_rank`** (`train.py:89, 277`) - it keys on `fused_rank` (within-S1 rank). Residual memory concern is A2.
- **Blocker rebuilt three times per country** - Pass B and C are merged; residual is the Pass A build (A5).

## D. Refuted claims (kept for the record)

Each of these was checked by the skeptics against the current tree and found to be either already implemented, immaterial, or describing code that does not exist:

- Legal-token removal "in any position" deletes identity tokens (`preprocess.py:294`) - behaviour real, impact immaterial; proposed fix measured as a regression on a comparable population.
- Single-token `CITY_ALIASES` rewrites ordinary address words (`preprocess.py:413`) - real but no measurable accuracy effect.
- `'59000 LILLE'` becomes a house number (`preprocess.py:70`) - already handled by `_CP_CITY_RE` for France (`postal='59000'`, verified); the generic proposed fix regresses US/India.
- `max_df` collapses to 1 for tiny country groups (`blocking.py:82`) - floored at 50 documents in the current code.
- Candidate cap / blocking params not persisted with the model (`config.py:84`, `inference.py`) - `train.py` writes them to `model_config.json`; inference default is the same `cfg.BLOCK_MAX_CANDIDATES`.
- CatBoost GPU decided by directory-existence check (`config.py:113`) - current code probes the GPU.
- `s1_name_dup` absolute-count skew (`features.py`) - already normalised in the working tree.
- Rank / top-k block features remain density-sensitive (`features.py:525`) - design choice, unquantified; not a defect.
- `legal_rel` breaks on catch-all legal tokens (`features.py:458`) - the first-legal-token `break` no longer exists.
- `size_adaptive` always drops the lowest link (`decision.py:138-157`) - it keeps `within == 0`; claim describes non-existent code.
- Expected-F config with floor 0 writes empty thresholds so US/India are treated as unseen (`train.py`, `inference.py:130`) - `seen_countries` is persisted explicitly in `model_config.json`.
- Expected-F selection validated on all candidates but inference filters `p_max >= 0.02` (`train.py:372-379`) - training applies the same filter before selection.
- France calibration targets the GT singleton share instead of the F0.5-optimal empty rate (`train.py`) - target is `max(0.0559, seen-country post-decide empty rate)`.
- Recall@K computed after the cap (`train.py:232`) - Pass A retrieves at `max(150, cap)`.
- Subsampled negatives not reweighted (`train.py:89`) - `w_sub` up-weights kept deep negatives by `1/NEG_RANDOM_FRAC`.
- France-robustness masking rewrites rows without extra tokens (`train.py:313-320`) - masked rows are restricted to `x_extra_cnt > 0`.
- `val_predictions` saves the mean ensemble instead of the selected probability (`train.py:591-604`) - saves `P_best`.
- Validator launched while inference holds all frames (`inference.py:313-314`) - frames are deleted before the subprocess.
- `KEEP_PROB=0.02` pre-filter changes the validated expected-F decision - same filter is applied in training (see above).
