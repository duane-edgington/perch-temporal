# CLAUDE_temporal.md

Project-memory for the **temporal layer** (repo: `perch-temporal`) on top of the
Perch 2 marine-mammal classifier. Companion files: `README.md`,
`consolidate_multispecies.py`, `perch2_temporal_colab.ipynb` (Colab pipeline),
`marine_temporal_pipeline.py` (standalone module).

---

## Goal

Reduce false detections and cross-species confusion between **humpback**,
**Bigg's orca**, and **Pacific white-sided dolphin (PWSD)** by modelling the
*pattern of vocal activity over time* instead of classifying isolated 5 s clips.
Phase 1 = label-free per-species stickiness HMM NOW; Phase 2 = learned BiLSTM/TCN
over frozen Perch 2 embeddings later. (orcAI's back half on a stronger front half.)

## Three-repo layout (settled)

- **perch-pytorch** — Perch 2 embedder reimplemented in PyTorch for the GB10 (the
  poster work). Emits embeddings + classifier posteriors.
- **perch-hoplite** — TensorFlow-free agile-modeling / search fork. Seeds labels.
- **perch-temporal** (this) — temporal smoothing + sequence modelling. Consumes the
  other two's file outputs; imports neither, so the embedding model can be swapped
  without touching ingestion/calibration/HMM. Live at
  github.com/duane-edgington/perch-temporal.

## Poster abstract facts (context; belong to perch-pytorch)

- Target HW: **NVIDIA GB10** (Grace Blackwell, sm_121, CUDA 13). Accelerated TF
  unavailable there; PyTorch runs natively -> motivation for the reimplementation.
  (The "Spark box", host `spark-ae0e`, is this GB10 machine; Mac is `icefish`.)
- Reimplemented Perch 2 embedder: **log-mel frontend + EfficientNet-B3** as a
  torch.nn.Module, weights from the TF graph. Cosine ~1.0 vs TF on the GB10.
- Reverse-engineered: frontend uses **log scaling (not PCEN)**; stem uses **VALID
  padding**. torch.compile ~2.5x faster than ONNX (~635 clips/s), fully on-device.
- perch-hoplite's TF calls removed -> embed->search->classify loop is 100% TF-free.
- **Per-window amplitude normalization essential** for parity (MARS audio is quiet;
  matches the RMS~1e-4 / quiet-audio note from the humpback spectrogram check).
- 1.56M MARS embeddings (2018, 2020, 2026); linear classifier **ROC-AUC 0.959**
  distinguishes Bigg's orca / PWSD / humpback / ship noise; **orca validated 2018**.

## Class taxonomy (RESOLVED)

- Four classes now: **Bigg's orca, PWSD, humpback, ship noise.**
- `dolphin_call` == **PWSD specifically** (the one currently identifiable + most
  common), NOT delphinids broadly.
- Planned expansion (needs experts + samples over time): per-vocalization-type
  classes — **echolocation clicks, burst pulses, whistles** — for each of the 4+
  Monterey delphinid species. => keep the pipeline **multi-label & extensible**
  (independent per-species/per-type chains; adding channels is additive).
- **v4 is the current pipeline**, but stay ready to swap the embedding model as
  agile modeling + more months of embeddings continue (ingestion is model-agnostic).

---

## Data reality (MARS / MBARI) and the SOURCE-TREE decision

- MBARI MARS cabled hydrophone, Monterey Bay. 10-min continuous WAV, 32 kHz,
  resampled to 24 kHz for the multispecies model, split into 120 x 5 s chunks.
  Filenames `MARS_YYYYMMDD_HHMMSS_..._chunk_NNN`. `epoch_seconds` = absolute UTC.
- Spark path `/mnt/PAM_Analysis/...` == Mac `/Volumes/PAM_Analysis/...`.
- The multispecies model writes **one JSON per 5 s chunk**. **Two output trees
  exist — this bit us, so it's the key thing to remember:**
  - **`scores_gpu/`  ✅ USE THIS.** Good format: keys `filename, scores,
    class_names, all_logits, all_probabilities`; **12 classes**; canonical order.
    Verified end-to-end on real data (Oo/Mn logits round-trip to the JSON exactly).
  - **`scores/`  ❌ DO NOT USE.** Older/thinner run: keys `filenames(!), scores,
    class_names`; **only 10 classes** (missing Ba, Be); **no all_logits/
    all_probabilities**. The consolidator's self-check correctly rejects these.
  - `resampled_24kHz_chunks/` holds the **.wav inputs**, not JSONs.
- Canonical class order (verified):
  `["Oo","Mn","Eg","Be","Upcall","Bp","Call","Gunshot","Echolocation","Bm","Whistle","Ba"]`
  The JSON `scores` array and per-chunk `class_names` are re-sorted per chunk ->
  never use them for labelling; only `all_logits`/`all_probabilities` are canonical.

## Consolidation (implemented + running)

`consolidate_multispecies.py`: per-chunk JSON -> one wide **logit** CSV per
recording (`epoch_seconds, <Class>_logit ...`), mirroring the input `YYYY/MM/` tree,
`_logits.csv` suffix (distinct from per-class `_epoch_*_scores.csv`), plus
`manifest.csv`. Reads `all_logits`; per-file self-check via the aligned
(class_names, scores) pair; resumable; skips non-matching JSONs.

- Verified: single-recording run on `scores_gpu` -> clean, no warnings, exact
  round-trip. **Full April 2018 run launched** under nohup:
  input `scores_gpu/2018/04`, output `.../logits/`. Scan found **518,400** JSONs
  (144 files/day x 30 x 120 = full month) -> expect ~4,320 recording CSVs.
- Output-root convention: write consolidated CSVs to a **separate `logits/` root**
  (not into `scores_gpu`) to keep inputs/outputs clean.

## Pipeline (`perch2_temporal_colab.ipynb`)

- **Runtime:** LIGHT. Heavy inference already done; Perch runs on the GB10, not the
  notebook. CPU high-RAM or T4 is plenty; A100 unnecessary.
- **Grid:** 5 s window / 5 s hop (matches precomputed chunks). Run Perch at 5 s hop.
- **Ingest:** `load_scores_csv()` reads the `logits/**/ *_logits.csv` tree
  recursively; `load_expanded_csv()` handles legacy expanded CSVs; both ->
  `{ts_key: (epoch_seconds, {ms_<Class>: logits})}`. Both self-check class order.
- **Perch handoff (GB10 -> Drive):** one `.npz` per recording, 5 s hop, with
  `embeddings [T,D]` + `perch_humpback/orca/pwsd/ship [T]` + `hop_sec`.
- **Then:** `build_frame_stacks` (align on epoch) -> `Calibrator` (Platt/isotonic
  on LOGITS; PWSD starved) -> per-species stickiness HMM (log-space Viterbi +
  forward-backward) -> intervals -> `stitch` (split on outages, decode per segment).
- **Payoff metric:** `ab_check(stacks, cal, "humpback")` -> `flipped_to_absent`.
- Class-order handling in the module is order-safe by construction (reads
  `model.metadata()`); the notebook hardcodes the verified `MS_CLASS_ORDER` + self-check.

## Monthly plots (earlier) — reminders

Humpback ~continuous, dolphin bursty, orca sparse: separability confirmed in raw
aggregates. BUT counts are **version-dominated, not biology** (April humpback:
v2=1,293 vs v4=24,701, ~19x) -> **calibration mandatory** before cross-month/version
comparison. v4 `other`->3 / `ship_noise`->4 collapse noted; keep `ship_noise` as a
vessel-masking covariate. Big single-day dolphin spikes = HMM test cases (preserve
coherent bouts, erase speckle). Monthly plots are the "before" picture.

---

## Repo / environment state (today)

- `perch-temporal` repo live on GitHub; `consolidate_multispecies.py` +
  `requirements.txt` committed and pushed (initial commit).
- venv at `~/perch-temporal/venv` (Python 3.12); deps satisfied
  (pandas 3.0.3, numpy 2.4.4, scipy 1.18, scikit-learn 1.9). `venv/` gitignored.
- `requirements.txt` = pandas/numpy/scipy/scikit-learn only. **No torch/TF here** —
  torch comes later for the sequence head, from the GB10/CUDA-13 build (NOT cu128).
- Suggest adding `*.log` to `.gitignore` (nohup logs land in the repo dir).

## TODO / next steps

- [x] April 2018 consolidation DONE: 4,303 `_logits.csv` on disk (log said "wrote
      4,208" because the first crashed run had already written ~95; resumable run
      skipped those -> 4,208 new + 95 existing = 4,303 on disk = authoritative).
      17 old-format recordings skipped -> `logits/skipped.csv`.
- [ ] Confirm `scores_gpu` covers 2020/2026 too (not mixed with old `/scores`).
- [ ] **Old-format stragglers:** April 2018 had 17/4,320 recordings in the old thin
      10-class format (no all_logits), logged to `logits/2018/04/skipped.csv` and
      skipped. Plan (another day): write a diagnostic scanning ALL months (2018/
      2020/2026) for old-format recordings + their file mtimes to see if clustered
      (one bad batch) or scattered (intermittent bug); then regenerate just those
      through the current 12-class model (in perch-pytorch; audio still under
      resampled_24kHz_chunks) and rerun consolidation (resumable) to fill gaps.
      Not blocking: 0.4%, spread out, `stitch` treats them as short gaps.
- [x] Perch export DESIGNED + written: `export_slim_npz.py` (belongs in perch-hoplite
      `tools/`). Reads embeddings back out of the Hoplite DB (no GPU), applies the
      linear classifier, writes slim `.npz` per recording. CPU, minutes.
  - Embeddings ALREADY in Hoplite DB (`.../db/MARS_20180401_20180430_32kHz_norm/`);
    do NOT re-embed on the GB10. 32 kHz WAVs live in `.../resampled_32kHz/YYYY/MM/`.
  - DB: SQLite `windows(id, recording_id, offsets=struct.pack('<dd',start,end))` +
    `recordings(id, filename)`; USearch key == windows.id;
    `SQLiteUSearchDB.load(db_path)` then `db.get_embedding(wid)` -> [1,1536].
  - Classifier `orca_v4.pt` (perch-hoplite): `beta [1536,5]`, `beta_bias [5]`,
    classes **5**: `dolphin_call, humpback_song, orca_call, other, ship_noise`
    (`other` = catch-all, kept as an HMM background/noise channel). Trained on
    `_norm` (peak-0.25) embeddings -> matches the DB. logits = emb@beta + bias.
  - TWO alignment bugs fixed vs the naive recipe: sort frames by UNPACKED start_s
    (blob byte-order != numeric for little-endian doubles); epoch is **UTC** to
    match the multispecies CSVs (naive `.timestamp()` would use local time).
    Verified: MARS_20180401_000914 -> epoch 1522541354.
  - Slim export omits embeddings (HMM doesn't need them); `--with-embeddings` adds
    the [T,1536] block for the future sequence head (heavy: ~3 GB/month, stays on
    thalassa, not to Drive).
  - [x] RAN for April 2018 -> **4,320 `.npz`** in `.../perch/2018/04/` (the DB has
        clean embeddings for ALL 4,320, incl. the 17 that were old-format on the
        multispecies side -> those 17 have Perch tracks but no ms logits; stitch
        handles the one-sided frames). Verified: T=120, hop=5.0, first epoch
        1522541354 == the April 1 logits CSV (UTC alignment holds end-to-end).
  - Two format surprises fixed on first contact (both now in the committed script):
    (1) `orca_v4.pt` is NOT a torch pickle -- it's JSON with base64 float32 arrays
        (LinearClassifier format): beta (embedding_dim,num_classes)=[1536,5],
        beta_bias [5]. Decode with numpy: np.frombuffer(b64decode(...),f32).
    (2) perch_hoplite.agile.classifier imports TensorFlow at module load -> can't
        use LinearClassifier.load in this TF-free venv; decode JSON+base64 directly.
        Only perch_hoplite import kept is db.sqlite_usearch_impl.
  - DB open is `SQLiteUSearchDB.create(db_path, readonly=True)` (not .load);
    `db.get_embeddings_batch(window_ids)` -> [T,1536] (one call/recording, fast).
  - Committed to perch-hoplite `tools/export_slim_npz.py` (commit 210213c).
- [ ] **Notebook integration (deferred):** reconcile Perch class names before the
      first HMM run. Export writes `perch_dolphin_call/humpback_song/orca_call/
      other/ship_noise`, but notebook `load_perch`/`emission_sources` still expect
      `perch_humpback/orca/pwsd` AND require an `embeddings` key the slim file lacks.
      Edits: (a) `load_perch` reads the 5 real channels, makes `embeddings`
      optional, uses the file's own `epoch_seconds`; (b) `emission_sources`: humpback
      <- perch_humpback_song + ms_Mn, orca <- perch_orca_call + ms_Oo, pwsd <-
      perch_dolphin_call, carry perch_other + perch_ship_noise as features;
      (c) reconcile `pwsd` vs `dolphin_call` naming in Config.
- [x] Colab data-transfer tooling built: `package_for_colab.sh` (tar.gz per input
      type from the mount + sha256 MANIFEST) and `colab_stage_inputs.py` (copy from
      Drive -> /content, checksum, untar, point PATHS local). Not yet folded into
      the notebook as a cell (deferred with the other notebook edits).
- [ ] Calibration `labels.csv` — start with orca-validated April 2018; on logits.
- [ ] First end-to-end HMM run on April 2018; watch `flipped_to_absent`.
- [ ] Regenerate April 2018 day-x-hour heatmap from smoothed output ("after" vs "before").
- [ ] Later: BiLSTM/TCN over `FrameStack.embeddings`; borrow orcAI masked BCE.
- [ ] Record the poster abstract in the **perch-pytorch** repo (not here).

## Regenerating multispecies scores (for the 17 old-format stragglers, someday)

The 12-class `scores_gpu` JSONs are produced by the multispecies model, run from a
SEPARATE repo (NOT perch-temporal):

- Repo: `github.com/duane-edgington/google-multispecies-whale-detection` (private),
  cloned on spark-ae0e at `~/gmwd/new3-12_whale_detection/gmwd/`.
- Runner: `run_model_gpu_optimized.py` (+ `.md` docs, `.sh` wrapper). GPU/TensorFlow.
  (Getting the TF dependency stack working in that env was hard-won — don't disturb it.)
- Model: Kaggle `google/multispecies-whale/TensorFlow2/default/2`.
- Input = the 24 kHz chunk WAVs (`resampled_24kHz_chunks/YYYY/MM/`); output = the
  per-chunk JSONs into `scores_gpu/YYYY/MM/`.
- Working example command (ran on spark-ae0e):
  ```
  nohup python3 run_model_gpu_optimized.py \
    --input_dir  /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_24kHz_chunks/2021/11/ \
    --output_dir /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2021/11/ \
    --model_url "https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/2" \
    --batch_size 8 > logs/nohup_run_model_gpu_optimized_256_2021_11.out &
  ```
- To fix the 17: rerun this on just their 24 kHz chunk dirs (from `skipped.csv`),
  then rerun `consolidate_multispecies.py` (resumable) to fill the gaps.

## Local Jupyter migration + first pipeline runs (DONE this session)

- Runs entirely on spark-ae0e now -- NO Colab/Drive. `perch2_temporal.ipynb`
  (dropped the `_colab` suffix; old colab notebook kept, not deleted). Committed to
  perch-temporal. Launch: `jupyter lab --no-browser --ip=127.0.0.1 --port=8888` on
  the box, then `ssh -N -L 8888:127.0.0.1:8888 duane@134.89.11.107` from the Mac.
- Notebook edits made: PATHS point at the /mnt logits/ and perch/ roots (loaders
  glob recursively); `load_perch` reads the 5 real channels
  (perch_dolphin_call/humpback_song/orca_call/other/ship_noise), embeddings
  OPTIONAL (slim files have none), uses the file's own epoch_seconds;
  `emission_sources` remapped (humpback<-perch_humpback_song+ms_Mn,
  orca<-perch_orca_call+ms_Oo, pwsd<-perch_dolphin_call).
- Pipeline RUNS end-to-end on April 2018: 4,303 stacks built (4,320 perch - 17
  ms-only stragglers, as expected). Load ~15 s when the kernel is healthy.

### Labels / calibration
- Analyst labels live in `/mnt/PAM_Analysis/perch-hoplite/provenance/labels/`
  (JSON: filename, offset_s, label, label_type). Two sessions so far:
  * July 9 (April 2018 DB): orca-query era -> orca_call / dolphin_call / other.
  * July 11 (April 2026 DB): per-class era -> humpback_song labels.
  Label evolution: early = orca/not-orca (so `other` = "not orca, class not
  recorded" -- NOT a humpback/pwsd negative); later = precise per class.
- `build_labels_csv.py` (in perch-temporal) converts analyst JSON -> the notebook's
  labels.csv (ts_key, frame_index, species, label). Maps orca_call->orca(+),
  dolphin_call->pwsd(+), humpback_song->humpback(+); `other`->negative for the
  session's query species only. frame_index = round(offset_s/5).
- April 2018 labels.csv written to `/mnt/PAM_Analysis/perch-hoplite/labels/`:
  **8 orca pos + 6 orca neg (calibrates), 9 pwsd pos + 0 neg (can't -> squash).**
- Calibrator hardened: skip any channel with <2 classes (falls back to sigmoid
  squash) instead of crashing. So orca calibrates (Platt, monotonic: logit 3.3->
  0.96), pwsd squashes.
- Month split of verified labels: **April 2018 = orca ground truth; April 2026 =
  humpback ground truth.** April 2026 has NO data yet (multispecies not run on
  2026 -> would need the full GPU run + consolidate + perch export first). So
  chose to validate ORCA on April 2018 now.

### HMM tuning findings (orca, verified rec 20180418T115912, frame 9)
- First run erased ALL verified orca (flipped_to_absent, smoothed_pos=0). Diagnosis
  found THREE compounding issues, needing two fixes (calibration was fine: Platt
  maps logit 3.3->0.96):
  1. **Duration prior was placeholder-wrong.** expected_present_sec orca was 180 s
     (36 frames!) but real present-runs are 1-4 frames (brief calls, not song).
     -> set orca expected_present_sec = 15 s.
  2. **Absent state too sticky to ENTER.** expected_absent_sec 1800 s -> a00=0.997,
     so the HMM never entered present even on confident frames. -> orca
     expected_absent_sec = 120 s. (This was the real blocker; present-prior alone
     didn't fix it.)
  3. **Channel combine diluted.** perch_orca_call and ms_Oo fire one frame APART
     (frame 8 vs 9); averaging -> ~0.5 non-committal. -> `combine_emission` default
     changed mean -> **max** (present if EITHER detector confident).
  Simulation on the verified window: only max-combine + absent=120 TOGETHER makes
  frame 9 survive. Both committed in the notebook.
- OPEN: absent=120 may be slightly loose (window sim flipped 6-14 all present);
  check on the full 120-frame decode (scratch cell added). Validate against the
  other 7 orca-positive recordings before calling the priors settled (don't overfit
  to one recording). humpback/pwsd priors still placeholders.

### Workflow lessons
- Pasting snippets into notebook cells corrupts the pipeline (clobbered a cell
  once). RULE: diagnostics go in a NEW scratch cell at the very bottom, never edited
  into pipeline cells; regenerate pipeline cells only by re-downloading the whole nb.
- A wedged kernel (interrupt did nothing) caused a 15-min "load hang"; standalone
  the reads are fast (200 npz in 0.6 s). Fix = restart kernel, single-step.
- To iterate on priors WITHOUT the ~15 s reload: after cell 6 builds `stacks`, edit
  only Config + calibration + the scratch check; reuse in-memory `stacks`. (Could
  add a stacks pickle-cache to the nb if reloads become painful.)

## References

- orcAI — Bonhoeffer et al. 2026, Marine Mammal Science e70083 (doi:10.1111/mms.70083);
  repo `ethz-tb/orcAI`.
- BiLSTM — Hochreiter & Schmidhuber 1997; Schuster & Paliwal 1997; Graves &
  Schmidhuber 2005 (framewise, Neural Networks 18(5-6):602-610).
- Humpback model — Allen et al. 2021, Front. Mar. Sci. 8:607321 (names temporal/
  song-unit modelling as future work).
- Perch 2 -> whale transfer — Google Research, NeurIPS 2025 workshop (Feb 2026);
  DCLDE, PIPAN, ReefSet; killer-whale subpopulation tasks.
- DCLDE 2026 NE Pacific — Sci. Data s41597-025-05281-5.
- Multispecies model = Google `multispecies-whale` (Kaggle, "Model2"); humpback =
  Google `humpback-whale`. Data: MBARI MARS; PIPAN 10.25921/Z787-9Y54.
