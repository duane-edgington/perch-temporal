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

- [ ] Finish + verify the April 2018 consolidation run (count ~4,320, zero WARNINGs).
      Confirm `scores_gpu` covers 2018/2020/2026 (not mixed with old `/scores`).
- [ ] Perch export contract from perch-pytorch/GB10: `.npz` per recording, 5 s hop,
      `embeddings` + 4 posteriors (`perch_orca/pwsd/humpback/ship`) + `hop_sec`.
      (Add a `ship` channel to the notebook `Config`.)
- [ ] Calibration `labels.csv` — start with orca-validated April 2018; on logits.
- [ ] First end-to-end HMM run on April 2018; watch `flipped_to_absent`.
- [ ] Regenerate April 2018 day-x-hour heatmap from smoothed output ("after" vs "before").
- [ ] Later: BiLSTM/TCN over `FrameStack.embeddings`; borrow orcAI masked BCE.
- [ ] Record the poster abstract in the **perch-pytorch** repo (not here).

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
