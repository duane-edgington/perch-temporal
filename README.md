# perch-temporal

A temporal layer over per-clip marine-mammal detections. It takes the 5-second
scores produced by upstream models (Google multispecies-whale logits + Perch 2
embeddings/classifier) and models the **pattern of vocal activity over time**, so
that species which are confusable in any single 5-second window can be separated
by their temporal signature — and isolated false detections can be smoothed away.

## Why

A 5-second window is too short to tell humpback, Bigg's orca, and Pacific
white-sided dolphin apart reliably: the distinguishing information is in the
minute-to-hour structure (humpback song is long and repetitive; orca calls come
in stereotyped bouts; dolphins are bursty). This repo adds that missing temporal
context on top of the existing per-clip classifier.

Approach, in two phases:

1. **Now — label-free smoothing.** A per-species 2-state (absent/present) Hidden
   Markov Model over the per-frame scores. The self-transition probability *is* a
   duration prior (`expected_bout_frames = 1/(1 - p_self)`), set from biology, so
   it needs no training labels. Viterbi for the hard track, forward-backward for a
   smoothed presence probability.
2. **Later — learned.** A small BiLSTM/TCN head over the frozen Perch 2 embedding
   sequence, once enough labelled contiguous data has accumulated.

## How it fits with the other repos

This repo consumes the **outputs** of the other two; it does not import them.

| repo | role | this repo uses |
|---|---|---|
| [`perch-pytorch`](https://github.com/duane-edgington) | Perch 2 embedding model reimplemented in PyTorch (runs on the NVIDIA GB10) | per-recording `.npz`: embeddings + classifier posteriors |
| [`perch-hoplite`](https://github.com/duane-edgington) | TensorFlow-free agile-modeling / search | the labelled examples that seed the classifier |
| **`perch-temporal`** (here) | temporal smoothing + sequence modelling | multispecies logit CSVs + the `.npz` above |

Because ingestion only reads files, the embedding model can be swapped later
(agile modeling continues, more months of embeddings arrive) without touching the
consolidation, calibration, or HMM code.

## Data

Source is the **MBARI MARS cabled hydrophone** (Monterey Bay): 10-minute
continuous WAV, 32 kHz, resampled to 24 kHz for the multispecies model, split into
120 × 5 s chunks per recording, named `MARS_YYYYMMDD_HHMMSS_..._chunk_NNN`.

The Google multispecies model writes **one JSON per chunk**. Two output trees
exist in the archive — use the right one:

- **`scores_gpu/`** ✅ — the good format: `all_logits` + `all_probabilities`, 12
  classes, canonical order. **This is the source.**
- `scores/` ❌ — an older, thinner run: 10 classes, no logits. Do **not** use.

Class order in `all_logits`/`all_probabilities` is fixed and canonical
(`Oo, Mn, Eg, Be, Upcall, Bp, Call, Gunshot, Echolocation, Bm, Whistle, Ba`); the
JSON's `scores` array and per-chunk `class_names` are re-sorted per chunk and must
not be used for labelling. The consolidator verifies this per file.

## Install

CPU-only; no CUDA, no TensorFlow (torch is added later, for the sequence head,
from the GB10/CUDA-13 build — not the cu128 index).

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # pandas, numpy, scipy, scikit-learn
```

## Usage — consolidation

`consolidate_multispecies.py` collapses the ~120 tiny per-chunk JSONs of each
recording into one wide CSV of **logits** in canonical order
(`epoch_seconds, Oo_logit, Mn_logit, …`), mirroring the input `YYYY/MM/` tree, plus
a `manifest.csv`. This turns ~500k tiny files/month into a few thousand compact
CSVs and is the one-time reduction every downstream step reads.

```bash
# one recording (quick sanity check)
python3 consolidate_multispecies.py \
    /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2018/04/<recording> \
    /tmp/scores_test

# a whole month, logged and detached
nohup python3 -u consolidate_multispecies.py \
    /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores_gpu/2018/04 \
    /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/logits \
    > consolidate_2018_04.log 2>&1 &
```

It is resumable (recordings already written are skipped) and prints a `WARNING`
listing any chunk that fails the class-order self-check rather than mislabelling.

## Pipeline stages

1. **Consolidate** — per-chunk JSON → per-recording wide logit CSV. *(implemented:
   `consolidate_multispecies.py`)*
2. **Frame stack** — join multispecies logits + Perch embeddings/posteriors on the
   absolute-UTC 5 s grid.
3. **Calibrate** — Platt/isotonic on the logits (they are not probabilities).
4. **HMM** — per-species stickiness smoothing; Viterbi + forward-backward.
5. **Stitch** — decode across recording boundaries; split on outages.
6. *(later)* **BiLSTM/TCN** over the embedding sequence.

Stages 2–5 live in the Colab notebook (`perch2_temporal_colab.ipynb`); the HMM,
alignment, calibration, and stitching logic also exist as a standalone module
(`marine_temporal_pipeline.py`).

## Classes

Current four-class target: **Bigg's orca**, **Pacific white-sided dolphin (PWSD)**,
**humpback**, **ship noise**. Designed multi-label and extensible — the intended
expansion is per-species vocalization types (echolocation clicks, burst pulses,
whistles) across the 4+ delphinid species in Monterey Bay, pending expert-labelled
samples.

## Status

- ✅ Consolidation working and verified against real `scores_gpu` data.
- 🔜 Perch embedding/posterior export contract from `perch-pytorch`.
- 🔜 Calibration labels (starting with the orca-validated April 2018).
- 🔜 First end-to-end HMM run on April 2018.
