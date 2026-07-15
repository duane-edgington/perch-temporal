#!/usr/bin/env python3
"""build_labels_csv.py -- build the temporal pipeline's labels.csv from the
verified-annotation JSONs in /mnt/PAM_Analysis/perch-hoplite/json_labels/.

(Supersedes the earlier version that read the provenance/labels analyst JSON;
those clean per-species exports are the authoritative source now.)

INPUT: per-species-per-month JSON files, each a list of records:
  {species, recording_32khz, annotation_offset_s, frame_index,
   recording_start_utc_epoch, label (1|0), label_type, annotator, month}
  `species` is the CLASSIFIER class: orca_call | dolphin_call | humpback_song |
  ship_noise | other.  label=1 = verified IS that class; label=0 = verified NOT.

OUTPUT: labels.csv with columns  ts_key, frame_index, species, label
  `species` is the pipeline/HMM name: orca | humpback | pwsd.  label 1=pos 0=neg.

NEGATIVE POLICY (one-vs-rest -- this is what fixes PWSD's zero-negative break):
  For target species T, a frame is a NEGATIVE if it is a verified POSITIVE of a
  DIFFERENT named class (orca/dolphin/humpback/ship/other) -- i.e. verified to be
  something that isn't T -- plus any explicit label=0 whose class maps to T.
  This gives every species hundreds of real negatives without over-claiming: we
  only call a frame not-T when it's verified to be some other *named* thing.
  Co-occurrence caveat: assumes one dominant class per 5 s window (matches the
  annotation regime). Explicit label=0 records of a class OTHER than T (e.g. the
  54 'not-orca' frames, true class unknown) are, by default, used only as negatives
  for their own species; --neg-universal also applies them to all species (honors
  'non-orca frames are also non-dolphin', though one-vs-rest already suffices).

USAGE:
  python3 build_labels_csv.py OUT.csv [--json-dir DIR] [--month 2018_04] [--neg-universal]
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict

CLASS_TO_SPECIES = {"orca_call": "orca", "dolphin_call": "pwsd", "humpback_song": "humpback"}
TARGETS = ("orca", "humpback", "pwsd")
_TS = re.compile(r'(\d{8})_(\d{6})')


def ts_key(fn):
    m = _TS.search(os.path.basename(fn))
    if not m:
        raise ValueError(f"no timestamp in {fn!r}")
    return m.group(1) + "T" + m.group(2)


def load_records(json_dir, month):
    recs = []
    for fp in sorted(glob.glob(f"{json_dir}/*.json")):
        if os.path.basename(fp) in ("inventory.json",):
            continue
        data = json.load(open(fp))
        rows = data if isinstance(data, list) else data.get("annotations", data.get("labels", []))
        for r in rows:
            fn = r.get("recording_32khz") or r.get("filename")
            k = ts_key(fn)
            fi = int(r["frame_index"]) if "frame_index" in r else int(round(r["annotation_offset_s"] / 5.0))
            lab = int(r.get("label", 1))
            mon = str(r.get("month") or k[:6]).replace("_", "")
            if month and mon != month:
                continue
            recs.append((k, fi, r["species"], lab))
    return recs


def build(recs, neg_universal=False):
    pos, negexp = defaultdict(set), defaultdict(set)
    for k, fi, cls, lab in recs:
        (pos if lab == 1 else negexp)[cls].add((k, fi))

    out, stats = [], {}
    for tclass, tsp in CLASS_TO_SPECIES.items():
        P = pos.get(tclass, set())
        # negatives: positives of every OTHER class (one-vs-rest) + explicit not-T
        N = set()
        for c, frames in pos.items():
            if c != tclass:
                N |= frames
        N |= negexp.get(tclass, set())
        if neg_universal:
            for c, frames in negexp.items():
                if c != tclass:
                    N |= frames
        N -= P                                   # never both pos and neg for same species
        for (k, fi) in P:
            out.append((k, fi, tsp, 1))
        for (k, fi) in N:
            out.append((k, fi, tsp, 0))
        stats[tsp] = (len(P), len(N))
    return sorted(set(out)), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv")
    ap.add_argument("--json-dir", default="/mnt/PAM_Analysis/perch-hoplite/json_labels")
    ap.add_argument("--month", default=None, help="filter, e.g. 2018_04 or 201804")
    ap.add_argument("--neg-universal", action="store_true",
                    help="also use explicit not-X frames as negatives for all species")
    a = ap.parse_args()
    month = a.month.replace("_", "") if a.month else None

    recs = load_records(a.json_dir, month)
    if not recs:
        raise SystemExit(f"no records under {a.json_dir}" + (f" for month {a.month}" if month else ""))
    rows, stats = build(recs, a.neg_universal)

    os.makedirs(os.path.dirname(a.out_csv) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["ts_key", "frame_index", "species", "label"]); w.writerows(rows)
    print(f"wrote {len(rows)} label rows -> {a.out_csv}"
          + (f"  (month {a.month})" if month else ""))
    for sp in TARGETS:
        p, n = stats.get(sp, (0, 0))
        print(f"  {sp:9s}: {p:4d} pos / {n:4d} neg")


if __name__ == "__main__":
    main()
