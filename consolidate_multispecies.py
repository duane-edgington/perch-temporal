#!/usr/bin/env python3
"""
consolidate_multispecies.py  --  run this on the Spark box, where the JSONs live.

Walks a tree of per-5s-chunk Google-multispecies JSONs and writes, per recording,
ONE wide CSV of logits in the canonical class order:

    epoch_seconds, Oo_logit, Mn_logit, Eg_logit, Be_logit, Upcall_logit,
    Bp_logit, Call_logit, Gunshot_logit, Echolocation_logit, Bm_logit,
    Whistle_logit, Ba_logit

Output mirrors the input tree by YYYY/MM (from the recording timestamp), e.g.
    OUT_DIR/2018/04/MARS_20180413_071913_resampled_24kHz_logits.csv
so it can live alongside your per-class files (..._epoch_oo_scores.csv). The
distinct `_logits.csv` suffix keeps the pipeline loader from confusing the two.
A manifest.csv at OUT_DIR summarises coverage (used later to find recording gaps).

Why logits (not the JSON's probabilities): the sigmoid outputs pile against 0, so
calibration downstream is far better conditioned on the logits.
Why not the JSON's `scores` / `class_names` fields: both are re-sorted per chunk;
only `all_logits`/`all_probabilities` are in the fixed canonical order. We verify
that per file via the internally-aligned (class_names, scores) pair.

Usage:
    python consolidate_multispecies.py JSON_ROOT OUT_DIR [--hop 5.0] [--flat] [--force]

    # e.g.
    python consolidate_multispecies.py \
        /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/resampled_24kHz_chunks \
        /mnt/PAM_Analysis/GoogleMultiSpeciesWhaleModel2/scores

Resumable: recordings whose CSV already exists are skipped unless --force.
"""
import argparse, csv, glob, json, os, re, sys, datetime as dt

# Canonical multispecies class order = model-card index map, verified against the
# aligned (class_names, scores) pair. This is the order of all_logits.
MS_CLASS_ORDER = ["Oo", "Mn", "Eg", "Be", "Upcall", "Bp",
                  "Call", "Gunshot", "Echolocation", "Bm", "Whistle", "Ba"]
SUFFIX = "_logits.csv"

_TS = re.compile(r"(\d{8})[_T](\d{6})")
_CHUNK = re.compile(r"chunk_(\d+)", re.I)


def ts_key(name):
    m = _TS.search(name)
    if not m:
        raise ValueError(f"no YYYYMMDD[_/T]HHMMSS timestamp in {name!r}")
    return m.group(1) + "T" + m.group(2)


def ts_epoch(key):
    return dt.datetime.strptime(key, "%Y%m%dT%H%M%S").replace(
        tzinfo=dt.timezone.utc).timestamp()


def recording_id(o, fallback_path):
    """Clean recording id: parent dir of the JSON's internal `filename`
    (.../MARS_20180413_071913_resampled_24kHz/..._chunk_039.wav), else the chunk
    filename with the _chunk_NNN... suffix stripped."""
    fn = o.get("filename", "")
    if fn:
        parent = os.path.basename(os.path.dirname(fn))
        if parent:
            return parent
    return _CHUNK.split(os.path.basename(fallback_path))[0].rstrip("_")


def order_ok(o, tol=1e-9):
    """all_logits/all_probabilities in MS_CLASS_ORDER? Check via (class_names, scores).
    Returns False (never raises) for the old thin format that lacks these keys or
    has the wrong class list, so such files are skipped and reported, not fatal."""
    if not {"class_names", "scores", "all_probabilities"} <= o.keys():
        return False
    ap = o["all_probabilities"]
    if len(ap) != len(MS_CLASS_ORDER):
        return False
    truth = dict(zip(o["class_names"], o["scores"]))
    try:
        want = [truth[c] for c in MS_CLASS_ORDER]
    except KeyError:
        return False
    return all(abs(a - b) <= tol for a, b in zip(want, ap))


def out_path(out_dir, rid, key, nested):
    if nested:                                   # OUT_DIR/YYYY/MM/<rid>_logits.csv
        sub = os.path.join(out_dir, key[0:4], key[4:6])
    else:
        sub = out_dir
    os.makedirs(sub, exist_ok=True)
    return os.path.join(sub, f"{rid}{SUFFIX}")


def recording_id_from_path(fp):
    """Recording id from the path: parent dir of the chunk JSON
    (.../MARS_20180413_071913_resampled_24kHz/..._chunk_039_output.json), else the
    chunk filename with the _chunk_NNN... suffix stripped. No file read needed."""
    parent = os.path.basename(os.path.dirname(fp))
    if _TS.search(parent):
        return parent
    return _CHUNK.split(os.path.basename(fp))[0].rstrip("_")


def consolidate(json_root, out_dir, hop=5.0, nested=True, force=False):
    os.makedirs(out_dir, exist_ok=True)
    print(f"scanning {json_root} for chunk JSONs ...", file=sys.stderr, flush=True)
    files = glob.glob(f"{json_root}/**/*chunk_*.json", recursive=True)
    print(f"found {len(files):,} chunk JSONs", file=sys.stderr, flush=True)

    # Group by recording via PATH only -- no file reads (the slow part before).
    groups = {}
    for fp in files:
        groups.setdefault(recording_id_from_path(fp), []).append(fp)
    print(f"{len(groups):,} recordings; writing CSVs ...", file=sys.stderr, flush=True)

    manifest, bad, skipped_recs = [], [], []
    for i, (rid, paths) in enumerate(sorted(groups.items()), 1):
        try:
            key = ts_key(rid)
        except ValueError as e:
            print(f"  skip {rid}: {e}", file=sys.stderr); continue
        dest = out_path(out_dir, rid, key, nested)
        if os.path.exists(dest) and not force:
            continue
        t0 = ts_epoch(key)
        paths = sorted(paths, key=lambda p: int(_CHUNK.search(p).group(1)))
        rows, n_bad = [], 0
        for fp in paths:
            o = json.load(open(fp))
            if not order_ok(o):
                bad.append(fp); n_bad += 1; continue
            idx = int(_CHUNK.search(fp).group(1))            # chunk_001 -> offset 0
            rows.append([int(t0 + (idx - 1) * hop)] + list(o["all_logits"]))
        if not rows:
            skipped_recs.append((rid, len(paths)))           # whole recording old-format
            continue
        with open(dest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch_seconds"] + [f"{c}_logit" for c in MS_CLASS_ORDER])
            w.writerows(rows)
        manifest.append((rid, rows[0][0], len(rows), os.path.relpath(dest, out_dir)))
        if i % 200 == 0:
            print(f"  {i:,}/{len(groups):,} recordings written", file=sys.stderr, flush=True)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["recording_id", "start_epoch", "n_chunks", "path"])
        w.writerows(sorted(manifest, key=lambda r: r[1]))

    if skipped_recs:
        with open(os.path.join(out_dir, "skipped.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["recording_id", "n_chunks"])
            w.writerows(sorted(skipped_recs))

    print(f"\nwrote {len(manifest):,} recording CSVs + manifest.csv under {out_dir}")
    if skipped_recs:
        print(f"SKIPPED {len(skipped_recs)} recording(s) in the old format (no "
              f"all_logits/12 classes) -> listed in skipped.csv:", file=sys.stderr)
        for rid, n in skipped_recs[:5]:
            print(f"    {rid}  ({n} chunks)", file=sys.stderr)
    if bad:
        print(f"WARNING: {len(bad)} chunk(s) FAILED the class-order self-check and "
              f"were skipped; the class list may differ. First few:", file=sys.stderr)
        for fp in bad[:5]:
            print(f"    {fp}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json_root")
    ap.add_argument("out_dir")
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--flat", action="store_true", help="don't mirror YYYY/MM tree")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    consolidate(a.json_root, a.out_dir, a.hop, nested=not a.flat, force=a.force)
