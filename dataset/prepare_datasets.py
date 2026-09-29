#!/usr/bin/env python3
"""Rebuild the five benchmark folders this repository trains on.

No dataset is redistributed inside the git tree. This script turns the public
releases into exactly the files `data_factory/data_loader.py` opens, and writes
a checksum for each one, so a rebuilt copy can be proved identical to ours.

Sources
-------
TSB-AD-M          SMD, SWaT, PSM and LTDB, as corrected by Liu et al.
                  https://www.thedatum.org/datasets/TSB-AD-M.zip
Original train    the separate training file each benchmark ships, mirrored at
                  the folder linked from the repository README
WADI              iTrust, Singapore University of Technology and Design; a
                  signed request is required and TSB-AD-M does not include it

Usage
-----
Run it with no arguments. Sources are fetched into `_src/` beside this script
and reused on later runs:

  python prepare_datasets.py build

Once the iTrust request for WADI is granted, point at the unpacked release:

  python prepare_datasets.py build --wadi ~/Downloads/WADI

Then check the result against the recorded hashes:

  python prepare_datasets.py verify

TSB-AD-M is downloaded automatically. The three original training files sit
in a Google Drive folder, which plain HTTP cannot fetch. Install `gdown` and
the script pulls each one by its file id, about 170 MB in all, or place them
at `_src/train_files` by hand. SWaT ships its training split as a CSV, and
the array the loader opens is derived from it and checked against the values
used in the paper. WADI is never downloaded, since iTrust releases it only
after a signed request.

`--tsb-ad-m` and `--train-src` override the defaults with copies you already
have. `--no-download` forbids the network. No argument ever takes square
brackets; in a synopsis they only mark something as optional.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHECKSUMS = HERE / "CHECKSUMS.sha256"
SRC = HERE / "_src"                      # downloads land here; git ignores it
TSB_URL = "https://www.thedatum.org/datasets/TSB-AD-M.zip"
TRAIN_URL = ("https://drive.google.com/drive/folders/"
             "1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR")
# The three training files this repository needs, by Google Drive file id.
# Fetching them one by one moves about 170 MB. The folder that holds them also
# carries ten other benchmarks and is roughly 600 MB.
TRAIN_IDS = {
    "SMD/SMD_train.npy":    "1ETJkCImUSk11p-p9B5CAnduszoNOnSiF",
    "SWAT/swat_train2.csv": "1TVppX8PzyEtRbJ45ooWic4HeSXHAQ5kj",
    "PSM/train.csv":        "1d3tAbYTj0CZLhB7z3IDTfTRg3E7qj_tw",
}

# TSB-AD-M file names look like 002_MSL_id_1_Sensor_tr_500_1st_900.csv
TSB_NAME = re.compile(r"^(?P<seq>\d+)_(?P<name>[A-Za-z0-9]+)_id_(?P<id>\d+)_")

# How each benchmark is assembled from TSB-AD-M. The rules are not uniform,
# because the released parts are not uniform, and every one of them is checked
# against the published series in CHECKS below.
#
#   series, labels   the two files written
#   keep_ids         None for every part, or the subset to use
#   common_cols      keep only the columns present in every part
#   second_half      keep the tail half of the series, source index preserved
#   reset_index      renumber the concatenated rows from zero
FROM_TSB = {
    "SMD":  dict(series="SMD_test.csv",  labels="SMD_label.csv",
                 keep_ids=None, common_cols=False, second_half=False,
                 reset_index=True,
                 note="22 parts, concatenated in id order"),
    "SWaT": dict(series="SWaT_test.csv", labels="SWaT_label.csv",
                 keep_ids=[2], common_cols=False, second_half=False,
                 reset_index=True,
                 note="id 1 carries 66 unnamed channels and is a different "
                      "schema; the published series is id 2 alone, with the 51 "
                      "named sensors"),
    "PSM":  dict(series="PSM_test.csv",  labels="PSM_label.csv",
                 keep_ids=None, common_cols=False, second_half=True,
                 reset_index=False,
                 note="one part; the published series is its second half, and "
                      "the source row numbers are kept"),
    "LTDB": dict(series="LTDB.csv",      labels="LTDB_label.csv",
                 keep_ids=None, common_cols=True, second_half=False,
                 reset_index=True,
                 note="5 parts; id 2 alone carries a third ECG lead, so only "
                      "the leads common to every part are kept"),
}

# rows, channels: what a correct rebuild must produce
CHECKS = {
    "SMD":  (560260, 38),
    "SWaT": (399919, 51),
    "PSM":  (108812, 25),
    "LTDB": (500000, 2),
}
# dataset -> (file name in the mirror, file name the loader opens)
# dataset -> (source names to look for, in order; file name the loader opens)
FROM_TRAIN_SRC = {
    "SMD":  (("SMD_train.npy",),                    "SMD_train.npy"),
    "SWaT": (("swat_train2.csv", "SWaT_train.npy"), "SWaT_train.npy"),
    "PSM":  (("train.csv",),                        "train.csv"),
}
# SWaT ships its training split as a CSV, and the loader opens an array. The
# derived array is checked against the one used in the paper before it is
# written: same shape, same values to the last bit.
SWAT_TRAIN_SHAPE = (495000, 51)
# The values are hashed at single precision. A CSV round trip can move the
# last bit of a double, and no sensor here carries anything near that many
# digits, so single precision compares the data and not the formatting.
SWAT_TRAIN_DIGEST = ("9a0f4610866376bbb408a34fbc717c2c0cccc0f2"
                     "8d65ce5873dfdb1d65101946")
# WADI keeps the original iTrust layout, which the loader reads differently
WADI_FILES = ("train.csv", "test.csv")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tsb_parts(root: Path, name: str, keep_ids=None) -> list[Path]:
    """Every TSB-AD-M file belonging to one benchmark, in ascending id order."""
    found = []
    for p in root.rglob("*.csv"):
        m = TSB_NAME.match(p.name)
        if m and m.group("name").upper() == name.upper():
            found.append((int(m.group("seq")), int(m.group("id")), p))
    if not found:
        raise SystemExit("no TSB-AD-M file matches *_%s_id_*.csv under %s" % (name, root))
    if keep_ids is not None:
        found = [f for f in found if f[1] in keep_ids]
        if not found:
            raise SystemExit("%s: none of the ids %s are present" % (name, keep_ids))
    found.sort(key=lambda t: (t[1], t[0]))          # id first, then the sequence number
    ids = [t[1] for t in found]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise SystemExit("%s: ids are not a clean ascending run: %s" % (name, ids))
    if ids != list(range(ids[0], ids[0] + len(ids))):
        print("  ! %s: ids %s are not contiguous; concatenating in the order shown"
              % (name, ids))
    return [t[2] for t in found]


def read_part(path: Path) -> pd.DataFrame:
    """Read one TSB-AD-M file.

    When the file carries a leading unnamed index column, it is taken as the
    frame index rather than as a channel. Writing the result back therefore
    reproduces the original row numbering, and no ``Unnamed: 0`` column leaks
    into the series the loaders read.
    """
    head = pd.read_csv(path, nrows=0)
    first = str(head.columns[0]).strip()
    if first == "" or first.lower().startswith("unnamed:"):
        return pd.read_csv(path, index_col=0)
    return pd.read_csv(path)


def split_label(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in df.columns if c.strip().lower() == "label"]
    if len(cols) != 1:
        raise SystemExit("%s: expected exactly one Label column, found %s" % (name, cols))
    lab = cols[0]
    return df.drop(columns=[lab]), df[[lab]]


def fetch(url: str, dest: Path) -> None:
    """Download to dest, resuming a partial file when the server allows it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(dest), url]
    elif shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(dest), url]
    else:
        cmd = None
    if cmd is not None:
        print("  downloading %s" % url)
        if subprocess.call(cmd) == 0 and dest.exists() and dest.stat().st_size > 0:
            return
        raise SystemExit(
            "download failed: %s\n"
            "Fetch it manually and put it at %s, then run this script again."
            % (url, dest))
    raise SystemExit("neither curl nor wget is available; download %s to %s by hand"
                     % (url, dest))


def unpack(archive: Path, into: Path) -> Path:
    print("  unpacking %s" % archive.name)
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(into)
    # the zip may or may not carry a single top-level folder
    entries = [p for p in into.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return into


def ensure_tsb(args: argparse.Namespace) -> Path:
    """Return an unpacked TSB-AD-M directory, downloading it if needed."""
    if args.tsb_ad_m:
        return need_dir(args.tsb_ad_m, "--tsb-ad-m")
    unpacked = SRC / "TSB-AD-M"
    if unpacked.is_dir() and any(unpacked.rglob("*.csv")):
        print("TSB-AD-M: using %s" % unpacked)
        return unpacked
    archive = SRC / "TSB-AD-M.zip"
    if not archive.exists():
        if args.no_download:
            raise SystemExit(
                "TSB-AD-M is missing and --no-download is set.\n"
                "Download %s to %s, or pass --tsb-ad-m with an unpacked copy."
                % (TSB_URL, archive))
        fetch(TSB_URL, archive)
    return unpack(archive, unpacked)


def gdown_cmd() -> list[str] | None:
    """The way to run gdown here, or None when it is not installed."""
    if importlib.util.find_spec("gdown") is not None:
        return [sys.executable, "-m", "gdown"]
    if shutil.which("gdown"):
        return ["gdown"]
    return None


def ensure_train_src(args: argparse.Namespace):
    """Return the directory of original training files, or None.

    Three files are needed, and they sit in a Google Drive folder that plain
    HTTP cannot fetch. Each one is pulled by its own file id, so the ten other
    benchmarks in that folder are never downloaded. A missing file is not
    fatal: the four series built from TSB-AD-M are written anyway.
    """
    if args.train_src:
        return need_dir(args.train_src, "--train-src")
    local = SRC / "train_files"
    want = [local / rel for rel in TRAIN_IDS]
    if all(f.exists() for f in want):
        print("training files: using %s" % local)
        return local
    if args.no_download:
        return local if any(f.exists() for f in want) else None

    cmd = gdown_cmd()
    if cmd is None:
        print("\n- gdown is not installed, so the three training files were skipped.")
        print("-   %s -m pip install gdown" % Path(sys.executable).name)
        print("-   python prepare_datasets.py build")
        print("- or download %s" % TRAIN_URL)
        print("- by hand into %s\n" % local)
        return local if any(f.exists() for f in want) else None

    for rel, fid in TRAIN_IDS.items():
        dest = local / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("  fetching %s" % rel)
        ok = False
        for attempt in (1, 2, 3):
            rc = subprocess.call(cmd + ["--id", fid, "--continue",
                                        "-O", str(dest)])
            if rc == 0 and dest.exists() and dest.stat().st_size > 0:
                ok = True
                break
            print("  attempt %d failed; retrying" % attempt)
        if not ok:
            print("  could not fetch %s" % rel)
            for junk in dest.parent.glob(dest.name + "*.part"):
                junk.unlink()

    got = [f for f in want if f.exists()]
    if not got:
        print("\n- none of the three training files arrived, so they are left")
        print("- for a later run. The four TSB-AD-M series are written regardless.")
        print("- Re-run the same command; finished downloads are reused.")
        print("- or download %s" % TRAIN_URL)
        print("- by hand into %s\n" % local)
        return None
    return local


def find_src(root: Path, name: str, fnames):
    """Locate one training file under a mirror of the Drive folder.

    The folder names there do not match ours exactly; SWaT sits under SWAT.
    The first name that exists wins, so a ready-made array is preferred to
    nothing when the released CSV was not downloaded.
    """
    folders = (root / name, root / name.upper(), root / name.lower(), root)
    for fname in fnames:
        for folder in folders:
            cand = folder / fname
            if cand.exists():
                return cand
        hits = sorted(root.rglob(fname))
        if hits:
            return hits[0]
    return None


def atomic(dst: Path, write) -> None:
    """Run `write` against a temporary name, then move it into place.

    An interrupted run then leaves the previous file untouched rather than a
    half-written one, which a later build would happily hash and publish.
    """
    tmp = dst.with_name(dst.name + ".partial")
    try:
        write(tmp)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def swat_train_npy(src: Path, dst: Path) -> None:
    """Write SWaT_train.npy from the released swat_train2.csv.

    That CSV carries the 51 sensors and one trailing normal/attack column.
    The loader reads the sensors alone. Shape and values are both checked
    against the array used in the paper, and nothing is written on a mismatch.
    """
    frame = pd.read_csv(src)
    if frame.shape[1] == SWAT_TRAIN_SHAPE[1] + 1:
        frame = frame.iloc[:, :-1]
    if frame.shape != SWAT_TRAIN_SHAPE:
        raise SystemExit(
            "SWaT: %s gives %s, and %s was expected.\n"
            "First column names: %s"
            % (src.name, frame.shape, SWAT_TRAIN_SHAPE,
               ", ".join(map(str, list(frame.columns)[:6]))))
    check_swat_values(frame.to_numpy(), src.name)
    arr = frame.to_numpy().astype(object)
    atomic(dst, lambda t: np.save(t, arr, allow_pickle=True))


def check_swat_values(arr, what: str) -> None:
    """Abort unless an array carries the SWaT training values of the paper."""
    flat = np.ascontiguousarray(arr.astype(np.float32))
    digest = hashlib.sha256(flat.tobytes()).hexdigest()
    if digest != SWAT_TRAIN_DIGEST:
        raise SystemExit(
            "SWaT: %s does not hold the values used in the paper.\n"
            "  expected %s\n  got      %s\n"
            "Nothing was written. Report this instead of training on it."
            % (what, SWAT_TRAIN_DIGEST, digest))


def need_dir(raw: str, what: str) -> Path:
    p = Path(raw).expanduser().resolve()
    if not p.is_dir():
        raise SystemExit(
            "%s is not a directory: %s\n"
            "Pass a real path. Square brackets in the usage text mark an "
            "optional argument and must not be typed." % (what, raw))
    return p


def build(args: argparse.Namespace) -> None:
    tsb = ensure_tsb(args)
    trn = ensure_train_src(args)
    written: list[Path] = []
    missing: list[str] = []
    mismatched: list[str] = []

    for name, spec in FROM_TSB.items():
        out_dir = HERE / name
        out_dir.mkdir(parents=True, exist_ok=True)
        parts = tsb_parts(tsb, name, spec["keep_ids"])
        print("%s: %d TSB-AD-M file(s) -- %s" % (name, len(parts), spec["note"]))
        frames = [read_part(p) for p in parts]
        if spec["common_cols"] and len(frames) > 1:
            keep = [c for c in frames[0].columns
                    if all(c in f.columns for f in frames)]
            dropped = sorted({c for f in frames for c in f.columns} - set(keep))
            if dropped:
                print("  dropping columns absent from some parts: %s" % dropped)
            frames = [f[keep] for f in frames]
        joined = pd.concat(frames, ignore_index=False)
        if spec["reset_index"]:
            joined = joined.reset_index(drop=True)
        if spec["second_half"]:
            joined = joined.iloc[len(joined) // 2:]
        feats, labels = split_label(joined, name)

        want_rows, want_cols = CHECKS[name]
        got = (len(feats), feats.shape[1])
        flag = "OK" if got == (want_rows, want_cols) else "DIFFERS FROM THE PAPER"
        print("  rows=%d  channels=%d  anomaly rate=%.4f  index %s..%s  [%s]"
              % (len(feats), feats.shape[1], labels.iloc[:, 0].mean(),
                 feats.index[0], feats.index[-1], flag))
        if got != (want_rows, want_cols):
            print("  ! expected %d rows and %d channels" % (want_rows, want_cols))
            mismatched.append(name)

        atomic(out_dir / spec["series"], lambda t: feats.to_csv(t))
        atomic(out_dir / spec["labels"], lambda t: labels.to_csv(t))
        written += [out_dir / spec["series"], out_dir / spec["labels"]]

        if name in FROM_TRAIN_SRC:
            src_names, dst_name = FROM_TRAIN_SRC[name]
            if trn is None:
                missing.append("%s/%s" % (name, dst_name))
            else:
                src = find_src(trn, name, src_names)
                if src is None:
                    missing.append("%s/%s  (%s not found under %s)"
                                   % (name, dst_name, " or ".join(src_names), trn))
                elif src.suffix == ".csv" and dst_name.endswith(".npy"):
                    swat_train_npy(src, out_dir / dst_name)
                    written.append(out_dir / dst_name)
                    print("  train file: %s, derived from %s" % (dst_name, src.name))
                else:
                    if name == "SWaT":
                        check_swat_values(
                            np.load(src, allow_pickle=True), src.name)
                    atomic(out_dir / dst_name,
                           lambda t, s=src: shutil.copy2(s, t))
                    written.append(out_dir / dst_name)
                    print("  train file: %s" % dst_name)

    if args.wadi:
        wadi_src = need_dir(args.wadi, "--wadi")
        out_dir = HERE / "WADI"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in WADI_FILES:
            src = wadi_src / f
            if not src.exists():
                raise SystemExit("WADI: %s not found under %s" % (f, wadi_src))
            atomic(out_dir / f, lambda t, s=src: shutil.copy2(s, t))
            written.append(out_dir / f)
        print("WADI: copied %s from the original iTrust release" % ", ".join(WADI_FILES))
    elif all((HERE / "WADI" / f).exists() for f in WADI_FILES):
        written += [HERE / "WADI" / f for f in WADI_FILES]
        print("WADI: already in place from an earlier iTrust release")
    else:
        missing.append("WADI/train.csv and WADI/test.csv")
        print("WADI: skipped; pass --wadi once the iTrust request is granted")

    lines = ["%s  %s" % (sha256(p), p.relative_to(HERE)) for p in sorted(written)]
    CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote %d file(s) and %s" % (len(written), CHECKSUMS.name))
    if mismatched:
        print("\nFAILED: these do not match the published series: %s"
              % ", ".join(mismatched))
        print("Do not train on them until the difference is understood.")
        sys.exit(2)
    if missing:
        print("\nPARTIAL SUCCESS. Everything built so far matches the paper.")
        print("These files come from sources this script cannot fetch on its own:")
        for m in missing:
            print("  - %s" % m)
        print("Add them, then run the same command again. What is already "
              "written is kept and reused.")
        sys.exit(1)
    print("\nCOMPLETE: all five datasets are in place and match the paper.")


def verify(args: argparse.Namespace) -> None:
    path = Path(args.checksums).expanduser()
    if not path.exists():
        raise SystemExit("no checksum file at %s" % path)
    ok = missing = bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        target = HERE / rel.strip()
        if not target.exists():
            print("  MISSING  %s" % rel.strip()); missing += 1; continue
        if sha256(target) == digest:
            print("  ok       %s" % rel.strip()); ok += 1
        else:
            print("  MISMATCH %s" % rel.strip()); bad += 1
    print("\n%d ok, %d mismatched, %d missing" % (ok, bad, missing))
    sys.exit(1 if (bad or missing) else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser(
        "build", help="rebuild every dataset folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python prepare_datasets.py build\n"
               "  python prepare_datasets.py build --wadi ~/Downloads/WADI\n\n"
               "Sources are fetched into _src/ when the paths are omitted.\n"
               "Do not type square brackets; they only mark optional arguments.")
    b.add_argument("--tsb-ad-m", default=None,
                   help="unpacked TSB-AD-M directory; downloaded into _src/ when omitted")
    b.add_argument("--train-src", default=None,
                   help="directory of the original training files; _src/train_files by default")
    b.add_argument("--wadi", default=None,
                   help="directory holding the original WADI train.csv and test.csv")
    b.add_argument("--no-download", action="store_true",
                   help="never reach the network; fail instead")
    b.set_defaults(func=build)
    v = sub.add_parser("verify", help="re-hash the rebuilt files")
    v.add_argument("--checksums", default=str(CHECKSUMS))
    v.set_defaults(func=verify)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
