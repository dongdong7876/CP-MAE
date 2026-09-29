# Datasets

No dataset is stored in this git tree. Two of the five carry terms that do not
permit redistribution, and the other three are large enough to make every clone
of this repository expensive. Everything needed to rebuild them is here instead:
a preparation script, the sources, and a checksum for every produced file.

`prepare_datasets.py verify` is the point of this directory. It proves that a
rebuilt copy is byte-identical to the one the paper was run on, which a copy of
the data could not prove on its own.

## What to fetch, and from where

| Dataset | Series and labels | Training file | Licence |
|---|---|---|---|
| SMD | TSB-AD-M, or the Release assets of this repository | `SMD_train.npy`, from the mirror linked below | MIT, via OmniAnomaly |
| PSM | TSB-AD-M, or the Release assets of this repository | `train.csv`, from the mirror linked below | CC BY 4.0, via eBay RANSynCoders |
| LTDB | TSB-AD-M, or the Release assets of this repository | none; the last 5% of the series is the validation split | ODC-BY 1.0, via PhysioNet |
| SWaT | TSB-AD-M | `swat_train2.csv`, from the mirror linked below | iTrust; signed request required |
| WADI | **not in TSB-AD-M**; original iTrust release | included in that release | iTrust; signed request required |

- TSB-AD-M: `https://www.thedatum.org/datasets/TSB-AD-M.zip`, curated by
  Liu et al. Its Apache 2.0 licence covers the curation, and each dataset keeps
  the licence of its own source.
- Training files: the mirror folder linked from the top-level `README.md`.
  SMD and PSM are used as released. SWaT is a CSV, and the array the loader
  opens is derived from it and checked against the paper's values.
- iTrust: `https://www.sutd.edu.sg/itrust/`. SWaT and WADI are released only
  after a signed request, so they are neither stored nor mirrored here.

SMD, PSM and LTDB permit redistribution with attribution, so their prepared
folders are attached to the tagged release of this repository. Downloading them
from there skips the TSB-AD-M step. SWaT and WADI must be obtained from iTrust.

## Rebuild

Run it with no arguments. TSB-AD-M is downloaded into `_src/` beside the script
and reused afterwards:

```bash
python prepare_datasets.py build
```

Once iTrust grants the WADI request, point at the unpacked release and run it
again:

```bash
python prepare_datasets.py build --wadi ~/Downloads/WADI
```

Then compare the result with the recorded hashes. It exits non-zero on any
mismatch:

```bash
python prepare_datasets.py verify
```

The three original training files sit in a Google Drive folder, which plain
HTTP cannot fetch. Install `gdown` and the script pulls each one by its file
id, about 170 MB in all:

```bash
pip install gdown
```

The folder also holds ten other benchmarks and is roughly 600 MB, so fetching
it whole moves about 3.5 times more data than is needed. Downloading by hand
into `_src/train_files` works too; these three paths are what `build` looks
for:

```
_src/train_files/SMD/SMD_train.npy
_src/train_files/SWAT/swat_train2.csv
_src/train_files/PSM/train.csv
```

SMD and PSM ship their training partition ready to use. SWaT ships a CSV of
51 sensors plus a trailing normal/attack column, and the array the loader
opens is derived from it: the label column is dropped and the remaining
495,000 x 51 values are written as `SWaT_train.npy`. Before anything is
written the derived values are hashed at single precision and compared with
the array the paper was run on. A mismatch aborts the build. Placing a
ready-made `SWaT_train.npy` there instead also works, and it is checked the
same way.

Every produced file is written to a `.partial` name and moved into place only
once it is complete, so an interrupted run leaves the previous copy intact
rather than a truncated one.

`--tsb-ad-m` and `--train-src` accept copies you already have, and
`--no-download` forbids the network. Do not type square brackets anywhere; in a
usage synopsis they only mark an argument as optional.

### How each benchmark is assembled

TSB-AD-M ships one file per series, named `<seq>_<NAME>_id_<id>_...csv`. The
parts are not uniform, so the rule is not uniform either. `build` prints what it
did for each dataset and checks the result against the row and channel counts
below; it exits non-zero if any of them differs.

| Dataset | Parts used | Rule | Result |
|---|---|---|---|
| SMD | all 22 | concatenate in `id` order, renumber the rows from zero | 560,260 x 38 |
| SWaT | `id` 2 only | `id` 1 holds 66 unnamed channels and a different schema; the published series is `id` 2 with its 51 named sensors | 399,919 x 51 |
| PSM | the single part | keep its second half, with the source row numbers | 108,812 x 25, rows 108,812 to 217,623 |
| LTDB | all 5 | `id` 2 alone carries a third ECG lead, so only the leads present in every part are kept | 500,000 x 2 |

The `Label` column of each part becomes `<NAME>_label.csv`; the remaining columns
become the series. Nothing is resampled, reordered or rescaled.

## Layout the loaders expect

```
dataset/SMD/   SMD_test.csv   SMD_label.csv   SMD_train.npy
dataset/SWaT/  SWaT_test.csv  SWaT_label.csv  SWaT_train.npy
dataset/PSM/   PSM_test.csv   PSM_label.csv   train.csv
dataset/LTDB/  LTDB.csv       LTDB_label.csv
dataset/WADI/  train.csv      test.csv
dataset/_src/                  downloaded sources; not part of the repository
```

WADI keeps the original two-file iTrust layout, which `WADISegLoader` reads with
its own column handling. The other four follow the series/label/train layout
above.

## Sanity check after rebuilding

The first column of every produced CSV is the row index, which the loaders skip.
Channel counts and native training contamination should match Table 1:

| Dataset | Channels | Training contamination |
|---|---|---|
| SMD | 38 | 4.5% |
| SWaT | 51 | 18.5% |
| LTDB | 2 | 19.8% |
| WADI | 127 | 6.5% |
| PSM | 25 | 16.7% |

`LTDB.csv` must hold exactly three columns: the index and the two ECG channels.
Its labels live only in `LTDB_label.csv`. A four-column variant with the label
appended would be read as a third input channel and would leak the target.

## Attribution

Cite the original source of any dataset you use, as TSB-AD asks. SMD is from
Su et al., KDD 2019. PSM is from Abdulaal et al., KDD 2021. LTDB is the MIT-BIH
Long-Term ECG Database on PhysioNet. SWaT and WADI are from iTrust, Singapore
University of Technology and Design. The corrected label variants are from
Liu et al.
