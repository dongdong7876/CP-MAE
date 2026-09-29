# Figure backend

Every data figure in the manuscript is produced by a script in this directory,
straight from the result files in `../results/`.  No figure is drawn by hand and
none is edited after generation, so `bash make_all.sh` reproduces the printed
figures byte for byte.

| Script | Output | Source data | Answers |
|---|---|---|---|
| `make_fig04_contamination.py` | `Fig4.pdf/.png` | `results/contamination_*.csv`, `results/synthetic_contamination/*.csv` | R2-C1, R2-C2 |
| `make_figG1_calibration.py` | `FigG1.pdf/.png` | `results/npz/*.reliability.csv` | R1-C46 |
| `make_figG2_uncertainty.py` | `FigG2.pdf/.png` | `results/npz/*.npz` | R1-C47 |
| `make_figF1_cost.py` | `FigF1.pdf/.png` | `results/cost.csv` | R1-C19, R1-C20, R1-C21, R1-C50 |

## Inputs

The summary CSVs the scripts read are versioned here, so three of the four
figures redraw from a clean clone with no extra download. `FigG2` is the
exception: it reads the raw per-window score arrays in `results/npz/*.npz`,
50 MB that is attached to the tagged release as `exp_results_npz.tgz` instead
of committed. Unpack it into `results/` to redraw that one figure.

`fig_style.py` holds the only copy of the palette, the type sizes and the text
width, so the four figures cannot drift apart visually.  The categorical palette
is Okabe--Ito, checked for colour-vision separation; every series also carries a
distinct marker and dash pattern, so identity survives greyscale printing.

## QA gates

`qa_gates.py` gives each script a set of assertions.  A script recomputes the
numbers it is about to draw and compares them with the values printed in the
manuscript and the response letter.  A mismatch aborts before anything is
written.  This is not decoration: the gates have already caught two real
defects.

1. `make_figG1_calibration.py` rejected the ECE figures quoted in the response to R1-C46.  The
   quoted pairs had been read off single seeds, and two of them came from
   different seeds of the same dataset.  The response now quotes seed-averaged
   values, which the gate reproduces.
2. `make_figF1_cost.py` rejected a naive merge of the two profiling passes in
   `cost.csv`.  Latency and throughput agree between passes to within 5%, but
   peak memory differs by up to 47%, so the memory panel uses one pass only.

## Figures 1 to 7

Figures 3, 5, 6 and 7 were typeset in plotting software rather than by a script
in this directory. Their underlying experiment outputs ship in `source_data/`,
one file per panel or per segment, with a README that maps each file to its
panel and states how to aggregate it. At every default setting the Figure 3
sweeps reproduce Tables 2 and 3 exactly.

Figures 1 and 2 are a conceptual sketch and an architecture diagram. They carry
no data and are drawn by hand.
