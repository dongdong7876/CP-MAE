# Source data for Figures 3, 5, 6 and 7

Figures 4, F.1, G.1 and G.2 are produced by the scripts in the parent directory.
Figures 3, 5, 6 and 7 were typeset in plotting software from the experiment
outputs collected here. This directory is what makes them checkable. Every
plotted value is present, together with the run that produced it.

Figures 1 and 2 are a conceptual sketch and an architecture diagram. They carry
no data and are drawn by hand.

## Figure 3 — hyperparameter sensitivity

`fig3_sensitivity/` holds one file per swept hyperparameter, exactly as the
evaluation loop wrote them. Each row is one completed run, and each file keeps
all eighteen metric columns rather than only the one plotted.

| File | Panel | Swept column | Values |
|---|---|---|---|
| `results_win_size.csv` | (a) | `win_size` | 128, 240, 320, 400, 480 |
| `results_K.csv` | (b) | `K` | 1, 4, 8, 16, 32 |
| `results_num_patch.csv` | (c) | `num_patch` | [4], [8], [16], [4,8], [4,16], [4,8,16] |
| `results_d_model.csv` | (d) | `d_model` | 16, 32, 64, 128, 256, 512 |
| `results_mt.csv` | (e) | `mt` | 0.10, 0.30, 0.50, 0.75, 0.90 |
| `results_mf.csv` | (f) | `mf` | 0.10, 0.30, 0.50, 0.70, 0.90 |
| `results_alpha.csv` | (g) | `alpha` | 0.0, 0.5, 1.0, 5.0, 10.0 |
| `results_beta.csv` | (h) | `beta` | 0.0, 0.5, 1.0, 5.0, 10.0 |
| `results_gamma.csv` | (i) | `gamma` | 0.0, 0.5, 1.0, 2.0, 5.0, 10.0 |

`results_CP-MAE.csv` holds the joint scale and variability-weight grid used to
fix the default configuration.

**How to reproduce a panel.** Group by `data_name` and the swept column, then
take the mean and the sample standard deviation of `VUS_PR`. Multiply by 100 for
the percentages printed in the manuscript. Every other setting stays at the
default of Table D.1.

```python
import pandas as pd
d = pd.read_csv("fig3_sensitivity/results_K.csv")
print(d.groupby(["data_name", "K"]).VUS_PR.agg(["mean", "std", "count"]) * 100)
```

At each sweep's default setting this reproduces Tables 2 and 3 exactly: SMD
23.4, SWaT 18.6, LTDB 33.4, WADI 36.5 and PSM 52.0.

**Run counts.** Every setting holds five runs, and every reported number
averages them.

**Zero-valued endpoints.** `alpha = 0`, `beta = 0` and `gamma = 0` are the
ablation endpoints reported as *w/o Time*, *w/o Freq* and *Mean-Only* in Table 4.
They are part of the sweep rather than separate experiments.

## Figure 5 — macroscopic case study

`fig5_case_study/fig5_smd_dim15_segment.csv` covers SMD time steps 64,000 to
78,999 on dimension 15. Columns: `original_signal`, the intermediate metrics
`mu_t`, `mu_f`, `sigma_t`, `sigma_f`, the normalised baseline scores
`score_MTGFlow` and `score_MSHTrans`, the fused `score_CP-MAE`, and the
ground-truth `label`. Column names were rewritten from the plotting software's
escape syntax; the values are untouched.

## Figure 6 — microscopic view

`fig6_microscopic/` covers SMD time steps 66,500 to 71,499.

- `fig6a_time_domain_mc_trajectories.csv` — the original channel alongside the
  sixteen Monte Carlo reconstructions, `MC_Recon_K0` to `MC_Recon_K15`.
- `fig6b_frequency_variance_spectrogram.csv` — the per-bin mask-induced variance
  of the frequency branch, one row per STFT frame.

`Original_Dim_32` is the one-based name of the channel the manuscript calls
dimension 31.

## Figure 7 — reconstruction against masking ratio

`fig7_masking_ratio/` covers SMD time steps 61,500 to 71,499 on channel 31.
Columns give the ground truth `GT`, then `Recon_<r>` and `Error_<r>` for each
masking ratio r in 10, 30, 50, 75 and 90 percent.

`fig7_masking_ratio_smd_dim31.csv` is the export behind the published figure.
`make_fig7_data.ipynb` is the notebook that produced these arrays. Its outputs
are cleared, so it carries no absolute paths from the machine it ran on.
