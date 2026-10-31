# expiMap: reproducing and fixing a course-project modification

Course project (Bioinformatics, Amirkabir University, 2024-25) built on
[scArches / expiMap](https://github.com/theislab/scarches) (Lotfollahi et al., *Nat. Biotechnol.* 2022).
The original project changed expiMap in three ways and judged the result only by looking at UMAPs.
This repo re-implements those changes so they actually train, and measures them with numbers.

## What changed in the model (`src/scarches/`, copy of upstream v0.6.1)
| Student change | Status in original | Here |
|---|---|---|
| Extra `Linear+ReLU` before the encoder mean/var heads | always on; broke loading of pretrained models | optional flag `use_extra_layer`, saved with the model so query mapping rebuilds it |
| `correlation_penalty(z)` (squared off-diagonal covariance) | computed but **never optimised**: trainer still unpacked 3 return values | model returns the raw penalty, trainer adds `alpha_corr * penalty` with the annealed KL, logs `corr_loss` |
| New args `use_corr_penalty`, `alpha_corr` | defaults were ON, not stored in the model | default OFF, stored and restored by `load_query_data` |

"Transfer-learning update" and "decoder update" from the report had no code change (the first is upstream's `load_query_data`), so they are not reproduced.

## Experiment (`run_experiment.py`, one run = reference + query)
Tutorial pipeline unchanged: PBMC (4 studies, 32,484 cells) reference, Reactome pathways, 2000 HVGs, hidden `[256,256,256]`, NB loss, group lasso 0.7, up to 400 epochs with early stopping; then architecture surgery with the Kang query (13,576 cells, control vs IFN-stimulated).
Variants: `baseline` (upstream), `extra`, `corr` (student's penalty, weight 1e-3), `both` (= student's full modification). Seeds 0, 1, 2.

Metrics (mean ± std over seeds, `results/results_summary.csv`): iLISI and batch ASW (mixing), cell-type ASW / NMI / ARI (separation), kNN label-transfer accuracy and macro-F1 on Kang (cell types absent from the reference are excluded; fraction reported), and pathway checks: active terms kept after surgery, mean |correlation| between pathway latents, drift of reference latents after surgery, and AUROC of the INTERFERON_SIGNALING latent separating stimulated from control Kang cells.

## Run
```
make env            # local venv from requirements.lock (arm64 macOS, Python 3.12)
python download_data.py
make smoke          # 3-epoch CPU test, a few minutes
make run            # full experiment (slow on CPU; used colab_run.ipynb with a GPU)
```