# expiMap: course project on scArches

Course project (Bioinformatics, Amirkabir University, 2024-25) built on
[scArches / expiMap](https://github.com/theislab/scarches) (Lotfollahi et al., *Nat. Biotechnol.* 2022).
The original project modified expiMap and judged the result by looking at UMAPs.
This repository holds the cleaned project files, together with scripts that measure the modifications with numbers.

## Changes to the model
The code lives in `src/scarches/`, a copy of scArches v0.6.1. Two changes were made to expiMap.

- **Extra encoder layer.** An additional `Linear + ReLU` layer is added to the encoder just before the mean and variance layers. It is controlled by the option `use_extra_layer`.
- **Correlation penalty.** A penalty on the latent variables is added to the loss. It is the sum of the squared off-diagonal entries of the latent covariance matrix, multiplied by the weight `alpha_corr` (1e-3). It is controlled by the option `use_corr_penalty`.

## Experiment
Each run uses the setup of the scArches expiMap tutorial. First, a reference model is trained on the PBMC data (4 studies, 32,484 cells) using Reactome pathways, 2000 highly variable genes, three hidden layers of 256 nodes, the negative binomial loss and a group lasso weight of 0.7. Training lasts up to 400 epochs with early stopping. Then the Kang data (13,576 cells, control and interferon-stimulated) is mapped onto the reference with architecture surgery, which keeps the reference model frozen and trains only the new input weights for the Kang study.

This is done for four versions of the model: the unmodified baseline, the extra layer alone, the correlation penalty alone, and both changes together. Each version is run with three random seeds (0, 1 and 2), and the results are reported as the mean and standard deviation over the seeds.

The results are measured in four ways. Batch mixing is measured with iLISI and batch ASW, and cell-type separation with cell-type ASW, NMI and ARI. Label transfer is measured by the accuracy and macro F1 of predicting the Kang cell types from the reference. Finally, three pathway checks look at how correlated the pathway latents are, how much the reference cells move after surgery, and how well the interferon signalling latent separates stimulated from control cells. All numbers are saved in `results/results_summary.csv`.

## Run
```
make env            # local venv from requirements.lock (arm64 macOS, Python 3.12)
python download_data.py
make smoke          # 3-epoch CPU test, a few minutes
make run            # full experiment (slow on CPU; use colab_run.ipynb with a GPU)
```
