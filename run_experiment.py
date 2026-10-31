"""One expiMap run: train reference on PBMC, map the Kang query by architecture surgery, compute metrics.

    python run_experiment.py --variant both --seed 0 --out results/runs

Variants (see VARIANTS): baseline = unmodified upstream behaviour; extra = student change #1 (extra
encoder layer); corr = student change #2 (correlation penalty); both = student's full modification.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))  # patched copy of scarches

warnings.simplefilter("ignore")

import scanpy as sc  # noqa: E402
import torch  # noqa: E402
import scarches as sca  # noqa: E402

from metrics import embedding_metrics, label_transfer, pathway_checks  # noqa: E402

# model options per variant; ALPHA_CORR is the penalty weight used in the trainer.
VARIANTS = {
    "baseline": dict(use_extra_layer=False, use_corr_penalty=False),
    "extra": dict(use_extra_layer=True, use_corr_penalty=False),
    # student's penalty as written: raw sum of squared off-diagonal covariances, weight 1e-3
    "corr": dict(use_extra_layer=False, use_corr_penalty=True),
    "both": dict(use_extra_layer=True, use_corr_penalty=True),
}
ALPHA = 0.7  # group-lasso strength, as in the tutorial
ALPHA_CORR = 1e-3  # student's value
EARLY_STOP = {
    "early_stopping_metric": "val_unweighted_loss",
    "threshold": 0,
    "patience": 50,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}


def prepare(data_dir, max_cells, seed):
    """Preprocessing exactly as in expimap_surgery_pipeline_basic.ipynb (optionally subsampled)."""
    adata = sc.read(os.path.join(data_dir, "pbmc_tutorial.h5ad"))
    adata.X = adata.layers["counts"].copy()
    sca.utils.add_annotations(adata, os.path.join(data_dir, "reactome.gmt"), min_genes=12, clean=True)
    adata._inplace_subset_var(adata.varm["I"].sum(1) > 0)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch", subset=True)
    keep = adata.varm["I"].sum(0) > 12
    adata.uns["terms"] = np.array(adata.uns["terms"])[keep].tolist()
    adata.varm["I"] = adata.varm["I"][:, keep]
    adata._inplace_subset_var(adata.varm["I"].sum(1) > 0)
    adata.X = adata.layers["counts"].copy()

    kang = sc.read(os.path.join(data_dir, "kang_tutorial.h5ad"))[:, adata.var_names].copy()
    kang.obs["study"] = "Kang"
    if "condition" not in adata.obs:  # tutorial treats reference cells as 'control'
        adata.obs["condition"] = "control"
    kang.uns["terms"] = adata.uns["terms"]

    if max_cells:
        rng = np.random.RandomState(0)  # fixed subsample, independent of the training seed
        adata = adata[np.sort(rng.choice(adata.n_obs, min(max_cells, adata.n_obs), replace=False))].copy()
        kang = kang[np.sort(rng.choice(kang.n_obs, min(max_cells // 3, kang.n_obs), replace=False))].copy()
    return adata, kang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "runs"))
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--epochs", type=int, default=400, help="max epochs (early stopping still active)")
    ap.add_argument("--max_cells", type=int, default=0, help="subsample reference to N cells (0 = all)")
    ap.add_argument("--save_models", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(args.out, args.variant, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"variant={args.variant} seed={args.seed} device={device}", flush=True)
    t0 = time.time()

    adata, kang = prepare(args.data, args.max_cells, args.seed)
    vkw = VARIANTS[args.variant]
    corr_kw = dict(alpha_corr=ALPHA_CORR) if vkw["use_corr_penalty"] else {}

    # ---- reference training
    ref = sca.models.EXPIMAP(adata=adata, condition_key="study", hidden_layer_sizes=[256, 256, 256],
                             recon_loss="nb", **vkw)
    ref.train(n_epochs=args.epochs, alpha_epoch_anneal=100, alpha=ALPHA, alpha_kl=0.5, weight_decay=0.0,
              early_stopping_kwargs=EARLY_STOP, use_early_stopping=True, monitor_only_val=False,
              seed=args.seed, **corr_kw)
    t_ref = time.time() - t0
    pd.DataFrame({k: pd.Series(v) for k, v in ref.trainer.logs.items()}).to_csv(
        os.path.join(run_dir, "ref_train_log.csv"), index=False)

    # ---- query mapping (architecture surgery)
    qry = sca.models.EXPIMAP.load_query_data(kang, ref)
    qry.train(n_epochs=args.epochs, alpha_epoch_anneal=100, weight_decay=0.0, alpha_kl=0.1,
              seed=args.seed, use_early_stopping=True, **corr_kw)
    t_all = time.time() - t0
    pd.DataFrame({k: pd.Series(v) for k, v in qry.trainer.logs.items()}).to_csv(
        os.path.join(run_dir, "query_train_log.csv"), index=False)
    if args.save_models:
        ref.save(os.path.join(run_dir, "ref_model"), overwrite=True)
        qry.save(os.path.join(run_dir, "query_model"), overwrite=True)

    # ---- embeddings (posterior mean => deterministic)
    z_ref = ref.get_latent(mean=True, only_active=True)
    adata.obsm["X_cvae"] = z_ref
    joint = sc.concat([adata, kang], label="batch_join", uns_merge="same")
    joint.obs_names_make_unique()
    joint.obsm["X_cvae"] = qry.get_latent(joint.X, joint.obs["study"], mean=True, only_active=True)
    is_q = (joint.obs["study"] == "Kang").values

    # ---- metrics
    res = dict(variant=args.variant, seed=args.seed, device=device, n_ref=adata.n_obs, n_query=kang.n_obs,
               n_latent_active_ref=int(z_ref.shape[1]), n_latent_active_joint=int(joint.obsm["X_cvae"].shape[1]),
               ref_epochs=len(ref.trainer.logs.get("epoch_loss", [])),
               query_epochs=len(qry.trainer.logs.get("epoch_loss", [])),
               time_ref_s=round(t_ref, 1), time_total_s=round(t_all, 1))
    res.update({f"ref_{k}": v for k, v in embedding_metrics(
        z_ref, adata.obs["study"].values, adata.obs["cell_type"].values, args.seed).items()})
    res.update({f"joint_{k}": v for k, v in embedding_metrics(
        joint.obsm["X_cvae"], joint.obs["study"].values, joint.obs["cell_type"].values, args.seed).items()})
    res.update(label_transfer(joint.obsm["X_cvae"][~is_q], joint.obs["cell_type"].values[~is_q],
                              joint.obsm["X_cvae"][is_q], joint.obs["cell_type"].values[is_q]))
    res.update(pathway_checks(ref, qry, adata, kang, joint))

    json.dump(res, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)

    # ---- figures
    sc.settings.figdir = run_dir
    for name, ad in (("reference", adata), ("joint", joint)):
        sc.pp.neighbors(ad, use_rep="X_cvae")
        sc.tl.umap(ad)
        sc.pl.umap(ad, color=["study", "cell_type"], frameon=False, wspace=0.6, show=False,
                   save=f"_{name}.png")
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
