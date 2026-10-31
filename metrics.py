"""Hand-coded integration / label-transfer / pathway metrics (numpy + scikit-learn + scanpy only).

All batch-mixing scores are scaled to [0, 1], higher = better mixed; all bio scores higher = better.
Simplifications versus scib: iLISI uses a plain k-nearest-neighbour inverse Simpson index (no
perplexity-based Gaussian weighting), ASW values are computed on random subsamples of at most
MAX_ASW cells, and clustering for NMI/ARI is Leiden at resolution 1.0.
"""
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import (adjusted_rand_score, f1_score, normalized_mutual_info_score, roc_auc_score,
                             silhouette_score)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

MAX_ASW = 5000
K_LISI = 90


def _codes(x):
    return np.unique(np.asarray(x).astype(str), return_inverse=True)[1]


def ilisi(z, batch, seed, k=K_LISI, max_points=10000):
    """Normalised integration LISI in [0, 1]: (inverse Simpson of batch labels among k neighbours - 1) / (n_batches - 1)."""
    b = _codes(batch)
    nb = b.max() + 1
    if nb < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(z), min(max_points, len(z)), replace=False)
    nn = NearestNeighbors(n_neighbors=min(k, len(z) - 1) + 1).fit(z)
    nbr = nn.kneighbors(z[idx], return_distance=False)[:, 1:]
    counts = np.stack([(b[nbr] == j).sum(1) for j in range(nb)], 1) / nbr.shape[1]
    inv_simpson = 1.0 / (counts ** 2).sum(1)
    return float(np.mean((inv_simpson - 1) / (nb - 1)))


def batch_asw(z, batch, ctype, seed):
    """scib-style batch ASW: per cell type, 1 - |silhouette over batch labels|, averaged over cell types."""
    b, c = _codes(batch), _codes(ctype)
    rng = np.random.RandomState(seed)
    vals = []
    for ct in np.unique(c):
        m = np.where(c == ct)[0]
        if len(np.unique(b[m])) < 2 or len(m) < 10:
            continue
        m = rng.choice(m, min(MAX_ASW, len(m)), replace=False)
        if len(np.unique(b[m])) < 2:
            continue
        vals.append(1 - abs(silhouette_score(z[m], b[m])))
    return float(np.mean(vals)) if vals else float("nan")


def celltype_asw(z, ctype, seed):
    c = _codes(ctype)
    rng = np.random.RandomState(seed)
    m = rng.choice(len(z), min(MAX_ASW, len(z)), replace=False)
    return float((silhouette_score(z[m], c[m]) + 1) / 2)


def cluster_scores(z, ctype, seed, resolution=1.0):
    ad = AnnData(np.zeros((len(z), 1), dtype=np.float32))
    ad.obsm["X_emb"] = z
    sc.pp.neighbors(ad, use_rep="X_emb", random_state=seed)
    sc.tl.leiden(ad, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    cl = ad.obs["leiden"].values
    ct = np.asarray(ctype).astype(str)
    return float(normalized_mutual_info_score(ct, cl)), float(adjusted_rand_score(ct, cl))


def embedding_metrics(z, batch, ctype, seed):
    nmi, ari = cluster_scores(z, ctype, seed)
    return {
        "ilisi": ilisi(z, batch, seed),
        "batch_asw": batch_asw(z, batch, ctype, seed),
        "celltype_asw": celltype_asw(z, ctype, seed),
        "nmi": nmi,
        "ari": ari,
    }


def label_transfer(z_ref, y_ref, z_qry, y_qry, k=15):
    """kNN classifier fitted on reference latent, evaluated on the query (only cell types present in the reference)."""
    clf = KNeighborsClassifier(n_neighbors=k).fit(z_ref, np.asarray(y_ref).astype(str))
    pred = clf.predict(z_qry)
    y = np.asarray(y_qry).astype(str)
    known = np.isin(y, np.unique(np.asarray(y_ref).astype(str)))
    return {
        "lt_accuracy": float((pred[known] == y[known]).mean()),
        "lt_macro_f1": float(f1_score(y[known], pred[known], average="macro")),
        "lt_frac_query_types_in_ref": float(known.mean()),
    }


def _mean_abs_offdiag_corr(z):
    keep = z.std(0) > 1e-8
    c = np.corrcoef(z[:, keep], rowvar=False)
    d = c.shape[0]
    return float((np.abs(c).sum() - d) / (d * (d - 1)))


def pathway_checks(ref, qry, adata, kang, joint):
    """Checks that the pathway latents of expiMap are still meaningful.

    - terms_active_*: number of pathway nodes with non-zero decoder weights (group lasso survivors).
    - terms_kept_after_surgery: fraction of reference-active terms still active in the query model.
    - offdiag_corr_ref: mean |correlation| between active pathway latents (what the penalty targets).
    - ref_drift_r: median Pearson r, per active term, between the reference latent from the reference
      model and from the query model on the same cells (1 = reference space untouched by surgery).
    - ifn_auc: AUROC of the INTERFERON_SIGNALING latent (sign-corrected by learned direction) for
      stimulated vs control Kang cells; the tutorial's biological sanity check (0.5 = no signal).
    """
    nz_ref, nz_q = ref.nonzero_terms(), qry.nonzero_terms()
    out = {
        "terms_active_ref": int(nz_ref.sum()),
        "terms_active_query": int(nz_q.sum()),
        "terms_kept_after_surgery": float((nz_ref & nz_q).sum() / max(nz_ref.sum(), 1)),
    }
    z_ref_active = ref.get_latent(mean=True, only_active=True)
    out["offdiag_corr_ref"] = _mean_abs_offdiag_corr(z_ref_active)

    z_a = ref.get_latent(mean=True)
    z_b = qry.get_latent(adata.X, adata.obs["study"], mean=True)
    active = np.where(nz_ref)[0]
    active = active[active < z_a.shape[1]]
    rs = [np.corrcoef(z_a[:, i], z_b[:, i])[0, 1] for i in active if z_a[:, i].std() > 1e-8 and z_b[:, i].std() > 1e-8]
    out["ref_drift_r"] = float(np.median(rs)) if rs else float("nan")

    qry.latent_directions(adata=joint)
    z = qry.get_latent(joint.X, joint.obs["study"], mean=True) * joint.uns["directions"]
    terms = list(joint.uns["terms"])
    is_q = (joint.obs["study"] == "Kang").values
    cond = joint.obs["condition"].astype(str).str.lower().values[is_q]
    stim = np.array([c.startswith("stim") for c in cond])
    if "INTERFERON_SIGNALING" in terms and 0 < stim.sum() < len(stim):
        out["ifn_auc"] = float(roc_auc_score(stim, z[is_q][:, terms.index("INTERFERON_SIGNALING")]))
    else:
        out["ifn_auc"] = float("nan")
    return out
