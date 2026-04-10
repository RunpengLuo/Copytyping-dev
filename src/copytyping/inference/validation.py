import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    adjusted_rand_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from copytyping.utils import is_tumor_label, NA_CELLTYPE
from copytyping.inference.model_utils import (
    clone_rdr_gk,
    clone_pi_gk,
)
from copytyping.inference.likelihood_funcs import (
    cond_betabin_logpmf,
    cond_betabin_logpmf_theta,
    cond_negbin_logpmf,
    cond_negbin_logpmf_theta,
)


def _eval_subset(anns_sub, cell_label, cell_type, tumor_post):
    """Compute metrics for a subset of annotations. Returns dict."""
    known_mask = ~anns_sub[cell_type].isin(NA_CELLTYPE)
    anns_known = anns_sub[known_mask]
    na_count = int((anns_sub[cell_label] == "NA").sum())
    total = len(anns_sub)

    y_true = anns_known[cell_type].apply(is_tumor_label).to_numpy(dtype=int)
    y_pred_hard = anns_known[cell_label].apply(is_tumor_label).to_numpy(dtype=int)

    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        precision = recall = f1 = accuracy = auc_hard = np.nan
        auc_soft = np.nan
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_hard, average="binary"
        )
        accuracy = accuracy_score(y_true, y_pred_hard)
        auc_hard = roc_auc_score(y_true, y_pred_hard)
        auc_soft = None
        if tumor_post in anns_known:
            auc_soft = roc_auc_score(y_true, anns_known[tumor_post])

    # per-clone counts
    label_counts = anns_sub[cell_label].value_counts()
    clone_cols = sorted([c for c in label_counts.index if c.startswith("clone")])
    metric = {
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "ROC-AUC (hard)": auc_hard,
        "ROC-AUC (soft)": auc_soft,
        "#normal": int(label_counts.get("normal", 0)),
    }
    for c in clone_cols:
        metric[f"#{c}"] = int(label_counts.get(c, 0))
    metric["#NA"] = na_count
    return metric


def evaluate_malignant_accuracy(
    anns: pd.DataFrame,
    cell_label="cell_label",
    cell_type="cell_type",
    tumor_post="tumor",
):
    """Evaluate tumor/normal classification accuracy.

    Returns per-REP_ID rows plus an "ALL" aggregate row.
    Cells with Unknown cell_type are excluded from all metrics.
    """
    # overall metrics
    metric_all = _eval_subset(anns, cell_label, cell_type, tumor_post)
    metric_all["REP_ID"] = "ALL"

    logging.info("==================================================")
    logging.info(f"ROC AUC (hard classification): {metric_all['ROC-AUC (hard)']}")
    if metric_all.get("ROC-AUC (soft)") is not None:
        logging.info(f"ROC AUC (soft classification): {metric_all['ROC-AUC (soft)']}")

    # per-clone cross-tab
    known_mask = ~anns[cell_type].isin(NA_CELLTYPE)
    anns_known = anns[known_mask]
    ct = pd.crosstab(anns_known[cell_label], anns_known[cell_type])
    logging.info(f"Confusion matrix:\n{ct}")
    logging.info(
        f"#NA={metric_all['#NA']}, precision={metric_all['precision']:.4f}, "
        f"recall={metric_all['recall']:.4f}, f1={metric_all['f1']:.4f}, "
        f"accuracy={metric_all['accuracy']:.4f}"
    )

    # per-REP_ID metrics
    rows = [metric_all]
    if "REP_ID" in anns.columns:
        for rep_id, anns_rep in anns.groupby("REP_ID", sort=True):
            m = _eval_subset(anns_rep, cell_label, cell_type, tumor_post)
            m["REP_ID"] = rep_id
            rows.append(m)

    metric_str = (
        f"precision={metric_all['precision']:.3f}, "
        f"recall={metric_all['recall']:.3f}, "
        f"f1={metric_all['f1']:.3f}, "
        f"accuracy={metric_all['accuracy']:.3f}"
    )
    return metric_all, metric_str, rows


def refine_labels_by_reference(
    anns: pd.DataFrame,
    ref_label="cell_type",
    cell_label="cell_label",
    out_label="refined_label",
):
    # TODO, ref label vals
    num_na_before = (anns[cell_label] == "NA").sum()
    anns[out_label] = anns[cell_label]
    ref_is_tumor = anns[ref_label].apply(is_tumor_label)
    anns.loc[ref_is_tumor & (anns[cell_label] == "normal"), out_label] = "NA"
    anns.loc[~ref_is_tumor & (anns[cell_label] != "normal"), out_label] = "NA"
    num_na_after = (anns[out_label] == "NA").sum()
    logging.info(
        f"#NA before/after refinement={num_na_before}->{num_na_after} / {len(anns)}"
    )
    return anns


def evaluate_clone_accuracy(
    anns: pd.DataFrame,
    pred_label: str,
    gt_label: str,
):
    """Evaluate clone-level clustering accuracy using ARI.

    Compares predicted clone labels against ground-truth clone labels on
    tumor spots only. Uses ARI (label-permutation invariant).

    Returns:
        dict with ARI, n_tumor, and crosstab. Empty dict if not evaluable.
    """
    if gt_label not in anns.columns:
        return {}

    gt_is_tumor = anns[gt_label].apply(is_tumor_label)
    tumor = anns[gt_is_tumor].copy()
    if len(tumor) < 2:
        return {}

    valid = tumor[~tumor[pred_label].isin({"normal", "NA"})]
    if len(valid) < 2:
        return {}

    ari = adjusted_rand_score(valid[gt_label], valid[pred_label])
    ct = pd.crosstab(valid[gt_label], valid[pred_label])

    logging.info(f"Clone evaluation: ARI={ari:.4f}, n_tumor={len(valid)}")
    logging.info(f"Crosstab:\n{ct}")

    return {"ARI": ari, "n_tumor": len(valid), "crosstab": ct}


def compute_cluster_discrimination(
    sx_data,
    params: dict,
    anns: pd.DataFrame,
    pred_label: str,
    gt_label: str,
    data_type: str = "gex",
    is_spot: bool = False,
    seg_data=None,
):
    """Compute per-cluster clone discrimination power.

    For each CNP cluster, assigns tumor spots to best clone using only that
    cluster's log-likelihood, then computes ARI against GT.

    Args:
        sx_data: segment or cluster-level SX_Data (used for EM).
        params: fitted model params (lambda, tau, inv_phi, theta).
        anns: annotations DataFrame with pred_label and gt_label.
        pred_label: predicted label column name.
        seg_data: optional segment-level SX_Data for chromosome mapping.
        gt_label: ground-truth label column name.
        data_type: e.g. "gex".
        is_spot: True for spot model (uses theta-adjusted likelihoods).

    Returns:
        DataFrame with per-cluster: cid, CNP, n_segs, ARI_baf, ARI_rdr, ARI_hyb.
    """
    if gt_label not in anns.columns:
        return pd.DataFrame()

    gt_is_tumor = anns[gt_label].apply(is_tumor_label)
    tumor_idx = np.where(gt_is_tumor.values)[0]
    gt_labels = anns.loc[gt_is_tumor, gt_label].values
    if len(tumor_idx) < 2:
        return pd.DataFrame()

    lambda_g = params.get(f"{data_type}-lambda")
    if lambda_g is None:
        return pd.DataFrame()

    tau = params.get(f"{data_type}-tau")
    inv_phi = params.get(f"{data_type}-inv_phi")
    theta = params.get(f"{data_type}-theta")

    K = sx_data.K
    tumor_clones = sx_data.clones[1:] if is_spot else sx_data.clones
    K_eval = len(tumor_clones)

    if is_spot:
        rdrs_gk = clone_rdr_gk(lambda_g, sx_data.C)[:, 1:]
        theta_tumor = theta[tumor_idx] if theta is not None else None
    else:
        rdrs_gk = None

    # Build cluster -> chromosomes mapping from segment-level data
    cid_to_chroms = {}
    if seg_data is not None and hasattr(sx_data, "cluster_ids"):
        seg_cnv = seg_data.cnv_blocks
        cluster_ids = sx_data.cluster_ids
        for cid in range(sx_data.G):
            members = np.where(cluster_ids == cid)[0]
            chroms = seg_cnv.iloc[members]["#CHR"].unique()
            short = ",".join(
                sorted(
                    chroms,
                    key=lambda x: int(
                        x.replace("chr", "").replace("X", "23").replace("Y", "24")
                    ),
                )
            ).replace("chr", "")
            cid_to_chroms[cid] = short

    rows = []
    for g in range(sx_data.G):
        cnp_parts = [f"{sx_data.A[g, k]}|{sx_data.B[g, k]}" for k in range(1, K)]
        cnp_str = ";".join(cnp_parts)
        is_imb = bool(np.any(sx_data.A[g, 1:] != sx_data.B[g, 1:]))
        is_ane = bool(np.any(sx_data.C[g, 1:] != 2))
        n_segs = (
            int(np.sum(sx_data.cluster_ids == g))
            if hasattr(sx_data, "cluster_ids")
            else 1
        )
        chroms = cid_to_chroms.get(g, "")

        ari_baf, ari_rdr, ari_hyb = 0.0, 0.0, 0.0
        ll_hyb = np.zeros((len(tumor_idx), K_eval))

        if is_imb and tau is not None:
            if is_spot and theta_tumor is not None:
                ll_bb = cond_betabin_logpmf_theta(
                    sx_data.Y[g : g + 1, tumor_idx],
                    sx_data.D[g : g + 1, tumor_idx],
                    tau[0:1],
                    sx_data.BAF[g : g + 1, 1:],
                    rdrs_gk[g : g + 1],
                    theta_tumor,
                )[0]
            else:
                ll_bb = cond_betabin_logpmf(
                    sx_data.Y[g : g + 1, tumor_idx],
                    sx_data.D[g : g + 1, tumor_idx],
                    tau[0:1],
                    sx_data.BAF[g : g + 1],
                )[0]
                if not is_spot:
                    ll_bb = ll_bb[:, 1:]  # drop normal for cell model comparison
            if ll_bb.shape[1] >= 2:
                pred = np.array(tumor_clones)[ll_bb.argmax(axis=1)]
                ari_baf = adjusted_rand_score(gt_labels, pred)
            ll_hyb += ll_bb

        if is_ane and inv_phi is not None:
            if is_spot and theta_tumor is not None:
                ll_nb = cond_negbin_logpmf_theta(
                    sx_data.X[g : g + 1, tumor_idx],
                    sx_data.T[tumor_idx],
                    lambda_g[g : g + 1],
                    inv_phi[0:1],
                    rdrs_gk[g : g + 1],
                    theta_tumor,
                )[0]
            else:
                props_gk = clone_pi_gk(lambda_g, sx_data.C)[g : g + 1]
                ll_nb = cond_negbin_logpmf(
                    sx_data.X[g : g + 1, tumor_idx],
                    sx_data.T[tumor_idx],
                    props_gk,
                    inv_phi[0:1],
                )[0]
                if not is_spot:
                    ll_nb = ll_nb[:, 1:]
            if ll_nb.shape[1] >= 2:
                pred = np.array(tumor_clones)[ll_nb.argmax(axis=1)]
                ari_rdr = adjusted_rand_score(gt_labels, pred)
            ll_hyb += ll_nb

        if (is_imb or is_ane) and ll_hyb.shape[1] >= 2:
            pred = np.array(tumor_clones)[ll_hyb.argmax(axis=1)]
            ari_hyb = adjusted_rand_score(gt_labels, pred)

        rows.append(
            {
                "cid": g,
                "chroms": chroms,
                "CNP": cnp_str,
                "n_segs": n_segs,
                "IMB": is_imb,
                "ANE": is_ane,
                "med_D": int(np.median(sx_data.D[g])),
                "ARI_baf": round(ari_baf, 4),
                "ARI_rdr": round(ari_rdr, 4),
                "ARI_hyb": round(ari_hyb, 4),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("ARI_hyb", ascending=False)
        .reset_index(drop=True)
    )
    return df
