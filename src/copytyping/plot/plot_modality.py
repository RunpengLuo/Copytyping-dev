"""Per-modality, per-REP_ID plot orchestrator shared by inference and validate."""

import logging
import os

from matplotlib.backends.backend_pdf import PdfPages

from copytyping.inference.inference_utils import adaptive_bin_bbc
from copytyping.inference.model_utils import compute_baseline_proportions
from copytyping.plot.plot_common import plot_cluster_observed_data
from copytyping.plot.plot_heatmap import plot_cnv_heatmap
from copytyping.plot.plot_scatter_1d import plot_rdr_baf_1d_pseudobulk
from copytyping.plot.plot_scatter_2d import plot_scatter_2d_per_cell
from copytyping.validation.metrics import compute_cluster_baf_metrics


def plot_modality_panel(
    *,
    sample,
    data_type,
    prefix,
    plot_dir,
    seg_sx,
    raw_clust,
    bbc_data,
    cnv_blocks,
    anns,
    is_normal,
    primary_label,
    plot_labels,
    theta,
    region_bed,
    genome_size,
    dpi,
    heatmap_agg,
    min_snp_count,
    max_bin_length,
    platform_str,
    compute_baf_metrics=False,
    ascn_profile=False,
):
    """Per-modality + per-REP_ID plots: cluster_obs, cluster_2d, heatmap, 1d_scatter.

    Files written:
        {plot_dir}/{prefix}.{data_type}.cluster_obs.{label}.pdf  (per plot_label)
        {plot_dir}/{prefix}.{data_type}.cluster_2d.pdf
        {plot_dir}/{prefix}.{data_type}.heatmap.{label}.agg{agg}.pdf
            (multi-page: rep_id outer × val inner)
        {plot_dir}/{prefix}.{data_type}.1d_scatter.{label}.pdf
            (multi-page: one page per rep_id)
    """
    raw_lambda = (
        compute_baseline_proportions(raw_clust.X, raw_clust.T, is_normal)
        if is_normal.sum() > 0
        else None
    )

    for obs_label in plot_labels:
        baf_metrics = None
        if compute_baf_metrics:
            baf_metrics = compute_cluster_baf_metrics(raw_clust, anns[obs_label].values)
            for g, m in sorted(baf_metrics.items()):
                logging.info(
                    f"  cluster {g} ({data_type}, {obs_label}): "
                    f"within_var={m['within_var']:.4f}"
                )
        plot_cluster_observed_data(
            raw_clust,
            anns,
            sample,
            os.path.join(plot_dir, f"{prefix}.{data_type}.cluster_obs.{obs_label}.pdf"),
            label_col=obs_label,
            base_props=raw_lambda,
            baf_metrics=baf_metrics,
            dpi=dpi,
        )
    plot_scatter_2d_per_cell(
        raw_clust,
        anns,
        sample,
        os.path.join(plot_dir, f"{prefix}.{data_type}.cluster_2d.pdf"),
        label_col=primary_label,
        base_props=raw_lambda,
        dpi=dpi,
    )

    bbc_df_dt, X_bbc, Y_bbc, D_bbc = bbc_data
    agg_bbc = None
    if bbc_df_dt is not None and genome_size and region_bed:
        agg_bbc = adaptive_bin_bbc(
            bbc_df_dt,
            X_bbc,
            Y_bbc,
            D_bbc,
            seg_sx,
            min_snp_count,
            max_bin_length,
        )

    rep_ids = sorted(seg_sx.barcodes["REP_ID"].unique())
    rep_views = {}
    for rep_id in rep_ids:
        seg_sx_rep, rep_mask = seg_sx.subset_by_rep(rep_id)
        is_normal_rep = is_normal[rep_mask]
        rep_views[rep_id] = {
            "seg_sx": seg_sx_rep,
            "rep_mask": rep_mask,
            "anns": anns.iloc[rep_mask].reset_index(drop=True),
            "is_normal": is_normal_rep,
            "seg_lambda": (
                compute_baseline_proportions(seg_sx_rep.X, seg_sx_rep.T, is_normal_rep)
                if is_normal_rep.sum() > 0
                else None
            ),
            "theta": theta[rep_mask] if theta is not None else None,
        }

    if region_bed:
        for agg in [1, heatmap_agg]:
            for my_label in plot_labels:
                fname = os.path.join(
                    plot_dir,
                    f"{prefix}.{data_type}.heatmap.{my_label}.agg{agg}.pdf",
                )
                with PdfPages(fname) as pdf:
                    for rep_id in rep_ids:
                        v = rep_views[rep_id]
                        for val in ["BAF", "log2RDR"]:
                            if val == "log2RDR" and v["seg_lambda"] is None:
                                continue
                            plot_cnv_heatmap(
                                sample,
                                data_type,
                                cnv_blocks,
                                v["seg_sx"],
                                v["anns"],
                                region_bed,
                                proportions=v["theta"],
                                val=val,
                                base_props=v["seg_lambda"],
                                agg_size=agg,
                                lab_type=my_label,
                                pdf_pages=pdf,
                                dpi=dpi,
                                figsize=(20, 6 if agg > 1 else 15),
                                title_info=f"rep={rep_id}",
                                ascn_profile=ascn_profile,
                            )

    if agg_bbc is not None:
        for my_label in plot_labels:
            fname = os.path.join(
                plot_dir, f"{prefix}.{data_type}.1d_scatter.{my_label}.pdf"
            )
            with PdfPages(fname) as pdf:
                for rep_id in rep_ids:
                    v = rep_views[rep_id]
                    agg_bbc_rep, _ = agg_bbc.subset_by_rep(rep_id)
                    agg_lambda_rep = (
                        compute_baseline_proportions(
                            agg_bbc_rep.X, agg_bbc_rep.T, v["is_normal"]
                        )
                        if v["is_normal"].sum() > 0
                        else None
                    )
                    plot_rdr_baf_1d_pseudobulk(
                        agg_bbc_rep,
                        v["anns"],
                        agg_lambda_rep,
                        sample,
                        data_type,
                        genome_size,
                        haplo_blocks=cnv_blocks,
                        region_bed=region_bed,
                        lab_type=my_label,
                        is_inferred=(my_label == primary_label),
                        pdf_pages=pdf,
                        platform=platform_str,
                        subtitle=f"rep={rep_id}",
                        ascn_profile=ascn_profile,
                    )
