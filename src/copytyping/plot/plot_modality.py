"""Per-modality, per-REP_ID plot orchestrator shared by inference and validate."""

import logging
import os

from matplotlib.backends.backend_pdf import PdfPages

from copytyping.inference.inference_utils import adaptive_bin_bbc
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
    baseline_fn,
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
    cluster_base_props=None,
    compute_baf_metrics=False,
    ascn_profile=False,
):
    """Per-modality + per-REP_ID plots: cluster_obs, cluster_2d, heatmap, 1d_scatter.

    Cluster-level plots reuse the model's fitted baseline (``cluster_base_props``,
    i.e. ``model_params["{dt}-lambda"]``); the finer-resolution heatmap (seg) and
    1d_scatter (bbc) derive theirs via ``baseline_fn(sx)``.

    Files written:
        {plot_dir}/{prefix}.{data_type}.cluster_obs.{label}.pdf  (per plot_label)
        {plot_dir}/{prefix}.{data_type}.cluster_2d.pdf
        {plot_dir}/{prefix}.{data_type}.heatmap.agg{agg}.pdf
            (multi-page: rep_id outer × val inner; labels shown as color strips)
        {plot_dir}/{prefix}.{data_type}.1d_scatter.{label}.pdf
            (multi-page: one page per rep_id)
    """
    raw_lambda = (
        cluster_base_props if cluster_base_props is not None else baseline_fn(raw_clust)
    )
    seg_lambda = baseline_fn(seg_sx)

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
    if bbc_df_dt is not None:
        agg_bbc = adaptive_bin_bbc(
            bbc_df_dt,
            X_bbc,
            Y_bbc,
            D_bbc,
            seg_sx,
            min_snp_count,
            max_bin_length,
        )
    agg_lambda = baseline_fn(agg_bbc) if agg_bbc is not None else None

    rep_ids = sorted(seg_sx.barcodes["REP_ID"].unique())
    rep_views = {}
    for rep_id in rep_ids:
        seg_sx_rep, rep_mask = seg_sx.subset_by_rep(rep_id)
        rep_views[rep_id] = {
            "seg_sx": seg_sx_rep,
            "anns": anns.iloc[rep_mask].reset_index(drop=True),
            "theta": theta[rep_mask] if theta is not None else None,
        }

    # one heatmap file per agg level; labels shown as color strips, no per-label spawn
    for agg in [1, heatmap_agg]:
        fname = os.path.join(
            plot_dir,
            f"{prefix}.{data_type}.heatmap.agg{agg}.pdf",
        )
        with PdfPages(fname) as pdf:
            for rep_id in rep_ids:
                v = rep_views[rep_id]
                for val in ["BAF", "log2RDR"]:
                    if val == "log2RDR" and seg_lambda is None:
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
                        base_props=seg_lambda,
                        agg_size=agg,
                        label_cols=plot_labels,
                        primary_label=primary_label,
                        pdf_pages=pdf,
                        dpi=dpi,
                        figsize=(20, 6 if agg > 1 else 15),
                        rep_id=rep_id,
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
                    plot_rdr_baf_1d_pseudobulk(
                        agg_bbc_rep,
                        v["anns"],
                        agg_lambda,
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
