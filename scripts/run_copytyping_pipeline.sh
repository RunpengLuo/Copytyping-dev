#!/bin/bash
set -euo pipefail

# ============================================================
# Global configuration — edit these paths
# ============================================================
COPYTYPING_ENV="/path/to/copytyping-env"
COPYTYPING_OUTDIR="/path/to/copytyping_results"
PREPROCESS_DIR="${COPYTYPING_OUTDIR}/preprocess"
GENOME_SIZE="/path/to/hg38.chrom.sizes"
REGION_BED="/path/to/hg38.segments.bed"
MSR=50
MSPB=3
BB_DIR_NAME="bb_msr${MSR}_mspb${MSPB}"

# ============================================================
# Usage
# ============================================================
# ./run_copytyping.sh <panel.tsv> [platform_filter]
#
# Panel TSV columns (tab-separated, header required):
#   SAMPLE  cancer_type  platform  celltype_file  bbc_phase
#   seg_ucn  PLOIDY  CLONE  sol_id
#
# Field conventions:
#   celltype_file : path or "UNKNOWN"
#   seg_ucn       : path or "UNKNOWN" (skip sample)
#   bbc_phase     : path or "UNKNOWN" (skip sample)
#   sol_id        : specific id (e.g. "0.01") | "UNKNOWN" (sweep) | "DEFAULT" (no solfile)
#   PLOIDY        : "diploid" | "tetraploid" | "UNKNOWN" (sweep both)
#   CLONE         : 2 | 3 | 4 | "UNKNOWN" (sweep all)
#
# platform_filter : optional, e.g. "spatial" or "single_cell"

PANEL_TSV="${1:?Usage: $0 <panel.tsv> [platform_filter]}"
PLATFORM_FILTER="${2:-}"

# ============================================================
# Helper: run a single copytyping inference
# ============================================================
run_one() {
    local out_dir="$1"; shift
    mkdir -p "${out_dir}"
    echo "==================================================="
    echo "${SAMPLE} ${platform} ${out_dir}"; date
    conda run -p "${COPYTYPING_ENV}" --no-capture-output \
        copytyping inference -o "${out_dir}" "${common_args[@]}" "$@" -v 2 \
        1>/dev/null
    awk '/tumor\/normal evaluation:/{f=1; print; next} f && /INFO  /{print; next} f{exit}' \
        "${out_dir}/copytyping.log"
}

# ============================================================
# Main loop
# ============================================================
tail -n +2 "${PANEL_TSV}" | while IFS=$'\t' read -r SAMPLE cancer_type platform celltype_file bbc_phase seg_ucn PLOIDY CLONE sol_id; do

    # Platform filter
    if [ -n "$PLATFORM_FILTER" ] && [ "$platform" != "$PLATFORM_FILTER" ]; then
        continue
    fi

    # Validate required inputs
    BB_DIR="${PREPROCESS_DIR}/${SAMPLE}/${BB_DIR_NAME}"
    if [ ! -d "${BB_DIR}" ]; then
        echo "SKIP ${SAMPLE}: missing ${BB_DIR}"
        continue
    fi
    if [ "$seg_ucn" == "UNKNOWN" ]; then
        echo "SKIP ${SAMPLE}: seg_ucn is UNKNOWN"
        continue
    fi
    if [ "$bbc_phase" == "UNKNOWN" ]; then
        echo "SKIP ${SAMPLE}: bbc_phase is UNKNOWN"
        continue
    fi

    # Build common args
    common_args=()
    common_args+=(--platform "${platform}")
    common_args+=(--sample "${SAMPLE}")
    common_args+=(--seg_ucn "${seg_ucn}")
    common_args+=(--bbc_phases "${bbc_phase}")
    common_args+=(--genome_size "${GENOME_SIZE}")
    common_args+=(--region_bed "${REGION_BED}")

    # Platform-specific dirs and ref_label
    if [ "$platform" == "spatial" ]; then
        common_args+=(--gex_dir "${BB_DIR}/VISIUM")
        common_args+=(--ref_label path_label)
    else
        [ -d "${BB_DIR}/scRNA" ]  && common_args+=(--gex_dir "${BB_DIR}/scRNA")
        [ -d "${BB_DIR}/scATAC" ] && common_args+=(--atac_dir "${BB_DIR}/scATAC")
        common_args+=(--ref_label cell_type)
    fi

    # Optional cell type annotation
    if [ "$celltype_file" != "UNKNOWN" ] && [ -f "$celltype_file" ]; then
        common_args+=(--cell_type "$celltype_file")
    fi

    # ----------------------------------------------------------
    # Dispatch based on sol_id / PLOIDY / CLONE
    # ----------------------------------------------------------
    RES_DIR="$(dirname "${seg_ucn}")"

    if [ "$sol_id" != "UNKNOWN" ] && [ "$sol_id" != "DEFAULT" ]; then
        # Specific solution: PLOIDY and CLONE must be known
        seg_file="${RES_DIR}/results.${PLOIDY}.n${CLONE}.seg.ucn.tsv"
        sol_file="${RES_DIR}/sols/${PLOIDY}_n${CLONE}/cd_sol${sol_id}_pool0.tsv"
        if [ ! -f "$sol_file" ]; then
            echo "SKIP ${SAMPLE}: missing ${sol_file}"
            continue
        fi
        run_one "${COPYTYPING_OUTDIR}/${SAMPLE}/${platform}/${PLOIDY}_n${CLONE}_sol${sol_id}" \
            --seg_ucn "${seg_file}" --solfile "${sol_file}"

    elif [ "$sol_id" == "UNKNOWN" ]; then
        # Sweep solfiles — narrow by known PLOIDY / CLONE
        if [ "$PLOIDY" != "UNKNOWN" ]; then
            ploidies=("$PLOIDY")
        else
            ploidies=(diploid tetraploid)
        fi
        if [ "$CLONE" != "UNKNOWN" ]; then
            clones=("$CLONE")
        else
            clones=(2 3 4)
        fi

        for ploidy in "${ploidies[@]}"; do
            for clone in "${clones[@]}"; do
                seg_file="${RES_DIR}/results.${ploidy}.n${clone}.seg.ucn.tsv"
                [ -f "$seg_file" ] || continue
                sol_dir="${RES_DIR}/sols/${ploidy}_n${clone}"
                [ -d "$sol_dir" ] || continue
                for sol_file in "${sol_dir}"/cd_sol*_pool*.tsv; do
                    [ -f "$sol_file" ] || continue
                    sid=$(echo "${sol_file##*/}" | sed 's/cd_sol\(.*\)_pool.*/\1/')
                    run_one "${COPYTYPING_OUTDIR}/${SAMPLE}/${platform}/${ploidy}_n${clone}_sol${sid}" \
                        --seg_ucn "${seg_file}" --solfile "${sol_file}"
                done
            done
        done

    else
        # DEFAULT: no solfile, use seg_ucn directly
        run_one "${COPYTYPING_OUTDIR}/${SAMPLE}/${platform}/${PLOIDY}_n${CLONE}_default"
    fi

done

echo "=== All done ==="
