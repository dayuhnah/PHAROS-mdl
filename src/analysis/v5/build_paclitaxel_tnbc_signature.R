library(SeuratObject)
library(Matrix)

INPUT <- "data/raw/cellresdb/rds/GSE169246_TNBC_1.rds"
OUTDIR <- "data/processed/v5/cellresdb/paclitaxel_tnbc"

MIN_CELLS <- 20
MIN_PATIENTS <- 5

dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

cat("Loading Seurat object...\n")
obj <- readRDS(INPUT)

meta <- obj[[]]

# Make sure metadata order matches assay columns
meta <- meta[colnames(obj), , drop = FALSE]

# ------------------------------------------------------------
# 1. PRE-TREATMENT R / NR ONLY
# ------------------------------------------------------------

keep <- (
    meta$treatment == "Pre" &
    meta$Response %in% c("R", "NR")
)

meta_pre <- meta[keep, , drop = FALSE]

cat("\nPre-treatment cells:", nrow(meta_pre), "\n")
cat("Patients:", length(unique(meta_pre$patient)), "\n")

# ------------------------------------------------------------
# 2. RAW COUNTS
# ------------------------------------------------------------

counts <- LayerData(
    obj[["RNA"]],
    layer = "counts"
)

counts <- counts[, rownames(meta_pre), drop = FALSE]

# ------------------------------------------------------------
# 3. PATIENT + CELL-TYPE GROUPS
# ------------------------------------------------------------

group_key <- paste(
    meta_pre$patient,
    meta_pre$Response,
    meta_pre$cluster1,
    sep = "|||"
)

group_factor <- factor(group_key)

group_meta <- unique(
    data.frame(
        group_key = group_key,
        patient = meta_pre$patient,
        response = meta_pre$Response,
        cell_type = meta_pre$cluster1,
        stringsAsFactors = FALSE
    )
)

group_meta <- group_meta[
    match(levels(group_factor), group_meta$group_key),
    ,
    drop = FALSE
]

# Sparse cell -> pseudobulk group matrix
G <- sparseMatrix(
    i = seq_along(group_factor),
    j = as.integer(group_factor),
    x = 1,
    dims = c(
        length(group_factor),
        nlevels(group_factor)
    )
)

cat("\nBuilding pseudobulk counts...\n")

pb_counts <- counts %*% G

group_meta$n_cells <- tabulate(
    as.integer(group_factor),
    nbins = nlevels(group_factor)
)

# ------------------------------------------------------------
# 4. REQUIRE ENOUGH CELLS PER PATIENT/CELL TYPE
# ------------------------------------------------------------

valid_groups <- group_meta$n_cells >= MIN_CELLS

pb_counts <- pb_counts[, valid_groups, drop = FALSE]
group_meta <- group_meta[valid_groups, , drop = FALSE]

cat(
    "\nPseudobulk samples after MIN_CELLS =",
    MIN_CELLS,
    ":",
    ncol(pb_counts),
    "\n"
)

# ------------------------------------------------------------
# 5. DETERMINE ELIGIBLE CELL TYPES
# ------------------------------------------------------------

cell_types <- sort(unique(group_meta$cell_type))

coverage_rows <- list()
eligible <- c()

for (ct in cell_types) {

    tmp <- group_meta[group_meta$cell_type == ct, ]

    n_r <- sum(tmp$response == "R")
    n_nr <- sum(tmp$response == "NR")

    coverage_rows[[length(coverage_rows) + 1]] <- data.frame(
        cell_type = ct,
        n_R = n_r,
        n_NR = n_nr
    )

    if (n_r >= MIN_PATIENTS && n_nr >= MIN_PATIENTS) {
        eligible <- c(eligible, ct)
    }
}

coverage <- do.call(rbind, coverage_rows)

write.csv(
    coverage,
    file.path(OUTDIR, "celltype_patient_coverage.csv"),
    row.names = FALSE
)

cat("\n=== COVERAGE AFTER CELL FILTER ===\n")
print(coverage)

cat("\nEligible cell types:\n")
print(eligible)

# ------------------------------------------------------------
# 6. LOG-CPM PSEUDOBULK NORMALIZATION
# ------------------------------------------------------------

pb_dense <- as.matrix(pb_counts)

library_sizes <- colSums(pb_dense)

logcpm <- log2(
    sweep(
        pb_dense,
        2,
        library_sizes,
        "/"
    ) * 1e6 + 1
)

# ------------------------------------------------------------
# 7. NR - R SIGNATURE FOR EACH CELL TYPE
# ------------------------------------------------------------

all_results <- list()

for (ct in eligible) {

    cat("\nProcessing:", ct, "\n")

    idx <- which(group_meta$cell_type == ct)

    gm <- group_meta[idx, , drop = FALSE]
    mat <- logcpm[, idx, drop = FALSE]

    r_idx <- which(gm$response == "R")
    nr_idx <- which(gm$response == "NR")

    Rmat <- mat[, r_idx, drop = FALSE]
    NRmat <- mat[, nr_idx, drop = FALSE]

    nR <- ncol(Rmat)
    nNR <- ncol(NRmat)

    mean_R <- rowMeans(Rmat)
    mean_NR <- rowMeans(NRmat)

    delta <- mean_NR - mean_R

    var_R <- apply(Rmat, 1, var)
    var_NR <- apply(NRmat, 1, var)

    se <- sqrt(
        var_R / nR +
        var_NR / nNR
    )

    t_stat <- delta / se

    # Welch-Satterthwaite df
    numerator <- (
        var_R / nR +
        var_NR / nNR
    )^2

    denominator <- (
        (var_R / nR)^2 / (nR - 1) +
        (var_NR / nNR)^2 / (nNR - 1)
    )

    df <- numerator / denominator

    bad <- (
        !is.finite(t_stat) |
        !is.finite(df)
    )

    t_stat[bad] <- 0
    df[bad] <- 1

    p_value <- 2 * pt(
        -abs(t_stat),
        df = df
    )

    padj <- p.adjust(
        p_value,
        method = "BH"
    )

    result <- data.frame(
        gene = rownames(mat),
        cell_type = ct,
        n_R = nR,
        n_NR = nNR,
        mean_R_logCPM = mean_R,
        mean_NR_logCPM = mean_NR,
        NR_minus_R = delta,
        abs_effect = abs(delta),
        t_stat = t_stat,
        p_value = p_value,
        padj = padj,
        stringsAsFactors = FALSE
    )

    all_results[[ct]] <- result
}

signatures <- do.call(rbind, all_results)

# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------

write.csv(
    signatures,
    gzfile(
        file.path(
            OUTDIR,
            "paclitaxel_TNBC_pre_NR_vs_R_signatures.csv.gz"
        )
    ),
    row.names = FALSE
)

saveRDS(
    list(
        logcpm = logcpm,
        group_metadata = group_meta
    ),
    file.path(
        OUTDIR,
        "paclitaxel_TNBC_pre_pseudobulk.rds"
    )
)

# Top genes for easy inspection
top <- do.call(
    rbind,
    lapply(
        split(signatures, signatures$cell_type),
        function(x) {
            x <- x[order(-x$abs_effect), ]
            head(x, 20)
        }
    )
)

write.csv(
    top,
    file.path(
        OUTDIR,
        "top20_genes_per_celltype.csv"
    ),
    row.names = FALSE
)

cat("\n========================================\n")
cat("DONE\n")
cat("========================================\n")
cat("Genes:", length(unique(signatures$gene)), "\n")
cat("Eligible cell types:", length(eligible), "\n")
cat("Signature rows:", nrow(signatures), "\n")
cat("Output:", OUTDIR, "\n")
