library(SeuratObject)
library(Matrix)
library(edgeR)
library(limma)
library(msigdbr)

INPUT <- "data/raw/cellresdb/rds/GSE169246_TNBC_1.rds"

OUTDIR <- paste0(
    "data/processed/v5/cellresdb/",
    "paclitaxel_tnbc/limma_voom"
)

MIN_CELLS <- 20
MIN_PATIENTS <- 5

dir.create(
    OUTDIR,
    recursive = TRUE,
    showWarnings = FALSE
)

cat("Loading CellResDB object...\n")

obj <- readRDS(INPUT)
meta <- obj[[]]
meta <- meta[colnames(obj), , drop = FALSE]


# ============================================================
# 1. PRE-TREATMENT ONLY
# ============================================================

keep <- (
    meta$treatment == "Pre" &
    meta$Response %in% c("R", "NR")
)

meta <- meta[keep, , drop = FALSE]

cat("Pre-treatment cells:", nrow(meta), "\n")
cat(
    "Patients:",
    length(unique(meta$patient)),
    "\n"
)


# ============================================================
# 2. RAW COUNTS
# ============================================================

counts <- LayerData(
    obj[["RNA"]],
    layer = "counts"
)

counts <- counts[
    ,
    rownames(meta),
    drop = FALSE
]


# ============================================================
# 3. PATIENT × CELL-TYPE PSEUDOBULK
# ============================================================

group_key <- paste(
    meta$patient,
    meta$Response,
    meta$cluster1,
    sep = "|||"
)

group_factor <- factor(group_key)

group_meta <- unique(
    data.frame(
        group_key = group_key,
        patient = meta$patient,
        response = meta$Response,
        cell_type = meta$cluster1,
        stringsAsFactors = FALSE
    )
)

group_meta <- group_meta[
    match(
        levels(group_factor),
        group_meta$group_key
    ),
    ,
    drop = FALSE
]

G <- sparseMatrix(
    i = seq_along(group_factor),
    j = as.integer(group_factor),
    x = 1,
    dims = c(
        length(group_factor),
        nlevels(group_factor)
    )
)

cat("Building patient pseudobulk matrix...\n")

pb_counts <- counts %*% G

group_meta$n_cells <- tabulate(
    as.integer(group_factor),
    nbins = nlevels(group_factor)
)

valid <- group_meta$n_cells >= MIN_CELLS

pb_counts <- pb_counts[
    ,
    valid,
    drop = FALSE
]

group_meta <- group_meta[
    valid,
    ,
    drop = FALSE
]

cat(
    "Pseudobulk samples:",
    ncol(pb_counts),
    "\n"
)


# ============================================================
# 4. ELIGIBLE CELL TYPES
# ============================================================

coverage_rows <- list()

for (ct in sort(unique(group_meta$cell_type))) {

    z <- group_meta[
        group_meta$cell_type == ct,
        ,
        drop = FALSE
    ]

    coverage_rows[[length(coverage_rows) + 1]] <-
        data.frame(
            cell_type = ct,
            n_R = sum(z$response == "R"),
            n_NR = sum(z$response == "NR"),
            stringsAsFactors = FALSE
        )
}

coverage <- do.call(
    rbind,
    coverage_rows
)

coverage$eligible <- (
    coverage$n_R >= MIN_PATIENTS &
    coverage$n_NR >= MIN_PATIENTS
)

write.csv(
    coverage,
    file.path(
        OUTDIR,
        "celltype_patient_coverage.csv"
    ),
    row.names = FALSE
)

eligible <- coverage$cell_type[
    coverage$eligible
]

cat("\n=== ELIGIBLE CELL TYPES ===\n")
print(eligible)


# ============================================================
# 5. HALLMARK GENE SETS
# ============================================================

cat("\nLoading MSigDB Hallmark pathways...\n")

hallmark_df <- tryCatch(

    msigdbr(
        species = "Homo sapiens",
        collection = "H"
    ),

    error = function(e) {

        msigdbr(
            species = "Homo sapiens",
            category = "H"
        )
    }
)

hallmark_sets <- split(
    hallmark_df$gene_symbol,
    hallmark_df$gs_name
)

hallmark_sets <- lapply(
    hallmark_sets,
    unique
)

cat(
    "Hallmark pathways:",
    length(hallmark_sets),
    "\n"
)


# ============================================================
# 6. LIMMA-VOOM + CAMERA
# ============================================================

all_de <- list()
all_pathways <- list()

for (ct in eligible) {

    cat(
        "\n========================================\n"
    )
    cat("CELL TYPE:", ct, "\n")
    cat(
        "========================================\n"
    )

    idx <- which(
        group_meta$cell_type == ct
    )

    gm <- group_meta[
        idx,
        ,
        drop = FALSE
    ]

    mat <- pb_counts[
        ,
        idx,
        drop = FALSE
    ]

    response <- factor(
        gm$response,
        levels = c("R", "NR")
    )

    design <- model.matrix(
        ~0 + response
    )

    colnames(design) <- c(
        "R",
        "NR"
    )

    contrast <- makeContrasts(
        NR_minus_R = NR - R,
        levels = design
    )


    # --------------------------------------------------------
    # edgeR filtering + normalization
    # --------------------------------------------------------

    y <- DGEList(
        counts = mat
    )

    keep_genes <- filterByExpr(
        y,
        design = design
    )

    y <- y[
        keep_genes,
        ,
        keep.lib.sizes = FALSE
    ]

    y <- calcNormFactors(y)

    cat(
        "Genes after expression filtering:",
        nrow(y),
        "\n"
    )


    # --------------------------------------------------------
    # limma-voom
    # --------------------------------------------------------

    v <- voom(
        y,
        design,
        plot = FALSE
    )

    fit <- lmFit(
        v,
        design
    )

    fit2 <- contrasts.fit(
        fit,
        contrast
    )

    fit2 <- eBayes(
        fit2,
        robust = TRUE
    )

    de <- topTable(
        fit2,
        coef = "NR_minus_R",
        number = Inf,
        sort.by = "P"
    )

    de$gene <- rownames(de)
    de$cell_type <- ct

    de <- de[
        ,
        c(
            "gene",
            "cell_type",
            "logFC",
            "AveExpr",
            "t",
            "P.Value",
            "adj.P.Val",
            "B"
        )
    ]

    all_de[[ct]] <- de


    # --------------------------------------------------------
    # CAMERA pathway analysis
    # --------------------------------------------------------

    indices <- ids2indices(
        hallmark_sets,
        rownames(v$E),
        remove.empty = TRUE
    )

    cam <- camera(
        v,
        index = indices,
        design = design,
        contrast = contrast
    )

    cam$pathway <- rownames(cam)
    cam$cell_type <- ct


    # --------------------------------------------------------
    # Add pathway-level effect sizes
    #
    # Positive = higher in NR
    # Negative = higher in R
    # --------------------------------------------------------

    effect_rows <- lapply(
        cam$pathway,
        function(pathway) {

            genes <- intersect(
                hallmark_sets[[pathway]],
                de$gene
            )

            z <- de[
                de$gene %in% genes,
                ,
                drop = FALSE
            ]

            if (nrow(z) == 0) {

                return(
                    data.frame(
                        pathway = pathway,
                        mean_logFC = NA_real_,
                        median_logFC = NA_real_,
                        mean_t = NA_real_,
                        stringsAsFactors = FALSE
                    )
                )
            }

            data.frame(
                pathway = pathway,
                mean_logFC = mean(
                    z$logFC,
                    na.rm = TRUE
                ),
                median_logFC = median(
                    z$logFC,
                    na.rm = TRUE
                ),
                mean_t = mean(
                    z$t,
                    na.rm = TRUE
                ),
                stringsAsFactors = FALSE
            )
        }
    )

    effects <- do.call(
        rbind,
        effect_rows
    )

    cam <- merge(
        cam,
        effects,
        by = "pathway",
        all.x = TRUE
    )

    # Positive means pathway shifted toward NR.
    cam$resistance_direction <- ifelse(
        cam$median_logFC > 0,
        "NR",
        "R"
    )

    # Signed enrichment-type score.
    # Positive = resistance-associated
    # Negative = response-associated
    cam$resistance_score <- (
        sign(cam$median_logFC) *
        -log10(
            pmax(
                cam$FDR,
                1e-300
            )
        )
    )

    all_pathways[[ct]] <- cam

    cat(
        "Significant DE genes (FDR < 0.05):",
        sum(de$adj.P.Val < 0.05),
        "\n"
    )

    cat(
        "Significant Hallmark pathways (FDR < 0.05):",
        sum(cam$FDR < 0.05),
        "\n"
    )
}


# ============================================================
# 7. COMBINE RESULTS
# ============================================================

de_all <- do.call(
    rbind,
    all_de
)

pathway_all <- do.call(
    rbind,
    all_pathways
)


# ============================================================
# 8. SAVE FULL RESULTS
# ============================================================

write.csv(
    de_all,
    gzfile(
        file.path(
            OUTDIR,
            "paclitaxel_TNBC_pre_limma_DE.csv.gz"
        )
    ),
    row.names = FALSE
)

write.csv(
    pathway_all,
    file.path(
        OUTDIR,
        "paclitaxel_TNBC_pre_hallmark_pathways.csv"
    ),
    row.names = FALSE
)


# ============================================================
# 9. BUILD CELL-TYPE × PATHWAY MATRIX
# ============================================================

pathway_matrix <- reshape(
    pathway_all[
        ,
        c(
            "cell_type",
            "pathway",
            "resistance_score"
        )
    ],
    idvar = "cell_type",
    timevar = "pathway",
    direction = "wide"
)

names(pathway_matrix) <- sub(
    "^resistance_score\\.",
    "",
    names(pathway_matrix)
)

write.csv(
    pathway_matrix,
    file.path(
        OUTDIR,
        "paclitaxel_TNBC_resistance_pathway_matrix.csv"
    ),
    row.names = FALSE
)


# ============================================================
# 10. TOP PATHWAYS FOR INSPECTION
# ============================================================

top_list <- list()

for (ct in eligible) {

    z <- pathway_all[
        pathway_all$cell_type == ct,
        ,
        drop = FALSE
    ]

    z <- z[
        order(
            z$FDR,
            -abs(z$resistance_score)
        ),
        ,
        drop = FALSE
    ]

    top_list[[ct]] <- head(
        z,
        10
    )
}

top_pathways <- do.call(
    rbind,
    top_list
)

write.csv(
    top_pathways,
    file.path(
        OUTDIR,
        "top10_pathways_per_celltype.csv"
    ),
    row.names = FALSE
)


# ============================================================
# SUMMARY
# ============================================================

cat(
    "\n========================================\n"
)
cat("DONE\n")
cat(
    "========================================\n"
)

cat(
    "Eligible cell types:",
    length(eligible),
    "\n"
)

cat(
    "DE result rows:",
    nrow(de_all),
    "\n"
)

cat(
    "Pathway result rows:",
    nrow(pathway_all),
    "\n"
)

cat(
    "Significant pathways:",
    sum(pathway_all$FDR < 0.05),
    "\n"
)

cat(
    "Output:",
    OUTDIR,
    "\n"
)
