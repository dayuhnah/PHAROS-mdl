suppressPackageStartupMessages({
    library(Matrix)
    library(edgeR)
    library(limma)
})


# ============================================================
# CONFIG
# ============================================================

DATASET_ID <- "GSE169246_TNBC_1"

DRUG <- "Paclitaxel"

DRUGBANK_ID <- "DB01229"

CANCER <- "Triple-negative breast cancer"

EVIDENCE_CONTEXT <- "direct_monotherapy"

PATIENT_QUALITY <- "strong"


DATA_DIR <- file.path(
    "data/processed/v6/cellresdb_expanded",
    DATASET_ID
)

COUNTS_PATH <- file.path(
    DATA_DIR,
    "pseudobulk_counts.rds"
)

META_PATH <- file.path(
    DATA_DIR,
    "pseudobulk_metadata.csv"
)

HALLMARK_PATH <- file.path(
    "data/processed/v6/cellresdb_expanded",
    "hallmark_reference",
    "hallmark_sets_v5_compatible.rds"
)

OUT_DIR <- file.path(
    DATA_DIR,
    "limma_camera"
)


MIN_PATIENTS_PER_CLASS <- 3


dir.create(
    OUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)


# ============================================================
# LOAD
# ============================================================

cat("============================================================\n")
cat("PHAROS V6 — TNBC LIMMA-VOOM + CAMERA\n")
cat("============================================================\n\n")


counts <- readRDS(
    COUNTS_PATH
)


meta <- read.csv(
    META_PATH,
    stringsAsFactors = FALSE,
    check.names = FALSE
)


hallmark_sets <- readRDS(
    HALLMARK_PATH
)


cat(
    "Genes:",
    nrow(counts),
    "\n"
)

cat(
    "Pseudobulks:",
    ncol(counts),
    "\n"
)

cat(
    "Hallmark pathways:",
    length(hallmark_sets),
    "\n"
)


# ============================================================
# VALIDATION
# ============================================================

if (
    ncol(counts) != nrow(meta)
) {
    stop(
        "Pseudobulk matrix columns do not match metadata rows."
    )
}


if (
    !identical(
        as.character(
            colnames(counts)
        ),
        as.character(
            meta$group_id
        )
    )
) {
    stop(
        "Pseudobulk counts are not aligned with metadata."
    )
}


if (
    is.null(
        rownames(counts)
    )
) {
    stop(
        "Pseudobulk matrix has no gene names."
    )
}


if (
    length(hallmark_sets) != 50
) {
    stop(
        paste(
            "Expected 50 Hallmark pathways, got",
            length(hallmark_sets)
        )
    )
}


if (
    anyDuplicated(
        names(hallmark_sets)
    )
) {
    stop(
        "Hallmark pathway names are duplicated."
    )
}


expected_pathways <- names(
    hallmark_sets
)


cat(
    "Counts/metadata alignment: TRUE ✅\n"
)

cat(
    "50-pathway Hallmark reference: TRUE ✅\n"
)


# ============================================================
# HELPERS
# ============================================================

sanitize_token <- function(x) {

    x <- gsub(
        "[^A-Za-z0-9]+",
        "_",
        x
    )

    x <- gsub(
        "^_+|_+$",
        "",
        x
    )

    x
}


make_token_id <- function(
    cell_type,
    signature_type
) {

    paste(
        DATASET_ID,
        sanitize_token(DRUG),
        sanitize_token(cell_type),
        sanitize_token(signature_type),
        sep = "__"
    )
}


# ------------------------------------------------------------
# Convert Hallmark gene symbols to row indices
# ------------------------------------------------------------

make_indices <- function(
    genes
) {

    idx <- ids2indices(
        hallmark_sets,
        genes
    )


    # We want all 50 pathways represented.

    missing <- setdiff(
        expected_pathways,
        names(idx)
    )


    if (
        length(missing) > 0
    ) {

        stop(
            paste(
                "Hallmark index missing pathways:",
                paste(
                    missing,
                    collapse = ", "
                )
            )
        )
    }


    idx <- idx[
        expected_pathways
    ]


    empty <- names(idx)[
        lengths(idx) == 0
    ]


    if (
        length(empty) > 0
    ) {

        stop(
            paste(
                "Hallmark pathways with zero genes:",
                paste(
                    empty,
                    collapse = ", "
                )
            )
        )
    }


    idx
}


# ------------------------------------------------------------
# Add model-facing pathway quantities
# ------------------------------------------------------------

decorate_camera <- function(
    camera_result,
    gene_effect,
    pathway_indices,
    token_id,
    cell_type,
    signature_type,
    nr_patients,
    r_patients,
    analysis_method
) {

    cam <- as.data.frame(
        camera_result
    )


    cam$pathway <- rownames(
        cam
    )


    rownames(
        cam
    ) <- NULL


    # CAMERA should return exactly the frozen 50 pathways.

    missing <- setdiff(
        expected_pathways,
        cam$pathway
    )


    if (
        length(missing) > 0
    ) {

        stop(
            paste(
                "CAMERA did not return all Hallmark pathways:",
                paste(
                    missing,
                    collapse = ", "
                )
            )
        )
    }


    cam <- cam[
        match(
            expected_pathways,
            cam$pathway
        ),
        ,
        drop = FALSE
    ]


    if (
        any(
            is.na(
                cam$pathway
            )
        )
    ) {

        stop(
            "Failed to reorder CAMERA results."
        )
    }


    # Mean gene-level contrast inside each pathway.
    # This is descriptive effect direction/magnitude,
    # separate from CAMERA's competitive test.

    pathway_mean_effect <- vapply(
        expected_pathways,
        function(pathway) {

            idx <- pathway_indices[[
                pathway
            ]]

            mean(
                gene_effect[
                    idx
                ],
                na.rm = TRUE
            )
        },
        numeric(1)
    )


    pathway_median_effect <- vapply(
        expected_pathways,
        function(pathway) {

            idx <- pathway_indices[[
                pathway
            ]]

            median(
                gene_effect[
                    idx
                ],
                na.rm = TRUE
            )
        },
        numeric(1)
    )


    cam$mean_gene_effect <- (
        pathway_mean_effect[
            cam$pathway
        ]
    )


    cam$median_gene_effect <- (
        pathway_median_effect[
            cam$pathway
        ]
    )


    # Direction is relative to NR - R.
    #
    # CAMERA:
    # Up   -> pathway tends higher in NR
    # Down -> pathway tends higher in R

    cam$HigherIn <- ifelse(
        cam$Direction == "Up",
        "NR",
        "R"
    )


    # Same general signed-score interpretation used
    # by the previous PHAROS CellResDB bank:
    #
    # sign(Direction) * -log10(FDR)

    safe_fdr <- pmax(
        cam$FDR,
        1e-300
    )


    direction_sign <- ifelse(
        cam$Direction == "Up",
        1,
        -1
    )


    cam$signed_score <- (
        direction_sign
        *
        -log10(
            safe_fdr
        )
    )


    cam$token_id <- token_id

    cam$dataset_id <- DATASET_ID

    cam$drug <- DRUG

    cam$drugbank_id <- DRUGBANK_ID

    cam$cancer <- CANCER

    cam$cell_type <- cell_type

    cam$signature_type <- signature_type

    cam$NR_patients <- nr_patients

    cam$R_patients <- r_patients

    cam$analysis_method <- analysis_method

    cam$evidence_context <- EVIDENCE_CONTEXT

    cam$patient_quality <- PATIENT_QUALITY


    cam
}


# ============================================================
# CROSS-SECTIONAL ANALYSIS
#
# Baseline:
#   NR_Pre - R_Pre
#
# Post:
#   NR_Post - R_Post
# ============================================================

analyze_cross_sectional <- function(
    cell_type,
    treatment
) {

    sub_meta <- meta[
        meta$cluster1 == cell_type
        &
        meta$treatment == treatment,
        ,
        drop = FALSE
    ]


    if (
        nrow(sub_meta) == 0
    ) {
        return(NULL)
    }


    nr_patients <- unique(
        sub_meta$patient[
            sub_meta$Response == "NR"
        ]
    )


    r_patients <- unique(
        sub_meta$patient[
            sub_meta$Response == "R"
        ]
    )


    n_nr <- length(
        nr_patients
    )

    n_r <- length(
        r_patients
    )


    if (
        n_nr < MIN_PATIENTS_PER_CLASS
        ||
        n_r < MIN_PATIENTS_PER_CLASS
    ) {

        return(NULL)
    }


    idx <- match(
        sub_meta$group_id,
        colnames(counts)
    )


    if (
        anyNA(idx)
    ) {

        stop(
            paste(
                "Failed to map pseudobulks for",
                cell_type,
                treatment
            )
        )
    }


    y_counts <- counts[
        ,
        idx,
        drop = FALSE
    ]


    # --------------------------------------------------------
    # RESPONSE FACTOR
    # R is reference, contrast is NR - R
    # --------------------------------------------------------

    group <- factor(
        sub_meta$Response,
        levels = c(
            "R",
            "NR"
        )
    )


    design <- model.matrix(
        ~ 0 + group
    )


    colnames(
        design
    ) <- c(
        "R",
        "NR"
    )


    contrast <- makeContrasts(
        NR_minus_R = NR - R,
        levels = design
    )


    # --------------------------------------------------------
    # EDGER + VOOM
    # --------------------------------------------------------

    dge <- DGEList(
        counts = y_counts
    )


    keep_genes <- filterByExpr(
        dge,
        group = group
    )


    cat(
        "  Genes retained:",
        sum(keep_genes),
        "/",
        length(keep_genes),
        "\n"
    )


    if (
        sum(keep_genes) < 1000
    ) {

        stop(
            paste(
                "Too few expressed genes retained for",
                cell_type,
                treatment
            )
        )
    }


    dge <- dge[
        keep_genes,
        ,
        keep.lib.sizes = FALSE
    ]


    dge <- calcNormFactors(
        dge,
        method = "TMM"
    )


    v <- voom(
        dge,
        design,
        plot = FALSE
    )


    # --------------------------------------------------------
    # LIMMA
    # --------------------------------------------------------

    fit <- lmFit(
        v,
        design
    )


    fit2 <- contrasts.fit(
        fit,
        contrast
    )


    fit2 <- eBayes(
        fit2
    )


    gene_table <- topTable(
        fit2,
        coef = 1,
        number = Inf,
        sort.by = "none"
    )


    gene_table$gene <- rownames(
        gene_table
    )


    gene_table$cell_type <- cell_type


    signature_type <- if (
        treatment == "Pre"
    ) {
        "baseline_NR_minus_R"
    } else {
        "post_NR_minus_R"
    }


    token_id <- make_token_id(
        cell_type,
        signature_type
    )


    gene_table$signature_type <- (
        signature_type
    )


    gene_table$token_id <- token_id

    gene_table$NR_patients <- n_nr

    gene_table$R_patients <- n_r


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    hallmark_indices <- make_indices(
        rownames(v)
    )


    cam <- camera(
        v,
        index = hallmark_indices,
        design = design,
        contrast = contrast[, 1]
    )


    pathway_table <- decorate_camera(
        camera_result = cam,
        gene_effect = fit2$coefficients[, 1],
        pathway_indices = hallmark_indices,
        token_id = token_id,
        cell_type = cell_type,
        signature_type = signature_type,
        nr_patients = n_nr,
        r_patients = n_r,
        analysis_method = "limma_voom_camera"
    )


    list(
        gene_table = gene_table,
        pathway_table = pathway_table
    )
}


# ============================================================
# PAIRED ADAPTATION ANALYSIS
#
# Patient-level:
#
# delta_i = Post_i - Pre_i
#
# Then:
#
# mean(delta_NR) - mean(delta_R)
# ============================================================

analyze_adaptation <- function(
    cell_type
) {

    sub_meta <- meta[
        meta$cluster1 == cell_type,
        ,
        drop = FALSE
    ]


    if (
        nrow(sub_meta) == 0
    ) {
        return(NULL)
    }


    patients <- unique(
        sub_meta$patient
    )


    paired_patients <- character()


    for (
        patient in patients
    ) {

        p <- sub_meta[
            sub_meta$patient == patient,
            ,
            drop = FALSE
        ]


        has_pre <- any(
            p$treatment == "Pre"
        )


        has_post <- any(
            p$treatment == "Post"
        )


        if (
            has_pre
            &&
            has_post
        ) {

            paired_patients <- c(
                paired_patients,
                patient
            )
        }
    }


    paired_meta <- sub_meta[
        sub_meta$patient
        %in%
        paired_patients,
        ,
        drop = FALSE
    ]


    if (
        nrow(paired_meta) == 0
    ) {
        return(NULL)
    }


    patient_response <- unique(
        paired_meta[
            ,
            c(
                "patient",
                "Response"
            )
        ]
    )


    nr_patients <- patient_response$patient[
        patient_response$Response == "NR"
    ]


    r_patients <- patient_response$patient[
        patient_response$Response == "R"
    ]


    n_nr <- length(
        nr_patients
    )


    n_r <- length(
        r_patients
    )


    if (
        n_nr < MIN_PATIENTS_PER_CLASS
        ||
        n_r < MIN_PATIENTS_PER_CLASS
    ) {

        return(NULL)
    }


    # --------------------------------------------------------
    # STRICTLY REQUIRE ONE PRE + ONE POST PSEUDOBULK
    # PER PAIRED PATIENT FOR THIS CELL TYPE
    # --------------------------------------------------------

    valid_patients <- character()


    for (
        patient in paired_patients
    ) {

        p <- paired_meta[
            paired_meta$patient == patient,
            ,
            drop = FALSE
        ]


        n_pre <- sum(
            p$treatment == "Pre"
        )


        n_post <- sum(
            p$treatment == "Post"
        )


        if (
            n_pre == 1
            &&
            n_post == 1
        ) {

            valid_patients <- c(
                valid_patients,
                patient
            )
        }
    }


    paired_meta <- paired_meta[
        paired_meta$patient
        %in%
        valid_patients,
        ,
        drop = FALSE
    ]


    patient_response <- unique(
        paired_meta[
            ,
            c(
                "patient",
                "Response"
            )
        ]
    )


    nr_patients <- patient_response$patient[
        patient_response$Response == "NR"
    ]


    r_patients <- patient_response$patient[
        patient_response$Response == "R"
    ]


    n_nr <- length(
        nr_patients
    )


    n_r <- length(
        r_patients
    )


    if (
        n_nr < MIN_PATIENTS_PER_CLASS
        ||
        n_r < MIN_PATIENTS_PER_CLASS
    ) {

        return(NULL)
    }


    # --------------------------------------------------------
    # LOAD PAIRED RAW COUNTS
    # --------------------------------------------------------

    idx <- match(
        paired_meta$group_id,
        colnames(counts)
    )


    if (
        anyNA(idx)
    ) {

        stop(
            paste(
                "Failed to map paired pseudobulks for",
                cell_type
            )
        )
    }


    y_counts <- counts[
        ,
        idx,
        drop = FALSE
    ]


    # Four biological groups are useful for
    # expression filtering / voom modelling.

    four_group <- interaction(
        paired_meta$Response,
        paired_meta$treatment,
        drop = TRUE
    )


    four_group <- factor(
        four_group
    )


    design_voom <- model.matrix(
        ~ 0 + four_group
    )


    # --------------------------------------------------------
    # EDGER + VOOM NORMALIZATION
    # --------------------------------------------------------

    dge <- DGEList(
        counts = y_counts
    )


    keep_genes <- filterByExpr(
        dge,
        group = four_group
    )


    cat(
        "  Paired genes retained:",
        sum(keep_genes),
        "/",
        length(keep_genes),
        "\n"
    )


    if (
        sum(keep_genes) < 1000
    ) {

        stop(
            paste(
                "Too few expressed paired genes retained for",
                cell_type
            )
        )
    }


    dge <- dge[
        keep_genes,
        ,
        keep.lib.sizes = FALSE
    ]


    dge <- calcNormFactors(
        dge,
        method = "TMM"
    )


    v <- voom(
        dge,
        design_voom,
        plot = FALSE
    )


    # ========================================================
    # BUILD WITHIN-PATIENT DELTA MATRIX
    #
    # Each column becomes ONE patient.
    # ========================================================

    delta_columns <- list()

    delta_response <- character()


    for (
        patient in valid_patients
    ) {

        p <- paired_meta[
            paired_meta$patient == patient,
            ,
            drop = FALSE
        ]


        pre_id <- p$group_id[
            p$treatment == "Pre"
        ]


        post_id <- p$group_id[
            p$treatment == "Post"
        ]


        pre_idx <- match(
            pre_id,
            colnames(v$E)
        )


        post_idx <- match(
            post_id,
            colnames(v$E)
        )


        if (
            is.na(pre_idx)
            ||
            is.na(post_idx)
        ) {

            stop(
                paste(
                    "Could not construct paired delta for",
                    patient,
                    cell_type
                )
            )
        }


        delta_columns[[
            patient
        ]] <- (
            v$E[
                ,
                post_idx
            ]
            -
            v$E[
                ,
                pre_idx
            ]
        )


        response_value <- unique(
            p$Response
        )


        if (
            length(response_value) != 1
        ) {

            stop(
                paste(
                    "Patient has inconsistent response labels:",
                    patient
                )
            )
        }


        delta_response[
            patient
        ] <- response_value
    }


    delta_matrix <- do.call(
        cbind,
        delta_columns
    )


    rownames(
        delta_matrix
    ) <- rownames(
        v$E
    )


    # --------------------------------------------------------
    # Ensure columns are patient IDs.
    # --------------------------------------------------------

    colnames(
        delta_matrix
    ) <- names(
        delta_columns
    )


    response_factor <- factor(
        delta_response[
            colnames(
                delta_matrix
            )
        ],
        levels = c(
            "R",
            "NR"
        )
    )


    design_delta <- model.matrix(
        ~ 0 + response_factor
    )


    colnames(
        design_delta
    ) <- c(
        "R",
        "NR"
    )


    contrast <- makeContrasts(
        NR_minus_R = NR - R,
        levels = design_delta
    )


    # --------------------------------------------------------
    # LIMMA ON WITHIN-PATIENT DELTAS
    # --------------------------------------------------------

    fit <- lmFit(
        delta_matrix,
        design_delta
    )


    fit2 <- contrasts.fit(
        fit,
        contrast
    )


    # Deltas are already voom-normalized log-expression.
    # trend=TRUE allows empirical Bayes variance modelling
    # to account for residual mean-variance structure.

    fit2 <- eBayes(
        fit2,
        trend = TRUE
    )


    gene_table <- topTable(
        fit2,
        coef = 1,
        number = Inf,
        sort.by = "none"
    )


    gene_table$gene <- rownames(
        gene_table
    )


    signature_type <- (
        "adaptation_delta_NR_minus_R"
    )


    token_id <- make_token_id(
        cell_type,
        signature_type
    )


    gene_table$cell_type <- cell_type

    gene_table$signature_type <- signature_type

    gene_table$token_id <- token_id

    gene_table$NR_patients <- n_nr

    gene_table$R_patients <- n_r


    # --------------------------------------------------------
    # CAMERA ON PATIENT-LEVEL DELTA MATRIX
    # --------------------------------------------------------

    hallmark_indices <- make_indices(
        rownames(
            delta_matrix
        )
    )


    cam <- camera(
        delta_matrix,
        index = hallmark_indices,
        design = design_delta,
        contrast = contrast[, 1]
    )


    pathway_table <- decorate_camera(
        camera_result = cam,
        gene_effect = fit2$coefficients[, 1],
        pathway_indices = hallmark_indices,
        token_id = token_id,
        cell_type = cell_type,
        signature_type = signature_type,
        nr_patients = n_nr,
        r_patients = n_r,
        analysis_method = "paired_voom_delta_limma_camera"
    )


    list(
        gene_table = gene_table,
        pathway_table = pathway_table
    )
}


# ============================================================
# RUN ALL CELL TYPES
# ============================================================

cell_types <- sort(
    unique(
        meta$cluster1
    )
)


all_gene_results <- list()

all_pathway_results <- list()

summary_records <- list()


for (
    cell_type in cell_types
) {

    cat("\n============================================================\n")
    cat("CELL TYPE:", cell_type, "\n")
    cat("============================================================\n")


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    cat("\nBaseline NR - R\n")


    baseline <- analyze_cross_sectional(
        cell_type,
        "Pre"
    )


    if (
        !is.null(baseline)
    ) {

        key <- make_token_id(
            cell_type,
            "baseline_NR_minus_R"
        )


        all_gene_results[[
            key
        ]] <- baseline$gene_table


        all_pathway_results[[
            key
        ]] <- baseline$pathway_table


        summary_records[[
            length(summary_records) + 1
        ]] <- data.frame(
            token_id = key,
            cell_type = cell_type,
            signature_type =
                "baseline_NR_minus_R",
            NR_patients =
                baseline$pathway_table$NR_patients[1],
            R_patients =
                baseline$pathway_table$R_patients[1],
            stringsAsFactors = FALSE
        )
    } else {

        cat(
            "  Not eligible\n"
        )
    }


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    cat("\nPost-treatment NR - R\n")


    post <- analyze_cross_sectional(
        cell_type,
        "Post"
    )


    if (
        !is.null(post)
    ) {

        key <- make_token_id(
            cell_type,
            "post_NR_minus_R"
        )


        all_gene_results[[
            key
        ]] <- post$gene_table


        all_pathway_results[[
            key
        ]] <- post$pathway_table


        summary_records[[
            length(summary_records) + 1
        ]] <- data.frame(
            token_id = key,
            cell_type = cell_type,
            signature_type =
                "post_NR_minus_R",
            NR_patients =
                post$pathway_table$NR_patients[1],
            R_patients =
                post$pathway_table$R_patients[1],
            stringsAsFactors = FALSE
        )
    } else {

        cat(
            "  Not eligible\n"
        )
    }


    # --------------------------------------------------------
    # PAIRED ADAPTATION
    # --------------------------------------------------------

    cat("\nPaired treatment adaptation\n")


    adaptation <- analyze_adaptation(
        cell_type
    )


    if (
        !is.null(adaptation)
    ) {

        key <- make_token_id(
            cell_type,
            "adaptation_delta_NR_minus_R"
        )


        all_gene_results[[
            key
        ]] <- adaptation$gene_table


        all_pathway_results[[
            key
        ]] <- adaptation$pathway_table


        summary_records[[
            length(summary_records) + 1
        ]] <- data.frame(
            token_id = key,
            cell_type = cell_type,
            signature_type =
                "adaptation_delta_NR_minus_R",
            NR_patients =
                adaptation$pathway_table$NR_patients[1],
            R_patients =
                adaptation$pathway_table$R_patients[1],
            stringsAsFactors = FALSE
        )
    } else {

        cat(
            "  Not eligible\n"
        )
    }
}


# ============================================================
# COMBINE
# ============================================================

if (
    length(all_pathway_results) == 0
) {

    stop(
        "No eligible CAMERA analyses were generated."
    )
}


pathway_results <- do.call(
    rbind,
    all_pathway_results
)


rownames(
    pathway_results
) <- NULL


gene_results <- do.call(
    rbind,
    all_gene_results
)


rownames(
    gene_results
) <- NULL


analysis_summary <- do.call(
    rbind,
    summary_records
)


rownames(
    analysis_summary
) <- NULL


# ============================================================
# VALIDATE 50 PATHWAYS PER TOKEN
# ============================================================

pathway_counts <- table(
    pathway_results$token_id
)


if (
    any(
        pathway_counts != 50
    )
) {

    print(
        pathway_counts
    )

    stop(
        "At least one token does not contain exactly 50 Hallmark pathways."
    )
}


cat(
    "\nAll tokens contain exactly 50 Hallmark pathways ✅\n"
)


# ============================================================
# BUILD MODEL-FACING 50-D TOKEN MATRIX
# ============================================================

token_ids <- unique(
    pathway_results$token_id
)


token_matrix <- matrix(
    NA_real_,
    nrow = length(token_ids),
    ncol = length(expected_pathways),
    dimnames = list(
        token_ids,
        expected_pathways
    )
)


for (
    token_id in token_ids
) {

    g <- pathway_results[
        pathway_results$token_id == token_id,
        ,
        drop = FALSE
    ]


    g <- g[
        match(
            expected_pathways,
            g$pathway
        ),
        ,
        drop = FALSE
    ]


    token_matrix[
        token_id,
        expected_pathways
    ] <- g$signed_score
}


if (
    anyNA(
        token_matrix
    )
) {

    stop(
        "Model-facing Hallmark token matrix contains NA values."
    )
}


token_matrix_df <- data.frame(
    token_id = rownames(
        token_matrix
    ),
    token_matrix,
    check.names = FALSE,
    row.names = NULL
)


# ============================================================
# TOKEN METADATA
# ============================================================

token_metadata <- unique(
    pathway_results[
        ,
        c(
            "token_id",
            "dataset_id",
            "drug",
            "drugbank_id",
            "cancer",
            "cell_type",
            "signature_type",
            "NR_patients",
            "R_patients",
            "analysis_method",
            "evidence_context",
            "patient_quality"
        )
    ]
)


# ============================================================
# TOP 10 PATHWAYS PER TOKEN
# ============================================================

top10_list <- list()


for (
    token_id in token_ids
) {

    g <- pathway_results[
        pathway_results$token_id == token_id,
        ,
        drop = FALSE
    ]


    g <- g[
        order(
            abs(
                g$signed_score
            ),
            decreasing = TRUE
        ),
        ,
        drop = FALSE
    ]


    top10_list[[
        token_id
    ]] <- head(
        g,
        10
    )
}


top10 <- do.call(
    rbind,
    top10_list
)


rownames(
    top10
) <- NULL


# ============================================================
# SAVE
# ============================================================

GENE_PATH <- file.path(
    OUT_DIR,
    "gene_level_limma_results.csv.gz"
)


PATHWAY_PATH <- file.path(
    OUT_DIR,
    "hallmark_camera_results.csv"
)


TOKEN_PATH <- file.path(
    OUT_DIR,
    "hallmark_token_matrix.csv"
)


TOKEN_META_PATH <- file.path(
    OUT_DIR,
    "token_metadata.csv"
)


TOP10_PATH <- file.path(
    OUT_DIR,
    "top10_pathways_per_token.csv"
)


SUMMARY_PATH <- file.path(
    OUT_DIR,
    "analysis_summary.csv"
)


write.csv(
    gene_results,
    gzfile(
        GENE_PATH
    ),
    row.names = FALSE
)


write.csv(
    pathway_results,
    PATHWAY_PATH,
    row.names = FALSE
)


write.csv(
    token_matrix_df,
    TOKEN_PATH,
    row.names = FALSE
)


write.csv(
    token_metadata,
    TOKEN_META_PATH,
    row.names = FALSE
)


write.csv(
    top10,
    TOP10_PATH,
    row.names = FALSE
)


write.csv(
    analysis_summary,
    SUMMARY_PATH,
    row.names = FALSE
)


# ============================================================
# FINAL AUDIT
# ============================================================

cat("\n============================================================\n")
cat("FINAL CAMERA AUDIT\n")
cat("============================================================\n")


cat(
    "\nEligible tokens:",
    nrow(
        token_metadata
    ),
    "\n"
)


cat(
    "Token matrix dimensions:",
    paste(
        dim(
            token_matrix
        ),
        collapse = " x "
    ),
    "\n"
)


cat(
    "\nTokens by signature type:\n"
)


print(
    table(
        token_metadata$signature_type
    )
)


cat(
    "\nPatient counts per token:\n"
)


print(
    token_metadata[
        order(
            token_metadata$signature_type,
            token_metadata$cell_type
        ),
        c(
            "cell_type",
            "signature_type",
            "NR_patients",
            "R_patients"
        )
    ],
    row.names = FALSE
)


cat(
    "\nSignificant Hallmark pathways per token (FDR < 0.05):\n"
)


sig_counts <- aggregate(
    FDR ~ token_id,
    data = pathway_results,
    FUN = function(x) {
        sum(
            x < 0.05,
            na.rm = TRUE
        )
    }
)


names(
    sig_counts
)[2] <- "significant_pathways"


print(
    sig_counts,
    row.names = FALSE
)


cat("\n============================================================\n")
cat("TOP PATHWAYS — MAJOR CELL TYPES\n")
cat("============================================================\n")


major_celltypes <- c(
    "B Cell",
    "CD4+ T Cell",
    "CD8+ T Cell",
    "Monocyte/Macrophage"
)


signature_types <- c(
    "baseline_NR_minus_R",
    "post_NR_minus_R",
    "adaptation_delta_NR_minus_R"
)


for (
    cell_type in major_celltypes
) {

    cat(
        "\n###",
        cell_type,
        "###\n"
    )


    for (
        signature_type in signature_types
    ) {

        g <- pathway_results[
            pathway_results$cell_type
            ==
            cell_type
            &
            pathway_results$signature_type
            ==
            signature_type,
            ,
            drop = FALSE
        ]


        if (
            nrow(g) == 0
        ) {
            next
        }


        g <- g[
            order(
                abs(
                    g$signed_score
                ),
                decreasing = TRUE
            ),
            ,
            drop = FALSE
        ]


        cat(
            "\n",
            signature_type,
            "\n",
            sep = ""
        )


        print(
            head(
                g[
                    ,
                    c(
                        "pathway",
                        "NGenes",
                        "Direction",
                        "PValue",
                        "FDR",
                        "mean_gene_effect",
                        "HigherIn",
                        "signed_score"
                    )
                ],
                10
            ),
            row.names = FALSE
        )
    }
}


cat("\n============================================================\n")
cat("SAVED\n")
cat("============================================================\n")


cat(
    GENE_PATH,
    "\n"
)

cat(
    PATHWAY_PATH,
    "\n"
)

cat(
    TOKEN_PATH,
    "\n"
)

cat(
    TOKEN_META_PATH,
    "\n"
)

cat(
    TOP10_PATH,
    "\n"
)

cat(
    SUMMARY_PATH,
    "\n"
)


cat("\nDONE ✅\n")
