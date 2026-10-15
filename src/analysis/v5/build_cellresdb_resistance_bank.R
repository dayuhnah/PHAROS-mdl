library(SeuratObject)
library(Matrix)
library(edgeR)
library(limma)
library(msigdbr)

# ============================================================
# CONFIG
# ============================================================

RDS_DIR <- "data/raw/cellresdb/rds"

MANIFEST_FILE <- paste0(
    "data/processed/v5/cellresdb/",
    "cellresdb_dataset_manifest.csv"
)

OUTDIR <- paste0(
    "data/processed/v5/cellresdb/",
    "resistance_bank"
)

MIN_CELLS <- 20
MIN_PATIENTS <- 5
SCORE_CAP <- 10

dir.create(
    OUTDIR,
    recursive = TRUE,
    showWarnings = FALSE
)

# ============================================================
# LOAD MANIFEST
# ============================================================

manifest <- read.csv(
    MANIFEST_FILE,
    stringsAsFactors = FALSE
)

# ============================================================
# LOAD HALLMARK SETS ONCE
# ============================================================

cat("Loading MSigDB Hallmark pathways...\n")

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
# INPUT FILES
# ============================================================

files <- list.files(
    RDS_DIR,
    pattern = "\\.rds$",
    full.names = TRUE
)

cat(
    "\nDownloaded datasets:",
    length(files),
    "\n"
)

# ============================================================
# STORAGE
# ============================================================

bank_list <- list()
de_list <- list()
dataset_summary <- list()
coverage_list <- list()

# ============================================================
# LOOP DATASETS
# ============================================================

for (file in files) {

    dataset_id <- sub(
        "\\.rds$",
        "",
        basename(file)
    )

    cat(
        "\n============================================================\n"
    )
    cat("DATASET:", dataset_id, "\n")
    cat(
        "============================================================\n"
    )

    # --------------------------------------------------------
    # Manifest information
    # --------------------------------------------------------

    m <- manifest[
        manifest$database_id == dataset_id,
        ,
        drop = FALSE
    ]

    if (nrow(m) > 0) {

        cancer_type <- m$cancer_type[1]
        therapy_type <- m$therapy_type[1]
        regimen <- m$therapeutic_regimen[1]
        matched_drugs <- m$matched_pharos_drugs[1]
        pmid <- m$pmid[1]

    } else {

        cancer_type <- NA
        therapy_type <- NA
        regimen <- NA
        matched_drugs <- NA
        pmid <- NA
    }

    evidence_scope <- ifelse(
        !is.na(regimen) &&
        grepl("\\+", regimen),
        "combination_regimen",
        "monotherapy"
    )

    # --------------------------------------------------------
    # Load object
    # --------------------------------------------------------

    cat("Loading RDS...\n")

    obj <- readRDS(file)
    meta <- obj[[]]

    required <- c(
        "patient",
        "treatment",
        "Response",
        "cluster1"
    )

    missing_cols <- setdiff(
        required,
        colnames(meta)
    )

    if (length(missing_cols) > 0) {

        cat(
            "SKIP: missing metadata:",
            paste(missing_cols, collapse = ", "),
            "\n"
        )

        dataset_summary[[dataset_id]] <- data.frame(
            dataset_id = dataset_id,
            status = "missing_metadata",
            reason = paste(
                missing_cols,
                collapse = ";"
            )
        )

        rm(obj)
        gc()

        next
    }

    meta <- meta[
        colnames(obj),
        ,
        drop = FALSE
    ]

    meta$treatment_clean <- tolower(
        trimws(
            as.character(
                meta$treatment
            )
        )
    )

    meta$response_clean <- toupper(
        trimws(
            as.character(
                meta$Response
            )
        )
    )

    # --------------------------------------------------------
    # PRE-TREATMENT R / NR
    # --------------------------------------------------------

    keep <- (
        meta$treatment_clean == "pre" &
        meta$response_clean %in% c(
            "R",
            "NR"
        )
    )

    meta_pre <- meta[
        keep,
        ,
        drop = FALSE
    ]

    n_pre_cells <- nrow(meta_pre)

    n_patients_R <- length(
        unique(
            meta_pre$patient[
                meta_pre$response_clean == "R"
            ]
        )
    )

    n_patients_NR <- length(
        unique(
            meta_pre$patient[
                meta_pre$response_clean == "NR"
            ]
        )
    )

    cat(
        "Pre-treatment cells:",
        n_pre_cells,
        "\n"
    )

    cat(
        "Patients R:",
        n_patients_R,
        "| NR:",
        n_patients_NR,
        "\n"
    )

    if (
        n_patients_R < MIN_PATIENTS ||
        n_patients_NR < MIN_PATIENTS
    ) {

        cat(
            "SKIP: insufficient overall R/NR patients\n"
        )

        dataset_summary[[dataset_id]] <- data.frame(
            dataset_id = dataset_id,
            cancer_type = cancer_type,
            regimen = regimen,
            status = "insufficient_patients",
            pre_cells = n_pre_cells,
            n_R = n_patients_R,
            n_NR = n_patients_NR
        )

        rm(obj)
        gc()

        next
    }

    # --------------------------------------------------------
    # RAW COUNTS
    # --------------------------------------------------------

    counts <- LayerData(
        obj[["RNA"]],
        layer = "counts"
    )

    counts <- counts[
        ,
        rownames(meta_pre),
        drop = FALSE
    ]

    # --------------------------------------------------------
    # PATIENT × CELL TYPE PSEUDOBULK
    # --------------------------------------------------------

    group_key <- paste(
        meta_pre$patient,
        meta_pre$response_clean,
        meta_pre$cluster1,
        sep = "|||"
    )

    group_factor <- factor(
        group_key
    )

    group_meta <- unique(
        data.frame(
            group_key = group_key,
            patient = meta_pre$patient,
            response = meta_pre$response_clean,
            cell_type = meta_pre$cluster1,
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

    cat("Building pseudobulk matrix...\n")

    pb_counts <- counts %*% G

    group_meta$n_cells <- tabulate(
        as.integer(group_factor),
        nbins = nlevels(group_factor)
    )

    valid <- (
        group_meta$n_cells >= MIN_CELLS
    )

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
        "Pseudobulk groups:",
        ncol(pb_counts),
        "\n"
    )

    # --------------------------------------------------------
    # CELL TYPE COVERAGE
    # --------------------------------------------------------

    cov_rows <- list()

    for (
        ct in sort(
            unique(
                group_meta$cell_type
            )
        )
    ) {

        z <- group_meta[
            group_meta$cell_type == ct,
            ,
            drop = FALSE
        ]

        cov_rows[[length(cov_rows) + 1]] <-
            data.frame(
                dataset_id = dataset_id,
                cell_type = ct,
                n_R = sum(
                    z$response == "R"
                ),
                n_NR = sum(
                    z$response == "NR"
                ),
                stringsAsFactors = FALSE
            )
    }

    coverage <- do.call(
        rbind,
        cov_rows
    )

    coverage$eligible <- (
        coverage$n_R >= MIN_PATIENTS &
        coverage$n_NR >= MIN_PATIENTS
    )

    coverage_list[[dataset_id]] <- coverage

    eligible <- coverage$cell_type[
        coverage$eligible
    ]

    cat(
        "Eligible cell types:",
        length(eligible),
        "\n"
    )

    print(eligible)

    if (length(eligible) == 0) {

        dataset_summary[[dataset_id]] <- data.frame(
            dataset_id = dataset_id,
            cancer_type = cancer_type,
            regimen = regimen,
            status = "no_eligible_celltypes",
            pre_cells = n_pre_cells,
            n_R = n_patients_R,
            n_NR = n_patients_NR
        )

        rm(
            obj,
            counts,
            pb_counts
        )

        gc()

        next
    }

    # --------------------------------------------------------
    # PROCESS EACH CELL TYPE
    # --------------------------------------------------------

    for (ct in eligible) {

        cat(
            "\n  Cell type:",
            ct,
            "\n"
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

        n_R <- sum(
            gm$response == "R"
        )

        n_NR <- sum(
            gm$response == "NR"
        )

        response <- factor(
            gm$response,
            levels = c(
                "R",
                "NR"
            )
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

        # ----------------------------------------------------
        # edgeR
        # ----------------------------------------------------

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
            "    Genes:",
            nrow(y),
            "\n"
        )

        # ----------------------------------------------------
        # limma-voom
        # ----------------------------------------------------

        v <- voom(
            y,
            design,
            plot = FALSE
        )

        fit <- lmFit(
            v,
            design
        )

        fit <- contrasts.fit(
            fit,
            contrast
        )

        fit <- eBayes(
            fit,
            robust = TRUE
        )

        de <- topTable(
            fit,
            coef = "NR_minus_R",
            number = Inf,
            sort.by = "P"
        )

        de$gene <- rownames(de)

        de$dataset_id <- dataset_id
        de$cell_type <- ct

        de_list[[
            paste(
                dataset_id,
                ct,
                sep = "|||"
            )
        ]] <- de

        # ----------------------------------------------------
        # CAMERA HALLMARK
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PATHWAY EFFECT SIZE
        # ----------------------------------------------------

        effects <- lapply(
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
                            median_logFC = NA_real_
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
                    )
                )
            }
        )

        effects <- do.call(
            rbind,
            effects
        )

        cam <- merge(
            cam,
            effects,
            by = "pathway",
            all.x = TRUE
        )

        # ----------------------------------------------------
        # STANDARDIZED RESISTANCE SCORE
        #
        # NR - R contrast:
        #
        # Up   = stronger in non-responders
        #      = resistance-associated
        #
        # Down = stronger in responders
        #      = response-associated
        # ----------------------------------------------------

        direction_sign <- ifelse(
            cam$Direction == "Up",
            1,
            -1
        )

        confidence <- -log10(
            pmax(
                cam$FDR,
                1e-300
            )
        )

        cam$confidence <- confidence

        cam$resistance_score_raw <- (
            direction_sign *
            confidence
        )

        cam$resistance_score <- (
            direction_sign *
            pmin(
                confidence,
                SCORE_CAP
            )
        )

        cam$dataset_id <- dataset_id
        cam$pmid <- pmid

        cam$cancer_type <- cancer_type
        cam$therapy_type <- therapy_type
        cam$regimen <- regimen

        cam$evidence_scope <- evidence_scope
        cam$matched_pharos_drugs <- matched_drugs

        cam$cell_type <- ct

        cam$n_R <- n_R
        cam$n_NR <- n_NR

        cam$available <- 1

        bank_list[[
            paste(
                dataset_id,
                ct,
                sep = "|||"
            )
        ]] <- cam

        cat(
            "    Significant pathways:",
            sum(
                cam$FDR < 0.05,
                na.rm = TRUE
            ),
            "/",
            nrow(cam),
            "\n"
        )
    }

    dataset_summary[[dataset_id]] <- data.frame(
        dataset_id = dataset_id,
        pmid = pmid,
        cancer_type = cancer_type,
        therapy_type = therapy_type,
        regimen = regimen,
        matched_pharos_drugs = matched_drugs,
        evidence_scope = evidence_scope,
        status = "processed",
        pre_cells = n_pre_cells,
        n_R = n_patients_R,
        n_NR = n_patients_NR,
        eligible_celltypes = length(
            eligible
        ),
        stringsAsFactors = FALSE
    )

    # Free RAM before next large RDS
    rm(
        obj,
        counts,
        pb_counts
    )

    gc()
}

# ============================================================
# CONSOLIDATE
# ============================================================

cat(
    "\n============================================================\n"
)
cat("CONSOLIDATING RESISTANCE BANK\n")
cat(
    "============================================================\n"
)

if (length(bank_list) > 0) {

    bank <- do.call(
        rbind,
        bank_list
    )

    rownames(bank) <- NULL

    write.csv(
        bank,
        gzfile(
            file.path(
                OUTDIR,
                "cellresdb_resistance_bank_long.csv.gz"
            )
        ),
        row.names = FALSE
    )

    significant_bank <- bank[
        bank$FDR < 0.05,
        ,
        drop = FALSE
    ]

    write.csv(
        significant_bank,
        gzfile(
            file.path(
                OUTDIR,
                "cellresdb_resistance_bank_significant.csv.gz"
            )
        ),
        row.names = FALSE
    )

    # --------------------------------------------------------
    # Create pathway token matrix
    # one row = dataset × cell type
    # --------------------------------------------------------

    bank$token_id <- paste(
        bank$dataset_id,
        bank$cell_type,
        sep = "|||"
    )

    matrix_input <- bank[
        ,
        c(
            "token_id",
            "pathway",
            "resistance_score"
        )
    ]

    pathway_matrix <- reshape(
        matrix_input,
        idvar = "token_id",
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
            "cellresdb_pathway_token_matrix.csv"
        ),
        row.names = FALSE
    )
}

# ============================================================
# COVERAGE
# ============================================================

if (length(coverage_list) > 0) {

    coverage_all <- do.call(
        rbind,
        coverage_list
    )

    rownames(coverage_all) <- NULL

    write.csv(
        coverage_all,
        file.path(
            OUTDIR,
            "celltype_patient_coverage.csv"
        ),
        row.names = FALSE
    )
}

# ============================================================
# DATASET SUMMARY
# ============================================================

if (length(dataset_summary) > 0) {

    all_names <- unique(
        unlist(
            lapply(
                dataset_summary,
                names
            )
        )
    )

    padded <- lapply(
        dataset_summary,
        function(x) {

            missing <- setdiff(
                all_names,
                names(x)
            )

            for (m in missing) {
                x[[m]] <- NA
            }

            x[
                ,
                all_names,
                drop = FALSE
            ]
        }
    )

    summary_df <- do.call(
        rbind,
        padded
    )

    rownames(summary_df) <- NULL

    write.csv(
        summary_df,
        file.path(
            OUTDIR,
            "dataset_processing_summary.csv"
        ),
        row.names = FALSE
    )
}

# ============================================================
# FINAL SUMMARY
# ============================================================

cat("\n============================================================\n")
cat("DONE\n")
cat("============================================================\n")

if (exists("bank")) {

    cat(
        "Processed resistance rows:",
        nrow(bank),
        "\n"
    )

    cat(
        "Datasets represented:",
        length(
            unique(
                bank$dataset_id
            )
        ),
        "\n"
    )

    cat(
        "Cell-type tokens:",
        length(
            unique(
                paste(
                    bank$dataset_id,
                    bank$cell_type
                )
            )
        ),
        "\n"
    )

    cat(
        "Significant pathway associations:",
        sum(
            bank$FDR < 0.05,
            na.rm = TRUE
        ),
        "\n"
    )
}

cat(
    "Output:",
    OUTDIR,
    "\n"
)
