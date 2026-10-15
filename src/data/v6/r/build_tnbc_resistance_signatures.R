suppressPackageStartupMessages({
    library(Matrix)
})


# ============================================================
# CONFIG
# ============================================================

DATA_DIR <- paste0(
    "data/processed/v6/cellresdb_expanded/",
    "GSE169246_TNBC_1"
)

OUT_DIR <- file.path(
    DATA_DIR,
    "signatures"
)

dir.create(
    OUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)


LOGCPM_PATH <- file.path(
    DATA_DIR,
    "pseudobulk_logcpm.rds"
)

META_PATH <- file.path(
    DATA_DIR,
    "pseudobulk_metadata.csv"
)


# Require at least this many independent patients
# PER response class for a signature.

MIN_PATIENTS_PER_CLASS <- 3


# ============================================================
# LOAD
# ============================================================

cat("============================================================\n")
cat("PHAROS V6 — TNBC RESISTANCE SIGNATURE BUILDER\n")
cat("============================================================\n\n")


x <- readRDS(
    LOGCPM_PATH
)

meta <- read.csv(
    META_PATH,
    stringsAsFactors = FALSE,
    check.names = FALSE
)


if (
    !identical(
        colnames(x),
        meta$group_id
    )
) {
    stop(
        "Pseudobulk matrix columns do not align with metadata."
    )
}


cat(
    "Genes:",
    nrow(x),
    "\n"
)

cat(
    "Pseudobulks:",
    ncol(x),
    "\n"
)

cat(
    "Patients:",
    length(
        unique(meta$patient)
    ),
    "\n"
)


# ============================================================
# HELPERS
# ============================================================

mean_signature <- function(
    matrix,
    indices
) {

    if (
        length(indices) == 1
    ) {
        return(
            as.numeric(
                matrix[, indices]
            )
        )
    }

    return(
        Matrix::rowMeans(
            matrix[, indices, drop = FALSE]
        )
    )
}


make_unpaired_signature <- function(
    celltype,
    treatment
) {

    sub <- meta[
        meta$cluster1 == celltype
        &
        meta$treatment == treatment,
        ,
        drop = FALSE
    ]


    nr <- sub[
        sub$Response == "NR",
        ,
        drop = FALSE
    ]

    r <- sub[
        sub$Response == "R",
        ,
        drop = FALSE
    ]


    nr_patients <- unique(
        nr$patient
    )

    r_patients <- unique(
        r$patient
    )


    if (
        length(nr_patients)
        <
        MIN_PATIENTS_PER_CLASS
        ||
        length(r_patients)
        <
        MIN_PATIENTS_PER_CLASS
    ) {

        return(NULL)
    }


    nr_idx <- match(
        nr$group_id,
        colnames(x)
    )

    r_idx <- match(
        r$group_id,
        colnames(x)
    )


    nr_mean <- mean_signature(
        x,
        nr_idx
    )

    r_mean <- mean_signature(
        x,
        r_idx
    )


    effect <- (
        nr_mean
        -
        r_mean
    )


    data.frame(
        gene =
            rownames(x),

        effect =
            effect,

        abs_effect =
            abs(effect),

        signature_type =
            ifelse(
                treatment == "Pre",
                "baseline_NR_minus_R",
                "post_NR_minus_R"
            ),

        cell_type =
            celltype,

        NR_patients =
            length(nr_patients),

        R_patients =
            length(r_patients),

        stringsAsFactors =
            FALSE
    )
}


make_adaptation_signature <- function(
    celltype
) {

    sub <- meta[
        meta$cluster1 == celltype,
        ,
        drop = FALSE
    ]


    patients <- unique(
        sub$patient
    )


    deltas <- list()
    responses <- character()


    for (
        patient in patients
    ) {

        p <- sub[
            sub$patient == patient,
            ,
            drop = FALSE
        ]


        pre <- p[
            p$treatment == "Pre",
            ,
            drop = FALSE
        ]

        post <- p[
            p$treatment == "Post",
            ,
            drop = FALSE
        ]


        if (
            nrow(pre) != 1
            ||
            nrow(post) != 1
        ) {
            next
        }


        pre_idx <- match(
            pre$group_id,
            colnames(x)
        )

        post_idx <- match(
            post$group_id,
            colnames(x)
        )


        delta <- (
            as.numeric(
                x[, post_idx]
            )
            -
            as.numeric(
                x[, pre_idx]
            )
        )


        deltas[[
            patient
        ]] <- delta

        responses[
            patient
        ] <- p$Response[1]
    }


    if (
        length(deltas) == 0
    ) {
        return(NULL)
    }


    delta_matrix <- do.call(
        cbind,
        deltas
    )


    nr_patients <- names(
        responses
    )[
        responses == "NR"
    ]

    r_patients <- names(
        responses
    )[
        responses == "R"
    ]


    if (
        length(nr_patients)
        <
        MIN_PATIENTS_PER_CLASS
        ||
        length(r_patients)
        <
        MIN_PATIENTS_PER_CLASS
    ) {
        return(NULL)
    }


    nr_mean <- rowMeans(
        delta_matrix[
            ,
            nr_patients,
            drop = FALSE
        ]
    )

    r_mean <- rowMeans(
        delta_matrix[
            ,
            r_patients,
            drop = FALSE
        ]
    )


    effect <- (
        nr_mean
        -
        r_mean
    )


    data.frame(
        gene =
            rownames(x),

        effect =
            effect,

        abs_effect =
            abs(effect),

        signature_type =
            "adaptation_delta_NR_minus_R",

        cell_type =
            celltype,

        NR_patients =
            length(nr_patients),

        R_patients =
            length(r_patients),

        stringsAsFactors =
            FALSE
    )
}


# ============================================================
# BUILD ALL SIGNATURES
# ============================================================

celltypes <- sort(
    unique(
        meta$cluster1
    )
)


results <- list()
coverage_records <- list()


for (
    celltype in celltypes
) {

    cat(
        "\nProcessing:",
        celltype,
        "\n"
    )


    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline <- make_unpaired_signature(
        celltype,
        "Pre"
    )


    if (
        !is.null(baseline)
    ) {

        key <- paste0(
            celltype,
            "__baseline"
        )

        results[[
            key
        ]] <- baseline


        coverage_records[[
            length(coverage_records) + 1
        ]] <- data.frame(
            cell_type =
                celltype,

            signature_type =
                "baseline_NR_minus_R",

            NR_patients =
                baseline$NR_patients[1],

            R_patients =
                baseline$R_patients[1],

            eligible =
                TRUE
        )
    }


    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    post <- make_unpaired_signature(
        celltype,
        "Post"
    )


    if (
        !is.null(post)
    ) {

        key <- paste0(
            celltype,
            "__post"
        )

        results[[
            key
        ]] <- post


        coverage_records[[
            length(coverage_records) + 1
        ]] <- data.frame(
            cell_type =
                celltype,

            signature_type =
                "post_NR_minus_R",

            NR_patients =
                post$NR_patients[1],

            R_patients =
                post$R_patients[1],

            eligible =
                TRUE
        )
    }


    # --------------------------------------------------------
    # ADAPTATION
    # --------------------------------------------------------

    adaptation <- make_adaptation_signature(
        celltype
    )


    if (
        !is.null(adaptation)
    ) {

        key <- paste0(
            celltype,
            "__adaptation"
        )

        results[[
            key
        ]] <- adaptation


        coverage_records[[
            length(coverage_records) + 1
        ]] <- data.frame(
            cell_type =
                celltype,

            signature_type =
                "adaptation_delta_NR_minus_R",

            NR_patients =
                adaptation$NR_patients[1],

            R_patients =
                adaptation$R_patients[1],

            eligible =
                TRUE
        )
    }
}


# ============================================================
# COMBINE
# ============================================================

if (
    length(results) == 0
) {
    stop(
        "No eligible signatures were generated."
    )
}


all_signatures <- do.call(
    rbind,
    results
)

rownames(
    all_signatures
) <- NULL


coverage <- do.call(
    rbind,
    coverage_records
)


# ============================================================
# ADD PROVENANCE
# ============================================================

all_signatures$dataset_id <- (
    "GSE169246_TNBC_1"
)

all_signatures$drug <- (
    "Paclitaxel"
)

all_signatures$drugbank_id <- (
    "DB01229"
)

all_signatures$cancer <- (
    "Triple-negative breast cancer"
)

all_signatures$evidence_context <- (
    "direct_monotherapy"
)

all_signatures$patient_quality <- (
    "strong"
)


# ============================================================
# SAVE FULL SIGNATURE TABLE
# ============================================================

signature_path <- file.path(
    OUT_DIR,
    "gene_signatures.csv"
)

coverage_path <- file.path(
    OUT_DIR,
    "signature_coverage.csv"
)


write.csv(
    all_signatures,
    signature_path,
    row.names = FALSE
)

write.csv(
    coverage,
    coverage_path,
    row.names = FALSE
)


# ============================================================
# TOP GENES
# ============================================================

top_records <- list()


signature_groups <- split(
    all_signatures,
    interaction(
        all_signatures$cell_type,
        all_signatures$signature_type,
        drop = TRUE
    )
)


for (
    name in names(signature_groups)
) {

    g <- signature_groups[[
        name
    ]]


    g <- g[
        order(
            g$abs_effect,
            decreasing = TRUE
        ),
        ,
        drop = FALSE
    ]


    top_records[[
        name
    ]] <- head(
        g,
        25
    )
}


top_genes <- do.call(
    rbind,
    top_records
)

rownames(
    top_genes
) <- NULL


top_path <- file.path(
    OUT_DIR,
    "top25_genes_per_signature.csv"
)


write.csv(
    top_genes,
    top_path,
    row.names = FALSE
)


# ============================================================
# AUDIT
# ============================================================

cat("\n============================================================\n")
cat("SIGNATURE AUDIT\n")
cat("============================================================\n")


cat(
    "\nEligible signatures:",
    nrow(coverage),
    "\n"
)


cat(
    "\nSignature counts by type:\n"
)

print(
    table(
        coverage$signature_type
    )
)


cat(
    "\nEligible cell types by signature:\n"
)

print(
    coverage[
        order(
            coverage$signature_type,
            coverage$cell_type
        ),
        ,
        drop = FALSE
    ],
    row.names = FALSE
)


cat(
    "\nEffect-size summary:\n"
)

print(
    aggregate(
        abs_effect ~ signature_type,
        data = all_signatures,
        FUN = function(z) {
            c(
                median = median(z),
                mean = mean(z),
                max = max(z)
            )
        }
    )
)


cat("\n============================================================\n")
cat("TOP GENES — SELECTED MAJOR CELL TYPES\n")
cat("============================================================\n")


major <- c(
    "CD4+ T Cell",
    "CD8+ T Cell",
    "Monocyte/Macrophage",
    "B Cell"
)


for (
    ct in major
) {

    cat(
        "\n###",
        ct,
        "###\n"
    )


    for (
        sig in c(
            "baseline_NR_minus_R",
            "post_NR_minus_R",
            "adaptation_delta_NR_minus_R"
        )
    ) {

        g <- all_signatures[
            all_signatures$cell_type == ct
            &
            all_signatures$signature_type == sig,
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
                g$abs_effect,
                decreasing = TRUE
            ),
            ,
            drop = FALSE
        ]


        cat(
            "\n",
            sig,
            "\n",
            sep = ""
        )


        print(
            head(
                g[
                    ,
                    c(
                        "gene",
                        "effect",
                        "NR_patients",
                        "R_patients"
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
    signature_path,
    "\n"
)

cat(
    coverage_path,
    "\n"
)

cat(
    top_path,
    "\n"
)

cat("\nDONE ✅\n")
