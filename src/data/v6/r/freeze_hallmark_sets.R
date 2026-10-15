suppressPackageStartupMessages({
    library(msigdbr)
})


# ============================================================
# CONFIG
# ============================================================

V5_TOKEN_PATH <- paste0(
    "data/processed/v5/cellresdb/",
    "resistance_bank/",
    "cellresdb_pathway_token_matrix.csv"
)

OUT_DIR <- paste0(
    "data/processed/v6/cellresdb_expanded/",
    "hallmark_reference"
)

dir.create(
    OUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)


# ============================================================
# READ EXACT V5 PATHWAY VOCABULARY
# ============================================================

cat("============================================================\n")
cat("PHAROS V6 — FREEZE V5-COMPATIBLE HALLMARK SETS\n")
cat("============================================================\n\n")

v5_header <- read.csv(
    V5_TOKEN_PATH,
    nrows = 1,
    check.names = FALSE
)

expected_pathways <- setdiff(
    colnames(v5_header),
    "token_id"
)

cat(
    "V5 expected pathways:",
    length(expected_pathways),
    "\n"
)

if (
    length(expected_pathways) != 50
) {
    stop(
        paste(
            "Expected exactly 50 V5 Hallmark pathways, got",
            length(expected_pathways)
        )
    )
}


# ============================================================
# LOAD MSIGDB HALLMARK COLLECTION
# ============================================================

cat(
    "msigdbr version:",
    as.character(
        packageVersion("msigdbr")
    ),
    "\n"
)

cat(
    "Loading MSigDB Hallmark collection...\n"
)

hallmark_df <- tryCatch(
    {
        msigdbr(
            species = "Homo sapiens",
            collection = "H"
        )
    },
    error = function(e) {

        cat(
            "collection='H' failed; trying legacy category='H'\n"
        )

        msigdbr(
            species = "Homo sapiens",
            category = "H"
        )
    }
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns <- c(
    "gs_name",
    "gene_symbol"
)

missing_columns <- setdiff(
    required_columns,
    colnames(hallmark_df)
)

if (
    length(missing_columns) > 0
) {
    stop(
        paste(
            "msigdbr output missing columns:",
            paste(
                missing_columns,
                collapse = ", "
            )
        )
    )
}


# ============================================================
# CLEAN GENE SET TABLE
# ============================================================

hallmark_df <- hallmark_df[
    ,
    c(
        "gs_name",
        "gene_symbol"
    )
]

names(hallmark_df) <- c(
    "pathway",
    "gene_symbol"
)

hallmark_df$pathway <- trimws(
    as.character(
        hallmark_df$pathway
    )
)

hallmark_df$gene_symbol <- trimws(
    as.character(
        hallmark_df$gene_symbol
    )
)

hallmark_df <- hallmark_df[
    hallmark_df$pathway != ""
    &
    hallmark_df$gene_symbol != "",
    ,
    drop = FALSE
]

hallmark_df <- unique(
    hallmark_df
)


# ============================================================
# COMPARE WITH V5 PATHWAY VOCABULARY
# ============================================================

current_pathways <- sort(
    unique(
        hallmark_df$pathway
    )
)

missing_from_current <- setdiff(
    expected_pathways,
    current_pathways
)

extra_current <- setdiff(
    current_pathways,
    expected_pathways
)

cat(
    "\nCurrent Hallmark pathways:",
    length(current_pathways),
    "\n"
)

cat(
    "Missing V5 pathways:",
    length(missing_from_current),
    "\n"
)

if (
    length(missing_from_current) > 0
) {

    print(
        missing_from_current
    )

    stop(
        "Current Hallmark collection is missing pathways required by V5."
    )
}

if (
    length(extra_current) > 0
) {

    cat(
        "\nAdditional pathways present in current collection:\n"
    )

    print(
        extra_current
    )
}


# ============================================================
# KEEP EXACT V5 VOCABULARY
# ============================================================

hallmark_df <- hallmark_df[
    hallmark_df$pathway
    %in%
    expected_pathways,
    ,
    drop = FALSE
]


hallmark_df$pathway <- factor(
    hallmark_df$pathway,
    levels = expected_pathways
)

hallmark_df <- hallmark_df[
    order(
        hallmark_df$pathway,
        hallmark_df$gene_symbol
    ),
    ,
    drop = FALSE
]

hallmark_df$pathway <- as.character(
    hallmark_df$pathway
)


# ============================================================
# BUILD CAMERA-COMPATIBLE GENE SET LIST
# ============================================================

hallmark_sets <- split(
    hallmark_df$gene_symbol,
    hallmark_df$pathway
)

hallmark_sets <- hallmark_sets[
    expected_pathways
]

hallmark_sets <- lapply(
    hallmark_sets,
    unique
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if (
    length(hallmark_sets) != 50
) {
    stop(
        paste(
            "Final Hallmark list contains",
            length(hallmark_sets),
            "pathways instead of 50."
        )
    )
}


if (
    !identical(
        names(hallmark_sets),
        expected_pathways
    )
) {
    stop(
        "Final pathway ordering does not match V5."
    )
}


gene_counts <- vapply(
    hallmark_sets,
    length,
    integer(1)
)

if (
    any(
        gene_counts == 0
    )
) {
    stop(
        "At least one Hallmark pathway contains zero genes."
    )
}


# ============================================================
# SAVE
# ============================================================

LONG_PATH <- file.path(
    OUT_DIR,
    "hallmark_gene_sets_v5_compatible.csv"
)

RDS_PATH <- file.path(
    OUT_DIR,
    "hallmark_sets_v5_compatible.rds"
)

MANIFEST_PATH <- file.path(
    OUT_DIR,
    "hallmark_manifest.csv"
)

PROVENANCE_PATH <- file.path(
    OUT_DIR,
    "hallmark_provenance.txt"
)


write.csv(
    hallmark_df,
    LONG_PATH,
    row.names = FALSE
)

saveRDS(
    hallmark_sets,
    RDS_PATH
)


manifest <- data.frame(
    pathway =
        names(hallmark_sets),

    pathway_index =
        seq_along(
            hallmark_sets
        ),

    gene_count =
        gene_counts,

    stringsAsFactors =
        FALSE
)

write.csv(
    manifest,
    MANIFEST_PATH,
    row.names = FALSE
)


writeLines(
    c(
        paste(
            "msigdbr_version=",
            as.character(
                packageVersion("msigdbr")
            ),
            sep = ""
        ),

        paste(
            "hallmark_pathway_count=",
            length(hallmark_sets),
            sep = ""
        ),

        paste(
            "source_v5_token_matrix=",
            V5_TOKEN_PATH,
            sep = ""
        )
    ),
    PROVENANCE_PATH
)


md5 <- tools::md5sum(
    LONG_PATH
)


# ============================================================
# REPORT
# ============================================================

cat("\n============================================================\n")
cat("HALLMARK REFERENCE AUDIT\n")
cat("============================================================\n")

cat(
    "\nExact V5 pathway count:",
    length(hallmark_sets),
    "\n"
)

cat(
    "Exact V5 pathway order:",
    identical(
        names(hallmark_sets),
        expected_pathways
    ),
    "\n"
)

cat(
    "Minimum genes/pathway:",
    min(gene_counts),
    "\n"
)

cat(
    "Median genes/pathway:",
    median(gene_counts),
    "\n"
)

cat(
    "Maximum genes/pathway:",
    max(gene_counts),
    "\n"
)

cat(
    "Unique genes across all pathways:",
    length(
        unique(
            hallmark_df$gene_symbol
        )
    ),
    "\n"
)

cat(
    "\nMD5:",
    unname(md5),
    "\n"
)

cat("\nFirst 10 pathways:\n")

print(
    head(
        manifest,
        10
    ),
    row.names = FALSE
)


cat("\n============================================================\n")
cat("SAVED\n")
cat("============================================================\n")

cat(LONG_PATH, "\n")
cat(RDS_PATH, "\n")
cat(MANIFEST_PATH, "\n")
cat(PROVENANCE_PATH, "\n")

cat("\nDONE ✅\n")
