suppressPackageStartupMessages({
    library(SeuratObject)
    library(Matrix)
})


# ============================================================
# CONFIG
# ============================================================

INPUT <- paste0(
    "data/external/cellresdb_full/raw_rds/",
    "tier1/strong/GSE169246_TNBC_1.rds"
)

OUT_DIR <- paste0(
    "data/processed/v6/cellresdb_expanded/",
    "GSE169246_TNBC_1"
)

MIN_CELLS <- 20


dir.create(
    OUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)


# ============================================================
# LOAD
# ============================================================

cat("============================================================\n")
cat("PHAROS V6 — TNBC PATIENT PSEUDOBULK\n")
cat("============================================================\n\n")

cat("Loading RDS...\n")


obj <- readRDS(
    INPUT
)


meta <- obj[[]]


counts <- GetAssayData(
    obj,
    assay = "RNA",
    layer = "counts"
)


cat(
    "Genes:",
    nrow(counts),
    "\n"
)

cat(
    "Cells:",
    ncol(counts),
    "\n"
)


# ============================================================
# VALIDATE REQUIRED METADATA
# ============================================================

required <- c(
    "patient",
    "Response",
    "treatment",
    "cluster1",
    "drug_name"
)


missing <- setdiff(
    required,
    colnames(meta)
)


if (
    length(missing) > 0
) {

    stop(
        paste(
            "Missing required metadata:",
            paste(
                missing,
                collapse = ", "
            )
        )
    )
}


# ============================================================
# VALIDATE CELL ALIGNMENT
# ============================================================

if (
    !identical(
        rownames(meta),
        colnames(counts)
    )
) {

    stop(
        paste(
            "Metadata rows do not align exactly",
            "with expression matrix columns."
        )
    )
}


cat(
    "Metadata/expression cell alignment: TRUE ✅\n"
)


# ============================================================
# CLEAN LABELS
# ============================================================

meta$patient <- trimws(
    as.character(
        meta$patient
    )
)


meta$Response <- trimws(
    as.character(
        meta$Response
    )
)


meta$treatment <- trimws(
    as.character(
        meta$treatment
    )
)


meta$cluster1 <- trimws(
    as.character(
        meta$cluster1
    )
)


# ============================================================
# FILTER TO USABLE CELLS
# ============================================================

keep <- (
    !is.na(meta$patient)
    &
    meta$patient != ""
    &
    !is.na(meta$Response)
    &
    meta$Response %in% c(
        "R",
        "NR"
    )
    &
    !is.na(meta$treatment)
    &
    meta$treatment %in% c(
        "Pre",
        "Post"
    )
    &
    !is.na(meta$cluster1)
    &
    meta$cluster1 != ""
)


meta <- meta[
    keep,
    ,
    drop = FALSE
]


counts <- counts[
    ,
    keep,
    drop = FALSE
]


cat(
    "Retained cells:",
    ncol(counts),
    "\n"
)


# ============================================================
# REVALIDATE ALIGNMENT AFTER FILTERING
# ============================================================

if (
    !identical(
        rownames(meta),
        colnames(counts)
    )
) {

    stop(
        paste(
            "Metadata/expression alignment was lost",
            "after cell filtering."
        )
    )
}


# ============================================================
# BUILD PATIENT × RESPONSE × TIME × CELL-TYPE GROUPS
# ============================================================

meta$group_id <- paste(
    meta$patient,
    meta$Response,
    meta$treatment,
    meta$cluster1,
    sep = "___"
)


group_levels <- unique(
    meta$group_id
)


group_factor <- factor(
    meta$group_id,
    levels = group_levels
)


cell_counts <- table(
    group_factor
)


group_meta <- data.frame(
    group_id =
        names(cell_counts),

    n_cells =
        as.integer(cell_counts),

    stringsAsFactors =
        FALSE
)


# ============================================================
# PARSE GROUP IDENTIFIERS
# ============================================================

parts <- strsplit(
    group_meta$group_id,
    "___",
    fixed = TRUE
)


part_lengths <- lengths(
    parts
)


if (
    any(
        part_lengths != 4
    )
) {

    stop(
        paste(
            "One or more group IDs did not split into",
            "patient / Response / treatment / cluster1."
        )
    )
}


group_meta$patient <- vapply(
    parts,
    `[`,
    character(1),
    1
)


group_meta$Response <- vapply(
    parts,
    `[`,
    character(1),
    2
)


group_meta$treatment <- vapply(
    parts,
    `[`,
    character(1),
    3
)


group_meta$cluster1 <- vapply(
    parts,
    `[`,
    character(1),
    4
)


# ============================================================
# MINIMUM CELL FILTER
# ============================================================

group_meta$passes_min_cells <- (
    group_meta$n_cells
    >=
    MIN_CELLS
)


cat(
    "Raw pseudobulk groups:",
    nrow(group_meta),
    "\n"
)


cat(
    "Groups with >=",
    MIN_CELLS,
    "cells:",
    sum(
        group_meta$passes_min_cells
    ),
    "\n"
)


# ============================================================
# SPARSE PSEUDOBULK SUM
# ============================================================

cat("\nAggregating raw counts...\n")


design <- sparse.model.matrix(
    ~ 0 + group_factor
)


# sparse.model.matrix creates its own column names.
# Replace them with our exact group IDs.

colnames(
    design
) <- levels(
    group_factor
)


pseudobulk_counts <- (
    counts
    %*%
    design
)


# Explicitly restore/assign dimnames.
# This protects against Matrix operations dropping names.

rownames(
    pseudobulk_counts
) <- rownames(
    counts
)


colnames(
    pseudobulk_counts
) <- levels(
    group_factor
)


# ============================================================
# VALIDATE RAW PSEUDOBULK
# ============================================================

if (
    ncol(
        pseudobulk_counts
    )
    !=
    nrow(
        group_meta
    )
) {

    stop(
        paste(
            "Raw pseudobulk column count does not",
            "match group metadata."
        )
    )
}


if (
    !identical(
        as.character(
            colnames(
                pseudobulk_counts
            )
        ),
        as.character(
            group_meta$group_id
        )
    )
) {

    stop(
        paste(
            "Raw pseudobulk columns do not align",
            "with group metadata."
        )
    )
}


cat(
    "Raw pseudobulk metadata alignment: TRUE ✅\n"
)


# ============================================================
# FILTER LOW-CELL PSEUDOBULKS
# ============================================================

valid_groups <- group_meta$group_id[
    group_meta$passes_min_cells
]


pseudobulk_counts <- pseudobulk_counts[
    ,
    valid_groups,
    drop = FALSE
]


group_meta_filtered <- group_meta[
    match(
        valid_groups,
        group_meta$group_id
    ),
    ,
    drop = FALSE
]


# ============================================================
# VALIDATE FILTERED PSEUDOBULK ALIGNMENT
# ============================================================

if (
    ncol(
        pseudobulk_counts
    )
    !=
    nrow(
        group_meta_filtered
    )
) {

    stop(
        paste(
            "Filtered pseudobulk column count does not",
            "match filtered metadata."
        )
    )
}


if (
    !identical(
        as.character(
            colnames(
                pseudobulk_counts
            )
        ),
        as.character(
            group_meta_filtered$group_id
        )
    )
) {

    stop(
        paste(
            "Filtered pseudobulk columns do not align",
            "with filtered metadata."
        )
    )
}


cat(
    "Filtered pseudobulk metadata alignment: TRUE ✅\n"
)


# ============================================================
# LIBRARY SIZE CALCULATION
# ============================================================

library_sizes <- Matrix::colSums(
    pseudobulk_counts
)


if (
    any(
        !is.finite(
            library_sizes
        )
    )
) {

    stop(
        "One or more pseudobulk library sizes are non-finite."
    )
}


if (
    any(
        library_sizes <= 0
    )
) {

    stop(
        "One or more pseudobulk groups have zero total counts."
    )
}


# ============================================================
# LOG2(CPM + 1) NORMALIZATION
# ============================================================

cat("Normalizing pseudobulks...\n")


inv_lib <- Diagonal(
    x =
        1e6
        /
        library_sizes
)


cpm <- (
    pseudobulk_counts
    %*%
    inv_lib
)


# IMPORTANT:
# Multiplication by an unnamed diagonal Matrix can drop
# matrix dimnames. Restore them explicitly.

dimnames(
    cpm
) <- dimnames(
    pseudobulk_counts
)


# ============================================================
# VALIDATE CPM DIMNAMES
# ============================================================

if (
    is.null(
        colnames(cpm)
    )
) {

    stop(
        "CPM matrix lost its column names."
    )
}


if (
    is.null(
        rownames(cpm)
    )
) {

    stop(
        "CPM matrix lost its gene row names."
    )
}


if (
    !identical(
        as.character(
            colnames(cpm)
        ),
        as.character(
            group_meta_filtered$group_id
        )
    )
) {

    stop(
        paste(
            "CPM columns lost alignment",
            "with pseudobulk metadata."
        )
    )
}


# ============================================================
# LOG TRANSFORMATION
# ============================================================

logcpm <- cpm


logcpm@x <- log2(
    logcpm@x
    +
    1
)


# Restore dimnames again defensively.

dimnames(
    logcpm
) <- dimnames(
    pseudobulk_counts
)


# ============================================================
# FINAL LOGCPM VALIDATION
# ============================================================

if (
    !identical(
        dim(
            logcpm
        ),
        dim(
            pseudobulk_counts
        )
    )
) {

    stop(
        "LogCPM dimensions differ from raw pseudobulk counts."
    )
}


if (
    is.null(
        colnames(
            logcpm
        )
    )
) {

    stop(
        "LogCPM matrix has no column names."
    )
}


if (
    is.null(
        rownames(
            logcpm
        )
    )
) {

    stop(
        "LogCPM matrix has no gene names."
    )
}


if (
    !identical(
        as.character(
            colnames(
                logcpm
            )
        ),
        as.character(
            group_meta_filtered$group_id
        )
    )
) {

    stop(
        paste(
            "LogCPM columns lost alignment",
            "with pseudobulk metadata."
        )
    )
}


if (
    !identical(
        as.character(
            rownames(
                logcpm
            )
        ),
        as.character(
            rownames(
                pseudobulk_counts
            )
        )
    )
) {

    stop(
        paste(
            "LogCPM gene ordering differs",
            "from pseudobulk counts."
        )
    )
}


cat(
    "LogCPM metadata alignment: TRUE ✅\n"
)


# ============================================================
# ADD LIBRARY SIZE TO METADATA
# ============================================================

group_meta_filtered$library_size <- (
    as.numeric(
        library_sizes
    )
)


# ============================================================
# FINAL PRE-SAVE VALIDATION
# ============================================================

if (
    !identical(
        as.character(
            colnames(
                pseudobulk_counts
            )
        ),
        as.character(
            group_meta_filtered$group_id
        )
    )
) {

    stop(
        paste(
            "Final counts/metadata alignment",
            "check failed."
        )
    )
}


if (
    !identical(
        as.character(
            colnames(
                logcpm
            )
        ),
        as.character(
            group_meta_filtered$group_id
        )
    )
) {

    stop(
        paste(
            "Final LogCPM/metadata alignment",
            "check failed."
        )
    )
}


# ============================================================
# SAVE
# ============================================================

saveRDS(
    pseudobulk_counts,
    file.path(
        OUT_DIR,
        "pseudobulk_counts.rds"
    )
)


saveRDS(
    logcpm,
    file.path(
        OUT_DIR,
        "pseudobulk_logcpm.rds"
    )
)


write.csv(
    group_meta_filtered,
    file.path(
        OUT_DIR,
        "pseudobulk_metadata.csv"
    ),
    row.names = FALSE
)


write.csv(
    data.frame(
        gene =
            rownames(
                pseudobulk_counts
            )
    ),
    file.path(
        OUT_DIR,
        "genes.csv"
    ),
    row.names = FALSE
)


# Preserve ALL groups, including groups that failed MIN_CELLS.

write.csv(
    group_meta,
    file.path(
        OUT_DIR,
        "pseudobulk_group_audit.csv"
    ),
    row.names = FALSE
)


# ============================================================
# PSEUDOBULK AUDIT
# ============================================================

cat("\n============================================================\n")
cat("PSEUDOBULK AUDIT\n")
cat("============================================================\n")


cat(
    "\nFinal matrix dimensions:\n"
)

print(
    dim(
        pseudobulk_counts
    )
)


cat(
    "\nMatrix dimnames:\n"
)

cat(
    "Gene names:",
    length(
        rownames(
            pseudobulk_counts
        )
    ),
    "\n"
)


cat(
    "Pseudobulk names:",
    length(
        colnames(
            pseudobulk_counts
        )
    ),
    "\n"
)


cat(
    "LogCPM pseudobulk names:",
    length(
        colnames(
            logcpm
        )
    ),
    "\n"
)


cat(
    "Exact metadata alignment:",
    identical(
        as.character(
            colnames(
                logcpm
            )
        ),
        as.character(
            group_meta_filtered$group_id
        )
    ),
    "\n"
)


# ============================================================
# CELL COUNTS
# ============================================================

cat(
    "\nCell counts per pseudobulk:\n"
)


print(
    summary(
        group_meta_filtered$n_cells
    )
)


# ============================================================
# LIBRARY SIZES
# ============================================================

cat(
    "\nLibrary sizes:\n"
)


print(
    summary(
        group_meta_filtered$library_size
    )
)


# ============================================================
# RESPONSE × TREATMENT GROUP COUNTS
# ============================================================

cat(
    "\nGroups by Response × Treatment:\n"
)


print(
    table(
        Response =
            group_meta_filtered$Response,

        Treatment =
            group_meta_filtered$treatment
    )
)


# ============================================================
# UNIQUE PATIENT COUNTS
# ============================================================

cat(
    "\nUnique patients by Response × Treatment:\n"
)


patient_groups <- unique(
    group_meta_filtered[
        ,
        c(
            "patient",
            "Response",
            "treatment"
        )
    ]
)


print(
    table(
        Response =
            patient_groups$Response,

        Treatment =
            patient_groups$treatment
    )
)


# ============================================================
# CELL TYPES
# ============================================================

cat(
    "\nGroups per standardized cell type:\n"
)


print(
    sort(
        table(
            group_meta_filtered$cluster1
        ),
        decreasing = TRUE
    )
)


# ============================================================
# PAIRED-PATIENT AUDIT
# ============================================================

paired_table <- table(
    patient_groups$patient,
    patient_groups$treatment
)


if (
    all(
        c(
            "Pre",
            "Post"
        )
        %in%
        colnames(
            paired_table
        )
    )
) {

    paired <- rownames(
        paired_table
    )[
        paired_table[
            ,
            "Pre"
        ] > 0
        &
        paired_table[
            ,
            "Post"
        ] > 0
    ]

} else {

    paired <- character(0)

}


paired_meta <- unique(
    patient_groups[
        patient_groups$patient
        %in%
        paired,

        c(
            "patient",
            "Response"
        )
    ]
)


cat(
    "\nPatients retaining both Pre and Post after pseudobulk filtering:",
    length(
        paired
    ),
    "\n"
)


if (
    nrow(
        paired_meta
    ) > 0
) {

    print(
        table(
            paired_meta$Response
        )
    )
}


# ============================================================
# SAVE REPORT
# ============================================================

cat("\nSaved:\n")


cat(
    file.path(
        OUT_DIR,
        "pseudobulk_counts.rds"
    ),
    "\n"
)


cat(
    file.path(
        OUT_DIR,
        "pseudobulk_logcpm.rds"
    ),
    "\n"
)


cat(
    file.path(
        OUT_DIR,
        "pseudobulk_metadata.csv"
    ),
    "\n"
)


cat(
    file.path(
        OUT_DIR,
        "pseudobulk_group_audit.csv"
    ),
    "\n"
)


cat("\n============================================================\n")
cat("DONE ✅\n")
cat("============================================================\n")
