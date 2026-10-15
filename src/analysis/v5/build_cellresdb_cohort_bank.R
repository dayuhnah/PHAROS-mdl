input_file <- paste0(
  "data/processed/v5/cellresdb/resistance_bank_model_ready/",
  "cellresdb_model_ready_long.csv.gz"
)

out_dir <- paste0(
  "data/processed/v5/cellresdb/",
  "resistance_bank_cohort"
)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

df <- read.csv(
  gzfile(input_file),
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

collapse_unique <- function(x) {
  x <- as.character(x)
  x <- x[!is.na(x)]
  x <- trimws(x)
  x <- x[x != "" & x != "NA"]

  if (length(x) == 0) {
    return(NA_character_)
  }

  paste(sort(unique(x)), collapse = ";")
}

safe_mean <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]

  if (length(x) == 0) return(NA_real_)

  mean(x)
}

safe_sd <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]

  if (length(x) <= 1) return(0)

  sd(x)
}

safe_max <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]

  if (length(x) == 0) return(NA_real_)

  max(x)
}

safe_min <- function(x) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]

  if (length(x) == 0) return(NA_real_)

  min(x)
}

# ------------------------------------------------------------
# Group by independent cohort × cell type × pathway
# ------------------------------------------------------------

group_key <- interaction(
  df$cohort_id,
  df$cell_type,
  df$pathway,
  drop = TRUE,
  lex.order = TRUE
)

groups <- split(df, group_key)

rows <- lapply(groups, function(g) {

  dirs <- unique(
    as.character(g$Direction[
      !is.na(g$Direction) &
      g$Direction != ""
    ])
  )

  direction_agreement <- as.integer(
    length(dirs) <= 1
  )

  direction_consensus <- if (
    length(dirs) == 1
  ) {
    dirs[1]
  } else if (
    length(dirs) > 1
  ) {
    "Mixed"
  } else {
    NA_character_
  }

  has_direct <- as.integer(
    any(
      g$evidence_role == "direct_monotherapy",
      na.rm = TRUE
    )
  )

  has_combo <- as.integer(
    any(
      g$evidence_role == "combination_context",
      na.rm = TRUE
    )
  )

  role <- if (
    has_direct == 1 && has_combo == 1
  ) {
    "direct_monotherapy_plus_context"
  } else if (
    has_direct == 1
  ) {
    "direct_monotherapy"
  } else {
    "combination_context"
  }

  data.frame(
    cohort_id = g$cohort_id[1],

    dataset_ids =
      collapse_unique(g$dataset_id),

    pmids =
      collapse_unique(g$pmid),

    cancer_type =
      collapse_unique(g$cancer_type),

    therapy_types =
      collapse_unique(g$therapy_type),

    regimens =
      collapse_unique(g$regimen),

    evidence_scopes =
      collapse_unique(g$evidence_scope),

    matched_pharos_drugs =
      collapse_unique(g$matched_pharos_drugs),

    cell_type = g$cell_type[1],
    pathway = g$pathway[1],

    NGenes =
      round(safe_mean(g$NGenes)),

    direction_consensus =
      direction_consensus,

    direction_agreement =
      direction_agreement,

    # Conservative significance:
    # duplicated patient representations are NOT independent.
    PValue =
      safe_max(g$PValue),

    FDR =
      safe_max(g$FDR),

    # Average biological effect across duplicate representations
    mean_logFC =
      safe_mean(g$mean_logFC),

    median_logFC =
      safe_mean(g$median_logFC),

    # Signed resistance evidence
    resistance_score =
      safe_mean(g$resistance_score),

    resistance_score_sd =
      safe_sd(g$resistance_score),

    # Conservative confidence
    confidence =
      safe_min(g$confidence),

    # Patient counts are not added across duplicates
    n_R =
      safe_min(g$n_R),

    n_NR =
      safe_min(g$n_NR),

    available =
      safe_min(g$available),

    n_dataset_representations =
      length(unique(g$dataset_id)),

    has_direct_monotherapy =
      has_direct,

    has_combination_context =
      has_combo,

    evidence_role =
      role,

    cohort_deduplicated = 1,

    stringsAsFactors = FALSE
  )
})

bank <- do.call(rbind, rows)
row.names(bank) <- NULL

# ------------------------------------------------------------
# Sort
# ------------------------------------------------------------

bank <- bank[
  order(
    bank$cohort_id,
    bank$cell_type,
    bank$pathway
  ),
]

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out_long <- file.path(
  out_dir,
  "cellresdb_cohort_resistance_bank.csv.gz"
)

write.csv(
  bank,
  gzfile(out_long),
  row.names = FALSE
)

# Compact model token table
token_cols <- c(
  "cohort_id",
  "cancer_type",
  "matched_pharos_drugs",
  "cell_type",
  "pathway",
  "resistance_score",
  "confidence",
  "FDR",
  "median_logFC",
  "direction_agreement",
  "resistance_score_sd",
  "n_R",
  "n_NR",
  "available",
  "has_direct_monotherapy",
  "has_combination_context"
)

write.csv(
  bank[, token_cols],
  file.path(
    out_dir,
    "cellresdb_pathway_tokens.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

cat("\n========================================\n")
cat("DEDUPLICATED CELLRESDB COHORT BANK\n")
cat("========================================\n")

cat("Rows:", nrow(bank), "\n")

cat(
  "Independent cohorts:",
  length(unique(bank$cohort_id)),
  "\n"
)

cat(
  "Cohort-cell-type tokens:",
  nrow(
    unique(
      bank[, c("cohort_id", "cell_type")]
    )
  ),
  "\n"
)

cat(
  "Unique pathways:",
  length(unique(bank$pathway)),
  "\n"
)

cat(
  "Direction disagreements:",
  sum(bank$direction_agreement == 0),
  "\n"
)

cat(
  "Significant associations (FDR < 0.05):",
  sum(bank$FDR < 0.05, na.rm = TRUE),
  "\n"
)

cat("\nRows per cohort:\n")
print(
  table(bank$cohort_id)
)

cat("\nCell types per cohort:\n")

ct <- aggregate(
  cell_type ~ cohort_id,
  data = unique(
    bank[, c("cohort_id", "cell_type")]
  ),
  FUN = length
)

print(ct)

cat("\nEvidence roles:\n")
print(table(bank$evidence_role))

cat("\nOutput:", out_dir, "\n")
cat("DONE\n")
