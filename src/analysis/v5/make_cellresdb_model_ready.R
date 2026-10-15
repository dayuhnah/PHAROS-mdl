input_file <- paste0(
  "data/processed/v5/cellresdb/resistance_bank/",
  "cellresdb_resistance_bank_long.csv.gz"
)

out_dir <- "data/processed/v5/cellresdb/resistance_bank_model_ready"

df <- read.csv(
  gzfile(input_file),
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------------
# Independent patient-cohort IDs
# ------------------------------------------------------------------

df$cohort_id <- df$dataset_id

df$cohort_id[
  df$dataset_id %in% c(
    "GSE161801_MM_1",
    "GSE161801_MM_2"
  )
] <- "GSE161801_MM"

df$cohort_id[
  df$dataset_id %in% c(
    "GSE169246_TNBC_1",
    "GSE169246_TNBC_2"
  )
] <- "GSE169246_TNBC"

# Number of dataset representations sharing the same patient cohort
dataset_map <- unique(
  df[, c("dataset_id", "cohort_id")]
)

cohort_counts <- aggregate(
  dataset_id ~ cohort_id,
  data = dataset_map,
  FUN = length
)

names(cohort_counts)[2] <- "n_dataset_representations"

dataset_map <- merge(
  dataset_map,
  cohort_counts,
  by = "cohort_id"
)

dataset_map$cohort_weight <-
  1 / dataset_map$n_dataset_representations

df <- merge(
  df,
  dataset_map,
  by = c("dataset_id", "cohort_id"),
  all.x = TRUE
)

# ------------------------------------------------------------------
# Evidence interpretation
# ------------------------------------------------------------------

df$patient_independent <- TRUE

df$patient_independent[
  df$n_dataset_representations > 1
] <- FALSE

df$evidence_role <- "cohort_context"

# Paclitaxel monotherapy is our strongest direct drug evidence
df$evidence_role[
  df$dataset_id == "GSE169246_TNBC_1"
] <- "direct_monotherapy"

# Combination-regimen records remain contextual evidence
df$evidence_role[
  df$dataset_id %in% c(
    "GSE123813_BCC_1",
    "GSE161801_MM_1",
    "GSE161801_MM_2",
    "GSE169246_TNBC_2"
  )
] <- "combination_context"

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

write.csv(
  df,
  gzfile(
    file.path(
      out_dir,
      "cellresdb_model_ready_long.csv.gz"
    )
  ),
  row.names = FALSE
)

write.csv(
  dataset_map,
  file.path(
    out_dir,
    "cellresdb_cohort_map.csv"
  ),
  row.names = FALSE
)

cat("\n========================================\n")
cat("MODEL-READY CELLRESDB BANK\n")
cat("========================================\n")

cat("Rows:", nrow(df), "\n")
cat("Dataset representations:",
    length(unique(df$dataset_id)), "\n")
cat("Independent patient cohorts:",
    length(unique(df$cohort_id)), "\n")

cat("\nCohort mapping:\n")
print(dataset_map)

cat("\nEvidence roles:\n")
print(table(df$evidence_role))

cat("\nDONE\n")
