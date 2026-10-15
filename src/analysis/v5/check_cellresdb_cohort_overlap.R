library(SeuratObject)

RDS_DIR <- "data/raw/cellresdb/rds"

files <- list.files(
    RDS_DIR,
    pattern = "\\.rds$",
    full.names = TRUE
)

all_rows <- list()

for (f in files) {

    id <- sub("\\.rds$", "", basename(f))

    cat("Checking:", id, "\n")

    obj <- readRDS(f)
    meta <- obj[[]]

    required <- c(
        "patient",
        "treatment",
        "Response"
    )

    if (!all(required %in% colnames(meta))) {
        rm(obj)
        gc()
        next
    }

    z <- unique(
        meta[
            tolower(trimws(meta$treatment)) == "pre" &
            toupper(trimws(meta$Response)) %in% c("R", "NR"),
            c("patient", "Response"),
            drop = FALSE
        ]
    )

    z$dataset_id <- id
    all_rows[[id]] <- z

    rm(obj)
    gc()
}

x <- do.call(rbind, all_rows)
ids <- unique(x$dataset_id)

cat("\n=== PAIRWISE PRE-TREATMENT PATIENT OVERLAP ===\n")

for (i in seq_along(ids)) {
    for (j in seq_along(ids)) {

        if (j <= i) next

        a <- x[x$dataset_id == ids[i], ]
        b <- x[x$dataset_id == ids[j], ]

        overlap <- intersect(
            as.character(a$patient),
            as.character(b$patient)
        )

        if (length(overlap) > 0) {

            merged <- merge(
                a[a$patient %in% overlap, ],
                b[b$patient %in% overlap, ],
                by = "patient",
                suffixes = c("_A", "_B")
            )

            cat(
                "\n",
                ids[i], "<->", ids[j],
                "\nShared patients:",
                length(overlap),
                "\nSame response labels:",
                sum(merged$Response_A == merged$Response_B),
                "/", nrow(merged),
                "\n"
            )
        }
    }
}
