# ============================================================
# fase3_inferencia.R
# Phase 3 – Full-scene inference with the selected ANN model
#
# What this script does:
#   1. Re-trains the ANN (size=20, decay=0.01) on the full
#      training split — same seed and normalization as Phase 2
#   2. Loads armero_recortado.tif and normalizes it
#   3. Predicts a land-cover class for every pixel
#   4. Exports the result as a georeferenced GeoTIFF
#   5. Prints area statistics per class
#
# Output: armero_predicho.tif
#   Pixel codes:  1 = cultivos
#                 2 = suelo_desnudo
#                 3 = urbano
#                 4 = vegetacion
#                 NA = no-data
# ============================================================

library(terra)
library(caret)
library(nnet)

# ── EDIT THESE PATHS ─────────────────────────────────────────
tsv_file <- "C:/Javeriana/8vo semestre/Tecnologias Emergentes/Proyecto Final/datos/training_data.tsv"
raster_file <- "C:/Javeriana/8vo semestre/Tecnologias Emergentes/Proyecto Final/datos/armero_recortado.tif"
output_file <- "C:/Javeriana/8vo semestre/Tecnologias Emergentes/Proyecto Final/datos/armero_predicho.tif"
# ─────────────────────────────────────────────────────────────

bands <- c("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")

# ── 1. Load training data — identical setup to fase2_resumen.R ──
cat("Loading training data...\n")
df <- read.table(tsv_file, sep = "\t", header = TRUE)
df$clase <- as.factor(df$clase)

# Same seed and split as Phase 2
set.seed(123)
idx <- createDataPartition(df$clase, p = 0.70, list = FALSE)
train <- df[idx, ]

# Normalization parameters — MUST come from the training split only
mins <- apply(train[, bands], 2, min)
maxs <- apply(train[, bands], 2, max)
normalize <- function(x, mn, mx) (x - mn) / (mx - mn)

train_n <- train
for (b in bands) {
    train_n[[b]] <- normalize(train[[b]], mins[b], maxs[b])
}

cat(
    "Training split:", nrow(train_n), "pixels |",
    "Classes:", paste(levels(df$clase), collapse = ", "), "\n\n"
)

# ── 2. Re-train ANN with best parameters (no CV needed) ──────
# trainControl(method = "none") trains once with the given
# tuneGrid — no cross-validation, much faster than Phase 2 HPO.
cat("Training ANN (size=20, decay=0.01) — may take ~1 minute...\n")
ctrl <- trainControl(method = "none")

set.seed(123)
m_ann <- train(
    clase ~ B2 + B3 + B4 + B5 + B6 + B7 + B8 + B8A + B11 + B12,
    data      = train_n,
    method    = "nnet",
    trControl = ctrl,
    tuneGrid  = expand.grid(size = 20, decay = 0.01),
    maxit     = 300,
    trace     = FALSE
)
cat("ANN trained successfully.\n\n")

# ── 3. Load raster ────────────────────────────────────────────
cat("Loading raster:", raster_file, "\n")
r <- rast(raster_file)

# Assign band names so they match the model's formula
names(r) <- bands
cat(
    "Raster dimensions:", nrow(r), "rows x", ncol(r), "cols |",
    "CRS:", crs(r, describe = TRUE)$name, "\n"
)

# ── 4. Normalize raster using training mins/maxs ──────────────
cat("Normalizing raster bands...\n")
r_norm <- r
for (b in bands) {
    r_norm[[b]] <- (r[[b]] - mins[b]) / (maxs[b] - mins[b])
}

# ── 5. Convert to data frame for prediction ───────────────────
# na.rm = FALSE keeps ALL pixels (including NA) so we can
# rebuild the raster at the original dimensions afterwards.
cat("Converting raster to data frame...\n")
df_r <- as.data.frame(r_norm, xy = TRUE, na.rm = FALSE)

# Identify pixels with valid data across all bands
valid <- complete.cases(df_r[, bands])
n_valid <- sum(valid)
cat("Valid pixels to classify:", format(n_valid, big.mark = ","), "\n\n")

# ── 6. Run inference ──────────────────────────────────────────
cat("Running inference...\n")
preds <- predict(m_ann, df_r[valid, bands])

# Convert factor labels to integers (alphabetical order):
#   1 = cultivos  |  2 = suelo_desnudo  |  3 = urbano  |  4 = vegetacion
pred_int <- as.integer(preds)

# ── 7. Rebuild output raster ──────────────────────────────────
out_vals <- rep(NA_integer_, nrow(df_r))
out_vals[valid] <- pred_int

out_rast <- rast(r, nlyr = 1) # same extent, resolution, CRS as input
values(out_rast) <- out_vals
names(out_rast) <- "clase"

# ── 8. Export GeoTIFF ─────────────────────────────────────────
# INT1U = unsigned 8-bit integer (values 0-255); ideal for 4 classes.
# Preserves the original CRS and affine transform automatically.
cat("Saving output raster to:", output_file, "\n")
writeRaster(out_rast, output_file, datatype = "INT1U", overwrite = TRUE)
cat("Done.\n\n")

# ── 9. Area statistics ────────────────────────────────────────
# At 10 m resolution: 1 pixel = 10 x 10 m = 100 m² = 0.01 ha
cat("============================================\n")
cat("       AREA PER CLASS (10 m resolution)\n")
cat("============================================\n")

freq_table <- as.data.frame(freq(out_rast))
class_names <- c(
    "1" = "cultivos",
    "2" = "suelo_desnudo",
    "3" = "urbano",
    "4" = "vegetacion"
)

freq_table$clase <- class_names[as.character(freq_table$value)]
freq_table$area_ha <- round(freq_table$count * 0.01, 2)
freq_table$area_km2 <- round(freq_table$area_ha / 100, 4)
freq_table$pct <- round(freq_table$count / sum(freq_table$count) * 100, 2)

print(freq_table[, c("value", "clase", "count", "area_ha", "area_km2", "pct")],
    row.names = FALSE
)

cat(
    "\nTotal classified area:",
    round(sum(freq_table$area_ha), 2), "ha /",
    round(sum(freq_table$area_km2), 4), "km²\n"
)
