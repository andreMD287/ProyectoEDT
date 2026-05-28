# ============================================================
# fase1_dataset.R
# Phase 1 - Step 2: Training dataset creation
# Extracts pixel values from training polygons and exports TSV
# ============================================================

library(terra)
library(dplyr)

# ── Adjust these paths ──────────────────────────────────────
raster_file    <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/armero_recortado.tif"
poligonos_file <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 1/PROYECTO/muestras_armero.gpkg"
output_file    <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/training_data.tsv"
# ────────────────────────────────────────────────────────────

# Load clipped raster and training polygons
r    <- rast(raster_file)
pols <- vect(poligonos_file)

# Reproject polygons to match raster CRS
pols <- project(pols, crs(r))

# Assign band names
names(r) <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")

# Extract pixel values within each polygon (including coordinates)
extracted <- extract(r, pols, xy = TRUE, ID = TRUE)
extracted$clase <- pols$clase[extracted$ID]

# Keep only relevant columns and remove NA rows
df <- extracted[, c("x","y","B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","clase")]
df <- na.omit(df)

# Show available pixels per class before sampling
message("Available pixels per class:")
print(table(df$clase))

# Balanced random sampling: 2000 pixels per class (8000 total)
set.seed(123)
df_balanced <- df %>%
  group_by(clase) %>%
  slice_sample(n = 2000) %>%
  ungroup()

# Export as Tab-Separated Values (TSV)
write.table(df_balanced, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
message("✓ Dataset saved: ", output_file)
message(paste("✓ Total pixels:", nrow(df_balanced)))
print(table(df_balanced$clase))
