# ============================================================
# fase1_stack.R
# Phase 1 - Step 1: Band stacking
# Loads Sentinel-2 L2A bands and creates a multi-band GeoTIFF
# ============================================================

library(terra)

# ── Adjust these paths ──────────────────────────────────────
ruta_r10 <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/S2A_MSIL2A_20240815T152651_N0511_R025_T18NWL_20240815T223702.SAFE/GRANULE/L2A_T18NWL_A047784_20240815T152653/IMG_DATA/R10m/"
ruta_r20 <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/S2A_MSIL2A_20240815T152651_N0511_R025_T18NWL_20240815T223702.SAFE/GRANULE/L2A_T18NWL_A047784_20240815T152653/IMG_DATA/R20m/"
salida   <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/armero_stack.tif"
# ────────────────────────────────────────────────────────────

# Load 10m bands natively
B2 <- rast(list.files(ruta_r10, pattern = "B02", full.names = TRUE))
B3 <- rast(list.files(ruta_r10, pattern = "B03", full.names = TRUE))
B4 <- rast(list.files(ruta_r10, pattern = "B04", full.names = TRUE))
B8 <- rast(list.files(ruta_r10, pattern = "B08_", full.names = TRUE))

# Load 20m bands
B5  <- rast(list.files(ruta_r20, pattern = "B05", full.names = TRUE))
B6  <- rast(list.files(ruta_r20, pattern = "B06", full.names = TRUE))
B7  <- rast(list.files(ruta_r20, pattern = "B07", full.names = TRUE))
B8A <- rast(list.files(ruta_r20, pattern = "B8A", full.names = TRUE))
B11 <- rast(list.files(ruta_r20, pattern = "B11", full.names = TRUE))
B12 <- rast(list.files(ruta_r20, pattern = "B12", full.names = TRUE))

# Resample 20m bands to 10m using bilinear interpolation
ref <- B2
B5  <- resample(B5,  ref, method = "bilinear")
B6  <- resample(B6,  ref, method = "bilinear")
B7  <- resample(B7,  ref, method = "bilinear")
B8A <- resample(B8A, ref, method = "bilinear")
B11 <- resample(B11, ref, method = "bilinear")
B12 <- resample(B12, ref, method = "bilinear")

# Create 10-band stack and assign names
stack <- c(B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
names(stack) <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")

# Save as GeoTIFF
writeRaster(stack, salida, overwrite = TRUE)
message("✓ Stack created: ", salida)
