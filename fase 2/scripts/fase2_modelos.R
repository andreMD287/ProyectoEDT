# ============================================================
# fase2_modelos.R
# Phase 2: Model training, HPO, and evaluation
# Models: Decision Tree (DT) and Artificial Neural Network (ANN)
# ============================================================

library(terra)
library(dplyr)
library(rpart)
library(nnet)
library(caret)

# ── Adjust these paths ──────────────────────────────────────
tsv_file   <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/training_data.tsv"
output_dir <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/"
# ────────────────────────────────────────────────────────────

# ============================================================
# 1. Load and prepare data
# ============================================================

df <- read.table(tsv_file, sep = "\t", header = TRUE)
df$clase <- as.factor(df$clase)

# Split: 70% training, 30% validation
set.seed(123)
idx   <- createDataPartition(df$clase, p = 0.7, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

# Normalized version for ANN (min-max scaling to [0,1])
bandas <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")
df_norm <- df
df_norm[bandas] <- lapply(df[bandas], function(x) (x - min(x)) / (max(x) - min(x)))

set.seed(123)
idx_n   <- createDataPartition(df_norm$clase, p = 0.7, list = FALSE)
train_n <- df_norm[idx_n, ]
test_n  <- df_norm[-idx_n, ]

# 5-fold cross-validation control
control <- trainControl(method = "cv", number = 5)

# ============================================================
# 2. Decision Tree (DT)
# ============================================================

# --- HPO ---
grid_dt <- expand.grid(cp = c(0.001, 0.005, 0.01, 0.05, 0.1))

modelo_dt_hpo <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data      = train,
                       method    = "rpart",
                       trControl = control,
                       tuneGrid  = grid_dt)

message("DT HPO results:")
print(modelo_dt_hpo)

# Save HPO plot
png(paste0(output_dir, "dt_hpo.png"), width = 1200, height = 700, res = 150)
plot(modelo_dt_hpo)
dev.off()
message("✓ dt_hpo.png saved")

# --- Train optimized DT (cp = 0.001) ---
modelo_dt_opt <- rpart(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data    = train,
                       method  = "class",
                       control = rpart.control(cp = 0.001))

pred_dt <- predict(modelo_dt_opt, test, type = "class")
cm_dt   <- confusionMatrix(pred_dt, test$clase)

message("\n=== Decision Tree (optimized, cp=0.001) ===")
print(cm_dt)

# ============================================================
# 3. Artificial Neural Network (ANN)
# ============================================================

# --- HPO ---
grid_ann <- expand.grid(size  = c(5, 10, 15, 20),
                        decay = c(0.001, 0.01, 0.1))

modelo_ann_hpo <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                        data      = train_n,
                        method    = "nnet",
                        trControl = control,
                        tuneGrid  = grid_ann,
                        maxit     = 300,
                        trace     = FALSE)

message("\nANN HPO results:")
print(modelo_ann_hpo)

# --- Train optimized ANN (size=20, decay=0.01) ---
modelo_ann_opt <- nnet(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data  = train_n,
                       size  = 20,
                       decay = 0.01,
                       maxit = 300,
                       trace = FALSE)

pred_ann <- as.factor(predict(modelo_ann_opt, test_n, type = "class"))
cm_ann   <- confusionMatrix(pred_ann, test_n$clase)

message("\n=== ANN (optimized, size=20, decay=0.01) ===")
print(cm_ann)

# ============================================================
# 4. Summary
# ============================================================

message("\n=== COMPARATIVE SUMMARY ===")
message(sprintf("Decision Tree  | Accuracy: %.2f%% | Kappa: %.4f",
                cm_dt$overall["Accuracy"] * 100,
                cm_dt$overall["Kappa"]))
message(sprintf("ANN            | Accuracy: %.2f%% | Kappa: %.4f",
                cm_ann$overall["Accuracy"] * 100,
                cm_ann$overall["Kappa"]))
