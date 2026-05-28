# ============================================================
# naive_bayes_armero.R
# Phase 2: Naive Bayes training and evaluation
# Model: Gaussian Naive Bayes
# ============================================================

library(caret)
library(e1071)

# ── Adjust these paths ──────────────────────────────────────
tsv_file   <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/training_data.tsv"
output_dir <- "C:/Users/valeh/OneDrive - Pontificia Universidad Javeriana/2610/Emergentes/Proyecto 2/"
# ────────────────────────────────────────────────────────────

# ============================================================
# 1. Load and prepare data
# ============================================================

df <- read.table(tsv_file, sep = "\t", header = TRUE)
df$clase <- as.factor(df$clase)

bandas <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")

# Check dataset
print(head(df))
print(table(df$clase))

# ============================================================
# 2. Split data: 70% training, 30% testing
# ============================================================

set.seed(123)

idx <- createDataPartition(df$clase, p = 0.7, list = FALSE)

train <- df[idx, ]
test  <- df[-idx, ]

# ============================================================
# 3. Naive Bayes HPO
# ============================================================

control <- trainControl(method = "cv", number = 5)

grid_nb <- expand.grid(
  fL = c(0, 0.5, 1),
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 2)
)

modelo_nb_hpo <- train(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data = train,
  method = "nb",
  trControl = control,
  tuneGrid = grid_nb
)

message("NB HPO results:")
print(modelo_nb_hpo)

# Save HPO plot
png(paste0(output_dir, "nb_hpo.png"), width = 1200, height = 700, res = 150)
plot(modelo_nb_hpo)
dev.off()

message("✓ nb_hpo.png saved")

# ============================================================
# 4. Train optimized Naive Bayes
# ============================================================

best_params <- modelo_nb_hpo$bestTune
print(best_params)

modelo_nb_opt <- train(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data = train,
  method = "nb",
  trControl = trainControl(method = "none"),
  tuneGrid = best_params
)

# ============================================================
# 5. Prediction and evaluation
# ============================================================

pred_nb <- predict(modelo_nb_opt, test)

cm_nb <- confusionMatrix(pred_nb, test$clase)

message("\n=== Naive Bayes Optimized ===")
print(cm_nb)

# ============================================================
# 6. Save confusion matrix as image
# ============================================================

png(paste0(output_dir, "nb_confusion_matrix.png"), width = 900, height = 700, res = 120)

fourfoldplot(
  cm_nb$table,
  color = c("#CC6666", "#99CC99"),
  conf.level = 0,
  margin = 1,
  main = "Confusion Matrix - Naive Bayes"
)

dev.off()

message("✓ nb_confusion_matrix.png saved")

# ============================================================
# 7. Summary
# ============================================================

message("\n=== NAIVE BAYES SUMMARY ===")
message(sprintf("Naive Bayes | Accuracy: %.2f%% | Kappa: %.4f",
                cm_nb$overall["Accuracy"] * 100,
                cm_nb$overall["Kappa"]))
