# ============================================================
# fase2_nb.R
# Phase 2 – Model 5: Naive Bayes (NB)
# HPO grid: laplace (3 values) x adjust (3 values) = 9 experiments
# ============================================================

library(dplyr)
library(caret)
library(naivebayes)

# ── EDIT THIS PATH ───────────────────────────────────────────
tsv_file <- "C:/Users/valeh/Downloads/proyecto-emergentes/training_data.tsv"
# ────────────────────────────────────────────────────────────

# ── 1. Load and prepare data ─────────────────────────────────
df       <- read.table(tsv_file, sep = "\t", header = TRUE)
df$clase <- as.factor(df$clase)

bands <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")

set.seed(123)
idx   <- createDataPartition(df$clase, p = 0.70, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")
cat("Classes:", levels(df$clase), "\n\n")

# ── 2. HPO Grid 3x3 (laplace x adjust) ───────────────────────
# laplace : smoothing parameter for zero-frequency problem
# adjust  : bandwidth multiplier for kernel density estimation
# usekernel = TRUE allows non-Gaussian distributions (better for spectral data)
control <- trainControl(method = "cv", number = 5)

grid_nb <- expand.grid(
  laplace   = c(0, 0.5, 1),
  usekernel = TRUE,
  adjust    = c(0.5, 1, 2)
)

cat("Running NB HPO (9 combinations x 5-fold CV)...\n")

set.seed(123)
modelo_nb_hpo <- train(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data      = train,
  method    = "naive_bayes",
  trControl = control,
  tuneGrid  = grid_nb
)

cat("\n=== NB HPO Results (3x3 grid: laplace x adjust) ===\n")
hpo_table <- modelo_nb_hpo$results[, c("laplace","adjust","Accuracy","Kappa")]
hpo_table$Accuracy <- round(hpo_table$Accuracy * 100, 2)
hpo_table$Kappa    <- round(hpo_table$Kappa, 4)
print(hpo_table)

# ── 3. Best parameters ────────────────────────────────────────
best_laplace <- modelo_nb_hpo$bestTune$laplace
best_adjust  <- modelo_nb_hpo$bestTune$adjust

cat(sprintf("\nBest params: laplace = %.1f | adjust = %.1f\n", best_laplace, best_adjust))

# ── 4. Evaluate on test set ───────────────────────────────────
pred_nb <- predict(modelo_nb_hpo, test)
cm_nb   <- confusionMatrix(pred_nb, test$clase)

cat("\n=== Naive Bayes — Final Evaluation ===\n")
print(cm_nb$table)
cat(sprintf("\nOverall Accuracy : %.4f\n", cm_nb$overall["Accuracy"]))
cat(sprintf("Kappa            : %.4f\n", cm_nb$overall["Kappa"]))
cat("\nPer-class metrics:\n")
print(round(cm_nb$byClass[, c("Precision","Recall","F1")], 4))
