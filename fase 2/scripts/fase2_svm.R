# ============================================================
# fase2_svm.R
# Phase 2 – Model 2: Support Vector Machine (SVM)
# HPO grid: C (3 values) x sigma (3 values) = 9 experiments
# Kernel: RBF (Radial Basis Function)
# ============================================================

library(dplyr)
library(caret)
library(kernlab)

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

# Feature scaling (center + scale), fit on training only
preProc        <- preProcess(train[, bands], method = c("center", "scale"))
train[, bands] <- predict(preProc, train[, bands])
test[, bands]  <- predict(preProc, test[, bands])

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")
cat("Classes:", levels(df$clase), "\n\n")

# ── 2. HPO Grid 3x3 (C x sigma) ──────────────────────────────
control <- trainControl(method = "cv", number = 5)

grid_svm <- expand.grid(
  C     = c(0.1, 1, 10),
  sigma = c(0.01, 0.1, 1)
)

cat("Running SVM HPO (9 combinations x 5-fold CV) — this may take a few minutes...\n")

set.seed(123)
modelo_svm_hpo <- train(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data      = train,
  method    = "svmRadial",
  trControl = control,
  tuneGrid  = grid_svm
)

cat("\n=== SVM HPO Results (3x3 grid: C x sigma) ===\n")
hpo_table <- modelo_svm_hpo$results[, c("C","sigma","Accuracy","Kappa")]
hpo_table$Accuracy <- round(hpo_table$Accuracy * 100, 2)
hpo_table$Kappa    <- round(hpo_table$Kappa, 4)
print(hpo_table)

# ── 3. Train best model ───────────────────────────────────────
best_C     <- modelo_svm_hpo$bestTune$C
best_sigma <- modelo_svm_hpo$bestTune$sigma

cat(sprintf("\nBest params: C = %.1f | sigma = %.2f\n", best_C, best_sigma))

# ── 4. Evaluate on test set ───────────────────────────────────
pred_svm <- predict(modelo_svm_hpo, test)
cm_svm   <- confusionMatrix(pred_svm, test$clase)

cat("\n=== SVM — Final Evaluation ===\n")
print(cm_svm$table)
cat(sprintf("\nOverall Accuracy : %.4f\n", cm_svm$overall["Accuracy"]))
cat(sprintf("Kappa            : %.4f\n", cm_svm$overall["Kappa"]))
cat("\nPer-class metrics:\n")
print(round(cm_svm$byClass[, c("Precision","Recall","F1")], 4))
