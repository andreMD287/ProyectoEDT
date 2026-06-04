# ============================================================
# fase2_ann.R
# Phase 2 – Model 3: Artificial Neural Network (ANN)
# HPO grid: size (3 values) x decay (3 values) = 9 experiments
# ============================================================

library(dplyr)
library(caret)
library(nnet)

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

# Min-max normalization to [0,1] — required for ANN
mins <- apply(train[, bands], 2, min)
maxs <- apply(train[, bands], 2, max)

normalize <- function(x, mn, mx) (x - mn) / (mx - mn)

for (b in bands) {
  train[[b]] <- normalize(train[[b]], mins[b], maxs[b])
  test[[b]]  <- normalize(test[[b]],  mins[b], maxs[b])
}

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")
cat("Classes:", levels(df$clase), "\n\n")

# ── 2. HPO Grid 3x3 (size x decay) ───────────────────────────
control <- trainControl(method = "cv", number = 5)

grid_ann <- expand.grid(
  size  = c(5, 10, 20),
  decay = c(0.001, 0.01, 0.1)
)

cat("Running ANN HPO (9 combinations x 5-fold CV) — this may take several minutes...\n")

set.seed(123)
modelo_ann_hpo <- train(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data      = train,
  method    = "nnet",
  trControl = control,
  tuneGrid  = grid_ann,
  maxit     = 300,
  trace     = FALSE
)

cat("\n=== ANN HPO Results (3x3 grid: size x decay) ===\n")
hpo_table <- modelo_ann_hpo$results[, c("size","decay","Accuracy","Kappa")]
hpo_table$Accuracy <- round(hpo_table$Accuracy * 100, 2)
hpo_table$Kappa    <- round(hpo_table$Kappa, 4)
print(hpo_table)

# ── 3. Best parameters ────────────────────────────────────────
best_size  <- modelo_ann_hpo$bestTune$size
best_decay <- modelo_ann_hpo$bestTune$decay

cat(sprintf("\nBest params: size = %d | decay = %.3f\n", best_size, best_decay))

# ── 4. Evaluate on test set ───────────────────────────────────
pred_ann <- predict(modelo_ann_hpo, test)
cm_ann   <- confusionMatrix(pred_ann, test$clase)

cat("\n=== ANN — Final Evaluation ===\n")
print(cm_ann$table)
cat(sprintf("\nOverall Accuracy : %.4f\n", cm_ann$overall["Accuracy"]))
cat(sprintf("Kappa            : %.4f\n", cm_ann$overall["Kappa"]))
cat("\nPer-class metrics:\n")
print(round(cm_ann$byClass[, c("Precision","Recall","F1")], 4))
