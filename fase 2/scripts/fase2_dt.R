# ============================================================
# fase2_dt.R
# Phase 2 – Model 1: Decision Tree (DT)
# HPO grid: cp (3 values) x maxdepth (3 values) = 9 experiments
# ============================================================

library(dplyr)
library(caret)
library(rpart)

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

# ── 2. HPO Grid 3x3 (cp x maxdepth) ─────────────────────────
control <- trainControl(method = "cv", number = 5)

grid_dt <- expand.grid(
  cp = c(0.001, 0.01, 0.1)
)

# caret's rpart only tunes cp; maxdepth is tested manually below
# We run the 3x3 by looping over maxdepth values

results_hpo <- data.frame()

for (md in c(5, 10, 20)) {
  set.seed(123)
  m <- train(
    clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
    data      = train,
    method    = "rpart",
    trControl = control,
    tuneGrid  = grid_dt,
    control   = rpart.control(maxdepth = md)
  )
  # Store ALL 3 cp results for this maxdepth (not just the best)
  for (i in 1:nrow(m$results)) {
    results_hpo <- rbind(results_hpo, data.frame(
      maxdepth = md,
      cp       = m$results$cp[i],
      Accuracy = round(m$results$Accuracy[i] * 100, 2),
      Kappa    = round(m$results$Kappa[i], 4)
    ))
  }
}

cat("=== DT HPO Results (3x3 grid: cp x maxdepth) ===\n")
print(results_hpo)

# ── 3. Train best model ──────────────────────────────────────
# Pick combination with highest CV Accuracy
best_row  <- results_hpo[which.max(results_hpo$Accuracy), ]
best_cp   <- best_row$cp
best_md   <- best_row$maxdepth

cat(sprintf("\nBest params: cp = %.3f | maxdepth = %d\n", best_cp, best_md))

modelo_dt <- rpart(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data    = train,
  method  = "class",
  control = rpart.control(cp = best_cp, maxdepth = best_md)
)

# ── 4. Evaluate on test set ──────────────────────────────────
pred_dt <- predict(modelo_dt, test, type = "class")
cm_dt   <- confusionMatrix(pred_dt, test$clase)

cat("\n=== Decision Tree — Final Evaluation ===\n")
print(cm_dt$table)
cat(sprintf("\nOverall Accuracy : %.4f\n", cm_dt$overall["Accuracy"]))
cat(sprintf("Kappa            : %.4f\n", cm_dt$overall["Kappa"]))
cat("\nPer-class metrics:\n")
print(round(cm_dt$byClass[, c("Precision","Recall","F1")], 4))
