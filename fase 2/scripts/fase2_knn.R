# ============================================================
# fase2_knn.R
# Phase 2 – Model 4: K-Nearest Neighbor (KNN)
# HPO grid: k (3 values) x distance metric (3 values) = 9 experiments
# Uses kknn method which supports both parameters
# ============================================================

library(dplyr)
library(caret)
library(kknn)

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

# Min-max normalization (distances are sensitive to scale)
mins <- apply(train[, bands], 2, min)
maxs <- apply(train[, bands], 2, max)

normalize <- function(x, mn, mx) (x - mn) / (mx - mn)

for (b in bands) {
  train[[b]] <- normalize(train[[b]], mins[b], maxs[b])
  test[[b]]  <- normalize(test[[b]],  mins[b], maxs[b])
}

cat("Train:", nrow(train), "| Test:", nrow(test), "\n")
cat("Classes:", levels(df$clase), "\n\n")

# ── 2. HPO Grid 3x3 (k x distance) ───────────────────────────
# distance: 1 = Manhattan, 2 = Euclidean, 3 = Minkowski p=3
control <- trainControl(method = "cv", number = 5)

grid_knn <- expand.grid(
  kmax     = c(3, 7, 15),
  distance = c(1, 2, 3),
  kernel   = "rectangular"
)

cat("Running KNN HPO (9 combinations x 5-fold CV)...\n")

set.seed(123)
modelo_knn_hpo <- train(
  clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
  data      = train,
  method    = "kknn",
  trControl = control,
  tuneGrid  = grid_knn
)

cat("\n=== KNN HPO Results (3x3 grid: k x distance) ===\n")
hpo_table <- modelo_knn_hpo$results[, c("kmax","distance","Accuracy","Kappa")]
hpo_table$Accuracy  <- round(hpo_table$Accuracy * 100, 2)
hpo_table$Kappa     <- round(hpo_table$Kappa, 4)
hpo_table$dist_name <- ifelse(hpo_table$distance == 1, "Manhattan",
                        ifelse(hpo_table$distance == 2, "Euclidean", "Minkowski"))
print(hpo_table[, c("kmax","dist_name","Accuracy","Kappa")])

# ── 3. Best parameters ────────────────────────────────────────
best_k    <- modelo_knn_hpo$bestTune$kmax
best_dist <- modelo_knn_hpo$bestTune$distance
dist_name <- ifelse(best_dist == 1, "Manhattan",
              ifelse(best_dist == 2, "Euclidean", "Minkowski"))

cat(sprintf("\nBest params: k = %d | distance = %s\n", best_k, dist_name))

# ── 4. Evaluate on test set ───────────────────────────────────
pred_knn <- predict(modelo_knn_hpo, test)
cm_knn   <- confusionMatrix(pred_knn, test$clase)

cat("\n=== KNN — Final Evaluation ===\n")
print(cm_knn$table)
cat(sprintf("\nOverall Accuracy : %.4f\n", cm_knn$overall["Accuracy"]))
cat(sprintf("Kappa            : %.4f\n", cm_knn$overall["Kappa"]))
cat("\nPer-class metrics:\n")
print(round(cm_knn$byClass[, c("Precision","Recall","F1")], 4))
