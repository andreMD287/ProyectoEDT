# ============================================================
# fase2_modelos.R
# Phase 2: Model training, HPO, and evaluation
# Models: Decision Tree (DT), Artificial Neural Network (ANN), KNN, and Naive Bayes (NB)
# ============================================================

library(terra)
library(dplyr)
library(rpart)
library(nnet)
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

set.seed(123)
idx   <- createDataPartition(df$clase, p = 0.7, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

bandas <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")

df_norm <- df
df_norm[bandas] <- lapply(df[bandas], function(x) (x - min(x)) / (max(x) - min(x)))

set.seed(123)
idx_n   <- createDataPartition(df_norm$clase, p = 0.7, list = FALSE)
train_n <- df_norm[idx_n, ]
test_n  <- df_norm[-idx_n, ]

control <- trainControl(method = "cv", number = 5)

# ============================================================
# 2. Decision Tree (DT)
# ============================================================

grid_dt <- expand.grid(cp = c(0.001, 0.005, 0.01, 0.05, 0.1))

modelo_dt_hpo <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data      = train,
                       method    = "rpart",
                       trControl = control,
                       tuneGrid  = grid_dt)

message("DT HPO results:")
print(modelo_dt_hpo)

png(paste0(output_dir, "dt_hpo.png"), width = 1200, height = 700, res = 150)
plot(modelo_dt_hpo)
dev.off()

modelo_dt_opt <- rpart(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data    = train,
                       method  = "class",
                       control = rpart.control(cp = 0.001))

pred_dt <- predict(modelo_dt_opt, test, type = "class")
cm_dt   <- confusionMatrix(pred_dt, test$clase)

message("\n=== Decision Tree optimized ===")
print(cm_dt)

# ============================================================
# 3. Artificial Neural Network (ANN)
# ============================================================

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

modelo_ann_opt <- nnet(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data  = train_n,
                       size  = 20,
                       decay = 0.01,
                       maxit = 300,
                       trace = FALSE)

pred_ann <- as.factor(predict(modelo_ann_opt, test_n, type = "class"))
cm_ann   <- confusionMatrix(pred_ann, test_n$clase)

message("\n=== ANN optimized ===")
print(cm_ann)

# ============================================================
# 4. K-Nearest Neighbor (KNN)
# ============================================================

grid_knn <- expand.grid(k = c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21))

modelo_knn_hpo <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                        data      = train_n,
                        method    = "knn",
                        trControl = control,
                        tuneGrid  = grid_knn)

message("\nKNN HPO results:")
print(modelo_knn_hpo)

pred_knn <- predict(modelo_knn_hpo, test_n)
cm_knn   <- confusionMatrix(pred_knn, test_n$clase)

message("\n=== KNN optimized ===")
print(cm_knn)

# ============================================================
# 5. Naive Bayes (NB)
# ============================================================

grid_nb <- expand.grid(
  fL = c(0, 0.5, 1),
  usekernel = c(TRUE, FALSE),
  adjust = c(0.5, 1, 2)
)

modelo_nb_hpo <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data      = train,
                       method    = "nb",
                       trControl = control,
                       tuneGrid  = grid_nb)

message("\nNB HPO results:")
print(modelo_nb_hpo)

png(paste0(output_dir, "nb_hpo.png"), width = 1200, height = 700, res = 150)
plot(modelo_nb_hpo)
dev.off()

best_nb <- modelo_nb_hpo$bestTune

modelo_nb_opt <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                       data      = train,
                       method    = "nb",
                       trControl = trainControl(method = "none"),
                       tuneGrid  = best_nb)

pred_nb <- predict(modelo_nb_opt, test)
cm_nb   <- confusionMatrix(pred_nb, test$clase)

message("\n=== Naive Bayes optimized ===")
print(cm_nb)

# ============================================================
# 6. Summary
# ============================================================

message("\n=== COMPARATIVE SUMMARY ===")

message(sprintf("Decision Tree  | Accuracy: %.2f%% | Kappa: %.4f",
                cm_dt$overall["Accuracy"] * 100,
                cm_dt$overall["Kappa"]))

message(sprintf("ANN            | Accuracy: %.2f%% | Kappa: %.4f",
                cm_ann$overall["Accuracy"] * 100,
                cm_ann$overall["Kappa"]))

message(sprintf("KNN            | Accuracy: %.2f%% | Kappa: %.4f",
                cm_knn$overall["Accuracy"] * 100,
                cm_knn$overall["Kappa"]))

message(sprintf("Naive Bayes    | Accuracy: %.2f%% | Kappa: %.4f",
                cm_nb$overall["Accuracy"] * 100,
                cm_nb$overall["Kappa"]))