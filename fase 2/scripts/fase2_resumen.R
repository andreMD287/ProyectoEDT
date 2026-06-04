# ============================================================
# fase2_resumen.R
# Phase 2 – Comparative summary of all 5 models
# Run AFTER all individual model scripts have been executed
# ============================================================

library(dplyr)
library(caret)
library(rpart)
library(kernlab)
library(nnet)
library(kknn)
library(naivebayes)

# ── EDIT THIS PATH ───────────────────────────────────────────
tsv_file <- "C:/Users/valeh/Downloads/proyecto-emergentes/training_data.tsv"
# ────────────────────────────────────────────────────────────

# ── 1. Load and split data (same seed for all models) ────────
df       <- read.table(tsv_file, sep = "\t", header = TRUE)
df$clase <- as.factor(df$clase)
bands    <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")

set.seed(123)
idx   <- createDataPartition(df$clase, p = 0.70, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]

# Normalized version (for ANN and KNN)
mins <- apply(train[, bands], 2, min)
maxs <- apply(train[, bands], 2, max)
normalize <- function(x, mn, mx) (x - mn) / (mx - mn)
train_n <- train; test_n <- test
for (b in bands) {
  train_n[[b]] <- normalize(train[[b]], mins[b], maxs[b])
  test_n[[b]]  <- normalize(test[[b]],  mins[b], maxs[b])
}

# Scaled version (for SVM)
preProc        <- preProcess(train[, bands], method = c("center","scale"))
train_s        <- train; test_s <- test
train_s[, bands] <- predict(preProc, train[, bands])
test_s[, bands]  <- predict(preProc, test[, bands])

control <- trainControl(method = "cv", number = 5)

# ── 2. Train all 5 models with best parameters ────────────────
cat("Training all models with best parameters...\n\n")

# Decision Tree
set.seed(123)
m_dt  <- rpart(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
               data = train, method = "class",
               control = rpart.control(cp = 0.001, maxdepth = 20))
p_dt  <- predict(m_dt, test, type = "class")
cm_dt <- confusionMatrix(p_dt, test$clase)

# SVM
set.seed(123)
m_svm  <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                data = train_s, method = "svmRadial", trControl = control,
                tuneGrid = expand.grid(C = 10, sigma = 1))
p_svm  <- predict(m_svm, test_s)
cm_svm <- confusionMatrix(p_svm, test_s$clase)

# ANN
set.seed(123)
m_ann  <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                data = train_n, method = "nnet", trControl = control,
                tuneGrid = expand.grid(size = 20, decay = 0.01),
                maxit = 300, trace = FALSE)
p_ann  <- predict(m_ann, test_n)
cm_ann <- confusionMatrix(p_ann, test_n$clase)

# KNN
set.seed(123)
m_knn  <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
                data = train_n, method = "kknn", trControl = control,
                tuneGrid = expand.grid(kmax = 15, distance = 2, kernel = "rectangular"))
p_knn  <- predict(m_knn, test_n)
cm_knn <- confusionMatrix(p_knn, test_n$clase)

# Naive Bayes
set.seed(123)
m_nb  <- train(clase ~ B2+B3+B4+B5+B6+B7+B8+B8A+B11+B12,
               data = train, method = "naive_bayes", trControl = control,
               tuneGrid = expand.grid(laplace = 0, usekernel = TRUE, adjust = 1))
p_nb  <- predict(m_nb, test)
cm_nb <- confusionMatrix(p_nb, test$clase)

# ── 3. Build comparison table ─────────────────────────────────
f1_mean <- function(cm) mean(cm$byClass[, "F1"], na.rm = TRUE)

resumen <- data.frame(
  Model    = c("Decision Tree","SVM","ANN","KNN","Naive Bayes"),
  BestParams = c("cp=0.001, maxdepth=20",
                 "C=10, sigma=1.0",
                 "size=20, decay=0.01",
                 "k=15, Euclidean",
                 "laplace=0, adjust=1.0"),
  Accuracy = c(cm_dt$overall["Accuracy"], cm_svm$overall["Accuracy"],
               cm_ann$overall["Accuracy"], cm_knn$overall["Accuracy"],
               cm_nb$overall["Accuracy"]),
  Kappa    = c(cm_dt$overall["Kappa"],    cm_svm$overall["Kappa"],
               cm_ann$overall["Kappa"],   cm_knn$overall["Kappa"],
               cm_nb$overall["Kappa"]),
  F1_mean  = c(f1_mean(cm_dt), f1_mean(cm_svm), f1_mean(cm_ann),
               f1_mean(cm_knn), f1_mean(cm_nb))
)

resumen$Accuracy <- round(resumen$Accuracy * 100, 2)
resumen$Kappa    <- round(resumen$Kappa, 4)
resumen$F1_mean  <- round(resumen$F1_mean, 4)
resumen <- resumen[order(-resumen$Accuracy), ]

cat("==========================================\n")
cat("       COMPARATIVE MODEL SUMMARY\n")
cat("==========================================\n")
print(resumen, row.names = FALSE)
cat("\nBest model:", resumen$Model[1], "\n")
cat(sprintf("Accuracy: %.2f%% | Kappa: %.4f | F1: %.4f\n",
            resumen$Accuracy[1], resumen$Kappa[1], resumen$F1_mean[1]))
