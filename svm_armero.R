# ============================================================
#  SVM Land Cover Classification - Armero
#  Methodology based on Lab 06 (caret + kernlab)
# ============================================================

# --- Libraries (same as Lab 06) ---
library(dplyr)     # data wrangling
library(caret)     # training / pre-processing / confusion matrix
library(kernlab)   # ksvm for fitting SVMs

# --- 1. Load data ---
df <- read.delim("training_data.tsv", sep = "\t")
df$clase <- as.factor(df$clase)

bands <- c("B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12")
cat("Classes:", levels(df$clase), "\n")
cat("Total samples:", nrow(df), "| Features:", length(bands), "\n\n")

# --- 2. Train/Test split (70/30, stratified, reproducible) ---
set.seed(42)
idx <- createDataPartition(df$clase, p = 0.70, list = FALSE)
training_set <- df[idx, ]
test_set     <- df[-idx, ]
cat("Train:", nrow(training_set), "| Test:", nrow(test_set), "\n\n")

# --- 3. Feature scaling (center + scale), fit on training only ---
preProc <- preProcess(training_set[, bands], method = c("center", "scale"))
training_set[, bands] <- predict(preProc, training_set[, bands])
test_set[, bands]     <- predict(preProc, test_set[, bands])

# Formula: clase ~ B2 + B3 + ... + B12
fml <- as.formula(paste("clase ~", paste(bands, collapse = " + ")))

# --- 4. Train SVM with three kernels (as in Lab 06) ---
kernels <- c(Linear = "vanilladot", Polynomial = "polydot", RBF = "rbfdot")
acc_results <- data.frame(Kernel = character(), Accuracy = numeric())
models <- list()

for (kname in names(kernels)) {
  set.seed(42)
  m <- ksvm(fml, data = training_set, kernel = kernels[[kname]])
  pred <- predict(m, test_set[, bands])
  cm <- confusionMatrix(pred, test_set$clase)
  acc <- as.numeric(cm$overall["Accuracy"])
  acc_results <- rbind(acc_results,
                       data.frame(Kernel = kname, Accuracy = round(acc * 100, 2)))
  models[[kname]] <- list(model = m, pred = pred, cm = cm)
  cat(sprintf("Kernel %-11s Accuracy: %.2f%%\n", kname, acc * 100))
}

cat("\n===== Accuracy comparison by kernel =====\n")
print(acc_results)

# --- 5. Detailed analysis of the best kernel ---
best <- acc_results$Kernel[which.max(acc_results$Accuracy)]
cat("\n===== Best kernel:", best, "=====\n")
cm_best <- models[[best]]$cm

cat("\n--- Confusion Matrix (rows = Prediction, cols = Reference) ---\n")
print(cm_best$table)

cat("\n--- Overall statistics ---\n")
cat(sprintf("Overall Accuracy: %.4f\n", cm_best$overall["Accuracy"]))
cat(sprintf("Kappa: %.4f\n", cm_best$overall["Kappa"]))

cat("\n--- Per-class metrics ---\n")
# caret reports Precision / Recall / F1 in byClass
per_class <- cm_best$byClass[, c("Precision", "Recall", "F1")]
print(round(per_class, 4))

# --- 6. Hyperparameter tuning for RBF (10-fold CV) -> plot ---
set.seed(42)
svm_tune <- train(
  fml,
  data = training_set,
  method = "svmRadial",
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)
cat("\n===== RBF hyperparameter tuning (10-fold CV) =====\n")
print(svm_tune$results[, c("sigma", "C", "Accuracy", "Kappa")])
cat("\nBest tune:\n"); print(svm_tune$bestTune)

# Save tuning plot
png("/home/claude/svm_tuning.png", width = 1400, height = 900, res = 180)
print(ggplot(svm_tune) + theme_light() +
        ggtitle("SVM RBF Hyperparameter Tuning (Armero)"))
dev.off()
cat("\nTuning plot saved.\n")
