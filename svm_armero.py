import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 1. Load data
df = pd.read_csv("training_data.tsv", sep="\t")
bands = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]
X = df[bands].values
y = df["clase"].values
classes = sorted(df["clase"].unique())
print("Classes:", classes)
print("Feature bands:", bands, "->", len(bands), "features")

# 2. Train/test split (stratified, 70/30, seed 42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

# 3. Scale features (essential for SVM)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Train SVM (RBF kernel)
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
svm.fit(X_train_s, y_train)

# 5. Predict + evaluate
y_pred = svm.predict(X_test_s)
oa = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=classes)

print("\n=== Overall Accuracy: %.2f%% ===" % (oa*100))
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print("Labels:", classes)
print(cm)
print("\n", classification_report(y_test, y_pred, digits=3))
