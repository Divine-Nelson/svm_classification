import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# --- Load MNIST ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(np.uint8)

# --- Split ---
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# --- Create Pipeline ---
svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', svm.SVC(kernel="linear" ,random_state=42))
])

# --- Correct Parameter Grid ---
param_grid = {
    'svm__C': [1, 10, 100, 1000, 10000]
}

# --- Randomized Search ---

rnd_search = RandomizedSearchCV(
    svm_clf,
    param_distributions=param_grid,
    n_iter=5,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2
)


# --- Timing Training ---
start_time = time.time()
rnd_search.fit(X_train, y_train)
end_time = time.time()

print(f"\nGrid Search Runtime: {end_time - start_time:.2f} seconds")
print("Best parameters:", rnd_search.best_params_)
print("Best CV Accuracy:", rnd_search.best_score_)

# --- Evaluate on Test Set ---
best_svm = rnd_search.best_estimator_

start_pred_time = time.time()
y_pred = best_svm.predict(X_test)
end_pred_time = time.time()

test_acc = accuracy_score(y_test, y_pred)
print(f"\nPrediction Runtime: {end_pred_time - start_pred_time:.2f} seconds")
print("Test Accuracy:", test_acc)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Blues', annot=False)
plt.title('Confusion Matrix - SVM (Linear Kernel)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
