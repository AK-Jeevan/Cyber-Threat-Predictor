# This model predicts whether a network session is safe or if someone might be trying to hack in using Support Vector Machines (SVM) and GridSearchCV.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load and clean data
data = pd.read_csv("C://Users//akjee//Documents//ML//cybersecurity_intrusion_data.csv")
print("Data Size:", data.shape)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print(data.head())

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target
x = data.drop(columns=['attack_detected'])  # Independent variables
y = data['attack_detected']   # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
print(f"X Train shape is :{X_train.shape}")
print(f"X Test shape is :{X_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

# Fit model
grid_search.fit(X_train_scaled, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
best_svc = grid_search.best_estimator_
print(best_svc)

# Predictions
y_pred = best_svc.predict(X_test_scaled)
print("Predictions:", y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svc.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Classification report and accuracy
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Model Accuracy:", grid_search.score(X_test_scaled, y_test))