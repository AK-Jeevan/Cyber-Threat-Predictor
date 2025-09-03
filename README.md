# 🛡️ Cyber-Threat-Predictor

This project uses machine learning to detect potential cyber intrusions in network sessions. By analyzing features like login attempts, session duration, and IP reputation, the model predicts whether a session is safe or suspicious.

📌 Objective

Train a supervised classification model to identify cyber threats based on network behavior and metadata.

📂 Dataset

Source: cybersecurity_intrusion_data.csv

Target Variable: attack_detected (binary: 0 = safe, 1 = intrusion)

Features: Includes session time, login count, IP reputation, and other behavioral indicators

Preprocessing:

Removed missing and duplicate entries

One-hot encoded categorical variables

Scaled features using StandardScaler

🧠 Model: Support Vector Machine (SVM)

SVM was chosen for its ability to handle high-dimensional data and non-linear decision boundaries. It’s particularly effective for binary classification tasks like intrusion detection.

🔍 Hyperparameter Tuning: 

GridSearchCV
To optimize model performance, we used GridSearchCV to tune:

C: Regularization parameter

kernel: 'linear', 'rbf', 'poly'

gamma: 'scale', 'auto'

python
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
Cross-validation: 3-fold

Scoring metric: Accuracy

📊 Evaluation

Confusion Matrix: Visualized with ConfusionMatrixDisplay

Classification Report: Precision, recall, F1-score

Accuracy: Evaluated on test set

✅ Results

Achieved strong classification performance with tuned SVM

Best parameters selected via GridSearchCV

Model generalizes well to unseen data

🚀 How to Run

python Cyberthreat.py
Make sure the dataset is in the same directory or update the path accordingly.

📌 Future Work

Try RandomizedSearchCV or Bayesian Optimization for faster tuning

Integrate real-time threat detection pipeline

Expand feature set with network flow metrics
