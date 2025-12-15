
# TEAM.NO.261 MARINE OIL SPILL PREDICTION USING MACHINE LEARNING


## About

This project focuses on predicting marine oil spill occurrences using machine learning techniques. Structured marine environmental data is collected and preprocessed to improve prediction accuracy. Different machine learning algorithms are trained and evaluated to classify oil spill and non-oil spill conditions. The system helps in early detection and supports effective marine environmental protection.


## Features

🌊 Upload marine environmental dataset

🤖 Machine learning–based oil spill prediction

📊 Prediction confidence and accuracy score

🧪 Automatic data preprocessing and feature scaling

📄 Downloadable prediction report

## Development Requirements

Operating System: Windows 10 / Linux

Programming Language: Python 3.x

IDE / Tool: Jupyter Notebook, VS Code

Libraries: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn

Machine Learning Framework: Scikit-learn

## System Architecture

<img width="821" height="345" alt="image" src="https://github.com/user-attachments/assets/ffeff1af-bec2-4b36-ac64-b6b26db98ce9" />


## Methodology
# 1. Data Preprocessing

i) The marine oil spill dataset was collected from publicly available sources and cleaned by removing incomplete, duplicate, and inconsistent records.

ii) Missing values were handled using statistical methods, and all numerical features were normalized using feature scaling techniques to ensure uniformity.

iii) The dataset was then split into training and testing sets to evaluate model performance effectively.

# 2. Model Training

i) Multiple machine learning algorithms were used for oil spill prediction and comparison:

Random Forest Classifier

Decision Tree Classifier

Naive Bayes Classifier

ii) The models were trained on the preprocessed dataset to learn patterns associated with oil spill and non-oil spill conditions.

iii) Hyperparameters were tuned to improve performance, and the best-performing model was selected as the final deployable model.

# 3. Model Evaluation

Evaluation metrics included: accuracy, precision, recall, F1-score, and confusion matrix.

The trained models were compared based on their performance, and the Random Forest classifier demonstrated superior accuracy and reliability.

The final selected model achieved consistent and accurate prediction results on the test dataset.

The final deployed model achieved:

<img width="527" height="625" alt="image" src="https://github.com/user-attachments/assets/d78b972b-dcea-490e-89f1-f536423261c4" />



## Key Model Implementation Code
```
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# LOAD DATASET
data = pd.read_csv("marine_oil_spill_dataset.csv")

# TARGET & FEATURES
X = data.drop("oil_spill", axis=1)
y = data["oil_spill"]

# DATA PREPROCESSING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# MODEL 1: Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# MODEL 2: Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

# ENSEMBLE PREDICTION (Soft Voting)
rf_pred_prob = rf_model.predict_proba(X_test)
gb_pred_prob = gb_model.predict_proba(X_test)

ensemble_prob = (rf_pred_prob + gb_pred_prob) / 2
ensemble_pred = np.argmax(ensemble_prob, axis=1)

# MODEL EVALUATION
print("Accuracy:", accuracy_score(y_test, ensemble_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, ensemble_pred))

# SAVE FINAL DEPLOYED MODEL
joblib.dump(rf_model, "final_oil_spill_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Final Deployed Model Saved Successfully")

```

## Results
The final deployed machine learning model achieved an accuracy of 94%, demonstrating strong performance in predicting marine oil spill occurrences from structured environmental data.

The system effectively classifies oil spill and non-oil spill conditions, enabling early detection and supporting timely environmental monitoring and preventive response actions.

## Output
# model accuracy comparison
<img width="846" height="591" alt="image" src="https://github.com/user-attachments/assets/6dfe13ce-020c-497d-bc59-826291319301" />

<img width="514" height="470" alt="image" src="https://github.com/user-attachments/assets/8f376187-83d4-4fda-8074-960deae65a55" />




## Future Enhancements

🔹 Integrate real-time marine and oceanographic data sources

🔹 Store historical oil spill records using MongoDB/Firebase

🔹 Add batch dataset prediction for large-scale analysis

🔹 Deploy the prediction system on cloud platforms for real-time inference

## References

[1] L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.

[2] T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: Data Mining, Inference, and Prediction, 2nd ed., Springer, 2009.

[3] S. M. Brekke and A. H. Solberg, “Oil spill detection by satellite remote sensing,” Remote Sensing of Environment, vol. 95, no. 1, pp. 1–13, 2005.

[4] J. Fingas and C. Brown, “Review of oil spill remote sensing,” Marine Pollution Bulletin, vol. 83, no. 1, pp. 9–23, 2014.
