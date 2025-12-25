# TEAM NO : 261 - MARINE OIL SPILL PREDICTION USING MACHINE LEARNING

## PROBLEM STATEMENT

Oil spills are a major environmental hazard that severely affect marine ecosystems, coastal biodiversity, and human livelihoods. Traditional monitoring methods are often time-consuming and may fail to provide early warnings. This project aims to develop an automated machine learning-based system to detect oil spills using satellite-derived numerical data, enabling faster and more reliable environmental monitoring.

## PROJECT DESCRIPTION

This project implements and compares multiple machine learning classification algorithms to identify oil spill occurrences from non-spill cases. The focus is on analyzing model performance using standard evaluation metrics and selecting the most suitable model for oil spill detection.

The project is intended for academic and research purposes, with emphasis on experimentation, evaluation, and result analysis rather than deployment.

In addition to the tabular machine learning approach, a single Synthetic Aperture Radar (SAR) satellite image was analyzed as an exploratory experiment to demonstrate image-based feature extraction using a pretrained CNN model. This image was used only for feature extraction and confidence estimation and was not included in the training dataset.

## DATASET

Type: Numerical satellite-derived data

Total samples: 936

Number of features: 49

Target variable:

1 → Oil Spill

0 → Non-Oil Spill

Dataset characteristics:

Real-world environmental data

Class imbalance present

The dataset is preprocessed before model training.

## FEATURES

1. Uses satellite-derived numerical data for oil spill detection
2. Implements multiple machine learning classification algorithms
3. Performs comparative analysis to identify the best-performing model
4. Handles imbalanced dataset characteristics
5. Evaluates models using accuracy, precision, recall, and F1-score
6. Generates confusion matrix for classification performance analysis
7. Plots ROC curve and precision–recall curve for model evaluation
8. Analyzes feature importance to identify key contributing attributes
9. Ensures reproducibility through notebook-based implementation

## DEVELOPMENT REQUIREMENTS

<img width="824" height="483" alt="image" src="https://github.com/user-attachments/assets/e03c0018-de2c-4d37-b580-b90627ba43a7" />

## SYSTEM ARCHITECTURE

<img width="861" height="402" alt="diagram-export-12-17-2025-11_58_38-PM" src="https://github.com/user-attachments/assets/88d041d4-9ab4-4be1-b81b-fbc7663bcf4e" />

## METHODOLOGY

The proposed system follows a structured machine learning workflow for oil spill detection using satellite-derived numerical data. The methodology consists of the following steps:

1. Data Collection
   
   The oil spill dataset containing numerical satellite-derived features is loaded into the Google Colab environment.

2. Data Preprocessing
   
   1. Removal of irrelevant columns
   2. Handling missing or inconsistent values
   3. Separation of input features and target variable
   4. Splitting the dataset into training and testing sets

3. Model Implementation
   
   Multiple machine learning classification algorithms are implemented, including:
   
   1. Logistic Regression
   2. K-Nearest Neighbors
   3. Naive Bayes
   4. Decision Tree
   5. Random Forest
   6. Gradient Boosting
   7. AdaBoost
   8. Support Vector Machine
   9. XGBoost
   10. Extra Trees Classifier

4. Model Training
   
   Each model is trained using the training dataset to learn patterns associated with oil spill and non-oil spill conditions.

5. Model Evaluation
   
   The trained models are evaluated using the testing dataset based on:
   
   1. Accuracy
   2. Precision
   3. Recall
   4. F1-score

6. Model Comparison and Selection
   
   Performance metrics of all models are compared to identify the most suitable model for oil spill detection.

7. Result Visualization
   
   The best-performing model is further analyzed using:
   
   Confusion Matrix
   ROC Curve
   Precision–Recall Curve
   Feature Importance Plot

8. Final Analysis
    
   The results are analyzed to assess the effectiveness of machine learning techniques for oil spill detection.

TABULAR FORMAT:

<img width="831" height="492" alt="image" src="https://github.com/user-attachments/assets/4fdcfe39-dd0e-4960-aab1-35778a39d601" />

## KEY MODEL IMPLEMENTATION CODE

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("oil_spill.csv")

# Separate features and target variable
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Marine Oil Spill Prediction")
plt.show()
from sklearn.metrics import roc_curve, auc

y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.head(10)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_cnn_feature(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = cnn_model.predict(img_array)
    return features.flatten()
features = extract_cnn_feature("sar_image.png")
print("Extracted Feature Shape:", features.shape)
```

## RESULTS

All models were trained and evaluated using the same dataset. Ensemble-based models performed better compared to individual classifiers.

The Extra Trees Classifier achieved the best overall performance with a good balance between precision and recall.

## OUTPUT 

### MODEL ACCURACY COMPARISON

<img width="1188" height="590" alt="download" src="https://github.com/user-attachments/assets/6b444923-8dc7-40b3-a1ec-bbb709b22ea9" />

### MODEL PRECISION COMPARISON

<img width="1188" height="590" alt="download" src="https://github.com/user-attachments/assets/5c07a075-68ce-4a06-868f-b462d6b8b896" />

### MODEL RECALL COMPARISON

<img width="1188" height="590" alt="download" src="https://github.com/user-attachments/assets/43cd1c26-26b5-4330-841a-9281eae1a0a4" />

### MODEL F1-SCORE COMPARISON

<img width="1189" height="590" alt="download" src="https://github.com/user-attachments/assets/7373de55-8662-44e1-80ab-b7e092c72cd9" />

### MODEL SUMMARY

<img width="583" height="386" alt="image" src="https://github.com/user-attachments/assets/6042b940-5276-4971-8815-38c23c0ccdfa" />

### CONFUSION MATRIX - BASE RANDOM FOREST

<img width="560" height="455" alt="download" src="https://github.com/user-attachments/assets/ad4e9734-55fd-4424-b80b-5a5f7007263b" />

### ROC CURVE - BASE MODEL

<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/69c737e0-2525-4741-b14f-811ecddb4409" />

### IMPORTANT FEATURES - BASE

<img width="555" height="455" alt="download" src="https://github.com/user-attachments/assets/5e520bad-f51d-4479-b056-5988231d8786" />

### CONFUSION MATRIX - SMOTE BALANCED MODEL

<img width="560" height="455" alt="download" src="https://github.com/user-attachments/assets/89a11c3b-483f-43bb-9ac0-0cbc3dd8cb42" />

### ROC CURVE - SMOTE BALANCED MODEL

<img width="613" height="547" alt="download" src="https://github.com/user-attachments/assets/3add8524-50b9-4937-b5fd-4cdee1774912" />

### FEATURE IMPORTANCE - SMOTE MODEL

<img width="854" height="547" alt="download" src="https://github.com/user-attachments/assets/9281303b-db40-4c98-b621-1baa73e1bdba" />

### CONFUSION MATRIX - FUSED MODEL

<img width="560" height="455" alt="download" src="https://github.com/user-attachments/assets/e885d397-7288-4d8e-b487-81a5b869f97e" />

### ROC CURVE - FUSED MODEL

<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/05e50d21-22ab-4f23-b654-9fb64dbaff05" />

### COMPARISON OF ROC CURVES

<img width="691" height="624" alt="download" src="https://github.com/user-attachments/assets/f8c4f6a6-f193-4083-964f-bd76eba21069" />

### PRECISION - RECALL CURVE

<img width="567" height="455" alt="download" src="https://github.com/user-attachments/assets/b8a128a5-a994-4358-ae21-43d1f51638aa" />

## FUTURE WORK

1. Full satellite image pipeline (end-to-end CV + CNN model)
2. Real-time monitoring dashboards
3. Integration with remote sensing APIs
4. Deployment with cloud services

## REFERENCES

[1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.

[2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.

[3] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

[4] García, V., Mollineda, R. A., & Sánchez, J. S. (2010). Theoretical analysis of a performance measure for imbalanced data. Pattern Recognition Letters.
