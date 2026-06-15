# ❤️ Heart Disease Prediction using Stacking Ensemble Learning

## 📌 Project Overview

This project predicts whether a patient is likely to have heart disease using Machine Learning.

The model utilizes a **Stacking Ensemble Classifier** that combines the strengths of multiple algorithms:

- XGBoost Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

The final predictions are generated using a **Logistic Regression Meta-Learner**, improving overall performance and robustness.

---

## 🚀 Features

- Data preprocessing and cleaning
- One-Hot Encoding for categorical variables
- Feature Scaling using MinMaxScaler
- Train-Test Split
- Ensemble Learning using Stacking
- Performance Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

## 📂 Dataset

The project uses the **Heart Disease Dataset** containing patient health information such as:

| Feature | Description |
|----------|------------|
| Age | Patient age |
| Sex | Gender |
| ChestPainType | Type of chest pain |
| RestingBP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FastingBS | Fasting blood sugar |
| RestingECG | Resting electrocardiogram results |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Slope of peak exercise ST segment |
| HeartDisease | Target Variable |

Target:

- 0 → No Heart Disease
- 1 → Heart Disease

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-Learn
- XGBoost
- NumPy

---

## ⚙️ Data Preprocessing

### 1. Handling Missing Values
The dataset was checked for null values.

```python
df.isnull().sum()
```

### 2. One-Hot Encoding

Categorical features were transformed into numerical representations using:

```python
pd.get_dummies()
```

### 3. Feature Scaling

Min-Max Normalization was applied:

```python
MinMaxScaler()
```

---

## 🤖 Model Architecture

### Base Models

1. XGBoost Classifier
2. Random Forest Classifier
3. Support Vector Machine (SVM)

### Meta Model

- Logistic Regression

### Stacking Structure

```
          Dataset
              |
    ---------------------
    |        |         |
   XGB      RF        SVM
    |        |         |
    ---------------------
              |
      Logistic Regression
              |
         Prediction
```

---

## 🧠 Training

```python
estimators = [
    ("xgb", XGBClassifier()),
    ("rf", RandomForestClassifier()),
    ("svc", SVC())
]

stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)
```

---

## 📊 Evaluation Metrics

The model performance is evaluated using:

```python
accuracy_score()
precision_score()
recall_score()
f1_score()
```

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```

Move into the project directory:

```bash
cd heart-disease-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

or execute the Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📈 Future Improvements

- Hyperparameter Tuning
- Feature Selection
- Cross Validation Analysis
- Explainable AI (SHAP)
- Streamlit Web Application
- Model Deployment using Flask/FastAPI

---
GitHub: https://github.com/SHADOWZERO93

---

## ⭐ If you found this project useful, consider giving it a star!
