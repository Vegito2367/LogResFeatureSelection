# **Heart Attack Prediction Using Custom Logistic Regression and Feature Selection**

## 📌 Project Overview
This project implements a **custom Logistic Regression model** using **NumPy** for **heart attack prediction**. It also includes a **Recursive Feature Elimination (RFE) algorithm** to optimize feature selection, reducing the number of predictors from **35 to 7** while achieving a **94% accuracy**.

By building the model from scratch, this project demonstrates an in-depth understanding of machine learning fundamentals, feature engineering, and model optimization techniques.

---

## ⚡ Key Features
✅ Custom **Logistic Regression** implementation using NumPy  
✅ Recursive Feature Elimination (**RFE**) for feature selection  
✅ **230K-row dataset** processed efficiently  
✅ Feature reduction from **35 → 7** without losing accuracy  
✅ Achieved **94% accuracy** on test data  
✅ Scalable and interpretable model for **healthcare analytics**  

---

## 🛠️ Tech Stack
- **Programming Language**: Python  
- **Libraries Used**:  
  - `numpy` – for numerical operations  
  - `pandas` – for data manipulation  
  - `sklearn StandardScaler` - for data normalization

---

## 📊 Dataset Columns

| Column Name                  | Description |
|------------------------------|--------------------------|
| PatientID                    | Unique identifier for each patient |
| State                        | State where the patient resides |
| Sex                          | Gender of the patient |
| GeneralHealth                | Self-reported general health status |
| AgeCategory                  | Age group/category of the patient |
| HeightInMeters               | Height of the patient in meters |
| WeightInKilograms            | Weight of the patient in kilograms |
| BMI                          | Body Mass Index (BMI) |
| HadHeartAttack               | Whether the patient had a heart attack (0/1) |
| HadAngina                    | Whether the patient had angina (0/1) |
| HadStroke                    | Whether the patient had a stroke (0/1) |
| HadAsthma                    | Whether the patient has asthma (0/1) |
| HadSkinCancer                | Whether the patient had skin cancer (0/1) |
| HadCOPD                      | Whether the patient has Chronic Obstructive Pulmonary Disease (COPD) (0/1) |
| HadDepressiveDisorder        | Whether the patient has been diagnosed with a depressive disorder (0/1) |
| HadKidneyDisease             | Whether the patient has kidney disease (0/1) |
| HadArthritis                 | Whether the patient has arthritis (0/1) |
| HadDiabetes                  | Whether the patient has diabetes (0/1) |
| DeafOrHardOfHearing          | Whether the patient has hearing difficulties (0/1) |
| BlindOrVisionDifficulty      | Whether the patient has vision difficulties (0/1) |
| DifficultyConcentrating      | Whether the patient has difficulty concentrating (0/1) |
| DifficultyWalking            | Whether the patient has difficulty walking (0/1) |
| DifficultyDressingBathing    | Whether the patient has difficulty dressing or bathing (0/1) |
| DifficultyErrands            | Whether the patient has difficulty running errands (0/1) |
| SmokerStatus                 | Smoking status of the patient (e.g., smoker/non-smoker) |
| ECigaretteUsage              | Whether the patient uses e-cigarettes (0/1) |
| ChestScan                    | Whether the patient has had a chest scan (0/1) |
| RaceEthnicityCategory        | Race/Ethnicity category of the patient |
| AlcoholDrinkers              | Whether the patient consumes alcohol (0/1) |
| HIVTesting                   | Whether the patient has been tested for HIV (0/1) |
| FluVaxLast12                 | Whether the patient received a flu vaccine in the last 12 months (0/1) |
| PneumoVaxEver                | Whether the patient has ever received a pneumococcal vaccine (0/1) |
| TetanusLast10Tdap            | Whether the patient received a Tdap/tetanus vaccine in the last 10 years (0/1) |
| HighRiskLastYear             | Whether the patient was considered high risk in the last year (0/1) |
| CovidPos                     | Whether the patient has tested positive for COVID-19 (0/1) |


---

## 📊 Model Implementation
### **1️⃣ Logistic Regression (Custom Implementation)**
Instead of using `sklearn`, the model is implemented from scratch using NumPy. It includes:  
✔️ **Sigmoid activation function**  
✔️ **Gradient descent optimization**  
✔️ **Objective function: Cross entropy loss**



### **2️⃣ Recursive Feature Elimination (RFE)**
To enhance model performance, a custom RFE algorithm was implemented:  
✔️ **Trains model with all features**  
✔️ **Ranks features based on their importance (weights)**  
✔️ **Eliminates least important features iteratively**  
✔️ **Stops when optimal feature subset is found**  

---

## 🚀 How to Run the Project

### **1. Install Dependencies**
```bash
pip install numpy pandas scikit-learn
```
### 2. Get the dataset from kaggle: (Link to dataset)[https://www.kaggle.com/datasets/tarekmuhammed/patients-data-for-medical-field]

### Sequentially run the cells in model.ipynb

