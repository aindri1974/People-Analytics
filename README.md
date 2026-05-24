# 👥 People Analytics: Workforce Retention & Attrition Prediction

> **Can we predict which employees are about to leave — before they do?**
> This project leverages HR data, exploratory analysis, and a deep learning model to answer exactly that.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/)
[![Dataset](https://img.shields.io/badge/Dataset-IBM%20HR%20Analytics-lightgrey?logo=kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/aindri1974/People-Analytics)

---

## 📌 Project Overview

Employee attrition is one of the most costly and disruptive challenges facing modern organizations. Replacing a single employee can cost anywhere from **50% to 200% of their annual salary** when factoring in recruitment, onboarding, and lost productivity.

This project takes a data-driven approach to the problem. Using a real-world IBM HR dataset, we perform in-depth exploratory data analysis to uncover the hidden patterns behind employee turnover, then build and train an **Artificial Neural Network (ANN)** to predict whether an employee is likely to leave or stay — giving HR teams the power to act before it's too late.

---

## � Business Problem & Motivation

Human Resources departments traditionally rely on exit interviews and gut instinct to understand attrition. By the time an employee hands in their resignation, it's already too late to intervene.

**The core business questions this project addresses:**

- What are the key factors that drive an employee to leave?
- Can we accurately predict attrition risk for individual employees?
- How can HR use these predictions to proactively improve retention?

> **Goal:** Build a predictive ML/DL model that classifies employees as likely to leave (`Attrition = 1`) or stay (`Attrition = 0`), enabling HR to take targeted, proactive action.

---

## 📊 Dataset Information

| Property | Details |
|---|---|
| **Source** | [IBM HR Analytics Employee Attrition & Performance Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| **Records** | 1,470 employees |
| **Features** | 35 columns |
| **Target Variable** | `Attrition` (Binary: `1` = Left, `0` = Stayed) |
| **Missing Values** | None |

**Key Features Include:**

| Category | Features |
|---|---|
| Demographics | `Age`, `Gender`, `MaritalStatus`, `EducationField` |
| Job Info | `JobRole`, `JobLevel`, `Department`, `BusinessTravel` |
| Satisfaction Metrics | `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`, `WorkLifeBalance` |
| Compensation | `MonthlyIncome`, `DailyRate`, `HourlyRate`, `PercentSalaryHike`, `StockOptionLevel` |
| Experience | `TotalWorkingYears`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsWithCurrManager` |
| Work Patterns | `OverTime`, `TrainingTimesLastYear`, `NumCompaniesWorked` |

> **Note:** Features with zero variance across all employees (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`) were dropped during preprocessing as they carry no predictive value.

---

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Environment:** Google Colab
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Seaborn, Matplotlib
- **Machine Learning & Preprocessing:** Scikit-learn (OneHotEncoder, MinMaxScaler, train_test_split)
- **Deep Learning:** TensorFlow / Keras (Sequential ANN)
- **Dataset Source:** [Kaggle — IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 🏗️ Project Architecture & Workflow

The project follows a structured, task-based pipeline:

```
Raw HR Data (CSV)
      │
      ▼
Task 1 & 2 ──► Business Understanding + Library Import + Data Loading
      │
      ▼
Task 3 ──────► Exploratory Data Analysis (EDA) & Visualization
      │
      ▼
Task 4 ──────► Data Preprocessing & Feature Engineering
      │
      ▼
Task 5 & 6 ──► ANN Model Building & Training (100 Epochs)
      │
      ▼
Task 7 ──────► Model Evaluation, Predictions & Classification Report
```

**Detailed Steps:**

1. **Task 1 & 2 — Setup & Data Loading:** Imported all required libraries, loaded the IBM HR dataset, and performed initial inspection (`head()`, `tail()`, `info()`, `describe()`).

2. **Task 3 — EDA & Visualization:** Conducted thorough exploratory analysis including:
   - Missing value heatmap (confirmed zero nulls)
   - Full dataset histogram for distribution analysis
   - Correlation heatmap across all numerical features
   - Count plots for `Age`, `JobRole`, `MaritalStatus`, `JobInvolvement`, `JobLevel` vs. Attrition
   - KDE plots comparing employees who left vs. stayed across `DistanceFromHome`, `YearsWithCurrManager`, and `TotalWorkingYears`
   - Box plots for `MonthlyIncome` by `Gender` and `JobRole`

3. **Task 4 — Data Preprocessing:**
   - Encoded binary columns (`Attrition`, `OverTime`, `Over18`) to integers
   - Applied `OneHotEncoder` to categorical features (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`)
   - Scaled all numerical features using `MinMaxScaler`
   - Split data into 80% training / 20% testing sets

4. **Task 5 & 6 — Model Building & Training:**
   - Built a deep **Artificial Neural Network** using Keras Sequential API
   - Compiled with **Adam optimizer** and **binary cross-entropy loss**
   - Trained over **100 epochs**

5. **Task 7 — Evaluation:** Generated predictions, confusion matrix, and a full classification report.

---

## 🔍 Key Insights from Data Exploration

> *(Placeholder — add your specific findings here after reviewing the EDA outputs)*

- 📌 **Insight 1:** *(e.g., Employees who work overtime have a significantly higher attrition rate)*
- 📌 **Insight 2:** *(e.g., Single employees tend to leave more than married or divorced employees)*
- 📌 **Insight 3:** *(e.g., Sales Representatives show the highest attrition rate among all job roles)*
- 📌 **Insight 4:** *(e.g., Employees living farther from home are more likely to leave)*
- 📌 **Insight 5:** *(e.g., Low job level and low job involvement are strong attrition indicators)*
- 📌 **Insight 6:** *(e.g., Employees with fewer years with their current manager show higher attrition)*

---

## 🤖 Modeling & Results

### Model Architecture

A deep **Artificial Neural Network (ANN)** was built using the Keras Sequential API:

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Training Epochs:** 100
- **Task:** Binary Classification (`Attrition`: Left vs. Stayed)

### Performance

| Metric | Value |
|---|---|
| **Testing Accuracy** | **89.13%** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **0 — Stayed** | 0.89 | 0.94 | **0.91** | — |
| **1 — Left** | 0.52 | 0.38 | **0.44** | — |

**Key Takeaway:**

The model performs strongly on the majority class (employees who stayed) with an F1-score of **0.91**. The lower F1-score of **0.44** for the minority class (employees who left) is a known challenge with imbalanced datasets — where only ~16% of employees actually left. Addressing this class imbalance is a primary target for future improvement.

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)

1. Upload `People_Analytics_Workforce_Retention_Prediction.ipynb` and `People Analytics.csv` to your **Google Drive (My Drive root)**
2. Open the notebook in **Google Colab** (right-click → Open with → Colaboratory)
3. Run the first cell to mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   import os
   os.chdir('/content/drive/MyDrive')
   ```
4. Click **Runtime → Run All**

### Option 2 — Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aindri1974/People-Analytics.git
   cd People-Analytics
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook:**
   ```bash
   jupyter notebook People_Analytics_Workforce_Retention_Prediction.ipynb
   ```

### Requirements

```
pandas
numpy
seaborn
matplotlib
scikit-learn
tensorflow
```

---

## � Future Scope & Improvements

- **Handle Class Imbalance:** Apply techniques like SMOTE (Synthetic Minority Oversampling), class weighting, or undersampling to improve recall for the minority class (employees who left)
- **Hyperparameter Tuning:** Use Keras Tuner or GridSearchCV to optimize the ANN architecture (layers, neurons, dropout rates, learning rate)
- **Try Ensemble Models:** Benchmark against Random Forest, XGBoost, and LightGBM to compare performance
- **Feature Importance Analysis:** Use SHAP values to explain which features drive individual predictions — critical for HR actionability
- **Deploy as a Web App:** Wrap the model in a Flask or Streamlit app so HR teams can input employee data and get real-time attrition risk scores
- **Cross-Validation:** Implement k-fold cross-validation for more robust model evaluation

---

## 📁 Repository Structure

```
People-Analytics/
│
├── People_Analytics_Workforce_Retention_Prediction.ipynb   # Main notebook
├── People Analytics.csv                                     # Dataset
├── People_Analytics_Png/                                    # Business case images
│   ├── download.png
│   ├── download_1.png
│   ├── download_2.png
│   └── download_3.png
├── People_Analytics_project_png/                            # EDA output plots
│   ├── heatmap.png
│   ├── boxplot.png
│   ├── countplot.png
│   ├── kdeplot.png
│   └── ...
└── README.md
```

---

## 🔗 Links

- 📂 **GitHub Repository:** [github.com/aindri1974/People-Analytics](https://github.com/aindri1974/People-Analytics)
- 📊 **Dataset:** [IBM HR Analytics Employee Attrition & Performance — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 👤 Author

**Aindri**
- GitHub: [@aindri1974](https://github.com/aindri1974)

---

*This project was built for learning, portfolio demonstration, and exploring the intersection of People Analytics and Machine Learning.*
