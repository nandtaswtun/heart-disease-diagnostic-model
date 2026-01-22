# Heart Disease Prediction & Statistical Analysis

This project explores a clinical dataset to identify key indicators of heart disease. It combines **statistical hypothesis testing**, **exploratory data analysis (EDA)**, and **machine learning** to classify the presence or absence of heart disease based on patient metrics like age and cholesterol.

---

## 🚀 Project Overview
The goal of this analysis is to determine the relationship between physiological factors and heart disease, and to build a predictive model to assist in clinical diagnosis.

### Key Features:
* **Statistical Analysis:** Conducted an Independent T-Test to determine if cholesterol levels significantly differ between healthy and diseased patients.
* **Data Visualization:** Created scatter plots and heatmaps to visualize age/cholesterol correlations and model performance.
* **Predictive Modeling:** Implemented and compared Linear and Logistic Regression models.
* **Model Evaluation:** Detailed performance metrics using Confusion Matrices, R-squared, and Classification Reports (Precision, Recall, F1-Score).

---

## 📊 Methodology

### 1. Data Processing
* Mapped categorical target variables (`Presence`/`Absence`) to numerical values (`1`/`0`) for model compatibility.
* Split the dataset into training (80%) and testing (20%) sets to ensure unbiased evaluation.

### 2. Exploratory Analysis
The analysis investigates the distribution of cholesterol levels across different heart disease statuses to identify patterns and outliers.



### 3. Machine Learning Pipeline
The project utilizes two distinct modeling approaches:
* **Linear Regression:** To explore trend predictions and calculate error metrics like $MAE$ and $R^2$.
* **Logistic Regression:** Used as the primary classifier for categorical prediction.



---

## 📈 Results

### Model Performance
The Logistic Regression model was evaluated using a confusion matrix to visualize True Positives and False Negatives.



* **Accuracy:** The model achieved a high accuracy rate (calculated in the script).
* **Statistical Significance:** The T-test provides a p-value to validate if the findings on cholesterol are statistically significant ($p < 0.05$).

---

## 🛠️ Requirements
To run this project, you will need the following Python libraries:
* `pandas`
* `seaborn`
* `matplotlib`
* `scikit-learn`
* `scipy`

---

## 📂 How to Use
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Prepare the data:** Ensure your dataset is named `Heart_Disease_Prediction.csv` in the project root or update the file path in the script.
3.  **Run the analysis:**
    ```bash
    python heart_disease_analysis.py
    ```
