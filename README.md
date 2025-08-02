# 💳 Credit Card Fraud Detection

This project focuses on building a machine learning model to **detect fraudulent credit card transactions** using real-world data. Fraudulent transactions are rare and difficult to detect, making class imbalance a key challenge. We apply classification models with effective preprocessing, normalization, and evaluation strategies to identify anomalies with high accuracy.

---
# Dataset link - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## 📂 Dataset Overview

The dataset contains transactions made by credit cards in September 2013 by European cardholders. Key characteristics:

* **Features**: 30 total, including anonymized PCA components (`V1` to `V28`), `Time`, and `Amount`
* **Target**: `Class`

  * `0`: Genuine transaction
  * `1`: Fraudulent transaction
* **Highly imbalanced**: Fraud cases represent \~0.17% of all records

---

## 🛠️ Tools & Libraries

* **Language**: Python
* **Libraries**:
  `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 📊 Project Workflow

### 🔹 1. Data Preprocessing

* Loaded and inspected data
* Applied **feature scaling** (`StandardScaler`)
* Performed **manual oversampling** of the minority class to balance the dataset
* Split the data into **training and testing sets** (80/20)

### 🔹 2. Models Used

* ✅ **Logistic Regression**
* ✅ **Random Forest Classifier**

---

## 🔍 Evaluation Metrics

| Model               | Accuracy | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
| ------------------- | -------- | ----------------- | -------------- | ---------------- |
| Logistic Regression | 97%      | 0.96              | 0.90           | 0.93             |
| Random Forest       | **98%**  | **1.00**          | 0.89           | **0.94**         |

* **Confusion Matrix** and **Classification Report** used for analysis
* **Random Forest** outperformed Logistic Regression in F1-score and overall precision

---

## 📌 Key Insights

* **Class imbalance** is critical in fraud detection. Oversampling improved model learning on rare fraud cases.
* **Random Forest** showed better overall performance, with **higher accuracy and fraud precision**.
* **Logistic Regression** still performed well and is a good interpretable baseline.

---

## 📈 Visuals

* Correlation heatmaps
* Confusion matrix heatmaps for both models

*(Charts not shown here, but included in the notebook for reference.)*

---

## 👨‍💻 How to Run

1. Clone this repository
2. Make sure `creditcard.csv` is in the root directory
3. Run the Jupyter notebook or `.py` script
4. Review evaluation metrics and plots

---

## 👤 Author

**Vansh Verma**
*Data Science & Machine Learning Enthusiast*

