# Evaluating AML Risk Through Data Science
_Aldo Alvarez_

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mq7fLoalaZ300CUeckJ4xvr9vtrX0R1J?usp=sharing)

---

<details>
  <summary><b>🔎 Executive Summary</b></summary>

This project applies classic, explainable Machine Learning models to assess Anti-Money Laundering (AML) risk on Bitcoin transactions.

Key contributions include:
- Wrangling and enriching transaction data with graph-based features (e.g., degree, PageRank, eigenvector centrality).
- Benchmarking Logistic Regression, Random Forest, and XGBoost classifiers.
- Prioritizing explainability and minimizing illicit transaction misclassification.

The codebase is modular and includes a Colab demo for easy experimentation.
</details>

---

## 📚 Dataset

The dataset used in this project comes from the [Elliptic Bitcoin Transaction Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).  
It contains Bitcoin transactions labeled as licit or illicit, alongside graph-based connections between transactions.

Steps to obtain the data:
1. Download the dataset from the provided link.
2. Ensure the following files are available in your working directory:
   - `elliptic_txs_edgelist.csv`
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`

---

## 🗂️ Project Structure

- `requirements.txt` — List of Python packages required to run the project.
- `aml_task_module.py` — Python module with auxiliary functions for feature engineering and model evaluation.
- `aml_task_demo.ipynb` — Jupyter Notebook with a step-by-step demonstration of the full solution.
- `README.md` — Overview and documentation of the project.

---

## 🛠️ Solution Overview

**Data Preprocessing**

- **Data Wrangling:** Minor adjustments were made to data types and column names to facilitate analysis.
- **Feature Engineering:** Added graph metrics to capture the network role of each transaction:
  - Degree
  - In-Degree
  - Out-Degree
  - Clustering Coefficient
  - PageRank
  - Eigenvector Centrality

**Modeling**

- Three models were implemented:
  - Logistic Regression
  - Random Forest
  - XGBoost

The focus was on model explainability and understanding general behavior rather than solely maximizing accuracy.

**Results**

- Logistic Regression struggled with the class imbalance, predicting most transactions as licit.
- Random Forest and XGBoost performed better, with Random Forest achieving lower misclassification of illicit transactions (critical for AML detection).
- Engineered features such as eigenvector centrality added meaningful predictive power.

**Further Improvements**

- Hyperparameter tuning.
- Advanced modeling (e.g., Graph Neural Networks) while balancing explainability.

---

