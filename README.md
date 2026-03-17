# 🔍 Uncertainty Estimation in Machine Learning

## 📌 Overview

In many real-world machine learning applications, models not only need to make accurate predictions but also estimate **how confident they are** in those predictions.

Traditional models like Neural Networks and Gradient Boosting often produce **overconfident predictions**, even on unfamiliar or out-of-distribution data. This can lead to **risky decisions**, especially in domains like finance, healthcare, and risk modeling.

This project focuses on **estimating predictive uncertainty** and using it to make more reliable decisions.

---

## 🎯 Objectives

* Quantify **model uncertainty** in predictions
* Detect **out-of-distribution (OOD)** data
* Improve decision-making using uncertainty scores
* Compare multiple uncertainty estimation techniques

---

## ⚙️ Techniques Implemented

This project implements and compares multiple approaches:

* 🔹 **Monte Carlo Dropout (MCD)**
* 🔹 **Deep Ensembles**
* 🔹 **Multiple XGBoost Models (MultiXGB)**
* 🔹 **Randomized XGBoost (RandomXGB)**

---

## 🧠 Key Idea

Instead of relying only on predictions:

👉 We also measure **uncertainty of predictions**

This helps in:

* Identifying unreliable predictions
* Handling unseen data
* Making safer decisions in high-risk scenarios

---

## 📂 Project Structure

```
uncertainty-estimation-machine-learning/
│
├── preprocessing/        # Data preprocessing configs & logic
├── training/             # Model training strategies
├── testing/              # Testing utilities
├── experiment/           # Experiment scripts
├── autotuning/           # Hyperparameter tuning
├── data/                 # Dataset / sample data
│
├── process_data.py       # Data preprocessing pipeline
├── train_data.py         # Model training script
├── test_data.py          # Testing script
├── Evaluation.py         # Model evaluation
├── Predict.py            # Prediction script
├── findCutoff.py         # Threshold / cutoff logic
├── param.json            # Model configuration
├── dependencies.txt      # Required libraries
```

---

## ⚡ Installation

Clone the repository:

```
git clone https://github.com/codexankit/uncertainty-estimation-machine-learning.git
cd uncertainty-estimation-machine-learning
```

Install dependencies:

```
pip install -r dependencies.txt
```

---

## 🛠️ Usage

### 1️⃣ Preprocessing

```
python process_data.py --dataPath DATA_PATH --dataSaveDir OUTPUT_PATH
```

---

### 2️⃣ Training

```
python train_data.py --algo MCD -dr DATA_PATH -tc TARGET_COLUMN
```

Available algorithms:

* `MCD`
* `DeepEnsmb`
* `MultiXGB`
* `RandomXGB`

---

### 3️⃣ Evaluation

```
python Evaluation.py --algo MCD -edr DATA_PATH -esd OUTPUT_PATH
```

---

### 4️⃣ Prediction

```
python Predict.py --algo MCD -pdr DATA_PATH -psd OUTPUT_PATH
```

---

## 📊 Dataset

This project is tested on the **Lending Club Loan Dataset**, which includes:

* Loan status (Fully Paid, Charged Off, etc.)
* Borrower financial details
* Payment history

We define:

* ✅ **Good Loan** → Positive profit
* ❌ **Bad Loan** → Negative profit

We also analyze:

* Out-of-distribution data (e.g., COVID-19 period shifts)
* High-risk segments (e.g., low FICO scores)

---

## 📈 Results & Insights

* Uncertainty estimation helps detect **high-risk predictions**
* Ensemble methods improve **confidence calibration**
* Models can identify **out-of-distribution samples**
* Useful in **risk-sensitive domains like finance**

---

## 🚀 Future Improvements

* Improve scalability for large datasets
* Add more advanced Bayesian methods
* Integrate real-time prediction pipeline
* Add visualization dashboard

---
