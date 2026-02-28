# 🩺 Medicine Recommendation System

A machine learning based disease prediction and recommendation system built using Scikit-learn and deployed with Streamlit.

---

## 🚀 Overview

This project predicts possible diseases based on selected symptoms using a trained Support Vector Classifier (SVC) model.

It also provides:
- 📝 Disease description
- 💊 Suggested medications
- 🛡️ Precautions
- 🥗 Diet plan
- 🏋️ Workout recommendations

⚠️ This project is for educational purposes only and is not a substitute for professional medical advice.

---

## 🧠 Machine Learning Approach

- Dataset: Structured healthcare symptom dataset
- Feature size: 130+ symptom features
- Models Compared:
  - SVC (Linear)
  - Random Forest
  - KNN
  - Gradient Boosting
  - Multinomial Naive Bayes
- Final Model: Linear SVM
- Model Serialization: Pickle

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py