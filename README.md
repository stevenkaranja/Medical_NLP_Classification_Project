# 🧠 Medical NLP Classification Project

A lean NLP project exploring automated classification of **medical transcripts** into their respective specialties using traditional and modern NLP methods.

---

## 📍 Project Overview

This project builds a multi-class text classifier using the **MTSamples dataset** (medical transcription notes).  
The goal is to assign each record to the correct medical specialty (e.g., *Surgery, Radiology, Pain Management*).

We benchmark classical models and progressively improve them with contextual embeddings (transformers).

---

## ⚙️ Tech Stack

- **Language:** Python  
- **Libraries:** pandas, scikit-learn, matplotlib, seaborn, XGBoost  
- **Environment:** Jupyter Notebooks (VS Code)  
- **ML Tasks:** NLP preprocessing, TF-IDF vectorization, classification, and evaluation  

---

## 📂 Folder Structure
```bash
med-nlp-classification-proj/
│
├── data/
│   └── mtsamples.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_tfidf_baseline_model.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_error_analysis.ipynb
│
├── models/
│   └── gradient_boosting_baseline.pkl
│
├── visuals/
│   ├── confusion_matrix.png
│   └── class_performance_chart.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧩 Steps Summary

| Step | Description |
|------|--------------|
| 1 | Data cleaning & preprocessing |
| 2 | Exploratory data analysis (EDA) |
| 3 | Feature extraction using TF-IDF |
| 4 | Baseline model training (Logistic Regression, Random Forest, Gradient Boosting) |
| 5 | Model evaluation & performance benchmarking |
| 6 | Error analysis & confusion matrix |
| 7 | Transformer-based model (to be added) |

---

## 📊 Results Summary

### **Baseline (TF-IDF + Gradient Boosting)**
- **Accuracy:** 12.8%  
- **F1 Score:** 12.4%  
- **Top Classes:** Pain Management, Sleep Medicine, Surgery  
- **Lowest Classes:** Hospice Care, Work Comp, Emergency Reports  

💬 **Interpretation:**  
The baseline demonstrates the limitations of sparse features in multi-class NLP with imbalanced data.  
Misclassifications like *Orthopedic → Radiology* reveal semantic overlap between specialties.

---

## 🚀 Next Steps

- Integrate transformer embeddings (e.g., BioClinicalBERT)
- Fine-tune and compare contextual models
- Deploy a minimal Streamlit interface for specialty prediction

---

## 👤 Author

**Stephen Karanja**  
📍 Nairobi, Kenya  
💼 Data Analyst | AI & Automation Specialist  
🔗 [LinkedIn](https://linkedin.com/in/steven-karanja)  
📧 muhurakaranja7@gmail.com  

---

## 📚 References

- [MTSamples Dataset – Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- [scikit-learn Documentation](https://scikit-learn.org/)
