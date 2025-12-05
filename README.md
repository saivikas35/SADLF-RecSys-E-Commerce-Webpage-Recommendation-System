
# 🛒 **SADLF-RecSys**  
### *Semantic + Behavioural Deep Learning Recommendation System*
An end-to-end deep-learning recommendation engine combining semantic understanding (BERT) and behavioral analytics, built using PyTorch, with automatic dataset mapping and visualization tools.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep--Learning-red.svg)
![BERT](https://img.shields.io/badge/BERT-Semantic%20Embedding-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Active-success.svg)

---

# 🔍 **Overview**
**SADLF-RecSys** is a deep-learning recommendation engine that fuses:

- **Semantic understanding (BERT embeddings)**  
- **Behavioural analytics (click rate, time spent)**  

It predicts personalised **0–1 relevance scores** for E‑commerce product ranking.

---

# 🧠 **System Architecture**

## **📌 High-Level Architecture Diagram (ASCII View)**

```
                 ┌─────────────────────────────┐
                 │        User Query           │
                 └───────────────┬─────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  BERT Encoder  │
                        └───────┬────────┘
                                │ 768-dim
                                ▼
         ┌────────────────────────────────────────────────┐
         │                                                │
         │         CONCATENATION LAYER                    │
         │  [BERT Embedding + click_rate + time_spent]    │
         │                                                │
         └───────────────────────┬────────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────┐
                      │  Linear (128)    │
                      └───────┬──────────┘
                              ▼
                      ┌──────────────────┐
                      │     ReLU         │
                      └───────┬──────────┘
                              ▼
                      ┌──────────────────┐
                      │  Linear (64)     │
                      └───────┬──────────┘
                              ▼
                      ┌──────────────────┐
                      │     ReLU         │
                      └───────┬──────────┘
                              ▼
                      ┌──────────────────┐
                      │  Linear (1)      │
                      │  Sigmoid Output  │
                      └──────────────────┘
```

---

# 📥 **Dataset Flow Diagram**

```
          ┌─────────────────────────┐
          │      Input Dataset      │
          └─────────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │ Amazon Format?      │
             └───────┬─────────────┘
       YES ───────────┘             └────────── NO
         ▼                                 ▼
┌───────────────────┐           ┌──────────────────────┐
│ Auto-Mapping to    │           │ Use Custom SADLF     │
│ SADLF Format        │           │ Format Directly      │
└─────────┬──────────┘           └───────────┬──────────┘
          ▼                                  ▼
      ┌─────────────────────────────────────────┐
      │ query, page_content, click_rate,        │
      │ time_spent, label                       │
      └─────────────────┬───────────────────────┘
                        ▼
                 Model Training
```

---

# 🚀 **Features**

### ✔ **Semantic + Behavioural Deep Fusion**
### ✔ **Automatic Dataset Mapping**
### ✔ **Supports Amazon & Custom Data**
### ✔ **Metric Visualisation**
### ✔ **End-to-End Training Pipeline**

---

# 🧱 **Model Architecture (Summary)**

```
Input → BERT → Concatenate Behaviour Features →
FC(128) → ReLU → FC(64) → ReLU → FC(1) → Sigmoid
```

- **Loss:** MSELoss  
- **Optimiser:** Adam (lr = 0.0005)  
- **Epochs:** 15  

---

# 🧪 **Training**

Run:

```bash
python recommendation.py
```

Choose `.csv` or `.zip` dataset from file picker.

### Example Training Log:
```
Epoch 3/15 — Loss: 0.028
Epoch 6/15 — Loss: 0.019
Epoch 9/15 — Loss: 0.014
```

---

# 📈 **Evaluation**

| Metric     | Value |
|-----------|--------|
| MSE       | 0.04   |
| Precision | 0.91   |
| Recall    | 0.88   |
| F1 Score  | 0.89   |

A **bar graph** of the metrics is also displayed.

---

# ⚙️ **Installation**

```bash
python -m venv venv
venv\Scripts\activate.ps1
pip install -r requirements.txt
```

---

# 📂 **Project Structure**

```
SADLF-RecSys/
│
├── recommendation.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset.csv          (optional)
└── uploaded_dataset/    (auto-generated)
```

---

# 🎯 **Future Enhancements**
- FastAPI / Flask model API  
- Streamlit dashboard  
- LIME / SHAP explainability  
- FAISS vector indexing  
- Hybrid collaborative filtering  

---

# ⚠️ Disclaimer
This project is for **research and educational** use only.  
Not intended for commercial deployment.

---

# 🙌 Credits
Semantic Model: **BERT (Devlin et al.)**  
Behavioural Feature Design: **User interaction metrics (click rate, time spent)**  
Fusion Architecture: **SADLF – Semantic + Adaptive Deep Learning Fusion**  
Machine Learning Framework: **PyTorch**  
Evaluation Tools: **Scikit-Learn**  
Dataset Sources: **Amazon Product Review Datasets / Custom E-commerce Behavioural Datasets**


Maintainer: **Sree Sai Vikas V.M**  
Powered by: **BERT**, **PyTorch**, **Scikit-Learn**

