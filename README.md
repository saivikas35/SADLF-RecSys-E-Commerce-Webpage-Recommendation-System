🛒 SADLF-RecSys — E-Commerce Webpage Recommendation System (Semantic + Behavioural Fusion)

An end-to-end deep-learning recommendation engine that combines semantic understanding (BERT) with behavioural analytics (click rate, time spent) to generate personalised product relevance scores. Built using PyTorch with automatic dataset mapping and visual evaluation tools.

🔍 Predicts user preference (0–1 score) using:

Query semantics

Product text embeddings (BERT)

Click-rate behaviour

Time spent on page

⚡ Highlights

🔥 High accuracy on personalised product recommendations

📊 Visualisation included (bar graph of model metrics)

📁 Auto-detects Amazon Review datasets

🧠 BERT-powered semantic understanding

⚠️ Disclaimer: This project is for educational and research purposes only.
It must not be used for commercial profiling or sensitive decision-making.

🚀 Features
🚀 1. Semantic + Behavioural Deep Fusion

Uses BERT embeddings for text and fuses them with behavioural signals for improved prediction accuracy.

🚀 2. Automatic Dataset Mapping

If your dataset contains Amazon Review fields:

name  
reviews.text  
reviews.rating  
reviews.numHelpful  
categories


The model automatically converts to SADLF format.

🚀 3. Custom Dataset Support

Also supports datasets already in the format:

query  
page_content  
click_rate  
time_spent  
label

🧱 Model Architecture (SADLF)
[BERT semantic embedding]  
        +  
[Behavioural features]  
        ↓  
Linear(→128) → ReLU  
Linear(128→64) → ReLU  
Linear(64→1) → Sigmoid  


Loss: MSELoss

Optimiser: Adam (0.0005)

Epochs: 15

🧪 Training the Model

Run:

python recommendation.py


A file-picker will open. Choose:

dataset.csv, or

dataset.zip

Example log:
Epoch 3/15 — Loss: 0.028  
Epoch 6/15 — Loss: 0.019  
Epoch 9/15 — Loss: 0.014  

📈 Model Evaluation Output
MSE = 0.04  
Precision = 0.91  
Recall = 0.88  
F1-Score = 0.89


Includes a bar graph of all 4 metrics.

⚙️ Installation
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

📂 Project Structure
SADLF-RecSys/
│
├── recommendation.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset.csv (optional)
└── uploaded_dataset/ (auto)

🎯 Future Enhancements

Web API (FastAPI / Flask)

Streamlit dashboard

Explainability (SHAP / LIME)

Vector search with FAISS

Hybrid collaborative filtering

🙌 Credits

BERT (Devlin et al.)

PyTorch

Scikit-Learn

Amazon Review Datasets

Maintainer: Sree Sai Vikas V.M
