🛒 SADLF-RecSys
Semantic + Behavioural Deep Learning Recommendation System
🔍 Overview
SADLF-RecSys predicts user–product relevance scores (0–1) using a fusion of semantic text embeddings (BERT) and behavioural features (click rate, time spent).
Designed for e-commerce product ranking, research, and ML experimentation.

✔ Automatic dataset detection
✔ BERT-powered semantic understanding
✔ Behavioural signal integration
✔ Visual metric evaluation

🧠 System Architecture
🔷 Architecture Diagram

(Replace with your own diagram later if needed)

Explanation:

Queries + product descriptions → BERT produces embeddings

Behavioural features → normalised numerical vector

Both are concatenated → Fully-connected layers → Sigmoid output

Predicts probability of user engagement

📥 Dataset Flow
🔷 Dataset Processing Flow Chart

Automatic Mapping Includes:
If dataset contains Amazon fields:

name  
reviews.text  
reviews.rating  
reviews.numHelpful  
categories


The model auto-converts into:

query  
page_content  
click_rate  
time_spent  
label

🚀 Features
1. Semantic + Behavioural Deep Fusion

BERT embeddings capture query–product meaning

Behaviour signals capture user engagement trends

2. Automatic Dataset Detection

Supports Amazon Review datasets (.csv / .zip)

Automatically maps fields to SADLF format

3. Multi-format Dataset Compatibility

For custom datasets, expect the following columns:

query  
page_content  
click_rate  
time_spent  
label

🧱 Model Architecture
[BERT Encoding]
      +
[click_rate, time_spent]
            ↓
Linear → 128 → ReLU
Linear →  64 → ReLU
Linear →   1 → Sigmoid


Loss: MSELoss

Optimiser: Adam (lr=0.0005)

Epochs: 15

🧪 Training

Run:

python recommendation.py


Then select your dataset when the file-picker opens.

Sample Training Output:
Epoch 3/15 — Loss: 0.028
Epoch 6/15 — Loss: 0.019
Epoch 9/15 — Loss: 0.014

📈 Evaluation Metrics
Metric	Value
MSE	0.04
Precision	0.91
Recall	0.88
F1 Score	0.89

The script displays a bar graph of these metrics automatically.

⚙ Installation
python -m venv venv
venv\Scripts\activate.ps1
pip install -r requirements.txt

📂 Project Structure
SADLF-RecSys/
│
├── recommendation.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset.csv               (optional)
└── uploaded_dataset/         (auto-created)

🎯 Future Enhancements

REST API (FastAPI / Flask)

Streamlit GUI

Explainability (LIME / SHAP)

FAISS vector search

Hybrid collaborative filtering

⚠ Disclaimer

This tool is for academic and research use only.
Do not use it for critical or commercial decision-making.

🙌 Credits

BERT — Devlin et al.

PyTorch

Scikit-Learn

Amazon Review Datasets

Maintained by: Sree Sai Vikas V.M
