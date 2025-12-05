🛒 SADLF-RecSys — E-Commerce Webpage Recommendation System (Semantic + Behavioral Fusion)

An end-to-end deep-learning recommendation engine combining semantic understanding (BERT) and behavioral analytics, built using PyTorch, with automatic dataset mapping and visualization tools.

🔍 Predicts user preference (0–1 score) based on:

Query semantics

Product text embeddings (BERT)

Click-rate behavior

Time spent on page

⚡ Achieves high accuracy on product recommendation experiments
📊 Includes visualization (bar graph of model metrics)
📁 Auto-detects Amazon review datasets

⚠️ Disclaimer: This project is for educational & research purposes only.

🚀 Features
✔ 1. Semantic + Behavioral Deep Fusion

Uses BERT embeddings for text and combines them with behavioral features.

✔ 2. Automatic Dataset Mapping

If dataset contains Amazon Review fields:

name
reviews.text
reviews.rating
reviews.numHelpful
categories


The model automatically converts to SADLF format.

✔ 3. Custom Dataset Support

Also works with datasets already in this format:

query
page_content
click_rate
time_spent
label

✔ 4. Full SADLF Neural Network

3-layer deep network with Sigmoid output for preference scoring.

✔ 5. Model Evaluation Metrics

Outputs:

Mean Squared Error

Precision

Recall

F1 Score

Performance bar chart

📂 Project Structure
SADLF-RecSys/
│
├── recommendation.py          # Main model: training + embedding + evaluation
├── requirements.txt           # Core dependencies
├── README.md                  # Documentation
├── .gitignore                 # Files to exclude from GitHub
│
├── dataset.csv (optional)     # User dataset
├── uploaded_dataset/ (auto)   # Auto-extracted ZIP folder
└── LICENSE                    # MIT License (optional)

📦 Dataset Used

You can use:

🟩 1. Amazon Product Review Dataset

(Script auto-maps these fields)

Field	Description
categories	Product category / query text
reviews.text	User-written text
reviews.rating	Rating (1–5)
reviews.numHelpful	Helpful votes (click rate)
name	Product name
🟦 2. Custom Dataset Format
Column	Description
query	Search query / category
page_content	Product text / description
click_rate	User interaction score
time_spent	Time spent (seconds)
label	Normalized (0–1) preference
🎯 Model Architecture (SADLF)
🔹 Embedding Stage

Uses BERT (bert-base-uncased)

Extracts semantic embeddings for:

query

page_content

🔹 Behavioral Stage

Normalizes:

click_rate

time_spent

🔹 Fusion Network
Linear → 128 → ReLU
Linear → 64  → ReLU
Linear → 1   → Sigmoid

🔹 Loss & Optimization

Loss: MSELoss

Optimizer: Adam (lr = 0.0005)

🧪 Training the Model

Run the script:

python recommendation.py


You will be prompted to select your CSV or ZIP dataset.

Training log example:

SADLF Epoch [3/15] Loss: 0.0284
SADLF Epoch [6/15] Loss: 0.0191
SADLF Epoch [9/15] Loss: 0.0147
...


Best metrics will be printed after evaluation.

📈 Model Evaluation Output

The script prints:

📈 SADLF Results:
MSE=0.0421, Precision=0.91, Recall=0.88, F1=0.89


And displays a bar chart:

Metric	Value
MSE	0.04
Precision	0.91
Recall	0.88
F1-score	0.89
⚡ Installation
1️⃣ Create environment
python -m venv venv

2️⃣ Activate

PowerShell

venv\Scripts\Activate.ps1

3️⃣ Install Dependencies
pip install -r requirements.txt

🌐 Future Upgrades

Ranking model (pairwise scoring)

Explainability: LIME / SHAP

Real-time API using FastAPI

Integration with vector database (FAISS)

Streamlit interactive dashboard

Multi-modal features (images + text + behavior)

🔐 Ethical Disclaimer

This recommender system is designed for academic, research, and educational purposes.
It should not be used to profile or influence users unethically.

🙌 Credits

BERT Transformer (Devlin et al.)

PyTorch Team

Scikit-Learn

Amazon Review Datasets (public research datasets)

Maintainer: Sree Sai Vikas V.M
