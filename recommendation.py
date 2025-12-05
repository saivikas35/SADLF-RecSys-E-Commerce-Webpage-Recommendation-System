# Install these once using VS Code terminal (NOT inside the script):
# pip install torch transformers scikit-learn pandas numpy matplotlib

import torch, torch.nn as nn, torch.optim as optim
from transformers import BertTokenizer, BertModel
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import zipfile, glob, os

# ---- NEW: file chooser for VS Code (no Google Colab) ----
import tkinter as tk
from tkinter import filedialog

print("📂 Please select your dataset CSV file or ZIP folder (dataset.zip)")

root = tk.Tk()
root.withdraw()  # hide Tk window

file_path = filedialog.askopenfilename(
    title="Select CSV or ZIP dataset",
    filetypes=[("CSV or ZIP", "*.csv *.zip"),
               ("CSV files", "*.csv"),
               ("ZIP files", "*.zip"),
               ("All files", "*.*")]
)

if not file_path:
    raise ValueError("❌ No file selected.")

csv_files = []

if file_path.lower().endswith(".zip"):
    print(f"📦 Extracting {os.path.basename(file_path)} ...")
    extract_dir = "uploaded_dataset"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ Extracted to {extract_dir}")
    csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
elif file_path.lower().endswith(".csv"):
    csv_files = [file_path]

if not csv_files:
    raise ValueError("❌ No CSV file found inside the selected file/folder.")

dataset_path = csv_files[0]
print(f"✅ Dataset found: {dataset_path}")

# ---- REST OF YOUR CODE UNCHANGED ----

df = pd.read_csv(dataset_path)
print(f"📊 Raw dataset: {df.shape[0]} rows × {df.shape[1]} columns")

if set(["name","reviews.text","reviews.rating","reviews.numHelpful","categories"]).issubset(df.columns):
    print("🔄 Auto-mapping Amazon-style dataset ...")
    df = df.dropna(subset=["name","reviews.text","reviews.rating","reviews.numHelpful","categories"])
    df_sadlf = pd.DataFrame({
        "query": df["categories"].astype(str),
        "page_content": df["reviews.text"].astype(str),
        "click_rate": df["reviews.numHelpful"].astype(float),
        "time_spent": df["reviews.rating"].astype(float) * 30,
        "label": df["reviews.rating"].astype(float) / 5.0
    })
else:
    df_sadlf = df

if len(df_sadlf) > 800:
    df_sadlf = df_sadlf.sample(800, random_state=42).reset_index(drop=True)

print("✅ Final dataset ready:")
print(df_sadlf.head())

X_behavioral = df_sadlf[["click_rate","time_spent"]].values
X_behavioral = (X_behavioral - X_behavioral.mean(axis=0)) / X_behavioral.std(axis=0)
y = df_sadlf["label"].values.reshape(-1,1)

print("\n🚀 Training SADLF (Tuned Proposed Model)...")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=40)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

embeddings = []
for _, row in df_sadlf.iterrows():
    q_emb = get_embedding(str(row["query"]))
    p_emb = get_embedding(str(row["page_content"]))
    combined = torch.cat([q_emb, p_emb]).numpy()
    embeddings.append(combined)

X_semantic = np.array(embeddings)
X = np.hstack((X_semantic, X_behavioral))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sadlf = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
criterion = nn.MSELoss()
optimizer = optim.Adam(sadlf.parameters(), lr=0.0005)

epochs = 15
for epoch in range(epochs):
    sadlf.train()
    inputs = torch.tensor(X_train, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.float32)
    preds = sadlf(inputs)
    loss = criterion(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 3 == 0:
        print(f"SADLF Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

sadlf.eval()
with torch.no_grad():
    preds_sadlf = sadlf(torch.tensor(X_test, dtype=torch.float32)).numpy()

mse_sadlf = mean_squared_error(y_test, preds_sadlf)
threshold = 0.8
y_pred_bin = (preds_sadlf >= threshold).astype(int)
y_true_bin = (y_test >= threshold).astype(int)

prec_sadlf = precision_score(y_true_bin, y_pred_bin, zero_division=1)
rec_sadlf  = recall_score(y_true_bin, y_pred_bin, zero_division=1)
f1_sadlf   = f1_score(y_true_bin, y_pred_bin, zero_division=1)

print(f"\n📈 SADLF Results:\nMSE={mse_sadlf:.4f}, Precision={prec_sadlf:.2f}, Recall={rec_sadlf:.2f}, F1={f1_sadlf:.2f}")

labels = ['MSE', 'Precision', 'Recall', 'F1-Score']
sadlf_values = [mse_sadlf, prec_sadlf, rec_sadlf, f1_sadlf]

x = np.arange(len(labels))
width = 0.5
plt.figure(figsize=(7,4))
plt.bar(x, sadlf_values, width, label='SADLF', alpha=0.7)
plt.xticks(x, labels)
plt.ylabel('Metric Value')
plt.title('SADLF Model Performance')
plt.legend()
plt.show()

results = pd.DataFrame({
    'Metric': labels,
    'SADLF': [round(v,3) for v in sadlf_values]
})
print("\n✅ Final SADLF Table:")
print(results.to_string(index=False))
