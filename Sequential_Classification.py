"""
LAB 2B: Sequential Models for Classification on SES Data

Task:
- Binary Classification (target variable: pov_exp)

Models Used:
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer

Important Note:
- ข้อมูล SES ที่ใช้ในแลปนี้เป็นข้อมูลแบบ cross-sectional data
- การแปลง feature ให้เป็น sequence ไม่ได้หมายความว่าข้อมูล SES มีลำดับเวลา (sequence) จริง
- จุดประสงค์ คือ เพื่อศึกษาแนวคิด และกลไกการทำงานของ Sequential Models เมื่อนำมาใช้กับข้อมูล SES ซึ่งเป็นข้อมูลแบบ cross-sectional
"""

# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
try:
    import seaborn as sns
except Exception:
    sns = None
import math

# =========================
# 2. Reproducibility
# =========================
torch.manual_seed(0)

# =========================
# 3. Load & Preprocess Data
# =========================
df = pd.read_csv("SES2023.csv", sep=None, engine="python")
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0}, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=["rspcexp23", "pov_exp"]).values
y = df["pov_exp"].values

# =========================
# 4. Train-Test Split & Scale
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# =========================
# 5. Reshape to Sequence 
# Concept: เนื่องจากข้อมูล SES เป็น cross-sectional data จึงต้องแปลง features เป็น sequence
# =========================
seq_len = 5
input_dim = X_train.shape[1] // seq_len

X_train = X_train[:, :seq_len * input_dim]
X_test  = X_test[:,  :seq_len * input_dim]

X_train_seq = X_train.reshape(-1, seq_len, input_dim)
X_test_seq  = X_test.reshape(-1, seq_len, input_dim)

# Use Dataset + DataLoader for mini-batch training
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None):
        # X: numpy array (N, seq_len, input_dim)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# Training config / device
batch_size = 32
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create datasets & dataloaders
train_ds = SequenceDataset(X_train_seq, y_train)
test_ds = SequenceDataset(X_test_seq, y_test)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# =========================
# 6. Models
# =========================

## RNN
class RNNModel(nn.Module):
    """
    Core idea:
    - Hidden state h_t ส่งต่อไปตาม sequence
    - มีปัญหา vanishing gradient
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # last timestep

## LSTM
class LSTMModel(nn.Module):
    """
    Core idea:
    - มี cell state (c_t)
    - แก้ปัญหา vanishing gradient
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

## Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerModel(nn.Module):
    """
    Core idea:
    - ไม่มี recurrence
    - ใช้ Self-Attention
    - เรียนรู้ในแต่ละ timestep ได้พร้อมกัน
    """
    def __init__(self, input_dim, d_model=32, nhead=4, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # project and scale
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        z = self.encoder(x)
        return self.fc(z[:, -1, :])


# =========================
# 7. Training Function
# =========================
def train_and_evaluate(model, name, train_loader, test_loader, device, epochs=num_epochs):
    """Train model using mini-batches from train_loader and evaluate on test_loader.
    Returns confusion matrix for the test set.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"{name} Epoch {ep+1}/{epochs} - loss: {epoch_loss:.4f}")

    # Evaluation on test_loader
    model.eval()
    ys_true = []
    ys_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            yhat = (probs >= 0.5).astype(int)
            ys_pred.extend(yhat.tolist())
            ys_true.extend(yb.numpy().ravel().tolist())

    y_true = np.array(ys_true)
    y_hat = np.array(ys_pred)

    acc = accuracy_score(y_true, y_hat)
    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)
    f1s = f1_score(y_true, y_hat, zero_division=0)

    print(f"{name} Accuracy : {acc:.4f}")
    print(f"{name} Precision: {prec:.4f}")
    print(f"{name} Recall   : {rec:.4f}")
    print(f"{name} F1-score : {f1s:.4f}")

    cm = confusion_matrix(y_true, y_hat)
    print(f"{name} Confusion Matrix:\n", cm)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(5, 4))
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix')
    else:
        plt.imshow(cm, cmap='Blues')
        plt.colorbar()
        ticks = np.arange(cm.shape[0])
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{name} Confusion Matrix')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.show()

    return cm


# =========================
# 8. Run Experiments
# =========================
cm_rnn = train_and_evaluate(RNNModel(input_dim), "RNN", train_loader, test_loader, device)
cm_lstm = train_and_evaluate(LSTMModel(input_dim), "LSTM", train_loader, test_loader, device)
cm_trf = train_and_evaluate(TransformerModel(input_dim, max_len=seq_len), "Transformer", train_loader, test_loader, device)

print("\nConfusion Matrices")
print("RNN\n", cm_rnn)
print("LSTM\n", cm_lstm)
print("Transformer\n", cm_trf)
