"""
LAB 2A: Sequential Models for Regression on SES Data

Task:
- Regression (target variable: rspcexp23)

Models Used:
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Transformer

Important Notes:
- ข้อมูล SES ที่ใช้ในแลปนี้เป็นข้อมูลแบบ cross-sectional data
- การแปลง feature ให้เป็น sequence ไม่ได้หมายความว่าข้อมูล SES มีลำดับเวลา (sequence) จริง
- จุดประสงค์ คือ เพื่อศึกษาแนวคิด และกลไกการทำงานของ Sequential Models กับข้อมูลที่ไม่มีโครงสร้างเชิงเวลา
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
from sklearn.metrics import mean_squared_error, r2_score
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
y = df["rspcexp23"].values.reshape(-1, 1)

# =========================
# 4. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# =========================
# 5. Scaling 
# =========================
x_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train)

X_train = x_scaler.transform(X_train)
X_test  = x_scaler.transform(X_test)

y_train = y_scaler.transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# =========================
# 6. Reshape to Sequence 
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
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# Training config and device
batch_size = 32
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets and loaders 
train_ds = SequenceDataset(X_train_seq, y_train)
test_ds = SequenceDataset(X_test_seq, y_test_scaled)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# =========================
# 7. Models
# =========================

## RNN
class RNNRegressor(nn.Module):
    """
    RNN: ใช้ hidden state ตัวสุดท้ายสำหรับ regression
    """
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

## LSTM
class LSTMRegressor(nn.Module):
    """
    LSTM: เพิ่ม cell state เพื่อจำข้อมูลระยะยาว
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
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    
class TransformerRegressor(nn.Module):
    """
    Transformer: Self-Attention ไม่มี recurrence
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
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        z = self.encoder(x)
        return self.fc(z[:, -1, :])

# =========================
# 8. Training & Evaluation
# =========================
def train_and_evaluate(model, name, train_loader, test_loader, device, epochs=num_epochs):
    """Train the model with mini-batches and evaluate on test_loader.
    Returns: (losses, y_pred (inverse-scaled), r2)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses = []

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        if (ep + 1) % 10 == 0 or ep == 0 or ep == epochs - 1:
            print(f"{name} Epoch {ep+1}/{epochs} - loss: {epoch_loss:.4f}")

    # Evaluation on test set (batched)
    model.eval()
    preds_all = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            out = model(xb)
            preds_all.append(out.cpu().numpy())

    y_pred_scaled = np.vstack(preds_all)
    # inverse scaling back to original units
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} RMSE: {rmse:,.2f} | R2: {r2:.4f}")

    return losses, y_pred, r2

# =========================
# 9. Run Experiments
# =========================
models = {
    "RNN": RNNRegressor(input_dim),
    "LSTM": LSTMRegressor(input_dim),
    "Transformer": TransformerRegressor(input_dim, max_len=seq_len)
}

results = {}

for name, model in models.items():
    losses, y_pred, r2 = train_and_evaluate(model, name, train_loader, test_loader, device)
    results[name] = (losses, y_pred, r2)

# =========================
# 10. Visualization (all models)
# =========================

# 1) Loss curves (all models on one plot)
plt.figure(figsize=(8, 5))
for name, val in results.items():
    losses = val[0]
    plt.plot(losses, label=name)
plt.title("Training Loss Curves (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()


# 2) Predicted vs Actual + Residuals for each model (grid)
n_models = len(results)
fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))
if n_models == 1:
    axes = np.expand_dims(axes, 0)

for i, (name, val) in enumerate(results.items()):
    losses = val[0]
    y_pred = val[1]
    r2 = val[2] if len(val) > 2 else None
    ax_scatter = axes[i, 0]
    ax_resid = axes[i, 1]

    # flatten
    y_true = y_test.reshape(-1)
    y_pred_flat = np.array(y_pred).reshape(-1)

    # Scatter: actual vs predicted
    ax_scatter.scatter(y_true, y_pred_flat, alpha=0.6)
    mn = min(y_true.min(), y_pred_flat.min())
    mx = max(y_true.max(), y_pred_flat.max())
    ax_scatter.plot([mn, mx], [mn, mx], 'r--')
    ax_scatter.set_xlabel('Actual')
    ax_scatter.set_ylabel('Predicted')
    title = f"{name}: Predicted vs Actual"
    if r2 is not None:
        title += f" (R2={r2:.3f})"
    ax_scatter.set_title(title)

    # Residuals
    resid = y_true - y_pred_flat
    ax_resid.scatter(y_pred_flat, resid, alpha=0.6)
    ax_resid.hlines(0, xmin=y_pred_flat.min(), xmax=y_pred_flat.max(), colors='r', linestyles='--')
    ax_resid.set_xlabel('Predicted')
    ax_resid.set_ylabel('Residual (Actual - Pred)')
    ax_resid.set_title(f"{name}: Residuals")

plt.tight_layout()
plt.show()

# 3) Summary RMSE table and optional bar plot
names = []
rmses = []
r2s = []
for name, (_, y_pred, *rest) in results.items():
    y_pred_flat = np.array(y_pred).reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_test.reshape(-1), y_pred_flat))
    names.append(name)
    rmses.append(rmse)
    r2s.append(rest[0] if rest else None)

print("Summary RMSE and R2:")
for n, r, r2 in zip(names, rmses, r2s):
    r2_str = f" | R2: {r2:.4f}" if r2 is not None else ""
    print(f" - {n}: {r:,.2f}{r2_str}")

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.bar(names, rmses, color=['C0','C1','C2'][:len(names)])
plt.ylabel('RMSE')
plt.title('Model RMSE Comparison')

plt.subplot(1,2,2)
plt.bar(names, [r if r is not None else 0 for r in r2s], color=['C0','C1','C2'][:len(names)])
plt.ylabel('R2')
plt.title('Model R2 Comparison')
plt.tight_layout()
plt.show()
