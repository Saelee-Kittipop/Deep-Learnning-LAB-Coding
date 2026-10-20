"""
LAB 1B: Feedforward Neural Network with PyTorch
Task:
- Classification: predict poverty status (pov_exp; 0 = not poor, 1 = poor)
"""
# 1. Import Libraries
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

# 2. Reproducibility
torch.manual_seed(0)  

# 3. Load Data & Preprocess
df = pd.read_csv("LAB3_Generative_Model/SES2023.csv", sep=None, engine="python")
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0}, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=["rspcexp23", "pov_exp"]).values # X = ตัวแปรอิสระ (ตัดตัวแปรตามออก)
y = df["pov_exp"].values  # y = ตัวแปรตามสำหรับ classification (ฺBinary Variable ที่บ่งบอกสถานะความยากจน)
# Plot class distribution ของตัวแปรตาม
df["pov_exp"].value_counts().plot(kind="bar")
plt.title("Class distribution")
plt.show()

# 4. Train-Test Set Split & Scale 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

x_scaler = StandardScaler().fit(X_train)
X_train = x_scaler.transform(X_train)
X_test  = x_scaler.transform(X_test)

# to tensors and DataLoader
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

# DataLoader config
batch_size = 64
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
test_ds = torch.utils.data.TensorDataset(X_test_t, torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# 5. Model 
class SimpleMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP(X_train.shape[1])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #อัลกอริทึมปรับ parameters
criterion = nn.BCEWithLogitsLoss() #Loss Function

# 6. Model Training (mini-batch using DataLoader)
loss_history = []
for ep in range(1, num_epochs + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    epoch_loss = running / len(train_loader.dataset)
    loss_history.append(epoch_loss)
    if ep % 10 == 0 or ep == 1 or ep == num_epochs:
        print(f"Epoch {ep}/{num_epochs} - loss: {epoch_loss:.4f}")

# 7. Model Evaluation
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

probs = np.array(ys_pred)
y_hat = probs

 # adapt y_true for metrics
y_true = np.array(ys_true)

acc = accuracy_score(y_true, y_hat)
print(f"Test Accuracy: {acc:.4f}")

# Classification evaluation metrics
prec = precision_score(y_true, y_hat, zero_division=0)
rec = recall_score(y_true, y_hat, zero_division=0)
f1  = f1_score(y_true, y_hat, zero_division=0)
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_hat)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix heatmap 
plt.figure(figsize=(6, 5))
if sns is not None:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
else:
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.6)
plt.scatter(range(len(y_hat)), y_hat, label="Predicted", alpha=0.6)
plt.legend()
plt.title("Classification: Predicted vs Actual")
plt.xlabel("Sample index")
plt.ylabel("Class")
plt.show()