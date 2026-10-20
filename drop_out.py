"""
LAB 1C: Regularization with Dropout in Feedforward Neural Network (FFN)
- ใช้กลไก Dropout เพื่อป้องกัน Overfitting
"""
# 1. Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 2. Reproducibility
torch.manual_seed(0)  

# 3. Load Data & Preprocess
df = pd.read_csv("SES2023.csv", sep=None, engine="python")
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df.replace({"Yes": 1, "No": 0, "yes": 1, "no": 0}, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(df.median(numeric_only=True), inplace=True)

X = df.drop(columns=["rspcexp23", "pov_exp"]).values # X = ตัวแปรอิสระ (ตัดตัวแปรตามออก)
y = df["rspcexp23"].values # y = ตัวแปรตามสำหรับ regression

# 4. Train-Test Set Split & Scale 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# scale X
x_scaler = StandardScaler().fit(X_train)
X_train = x_scaler.transform(X_train)
X_test  = x_scaler.transform(X_test)

# scale y (Different from FFN_regression_no scale.py)
y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_s, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)

# DataLoader: create TensorDataset and DataLoader for mini-batch training
batch_size = 32
train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
# include scaled y_test in test dataset for shape consistency 
test_ds = torch.utils.data.TensorDataset(X_test_t, torch.tensor(y_test_s, dtype=torch.float32).unsqueeze(1))

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. Model (with Dropout)

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, dropout: float = 0.0):
        """MLP with Dropout layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            # Dropout after activation reduces reliance on specific neurons
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# Set dropout probability 
dropout_rate = 0.3 #สุ่มปิดนิวรอน 30% ใน hidden layers ระหว่าง training
model = SimpleMLP(X_train.shape[1], dropout=dropout_rate)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # optimizer
criterion = nn.MSELoss()  # Loss Function

# 6. Model Training
model.train() #โหมด training เพื่อเปิดใช้งาน Dropout
epochs = 50
loss_hist = []

for ep in range(1, epochs + 1):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad() #ล้างค่า Gradient ที่คำนวณในรอบก่อนหน้า
        preds = model(xb)
        loss = criterion(preds, yb) #compute loss
        loss.backward() #backpropagation เพื่อหา gradients
        optimizer.step() #update parameters ตามแนวคิด SGD

        losses.append(loss.item())

    loss_hist.append(np.mean(losses))
    if ep % 10 == 0 or ep == 1:
        print(f"Epoch {ep:03d} | Train Loss: {loss_hist[-1]:.4f}")

# ---------------- plots ----------------
plt.figure()
plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve (Dropout)")
plt.show()

# 7. Model Evaluation
model.eval() #โหมด evaluation เพื่อปิดใช้งาน Dropout
preds_batches = []
with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(device)
        out = model(xb).cpu().numpy()
        preds_batches.append(out)

preds_s = np.vstack(preds_batches)
preds = y_scaler.inverse_transform(preds_s).ravel()

# Compute R-squared alongside RMSE
r2 = r2_score(y_test, preds)

plt.figure()
plt.scatter(y_test, preds, alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"Regression: Predicted vs Actual (R2={r2:.4f})")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "--")
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Test RMSE (Dropout): {rmse:.2f} | R2: {r2:.4f}")