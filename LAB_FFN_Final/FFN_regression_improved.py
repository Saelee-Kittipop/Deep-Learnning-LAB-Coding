"""
FFN regression แบบไฟล์เดียว: อ่านข้อมูล ฝึก baseline + improved ประเมิน บันทึกโมเดล และแสดงกราฟ
ใช้จาก root: python LAB_FFN_Final/FFN_regression_improved.py
ไม่เปิดหน้าต่าง: เติม --no-show | ข้อมูลอื่น: --data /path/to/SES2023.csv
ทำนายโดยไม่ฝึกใหม่: --predict new_data.csv --model /path/to/regression_models.pt
ต้องติดตั้ง: torch, numpy, pandas>=2.1, scikit-learn, matplotlib
ไฟล์นี้ทำงานได้โดยไม่ต้องมี train.py, predict.py หรือ originals/ อยู่ข้างกัน
"""

from pathlib import Path
import argparse
import copy
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TASK = "regression"
TARGET = "rspcexp23"
# วาง CSV ข้างไฟล์นี้ได้ หรือใช้ SES2023.csv ที่ root ของ repository
DATA = (
    ROOT / "SES2023.csv"
    if (ROOT / "SES2023.csv").exists()
    else ROOT.parent / "SES2023.csv"
)
SEEDS = [0, 1, 2]
MAX_EPOCHS = 50
PATIENCE = 12
BATCH_SIZE = 512
LEARNING_RATE = 0.001


# 1. โครงข่าย: predictors -> 64 -> 32 -> 1; ทั้ง baseline และ improved ใช้ ReLU
class FFN(nn.Module):
    def __init__(self, inputs=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def clean(frame):
    return (
        frame.map(lambda x: x.strip() if isinstance(x, str) else x)
        .replace({"Yes": 1, "No": 0, "yes": 1, "no": 0})
        .apply(pd.to_numeric, errors="coerce")
    )


def metrics(y, p, t=0.5):
    return dict(
        rmse=float(np.sqrt(mean_squared_error(y, p))),
        mae=float(mean_absolute_error(y, p)),
        r2=float(r2_score(y, p)),
    )


# 2. ฝึกโมเดลปรับปรุงหนึ่ง seed และคืน checkpoint ที่ validation ดีที่สุด
def fit(x, y, xv, yv, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FFN(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mu, sd = (float(y.mean()), float(y.std()))
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    vt = torch.tensor(xv, dtype=torch.float32)
    lossfn = nn.SmoothL1Loss(beta=1.0)
    best = np.inf
    wait = 0
    history = []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for ix in torch.randperm(len(x)).split(BATCH_SIZE):
            opt.zero_grad()
            loss = lossfn(model(xt[ix]), yt[ix])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            z = model(vt)
            p = z.numpy() * sd + mu
        objective = np.sqrt(mean_squared_error(yv, p))
        history.append(
            dict(
                epoch=ep,
                training_loss=float(np.mean(losses)),
                validation_objective=float(objective),
            )
        )
        if objective < best:
            best = objective
            bestep = ep
            state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if ep == 1 or ep % 10 == 0:
            print(
                f"Improved seed {seed}, epoch {ep}: validation={objective:.6f}",
                flush=True,
            )
        if wait >= PATIENCE:
            break
    model.load_state_dict(state)
    model.eval()
    return model, mu, sd, bestep, history


def predict(model, x, mu, sd):
    with torch.no_grad():
        z = model(torch.tensor(x, dtype=torch.float32))
        return z.numpy() * sd + mu


# 3. Baseline ในไฟล์เดียว: คงวิธีฝึกของ LAB เดิมเพื่อใช้เทียบกับรายงาน
# คง median ก่อน split ตาม LAB เดิม (มี leakage); improved fit median เฉพาะ train
# ใช้ CPU ทั้งสองแบบเพื่อทำซ้ำได้; ไม่อ่านหรือ exec ไฟล์ Python อื่น


def baseline(frame):
    torch.manual_seed(0)
    d = frame.fillna(frame.median(numeric_only=True))
    x = d.drop(columns=["pov_exp", "rspcexp23"]).to_numpy()
    y = d[TARGET].to_numpy()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=1)
    scaler = StandardScaler().fit(xtr)
    xtr = scaler.transform(xtr)
    xte = scaler.transform(xte)
    y_scaler = StandardScaler().fit(ytr.reshape(-1, 1))
    yt = y_scaler.transform(ytr.reshape(-1, 1)).ravel()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(xtr, dtype=torch.float32), torch.tensor(yt, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = FFN(xtr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossfn = nn.MSELoss()
    history = []
    for epoch in range(1, 51):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        history.append(dict(epoch=epoch, training_loss=running / len(dataset)))
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Baseline epoch {epoch}/50: loss={history[-1]['training_loss']:.6f}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(xte, dtype=torch.float32)).numpy()
    p = y_scaler.inverse_transform(z[:, None]).ravel()
    return yte, p, history


# 4. การทดสอบผลต่างแบบจับคู่: ช่วงความเชื่อมั่นและ p-value จาก test rows เดียวกัน
def paired_stats(y, a, b, t, reps=2000):
    rng = np.random.default_rng(2026)

    # t ใช้เฉพาะ classification; ไม่มี threshold สำหรับค่าใช้จ่าย
    def statistic(yy, aa, bb):
        return np.sqrt(np.mean((yy - aa) ** 2)) - np.sqrt(np.mean((yy - bb) ** 2))

    observed = statistic(y, a, b)
    gains = []
    extreme = 0
    for _ in range(reps):
        ix = rng.integers(0, len(y), len(y))
        gains.append(statistic(y[ix], a[ix], b[ix]))
        swap = rng.random(len(y)) < 0.5
        val = statistic(y, np.where(swap, b, a), np.where(swap, a, b))
        extreme += abs(val) >= abs(observed)
    p = (extreme + 1) / (reps + 1)
    return dict(
        observed_gain=float(observed),
        gain_ci95=np.quantile(gains, [0.025, 0.975]).tolist(),
        two_sided_randomization_p=p,
        bonferroni_p=min(1, 2 * p),
        resamples=reps,
        limitation="Conditional on fitted models; independent-row assumption; previously inspected holdout.",
    )


# 5. ทำนายข้อมูลใหม่ด้วย weights, median, scaler และ threshold ที่บันทึกไว้ในไฟล์ .pt
# ใช้ได้ด้วย --predict new_data.csv โดยไม่ต้องเปิด predict.py


def infer(model_path, csv_path):
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    if ck["task"] != TASK:
        raise ValueError(f"This file expects a {TASK} checkpoint")
    x = clean(pd.read_csv(csv_path))[ck["features"]].to_numpy(dtype=float)
    x = np.where(np.isnan(x), np.array(ck["imputer_statistics"]), x)
    x = ((x - np.array(ck["scaler_mean"])) / np.array(ck["scaler_scale"])).astype(
        "float32"
    )
    predictions = []
    for member in ck["members"]:
        m = FFN(len(ck["features"]))
        m.load_state_dict(member["state"])
        m.eval()
        predictions.append(predict(m, x, member["mu"], member["sd"]))
    p = np.mean(predictions, axis=0)
    result = pd.DataFrame({"predicted_consumption": p})
    return result


# 6. กราฟ: บันทึก PNG เสมอ; แสดงหน้าต่าง/ผลใน notebook ถ้าไม่ใช้ --no-show
def plot_results(out, y, bp, p, result, histories, baseline_history, unused, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{TASK}_{name}.png", dpi=160, bbox_inches="tight")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(
        [h["epoch"] for h in baseline_history],
        [h["training_loss"] for h in baseline_history],
    )
    axes[0].set(
        title="Baseline training loss",
        xlabel="Epoch",
        ylabel="MSE on standardized target",
    )
    for seed, h, best in zip(SEEDS, histories, result["checkpoints"]):
        ep = [v["epoch"] for v in h]
        axes[1].plot(ep, [v["training_loss"] for v in h], label=f"Seed {seed}")
        objective = [v["validation_objective"] for v in h]
        (line,) = axes[2].plot(ep, objective, label=f"Seed {seed}")
        axes[2].scatter(
            best, objective[best - 1], color=line.get_color(), marker="*", s=100
        )
    axes[1].set(
        title="Improved training loss",
        xlabel="Epoch",
        ylabel="SmoothL1 on standardized target",
    )
    axes[2].set(
        title="Validation: stars = chosen epochs",
        xlabel="Epoch",
        ylabel="RMSE (lower is better)",
    )
    axes[1].legend()
    axes[2].legend()
    # Loss สองแบบมีนิยาม/สเกลต่างกัน ไม่ใช้ระดับเส้นเทียบกันโดยตรง
    save(fig, "training")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pred, label in zip(axes, [bp, p], ["Baseline", "Improved"]):
        ax.scatter(y, pred, s=8, alpha=0.25)
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set(
            xlabel="Actual consumption",
            ylabel="Predicted consumption",
            title=f"{label}: R2={r2_score(y, pred):.4f}",
        )
    save(fig, "actual_predicted")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for pred, label in [(bp, "Baseline"), (p, "Improved")]:
        residual = y - pred
        axes[0].scatter(pred, residual, s=8, alpha=0.2, label=label)
        axes[1].hist(residual, bins=80, alpha=0.5, label=label)
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].set(
        title="Residuals on test data",
        xlabel="Predicted consumption",
        ylabel="Actual - predicted",
    )
    axes[1].set(
        title="Residual distribution (full range)",
        xlabel="Actual - predicted",
        ylabel="Count",
    )
    axes[0].legend()
    axes[1].legend()
    save(fig, "residuals")

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, metric in zip(axes, ["rmse", "mae", "r2"]):
        vals = [result["baseline"][metric], result["improved"][metric]]
        ax.bar(["Baseline", "Improved"], vals, color=["#6488aa", "#d38a44"])
        ax.set(
            title=metric.upper(),
            ylabel="Original target units" if metric != "r2" else "Score",
        )
        for i, value in enumerate(vals):
            ax.text(i, value, f"{value:,.4f}", ha="center", va="bottom")
        ax.margins(y=0.2)
    save(fig, "metrics")

    if show:
        plt.show()
    plt.close("all")


# 7. รันครบวงจร: เตรียมข้อมูล -> ฝึก 3 seeds -> baseline -> test -> บันทึก -> กราฟ


def run(data_path=DATA, output_dir=OUT, show=True):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    d = clean(pd.read_csv(data_path))
    y = d[TARGET].to_numpy()
    if not np.isfinite(y).all():
        raise ValueError(
            "Target contains missing/non-finite values; do not impute targets"
        )
    # ต้องเก็บ pov_exp ไว้เพื่อ stratify split แบบเดิม แต่ไม่นำเข้า predictors
    features = d.drop(columns=["pov_exp", "rspcexp23"]).columns.tolist()
    x = d[features].to_numpy()
    # คง test เดิม 20%; train 60%, validation/calibration 10%, reserved assessment 10%
    pool, test = train_test_split(np.arange(len(d)), test_size=0.2, random_state=1)
    tr, remaining = train_test_split(
        pool, test_size=0.25, random_state=20261020, stratify=d.pov_exp.iloc[pool]
    )
    val, assessment = train_test_split(
        remaining, test_size=0.5, random_state=7, stratify=d.pov_exp.iloc[remaining]
    )
    groups = [set(s) for s in [tr, val, assessment, test]]
    assert sum(map(len, groups)) == len(set.union(*groups)) == len(d)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xt = scaler.fit_transform(imp.fit_transform(x[tr])).astype("float32")
    xv = scaler.transform(imp.transform(x[val])).astype("float32")
    members, pv, histories = [], [], []
    for seed in SEEDS:
        m, mu, sd, ep, h = fit(xt, y[tr], xv, y[val], seed)
        members.append((m, mu, sd, ep))
        pv.append(predict(m, xv, mu, sd))
        histories.append(h)
        print(TASK, "seed", seed, "checkpoint", ep, flush=True)
    vp = np.mean(pv, axis=0)
    t = 0.5  # unused for regression
    # ไม่เลือก parameters/threshold จาก test; assessment ถูกสงวน ไม่ใช้ในการรันนี้
    xe = scaler.transform(imp.transform(x[test])).astype("float32")
    p = np.mean([predict(m, xe, mu, sd) for m, mu, sd, _ in members], axis=0)
    by, bp, baseline_history = baseline(d)
    np.testing.assert_allclose(by, y[test])
    result = dict(
        task=TASK,
        source="fresh_execution",
        baseline=metrics(by, bp),
        improved=metrics(y[test], p, t),
        threshold=t,
        checkpoints=[ep for _, _, _, ep in members],
        n_train=len(tr),
        n_calibration=len(val),
        n_assessment=len(assessment),
        n_test=len(test),
        statistics=paired_stats(y[test], bp, p, t),
    )
    curves = None
    checkpoint = dict(
        task=TASK,
        features=features,
        threshold=t,
        imputer_statistics=imp.statistics_.tolist(),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        members=[dict(state=m.state_dict(), mu=mu, sd=sd) for m, mu, sd, _ in members],
    )
    torch.save(checkpoint, out / f"{TASK}_models.pt")
    pd.DataFrame(dict(row_index=test, target=y[test], baseline=bp, improved=p)).to_csv(
        out / f"{TASK}_predictions.csv", index=False
    )
    np.savez_compressed(
        out / f"{TASK}_splits.npz",
        train=tr,
        calibration=val,
        assessment=assessment,
        test=test,
    )
    (out / f"{TASK}_history.json").write_text(json.dumps(histories), encoding="utf-8")
    (out / f"{TASK}_baseline_history.json").write_text(
        json.dumps(baseline_history), encoding="utf-8"
    )
    (out / f"{TASK}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        {"baseline": result["baseline"], "improved": result["improved"]}
    ).rename_axis("metric").to_csv(out / f"{TASK}_comparison.csv")
    environment = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        device="cpu",
        seeds=SEEDS,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )
    (out / f"{TASK}_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (out / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    plot_results(out, y[test], bp, p, result, histories, baseline_history, curves, show)
    print(f"Saved results and graphs to: {out.resolve()}", flush=True)
    return result


# 8. กด Run ใน IDE ได้เลย หรือรันคำสั่งจาก terminal ตามตัวอย่างด้านบน


def main():
    parser = argparse.ArgumentParser(
        description="Standalone FFN regression: baseline, improved model, metrics, inference and graphs"
    )
    parser.add_argument("--data", type=Path, default=DATA, help="Path to SES2023.csv")
    parser.add_argument(
        "--output", type=Path, default=OUT, help="Folder for models, tables and PNGs"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Save graphs without opening windows"
    )
    parser.add_argument(
        "--predict", type=Path, help="Predict new CSV instead of training"
    )
    parser.add_argument("--model", type=Path, default=OUT / f"{TASK}_models.pt")
    parser.add_argument(
        "--predictions", type=Path, default=OUT / f"{TASK}_new_predictions.csv"
    )
    args = parser.parse_args()
    if args.predict:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        infer(args.model, args.predict).to_csv(args.predictions, index=False)
        print(f"Saved predictions to: {args.predictions.resolve()}")
    else:
        run(args.data, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
