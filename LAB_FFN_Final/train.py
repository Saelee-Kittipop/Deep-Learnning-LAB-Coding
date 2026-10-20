"""Reconstructed final experiment; rerun results replace no historical claims."""
from pathlib import Path
import argparse, copy, hashlib, json, platform, sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, confusion_matrix, precision_recall_curve)
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT.parent / "SES2023.csv"
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class FFN(nn.Module):
    def __init__(self, inputs=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs,64), nn.ReLU(),
            nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self,x):
        return self.net(x).squeeze(1)

def clean(frame):
    return frame.map(lambda x:x.strip() if isinstance(x,str) else x).replace(
        {"Yes":1,"No":0,"yes":1,"no":0}).apply(pd.to_numeric,errors="coerce")

def metrics(y,p,task,t=.5):
    if task=="classification":
        h=p>=t
        return dict(accuracy=float(accuracy_score(y,h)),
            precision=float(precision_score(y,h,zero_division=0)),
            recall=float(recall_score(y,h,zero_division=0)),
            f1=float(f1_score(y,h,zero_division=0)),
            pr_auc=float(average_precision_score(y,p)),
            roc_auc=float(roc_auc_score(y,p)))
    return dict(rmse=float(np.sqrt(mean_squared_error(y,p))),
        mae=float(mean_absolute_error(y,p)),r2=float(r2_score(y,p)))

def threshold(y,p):
    pr,re,th=precision_recall_curve(y,p)
    fs=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12)
    return float(th[np.argmax(fs)])

def fit(x,y,xv,yv,task,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=FFN(x.shape[1])
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    mu,sd=(float(y.mean()),float(y.std())) if task=="regression" else (0.,1.)
    xt=torch.tensor(x,dtype=torch.float32)
    yt=torch.tensor((y-mu)/sd,dtype=torch.float32)
    vt=torch.tensor(xv,dtype=torch.float32)
    lossfn=(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.))
        if task=="classification" else nn.SmoothL1Loss(beta=1.))
    best=np.inf; wait=0; history=[]
    for ep in range(1,51):
        model.train(); losses=[]
        for ix in torch.randperm(len(x)).split(512):
            opt.zero_grad(); loss=lossfn(model(xt[ix]),yt[ix])
            loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            z=model(vt)
            p=torch.sigmoid(z).numpy() if task=="classification" else z.numpy()*sd+mu
        objective=(-average_precision_score(yv,p) if task=="classification"
            else np.sqrt(mean_squared_error(yv,p)))
        history.append(dict(epoch=ep,training_loss=float(np.mean(losses)),
            validation_objective=float(objective)))
        if objective<best:
            best=objective; bestep=ep; state=copy.deepcopy(model.state_dict()); wait=0
        else:
            wait+=1
        if wait>=12: break
    model.load_state_dict(state); model.eval()
    return model,mu,sd,bestep,history

def predict(model,x,task,mu,sd):
    with torch.no_grad():
        z=model(torch.tensor(x,dtype=torch.float32))
        return torch.sigmoid(z).numpy() if task=="classification" else z.numpy()*sd+mu

def original(task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src=(ROOT/"originals"/f"FFN_{task}.py").read_text(encoding="utf-8")
    src=src.replace("df.applymap(","df.map(")
    src=src.replace("LAB3_Generative_Model/SES2023.csv",DATA.as_posix())
    src=src.replace('"SES2023.csv"',repr(DATA.as_posix()))
    src=src.replace("plt.show()", 'plt.close("all")')
    ns={"__name__":"__main__"}
    exec(compile(src,f"original_{task}","exec"),ns)
    m=ns["model"]; m.eval()
    with torch.no_grad():
        z=m(ns["X_test_t"].to(ns["device"])).cpu().numpy().ravel()
    p=(1/(1+np.exp(-np.clip(z,-80,80))) if task=="classification"
        else ns["y_scaler"].inverse_transform(z[:,None]).ravel())
    return ns["y_test"],p

def paired_stats(y,a,b,task,t,reps=2000):
    rng=np.random.default_rng(2026)
    if task=="classification":
        a=a>=.5; b=b>=t
        def statistic(yy,aa,bb): return f1_score(yy,bb,zero_division=0)-f1_score(yy,aa,zero_division=0)
        observed=statistic(y,a,b)
    else:
        def statistic(yy,aa,bb): return np.sqrt(np.mean((yy-aa)**2))-np.sqrt(np.mean((yy-bb)**2))
        observed=statistic(y,a,b)
    gains=[]; extreme=0
    for _ in range(reps):
        ix=rng.integers(0,len(y),len(y))
        gains.append(statistic(y[ix],a[ix],b[ix]))
        swap=rng.random(len(y))<.5
        val=statistic(y,np.where(swap,b,a),np.where(swap,a,b))
        extreme+=abs(val)>=abs(observed)
    p=(extreme+1)/(reps+1)
    return dict(observed_gain=float(observed),gain_ci95=np.quantile(gains,[.025,.975]).tolist(),
        two_sided_randomization_p=p,bonferroni_p=min(1,2*p),resamples=reps,
        limitation="Conditional on fitted models; independent-row assumption; previously inspected holdout.")

def run(task):
    OUT.mkdir(exist_ok=True)
    d=clean(pd.read_csv(DATA))
    target="pov_exp" if task=="classification" else "rspcexp23"
    y=d[target].to_numpy()
    assert np.isfinite(y).all(), "Do not impute target values"
    features=d.drop(columns=["pov_exp","rspcexp23"]).columns.tolist()
    x=d[features].to_numpy()
    pool,test=train_test_split(np.arange(len(d)),test_size=.2,random_state=1)
    tr,remaining=train_test_split(pool,test_size=.25,random_state=20261020,stratify=d.pov_exp.iloc[pool])
    val,assessment=train_test_split(remaining,test_size=.5,random_state=7,stratify=d.pov_exp.iloc[remaining])
    groups=[set(s) for s in [tr,val,assessment,test]]
    assert sum(map(len,groups))==len(set.union(*groups))==len(d)
    imp=SimpleImputer(strategy="median"); scaler=StandardScaler()
    xt=scaler.fit_transform(imp.fit_transform(x[tr])).astype("float32")
    xv=scaler.transform(imp.transform(x[val])).astype("float32")
    members=[]; pv=[]; histories=[]
    for seed in [0,1,2]:
        m,mu,sd,ep,h=fit(xt,y[tr],xv,y[val],task,seed)
        members.append((m,mu,sd,ep)); pv.append(predict(m,xv,task,mu,sd)); histories.append(h)
        print(task,"seed",seed,"checkpoint",ep,flush=True)
    t=threshold(y[val],np.mean(pv,axis=0)) if task=="classification" else .5
    # Frozen configuration from prior development study. Never select on test.
    xe=scaler.transform(imp.transform(x[test])).astype("float32")
    p=np.mean([predict(m,xe,task,mu,sd) for m,mu,sd,_ in members],axis=0)
    by,bp=original(task)
    np.testing.assert_allclose(by,y[test])
    result=dict(task=task,source="fresh_execution",baseline=metrics(by,bp,task),
        improved=metrics(y[test],p,task,t),threshold=t,
        checkpoints=[ep for _,_,_,ep in members],n_train=len(tr),n_calibration=len(val),
        n_assessment=len(assessment),n_test=len(test),
        statistics=paired_stats(y[test],bp,p,task,t))
    if task=="classification":
        result["confusion_matrix"]=confusion_matrix(y[test],p>=t).tolist()
        curves=[dict(threshold=float(v),**metrics(y[val],np.mean(pv,axis=0),task,v)) for v in np.arange(.01,1,.01)]
        pd.DataFrame(curves).to_csv(OUT/"calibration_thresholds.csv",index=False)
    checkpoint=dict(task=task,features=features,threshold=t,
        imputer_statistics=imp.statistics_.tolist(),scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),members=[dict(state=m.state_dict(),mu=mu,sd=sd)
            for m,mu,sd,_ in members])
    torch.save(checkpoint,OUT/f"{task}_models.pt")
    pd.DataFrame(dict(row_index=test,target=y[test],baseline=bp,improved=p)).to_csv(
        OUT/f"{task}_predictions.csv",index=False)
    np.savez_compressed(OUT/f"{task}_splits.npz",train=tr,calibration=val,assessment=assessment,test=test)
    (OUT/f"{task}_history.json").write_text(json.dumps(histories),encoding="utf-8")
    (OUT/f"{task}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=["all","classification","regression"],default="all")
    a=p.parse_args(); OUT.mkdir(exist_ok=True)
    (OUT/"environment.json").write_text(json.dumps(dict(python=platform.python_version(),
        torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest()),indent=2))
    for task in (["classification","regression"] if a.task=="all" else [a.task]): run(task)
