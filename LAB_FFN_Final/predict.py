"""Load only trusted model checkpoints. Inputs require original predictor names."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from train import FFN,clean
def infer(model_path,csv_path):
    ck=torch.load(model_path,map_location="cpu",weights_only=True)
    x=clean(pd.read_csv(csv_path))[ck["features"]].to_numpy(dtype=float)
    x=np.where(np.isnan(x),np.array(ck["imputer_statistics"]),x)
    x=(x-np.array(ck["scaler_mean"]))/np.array(ck["scaler_scale"])
    xt=torch.tensor(x,dtype=torch.float32); pred=[]
    for member in ck["members"]:
        m=FFN(len(ck["features"])); m.load_state_dict(member["state"]); m.eval()
        with torch.no_grad():
            z=m(xt)
            pred.append(torch.sigmoid(z).numpy() if ck["task"]=="classification"
                else z.numpy()*member["sd"]+member["mu"])
    p=np.mean(pred,axis=0)
    out=pd.DataFrame({"score" if ck["task"]=="classification" else "predicted_consumption":p})
    if ck["task"]=="classification": out["predicted_poverty"]=(p>=ck["threshold"]).astype(int)
    return out
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--csv",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();infer(a.model,a.csv).to_csv(a.output,index=False)
