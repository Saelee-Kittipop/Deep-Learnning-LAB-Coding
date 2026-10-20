"""Verify saved inference and package source, data and fresh results."""
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
from predict import infer
from train import ROOT,OUT,DATA
for task in ["classification","regression"]:
    saved=pd.read_csv(OUT/f"{task}_predictions.csv")
    # Test inference on a bounded subset; same feature ordering and preprocessing.
    sample=pd.read_csv(DATA).iloc[saved.row_index.to_numpy()[:100]]
    temporary=OUT/"inference_check_input.csv";sample.to_csv(temporary,index=False)
    actual=infer(OUT/f"{task}_models.pt",temporary).iloc[:,0].to_numpy()
    np.testing.assert_allclose(actual,saved.improved.to_numpy()[:100],rtol=2e-5,atol=1e-3)
    temporary.unlink()
    assert (ROOT/"originals"/f"FFN_{task}.py").read_bytes()==(ROOT.parent/f"FFN_{task}.py").read_bytes()
rows=[]
for task in ["classification","regression"]:
    d=json.loads((OUT/f"{task}.json").read_text())
    for key,v in d["baseline"].items():rows.append(dict(task=task,metric=key,baseline=v,improved=d["improved"][key]))
pd.DataFrame(rows).to_csv(OUT/"comparison.csv",index=False)
archive=ROOT.parent/"LAB_FFN_Final_TH.zip"
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
    z.write(DATA,"SES2023.csv")
    # Root originals enable byte-for-byte checks after extracting the bundle.
    for task in ["classification","regression"]:
        z.write(ROOT.parent/f"FFN_{task}.py",f"FFN_{task}.py")
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            z.write(path,path.relative_to(ROOT.parent))
print(archive)
