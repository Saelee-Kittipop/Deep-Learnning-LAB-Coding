# FFN ปรับปรุง: ใช้งานเพียง 2 ไฟล์

เปิด `FFN_classification_improved.py` หรือ `FFN_regression_improved.py` ได้เลย แต่ละไฟล์ทำงานได้ด้วยตัวเอง ไม่ import โค้ดจาก `train.py`, `predict.py` หรือ `originals/` และไม่ต้องเรียกไฟล์ Python อื่น

## แต่ละไฟล์ทำอะไร

| ขั้นตอน | Classification | Regression |
|---|---|---|
| สิ่งที่ทำนาย | `pov_exp`: 0 ไม่ยากจน, 1 ยากจน | `rspcexp23`: ค่าใช้จ่ายบริโภคต่อหัว |
| ข้อมูลเข้า | ตัดตัวแปรตามทั้งสองออก เหลือ 32 predictors | เหมือน classification |
| โครงข่าย | 32 → 64 → 32 → 1, ReLU | เหมือน classification |
| เตรียมข้อมูลปรับปรุง | median และ StandardScaler fit เฉพาะ train | เหมือน classification และ standardize target จาก train |
| Loss เดิม → ใหม่ | BCE → Weighted BCE, `pos_weight=5` | MSE → `SmoothL1Loss(beta=1)` บน standardized target |
| Batch เดิม → ใหม่ | 64 → 512 | 32 → 512 |
| การเลือก epoch | PR-AUC (average precision) บน validation สูงสุด | RMSE บน validation ต่ำสุด ในหน่วยค่าใช้จ่ายเดิม |
| Early stopping | สูงสุด 50 epochs, patience 12 | เหมือน classification |
| รวมผล | เฉลี่ยคะแนนจาก seeds 0, 1, 2 | เฉลี่ยค่าทำนายจาก seeds 0, 1, 2 |
| เกณฑ์ตัดสิน | เลือก threshold ที่ F1 สูงสุดบน calibration | ไม่ใช้ threshold |
| ตัวชี้วัด test | Accuracy, Precision, Recall, F1, PR-AUC, ROC-AUC | RMSE, MAE, R² |
| การทดสอบผลต่าง | F1 gain แบบจับคู่ 2,000 ครั้ง | RMSE reduction แบบจับคู่ 2,000 ครั้ง |

ทั้งสองใช้ Adam และ learning rate 0.001 ไม่มี Dropout หรือ Batch Normalization ค่าปรับและ split เหมือน `train.py` รุ่นก่อน การรวมไฟล์ครั้งนี้ไม่ได้ค้นหา hyperparameters ใหม่

ในโค้ดมีหมายเลขขั้นตอน 1–8: โครงข่าย → ฝึก improved → ฝึก baseline → ทดสอบผลต่าง → โหลดโมเดลทำนาย → กราฟ → รันครบวงจร → รับคำสั่ง

## ติดตั้งครั้งเดียว

ใช้ Python 3.12 เพื่อให้ใกล้เคียง workflow เดิม รันจาก root ของ repository:

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r LAB_FFN_Final/requirements.txt
```

`python-docx` ใน requirements ใช้เมื่อสร้างรายงาน Word เท่านั้น โมเดลทั้งสองต้องใช้ torch, numpy, pandas, scikit-learn และ matplotlib

## วิธีรันปกติ

```bash
python LAB_FFN_Final/FFN_classification_improved.py
python LAB_FFN_Final/FFN_regression_improved.py
```

หรือเปิดไฟล์ใดไฟล์หนึ่งแล้วกด Run ใน IDE โปรแกรมจะใช้ CPU, ฝึก improved 3 seeds และ baseline 1 โมเดล, พิมพ์ผล, บันทึกไฟล์ และแสดงกราฟหลังฝึกเสร็จ ปิดหน้าต่างกราฟเมื่อดูเสร็จ

ถ้ารันบน server หรือไม่ต้องการเปิดหน้าต่างกราฟ:

```bash
python LAB_FFN_Final/FFN_classification_improved.py --no-show
python LAB_FFN_Final/FFN_regression_improved.py --no-show
```

ยังบันทึก PNG ครบเหมือนเดิม

**หากต้องการคัดลอกไปใช้แยก:** นำเพียงไฟล์ Python ของงานที่ต้องการและ `SES2023.csv` ไปไว้โฟลเดอร์เดียวกัน แล้วรัน `python FFN_classification_improved.py` หรือ `python FFN_regression_improved.py` ได้เลย หลังติดตั้ง dependencies ข้างต้นแล้ว โปรแกรมค้นหาข้อมูลข้างไฟล์ก่อน แล้วจึงค้นหาในโฟลเดอร์แม่ ไม่อิง current working directory

กำหนดที่อยู่ข้อมูลและผลเองได้ เช่น:

```bash
python LAB_FFN_Final/FFN_classification_improved.py --data /path/to/SES2023.csv --output /path/to/results --no-show
```

## กราฟและผลอยู่ที่ไหน

ค่าปกติบันทึกใน `LAB_FFN_Final/outputs/` แต่ละงานมีชื่อขึ้นต้นต่างกัน จึงไม่ทับกัน การรันงานเดิมซ้ำจะเขียนผลของงานนั้นใหม่

| Classification: PNG | แสดงอะไร |
|---|---|
| `classification_training.png` | loss baseline, weighted loss improved และ validation PR-AUC พร้อม epoch ที่เลือก |
| `classification_confusion_matrix.png` | confusion matrix ของ baseline และ improved |
| `classification_precision_recall_threshold.png` | PR curve บน test และ trade-off ของ threshold บน calibration |
| `classification_metrics.png` | ตัวชี้วัด baseline เทียบ improved |

| Regression: PNG | แสดงอะไร |
|---|---|
| `regression_training.png` | loss baseline, SmoothL1 loss improved และ validation RMSE พร้อม epoch ที่เลือก |
| `regression_actual_predicted.png` | actual เทียบ predicted ของทั้งสองแบบ พร้อมเส้นทำนายตรงค่าจริง |
| `regression_residuals.png` | residual เทียบ predicted และ histogram ของ residual |
| `regression_metrics.png` | RMSE, MAE, R² ของ baseline เทียบ improved |

ระดับ loss คนละชนิดเปรียบเทียบกันตรง ๆ ไม่ได้ จึงแยก panel และระบุชนิด loss ไว้ชัดเจน

แต่ละงานยังสร้าง `*_models.pt`, `*_predictions.csv`, `*_comparison.csv`, `*.json`, `*_history.json`, `*_baseline_history.json`, `*_splits.npz`, `*_environment.json` และ classification มี `calibration_thresholds.csv` ผลและ p-value คำนวณจากการรันจริง ไม่ได้คัดลอกตัวเลขเก่า checkpoint มี weights, feature order, median, scaler, target scaling และ threshold

## ทำนายข้อมูลใหม่โดยใช้ไฟล์เดิม ไม่ฝึกซ้ำ

```bash
python LAB_FFN_Final/FFN_classification_improved.py --predict new_data.csv
python LAB_FFN_Final/FFN_regression_improved.py --predict new_data.csv
```

input ต้องมีชื่อ predictors เดิมครบ แต่ไม่ต้องมี target ผลอยู่ใน `outputs/classification_new_predictions.csv` และ `outputs/regression_new_predictions.csv` หากเก็บ checkpoint ที่อื่น ให้ใช้ `--model /path/to/model.pt` และกำหนด CSV ผลลัพธ์ด้วย `--predictions /path/to/predictions.csv`

## วิธีเปรียบเทียบและข้อจำกัด

คง test split เดิม 20% (`random_state=1`) ส่วนที่เหลือแบ่ง train ประมาณ 60%, validation/calibration 10% และ assessment ที่สงวนไว้ 10% โดย stratify ตามความยากจน ใช้ชุด calibration เดียวกันเลือก checkpoint และ threshold ตามการทดลองเดิม จึงไม่ใช่ independent calibration สำหรับยืนยันประสิทธิภาพ

baseline ที่ฝังในแต่ละไฟล์คงวิธี LAB เดิม: seed 0, 50 epochs, train 80%, batch 64/32, BCE/MSE, threshold 0.5 และ median ก่อนแบ่งข้อมูล ซึ่งมีข้อจำกัดด้าน leakage ขณะที่ improved fit preprocessing เฉพาะ train ใช้ CPU ทั้งสองแบบเพื่อให้สภาพแวดล้อมเดียวกัน ผลไม่ควรถูกอธิบายว่าเป็นผลของ loss อย่างเดียว เพราะวิธีฝึกหลายอย่างเปลี่ยนพร้อมกัน

paired bootstrap และ prediction randomization ใช้ 2,000 ครั้ง พร้อม Bonferroni สำหรับสองงาน เป็น inference แบบมีเงื่อนไขบนโมเดลที่ฝึกแล้วและสมมติแถวเป็นอิสระ ไม่ได้รองรับ survey weights/clusters และ test นี้เคยถูกตรวจในรอบก่อน จึงไม่ใช่ external validation ค่าที่รันได้อาจต่างตามเวอร์ชัน ไม่รับประกันว่าจะชนะ baseline ทุก metric

classification เป็นเครื่องมือช่วยตรวจสอบ ยังไม่ควรตัดสิทธิสวัสดิการอัตโนมัติ หากต้องการ recall สูงขึ้น ให้ลองลด threshold และเลือกจาก calibration ตามจำนวน false positives ที่รับได้ ไม่เลือกจาก test คะแนนจาก weighted BCE ยังไม่ใช่ความน่าจะเป็นที่สอบเทียบแล้ว regression ไม่ควรบวก/ลบ MAE หรือ RMSE จากค่าทำนายทุกคน

## ตรวจการรวมไฟล์แล้วอย่างไร

รันสองไฟล์กับ SES2023 ครบ และรัน `train.py` รุ่นก่อนรวมไฟล์จาก commit `147fbc4` แยกในสภาพแวดล้อมเดียวกัน พบว่า predictions ของ baseline และ improved บน test 9,933 แถว ตลอดจน metrics และ weights ของ improved ตรงกันทุกค่า ตรวจกราฟ PNG ครบ 8 ไฟล์ และตรวจคำสั่งของแต่ละไฟล์เมื่อคัดลอกไปอยู่นอก repository แล้ว

## ไฟล์อื่นต้องเปิดหรือไม่

**สำหรับเรียนและรันโมเดล ไม่ต้องเปิดเพิ่ม** ไฟล์ต่อไปนี้เก็บไว้รองรับชุดรายงานเดิม:

| ไฟล์ | ใช้เมื่อ |
|---|---|
| `train.py` | workflow เดิมเรียกรันสองไฟล์หลักอัตโนมัติ ไม่มีโค้ดฝึกซ้ำอีกชุด |
| `predict.py` | รองรับคำสั่งทำนายแบบเดิม; ใช้ `--predict` ในสองไฟล์หลักแทนได้ |
| `build_report.py` | สร้าง Word จากผลรันครบสองงาน |
| `verify_and_package.py` | ตรวจโหลดโมเดลและแพ็ก ZIP |
| `originals/` | เก็บ LAB เดิมสำหรับอ้างอิง ไม่ถูกเรียกโดยสองไฟล์หลัก |
| `historical_results.json` | เก็บผลรอบก่อน แยกจากผลรันปัจจุบัน |

หากต้องการสร้างรายงาน/ZIP ต่อ หลังรันทั้งสองไฟล์แล้ว:

```bash
python LAB_FFN_Final/build_report.py
python LAB_FFN_Final/verify_and_package.py
```

รายงานใช้ฟอนต์ Garuda ต้องเปิดตรวจจำนวนหน้าใน Word/LibreOffice ก่อนส่ง ส่วน GitHub Actions จะตรวจความยาว 1–2 หน้าและเก็บ ZIP ที่ Actions > Build LAB FFN Thai package > run ที่สำเร็จ > Artifacts > LAB_FFN_Final_TH
