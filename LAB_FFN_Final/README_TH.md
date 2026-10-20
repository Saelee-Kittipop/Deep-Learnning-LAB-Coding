# LAB FFN ฉบับรายงานภาษาไทย

ชุดนี้เพิ่มจากไฟล์เดิมโดยไม่แก้ไขต้นฉบับ ใช้ SES2023.csv ที่ root ของ repository

## สถานะและที่มาของผล
โค้ดชุดนี้สร้างขึ้นใหม่จากวิธีทดลองที่ระบุในบทสนทนา ไม่ใช่สำเนา binary model เดิมที่เรียกคืนมาได้
ตัวเลขจากการทดลองก่อนหน้าและ baseline ที่ผู้ใช้รันอยู่ใน historical_results.json โดยระบุแหล่งที่มา
ผลที่รันจริงครั้งใหม่เท่านั้นจะอยู่ใน outputs/ และถูกนำไปสร้างรายงาน Word
ค่า metric อาจต่างจากครั้งก่อนตามเวอร์ชันและอุปกรณ์ ไม่รับประกันว่าทุก metric จะสูงขึ้น
การศึกษารอบก่อนใช้ test นี้มาแล้ว การรันซ้ำจึงไม่ใช่การยืนยันบน holdout ใหม่
ไม่ได้แนบหลักฐานการเปรียบเทียบเทคนิคทั้ง 57 runs เดิม เพราะไฟล์เหล่านั้นเรียกคืนไม่ได้ในรอบจัดทำ GitHub นี้

## ดาวน์โหลดชุดสมบูรณ์
เปิดแท็บ Actions > Build LAB FFN Thai package > เลือก run ที่สำเร็จ > Artifacts > LAB_FFN_Final_TH
ใน artifact มี ZIP รวมโค้ด ข้อมูล โมเดลที่ฝึกใหม่ ผลดิบ รายงาน Word และไฟล์สำหรับตรวจรูปแบบหน้า
Download ZIP จากปุ่ม Code ของ repository จะมีเพียง source; โมเดล/รายงานที่ CI สร้างอยู่ใน Actions artifact
หาก Actions ไม่ทำงาน ให้กด Run workflow หรือใช้คำสั่งด้านล่าง

## รันบนเครื่องหรือ Colab
ใช้ Python 3.12; รันคำสั่งจาก root ของ repository หลังดาวน์โหลดทุกไฟล์

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r LAB_FFN_Final/requirements.txt
python LAB_FFN_Final/train.py --task all
python LAB_FFN_Final/build_report.py
python LAB_FFN_Final/verify_and_package.py
```

หรือรันแต่ละงาน:
```bash
python LAB_FFN_Final/FFN_classification_improved.py
python LAB_FFN_Final/FFN_regression_improved.py
```

รายงานใช้ฟอนต์ Garuda (Linux: fonts-thai-tlwg) หากไม่มีฟอนต์นี้รูปแบบหน้าอาจเปลี่ยน
การสร้าง Word บนเครื่องอย่างเดียวไม่ได้ยืนยันว่าได้ 2 หน้า; เปิดตรวจใน Word/LibreOffice ก่อนส่ง
workflow แปลง PDF และตรวจจำนวนหน้า 1–2 หน้า พร้อมสร้างภาพหน้ารายงานสำหรับตรวจด้วยสายตา

## ทำนายข้อมูลใหม่
```bash
python LAB_FFN_Final/predict.py --model LAB_FFN_Final/outputs/classification_models.pt --csv SES2023.csv --output poverty_predictions.csv
python LAB_FFN_Final/predict.py --model LAB_FFN_Final/outputs/regression_models.pt --csv SES2023.csv --output consumption_predictions.csv
```
input ต้องมีชื่อ predictors 32 ตัวครบ; target ไม่จำเป็นต่อการทำนาย
checkpoint เก็บ feature order, median imputation, scaler, target scaling, threshold และ weights ครบ

## โมเดลที่เลือก
- FFN 32–64–32–1, ReLU, Adam lr=0.001, batch=512, สูงสุด 50 epochs, early-stopping patience=12
- Classification: Weighted BCE pos_weight=5, เลือก epoch จาก PR-AUC, threshold จาก calibration F1
- Regression: SmoothL1Loss beta=1 บน standardized y, เลือก epoch จาก RMSE ในหน่วยเดิม
- ทั้งสองงานเฉลี่ยสาม seeds 0,1,2; ไม่มี dropout หรือ batch normalization
- Split: test เดิม 20%, train ประมาณ60%, calibration10%, assessment10%; 10% หลังสงวนในการรันค่าที่เลือกแล้ว
- Baseline รันจากสำเนาต้นฉบับ: เปลี่ยนเฉพาะ path, pandas map และปิด plot; คง global-median imputation เดิมพร้อมเปิดเผยข้อจำกัด
- Bootstrap/paired randomization 2,000 ครั้ง; Bonferroni สำหรับสอง primary tasks; สมมติ independent rows ไม่ใช่ survey-aware inference

## ปรับเพื่อใช้งานจริง
ไฟล์ calibration_thresholds.csv เปรียบเทียบเกณฑ์ 0.01–0.99 ทีละ 0.01 เฉพาะ calibration
เลือกตามเป้าหมาย Recall/Precision และจำนวนรายที่ตรวจสอบต่อได้ ไม่ดู test เพื่อเลือก threshold
การเปลี่ยน threshold ต้องบันทึกใน checkpoint ที่ใช้ deployment หลังผ่านการตรวจสอบ
หากปรับ pos_weight/loss/architecture ต้องฝึกใหม่และเลือก threshold ใหม่
Regression ไม่ควรนำ MAE หรือ RMSE ไปบวก/ลบ predictions; ต้องประเมิน bias จากข้อมูล calibration ใหม่ก่อน
โมเดลนี้ยังไม่ใช่ระบบตัดสินสิทธิสวัสดิการอัตโนมัติ

## ไฟล์
- originals/: สำเนา lab แบบ byte-for-byte
- train.py / predict.py: ฝึก ประเมิน และทำนาย
- outputs/*_models.pt: weights ที่สร้างใหม่
- outputs/*_predictions.csv, *_history.json, *_splits.npz: หลักฐานการรัน
- outputs/classification.json และ regression.json: metrics และการทดสอบแบบจับคู่
- outputs/LAB_FFN_Report_TH.docx: รายงานจากผลรันใหม่
- historical_results.json: บันทึกผลก่อนหน้า ไม่ปะปนกับผลใหม่
