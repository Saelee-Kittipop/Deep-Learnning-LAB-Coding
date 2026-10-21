# Deep Learning LAB — เริ่มจากสองไฟล์นี้

สำหรับงาน FFN ปรับปรุง **เปิด Python เพียง 2 ไฟล์** ในโฟลเดอร์ `LAB_FFN_Final`:

| ต้องการทำอะไร | ไฟล์ที่เปิด |
|---|---|
| ทำนายความยากจน 0/1 | [FFN_classification_improved.py](LAB_FFN_Final/FFN_classification_improved.py) |
| ทำนายค่าใช้จ่ายบริโภคต่อหัว | [FFN_regression_improved.py](LAB_FFN_Final/FFN_regression_improved.py) |

แต่ละไฟล์มีโค้ดครบ: เตรียมข้อมูล ฝึก baseline และโมเดลปรับปรุง ประเมินผล ทดสอบผลต่าง บันทึกโมเดล แสดง/บันทึกกราฟ และทำนายข้อมูลใหม่ **ไม่ต้องเปิด `train.py` หรือ `predict.py` เพิ่ม**

อ่าน [วิธีรันและสรุปเทคนิคภาษาไทย](LAB_FFN_Final/README_TH.md) แล้วรันจาก root ของ repository:

```bash
python LAB_FFN_Final/FFN_classification_improved.py
python LAB_FFN_Final/FFN_regression_improved.py
```

ไฟล์ `FFN_classification.py` และ `FFN_regression.py` ที่ root เป็น **LAB ต้นฉบับ** สำหรับอ้างอิง ส่วน `Sequential_*`, `drop_out.py`, `early_stopping.py` เป็นตัวอย่างบทเรียนอื่น ไม่ใช่ไฟล์ที่ต้องเปิดเพื่อรัน FFN ปรับปรุง
