# K-Nearest-Neighbor (KNN) from scratch

KNN สำหรับข้อมูล 2 มิติ เขียนเองด้วย NumPy ล้วน (ไม่ใช้ scikit-learn) มี 2 แบบ

| แบบ | ไฟล์ | เหมาะกับ |
|---|---|---|
| Notebook | `KNN.ipynb` | อ่านทีละขั้น: ระยะทาง → k เพื่อนบ้าน → majority vote บนข้อมูลบัญชีสตรีมเมอร์ 19 บัญชี |
| CLI | `main.py` + `config_dev.yaml` + `Process/` | รันจาก config, ลองค่า k / metric / scaling, บันทึกกราฟ |

## ติดตั้ง

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # ถ้าจะรันเทสต์
```

## รัน CLI

```bash
python main.py                                   # ใช้ config_dev.yaml
python main.py --k 5 --point 3 4                 # เปลี่ยน k และจุดทดสอบ
python main.py --metric manhattan                # ระยะแบบ Manhattan
python main.py --tie-break nearest               # คะแนนเสมอ -> เลือกคลาสที่มีเพื่อนบ้านใกล้สุด
python main.py --standardize --no-plot           # z-score feature ก่อนคำนวณระยะ
python main.py --tune-k                          # เลือก k ด้วย leave-one-out CV
python main.py --config my_config.yaml
```

ตัวอย่างผลลัพธ์

```
9 training points, test point = (4.0, 4.0), k = 3, metric = euclidean, standardize = False
nearest neighbors:
  1. point  2 (3, 2.5)  Class A    d=1.803
  2. point  3 (6, 6)  Class B    d=2.828
  3. point  1 (2, 1.5)  Class A    d=3.202
votes: {'Class A': 2, 'Class B': 1}
prediction: 0 -> Class A
```

กราฟบันทึกที่ `output/`: `data.png`, `neighbors.png` (เส้นไปหา k เพื่อนบ้าน + รัศมี), `decision_boundary.png`

## Config (`config_dev.yaml`)

| key | ความหมาย | ค่าเริ่มต้น |
|---|---|---|
| `data.x`, `data.y`, `data.label` | พิกัดและคลาส (label เริ่มที่ 0) | จำเป็น |
| `data.label_names`, `data.colors` | ชื่อคลาส / สีในกราฟ | ตัวเลข / สีมาตรฐาน |
| `data.xlabel`, `data.ylabel` | ชื่อแกน | `x`, `y` |
| `test_point` | `[x, y]` ที่จะทำนาย | จำเป็น |
| `k` | จำนวนเพื่อนบ้าน (1 ถึงจำนวนจุด) | จำเป็น |
| `metric` | `euclidean` \| `manhattan` | `euclidean` |
| `tie_break` | `smallest` (เหมือน notebook) \| `nearest` | `smallest` |
| `standardize` | z-score ก่อนคำนวณระยะ | `false` |
| `output_dir` | โฟลเดอร์เก็บกราฟ | `output` |

## โครงสร้าง

```
KNN.ipynb              walkthrough แบบ notebook
main.py                CLI
config_dev.yaml        ชุดข้อมูลตัวอย่าง 3 คลาส x 3 จุด
Process/Data.py        class Data + load_config (ตรวจ key / ความยาว / ช่วงของ k)
Process/Calculate.py   distance, k-nearest, majority vote, standardize, LOOCV
Process/Plot.py        scatter, neighbors, decision boundary
tests/                 pytest (Calculate, Data, CLI)
```

## เทสต์

```bash
python -m pytest -q
```

GitHub Actions รันเทสต์และ `main.py --no-plot` ทุกครั้งที่ push หรือเปิด PR

## ข้อจำกัด

- รองรับแค่ 2 feature (`x`, `y`) ตามรูปแบบของ notebook
- เมื่อเปิด `--standardize` จะไม่วาดกราฟ (กราฟใช้พิกัดดิบ)
- เป็นโค้ดเพื่อการเรียนรู้ วนลูปทำนายทีละจุด ไม่เหมาะกับข้อมูลขนาดใหญ่
