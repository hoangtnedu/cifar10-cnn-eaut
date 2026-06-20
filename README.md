# CIFAR-10 CNN EAUT Experiment

Project này dùng để chạy lại thực nghiệm cho bài báo **so sánh hiệu năng và chi phí tính toán của một số mô hình mạng nơ-ron tích chập nhẹ trên tập dữ liệu CIFAR-10**.

Mục tiêu chính của project:

- Huấn luyện và so sánh 5 mô hình CNN trên CIFAR-10.
- Chạy lặp lại nhiều seed để báo cáo kết quả ổn định hơn.
- Cho phép resume khi Google Colab bị ngắt runtime.
- Lưu checkpoint, lịch sử huấn luyện và kết quả từng lần chạy.
- Tổng hợp tự động `mean ± std` cho Accuracy, F1-macro, Params, FLOPs và latency CPU.
- Có notebook Colab để chạy trực tiếp từ GitHub.

---

## 1. Mô hình được so sánh

Project hiện hỗ trợ 5 mô hình:

| Tên mô hình trong code | Vai trò |
|---|---|
| `SimpleCNN` | Baseline CNN nhẹ |
| `VGG11TinyBN` | Phiên bản VGG-11 nhỏ có BatchNorm |
| `ResNet18CIFAR` | ResNet-18 chỉnh cho ảnh CIFAR-10 |
| `MobileNetV2CIFAR` | MobileNetV2 chỉnh cho ảnh 32×32 |
| `EfficientNetB0CIFAR` | EfficientNet-B0 chỉnh cho CIFAR-10 |

Các seed mặc định:

```text
42, 43, 44, 45, 46
```

---

## 2. Cấu trúc project

```text
cifar10-cnn-eaut/
├── configs/
│   ├── cifar10_5seeds.yaml              # cấu hình chính chạy local
│   ├── cifar10_5seeds_colab_drive.yaml  # cấu hình lưu kết quả vào Google Drive
│   └── debug.yaml                       # cấu hình test nhanh 1-2 epoch
├── docs/
│   └── reviewer_checklist.md            # checklist chỉnh sửa theo phản biện
├── notebooks/
│   └── CIFAR10_EAUT_Colab.ipynb         # notebook chạy trên Google Colab
├── src/
│   ├── data.py                          # tải CIFAR-10, transform, train/val split
│   ├── engine.py                        # train/evaluate loop
│   ├── metrics.py                       # accuracy, macro-F1
│   ├── models.py                        # định nghĩa các mô hình CNN
│   ├── measure.py                       # đo params, FLOPs, latency CPU
│   └── utils.py                         # seed, yaml, json, checkpoint helpers
├── train.py                             # chạy 1 mô hình + 1 seed
├── run_experiments.py                   # chạy nhiều model × nhiều seed
├── aggregate_results.py                 # tổng hợp mean ± std
├── requirements.txt
└── README.md
```

---

## 3. Cài đặt nhanh

### 3.1. Chạy trên máy local

```bash
git clone https://github.com/hoangtnedu/cifar10-cnn-eaut.git
cd cifar10-cnn-eaut
pip install -r requirements.txt
```

Test nhanh project:

```bash
python run_experiments.py --config configs/debug.yaml
python aggregate_results.py --config configs/debug.yaml
```

### 3.2. Chạy trên Google Colab

Mở notebook:

```text
notebooks/CIFAR10_EAUT_Colab.ipynb
```

Hoặc chạy thủ công trong Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
git clone https://github.com/hoangtnedu/cifar10-cnn-eaut.git
cd cifar10-cnn-eaut
pip install -r requirements.txt
```

---

## 4. Chạy thực nghiệm chính 5 seed

### 4.1. Chạy local

```bash
python run_experiments.py --config configs/cifar10_5seeds.yaml
python aggregate_results.py --config configs/cifar10_5seeds.yaml
```

### 4.2. Chạy Colab và lưu vào Google Drive

Khuyến nghị dùng cấu hình này khi chạy lâu trên Colab:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml
python aggregate_results.py --config configs/cifar10_5seeds_colab_drive.yaml
```

Với cấu hình Colab Drive, kết quả nằm tại:

```text
/content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds/
```

---

## 5. Resume khi Colab bị ngắt

Project đã bật `resume: true` trong config. Nếu Colab bị ngắt, chỉ cần chạy lại đúng lệnh cũ:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml
```

Chương trình sẽ tự đọc file:

```text
checkpoint_last.pt
```

và chạy tiếp từ epoch gần nhất.

Ví dụ đường dẫn checkpoint:

```text
/content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds/SimpleCNN/seed_46/checkpoint_last.pt
```

Nếu một run đã đủ 100 epoch, chương trình sẽ báo dạng:

```text
Run already completed: epoch 100/100
```

Khi đó seed/model đó đã chạy xong, không cần train lại.

---

## 6. Chạy một mô hình hoặc một seed cụ thể

### 6.1. Chạy một mô hình, một seed

```bash
python train.py --config configs/cifar10_5seeds_colab_drive.yaml --model SimpleCNN --seed 42
```

### 6.2. Chạy một mô hình với nhiều seed

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models SimpleCNN --seeds 42 43 44 45 46
```

### 6.3. Chạy nhiều mô hình còn lại

Ví dụ nếu đã chạy xong `SimpleCNN` và `VGG11TinyBN`, chỉ chạy tiếp 3 mô hình còn lại:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models ResNet18CIFAR MobileNetV2CIFAR EfficientNetB0CIFAR
```

### 6.4. Chạy tiếp riêng seed còn thiếu

Ví dụ chỉ chạy seed 46 cho ResNet18-CIFAR:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models ResNet18CIFAR --seeds 46
```

### 6.5. Tắt resume và chạy lại từ đầu

Chỉ dùng khi thật sự muốn huấn luyện lại run đó:

```bash
python train.py --config configs/cifar10_5seeds_colab_drive.yaml --model SimpleCNN --seed 42 --no-resume
```

---

## 7. Chạy trên nhiều tài khoản Google Colab

Có thể chạy project trên nhiều tài khoản Google khác nhau để tiết kiệm thời gian, nhưng cần chú ý:

1. Nếu mỗi tài khoản dùng Google Drive riêng, kết quả **không tự kế thừa** nhau.
2. Muốn resume/kế thừa kết quả của tài khoản khác, các tài khoản phải truy cập được cùng một thư mục output.
3. Cách an toàn nhất là dùng chung một thư mục Drive/Shared Drive, hoặc copy toàn bộ thư mục kết quả về đúng cấu trúc.
4. Không đổi tên thư mục model hoặc seed, vì chương trình resume theo cấu trúc đường dẫn.

Cấu trúc cần giữ nguyên:

```text
cifar10_eaut_outputs/
└── cifar10_5seeds/
    ├── SimpleCNN/
    │   ├── seed_42/
    │   ├── seed_43/
    │   └── ...
    ├── VGG11TinyBN/
    ├── ResNet18CIFAR/
    ├── MobileNetV2CIFAR/
    └── EfficientNetB0CIFAR/
```

Nếu tài khoản B không thấy thư mục Drive được chia sẻ trong Colab, có thể copy thư mục kết quả sang `MyDrive` của tài khoản B rồi chạy tiếp bằng cùng cấu hình Colab Drive.

---

## 8. Kết quả đầu ra

Mỗi run tạo ra thư mục riêng theo model và seed:

```text
/content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds/
└── <ModelName>/
    └── seed_<seed>/
        ├── checkpoint_last.pt
        ├── checkpoint_best.pt
        ├── history.csv
        └── run_summary.json
```

Sau khi tổng hợp, thư mục `aggregate/` sẽ có:

```text
aggregate/
├── all_runs.csv
├── summary_mean_std.csv
└── summary_markdown.md
```

Ý nghĩa các file chính:

| File | Ý nghĩa |
|---|---|
| `checkpoint_last.pt` | checkpoint epoch mới nhất, dùng để resume |
| `checkpoint_best.pt` | checkpoint tốt nhất theo validation accuracy |
| `history.csv` | lịch sử train/validation theo từng epoch |
| `run_summary.json` | kết quả test và chi phí tính toán của một run |
| `all_runs.csv` | toàn bộ kết quả từng model, từng seed |
| `summary_mean_std.csv` | bảng tổng hợp mean ± std cho bài báo |
| `summary_markdown.md` | bảng Markdown dễ copy sang Word/GitHub |

---

## 9. Kiểm tra một seed đã chạy xong chưa

Một seed được xem là đã chạy xong khi trong thư mục seed có đủ:

```text
checkpoint_last.pt
checkpoint_best.pt
history.csv
run_summary.json
```

và khi chạy lại, log báo:

```text
Run already completed: epoch 100/100
```

Có thể kiểm tra nhanh bằng lệnh:

```bash
ls /content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds/<ModelName>/seed_<seed>/
```

Ví dụ:

```bash
ls /content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds/SimpleCNN/seed_46/
```

---

## 10. Tổng hợp kết quả cho bài báo

Sau khi các model/seed cần thiết đã hoàn thành, chạy:

```bash
python aggregate_results.py --config configs/cifar10_5seeds_colab_drive.yaml
```

Sau đó dùng các file:

- `summary_mean_std.csv`: bảng số liệu chính cho bài báo.
- `summary_markdown.md`: bảng đọc nhanh để copy sang Word.
- `all_runs.csv`: kết quả từng seed, dùng khi phản biện yêu cầu kiểm chứng.
- `run_summary.json`: thông tin chi tiết từng mô hình từng seed.

---

## 11. Gợi ý quy trình chạy thực tế trên Colab

### Trường hợp 1: Chạy toàn bộ từ đầu

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml
```

### Trường hợp 2: Đã chạy xong 2 mô hình, chạy tiếp 3 mô hình còn lại

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models ResNet18CIFAR MobileNetV2CIFAR EfficientNetB0CIFAR
```

### Trường hợp 3: Chỉ chạy lại một seed bị thiếu

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models EfficientNetB0CIFAR --seeds 46
```

### Trường hợp 4: Sau khi chạy xong, tổng hợp bảng kết quả

```bash
python aggregate_results.py --config configs/cifar10_5seeds_colab_drive.yaml
```

---

## 12. Lưu ý thực nghiệm

- Nên dùng GPU cho quá trình huấn luyện.
- Nên mount Google Drive trước khi chạy nếu muốn resume bền vững.
- Không nên đổi `output_dir` giữa các lần chạy nếu muốn chương trình nhận lại checkpoint cũ.
- Độ trễ suy luận được đo trên CPU, batch size = 1, sau warm-up.
- FLOPs là ước lượng lý thuyết trên đầu vào `(1, 3, 32, 32)`.
- Latency là đo thực tế nên có thể không tỉ lệ thuận hoàn toàn với FLOPs.
- `MobileNetV2CIFAR` và `EfficientNetB0CIFAR` đã được chỉnh `stem stride = 1` để phù hợp hơn với ảnh CIFAR-10 kích thước `32×32`.

---

## 13. Lỗi thường gặp

### 13.1. Colab bị ngắt giữa chừng

Chạy lại đúng lệnh cũ. Không thêm `--no-resume`.

### 13.2. Không thấy kết quả cũ

Kiểm tra lại đang dùng đúng config Drive:

```bash
configs/cifar10_5seeds_colab_drive.yaml
```

và đúng thư mục:

```text
/content/drive/MyDrive/cifar10_eaut_outputs/cifar10_5seeds/
```

### 13.3. Muốn chạy tiếp model chưa xong

Dùng tham số `--models` để chỉ định model cần chạy:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models MobileNetV2CIFAR EfficientNetB0CIFAR
```

### 13.4. Muốn chạy tiếp seed chưa xong

Dùng tham số `--seeds`:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml --models ResNet18CIFAR --seeds 45 46
```

---

## 14. License

Project sử dụng MIT License. Có thể dùng, chỉnh sửa và chia sẻ lại mã nguồn theo điều kiện của giấy phép MIT.
