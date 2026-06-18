# CIFAR-10 CNN EAUT Experiment

Project này dùng để chạy lại thực nghiệm bài báo **so sánh hiệu năng và chi phí tính toán của các mô hình CNN nhẹ trên CIFAR-10** theo yêu cầu phản biện:

- Chạy đồng đều **5 seed** cho mọi mô hình.
- Có khả năng **resume** khi Colab bị ngắt.
- Cấu hình thí nghiệm tập trung trong file `configs/*.yaml`.
- Lưu checkpoint, lịch sử huấn luyện, kết quả từng lần chạy.
- Tự động tổng hợp `mean ± std` cho Accuracy, F1-macro, Params, FLOPs, latency CPU.
- Có notebook Colab để chạy trực tiếp từ GitHub.

## 1. Cấu trúc project

```text
cifar10-cnn-eaut/
├── configs/
│   ├── cifar10_5seeds.yaml              # cấu hình chính để chạy bài báo
│   ├── cifar10_5seeds_colab_drive.yaml # cấu hình lưu checkpoint/kết quả vào Google Drive
│   └── debug.yaml                       # cấu hình test nhanh 1-2 epoch
├── docs/
│   └── reviewer_checklist.md    # checklist chỉnh sửa theo phản biện
├── notebooks/
│   └── CIFAR10_EAUT_Colab.ipynb # notebook chạy trên Google Colab
├── src/
│   ├── data.py                  # tải CIFAR-10, transform, split train/val
│   ├── engine.py                # train/evaluate loop
│   ├── metrics.py               # accuracy, macro-F1
│   ├── models.py                # SimpleCNN, VGG, ResNet18, MobileNetV2, EfficientNet-B0
│   ├── measure.py               # params, FLOPs, latency CPU
│   └── utils.py                 # seed, yaml, json, checkpoint helpers
├── train.py                     # chạy 1 mô hình + 1 seed
├── run_experiments.py           # chạy toàn bộ model × seed
├── aggregate_results.py         # tổng hợp mean ± std
├── requirements.txt
└── .gitignore
```

## 2. Chạy nhanh trên máy local hoặc Colab

```bash
pip install -r requirements.txt
python run_experiments.py --config configs/debug.yaml
python aggregate_results.py --config configs/debug.yaml
```

## 3. Chạy lại thực nghiệm chính 5 seed

```bash
python run_experiments.py --config configs/cifar10_5seeds.yaml
python aggregate_results.py --config configs/cifar10_5seeds.yaml
```

Nếu chạy trên Colab và muốn resume bền hơn sau khi runtime bị reset, mount Google Drive và dùng:

```bash
python run_experiments.py --config configs/cifar10_5seeds_colab_drive.yaml
python aggregate_results.py --config configs/cifar10_5seeds_colab_drive.yaml
```

Kết quả sẽ nằm trong:

```text
outputs/cifar10_5seeds/
├── SimpleCNN/seed_42/
│   ├── checkpoint_last.pt
│   ├── checkpoint_best.pt
│   ├── history.csv
│   └── run_summary.json
├── ...
└── aggregate/
    ├── all_runs.csv
    ├── summary_mean_std.csv
    └── summary_markdown.md
```

## 4. Resume khi Colab bị ngắt

Chỉ cần chạy lại đúng lệnh cũ:

```bash
python run_experiments.py --config configs/cifar10_5seeds.yaml
```

Nếu `resume: true` trong config, chương trình sẽ tự đọc `checkpoint_last.pt` và chạy tiếp từ epoch gần nhất.

## 5. Chạy một mô hình, một seed riêng lẻ

```bash
python train.py --config configs/cifar10_5seeds.yaml --model SimpleCNN --seed 42
```

Có thể tắt resume:

```bash
python train.py --config configs/cifar10_5seeds.yaml --model SimpleCNN --seed 42 --no-resume
```

## 6. Đưa project lên GitHub

Sau khi giải nén project, mở terminal tại thư mục project:

```bash
git init
git add .
git commit -m "Initial CIFAR-10 CNN experiment project"
git branch -M main
git remote add origin https://github.com/<your-username>/cifar10-cnn-eaut.git
git push -u origin main
```

Sau đó mở notebook:

```text
notebooks/CIFAR10_EAUT_Colab.ipynb
```

Trong notebook, sửa dòng:

```python
REPO_URL = "https://github.com/<your-username>/cifar10-cnn-eaut.git"
```

thành URL GitHub thật của anh.

## 7. Gợi ý ghi vào bài báo

Sau khi chạy xong, dùng các file sau:

- `summary_mean_std.csv`: bảng số liệu chính cho bài báo.
- `summary_markdown.md`: bảng đọc nhanh để copy sang Word.
- `all_runs.csv`: kết quả từng seed, dùng khi phản biện yêu cầu kiểm chứng.
- `run_summary.json`: thông tin chi tiết từng mô hình từng seed.

## 8. Lưu ý thực nghiệm

- Nên dùng cùng một runtime GPU cho toàn bộ thí nghiệm nếu có thể.
- Độ trễ suy luận được đo trên CPU, batch size = 1, sau warm-up.
- FLOPs là ước lượng lý thuyết trên đầu vào `(1, 3, 32, 32)`; latency là đo thực tế nên có thể không tỉ lệ thuận với FLOPs.
- MobileNetV2/EfficientNet-B0 đã được chỉnh `stem stride = 1` để phù hợp hơn với ảnh CIFAR-10 kích thước `32×32`.
