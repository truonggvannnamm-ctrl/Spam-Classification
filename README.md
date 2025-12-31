#  Spam Email Detection using Machine Learning

##  Giới thiệu
Đề tài xây dựng hệ thống **phân loại email Spam/Ham** sử dụng Machine Learning, bao gồm:
- Notebook để huấn luyện và phân tích mô hình
- Ứng dụng web Flask để demo dự đoán email mới

---

##  Dataset
- Dataset: SMS Spam Collection
- Nguồn: UCI Machine Learning Repository  
- Link: https://www.kaggle.com/datasets/zubairmustafa/spam-and-ham-classification-balanced-dataset

### Cấu trúc dữ liệu
| Cột | Mô tả |
|---|---|
| label | Nhãn email (`spam`, `ham`) |
| text | Nội dung email |

---

## 🔁 Pipeline xử lý
1. Tiền xử lý văn bản (lowercase, remove URL, ký tự đặc biệt)
2. Vector hóa TF-IDF (n-gram 1–2)
3. Chia train/test
4. Huấn luyện Naive Bayes
5. Đánh giá (Accuracy, Classification Report)
6. Dự đoán email mới qua web demo

---

## 🤖 Mô hình sử dụng
- **Multinomial Naive Bayes**
- Lý do: phù hợp dữ liệu văn bản, nhanh, hiệu quả cho bài toán spam

---

## 📊 Kết quả
- Đánh giá bằng Accuracy và Classification Report trên tập test
- Kết quả in trực tiếp ra terminal khi chạy `app.py`

---

## ▶️ HƯỚNG DẪN CHẠY CHƯƠNG TRÌNH

### 1️⃣ Cài đặt môi trường
Yêu cầu: Python >= 3.8

Cài các thư viện cần thiết:
```bash
pip install pandas numpy scikit-learn flask
```

---

### 2️⃣ Chạy notebook huấn luyện (tùy chọn)
File: `Untitled9.ipynb`

Mục đích:
- Phân tích dữ liệu
- Trực quan hóa
- Thử nghiệm mô hình

Cách chạy:
- Mở bằng Jupyter Notebook hoặc VS Code
- Chạy lần lượt các cell

📌 **Notebook chỉ dùng để học & phân tích, không cần để chạy demo**

---

### Chạy ứng dụng web demo (bắt buộc)
File: `app.py`

Trong thư mục chứa `app.py` và file CSV, chạy:
```bash
python app.py
```

Sau khi chạy thành công, terminal sẽ hiển thị:
```
Running on http://127.0.0.1:5000
```

Mở trình duyệt và truy cập:
```
http://127.0.0.1:5000
```

---

### Dự đoán email mới
- Nhập nội dung email vào ô textarea
- Nhấn **Dự đoán**
- Kết quả hiển thị:
  - 📛 Spam Email
  - ✅ Ham (Email hợp lệ)

---

##  Cấu trúc thư mục dự án
```
spam_email_detection/
├── app.py              # Web demo + train/test
├── Untitled9.ipynb     # Notebook huấn luyện & phân tích
├── spam_and_ham_classification.csv
└── README.md
```

---

## Sinh viên thực hiện
- Họ và tên: Trần Đình Mạnh
- Mã sinh viên: 12423022
- Lớp: 124231

- Họ và tên: Trương Văn Nam
- Mã sinh viên: 12423025
- Lớp: 124231
