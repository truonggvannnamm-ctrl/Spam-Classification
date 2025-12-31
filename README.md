# PHÂN LOẠI THƯ RÁC (SPAM) BẰNG MACHINE LEARNING

## 1. GIỚI THIỆU ĐỀ TÀI
Dự án tập trung vào việc xây dựng mô hình **Machine Learning** để phân loại email thành **Spam** hoặc **Ham** (không spam).  
Việc phát hiện sớm và chính xác các email spam giúp giảm rủi ro từ các chiêu lừa đảo, quảng cáo không mong muốn và cải thiện trải nghiệm người dùng.

- **Mục tiêu:** Xây dựng mô hình phân loại nhị phân Spam/Ham.  
- **Định hướng:** Ưu tiên các chỉ số **Recall** để giảm bỏ sót email spam.  
- **Thách thức:** Dữ liệu gần như cân bằng nhưng vẫn cần xử lý kỹ thuật tiền xử lý và vector hóa văn bản.

---

## 2. CHI TIẾT DỮ LIỆU (DATASET)
- **Cột dữ liệu:** 2 (`text`, `label`)  
- **Mô tả cột:**
| Thuộc tính | Kiểu dữ liệu | Mô tả |
|------------|--------------|-------|
| text      | object       | Nội dung email |
| label     | int64        | Nhãn (0: Ham, 1: Spam) |
- **Nguồn dữ liệu:** (vd: local file `data/emails.csv` hoặc link dataset)  
- **Tỷ lệ nhãn:** 53% Ham, 47% Spam

---

## 3. QUY TRÌNH THỰC HIỆN (PIPELINE)
Pipeline được triển khai trong notebook: `demo/demo.ipynb`.

### 3.1. Chuẩn bị dữ liệu
- Đọc file dữ liệu: `data/emails.csv`  
- **Target:** `label`  
- Làm sạch dữ liệu:
  - Loại bỏ giá trị thiếu / không hợp lệ.  
  - Tiền xử lý văn bản: 
    - Loại bỏ ký tự đặc biệt, số, stopwords  
    - Chuyển về chữ thường  
    - Lemmatization / stemming (tùy chọn)  

### 3.2. Chia train/test
- 70% train, 15% test (`train_test_split(test_size=0.25, stratify=y, random_state=42)`)  
- Giữ tỷ lệ lớp để cân bằng nhãn.

### 3.3. Tiền xử lý văn bản (TF-IDF)
- Vector hóa email bằng `TfidfVectorizer`:
  - `ngram_range=(1,2)` (unigram + bigram)  
  - `min_df=2`, `max_df=0.8`  
  - Loại bỏ stopwords tiếng Anh  
  - `sublinear_tf=True`  

### 3.4. Huấn luyện mô hình + xử lý mất cân bằng
- 3 mô hình chính:
  - **Naive Bayes (MultinomialNB)**  
  - **Logistic Regression (LR)**  
  - **Linear SVM (LinearSVC)**
- Xử lý cân bằng nhãn:
  - `class_weight="balanced"` (LR, SVM)  
  - `RandomOverSampler` (NB, nếu cần)

### 3.5. Đánh giá mô hình
- Metrics: **Accuracy, Precision, Recall, F1-score**  
- Ma trận nhầm lẫn (Confusion Matrix)  
- Lựa chọn threshold dựa trên **Precision–Recall curve** ưu tiên Recall

### 3.6. Inference
- Dự đoán email mới bằng `demo/app.py`  
- Trả nhãn **Spam/Ham** kèm xác suất dự đoán cho từng email.

---

## 4. CẤU TRÚC THƯ MỤC


## Sinh viên thực hiện
- Họ và tên: Trần Đình Mạnh
- Mã sinh viên: 12423022
- Lớp: 124231

- Họ và tên: Trương Văn Nam
- Mã sinh viên: 12423025
- Lớp: 124231
