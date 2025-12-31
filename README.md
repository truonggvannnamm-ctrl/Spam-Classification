# PHÂN LOẠI THƯ RÁC (SPAM) BẰNG MACHINE LEARNING

## 1. GIỚI THIỆU ĐỀ TÀI
Dự án tập trung vào việc xây dựng mô hình **Machine Learning** để phân loại email thành **Spam** hoặc **Ham** (không spam). Việc phát hiện sớm và chính xác các email spam giúp:
- Giảm rủi ro từ các chiêu lừa đảo và quảng cáo không mong muốn.
- Nâng cao trải nghiệm người dùng khi hộp thư ít bị spam.
- Hỗ trợ các hệ thống email tự động sàng lọc email hiệu quả.
**Mục tiêu chính:**
- Xây dựng mô hình phân loại nhị phân email Spam/Ham.
- Ưu tiên **Recall** để hạn chế bỏ sót email spam.
  
---

## 2. CHI TIẾT DỮ LIỆU (DATASET)

Bộ dữ liệu sử dụng trong dự án gồm **9.990 email**.  

### Các cột dữ liệu
- **label (int64):** Nhãn email, 0 = Ham, 1 = Spam  
- **text (object):** Nội dung văn bản của email  

### Tỷ lệ nhãn
Bộ dữ liệu ở trạng thái gần **cân bằng**:  
- 53% email không phải rác (Ham)  
- 47% email rác (Spam)

### Nguồn dữ liệu
- Nguồn dữ liệu: https://www.kaggle.com/datasets/zubairmustafa/spam-and-ham-classification-balanced-dataset

---

## 3. QUY TRÌNH THỰC HIỆN (PIPELINE)

### 3.1. Chuẩn bị dữ liệu
- Đọc file dữ liệu: `data/spam_and_ham_classification.csv`  
- Target: **label**  
- Làm sạch dữ liệu:
  - Loại bỏ giá trị thiếu hoặc không hợp lệ  
  - Tiền xử lý văn bản:
    - Loại bỏ ký tự đặc biệt và số
    - Chuyển về chữ thường
    - Loại bỏ stopwords
    - Lemmatization / stemming (tùy chọn)

### 3.2. Chia train/val/test
- Chia dữ liệu:
  - 70% train  
  - 15% validation  
  - 15% test  
- Sử dụng `train_test_split` với `stratify=y` để giữ tỷ lệ nhãn cân bằng giữa các tập

### 3.3. Tiền xử lý văn bản (TF-IDF)
- Vector hóa email bằng `TfidfVectorizer`:
  - `ngram_range=(1,2)` (unigram + bigram)  
  - `min_df=2`, `max_df=0.8`  
  - Loại bỏ stopwords tiếng Anh  
  - `sublinear_tf=True`  

### 3.4. Huấn luyện mô hình
Trong dự án này, chúng tôi sử dụng 3 mô hình chính để phân loại email Spam/Ham:

- Naive Bayes (MultinomialNB)
  - **Lý do chọn:**  
    - Phù hợp với dữ liệu văn bản, đặc biệt khi sử dụng TF-IDF hoặc Bag-of-Words.  
    - Dựa trên xác suất điều kiện của từng từ, mô hình dễ huấn luyện và rất nhanh trên tập dữ liệu lớn.  
  - **Ưu điểm:**  
    - Xử lý tốt các đặc trưng rời rạc (discrete features) như TF-IDF.  
    - Khả năng generalize tốt với các từ phổ biến và ít phổ biến.  
  - **Ứng dụng:**  
    - Dùng để đánh giá nhanh các mẫu email spam/ham, đặc biệt khi cần dự đoán trên số lượng lớn email.

- Logistic Regression (LR)
  - **Lý do chọn:**  
    - Là mô hình tuyến tính, dễ giải thích và triển khai.  
    - Hỗ trợ `class_weight` nếu dữ liệu mất cân bằng (tuy dataset của bạn cân bằng gần như 50/50).  
  - **Ưu điểm:**  
    - Hoạt động tốt trên các dữ liệu TF-IDF nhiều chiều.  
    - Có khả năng ra xác suất dự đoán, thuận tiện cho việc thresholding hoặc đánh giá độ tin cậy của dự đoán.  
  - **Ứng dụng:**  
    - Cung cấp một baseline ổn định và dễ so sánh với các mô hình khác.  

- Linear SVM (LinearSVC)
  - **Lý do chọn:**
    - Thường đạt hiệu suất tốt trên dữ liệu văn bản nhiều chiều, đặc biệt với TF-IDF vectorization.  
    - Tối ưu cho bài toán phân loại nhị phân với số lượng đặc trưng lớn.  
  - **Ưu điểm:**  
    - Tăng cường khả năng **Recall**, rất quan trọng để hạn chế bỏ sót email spam.  
    - Khả năng phân loại tuyến tính mạnh mẽ, ổn định với dữ liệu lớn.  
  - **Ứng dụng:**  
    - Dùng làm mô hình chính để sàng lọc email spam do khả năng phát hiện hầu hết các email spam mà vẫn giữ lỗi nhầm Ham thấp.
, phù hợp mục tiêu hạn chế bỏ sót email spam.

### 3.5. Đánh giá mô hình
- Metrics: **Accuracy, Precision, Recall, F1-score**  
- Ma trận nhầm lẫn (Confusion Matrix)  

### 3.6. Demo
- Dự đoán email mới bằng `demo/app.py`  
- Trả nhãn **Spam/Ham** kèm xác suất dự đoán cho từng email

---

## 4. CẤU TRÚC THƯ MỤC

Cấu trúc thư mục dự án:

<img width="682" height="340" alt="image" src="https://github.com/user-attachments/assets/b9e7670e-303d-43ce-aee8-e8e155c84131" />

---
## 5.HƯỚNG DẪN CÀI ĐẶT VÀ THỨ TỰ THỰC THI
1. Huấn luyện mô hình và đánh giá

Mở notebook huấn luyện:

jupyter notebook app/Class_Classification.ipynb


Các bước thực hiện trong notebook:

Đọc dữ liệu:

  File: data/spam_and_ham_classification.csv

Tiền xử lý văn bản:

- Loại bỏ ký tự đặc biệt

- Loại bỏ stopwords

- Chuyển đổi sang TF-IDF

Huấn luyện 3 mô hình chính:

- Naive Bayes (MultinomialNB)

- Logistic Regression (LR)

- Linear SVM (LinearSVC)

Đánh giá và xuất báo cáo:

- Kết quả được lưu vào thư mục reports/

2. Demo / Inference

Chạy ứng dụng demo:

python app/app.py

**Chú ý: Đảm bảo các mô hình đã được huấn luyện và lưu trước khi chạy app.**
---
## 6. PHÂN TÍCH KẾT QUẢ

- **Hiệu suất mô hình:**  
  - Mô hình **Linear SVM** thường được chọn là tối ưu do đạt Recall cao, phù hợp với mục tiêu hạn chế bỏ sót email spam.  
  - Các mô hình khác như **Logistic Regression** và **Naive Bayes** cũng cho kết quả tốt, nhưng Linear SVM có sự cân bằng tốt giữa Recall và Accuracy.

- **Tầm quan trọng của Recall:**  
  - Trong phân loại email, bỏ sót một email spam (False Negative) gây phiền toái và rủi ro cho người dùng hơn việc nhầm email Ham thành Spam (False Positive).  
  - Vì vậy, chỉ số **Recall** là ưu tiên hàng đầu để đảm bảo hệ thống phát hiện spam hiệu quả.

- **Precision và F1-score:**  
  - Precision giúp hạn chế cảnh báo nhầm email Ham.  
  - F1-score cân bằng giữa Precision và Recall; tuy F1-score có thể thấp do đặc tính văn bản và độ khó của dữ liệu, mô hình vẫn đảm bảo khả năng sàng lọc email spam hiệu quả.

- **Ma trận nhầm lẫn:**  
  - Giúp trực quan hóa số lượng email dự đoán đúng và nhầm giữa nhãn Ham/Spam.  
  - Sử dụng ma trận này để đánh giá chi tiết hơn từng mô hình trên tập validation/test.

- **Kết luận chung:**  
  - Mô hình Linear SVM kết hợp với tiền xử lý TF-IDF là giải pháp hiệu quả cho bài toán phân loại thư rác.  
  - Mức Recall cao đảm bảo rằng hầu hết email spam được phát hiện, giảm rủi ro bỏ sót spam.
    
Bảng kết quả Train/Val/Test

<img width="779" height="218" alt="image" src="https://github.com/user-attachments/assets/3e6bd7e2-df11-4409-8851-38a5e5de177a" />

 CONFUSION MATRIX
 
 <img width="1473" height="989" alt="image" src="https://github.com/user-attachments/assets/a04f5c72-c7ed-4737-ba93-15990ca8192e" />

---
## Sinh viên thực hiện
- Họ và tên: Trần Đình Mạnh
- Mã sinh viên: 12423022
- Lớp: 124231

- Họ và tên: Trương Văn Nam
- Mã sinh viên: 12423025
- Lớp: 124231
