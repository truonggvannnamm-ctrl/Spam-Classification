from flask import Flask, request
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# =====================
# 1. TIỀN XỬ LÝ VĂN BẢN
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================
# 2. LOAD DATA
# =====================
print("🔄 Loading dataset...")
df = pd.read_csv("spam_and_ham_classification.csv")

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']


# =====================
# 3. TRAIN / TEST SPLIT
# =====================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================
# 4. TF-IDF VECTORIZE
# =====================
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words='english'
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)


# =====================
# 5. TRAIN MODEL
# =====================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# =====================
# 6. ĐÁNH GIÁ TRAIN / TEST
# =====================
y_train_pred = model.predict(X_train_tfidf)
y_test_pred = model.predict(X_test_tfidf)

print("\n📊 TRAIN ACCURACY:", accuracy_score(y_train, y_train_pred))
print("📊 TEST ACCURACY:", accuracy_score(y_test, y_test_pred))

print("\n📋 CLASSIFICATION REPORT (TEST):")
print(classification_report(y_test, y_test_pred))


# =====================
# 7. FLASK DEMO
# =====================
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    email_text = ""
    result_text = ""
    result_class = ""

    if request.method == "POST":
        email_text = request.form.get("email")
        email_clean = clean_text(email_text)
        X_input = tfidf.transform([email_clean])
        pred = model.predict(X_input)[0]

        if pred == "spam":
            result_text = "📛 SPAM EMAIL"
            result_class = "danger"
        else:
            result_text = "✅ HAM (Email hợp lệ)"
            result_class = "success"

    return f"""
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Spam Email Detection</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">

    <style>
        body {{
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', sans-serif;
        }}

        .card {{
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
            padding: 30px;
        }}

        textarea {{
            border-radius: 12px;
            resize: none;
        }}

        .result-box {{
            margin-top: 25px;
            padding: 15px;
            border-radius: 15px;
            font-size: 20px;
            font-weight: bold;
        }}
    </style>
</head>

<body>
<div class="container">
    <div class="card mx-auto" style="max-width: 650px;">
        <h2 class="text-center mb-3">
            <i class="bi bi-envelope-check-fill text-primary"></i>
            Spam Email Detection
        </h2>

        <form method="post">
            <textarea class="form-control mb-3" name="email" rows="6"
                placeholder="Nhập nội dung email...">{email_text}</textarea>

            <button type="submit" class="btn btn-primary w-100">
                🔍 Dự đoán
            </button>
        </form>

        {f"""
        <div class="result-box text-center bg-{'danger' if result_class=='danger' else 'success'} text-white">
            {result_text}
        </div>
        """ if result_text else ""}
    </div>
</div>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
