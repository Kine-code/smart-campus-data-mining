# Smart Campus IoT Data Mining

Bài tập lớn môn Khai phá dữ liệu: phân tích, dự đoán tiêu thụ năng lượng và phát hiện bất thường từ dữ liệu IoT Smart Campus.

## Chức năng
- Sinh bộ dữ liệu IoT mô phỏng 10 phòng, 30 ngày, chu kỳ 5 phút (~86.400 bản ghi gốc).
- Cài lỗi dữ liệu có chủ đích: missing value, duplicate và anomaly để đánh giá pipeline làm sạch.
- Tiền xử lý và feature engineering.
- EDA: điện năng theo ngày/giờ/phòng, correlation heatmap.
- K-Means + Elbow + Silhouette + PCA.
- Classification: Decision Tree, KNN, SVM, Random Forest.
- Regression: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
- Baseline dự đoán năng lượng để so sánh.
- Isolation Forest phát hiện bất thường và đánh giá trên anomaly đã cài.
- HMM/Viterbi minh họa suy luận trạng thái hoạt động phòng từ sensor observation.
- Streamlit dashboard.

## Cài đặt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy toàn bộ
```bash
python src/main.py
```

## Chạy dashboard
```bash
streamlit run dashboard/app.py
```

## Kết quả
- `data/raw/iot_raw.csv`: dữ liệu gốc có lỗi.
- `data/processed/iot_clean.csv`: dữ liệu sau xử lý.
- `results/figures/`: biểu đồ PNG.
- `results/metrics/`: bảng so sánh mô hình và summary.json.
- `models/`: mô hình tốt nhất đã lưu.
