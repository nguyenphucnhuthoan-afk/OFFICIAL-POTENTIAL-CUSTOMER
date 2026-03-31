import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Load model, scaler, features
model = joblib.load('best_model.pkl')  # Đảm bảo tên file đúng với mô hình đã lưu
scaler = joblib.load('scaler_customer_data.joblib') # Đảm bảo tên file đúng với scaler đã lưu
with open('features_list.json', 'r') as f: # Đảm bảo tên file đúng với file đã lưu chứa danh sách features
    features = json.load(f)  # Đảm bảo rằng features được load đúng với thứ tự và tên đã sử dụng trong quá trình huấn luyện

# Preprocessing function (must match training) # Hàm này sẽ biến đổi dữ liệu đầu vào từ người dùng thành dạng mà model đã được huấn luyện để hiểu, bao gồm việc đảm bảo đủ các feature cần thiết và chuẩn hóa dữ liệu theo cách giống như khi huấn luyện model.
def preprocess_input(input_data: Dict[str, Any]) -> np.ndarray: #Biến dữ liệu người dùng nhập → dạng model hiểu được
    # Chuyển input thành DataFrame
    df = pd.DataFrame([input_data])
    # Đảm bảo đủ các feature, fill thiếu bằng 0 hoặc giá trị mặc định
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features] #Sắp xếp đúng thứ tự feature

    # 👇 THÊM DEBUG Ở ĐÂY
    print("Expected features:", list(scaler.feature_names_in_))
    print("Input features:", df.columns.tolist())


    # Scaling chuẩn hóa dữ liệu
    X_scaled = scaler.transform(df.values)
    return X_scaled

def predict_potential_customer(input_data: Dict[str, Any]) -> Dict[str, Any]: #Biến dữ liệu người dùng nhập → dự đoán của model, Hàm dự đoán khách hàng tiềm năng
    X = preprocess_input(input_data) #Dữ liệu đã được chuẩn hóa, sẵn sàng để đưa vào model
    proba = model.predict_proba(X)[0,1] # Lấy xác suất của lớp 1 (khách hàng tiềm năng)
    return {
        'prediction': pred, #pred = int(proba >= 0.5)
        'probability': float(proba)} #probability được chuyển thành float, Đây là xác suất model tin rằng khách hàng là tiềm năng.

# Example usage
if __name__ == '__main__':
    sample = {f: 0 for f in features}
    sample['AnnualIncome'] = 50000
    sample['NumberOfPurchases'] = 10
    sample['TimeSpentOnWebsite'] = 30
    sample['CustomerSegment_LE'] = 1
    print(predict_potential_customer(sample))
