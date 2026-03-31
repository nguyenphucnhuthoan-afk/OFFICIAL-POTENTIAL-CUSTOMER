from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
import sqlite3
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

# Load model, scaler, features
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler_customer_data.joblib')
with open('features_list.json', 'r') as f:
    features = json.load(f)

def preprocess_input(input_data: Dict[str, Any]) -> np.ndarray:
    df = pd.DataFrame([input_data])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features]
    X_scaled = scaler.transform(df)
    return X_scaled

def predict_potential_customer(input_data: Dict[str, Any]) -> Dict[str, Any]:
    X = preprocess_input(input_data)
    proba = model.predict_proba(X)[0,1]
    pred = int(proba >= 0.5)
    return {
        'prediction': pred,
        'probability': float(proba)
    }

# FastAPI app
app = FastAPI() #khởi tạo web server cho model ML để triển khai API dự đoán khách hàng tiềm năng, cho phép người dùng gửi dữ liệu đầu vào và nhận kết quả dự đoán cùng với xác suất.

# SQLite setup # Thiết lập cơ sở dữ liệu SQLite để lưu trữ lịch sử dự đoán, bao gồm các cột như id, input, output và timestamp, giúp theo dõi và phân tích các dự đoán đã thực hiện.
conn = sqlite3.connect('prediction_history.db', check_same_thread=False) #Tạo database lưu lịch sử dự đoán.
c = conn.cursor() #con trỏ để gửi lệnh SQL vào database
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input TEXT,
    output TEXT,
    timestamp TEXT
)''')#Nếu bảng chưa tồn tại → tạo bảng.
conn.commit() #Lưu thay đổi vào database

class PredictRequest(BaseModel): # Xác thực dữ liệu gửi vào từ user, Nếu user gửi sai format → FastAPI sẽ báo lỗi ngay.
    input_data: Dict[str, Any]

@app.post('/api/predict') #API dự đoán model
def api_predict(req: PredictRequest): #FastAPI sẽ tự động chuyển JSON thành:
    result = predict_potential_customer(req.input_data)
    # Save to DB
    c.execute('INSERT INTO predictions (input, output, timestamp) VALUES (?, ?, ?)',
              (json.dumps(req.input_data), json.dumps(result), datetime.now().isoformat()))
    conn.commit()
    return result

@app.get('/api/history') #API xem lịch sử dự đoán
def api_history():
    c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 100')
    rows = c.fetchall()
    return [
        {'id': row[0], 'input': json.loads(row[1]), 'output': json.loads(row[2]), 'timestamp': row[3]}
        for row in rows
    ]
