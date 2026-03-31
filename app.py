import sqlite3 
import altair as alt #vẽ biểu đồ interactive
import streamlit as st #tạo web app ML (UI cho người dùng nhập dữ liệu)
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from typing import Dict, Any #định nghĩa kiểu dữ liệu (code chuyên nghiệp hơn)

st.set_page_config(page_title="Hệ thống Phân tích & Dự báo Khách hàng Tiềm năng 2026", layout="wide")

# 1. Load model, scaler, features, feature importance
model = joblib.load('best_model.pkl') #Load model tốt nhất bạn đã lưu sau khi train.
scaler = joblib.load('scaler_customer_data.joblib') #Chuẩn hóa dữ liệu giống lúc training
with open('features_list.json', 'r') as f:
    features = json.load(f) #Load danh sách feature
try:
    feature_imp = pd.read_csv('feature_importance.csv', index_col=0) #Load feature importance (nếu có) không thì thoi
except:
    feature_imp = None

# 2. Load customerData_50k.csv để lấy giá trị trung bình khách tiềm năng
try:
    df = pd.read_csv('customerData_50k.csv')
    df_potential = df[df['PurchaseStatus'] == 1] #Lọc khách hàng tiềm năng
except: #Load feature importance (nếu có) không thì thoi
    df = None
    df_potential = None

# 3. Sidebar - Form nhập liệu
st.sidebar.header('Nhập thông tin khách hàng')

annual_income = st.sidebar.slider('Annual Income (USD)', 15000, 200000, 50000, step=1000)
time_spent = st.sidebar.slider('Time Spent On Website (phút)', 1, 180, 30)
CustomerTenureYears = st.sidebar.slider('Customer Tenure (năm)', 1, 15, 8)
number_of_purchases = st.sidebar.slider('Number Of Purchases', 4, 40, 20)
session_count = st.sidebar.slider('Session Count', 8, 58, 30)
customer_satisfaction = st.sidebar.selectbox( "Customer Satisfaction",[1, 2, 3, 4, 5])
Discounts_Availed= st.sidebar.selectbox('Discounts Availed',['Yes', 'No'])


# Dropdown options (khớp với nhãn đã học)
#product_categories = ['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports']
#customer_segments = ['Low Value', 'Medium Value', 'High Value']
#regions = ['North', 'South', 'East', 'West']

#product_category = st.sidebar.selectbox('Product Category', product_categories)
#customer_segment = st.sidebar.selectbox('Customer Segment', customer_segments)
#region = st.sidebar.selectbox('Region', regions)
#loyalty_program = st.sidebar.selectbox('Loyalty Program', loyalty_programs)

# 4. Chuẩn bị input cho model

# Tạo DataFrame rỗng với đầy đủ các cột mà Model/Scaler yêu cầu (từ features_list.json)
X_input = pd.DataFrame(0, index=[0], columns=features)

# Điền giá trị các biến số (Numerical)
X_input['AnnualIncome'] = annual_income
X_input['TimeSpentOnWebsite'] = time_spent
X_input['CustomerSatisfaction'] = customer_satisfaction
X_input['NumberOfPurchases'] = number_of_purchases
X_input['SessionCount'] = session_count
X_input['DiscountsAvailed'] = Discounts_Availed

# Điền giá trị các biến Category (One-Hot Encoding)
# Lưu ý: Kiểm tra chính xác tên cột trong features_list.json (ví dụ: 'ProductCategory_Electronics')
#if f'ProductCategory_{product_category}' in X_input.columns:
    #X_input[f'ProductCategory_{product_category}'] = 1

#if f'Region_{region}' in X_input.columns:
    #X_input[f'Region_{region}'] = 1

#if 'LoyaltyProgram' in X_input.columns:
    #X_input['LoyaltyProgram'] = 1 if loyalty_program == 'Yes' else 0

if 'DiscountsAvailed' in X_input.columns:
    X_input['DiscountsAvailed'] = 1 if Discounts_Availed == 'Yes' else 0

# Label Encoding cho Segment
#segment_map = {'Low Value': 0, 'Medium Value': 1, 'High Value': 2}
#if 'CustomerSegment_LE' in X_input.columns:
    #X_input['CustomerSegment_LE'] = segment_map[customer_segment]

# Tính toán các biến phái sinh (Feature Engineering) khớp với lúc Train
X_input['AVG_PURCHASE_3M'] = number_of_purchases / 4
X_input['MAX_SPENT_6M'] = (annual_income / 2) * (session_count / 24)
X_input['STDDEV_SATISFACTION_12M'] = 0.5
X_input['Income_per_Session'] = annual_income / (session_count + 1)

X_input['RecencyScore'] = 1 / (X_input['LastPurchaseDaysAgo'] + 1)
X_input['Engagement'] = X_input['TimeSpentOnWebsite'] / (X_input['SessionCount'] + 1)
X_input['PurchaseRate'] = X_input['NumberOfPurchases'] / (X_input['SessionCount'] + 1)
X_input['Time_per_Purchase'] = X_input['TimeSpentOnWebsite'] / (X_input['NumberOfPurchases'] + 1)

X_input["EngagementScore"] = (X_input["SessionCount"] * 0.4 +X_input["TimeSpentOnWebsite"] * 0.3 +X_input["NumberOfPurchases"] * 0.3)
X_input["LoyaltyScore"] = (X_input["CustomerTenureYears"] * 0.5 +X_input["NumberOfPurchases"] * 0.5)
X_input["IncomePerPurchase"] = X_input["AnnualIncome"] / (X_input["NumberOfPurchases"] + 1)

# ĐẢM BẢO THỨ TỰ CỘT: Đây là bước quan trọng nhất để fix lỗi ValueError

X_input = X_input[scaler.feature_names_in_]

# Thực hiện Transform
try:
    X_scaled = scaler.transform(X_input)
except ValueError as e:
    st.error(f"Lỗi lệch cột dữ liệu: {e}")
    st.write("Cột Scaler mong đợi:", scaler.feature_names_in_)
    st.write("Cột bạn cung cấp:", X_input.columns.tolist())
    st.stop()

# 5. Main page
conn = sqlite3.connect('prediction_history.db', check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input TEXT,
    output TEXT,
    timestamp TEXT
)''')
conn.commit()
st.title('Hệ thống Phân tích & Dự báo Khách hàng Tiềm năng 2026')
# --- THÊM ĐOẠN NÀY ĐỂ SETUP DATABASE ---


if st.button('Dự đoán'):
    # 1. Tính toán xác suất và dự đoán (Giữ nguyên)
    proba = model.predict_proba(X_scaled)[0,1]
    pred = int(proba >= 0.5)
    
    # --- QUAN TRỌNG: Định nghĩa biến này NGAY TẠI ĐÂY ---
    result_to_save = {'prediction': pred, 'probability': float(proba)}
    
    # 2. Hiển thị thông báo (Giữ nguyên)
    if pred == 1:
        st.success(f'Khách hàng này là TIỀM NĂNG! (Xác suất: {proba*100:.1f}%)')
    else:
        st.warning(f'Khách hàng này KHÔNG tiềm năng (Xác suất: {proba*100:.1f}%)')

    # 3. Lưu vào Database (Nằm trong if st.button)
    try:
        # Chuyển đổi dữ liệu input hàng đầu tiên thành dictionary
        input_data_dict = X_input.iloc[0].to_dict()
        
        input_json = json.dumps(input_data_dict)
        output_json = json.dumps(result_to_save) 
        current_time = pd.Timestamp.now().isoformat()
        
        # Đảm bảo các dòng dưới đây thẳng hàng nhau
        c.execute('INSERT INTO predictions (input, output, timestamp) VALUES (?, ?, ?)',
                  (input_json, output_json, current_time))
        conn.commit()
    except Exception as e:
        st.error(f"Không thể lưu lịch sử: {e}")

# 6. Radar/Bar chart so sánh với khách tiềm năng trung bình
if df_potential is not None:
        df_potential['DiscountsAvailed'] = df_potential['DiscountsAvailed'].map({'Yes': 1,'No': 0})

        compare_cols = ['AnnualIncome', 'CustomerSatisfaction', 'TimeSpentOnWebsite', 'DiscountsAvailed']
        avg_potential = df_potential[compare_cols].mean()
        user_vals = [X_input.iloc[0][c] for c in compare_cols]
        
        # Tính toán tỷ lệ (giữ nguyên số thập phân để Altair dễ xử lý)
        percent_vals = [(u / a) if a != 0 else 0 for u, a in zip(user_vals, avg_potential)]
        
        df_pct_compare = pd.DataFrame({
            'Feature': compare_cols,
            'Value': percent_vals
        })

        st.subheader('Chỉ số khách hiện tại so với chuẩn tiềm năng (100%)')
        
        # Tạo biểu đồ cột chính
        base_comp =alt.Chart(df_pct_compare).mark_bar().encode(
            x=alt.X('Feature:N', title='Chỉ số', axis=alt.Axis(labelAngle=0)), # Để label nằm ngang cho dễ đọc
            y=alt.Y('Value:Q', title='Tỷ lệ so với chuẩn', axis=alt.Axis(format='%')),
            color=alt.condition(
                alt.datum.Value >= 1,
                alt.value('#2ecc71'), # Màu xanh (Vượt chuẩn)
                alt.value('#e74c3c')  # Màu đỏ (Dưới chuẩn)
            ),
            tooltip=[alt.Tooltip('Feature'), alt.Tooltip('Value', format='.2%')]
        )

        # Thêm con số % trên đầu mỗi cột
        text_comp = base_comp.mark_text(align='center', baseline='bottom', dy=-5).encode(
            text=alt.Text('Value:Q', format='.1%')
        )

        # TẠO ĐƯỜNG KẺ CHUẨN 100% (Rule Mark)
        rule = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(
            color='gray', 
            strokeDash=[5, 5], 
            size=2
        ).encode(y='y:Q')

        # Hiển thị biểu đồ kết hợp (Cột + Chữ + Đường kẻ)
        st.altair_chart(base_comp + text_comp + rule, use_container_width=True)

    # 7. Feature Importance
if feature_imp is not None:
        st.subheader('Tỷ trọng ảnh hưởng của các yếu tố (%)')
fi = feature_imp.head(10).reset_index()
fi.columns = ['Feature', 'Importance']

bars_fi = alt.Chart(fi).mark_bar(color='#3498db').encode(
        x=alt.X('Feature:N', sort='-y', axis=alt.Axis(labelAngle=45)),
        y=alt.Y('Importance:Q', axis=alt.Axis(format='%'), title='Mức độ quan trọng')
    )

text_fi = bars_fi.mark_text(align='center', baseline='bottom', dy=-5).encode(
        text=alt.Text('Importance:Q', format='.1%')
    )

st.altair_chart(bars_fi + text_fi, use_container_width=True)

st.caption('Lưu ý: Kết quả chỉ mang tính chất tham khảo, không thay thế quyết định kinh doanh thực tế.')





# 8. Hướng dẫn chạy
#st.sidebar.markdown('---')
#st.sidebar.markdown('**Hướng dẫn chạy ứng dụng:**')
#st.sidebar.code('python -m streamlit run app.py')
#dòng 34-58-78 dòng đã thêm # để comment ra, không ảnh hưởng đến chương trình