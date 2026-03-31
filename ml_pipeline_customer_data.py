import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import joblib

# 1. Load data
file_path = 'customerData_50k.csv'
df = pd.read_csv(file_path)

# 1.1 EDA: Target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='PurchaseStatus', data=df)
plt.title('Target Distribution: PurchaseStatus')
plt.savefig('eda_target_distribution.png')
plt.close()
 #để xác định là dữ liệu có bị lệch lớp không, nếu lệch thì có thể cần xử lý
 # bằng oversampling hoặc undersampling sau này.

# 1.2 EDA: AnnualIncome vs PurchaseStatus by ProductCategory
plt.figure(figsize=(10,6))
sns.boxplot(x='ProductCategory', y='AnnualIncome', hue='PurchaseStatus', data=df)
plt.title('AnnualIncome vs PurchaseStatus by ProductCategory')
plt.savefig('eda_income_vs_status_by_category.png')
plt.close()

    #Thu nhập ảnh hưởng đến mua hàng không? Có sự khác biệt giữa các
    # loại sản phẩm không? Đây là những câu hỏi mà biểu đồ này giúp trả lời.
    # Nếu có sự khác biệt rõ ràng, có thể cân nhắc tạo thêm các feature tương tác
    # giữa AnnualIncome và ProductCategory trong bước Feature Engineering.
    # Với  biểu đồ này, Income có tác động đến tỉ lệ mua hàng của khách -> thu nhập cao tỉ lệ mua cao
    # và partern giữa các ngành hàng không có sự khác biệt  nhiều

# 1.3 EDA: Correlation heatmap
# 1.3 EDA: Correlation heatmap + remove highly correlated features
plt.figure(figsize=(10,8))

# Lấy các cột số
numeric_df = df.select_dtypes(include=[np.number])

# Tính correlation
corr = numeric_df.corr().abs()

# Lấy nửa trên của ma trận
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

# Tìm các feature có correlation > 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

print("Các feature bị loại do tương quan cao:", to_drop)

# Dataset sau khi loại
df_filtered = numeric_df.drop(columns=to_drop)

# Tính lại correlation
corr_filtered = df_filtered.corr()

# Vẽ heatmap
sns.heatmap(corr_filtered, annot=True, cmap='coolwarm', fmt='.2f')

plt.title('Correlation Matrix Heatmap (Removed corr > 0.8)')
plt.savefig('eda_correlation_heatmap.png')
plt.close()

     #Nhìn vào biểu đồ ma trận tương quan ta thấy được là Income và TimeSpentOnWebsite
     # có tương quan dương với PurchaseStatus, trong khi LastPurchaseDaysAgo có tương quan âm.

# 2. Feature Engineering
# 2.1 AVG_PURCHASE_3M: avg purchases per 3 months (assume uniform) (nhân 4 là vì 1 năm có 4 quý và +1 để tránh chia cho 0)
df['AVG_PURCHASE_3M'] = df['NumberOfPurchases'] / (df['CustomerTenureYears'] * 4 + 1)

# 2.2 MAX_SPENT_6M: proxy = (AnnualIncome/2) * (SessionCount/CustomerTenureYears/12)
# Khả năng chi tiêu ≈ Thu nhập × Mức độ hoạt động
df['MAX_SPENT_6M'] = (df['AnnualIncome'] / 2) * (df['SessionCount'] / (df['CustomerTenureYears'] * 12 + 1))

# 2.3 STDDEV_SATISFACTION_12M: simulate with random noise around CustomerSatisfaction
# Giả sử sự biến động của sự hài lòng có thể được ước tính từ mức độ tương tác và thời gian kể từ lần mua hàng cuối cùng
#engagement_score là số lần truy cập trung bình mỗi tháng.
#Score = a*Engagement + b*Recency + c*Satisfaction
#Trong TMDT thì tương tác với web nhiều thì cơ hội chuyển đôi khá cao nên là 40%, Khoảng thời gian mua gần nhất quyết dịnh là khách có tần suất mua hang nhiều không nên là 40,
#  còn lại mức do hài long ở cuối 20% (vì nếu hài long cao nhưng tương tác và ngày mua dài thì lượt chuyển doi cũng không cao)

df['ENGAGEMENT_SCORE'] = df['SessionCount'] / (df['CustomerTenureYears'] * 12 + 1)
df['STDDEV_SATISFACTION_EST'] = ( df['ENGAGEMENT_SCORE'] * 0.4 + (df['LastPurchaseDaysAgo'] / 365) * 0.4 + (5 - df['CustomerSatisfaction']) * 0.2 )
# 2.4 Income_per_Session-Thu nhập trung bình "ứng với mỗi lần truy cập"

df['Income_per_Session'] = df['AnnualIncome'] / (df['SessionCount'] + 1) # thu nhập trung bình mỗi lần truy cập
df["IncomePerPurchase"] = df["AnnualIncome"] / (df["NumberOfPurchases"] + 1)
df['RecencyScore'] = 1 / (df['LastPurchaseDaysAgo'] + 1) # điểm đánh giá độ mới của khách hàng
df['Engagement'] = df['TimeSpentOnWebsite'] / (df['SessionCount'] + 1) #thời gian trung bình một ở lại tronng 1 lần truy cập
df['PurchaseRate'] = df['NumberOfPurchases'] / (df['SessionCount'] + 1)
df['Time_per_Purchase'] = df['TimeSpentOnWebsite'] / (df['NumberOfPurchases'] + 1)

df["EngagementScore"] = (df["SessionCount"] * 0.4 +df["TimeSpentOnWebsite"] * 0.3 +df["NumberOfPurchases"] * 0.3) # điểm đánh giá mức độ tương tác tổng thể của khách hàng
df["LoyaltyScore"] = (df["CustomerTenureYears"] * 0.5 +df["NumberOfPurchases"] * 0.5)


# Convert 'DiscountsAvailed' to numerical before using it in calculations
df['DiscountsAvailed'] = df['DiscountsAvailed'].map({'Yes': 1, 'No': 0}).astype(int)

df["DiscountRate"] = df["DiscountsAvailed"] / (df["NumberOfPurchases"] + 1)
df["ExperienceScore"] = (
df["CustomerSatisfaction"] * df["CustomerTenureYears"])

# 3. Preprocessing
# 3.1 Encoding
# OneHot for ProductCategory, Region
#vì model không hiểu text nên phải chuyển thành số, drop_first=True để tránh đa cộng tuyến.
# Dùng OneHot Encoding dùng khi các giá trị không có thứ tự,
onehot_cols = ['ProductCategory', 'Region']
df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

# LabelEncoder for CustomerSegment
#còn Label Encoding dùng khi các giá trị có thứ tự (ví dụ: CustomerSegment có thể là 'Low', 'Medium', 'High' -> có thứ tự nên dùng LabelEncoder)
le_segment = LabelEncoder()
df_encoded['CustomerSegment_LE'] = le_segment.fit_transform(df_encoded['CustomerSegment'])

# 3.2 Handle Outliers in AnnualIncome (IQR) - bước này để xử lí các giá trị ngoại lai trong AnnualIncome,
# vì thu nhập có thể có sự chênh lệch lớn giữa các khách hàng, và những giá trị quá cao hoặc quá thấp có thể ảnh hưởng đến hiệu suất của model.
Q1 = df_encoded['AnnualIncome'].quantile(0.25)
Q3 = df_encoded['AnnualIncome'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
#Nếu giá trị < lower  → đổi thành lower
#Nếu giá trị > upper  → đổi thành upper
#Nếu nằm giữa        → giữ nguyên
df_encoded['AnnualIncome'] = np.clip(df_encoded['AnnualIncome'], lower, upper)

# 3.3 Tạo danh sách các feature dùng để train model
feature_cols = [
    'Age', 'AnnualIncome', 'NumberOfPurchases', 'TimeSpentOnWebsite',
    'CustomerTenureYears', 'LastPurchaseDaysAgo', 'SessionCount', 'CustomerSatisfaction',
    'AVG_PURCHASE_3M', 'MAX_SPENT_6M', 'STDDEV_SATISFACTION_EST', 'Income_per_Session','RecencyScore','Engagement','PurchaseRate','Time_per_Purchase',"EngagementScore","LoyaltyScore","IncomePerPurchase",'DiscountsAvailed',"ExperienceScore","CustomerSatisfaction"
] + [col for col in df_encoded.columns if col.startswith('ProductCategory_') or col.startswith('Region_')] + ['CustomerSegment_LE']

# Remove duplicates from feature_cols to ensure consistent column count
feature_cols = list(dict.fromkeys(feature_cols))

X = df_encoded[feature_cols]
y = df_encoded['PurchaseStatus']

# Use RandomForest for feature importance - sử dụng Random Forest để đánh giá tầm quan trọng của các feature, từ đó chọn ra những feature có ảnh hưởng nhất đến việc dự đoán PurchaseStatus.
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
#train model để đánh giá tầm quan trọng của các feature, feature_importances_ có sẵn trong RandomForestClassifier, giúp chúng ta dễ dàng xác định được những feature nào đóng góp nhiều nhất vào việc dự đoán. Trong khi đó, Logistic Regression và XGBoost cũng có thể đánh giá tầm quan trọng nhưng thường phức tạp hơn và không trực quan bằng Random Forest. 
# Do đó, nếu mục tiêu chính là đánh giá tầm quan trọng của các feature một cách nhanh chóng và trực quan, Random Forest thường là lựa chọn ưu tiên.
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances[importances > 0].sort_values(ascending=False).index.tolist() #chọn ra các feature >0 có tầm quan trọng cao nhất để sử dụng trong bước tiếp theo.

# 4. Train/Test split #stratify = y để đảm bảo rằng tỷ lệ của các lớp trong tập train và test giống nhau, giúp model học tốt hơn và đánh giá chính xác hơn.
X_selected = X[top_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Chuẩn hóa dữ liệu (Scaling) #StandardScaler lấy mean và std của tập train để chuẩn hóa cả train và test, đảm bảo rằng thông tin từ tập test không bị rò rỉ vào quá trình huấn luyện.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Save processed data and scaler  
processed = pd.DataFrame(X_train_scaled, columns=top_features) #tạo dataset mới chứa các feature
processed['PurchaseStatus'] = y_train.values
processed.to_csv('processed_customer_data.csv', index=False)
joblib.dump(scaler, 'scaler_customer_data.joblib') #lưu scaler đã fit trên tập train để sử dụng trong bước tiếp theo, đảm bảo rằng dữ liệu mới được chuẩn hóa theo cùng một cách như dữ liệu huấn luyện.

print('EDA plots saved: eda_target_distribution.png, eda_income_vs_status_by_category.png, eda_correlation_heatmap.png')
print('Processed data saved: processed_customer_data.csv')
print('Scaler saved: scaler_customer_data.joblib')




#tóm lại mục đích của pipeline này là để chuẩn bị dữ liệu cho bước tiếp theo là training model,




#tại sao lại chọn random forest mà không phải là logistic regression hay xgboost?
#  Vì Random Forest có khả năng đánh giá tầm quan trọng của các feature một cách trực quan thông qua thuộc tính feature_importances_, giúp chúng ta dễ dàng xác định được những feature nào đóng góp nhiều nhất vào việc dự đoán. Trong khi đó, Logistic Regression và XGBoost cũng có thể đánh giá tầm quan trọng nhưng thường phức tạp hơn và không trực quan bằng Random Forest.
#vậy khi nào dùng logistic regression hay xgboost để đánh giá tầm quan trọng của feature?
# Logistic Regression có thể được sử dụng để đánh giá tầm quan trọng của các feature thông qua hệ số (coefficients) của mô hình, nhưng nó chỉ phù hợp khi mối quan hệ giữa các feature và target là tuyến tính. XGBoost có thể đánh giá tầm quan trọng của các feature thông qua các phương pháp như gain, cover hoặc frequency, và nó có thể xử lý mối quan hệ phi tuyến tính tốt hơn. Tuy nhiên, việc sử dụng XGBoost để đánh giá tầm quan trọng của feature có thể phức tạp hơn và đòi hỏi nhiều thời gian huấn luyện hơn so với Random Forest. Do đó, nếu mục tiêu chính là đánh giá tầm quan trọng của các feature một cách nhanh chóng và trực quan, Random Forest thường là lựa chọn ưu tiên.
