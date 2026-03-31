# Hệ Thống Dự Báo Khách Hàng Tiềm Năng (E-commerce Customer Prediction)
Dự án này tập trung vào việc xây dựng và triển khai mô hình học máy để phân loại và dự đoán khả năng mua hàng của người dùng trên nền tảng Thương mại điện tử, giúp tối ưu hóa chuyển đổi và chi phí vận hành.

 # Mô tả dữ liệu file customerData_50k.csv
1. **Tổng quan bộ dữ liệu**
   - Số lượng bản ghi: 50,000 dòng.
   - Số lượng cột: 17 cột (bao gồm các thông tin định danh, hành vi và mục tiêu).
   - Mục tiêu (Target Variable): PurchaseStatus (0: Không mua, 1: Có mua hàng).

2. **Chi tiết các biến số (Features)**

***Nhóm 1: Thông tin nhân khẩu học (Demographics)***
   - Age: Độ tuổi của khách hàng.
   - Gender: Giới tính (Male/Female).
   - AnnualIncome: Thu nhập hàng năm (Đây là biến quan trọng nhất ảnh hưởng đến khả năng mua hàng).
   - Region: Khu vực địa lý (North, South, East, West).

***Nhóm 2: Hành vi trên Website (Behavioral Data)***
   - TimeSpentOnWebsite: Thời gian khách hàng truy cập trang web (tính bằng phút).
   - SessionCount: Số lượng phiên truy cập trong một khoảng thời gian nhất định.
   - PreferredDevice: Thiết bị ưu tiên sử dụng (Mobile, Desktop, Tablet).
   - ReferralSource: Nguồn dẫn khách hàng đến website (Social Media, Search Engine, Direct, Referral).

***Nhóm 3: Lịch sử mua hàng và Quan hệ khách hàng (Transaction & CRM)***
   - NumberOfPurchases: Tổng số đơn hàng đã thực hiện trước đó.
   - CustomerTenureYears: Số năm khách hàng đã gắn bó với hệ thống.
   - LastPurchaseDaysAgo: Số ngày kể từ lần mua hàng cuối cùng (Chỉ số Recency).
   - CustomerSatisfaction: Điểm hài lòng của khách hàng (thang điểm 1-5).
   - ProductCategory: Danh mục sản phẩm quan tâm nhất (Electronics, Fashion, Beauty, Sports, Home).
   - CustomerSegment: Phân khúc khách hàng (Low Value, Medium Value, High Value).

***Nhóm 4: Chương trình ưu đãi & Mục tiêu***
   - LoyaltyProgram: Có tham gia chương trình khách hàng thân thiết hay không (Yes/No).
   - DiscountsAvailed: Có sử dụng mã giảm giá hay không (Yes/No).
   - PurchaseStatus (Target): Biến phân loại cho biết khách hàng đó có thực hiện mua hàng trong kỳ quan sát hay không.

1. **Mục Tiêu Dự Án**
   - Xây dựng một hệ thống hoàn chỉnh từ xử lý dữ liệu đến triển khai ứng dụng thực tế nhằm:
   - Phân loại chính xác nhóm khách hàng có khả năng mua hàng cao (Potential).
   - Cung cấp cơ sở dữ liệu để bộ phận Marketing tối ưu hóa các chiến dịch cho từng đối tượng khách hàng

2. **Vấn đề Kinh doanh**
   - Trong môi trường TMĐT cạnh tranh, việc đổ ngân sách marketing đại trà gây lãng phí lớn.
   - Thách thức: Xác định ai là người thực sự có ý định mua hàng trong hàng triệu lượt truy cập.
   - Giải pháp: Sử dụng các thuật toán ML để dự báo hành vi, giúp giảm chi phí tiếp cận khách hàng (CAC) và tăng giá trị vòng đời khách hàng (LTV).

3. **Tổng quan Quy trình**
   - Dữ liệu: Sử dụng tập dữ liệu 50,000 dòng (customerData_50k.csv).
   - Xử lý: Làm sạch, chuẩn hóa và tạo biến phái sinh (Feature Engineering) để tăng hiệu quả chính xác của Model ML
   - Mô hình được sử dụng trong bài này là Logistic Regression, Random Forest, XGBoost
   - Triển khai: Đóng gói mô hình thành API và giao diện Web (Streamlit).

4. **Khám phá Dữ liệu (EDA)**
   - Qua phân tích, các yếu tố then chốt ảnh hưởng đến hành vi mua sắm bao gồm:
   - Annual Income: Thu nhập hàng năm là yếu tố dẫn đầu trong việc phân loại phân khúc khách hàng.
   - TimeSpentOnWebsite : Thời gian truy cập vào ứng dụng, website để lướt xem sản phẩm
   - Customer Satisfaction: Chỉ số hài lòng có mối tương quan thuận với tỷ lệ quay lại mua hàng.
   

5. **Feature Engineering**
   - Dữ liệu được tinh chỉnh qua các bước:
   - Tạo biến mới: Income_per_Session, AVG_PURCHASE_3M,PurchaseRate,IncomePerPurchase,Engagement,...
   - Xử lý số liệu: Chuẩn hóa dữ liệu (Scaling) bằng StandardScaler để đưa các biến về cùng một thang đo.
   - Lựa chọn đặc trưng: Tập trung vào Feature >0  dựa trên điểm số Feature Importance trong Random Forest.

6. **Huấn luyện & So sánh Mô hình**
   - Dựa trên kết quả thực tế từ file model_comparison.csv
   - Nhận xét:XGBoost (Tuned) là mô hình tốt nhất với các chỉ số ROC-AUC (0.84) cao nhất, cho khả năng phân loại 2 lớp ổn định và chính xác hơn Random Forest và Logistic Regression

7. **Kết quả Mô hình**
   - Bảng so sánh chỉ số, ROC, Feature Importance
   - AnnualIncome	23.15%
   - DiscountsAvailed	15.12%
   - TimeSpentOnWebsite	13.16%
   - CustomerSatisfaction	11.19%

8. **Triển khai Ứng dụng (API)**
   - Quy trình dự đoán, kiến trúc API, lưu lịch sử dự đoán
   - Xếp hạng các yếu tố ảnh hưởng nhất đến mô hình:
   - Backend: FastAPI dùng để phục vụ mô hình (model.pkl), tiếp nhận dữ liệu đầu vào và trả về kết quả dự báo.
   - Frontend: Giao diện Streamlit thân thiện cho phép nhân viên kinh doanh nhập liệu trực tiếp hoặc upload file.
   - Lưu trữ: Tích hợp tính năng lưu lịch sử dự đoán để theo dõi độ chính xác theo thời gian.

9. **Demo Ứng dụng**
   - Minh họa quy trình nhập liệu, dự báo và lưu kết quả
   - Tiết kiệm chi phí: Tập trung ngân sách quảng cáo vào nhóm khách hàng được AI dự báo là "Tiềm năng".
   - Chiến lược cá nhân hóa: Tự động gửi mã giảm giá (Discount) cho nhóm có LastPurchaseDaysAgo cao nhưng có AnnualIncome tốt.
   - Hướng phát triển: Tích hợp dữ liệu thời gian thực (Real-time tracking) để dự báo ngay trong phiên truy cập của khách hàng.

10. **Giá trị Kinh doanh & Khuyến nghị**
    - Lợi ích thực tiễn, chiến lược marketing, đề xuất phát triển tiếp
