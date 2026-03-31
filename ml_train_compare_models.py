import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
import joblib

# Nếu chưa cài xgboost hoặc lightgbm, hãy cài bằng pip trước khi chạy script này
try:
    from xgboost import XGBClassifier
    xgb_installed = True
except ImportError:
    xgb_installed = False
try:
    from lightgbm import LGBMClassifier
    lgbm_installed = True
except ImportError:
    lgbm_installed = False

# 1. Load processed data
train_df = pd.read_csv('processed_customer_data.csv')
features = [col for col in train_df.columns if col != 'PurchaseStatus']
X = train_df[features]
y = train_df['PurchaseStatus']

# 2. Train/Test split (chia dữ liệu đã qua xử lý thành train và test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Áp dụng SMOTETomek chỉ trên tập huấn luyện
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

print("Before SMOTETomek:", X_train.shape)
print("After SMOTETomek:", X_train_res.shape)

# 3. K-Fold Cross Validation (trên dữ liệu huấn luyện đã qua resample)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3.1 Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg_scores = cross_val_score(logreg, X_train_res, y_train_res, cv=kfold, scoring='roc_auc') # Use resampled data
logreg.fit(X_train_res, y_train_res) # Fit on resampled data

# 3.2 Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(rf, X_train_res, y_train_res, cv=kfold, scoring='roc_auc') # Use resampled data
rf.fit(X_train_res, y_train_res) # Fit on resampled data

# 3.3 XGBoost/LightGBM
if xgb_installed:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_scores = cross_val_score(xgb, X_train_res, y_train_res, cv=kfold, scoring='roc_auc') # Use resampled data
    xgb.fit(X_train_res, y_train_res) # Fit on resampled data
    best_gbm = xgb
    model_name = 'XGBoost'
elif lgbm_installed:
    lgbm = LGBMClassifier(random_state=42)
    xgb_scores = cross_val_score(lgbm, X_train_res, y_train_res, cv=kfold, scoring='roc_auc') # Use resampled data
    lgbm.fit(X_train_res, y_train_res) # Fit on resampled data
    best_gbm = lgbm
    model_name = 'LightGBM'
else:
    raise ImportError('Cần cài xgboost hoặc lightgbm')

# 4. Hyperparameter Tuning cho XGBoost/LightGBM (trên dữ liệu huấn luyện đã qua resample)
if xgb_installed:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                        param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_res, y_train_res) # Fit on resampled data
    best_gbm = grid.best_estimator_
    model_name = 'XGBoost (Tuned)'
    print('Best XGBoost params:', grid.best_params_)
elif lgbm_installed:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid = GridSearchCV(LGBMClassifier(random_state=42),
                        param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    best_gbm = grid.best_estimator_
    model_name = 'LightGBM (Tuned)'
    print('Best LightGBM params:', grid.best_params_)

# 5. Đánh giá trên tập test (sử dụng X_test, y_test)
def evaluate_model(model, X_eval, y_eval):
    threshold = 0.3  # thử 0.3 hoặc 0.4
    y_proba = model.predict_proba(X_eval)[:,1] # Calculate y_proba first
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'Accuracy': accuracy_score(y_eval, y_pred),
        'Precision': precision_score(y_eval, y_pred),
        'Recall': recall_score(y_eval, y_pred),
        'F1-Score': f1_score(y_eval, y_pred),
        'ROC-AUC': roc_auc_score(y_eval, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }

results = {}
results['Logistic Regression'] = evaluate_model(logreg, X_test, y_test)
results['Random Forest'] = evaluate_model(rf, X_test, y_test)
results[model_name] = evaluate_model(best_gbm, X_test, y_test) # Use model_name for the key

# 6. So sánh kết quả
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
summary = pd.DataFrame({model: {m: results[model][m] for m in metrics} for model in results})
print('Model Comparison Table (on Test Data):') # Added (on Test Data)
print(summary)
summary.to_csv('model_comparison.csv')

# 7. Vẽ ROC và Precision-Recall Curve (on Test Data)
plt.figure(figsize=(8,6))
for model, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba']) # Changed y to y_test
    plt.plot(fpr, tpr, label=f'{model} (AUC={res["ROC-AUC"]:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (on Test Data)') # Added (on Test Data)
plt.legend()
plt.savefig('roc_curve_comparison.png')
plt.close()

plt.figure(figsize=(8,6))
for model, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_proba']) # Changed y to y_test
    plt.plot(rec, prec, label=model)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison (on Test Data)') # Added (on Test Data)
plt.legend()
plt.savefig('pr_curve_comparison.png')
plt.close()

# 8. Confusion Matrix cho mô hình tốt nhất (on Test Data)
best_model_name = summary.T['ROC-AUC'].idxmax()
# Fixed model selection for best_model_instance to include the actual model_name as key
best_model_instance = {
    'Logistic Regression': logreg,
    'Random Forest': rf,
    model_name: best_gbm # Use model_name dynamically
}[best_model_name]
best_pred = results[best_model_name]['y_pred']
cm = confusion_matrix(y_test, best_pred) # Changed y to y_test
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title(f'Confusion Matrix: {best_model_name} (on Test Data)') # Added (on Test Data)
plt.savefig('confusion_matrix_best_model.png')
plt.close()

# 9. Feature Importance
if hasattr(best_model_instance, 'feature_importances_'): # Used best_model_instance
    importances = best_model_instance.feature_importances_
    feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)
else:
    # Logistic Regression
    importances = np.abs(best_model_instance.coef_[0])
    feature_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print('Top 10 Important Features:')
print(feature_imp.head(10))
feature_imp.head(10).to_csv('feature_importance.csv')

# 10. Lưu mô hình và features
joblib.dump(best_model_instance, 'best_model.pkl') # Used best_model_instance
with open('features_list.json', 'w') as f:
    json.dump(list(features), f)

# 11. Nhận xét cuối cùng
print(f'\nBest model for production (based on test data): {best_model_name}') # Added (based on test data)
print('Lý do chọn:')
if best_model_name == 'Logistic Regression':
    print('- Đơn giản, dễ giải thích, tốc độ nhanh, phù hợp nếu yêu cầu explainable AI.')
elif best_model_name == 'Random Forest':
    print('- Cân bằng giữa hiệu năng và khả năng giải thích, robust với dữ liệu nhiễu.')
else:
    print('- Hiệu năng cao nhất (ROC-AUC), khả năng tổng quát hóa tốt, phù hợp cho production.')
print('Các file kết quả: model_comparison.csv, roc_curve_comparison.png, pr_curve_comparison.png, confusion_matrix_best_model.png, feature_importance.csv, best_model.pkl, features_list.json')