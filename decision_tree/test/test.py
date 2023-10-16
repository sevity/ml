from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np

# 데이터 로딩
data = load_wine()
X, y = data.data, data.target

# 디시전 트리 모델 생성
dt = DecisionTreeClassifier()

# 랜덤 포레스트 모델 생성
rf = RandomForestClassifier()

# XGBoost 모델 생성
xg_cls = xgb.XGBClassifier()

# 교차 검증 수행 (5-fold CV)
cv_scores_dt = cross_val_score(dt, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
cv_scores_xgb = cross_val_score(xg_cls, X, y, cv=5)

# 평균 정확도 출력
print(f'Decision Tree CV Accuracy: {np.mean(cv_scores_dt):.2f}')
print(f'Random Forest CV Accuracy: {np.mean(cv_scores_rf):.2f}')
print(f'XGBoost CV Accuracy: {np.mean(cv_scores_xgb):.2f}')

