from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 데이터 로딩
data = load_wine()
X, y = data.data, data.target

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 디시전 트리 모델 생성 및 학습
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# 랜덤 포레스트 모델 생성 및 학습
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# XGBoost 모델 생성 및 학습
xg_cls = xgb.XGBClassifier()
xg_cls.fit(X_train, y_train)

# 모델 평가
y_pred_dt = dt.predict(X_test)
print(f'Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}')

y_pred_rf = rf.predict(X_test)
print(f'Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}')

y_pred_xgb = xg_cls.predict(X_test)
print(f'XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}')

