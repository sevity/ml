from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 모델 생성 및 학습
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# 평가
y_pred = dt.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

