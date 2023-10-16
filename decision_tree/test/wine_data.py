import pandas as pd
from sklearn.datasets import load_wine

# 와인 데이터셋 로드
wine = load_wine(as_frame=True)
df = wine.data
# 첫 5행 출력
print(df.head())

df = wine.target
# 첫 5행 출력
print(df.head())
