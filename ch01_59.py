import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# 데이터 준비
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
print(X)
y = lifesat[["Life satisfaction"]].values
print(y)

# 데이터 시각화
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# 모델 선택(선형 모델)
model = KNeighborsRegressor(n_neighbors=3)

model.fit()

# 키프로스(*국가명이다.)에 대한 예측 생성
X_new = [[37_655.2]] # 2020년 키프로스 1인당 GDP
print(model.predict(X_new)) # 출력 : [[6.30165767]]
