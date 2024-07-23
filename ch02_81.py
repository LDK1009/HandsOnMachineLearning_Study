
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

# ==========데이터 불러오기==========
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

# print("==========데이터 구조 훑어 보기==========")
# print(housing.head())

# print("==========데이터 설명 출력==========")
# print(housing.info())

# print("==========해당 컬럼의 고유값과 행 수==========")
# print(housing["ocean_proximity"].value_counts())

# print("==========숫자형 특성 요약 정보==========")
# print(housing.describe())

# print("==========상관관계 조사==========")
# corr_matrix = housing.corr(numeric_only=True)
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ==========모든 숫자형 특성의 히스토그램 출력==========
import matplotlib.pyplot as plt

# 추가 코드 – 다음 다섯 라인은 기본 폰트 크기를 지정합니다
# plt.rc('font', size=14)
# plt.rc('axes', labelsize=14, titlesize=14)
# plt.rc('legend', fontsize=14)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)

# housing.hist(bins=50, figsize=(12, 8))
# plt.show()


# ==========폰트 설정==========
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# ==========테스트 세트 만들기==========
from sklearn.model_selection import train_test_split
import numpy as np


# 데이터셋을 학습 세트와 테스트 세트로 분할
train_set, test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_category"],random_state=42)

# ==========계층적 샘플링==========
# 'median_income' 열을 기반으로 'income_category'라는 새로운 범주형 속성 추가
housing["income_category"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],  # 'median_income'을 카테고리로 나누는 기준(1.2는 label 1에 속한다)
                                    labels=[1, 2, 3, 4, 5]  # 각 소득 카테고리에 대한 레이블
                                   )


# 소득 카테고리의 분포를 막대 그래프로 표시
housing["income_category"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("소득 카테고리")  # X축 레이블
plt.ylabel("구역 개수")  # Y축 레이블
plt.show()  # 그래프 출력
