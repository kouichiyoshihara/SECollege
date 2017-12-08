
# coding: utf-8

# In[4]:

import pandas as pd
# CSVデータの読み込み
iris = pd.read_csv("iris.data")
# 基本情報の表示
iris.info()


# In[6]:

# 行数の表示
len(iris)


# In[8]:

# 行数と列数の表示
iris.shape


# In[12]:

# 先頭の５行を表示
iris.head()
# 末尾の５行を表示
iris.tail()


# In[18]:

# 列名を指定して表示
#iris["SepalLength"]
iris.SepalLength.head(3)
# 複数列を指定
iris[["SepalLength", "SepalWidth"]].tail()


# In[20]:

# 特定のレコード範囲を指定
iris[2:5]


# In[22]:

# 品種ごとにデータを区分け
setosa = iris[iris.Class == "Iris-setosa"]
versicolor = iris[iris.Class == "Iris-versicolor"]
virgincia = iris[iris.Class == "Iris-virginica"]


# In[26]:

# 基本統計量の算出
setosa.sum() # 合計値
setosa.min() # 最小値
setosa.max() # 最大値
setosa.mean() # 平均値
setosa.median() # 中央値
setosa.var() # 分散
setosa.std() # 標準偏差


# In[28]:

import numpy as np
# ピボットテーブルの作成

pd.pivot_table(iris, index="Class", aggfunc=np.mean)


# In[33]:

import matplotlib.pyplot as plt
# ヒストグラムの作成
plt.hist(iris.SepalLength)
plt.xlabel("Sepal Length") 
plt.ylabel("Freq") # Frequency: 度数
plt.show() # 表示


# In[34]:

# 品種ごとのがく片の長さに対する箱ひげ図の作成
data = [setosa.SepalLength, 
        versicolor.SepalLength,
        virgincia.SepalLength]
plt.boxplot(data) # 箱ひげ図の作成
plt.xlabel("Class") # 品種
plt.ylabel("Sepal Length") # がく片の長さ
plt.setp(plt.gca(), xticklabels=["setosa", 
                                 "versicolor", 
                                 "virgincia"])
plt.show()


# In[36]:

# 散布図の作成
plt.scatter(setosa.SepalLength, setosa.SepalWidth) 
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# In[39]:

import sklearn.linear_model as lm
# 単回帰分析を行う

# 相関係数の算出
corr = np.corrcoef(setosa.SepalLength, 
                   setosa.SepalWidth)
corr


# In[49]:

lr = lm.LinearRegression() # 線形回帰モデルの作成
x = setosa[["SepalLength"]]
y = setosa[["SepalWidth"]]
lr.fit(x, y) # モデルにデータを当て嵌め
# arange([start], stop, [step], [dtype])
px = np.arange(x.min(), x.max(), .01)[:, np.newaxis]
py = lr.predict(px) # 線形回帰予測モデルの作成
plt.plot(px, py, color="blue", 
                 linewidth=3) # 図のプロット
plt.scatter(x, y, color="black") # 散布図の作成
plt.show()
print(lr.coef_) # 回帰係数
print(lr.intercept_) # 切片

