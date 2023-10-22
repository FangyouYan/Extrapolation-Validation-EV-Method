import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

data = pd.read_csv(r"House-Price-Prediction-clean.csv")
print(data.shape)
data_ = data.dropna(axis=0, subset=['LotArea'])
print(data_.shape)
print('每一列的属性名称：', data_.columns)
mean = data_['LotArea'].mean()
std = data_['LotArea'].std()
print('平均值：', mean)
print('标准差：', std)
# data_['MSSubClass'].plot(kind='hist')
# plt.savefig(fname="1.png")
# data_['LotArea'].plot(kind='density')
# plt.savefig(fname="2.png")
data_d = data_[(data_['LotArea'] <= mean + 3 * std) & (data_['LotArea'] >= mean - 3 * std)]
print(data_d.shape)
data_d['LotArea_norm'] = (data_d['LotArea'] - mean) / std
print('根据平均值与标准差将数据归一化：')
print(data_d.shape)
print(data_d.columns)
# data_d['MSSubClass'].plot(kind='hist')
# plt.savefig(fname="3.png")
# data_d['LotArea_norm'].plot(kind='density')
# plt.savefig(fname="4.png")
data_d.to_csv('output.csv')