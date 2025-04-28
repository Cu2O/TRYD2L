import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

print(train_data.shape)
print(test_data.shape)

# all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(train_data.columns.tolist())
columns_to_drop = ['Id', 'Address','Summary', 'Elementary School',
                   'Middle School', 'High School']
train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)
all_features = pd.concat((train_data, test_data))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 把某些杂合项分开
column_to_split = ['Cooling', 'Parking','Flooring', 'Heating features', 'Cooling features', 'Appliances included', 'Laundry features', 'Parking features']

# 将列内容按逗号分隔，并展开为多个独热编码列
for i in range(len(column_to_split)):
    if column_to_split[i] in all_features.columns:
        # 拆分列内容
        split_columns = all_features[column_to_split[i]].str.get_dummies(sep=',')
        # 将拆分后的独热编码列拼接回原数据
        all_features = pd.concat([all_features, split_columns], axis=1)
        # 删除原始列
        all_features = all_features.drop(columns=[column_to_split[i]])

all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32)

print(train_features.shape, train_labels.shape, test_features.shape)

