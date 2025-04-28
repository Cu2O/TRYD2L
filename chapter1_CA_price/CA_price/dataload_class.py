import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

# 检测是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

class DataLoad:
    def __init__(self, train_path, test_path):
        """
        初始化数据加载类
        :param train_path: 训练数据的文件路径
        :param test_path: 测试数据的文件路径
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.all_features = None
        self.train_features = None
        self.test_features = None
        self.train_labels = None

    def load_data(self):
        """加载训练和测试数据"""
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        print(f"训练数据形状: {self.train_data.shape}")
        print(f"测试数据形状: {self.test_data.shape}")
        print(f"训练数据列名: {self.train_data.columns.tolist()}")

    def preprocess_data(self):
        """数据预处理，包括删除无关列、标准化和独热编码"""
        # 删除无关列
        columns_to_drop = ['Id', 'Address', 'Summary', 'Elementary School',
                           'Middle School', 'High School']
        self.train_data = self.train_data.drop(columns=columns_to_drop)
        self.test_data = self.test_data.drop(columns=columns_to_drop)

        # 合并训练和测试数据
        self.all_features = pd.concat((self.train_data, self.test_data))

        # 标准化数值特征
        numeric_features = self.all_features.dtypes[self.all_features.dtypes != 'object'].index
        self.all_features[numeric_features] = self.all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        self.all_features[numeric_features] = self.all_features[numeric_features].fillna(0)

        # 处理需要拆分的列
        column_to_split = ['Cooling', 'Parking', 'Flooring', 'Heating features',
                           'Cooling features', 'Appliances included', 'Laundry features', 'Parking features']
        for col in column_to_split:
            if col in self.all_features.columns:
                # 拆分列内容并生成独热编码
                split_columns = self.all_features[col].str.get_dummies(sep=',')
                self.all_features = pd.concat([self.all_features, split_columns], axis=1)
                self.all_features = self.all_features.drop(columns=[col])

        # 对所有列进行独热编码
        self.all_features = pd.get_dummies(self.all_features, dummy_na=True)
        print(f"预处理后数据形状: {self.all_features.shape}")

    def generate_tensors(self):
        """生成训练和测试特征及标签的张量"""
        n_train = self.train_data.shape[0]
        self.train_features = torch.tensor(self.all_features[:n_train].values, dtype=torch.float32).to(device)
        self.test_features = torch.tensor(self.all_features[n_train:].values, dtype=torch.float32).to(device)
        self.train_labels = torch.tensor(
            self.train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32).to(device)

        print(f"训练特征形状: {self.train_features.shape}, 训练标签形状: {self.train_labels.shape}, 测试特征形状: {self.test_features.shape}")

    def get_data(self):
        self.load_data()
        self.preprocess_data()
        self.generate_tensors()
        return self.train_features, self.train_labels, self.test_features

def get_net(in_features):
    net = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1)
    )
    return net.to(device)

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train.to(device), y_train.to(device), X_valid.to(device), y_valid.to(device)

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    in_features = train_features.shape[1]
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(in_features)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

if __name__ == '__main__':
    '''
    train_features, train_labels, test_features = DataLoad('./data/train.csv', './data/test.csv').get_data()
    torch.save(train_features, './data/train_features.pt')
    torch.save(train_labels, './data/train_labels.pt')
    torch.save(test_features, './data/test_features.pt')
    '''
    train_features = torch.load('./data/train_features.pt').to(device)
    train_labels = torch.load('./data/train_labels.pt').to(device)
    test_features = torch.load('./data/test_features.pt').to(device)
    print(f"训练特征形状: {train_features.shape}, 训练标签形状: {train_labels.shape}, 测试特征形状: {test_features.shape}")
    loss = nn.MSELoss()

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                              weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
          f'平均验证log rmse: {float(valid_l):f}')

