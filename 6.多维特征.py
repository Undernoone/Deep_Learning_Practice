import torch
import numpy as np
import matplotlib.pyplot as plt

# 多维特征的处理
# 例如二手车的价格预测受很多因素影响，包括品牌、型号、年份、驾驶情况、使用情况等等
# 每个因素看做一个维度，可以用一个10维的向量来表示
file_path = r'D:\Anaconda\pkgs\scikit-learn-1.2.2-py311hd77b12b_1\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv'
xy = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1]) # 不取最后一列
y_data = torch.from_numpy(xy[:, [-1]]) # 只取最后一列

test = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
test_x = torch.from_numpy(test[:, :-1]) # 取前9列

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 因为数据集文件一行中有10个数据，不取最后一列，所以输入层有9个神经元
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch_list = []
loss_list = []

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print('Epoch:', epoch,'loss:',loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# y_pred = model(x_data)
# print(y_pred.detach().numpy())

y_pred2 = model(test_x)
print(y_pred2.detach().numpy())

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()