import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

xy = np.loadtxt(r'D:\Anoconda\pkgs\scikit-learn-1.2.2-py311hd77b12b_1\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

test = np.loadtxt(r'D:\Anoconda\pkgs\scikit-learn-1.2.2-py311hd77b12b_1\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv', delimiter=',', dtype=np.float32)
test_x = torch.from_numpy(test)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    #Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    #Backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()

y_pred = model(x_data)

print(y_pred.detach().numpy())

y_pred2 = model(test_x)
print(y_pred2.data.item())
