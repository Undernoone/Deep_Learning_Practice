# import torch
#
# input_data = torch.Tensor([[1.0], [2.0], [3.0]])
# target_data = torch.Tensor([[0], [0], [1]])
#
# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(1,1)
#
#     def forward(self, x):
#         y_pred = torch.sigmoid(self.linear(x))
#         return y_pred
#
# model = LogisticRegressionModel()
#
# criterion = torch.nn.BCELoss(size_average=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# for epoch in range(1000):
#     y_pred = model(input_data)
#     loss = criterion(y_pred, target_data)
#     print("epoch:",epoch, "loss:",loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w = ', model.linear.weight.item())
# print('b = ', model.linear.bias.item())
#
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred = ', y_test.data)

import math
import torch
pred = torch.tensor([[-0.2],[0.2],[0.8]])
target = torch.tensor([[0.0],[0.0],[1.0]])

sigmoid = torch.nn.Sigmoid()
pred_s = sigmoid(pred)
print(pred_s)
result = 0
i = 0
for label in target:
    if label.item() == 0:
        result += math.log(1-pred_s[i].item())
    else:
        result += math.log(pred_s[i].item())
    i += 1
result /= 3
print("bce：", -result)
loss = torch.nn.BCELoss()
print('BCELoss:',loss(pred_s,target).item())

