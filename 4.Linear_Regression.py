import torch

input_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
input_dimension = 1
output_dimension = 1

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dimension, output_dimension, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

LinearModel = LinearModel()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(LinearModel.parameters(), lr=0.01)

for epoch in range(10000):
    y_pred = LinearModel(input_data)
    loss = criterion(y_pred, target_data)
    print(f'第 {epoch} 次迭代: 损失 {loss.item()}')
    optimizer.zero_grad() # 清空梯度
    loss.backward()
    optimizer.step() # 更新梯度

print("w =", LinearModel.linear.weight.item())
print("b =", LinearModel.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = LinearModel(x_test)
print('预测值 = ', y_test.data)


