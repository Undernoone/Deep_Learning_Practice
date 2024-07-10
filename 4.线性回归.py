import torch



input_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]]).to(device)
target_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]]).to(device)

# 定义线性模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

# 实例化模型并将其移动到选择的设备
model = LinearModel().to(device)

# 定义损失函数和优化器
criterion = torch.nn.MSELoss(reduction='sum')  # Pytorch 版本更新，使用 reduction 代替 size_average
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(1000000):  # 将训练次数调整为合理值
    # 前向传播
    y_pred = model(input_data)

    # 计算损失
    loss = criterion(y_pred, target_data)
    print(f'第 {epoch} 次迭代: 损失 {loss.item()}')

    # 清零梯度，进行反向传播并更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 打印学习到的权重和偏置
print("w =", model.linear.weight.item())
print("b =", model.linear.bias.item())

# 用新的数据测试模型
x_test = torch.tensor([[4.0]]).to(device)
y_test = model(x_test)
print('预测值 = ', y_test.data)

# 确认运行环境
print(f'当前计算是在 {"GPU" if device.type == "cuda" else "CPU"} 上进行的。')
