import numpy as np
import torch
import matplotlib.pyplot as plt

input_data = [1.0, 2.0, 3.0]
target_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True # 声明需要计算梯度

def forward(x):
    return w * x

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

epoch_list = []
loss_list = []
print("predict before training:", 4, forward(4).item())
for epoch in range(30):
    for x,y in zip(input_data, target_data):
        loss_value = loss(x,y)
        loss_value.backward() # 执行反向传播，计算梯度
        print("\tx:",x,"y:",y,"loss:",loss_value.item())
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()

    epoch_list.append(epoch)
    loss_list.append(loss_value.item())
    print("process:",epoch,loss_value.item())

epoch_list_np = np.array(epoch_list)
loss_list_np = np.array(loss_list)

print("predict after training:", 4, forward(4).item())
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

