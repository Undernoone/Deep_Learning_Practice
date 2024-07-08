import numpy as np
import matplotlib.pyplot as plt
#
# input_data = [1.0, 2.0, 3.0]
# target_data = [2.0, 4.0, 6.0]
#
# w = 1.0
#
# def forward(x):
#     return x * w # 预测函数
#
# def cost(xs, ys):
#     cost = 0
#     for x,y in zip(xs, ys):
#         y_pred = forward(x)
#         cost += (y_pred - y)**2
#     return cost / len(xs)
#
# # 梯度函数
# def gradient(xs, ys):
#     grad = 0
#     for x,y in zip(xs, ys):
#         grad += 2 * x * (x * w - y)
#     return grad / len(xs)
#
# epoch_list = []
# cost_list = []
# print('Predict(before training)',4, forward(4))
# for epoch in range(100):
#     cost_val = cost(input_data, target_data)
#     grad_val = gradient(input_data, target_data)
#     w -= 0.01 * grad_val
#     print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
#     epoch_list.append(epoch)
#     cost_list.append(cost_val)
#
# print('Predict(after training)',4, forward(4))
# plt.plot(epoch_list, cost_list)
# plt.xlabel('Epoch')
# plt.ylabel('Cost')
# plt.title('Training Loss')
# plt.show()

input_data = [1.0, 2.0, 3.0]
target_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

def gradient(x,y):
    return 2 * x * (x * w - y)

epoch_list = []
loss_list = []
print('Predict(before training)',4, forward(4))
for epoch in range(100):
    for x,y in zip(input_data, target_data):
        grad = gradient(x,y)
        w = w-0.01*grad
        print("\tgrad:",x,y,grad)
        l = loss(x,y)
    print("progress:",epoch,"w=",w,"loss=",l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('Predict(after training)',4, forward(4))
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()