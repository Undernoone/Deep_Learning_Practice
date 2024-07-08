import numpy as np
import matplotlib.pyplot as plt

input_data = [1.0, 2.0, 3.0]
target_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w # 预测函数`

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# 记录权重和均方误差列表
w_list = []
mse_list = []

for w in np.arange(0.0, 6.1, 0.1): # 遍历权重，步长为0.1，范围为0-4
    print('w=',w)
    total_loss = 0
    for x_val, y_val in zip(input_data, target_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        total_loss += loss_val
        print('\t',x_val,y_pred_val,loss_val)
    print('MSE=',total_loss/3)
    w_list.append(w)
    mse_list.append(total_loss/3)

plt.plot(w_list, mse_list)
plt.xlabel('Weights (w)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Weight vs. Mean Squared Error')
plt.show()


# # 使用当前权重 w 和偏置 b 预测输出的函数
# def forward(x):
#     return x * w + b  # 增加偏置 b
#
# # 计算预测值与实际值之间平方误差的函数
# def loss(x, y):
#     pred = forward(x)  # 计算预测值 pred
#     return (pred - y) ** 2  # 返回平方误差
#
# # 记录权重、偏置和对应的均方误差（MSE）的列表
# w_list = []
# b_list = []
# mse_list = []
#
# # 遍历权重 w 从 0.0 到 4.0，步长为 0.1
# for w in np.arange(0.0, 4.1, 0.1):
#     # 遍历偏置 b 从 -2.0 到 2.0，步长为 0.1
#     for b in np.arange(-2.0, 2.1, 0.1):
#         print('w=', w, 'b=', b)
#         l_sum = 0  # 初始化当前权重和偏置的损失和
#         for x_val, y_val in zip(x_data, y_data):  # 遍历每个数据点
#             pred_val = forward(x_val)  # 计算预测值
#             loss_val = loss(x_val, y_val)  # 计算当前数据点的损失值
#             l_sum += loss_val  # 累加损失
#             print('\t', x_val, pred_val, loss_val)
#         # 计算当前权重和偏置下的平均均方误差（MSE）
#         mse = l_sum / len(x_data)
#         print('MSE=', mse)
#         # 将当前权重、偏置和 MSE 记录下来
#         w_list.append(w)
#         b_list.append(b)
#         mse_list.append(mse)
#
# # 绘制权重和偏置的 3D 图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(w_list, b_list, mse_list, c='r', marker='o')
# ax.set_xlabel('Weight (w)')
# ax.set_ylabel('Bias (b)')
# ax.set_zlabel('Mean Squared Error (MSE)')
# plt.title('Weight, Bias vs. Mean Squared Error')
# plt.show()
