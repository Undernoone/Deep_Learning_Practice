import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# 记录权重和均方误差列表
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1): # 遍历权重，步长为0.1，范围为0-4
    print('w=',w)
    total_loss = 0
    for x_val, y_val in zip(x_data, y_data):
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