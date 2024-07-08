import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def load_data(self):
        try:
            # 使用空格作为分隔符读取数据
            data = np.loadtxt(self.file_path, delimiter=' ', dtype=np.float32)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回一个样本
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return sample

# 创建数据集实例
dataset = DiabetesDataset(r'D:\Anoconda\pkgs\scikit-learn-1.2.2-py311hd77b12b_1\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv')

# 创建数据加载器
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

# 打印一些批次数据以确认加载成功
for batch in train_loader:
    print(batch)
    break  # 打印一个批次即可
