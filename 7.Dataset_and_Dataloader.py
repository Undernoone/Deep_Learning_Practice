import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

file_path = r'D:\Anoconda\pkgs\scikit-learn-1.2.2-py311hd77b12b_1\Lib\site-packages\sklearn\datasets\data\diabetes_data.csv'

class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        xy = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
        self.len = xy.shape[0] # 取数据的行数
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.y_data = (self.y_data > 0).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset(file_path)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)  # Adjust input dimension to 9
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(10):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print("epoch:", epoch, "i:", i, "loss:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


