import torch
# from torch import nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.optim as optim
#
# batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = datasets.MNIST('../Deeplearing_data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = datasets.MNIST('../Deeplearing_data', train=False, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.pooling = nn.MaxPool2d(2)
#         self.fc = nn.Linear(320, 10)
#
#     def forward(self, x):
#         batchsize = x.size(0)
#         x = F.relu(self.pooling(self.conv1(x)))
#         x = F.relu(self.pooling(self.conv2(x)))
#         x = x.view(batchsize, -1)
#         x = F.relu(self.fc(x))
#         return x
#
# model = Net()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#
# def train(epoch):
#     running_loss = 0.0
#     for batch_idx, data in enumerate(train_loader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if batch_idx % 300 == 299:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
#             running_loss = 0.0
#
# def test(epoch_val):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print('Accuracy on test set: %d %%' % (100 * correct / total))
#
# if __name__ == '__main__':
#     for epoch in range(10):
#         train(epoch)
#         test(epoch)
#
#

input_channels, output_channels = 5,10
width, height = 100,100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size, input_channels, width, height)

conv_layer = torch.nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size)
output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)