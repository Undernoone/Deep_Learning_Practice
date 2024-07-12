import torch

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