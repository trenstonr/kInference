import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import struct

import os

class ThreeLayerNet(nn.Module):
    def __init__(self):
        super(ThreeLayerNet, self).__init__()
        # Input: 3072 (32x32x3 flattened)
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 3072)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def save_binary(filename, array):
    array = array.astype(np.float32)
    with open(filename, 'wb') as f:
        f.write(struct.pack('i', len(array.shape)))
        for s in array.shape:
            f.write(struct.pack('i', s))
        f.write(array.tobytes())


torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

print("loading CIFAR-10...")
train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

model = ThreeLayerNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("training 3-layer network...")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print(f"batch {batch_idx}, loss: {loss.item():.4f}")

print("testing...")
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        correct += output.argmax(1).eq(target).sum().item()

print(f"accuracy: {100. * correct / len(test_data):.2f}%")


os.makedirs('exported_data_cifar10', exist_ok=True)

print("exporting weights and biases...")
# Export layer 1
save_binary('exported_data_cifar10/weights_fc1_weight.bin', model.fc1.weight.data.numpy().T)
save_binary('exported_data_cifar10/weights_fc1_bias.bin', model.fc1.bias.data.numpy())

# Export layer 2
save_binary('exported_data_cifar10/weights_fc2_weight.bin', model.fc2.weight.data.numpy().T)
save_binary('exported_data_cifar10/weights_fc2_bias.bin', model.fc2.bias.data.numpy())

# Export layer 3
save_binary('exported_data_cifar10/weights_fc3_weight.bin', model.fc3.weight.data.numpy().T)
save_binary('exported_data_cifar10/weights_fc3_bias.bin', model.fc3.bias.data.numpy())

print("exporting test images and expected outputs...")
images, labels = next(iter(test_loader))
for i in range(10):
    img = images[i:i+1].numpy().reshape(1, 3072)
    save_binary(f'exported_data_cifar10/test_image_{i}.bin', img)
    with torch.no_grad():
        out = model(images[i:i+1])
        # Apply softmax to convert logits to probabilities
        out = F.softmax(out, dim=1)
    save_binary(f'exported_data_cifar10/expected_output_{i}.bin', out.numpy())

print("done!")
print("\nExported files:")
print("  - weights_fc1_weight.bin, weights_fc1_bias.bin")
print("  - weights_fc2_weight.bin, weights_fc2_bias.bin")
print("  - weights_fc3_weight.bin, weights_fc3_bias.bin")
print("  - test_image_0.bin to test_image_9.bin")
print("  - expected_output_0.bin to expected_output_9.bin")
