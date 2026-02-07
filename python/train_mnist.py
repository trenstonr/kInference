import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import struct

import os

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("training...")
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 200 == 0:
        print(f"batch {batch_idx}, loss: {loss.item():.4f}")

print("testing...")
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        correct += output.argmax(1).eq(target).sum().item()

print(f"accuracy: {100. * correct / len(test_data):.2f}%")


os.makedirs('exported_data', exist_ok=True)

print("exporting...")
save_binary('exported_data/weights_fc1_weight.bin', model.fc1.weight.data.numpy().T)
save_binary('exported_data/weights_fc1_bias.bin', model.fc1.bias.data.numpy())
save_binary('exported_data/weights_fc2_weight.bin', model.fc2.weight.data.numpy().T)
save_binary('exported_data/weights_fc2_bias.bin', model.fc2.bias.data.numpy())

images, labels = next(iter(test_loader))
for i in range(10):
    img = images[i:i+1].numpy().reshape(1, 784)
    save_binary(f'exported_data/test_image_{i}.bin', img)
    with torch.no_grad():
        out = model(images[i:i+1])
    save_binary(f'exported_data/expected_output_{i}.bin', out.numpy())

print("done")
