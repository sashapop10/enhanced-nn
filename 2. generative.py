import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

"""
Определяем, доступны ли какие-либо графические процессоры
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Сетевые архитектуры
Ниже приведены архитектуры дискриминатора и генератора
"""


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)


"""
Процедура обучения сети.
Каждый шаг потери обновляется как для дискиминатора, так и для генератора.
Дискриминатор стремится классифицировать реальные и fakes
Генератор стремится генерировать как можно более реалистичные изображения
"""

epochs = 150
lr = 2e-4
batch_size = 64
loss = nn.BCELoss()

G = generator().to(device)
D = discriminator().to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


"""
Image transformation and dataloader creation
Note that we are training generation and not classification, and hence
only the train_loader is loaded
"""
# Transform
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
# Load data
train_set = datasets.MNIST("mnist/", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for idx, (imgs, _) in enumerate(train_loader):
        idx += 1
        # Обучаем дискриминатор
        # real_inputs - изображения из набора данных MNIST
        # fake_inputs - изображения от генератора
        # real_inputs должны быть классифицированы как 1, а fake_inputs - как 0
        real_inputs = imgs.to(device)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)
        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)
        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        # Обучаем генератор
        # Цель генератора получить от дискриминатора 1 по всем изображениям
        noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        if idx % 100 == 0 or idx == len(train_loader):
            print(
                "Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}".format(
                    epoch, idx, D_loss.item(), G_loss.item()
                )
            )
    if (epoch + 1) % 10 == 0:
        torch.save(G, "Generator_epoch_{}.pth".format(epoch))
        print("Model saved.")


"""
Epoch 0 Iteration 100: discriminator_loss 0.685 generator_loss 0.874
Epoch 0 Iteration 200: discriminator_loss 0.661 generator_loss 0.786
Epoch 0 Iteration 300: discriminator_loss 0.665 generator_loss 0.733
Epoch 0 Iteration 400: discriminator_loss 0.636 generator_loss 0.761
Epoch 0 Iteration 500: discriminator_loss 0.531 generator_loss 0.958
Epoch 0 Iteration 600: discriminator_loss 0.526 generator_loss 0.875
Epoch 0 Iteration 700: discriminator_loss 0.506 generator_loss 0.938
Epoch 0 Iteration 800: discriminator_loss 0.531 generator_loss 0.911
Epoch 0 Iteration 900: discriminator_loss 0.435 generator_loss 1.137
Epoch 0 Iteration 938: discriminator_loss 0.424 generator_loss 1.216
Epoch 1 Iteration 100: discriminator_loss 0.419 generator_loss 1.260
...
Epoch 9 Iteration 700: discriminator_loss 0.593 generator_loss 0.845
Epoch 9 Iteration 800: discriminator_loss 0.659 generator_loss 0.757
Epoch 9 Iteration 900: discriminator_loss 0.667 generator_loss 0.638
Epoch 9 Iteration 938: discriminator_loss 0.688 generator_loss 0.704
Model saved.
Epoch 10 Iteration 100: discriminator_loss 0.602 generator_loss 0.840
Epoch 10 Iteration 200: discriminator_loss 0.609 generator_loss 1.112
Epoch 10 Iteration 300: discriminator_loss 0.573 generator_loss 0.935
"""
