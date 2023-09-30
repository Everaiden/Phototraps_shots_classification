import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import torchvision.transforms.functional as F

# Предварительная обработка данных и загрузка
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()])

train_data = datasets.ImageFolder("E:\Python\Phototraps_shots_classification\Classes", transform=transform)
test_data = datasets.ImageFolder("E:\Python\Phototraps_shots_classification\Train", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Создание сверточной нейронной сети (CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 2)  # Два класса: качественные и некачественные

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Обучение модели
for epoch in range(1000):  # Пример: 1000 эпох
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

print('Обучение завершено')

# Создайте папки "Useful" и "Trash" если их нет
if not os.path.exists('Useful'):
    os.makedirs('Useful')
if not os.path.exists('Trash'):
    os.makedirs('Trash')

# Функция для переноса изображений в соответствующие папки
def move_images(predictions):
    for i, prediction in enumerate(predictions):
        image_path = test_data.samples[i][0]  # Путь к изображению
        if prediction == 0:  # Предсказание: качественное
            shutil.copy(image_path, 'Useful')
        else:  # Предсказание: некачественное
            shutil.copy(image_path, 'Trash')

# Оценка модели на тестовом наборе данных
correct = 0
total = 0
predictions = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.tolist())

print(f'Точность на тестовом наборе данных: {100 * correct / total}%')

# Перемещение изображений в соответствующие папки
move_images(predictions)