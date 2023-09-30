import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Предварительная обработка данных и загрузка
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()])

train_data = datasets.ImageFolder(".\Classes", transform=transform)
test_data = datasets.ImageFolder(".\Train", transform=transform)

#train_data = datasets.ImageFolder('.\Classes', transform=transform)
#test_data = DataLoader(train_data, batch_size=32, shuffle=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

image_paths = [sample[0] for sample in train_data.imgs]

# Создание сверточной нейронной сети (CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        # Вычислите размерность после пулинга на основе размера изображения и операции MaxPool2d
        # 32 - количество фильтров в сверточном слое, 30x30 - размер данных после свертки и пулинга
        # При условии, что размер изображения после изменения (transforms.Resize) составляет 128x128

        # Определение линейных слоев
        self.fc2 = nn.Linear(128, 2)
        if self.fc2 == 3969:
            self.fc1 = nn.Linear(3969, 128)  # 32 channels, 30x30 output size after conv and pool
        else:
            self.fc1 = nn.Linear(127008, 128)  # 32 channels, 30x30 output size after conv and pool
        #self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Обучение модели
for epoch in range(1):  # Пример: 1000 эпох
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
        image_path = [sample[0] for sample in train_data.imgs]  # Путь к изображению
        if prediction == 0:  # Предсказание: качественное
            shutil.copy(image_path, 'Useful')
        else:  # Предсказание: некачественное
            shutil.copy(image_path, 'Trash')

# Оценка модели на тестовом наборе данных
correct = 0
total = 0
predictions = []
# Создайте список путей к изображениям на основе предсказаний
image_paths_to_move = [str(test_data.dataset.samples[i][0]) for i, prediction in enumerate(predictions)]

with torch.no_grad():
    for i, data in enumerate(test_data):  # Используем enumerate, чтобы получить индекс
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Точность на тестовом наборе данных: {100 * correct / total}%')

# Переместите изображения на основе предсказаний
move_images(image_paths_to_move)