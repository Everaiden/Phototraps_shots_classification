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
import zipfile

## Замените 'имя_архива.zip' на фактическое имя вашего архива.
#архив = zipfile.ZipFile('имя_архива.zip', 'r')

## Указываете папку, в которую хотите разархивировать содержимое.
#папка_разархивации = 'путь_к_папке_разархивации'

## Создайте папку, если она не существует
#if not os.path.exists(папка_разархивации):
#    os.makedirs(папка_разархивации)

## Извлекаем все файлы из архива
#архив.extractall(папка_разархивации)

## Закрываем архив
#архив.close()

# Предварительная обработка данных и загрузка
transform = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor()])

train_data = datasets.ImageFolder('.\Classes', transform=transform)
test_data = DataLoader(train_data, batch_size=32, shuffle=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

image_paths = [sample[0] for sample in train_data.imgs]

# Создание сверточной нейронной сети (CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)

        # Определение линейных слоев
        self.fc2 = nn.Linear(128, 2)
        if self.fc2 == 3969:
            self.fc1 = nn.Linear(3969, 128)
        else:
            self.fc1 = nn.Linear(127008, 128)


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
for epoch in range(1):
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

# Сохранение модели
torch.save(net.state_dict(), 'Model.pth')
print('Модель сохранена в файл модель.pth')
# Загрузка модели
#loaded_net = Net()
#loaded_net.load_state_dict(torch.load('модель.pth'))
#loaded_net.eval()  # Убедитесь, что модель находится в режиме оценки, если вы не планируете продолжать обучение

# Создайте папки "Useful" и "Trash" если их нет
if not os.path.exists('Useful'):
    os.makedirs('Useful')
if not os.path.exists('Trash'):
    os.makedirs('Trash')

# Функция для переноса изображений в соответствующие папки
def move_images(predictions, image_paths):
    for prediction, image_path in zip(predictions, image_paths):
        if prediction == 0:  # Предсказание: качественное
            shutil.copy(image_path, 'Useful')
        else:  # Предсказание: некачественное
            shutil.copy(image_path, 'Trash')

# Оценка модели на тестовом наборе данных
correct = 0
total = 0
predictions = []  # Список для хранения предсказаний (0 - качественное, 1 - некачественное)
image_paths_to_move = []  # Список для хранения путей к изображениям
# Создайте список путей к изображениям на основе предсказаний
#image_paths_to_move = [str(test_data.dataset.samples[i][0]) for i, prediction in enumerate(predictions)]

with torch.no_grad():
    for image_path, label in train_data.samples:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Преобразование и добавление размерности батча
        label = torch.tensor([label])  # Преобразование в тензор
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.tolist())  # Добавление предсказаний в список
        image_paths_to_move.append(image_path)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print(f'Точность на тестовом наборе данных: {100 * correct / total}%')

# Переместите изображения на основе предсказаний
move_images(predictions, image_paths_to_move)