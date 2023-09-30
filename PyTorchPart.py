import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Определение модели нейронной сети
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Linear(32 * 8 * 8, 2)  # 2 класса: качественные и некачественные фотографии

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Загрузка и предобработка данных
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder('путь_к_папке_с_обучающими_изображениями', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Обучение модели
model = ImageClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Эпоха [{epoch+1}/{num_epochs}], Потеря: {loss.item()}')

# Теперь модель обучена. Можно использовать ее для классификации новых изображений.
# Например, для классификации изображения 'image.jpg':

test_image = Image.open('image.jpg')
test_image = transform(test_image).unsqueeze(0)
test_image = test_image.to(device)

model.eval()
with torch.no_grad():
    outputs = model(test_image)
    _, predicted = torch.max(outputs.data, 1)

if predicted.item() == 0:
    print('Изображение является качественным.')
else:
    print('Изображение является некачественным.')