import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 전처리 설정
transform = transforms.Compose([
    transforms.Grayscale(),               # 흑백 이미지로 처리
    transforms.Resize((224, 224)),        # ResNet 입력 사이즈
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터셋 로딩
train_dataset = datasets.ImageFolder('your_dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 분류 클래스 개수
num_classes = len(train_dataset.classes)

# 모델 설정 (ResNet18)
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 채널수 변경
model.fc = nn.Linear(model.fc.in_features, num_classes)

# GPU 사용 여부
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 손실 함수와 최적화기
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {correct/total:.4f}')

# 학습 끝난 뒤 저장
torch.save(model.state_dict(), 'saved_model5.pth')
