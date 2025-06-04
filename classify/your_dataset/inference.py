import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pix2tex.cli import LatexOCR

# ----------------------------
# 설정
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = 'your_dataset/test/0023.jpg'  # 예측할 이미지 경로
model_path = 'saved_model5.pth'
label_map_path = 'label_map.txt'

# ----------------------------
# 라벨 매핑 복원 (인덱스 → 수식, 카테고리)
# ----------------------------
id2label = {}
id2category = {}
with open(label_map_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            idx, formula, category = parts
            id2label[int(idx)] = formula
            id2category[int(idx)] = category

# ----------------------------
# 분류 모델 정의 및 불러오기
# ----------------------------
num_classes = len(id2label)
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 흑백 처리
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# 이미지 전처리
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# pix2tex 수식 전체 예측
# ----------------------------
ocr_model = LatexOCR()

def predict_latex_formula(image_path):
    img = Image.open(image_path).convert("RGB")
    return ocr_model(img)

# ----------------------------
# 분류 모델 예측
# ----------------------------
def predict_formula_class(img_path):
    img = Image.open(img_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_idx = output.argmax(1).item()
    return predicted_idx, id2label[predicted_idx], id2category[predicted_idx]

# ----------------------------
# 결과 출력
# ----------------------------
print(f"[이미지 파일]: {img_path}")

# 1. 수식 전체 예측 (OCR)
formula_pred = predict_latex_formula(img_path)
print(f"[수식 전체 예측 (pix2tex)]: {formula_pred}")

# 2. 분류 모델 예측
idx, label_formula, label_category = predict_formula_class(img_path)
print(f"[분류 수식]: {label_formula}")
print(f"[분류 카테고리]: {label_category}")
