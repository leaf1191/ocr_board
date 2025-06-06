from pix2tex.models import get_model
from pix2tex.dataset.dataset import Im2LatexDataset
from pix2tex.utils import post_process, token2str
from PIL import Image
import torch
import numpy as np

# 1. config와 tokenizer 불러오기
from munch import Munch
import yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    args = Munch(yaml.safe_load(f))

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. 모델 로딩
model = get_model(args).to(args.device)
model.eval()

# 체크포인트 로딩 (수정 필요시)
model.load_state_dict(torch.load(r'C:\Users\SS\PycharmProjects\CvTermproject\LaTeX-OCR\checkpoints\my_new30\my_new30_e75_step49.pth', map_location=args.device))

# 3. 이미지 불러오기 및 전처리
img = Image.open('0001004.png').convert('L')

# 4. 토크나이저와 전처리기
dataset = Im2LatexDataset().load(args.data)
img_np = np.array(img)  # PIL 이미지를 numpy 배열로 변환
tensor_img = dataset.transform(image=img_np)["image"].unsqueeze(0).to(args.device)
# 5. 수식 예측
with torch.no_grad():
    output = model.generate(tensor_img, temperature=0.2)

    pred_latex = post_process(token2str(output, dataset.tokenizer)[0])

print("🧪 예측 수식:", pred_latex)
