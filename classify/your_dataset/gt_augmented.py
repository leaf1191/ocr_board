import cv2
import os
from tqdm import tqdm
from pathlib import Path
import albumentations as A

# 설정
BASE_DIR = ''
IMG_ROOT = os.path.join(BASE_DIR, 'images')         # 모든 이미지가 저장될 디렉토리
GT_PATH = os.path.join(BASE_DIR, 'gt.txt')          # 원본 LaTeX 매핑 파일
OUT_GT_PATH = os.path.join(BASE_DIR, 'gt_augmented.txt')  # 저장될 새 GT 파일
REPEAT_N = 10  # 이미지 하나당 증강 횟수

# 증강기 정의
augment = A.Compose([
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=0.05, shear=5, p=0.5),
    A.GaussNoise(var_limit=(5, 20), p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.Resize(64, 256),
])

# 디렉터리 준비
os.makedirs(IMG_ROOT, exist_ok=True)
lines = open(GT_PATH, encoding='utf-8').readlines()
aug_lines = []

counter = 1

for line in tqdm(lines):
    line = line.strip()
    if not line or '\t' not in line:
        print(f"⚠️ 잘못된 줄 건너뜀: {repr(line)}")
        continue

    try:
        fname, formula = line.split('\t')
    except ValueError:
        print(f"⚠️ split 실패: {repr(line)}")
        continue

    img_path = os.path.join(IMG_ROOT, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ 이미지 로드 실패: {img_path}")
        continue

    # 원본 복사 저장
    new_name = f"{counter:04d}.png"
    save_path = os.path.join(IMG_ROOT, new_name)
    cv2.imwrite(save_path, img)
    aug_lines.append(f"{new_name}\t{formula}\n")
    counter += 1

    # 증강 이미지 저장
    for _ in range(REPEAT_N):
        aug_img = augment(image=img)['image']
        aug_name = f"{counter:04d}.png"
        aug_path = os.path.join(IMG_ROOT, aug_name)
        cv2.imwrite(aug_path, aug_img)
        aug_lines.append(f"{aug_name}\t{formula}\n")
        counter += 1

# gt_augmented.txt 저장
with open(OUT_GT_PATH, 'w', encoding='utf-8') as f:
    f.writelines(aug_lines)

print(f"\n✅ 총 {counter - 1}개의 이미지가 '{IMG_ROOT}'에 저장되었고,")
print(f"📄 전체 라벨은 '{OUT_GT_PATH}'에 저장되었습니다.")
