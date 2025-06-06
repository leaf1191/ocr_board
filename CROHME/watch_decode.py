import pickle
from pix2tex.dataset.dataset import Im2LatexDataset

# 🔽 pkl 파일 경로
pkl_path = "C:/Users/SS/PycharmProjects/CvTermproject/CROHME/train3.pkl"

# 🔽 pkl 로드
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"총 {len(data)}개 배치 있음")

# 🔽 첫 5개 배치 출력
for i, (tok, img) in zip(range(5), iter(data)):
    print(f"\n샘플 {i + 1}:")
    print(f"  수식 토큰 길이: {tok['input_ids'].shape}")  # (배치크기, 시퀀스길이)
    print(f"  이미지 텐서 크기: {img.shape}")  # (배치크기, 채널, 높이, 너비)

    # 디코딩 및 후처리
    decoded = [data.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
               for ids in tok['input_ids']]
    cleaned = decoded[0].replace("Ġ", " ").replace("~", " ").strip()
    print(f"  디코딩된 수식: {cleaned}")
