import os
import argparse
from pix2tex.dataset.dataset import Im2LatexDataset
from pix2tex.utils.utils import in_model_path


def create_dataset(images_dir, equation_file, tokenizer_file, output_file):
    # 경로 정리
    images_dir = os.path.abspath(images_dir)
    equation_file = os.path.abspath(equation_file)
    tokenizer_file = os.path.abspath(tokenizer_file)
    output_file = os.path.abspath(output_file)

    # tokenizer 파일 확인
    if not os.path.exists(tokenizer_file):
        with in_model_path():
            tokenizer_file = os.path.realpath(os.path.join('dataset', 'tokenizer.json'))

    # 데이터셋 생성
    dataset = Im2LatexDataset(
        equations=equation_file,
        images=images_dir,
        tokenizer=tokenizer_file,
        batchsize=1,
        keep_smaller_batches=True,
        pad=False,
        test=False
    )

    # 저장
    dataset.save(output_file)
    print(f"✅ 데이터셋 저장 완료: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pix2tex 학습용 .pkl 데이터셋 생성기")
    parser.add_argument("--images", type=str, required=True, help="이미지 폴더 경로")
    parser.add_argument("--equations", type=str, required=True, help="수식 텍스트(gt.txt) 파일 경로")
    parser.add_argument("--tokenizer", type=str, required=True, help="토크나이저 JSON 경로")
    parser.add_argument("--out", type=str, default="train.pkl", help="출력할 .pkl 파일 경로")

    args = parser.parse_args()
    create_dataset(args.images, args.equations, args.tokenizer, args.out)
