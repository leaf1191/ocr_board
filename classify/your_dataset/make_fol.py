import os
import shutil

src_img_folder = 'images'
gt_file = 'gt_augmented.txt'
target_folder = 'your_dataset/train'

label2id = {}
label2category = {}  # 수식 라텍스 → 카테고리
id_counter = 0

def get_category(latex):
    if any(op in latex for op in ['\\le', '\\ge', '<', '>']):
        return '부등식'
    elif any(sym in latex for sym in ['\\in', '\\cup', '\\cap', '\\forall', '\\exists']):
        return '집합'
    elif any(sym in latex for sym in ['\\vec', '\\mathbf', '\\overrightarrow']):
        return '벡터'
    elif '\\sqrt' in latex:
        return '루트'
    elif '^' in latex:
        return '지수'
    elif '+' in latex or '-' in latex:
        return '덧셈/뺄셈'
    elif '*' in latex or '\\cdot' in latex or '\\times' in latex:
        return '곱셈'
    elif '\\frac' in latex or '/' in latex:
        return '분수'
    elif '\\sum' in latex or '\\prod' in latex:
        return '합/곱'
    elif '\\int' in latex:
        return '적분'
    elif '\\lim' in latex:
        return '극한'
    elif any(term in latex for term in ['\\frac{dy}{dx}', '\\frac{d^2y}{dx^2}', 'dy', 'dx', '\\partial']):
        return '미분'
    elif any(trig in latex for trig in ['\\tan', '\\sin', '\\cos']):
        return '삼각함수'
    elif '=' in latex or '\\equiv' in latex:
        return '방정식'
    elif '\\log' in latex or '\\ln' in latex:
        return '로그/지수함수'
    elif '|' in latex:
        return '절댓값'
    elif '\\binom' in latex:
        return '조합/이항계수'
    else:
        return '기타'


with open(gt_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or '\t' not in line:
            continue

        filename, label = line.split('\t')
        label = label.strip()

        if label not in label2id:
            category = get_category(label)
            label2category[label] = category
            label2id[label] = f'{id_counter:02d}_{category}'
            id_counter += 1

        folder_name = label2id[label]
        label_folder = os.path.join(target_folder, folder_name)
        os.makedirs(label_folder, exist_ok=True)

        src_path = os.path.join(src_img_folder, filename)
        dst_path = os.path.join(label_folder, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

# label_map.txt 저장: index, latex, category
with open('label_map.txt', 'w', encoding='utf-8') as f:
    for idx, (latex, foldername) in enumerate(label2id.items()):
        category = label2category[latex]
        f.write(f"{idx}\t{latex}\t{category}\n")
