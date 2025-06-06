from tokenizers import Tokenizer

# 저장된 tokenizer 불러오기
tokenizer_path = "C:/Users/SS/PycharmProjects/CvTermproject/CROHME/tokenizerfix.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

# 🔍 1. 특정 토큰이 vocab에 있는지 확인
def check_token(token):
    try:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            print(f"✅ 토큰 '{token}' 은 vocab에 있음 (ID: {token_id})")
        else:
            print(f"❌ 토큰 '{token}' 은 vocab에 없음")
    except Exception as e:
        print(f"오류 발생: {e}")

check_token("v")
check_token("(")
check_token("x")
check_token("v(x)")  # 공백 없이도 체크해보자
check_token("[UNK]")  # 항상 포함되어야 함

# 📋 2. 전체 vocab 일부 출력 (예: 처음 20개)
print("\n📚 vocab 일부:")
vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])  # ID 기준 정렬
for token, token_id in sorted_vocab[:20]:
    print(f"{token_id}: {token}")
