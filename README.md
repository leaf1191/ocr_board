# 수식 인식 칠판 OCR 프로젝트

이 프로젝트는 OpenCV를 활용하여 촬영된 칠판 사진에서 수식을 자동으로 인식하고 분류하는 시스템입니다. 이미지 전처리부터 수식 분류까지의 전체 파이프라인을 구현했습니다.

## 개요

- **목적**: 칠판에 쓰인 수식 이미지를 자동으로 인식하고 분류
- **주요 기능**:
  - 칠판 이미지에서 수식 영역 자동 검출
  - 이미지 전처리(잡음 제거, 이진화 등)
  - 수식 분류를 위한 딥러닝 모델 적용
  - 간단한 GUI 제공
  - 노션에 결과 저장

## 주요 기술 스택

- **이미지 처리**: OpenCV, scikit-image
- **딥러닝 프레임워크**: PyTorch, Hugging Face Transformers, 맞는 지 확인 점
- **GUI**: Tkinter
- **기타 라이브러리**: NumPy, Albumentations, OpenCV-Python

## 설치 방법

1. 저장소 클론:
   ```bash
   git clone [저장소 URL]
   cd ocr_board
   ```

2. Python 3.11 가상 환경 생성 및 활성화:
   - Windows:
     ```bash
     # Python 3.11 가상 환경 생성
     py -3.11 -m venv venv311
     
     # 가상 환경 활성화
     .\venv311\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     # Python 3.11 가상 환경 생성
     python3.11 -m venv venv311
     
     # 가상 환경 활성화
     source venv311/bin/activate
     ```

3. 의존성 설치:
   ```bash
   pip install -r requirements.txt
   ```

4. (선택사항) 가상 환경 비활성화:
   ```bash
   deactivate
   ```

## 노션(Notion) 연동 설정(이 내용 검증 좀)

1. **노션 API 토큰 발급**:
   - [Notion Developers](https://www.notion.so/my-integrations) 사이트에서 새 integration을 생성합니다.
   - 생성된 "Internal Integration Token"을 복사합니다.

2. **노션 페이지 공유**:
   - 결과를 저장할 노션 페이지를 엽니다.
   - 페이지 URL에서 페이지 ID를 복사합니다 (32자리 16진수).
   - 페이지 우측 상단의 `...` 메뉴에서 `Add connections`를 선택하여 생성한 integration을 연결합니다.

3. **환경 변수 설정**:
   [image_gui.py](cci:7://file:///c:/Users/jwlv1/OneDrive/%EB%B0%94%ED%83%95%20%ED%99%94%EB%A9%B4/ocr_board/image_gui.py:0:0-0:0) 파일을 열고 다음 부분을 수정하세요:
   ```python
   # Notion 정보
   NOTION_TOKEN = "여기에_발급받은_토큰_입력"  # 시크릿 노션 토큰 기입 (필수)
   NOTION_PAGE_ID = "여기에_페이지_ID_입력"    # 노션 사이트 ID 기입 (필수)

## 사용 방법

1. 애플리케이션 실행:
   ```bash
   python main.py
   ```

2. GUI가 실행되면 다음 단계를 따르세요:
   - `Alt+S`를 눌러 스크린샷 촬영 혹은 `Load Image` 버튼을 눌러 이미지 파일을 직접 선택
   - `Process Image` 버튼을 눌러 이미지 처리 및 수식 인식
   - `Copy Result` 버튼을 눌러 결과를 클립보드에 복사

## 프로젝트 구조

```
ocr_board/
├── CROHME/               # 학습 데이터 및 모델 관련 파일
├── demo_photos/          # 데모용 수식 이미지
├── __pycache__/
├── image_convert.py      # 이미지 변환 유틸리티
├── image_gui.py          # 그래픽 사용자 인터페이스
├── image_process.py      # 이미지 전처리 함수들
├── main.py               # 메인 애플리케이션 진입점
└── requirements.txt      # 의존성 목록
```

## 주요 기능 상세

### 이미지 전처리
- 그레이스케일 변환
- 양방향 필터링을 통한 잡음 제거
- CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
- 이진화 및 모폴로지 연산
- 스켈레토나이제이션(뼈대화)

### 모델 (알아서 추가)
- CROHME 데이터셋을 기반으로 한 사용자 정의 모델
- 수식 분류를 위한 딥러닝 아키텍처

## 실행 결과

![실행 결과](demo_result.png)
결과 사진으로 채우고
![실행 결과](demo_result.png)
밑에다 설명 쓰면 됨

## 한계점
위와 비슷하게 가면 될듯

## 참여자

- [심종우](https://github.com/leaf1191)
- [김지민](https://github.com/링크)


