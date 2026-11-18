# Fetal-Ultrasound-UNet-Segmentation
PyTorch U-Net implementation for fetal head circumference ultrasound image segmentation (Dice score: 0.9423)
## 프로젝트명: U-Net 기반 태아 초음파 의료 영상 분할(Segmentation) 모델 구현

### 1. 개요
* **목적**: 태아 머리 둘레(HC18) 초음파 영상에서 핵심 영역(머리 윤곽)을 정밀하게 분할하는 딥러닝 모델 구현 및 성능 평가
* **데이터셋**: 총 999장의 초음파 이미지 및 정답 마스크(Ground Truth) 데이터 (Train 629장 / Validation 270장 / Test 100장)
* **개발 환경**: Python, PyTorch, OpenCV, NumPy, Google Colab (GPU T4)

### 2. 주요 수행 내용
* **데이터 전처리 및 파이프라인 구축**: PyTorch `Dataset` 및 `DataLoader`를 직접 정의하여 Grayscale 변환, `256x256` 리사이징, 픽셀 정규화(`/255.0`), 마스크 이진화(Threshold 0.5) 처리 수행
* **모델 아키텍처 설계**: 4단계 Encoder-Bottleneck-Decoder 구조와 Skip Connection을 결합한 U-Net 모델(총 3,104만 파라미터) 직접 구현
* **손실 함수 및 학습 최적화**: 클래스 불균형 보완을 위해 Binary Cross Entropy와 Dice Loss를 결합한 `DiceBCELoss` 정의 및 Adam Optimizer, `ReduceLROnPlateau` 스케줄러 적용

### 3. 정량적 평가 결과 (Test Set 100장 기준)
* **Dice Coefficient (F1 Score)**: **0.9423**
* **Jaccard Index (IoU)**: **0.8982**
* **Pixel Accuracy**: **97.53%**
* **추론 속도**: **130.62 FPS**
