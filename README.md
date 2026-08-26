# Driver Drowsiness Detection

웹캠 영상에서 운전자의 눈 감김과 하품을 실시간으로 분석해 졸음 가능성을 표시하는 프로젝트입니다.

얼굴 검출에는 YuNet, 눈과 하품 분류에는 CNN을 사용합니다. 눈 감김의 단기 변화, 최근 구간의 PERCLOS, 시간축으로 누적한 하품 점수를 결합해 `NORMAL`, `DROWSY`, `NO FACE` 상태를 화면에 표시합니다.

최종 코드는 [`Drowsiness-Detection-260824`](https://github.com/lcsvvo/Driver-Drowsiness-Detection/tree/Drowsiness-Detection-260824) 브랜치에 있습니다. 아래 구조와 실행 방법도 이 브랜치를 기준으로 합니다.

> 이 프로젝트는 실험용 프로토타입입니다. 실제 차량의 안전 장치나 의료 진단 도구로 사용할 수 없습니다.

## 프로젝트 소개

최종 구현 범위는 다음과 같습니다.

- 웹캠 영상에서 가장 큰 얼굴 1개 검출
- 좌·우 눈 crop의 감김 확률 추론
- 최근 60초 기준 PERCLOS 계산
- 입 벌림 게이트, CNN, 시간 누적을 이용한 하품 판정
- 눈 감김과 하품 신호를 결합한 졸음 점수 표시
- 세션 종료 후 눈 감김, PERCLOS, 하품, 졸음 경고 횟수 요약

머리 방향, 휴대전화 사용, 안전벨트, 주의 산만 분류는 현재 구현에 포함되지 않습니다.

## 주요 기능

### 눈 감김 감지

YuNet이 제공하는 눈 좌표를 기준으로 좌·우 눈을 잘라 128×128 그레이스케일로 전처리합니다. Eye CNN의 두 출력 중 `class 0`을 눈 감김 확률로 사용하며, 좌·우 눈 중 높은 값을 선택합니다.

선택된 모델의 판정 임계값은 `p(Closed) >= 0.93`입니다. 실시간 루프에서는 이 임계값이 내부 점수 `0.5`가 되도록 변환한 뒤 EMA와 PERCLOS 계산에 사용합니다.

### PERCLOS

PERCLOS는 최근 시간 창에서 얼굴이 검출된 시간 중 눈이 감겨 있던 시간의 비율입니다.

```text
PERCLOS = 눈이 감긴 시간 / 얼굴이 검출된 시간
```

기본 창은 60초, 경고 기준은 0.15입니다. 처음 10초 동안은 준비 구간이며, 얼굴 검출 비율이 30%보다 낮으면 결합 점수에서 제외합니다.

### 하품 감지

하품은 한 프레임의 CNN 결과만으로 판정하지 않습니다.

1. MediaPipe FaceLandmarker로 입 벌림 정도를 계산합니다.
2. `open_ratio > 0.05`일 때만 Yawn CNN을 실행합니다.
3. CNN의 `p(Yawn) >= 0.50`인 증거를 시간축으로 누적합니다.
4. 누적값과 최근 3초의 최대 입 벌림 정도로 하나의 하품 이벤트를 판정합니다.

입 벌림 정도는 다음과 같이 계산합니다.

```text
open_ratio = 입술 안쪽 세로 거리(13, 14) / 양쪽 눈 바깥점 거리(33, 263)
```

시간 누적은 프레임 수가 아니라 실제 경과 시간을 사용합니다. 하품 상태가 `False`에서 `True`로 바뀌는 순간만 1회로 집계합니다.

### 최종 졸음 점수

눈의 EMA와 PERCLOS는 같은 눈 신호이므로 둘 중 큰 값을 사용합니다. 여기에 독립 신호인 하품 누적값을 결합합니다.

```text
eye_family = max(Eye EMA, normalized PERCLOS)
drowsy = eye_family + (1 - eye_family) × 0.6 × Yawn
```

기본 졸음 판정 임계값은 `0.6`입니다.

## 시스템 구조

```text
Webcam frame
    ↓
YuNet face detection
    ├─ Eye crops → Eye CNN → Eye EMA ─┐
    │                                ├─ Drowsiness score → NORMAL / DROWSY
    │              PERCLOS ──────────┤
    └─ Face crop                     │
         ↓                           │
       FaceLandmarker                │
         ↓                           │
       MouthGate → Yawn CNN → YawnAccumulator ─┘
```

실시간 실행을 종료하면 측정 시간, 의심 눈 감김 횟수, 최대 눈 감김 시간, 세션 PERCLOS, 하품 횟수, 졸음 경고 횟수를 `DRIVER MONITORING REPORT`로 출력합니다. 영상이나 얼굴 이미지는 저장하지 않습니다.

## 프로젝트 구조

```text
Driver-Drowsiness-Detection/
├── README.md
├── config.py                         # 공통 경로 설정
├── requirements.txt                 # 실행 패키지
├── model/
│   ├── 02_INFER_YuNet.ipynb          # 실시간 실행 진입점
│   ├── detectors/
│   │   └── face_detection_yunet_2023mar.onnx
│   └── artifacts/
│       ├── README.md                 # 가중치와 평가 결과 설명
│       └── *_metrics.json            # 모델별 입력 규격과 평가 지표
├── src/
│   ├── eye_preprocess.py             # 눈·얼굴 crop 공통 전처리
│   ├── mouth_gate.py                 # 입 벌림 측정과 게이트
│   ├── yawn_accumulator.py            # 하품 시간 누적
│   ├── yawn_dataset.py               # 하품 데이터 로딩과 추론 crop
│   ├── train_eye.py                  # Eye CNN 학습·평가
│   ├── train_yawn.py                 # 기본 Yawn CNN 학습·평가
│   ├── train_yawn_zoo.py             # 하품 모델 비교 학습
│   └── distill_yawn.py               # 지식 증류 실험
├── scripts/                          # DMD·YawDD 하품 데이터 전처리
├── outputs/eye_dataset/
│   ├── eye_manifest.csv              # Eye CNN 통합 학습 manifest
│   ├── class_balance.csv             # source·split별 클래스 분포
│   ├── leakage_report.csv            # split 누수 점검 결과
│   └── step12_comparison.csv          # Eye CNN 실험 비교 요약
└── docs/yawn_model.md                # 하품 모델 실험 기록
```

`data/`와 대부분의 `outputs/`는 Git에서 제외됩니다. Eye CNN 재학습에 필요한 통합 manifest와 검증 요약만 `outputs/eye_dataset/`에 포함합니다. 원본 데이터셋, crop 이미지, 학습된 `.keras` 가중치는 저장소에 포함되지 않습니다.

## 모델 및 데이터

### 최종 추론 모델

| 용도 | 파일 | 입력 | 출력 | 판정 기준 |
|---|---|---|---|---|
| 얼굴 검출 | `face_detection_yunet_2023mar.onnx` | BGR 프레임 | 얼굴 박스와 주요 좌표 | score 0.6 |
| 눈 개폐 | `eye_mrl+dmd__eval-dmd__gray128.keras` | 128×128 gray | `Closed`, `Open` | `p(Closed) >= 0.93` |
| 하품 | `yawn_yawn_mouthopen_v2__zoo-cnn_large__eval-face__gray128.keras` | 128×128 gray 얼굴 crop | `yawn`, `no_yawn` | 게이트 0.05, `p(Yawn) >= 0.50` |
| 입 랜드마크 | `face_landmarker.task` | 얼굴 crop | 478개 얼굴 랜드마크 | `open_ratio > 0.05` |

Eye CNN은 MRL Eye와 DMD를 합쳐 학습했습니다. Yawn CNN은 DMD와 YawDD에서 입이 열린 프레임을 구성해 학습한 `cnn_large` 모델입니다. 지식 증류와 다른 model-zoo 결과도 저장되어 있지만 최종 실시간 추론에는 사용하지 않습니다.

### 저장된 평가 결과

| 모델과 평가 조건 | Accuracy | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| Eye CNN, DMD hold-out 3,805프레임, 임계값 0.93 | 0.9624 | 0.9138 | 0.7871 | 0.8457 |
| Yawn CNN, 입 벌림 게이트 통과 test 1,553장, 임계값 0.50 | 0.7521 | 0.6899 | 0.8118 | 0.7459 |
| Yawn end-to-end, DMD test 1,131장, 게이트 포함 | 0.779 | 0.518 | 0.993 | - |

평가 단위와 데이터 구성이 서로 다르므로 세 행을 직접 비교하면 안 됩니다. 하품 end-to-end 평가는 입을 다문 프레임과 손으로 입을 가린 하품까지 포함합니다.

### 데이터와 재학습 범위

원본 DMD, MRL Eye, YawDD 데이터는 라이선스와 용량 문제로 저장소에 포함하지 않습니다.

코드에 남아 있는 흐름은 다음과 같습니다.

```text
DMD / MRL Eye / YawDD 원본
    ↓
전처리 및 subject 단위 split
    ↓
crop 이미지와 manifest 생성
    ↓
Eye CNN / Yawn CNN 학습과 평가
    ↓
.keras + _metrics.json 생성
    ↓
02_INFER_YuNet.ipynb에서 로드
```

하품 데이터는 `prepare_yawdd.py`, `build_yawdd_yawn_dataset.py`, `build_dmd_yawn_dataset.py`, `build_yawn_mouthopen_dataset.py` 순서의 전처리 도구가 남아 있습니다.

눈 학습의 최종 스크립트인 `src/train_eye.py`는 저장소에 포함된 `outputs/eye_dataset/eye_manifest.csv`를 입력으로 사용합니다. manifest의 `path`는 저장소 루트 기준 상대경로이므로, DMD와 MRL Eye 원본 및 crop을 같은 경로에 준비해야 합니다. 통합 manifest를 새로 만드는 과거 노트북은 최종 저장소에 포함되어 있지 않으므로, manifest 생성부터 시작하는 전체 전처리 과정은 저장소만으로 재현할 수 없습니다. `src/train_eye_mrl.py`도 현재 없는 과거 manifest 경로를 참조하므로 최종 실행 경로가 아닙니다.

## 설치 방법

### 1. 저장소와 가상환경

```bash
git clone --branch Drowsiness-Detection-260824 --single-branch https://github.com/lcsvvo/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
python -m venv .venv
```

가상환경을 활성화합니다.

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`에는 YuNet, CNN 추론, 노트북 실행, MediaPipe FaceLandmarker에 필요한 패키지가 포함되어 있습니다.

저장소에는 Python 3.11.7로 실행한 노트북 기록과 Python 3.12.4 개발 환경 기록이 함께 남아 있습니다. 정확한 patch 버전과 전체 패키지 버전을 고정한 lock file은 제공하지 않습니다.

### 3. CNN 가중치 다운로드

[GitHub Release `weights-260823`](https://github.com/lcsvvo/Driver-Drowsiness-Detection/releases/tag/weights-260823)에서 다음 파일을 받아 `model/artifacts/`에 넣습니다.

```text
model/artifacts/eye_mrl+dmd__eval-dmd__gray128.keras
model/artifacts/yawn_yawn_mouthopen_v2__zoo-cnn_large__eval-face__gray128.keras
```

GitHub CLI를 사용하는 경우 저장소 루트에서 실행합니다.

```bash
gh release download weights-260823 -R lcsvvo/Driver-Drowsiness-Detection -D model/artifacts
```

### 4. FaceLandmarker 다운로드

`face_landmarker.task`를 `model/detectors/`에 저장합니다.

```bash
curl -L -o model/detectors/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Windows PowerShell에서 `curl` 명령이 동작하지 않으면 다음을 사용합니다.

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "model/detectors/face_landmarker.task"
```

이 파일이 없으면 노트북은 중단되지 않지만 입 벌림 게이트가 비활성화됩니다. 최종 하품 판정 구조를 재현하려면 반드시 필요합니다.

## 실행 방법

1. VS Code 또는 Jupyter를 지원하는 편집기에서 `model/02_INFER_YuNet.ipynb`를 엽니다.
2. 위에서 만든 가상환경을 노트북 커널로 선택합니다.
3. 셀을 위에서 아래로 순서대로 실행합니다.
4. `경로 · 가중치 · 게이트` 셀에서 Eye CNN, Yawn CNN, YuNet이 모두 `[o]`이고 입 벌림 게이트가 활성화되었는지 확인합니다.
5. 카메라 점검 셀에서 사용할 카메라와 OpenCV 백엔드를 확인합니다.
6. 마지막 셀을 실행해 실시간 모니터를 시작합니다.
7. 기본 실행은 180초 후 종료됩니다. 그 전에 멈추려면 노트북의 interrupt 버튼을 누릅니다.

마지막 셀의 기본 호출은 다음과 같습니다.

```python
summary = run_realtime_scores(
    max_seconds=180,
    threshold=0.6,
    perclos_window=60.0,
    perclos_threshold=0.15,
)
```

경로 설정만 확인하려면 저장소 루트에서 다음을 실행할 수 있습니다.

```bash
python config.py
```

## 결과 / Demo

실시간 화면에는 얼굴·눈·하품 crop 박스와 다음 값이 표시됩니다.

- `EYE CLOSED`: 현재 눈 감김 점수
- `MOUTH OPEN`: 현재 프레임의 하품 CNN 점수
- `YAWN ACC`: 시간 누적 하품 점수
- `DROWSY`: 최종 졸음 점수
- `PERCLOS`: 최근 구간의 눈 감김 비율
- 눈 감김, 하품, 졸음 경고 누적 횟수

실행 종료 후에는 `DRIVER MONITORING REPORT`와 같은 세션 요약이 출력됩니다. 저장소에는 별도의 데모 영상이나 결과 이미지가 포함되어 있지 않습니다.

## 한계 및 향후 개선

- 손으로 입을 가린 하품의 DMD Recall은 0.162입니다. 랜드마커가 가려진 입을 닫힌 상태로 판단해 게이트에서 차단하는 것이 주된 원인입니다.
- Yawn CNN의 피험자별 성능 차이가 큽니다. 저장된 test의 피험자별 정확도 범위는 0.4404~0.9661입니다.
- PERCLOS는 기본 60초 창을 사용하므로 실행 초반에는 충분한 누적 시간이 필요합니다.
- 조명, 안경, 얼굴 각도, 카메라 위치가 얼굴·눈 검출 결과에 영향을 줄 수 있습니다.
- 현재 평가는 제한된 공개 데이터셋의 hold-out 결과이며 실제 도로 환경의 안전성을 보장하지 않습니다.
- 주의 산만, 머리 방향, 휴대전화 사용, 안전벨트 감지는 구현되어 있지 않습니다.
- 재학습을 완전히 재현하려면 원본 데이터와 눈 데이터 통합 manifest 생성 절차를 별도로 정리해야 합니다.
