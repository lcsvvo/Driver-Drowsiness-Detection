# Driver Drowsiness Detection

주행 영상·웹캠에서 운전자의 **졸음**을 감지하는 시스템. 4인 팀 프로젝트, 학부 수준.

`Python 3.10+` · YuNet 얼굴 검출 + CNN 눈/하품 분류 + PERCLOS 시간 누적

> 이 README 는 **실제 구현된 코드 기준**으로 작성한다. 계획만 있고 미구현인 항목은 "예정"으로 표기한다.

## 1. 프로젝트 소개

- **무엇**: 카메라 영상에서 얼굴을 검출하고, 눈 감김·하품 상태를 CNN으로 분류한 뒤, 시간 창에서 PERCLOS·최장 감김·하품 빈도 등을 이용해 졸음 위험을 판정하는 시스템을 구현한다.
- **왜**: 졸음운전은 사고의 큰 원인이다. 별도 센서 없이 카메라만으로 동작하는 저비용 감지의 실현 가능성을 직접 구현해 확인한다.
- **범위**: 눈 상태 분류기(눈 트랙)와 하품 분류기(하품 트랙)를 학습·검증하고, 두 지표를 시간 누적해 위험도로 통합한다.

## 2. 문제 정의

단일 프레임의 눈 감김만으로는 졸음을 판정할 수 없다. 깜빡임과 졸음, 하품과 대화는 **정지 영상이 아니라 시간 축에서만 구분된다.** 그래서 프레임 단위 판정을 그대로 쓰지 않고, 시간 창 위에서 누적·평활한 값(PERCLOS 등)으로 상태를 판정한다.

또한 **눈 감김이라는 단일 신호만으로는 졸음과 하품을 명확하게 구분하기 어려운 결과가 관찰됐다**(눈 트랙 STEP14·15). 하품 상황에서도 Closed 판정이 상당 비율 나타났기 때문이다. 따라서 후속 위험도 지수에서는 눈 지표와 하품 등 다른 행동 정보를 함께 고려하는 방안을 검토한다.

## 3. 목표

| # | 목표 | 상태 |
|---|---|---|
| G1 | YuNet 얼굴 검출 + 눈/하품 crop 추출 | 완료 (`model/02_INFER_YuNet.ipynb`) |
| G2 | 눈 감김 CNN 분류 + MRL·DMD 로 학습·검증 | **완료 (눈 트랙 10–15)** |
| G3 | 하품 CNN 분류 | 예정 (하품 트랙 20·21, 팀원) |
| G4 | PERCLOS·최장 감김·하품 빈도 누적 | 부분 (02 에 PERCLOS 실시간, 통합은 예정) |
| G5 | 위험도 지수 통합 + 경고 | 예정 (STEP30) |

> 초기 계획의 EAR/MAR(기하 랜드마크)·머리 방향 주의산만은 현재 코드에 없다. 눈·하품은 **CNN 분류**로 방향을 바꿨고, 주의산만은 범위에서 제외됐다. 이 표는 실제 코드 기준이다.

## 4. 접근 방법

| 구성 요소 | 방식 | 상태 |
|---|---|---|
| 얼굴 검출 | YuNet (`FaceDetectorYN`, OpenCV) | 완료 |
| 눈 상태 분류 | CNN (Closed / Open), 128×128 grayscale | 완료 |
| 하품 분류 | CNN (yawn / no_yawn) | baseline(01) + 예정(21) |
| 졸음 누적 지표 | PERCLOS, 최장 연속 감김 | 02 에 실시간 구현 |
| 위험도 통합 | 눈 + 하품 지표 결합 | 예정 |
| 눈 CNN 학습 데이터 | MRL Eye(주) + DMD(도메인 보강) | 완료 |
| 눈 CNN 검증 | DMD hold-out + NITYMED microsleep | 완료 |

### 눈 트랙 핵심 결과 (STEP10–15)

- **눈 CNN은 DMD hold-out에서 Closed-Recall 0.914**를 기록했으며, 실제 영상 파이프라인에서도 Closed-Recall **0.916**을 기록했다.
- **MRL+DMD 합본 학습(C)**은 네 학습 조건 중 가장 높은 Closed-Recall과 F1을 기록했다. MRL only 조건은 DMD 환경에서 Precision이 낮았고, DMD 학습 데이터를 포함한 조건에서 개선이 관찰됐다.
- **NITYMED에서는 영상 길이와 창 개수의 영향이 확인됐다.** 창 개수만으로 AUC가 **0.945**였으며, 창 개수의 영향을 고려한 후 세 눈 지표의 AUC는 **0.402~0.424**로 낮아졌다.
- **하품 구간의 Closed 판정 비율은 25.7%**였으며, microsleep 영상에서 샘플링한 프레임은 19.9%였다. 영상 단위 Closed 비율 AUC는 **0.440 [0.304, 0.577]**로, 이 표본에서는 눈 감김 단독의 명확한 판별력을 확인하지 못했다.
- 따라서 **눈 CNN의 눈 상태 분류 성능과 졸음·하품을 구분하는 성능은 별개의 문제**이며, 후속 위험도 지수에서는 눈 감김 외의 행동 정보를 함께 고려할 필요가 있다.

## 5. 기술 스택

| 구분 | 도구 |
|---|---|
| 언어 | Python 3.10+ |
| 얼굴 검출 | OpenCV `FaceDetectorYN` (YuNet) |
| 분류 모델 | TensorFlow / Keras CNN |
| 수치·평가 | NumPy, pandas, scikit-learn |
| 시각화 | matplotlib |
| 개발 환경 | VS Code (로컬), 필요시 Colab(GPU 학습) |

의존성은 `requirements.txt` 참고. `pip install -r requirements.txt`.

## 6. 저장소 구조

```
Driver-Drowsiness-Detection/
├── README.md
├── requirements.txt
├── config.py               # 공통 경로 (팀원 PC 간 통일). 절대경로 금지
├── model/
│   ├── 01_TRAIN.ipynb          # Eye/Yawn CNN 학습 (수정 금지)
│   ├── 02_INFER_YuNet.ipynb    # YuNet + 추론 + 실시간 PERCLOS (수정 금지)
│   ├── artifacts/              # 학습된 .keras (git 제외, Release 배포)
│   └── detectors/
│       └── face_detection_yunet_2023mar.onnx   # YuNet (git 추적)
├── notebooks/              # 눈 트랙 (이 저장소의 핵심 작업)
│   ├── 00_overview.ipynb       # 전체 지도 — 여기서 시작
│   ├── 10_eye_gt_dmd.ipynb
│   ├── 11_eye_dataset_build.ipynb
│   ├── 12_eye_train_eval.ipynb
│   ├── 13_eye_frame_eval_dmd.ipynb
│   ├── 14_eye_nitymed_video_eval.ipynb
│   └── 15_eye_yawn_falsepositive.ipynb
│   └── 19_eye_train_colab.ipynb  # 학습 보조 노트북
├── src/                    # 눈 트랙 파이썬 모듈
│   ├── dmd_annotation.py  dmd_crop.py  build_dmd_eye_dataset.py
│   ├── mrl_split.py  mrl_dataset.py  eye_preprocess.py
│   └── train_eye.py
├── data/                   # 데이터셋 (git 제외)
│   ├── raw/{DMD, MRL Eye, NITYMED, Yawn-Eye-Dataset New}
│   └── proceed/NITYMED     # NITYMED 전처리 얼굴 crop
└── outputs/                # 실행 결과 (git 제외)
```

git 제외: `data/`, `outputs/`, `model/artifacts/`, `private/`. `model/detectors/` 는 예외적으로 추적(§8).

## 7. 실행 방법

```bash
git clone <repo-url> && cd Driver-Drowsiness-Detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python config.py                                     # 경로 확인 · data/·outputs/ 생성
```

### 눈 트랙 실행 순서

`notebooks/00_overview.ipynb` 를 먼저 열어 전체 지도와 산출물 점검을 본다. 그다음:

```
[사전] model/01_TRAIN 실행 → model/artifacts/eye_model.keras
       python src/mrl_split.py --mrl_root "data/raw/MRL Eye/data" --out_dir outputs/mrl_split

10 → 11 → 12 → (13 · 14 · 15)
```

13·14·15 는 서로 독립이며 모두 12 의 C 모델(`eye_mrl+dmd__eval-dmd__gray128.keras`)을 공유한다. 각 노트북은 첫 셀에서 `config.py` 를 찾아 경로를 가져오므로 저장소 안 어디서 열어도 동작한다.

### 기준본 (수정 금지)

`model/01_TRAIN.ipynb`(CNN 학습), `model/02_INFER_YuNet.ipynb`(YuNet 추론·실시간 데모)는 팀 기준본이라 눈 트랙 작업에서 수정하지 않는다. 눈 트랙 모델은 규격이 달라(128×128×1 vs 256×256×3) 02 에 그대로 교체할 수 없다. 실시간 시연에 쓰려면 별도 추론 노트북이 필요하다(STEP12·13 참고).

## 8. 데이터·모델 관리

- **개인 PC 절대경로를 코드에 쓰지 않는다.** 경로는 전부 `config.py` 에서 가져온다.
- 데이터·영상·`.keras` 는 커밋하지 않는다. `model/detectors/*.onnx`(232 KB)만 예외로 추적한다 — 사전학습 검출기라 모든 팀원이 같은 파일을 써야 하기 때문.
- 노트북은 **출력을 지우고 커밋**한다. 이미지 출력이 들어가면 diff 가 안 읽히고 용량이 커진다.
- CNN 가중치(`eye_model.keras`, `yawn_model.keras`)는 저장소 Release 로 배포한다. 받아서 `model/artifacts/` 에 넣거나 `01_TRAIN` 으로 직접 학습한다.

## 9. 진행 상황

| 트랙 | 상태 |
|---|---|
| 기준본 01·02 (YuNet + CNN + 실시간 PERCLOS) | 완료 |
| **눈 트랙 10–15 (MRL·DMD 학습·검증)** | **완료** |
| 하품 트랙 20·21 | 예정 (팀원) |
| 위험도 지수 30 | 예정 |

눈 트랙 상세 결과는 `notebooks/00_overview.ipynb` 참고.

## 10. 협업 방식

4인이 각자 다른 PC 에서 작업하므로 아래를 지킨다.

1. 작업 전 `main` 에 `git pull`
2. 작업 단위 브랜치 — `<type>/<설명>` (예: `feature/yawn-cnn`). 에이전트 자동 브랜치명(`claude/~` 등)은 쓰지 않는다
3. 작업 후 `commit` → `push` → PR (무엇을/왜/확인할 점)
4. **셀프 머지 금지.** 최소 1명 리뷰 후 머지
5. 개인 PC 절대경로 금지, 데이터·모델 커밋 금지, 노트북 출력 지우고 커밋

### 참고 문헌

- Soukupová & Čech (2016), *Real-Time Eye Blink Detection using Facial Landmarks*, CVWW
- Albadawi et al. (2023), *Real-Time Machine Learning-Based Driver Drowsiness Detection Using Visual Features*, J. Imaging
