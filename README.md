# 운전자 졸음 감지 — 추론 · 학습

YuNet 얼굴 검출 + 직접 학습한 눈 개폐 CNN / 하품 CNN. 웹캠으로 실시간 판정하고,
PERCLOS(최근 1분 중 눈 감긴 시간 비율)를 함께 낸다.

**데이터셋 이미지는 이 저장소에 없다.** 추론은 데이터셋 없이 돌고, 학습만 별도로
데이터를 받아야 한다 (§4).

---

## 1. 5분 안에 돌려보기

```bash
pip install -r requirements.txt
```

**가중치를 받아야 한다.** 저장소에는 코드와 평가 지표만 있고 `.keras` 는 Release 로 뺐다.

    https://github.com/lcsvvo/Driver-Drowsiness-Detection/releases/tag/weights-260820

받은 `.keras` 두 개를 `model/artifacts/` 에 넣는다. 자세한 건
[`model/artifacts/README.md`](model/artifacts/README.md).

그 다음 `model/02_INFER_YuNet.ipynb` 을 열고 **1~4번 섹션(셀 6개)** 을 실행하면 추론 준비가 끝난다.

| 섹션 | 내용 | |
|---|---|---|
| 1~4 | 환경 점검 · 가중치 로딩 · detector 정의 | **필수** |
| 5 | 이미지 파일로 테스트 | 데이터가 있을 때만 |
| 6 | 카메라 점검 | 웹캠 쓸 때 |
| 7 | 사진 한 장 촬영 후 판정 | 수동 실행 |
| 8 | **실시간 점수 모니터** | 수동 실행 |

> 7·8번은 카메라를 열고 대기하므로 `Run All` 에 맞지 않는다. 6번까지 Run All 한 뒤
> 7·8은 직접 실행한다.

가장 흔한 실패는 **OpenCV 버전**이다. `cv2.FaceDetectorYN` 이 4.5.4 에서 들어와서,
그보다 낮으면 두 번째 셀에서 바로 멈춘다.

## 2. 폴더 구조

```
├── config.py                     공통 경로. 코드에 절대경로를 쓰지 않는다
├── requirements.txt
├── model/
│   ├── 02_INFER_YuNet.ipynb      ← 추론은 전부 여기
│   ├── detectors/                YuNet onnx (사전학습, 227KB)
│   └── artifacts/                학습된 가중치 + 평가 지표
│       └── README.md             ← 어떤 가중치가 무엇인지
├── src/                          학습 · 데이터셋 생성 코드
├── scripts/                      DMD 하품 데이터셋 생성 스크립트
└── docs/
    └── yawn_model.md             ← 하품 모델의 성능과 한계
```

## 3. 두 모델

| | 입력 | 출력 | 임계값 |
|---|---|---|---|
| 눈 개폐 | 눈 crop 128x128 gray | `class 0 = Closed` | 0.93 |
| 하품 | 얼굴 crop 128x128 gray | `class 0 = yawn` | 0.50 |

둘 다 `class 0` 이 "위험한 쪽"이다. 순서를 뒤집으면 에러 없이 조용히 반대로 동작한다.

학습과 추론이 **같은 전처리 함수**(`src/eye_preprocess.py` 의 `preprocess_eye`)를
통과한다. 리사이즈나 채널 변환을 추론 쪽에서 따로 손으로 하면 입력 분포가 어긋나
성능만 조용히 떨어지므로, 새 모델을 붙일 때도 이 경로를 지킬 것.

crop 규격은 손으로 적지 않고 `_metrics.json` 에서 읽는다. 가중치 파일만 바꾸면
입력 규격도 따라온다.

### 하품 모델은 아직 실사용 수준이 아니다

실제 하품 비율로 환산한 Precision 이 **0.203** 이다 — 경보 5번 중 4번이 오경보다.
test 피험자 4명에서 정확도가 0.333 ~ 0.864 로 널뛰기도 한다.
**경보를 켜기 전에 `docs/yawn_model.md` 를 읽을 것.**

눈 모델은 Closed-Recall 0.914 / 정확도 0.962 로 쓸 만하다.

## 4. 다시 학습하려면

데이터셋 이미지가 필요하다. 팀 공용 드라이브에서 받아 `Dataset/` 을 이 폴더의
**형제 위치**에 두면 스크립트가 알아서 찾는다. 다른 곳에 두었으면 `--dataset-root` 로 준다.

```
miniProject1/
├── Dataset/                       ← 여기
└── Drowsiness-Detection_260820/   ← 이 폴더
```

```bash
# 하품 — 학습 소스(뷰)만 바꿔 A/B/C 비교
python src/train_yawn.py --train face
python src/train_yawn.py --train body
python src/train_yawn.py --train face body

# 눈 — 학습 소스만 바꿔 A/B/C/C' 비교
python src/train_eye.py --train mrl --eval dmd
python src/train_eye.py --train mrl dmd --eval dmd
```

파이프라인만 빠르게 점검하려면 `--limit 400 --epochs 2` 를 붙인다.

결과는 `model/artifacts/` 에 `.keras` + `_metrics.json` 으로 떨어진다.
학습 없이 평가만 다시 하려면 `train_yawn.reevaluate(tag)`,
조건 비교표는 `train_yawn.compare()`.

데이터셋 자체를 다시 만들려면 DMD 원본 영상과 어노테이션이 필요하다
(`scripts/build_dmd_yawn_dataset.py`, `src/build_dmd_eye_dataset.py`).

## 5. 평가에서 지킨 것

숫자를 인용하기 전에 알아 둘 것.

- **split 은 피험자 단위다.** test 에 나오는 사람은 학습에서 본 적이 없다.
  눈·하품 데이터셋이 같은 매핑을 써서, 두 모델을 합쳐 평가해도 test 피험자가
  학습에 노출되지 않는다.
- **판정 임계값은 val 에서 고르고 test 에 그대로 적용한다.** test 에서 고르면
  낙관 편향이 생긴다. (하품은 이 방식이 역효과여서 0.50 고정을 쓴다 — `docs/yawn_model.md` §3)
- **하품 test 의 클래스 비율은 실제와 다르다.** 샘플링 때문에 하품이 45% 인데
  실제 운전 중에는 14% 안팎이다. 그래서 실제 비율로 환산한 Precision 을
  `precision_at_real_prior` 로 함께 기록한다.
- **여기 숫자는 전부 프레임 단위다.** 실시간 경로는 EMA 로 시간축 누적을 하므로
  더 나을 것이고, 지금 숫자는 하한에 가깝다. 다만 얼마나 나은지는 영상 단위로
  재 봐야 안다 — 아직 안 했다.

## 6. 데이터를 커밋하지 않는 이유

`.gitignore` 가 `data/`, 영상 파일, 대용량 포맷을 막는다.

1. **용량** — 영상 데이터셋은 GB 단위. GitHub 는 파일당 100MB 제한이 있고 히스토리에 남는다.
2. **라이선스** — DMD 등은 재배포 제한이 있다. 학습된 가중치는 파생물이라 괜찮지만
   **원본 프레임을 저장소에 올리면 안 된다.**
3. **초상권** — 직접 촬영한 얼굴 영상이 섞인다.

학습된 `.keras` 도 저장소에 넣지 않고 **Release 로 배포한다**(태그 `weights-260820`).
저장소가 무거워지지 않고, 가중치를 갱신해도 코드 히스토리가 지저분해지지 않는다.
평가 지표(`_metrics.json`)만 추적한다 — 전부 40KB 이고 근거 자료이기 때문이다.
