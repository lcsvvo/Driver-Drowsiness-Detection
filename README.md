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
├── scripts/                      데이터셋 생성 스크립트 (DMD · YawDD)
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

### YawDD (하품 데이터 추가분)

DMD 하품 데이터가 16세션뿐이라 YawDD(운전 중 하품 영상 349개)를 붙였다. 원본
`data/user06.tar` 에서 시작해 두 단계로 만든다.

```bash
python scripts/prepare_yawdd.py             # 압축 해제 결과 정규화 + 검증 -> raw/YawDD/
python scripts/build_yawdd_yawn_dataset.py --all   # 얼굴 crop + manifest -> processed_yawdd/
python scripts/build_yawn_mouthopen_dataset.py --measure --build   # 입 벌린 것만 -> yawn_mouthopen_v2/
```

**YawDD 의 라벨은 영상 단위다.** `Yawning` 영상에 하품이 들어 있다는 뜻이지 그 영상의
모든 프레임이 하품이라는 뜻이 아니다. 반면 `Normal`·`Talking` 영상은 모든 프레임이
하품이 아닌 것이 확실하다(`Talking` 은 "입은 열리는데 하품은 아닌" 표본이라 오경보를
줄이는 데 특히 값이 있다). 그래서 manifest 에 `label_quality`(strong/weak)를 함께
적었다. **약한 라벨을 거르지 않고 그대로 학습하면 안 된다** —
`Dataset/processed_yawdd/README.md` §3 을 먼저 읽을 것.

학습은 `--manifest` 로 다른 manifest 를 지정해 돌린다(기본값은 DMD 그대로다).

```bash
python src/train_yawn.py --train face \
    --manifest ../Dataset/processed_yawdd/metadata/manifest.csv
```

### 입 벌린 negative (`yawn_mouthopen_v2`)

하품 모델의 오경보가 많은 이유는 negative 가 대부분 **입을 다물고** 있어서다. 모델이
하품이 아니라 "입이 벌어졌는가"를 배우고, 말하는 운전자에게 그대로 경보를 울린다.

`build_yawn_mouthopen_dataset.py` 는 mediapipe 랜드마커로 입 벌림(`open_ratio`)을 재서
**두 클래스 양쪽에서** 입 다문 프레임을 뺀다. DMD 와 YawDD 를 함께 쓴다.

```bash
python scripts/build_yawn_mouthopen_dataset.py --measure --sources dmd yawdd
python scripts/build_yawn_mouthopen_dataset.py --stats     # 임계값 고르기
python scripts/build_yawn_mouthopen_dataset.py --build     # -> yawn_mouthopen_v2/
```

임계값 0.05 로 7,919장(no_yawn 4,103 / yawn 3,816), 피험자 81명. 이미지는 하드링크라
용량이 늘지 않는다.

**두 출처의 역할이 정반대다.** DMD 의 `no_yawn` 은 정상 주행이라 거의 다 입을 다물고
있어서(임계값을 넘는 것이 5%) 합본에서 DMD 는 사실상 positive 공급원이다 - 대신
프레임 단위 GT 라 라벨이 정확하다. "입은 벌렸는데 하품이 아닌" negative 는 89% 가
YawDD 의 `talking` 에서 온다. 한쪽만으로는 이 데이터셋이 성립하지 않는다.

`yawn_with_hand`(손이 입을 가려 벌림을 잴 수 없다)와 `talking_yawning`(한 영상에
말하기와 하품이 섞여 프레임 라벨 불가)은 제외한다.

```bash
python src/train_yawn_zoo.py --arch cnn_large \
    --manifest ../Dataset/yawn_mouthopen_v2/metadata/manifest.csv
```

임계값을 바꾸거나(`--threshold`), 두 클래스의 입 벌림 분포를 아예 같게 맞춰
"입 벌린 정도"라는 지름길을 막는(`--match-openness`) 것도 된다. 측정값은 CSV 로 남아서
다시 자를 때는 몇 초면 끝난다. 옛 YawDD 전용 데이터셋은
`--sources yawdd --threshold 0.10 --out yawn_mouthopen` 으로 그대로 재현된다.

### 추론에도 같은 문을 단다 (`src/mouth_gate.py`)

데이터셋을 이렇게 만들었으면 **추론도 같은 조건이어야 한다.** 입 다문 프레임은 CNN 이
본 적 없는 입력이라 넣으면 값이 튄다. `02_INFER_YuNet.ipynb` 는 CNN 앞에 게이트를 둔다.

```
open_ratio <= 0.05  ->  CNN 을 부르지 않고 p(yawn) = 0
open_ratio >  0.05  ->  CNN 에 물어본다
```

측정 함수는 `src/mouth_gate.py` 한 곳에만 있고 데이터셋 빌더가 그것을 import 한다.
두 곳에 따로 두면 값이 조용히 갈라지기 때문이다. 노트북은 실행할 때 게이트 임계값과
가중치를 만든 데이터셋의 임계값이 같은지 확인하고, 다르면 경고한다.

**임계값이 곧 Recall 상한이다.** 게이트에 걸린 하품은 모델이 아무리 좋아도 못 잡는다.
DMD 하품 프레임(n=1,148, 프레임 단위 GT) 기준:

| 게이트 | 하품인데 걸리는 비율 | Recall 상한 |
|---|---|---|
| 0.03 | 7.8% | 0.922 |
| **0.05** | **10.5%** | **0.895** |
| 0.08 | 14.4% | 0.856 |
| 0.10 | 18.3% | 0.817 |

게이트는 실시간 루프(`run_realtime_scores`)에도 들어 있다. 그 루프는 속도 때문에
`yawn_detection()` 을 거치지 않고 모델을 직접 부르므로, 게이트를 양쪽에 두지 않으면
낱장 추론에서만 걸리고 실시간에서는 빠진다.

### 하품은 시간으로 누적한다 (`src/yawn_accumulator.py`)

게이트를 통과한 다음에도 문제가 남는다. 프레임 한 장으로는 하품과 말하기를 가를 수
없어서, **말하기 세션의 68% 에서 최소 한 번 하품 경보가 뜬다**(YawDD val+test 41세션).

재학습으로 풀려면 시퀀스 모델이 필요한데 하품 사건이 180개뿐이라 지금 데이터로는
과적합한다. 대신 이미 매 프레임 나오는 두 신호를 시간축으로 묶는다.

| | 하는 일 |
|---|---|
| 누수 적분기 | `p_yawn` 이 판정선을 넘는 동안 시간을 쌓고, 못 넘으면 3배로 깎는다 |
| 최대 벌림 | 최근 3초의 최대 `open_ratio`. 말하기는 0.20 을 잘 못 넘는다 |

근거 (YawDD 세션 단위 중앙값): 말하기는 입이 **3배 빠르게** 움직이고(|기울기| 0.056
대 0.018/초), 하품은 **2배 넓게** 벌어진다(최대 open_ratio 0.453 대 0.195).

| | YawDD 하품 | YawDD 말하기 | DMD 사건 Recall | DMD 오경보 |
|---|---|---|---|---|
| 누적 없음 | 0.977 | 0.683 | 0.941 | 0.0032 |
| **누적 적용** | 0.860 | **0.195** | 0.824 | 0.0016 |

말하기 오경보가 **71% 줄어든다.** 대가로 하품 검출이 0.977 -> 0.860 으로 내려간다.
split 을 나눠도 방향은 같지만(val 말하기 0.650->0.300, test 0.714->0.095) split 당
세션이 20개 남짓이라 소수점 둘째 자리는 믿을 값이 아니다.

놓치는 쪽이 더 급하면 `YawnAccumulator(peak_min=0.0, fire=0.20)` 으로 두면 된다.
말하기가 0.439 로 덜 줄지만 DMD 사건 Recall 이 0.941 로 손실이 없다 — 손으로 가린
하품은 입이 안 보여 `open_ratio` 가 낮으므로 `peak_min` 이 그쪽을 깎기 때문이다.

**프레임이 아니라 시간을 센다.** 추론 루프가 입을 벌리면 22Hz, 다물면 30Hz 로 돌고
PC 마다도 다르기 때문이다(실측). `PerclosTracker` 가 dt 를 더하는 것과 같은 이유다.

`--measure` 에는 mediapipe 랜드마커 모델이 필요하다. `.gitignore` 가 `*.task` 를
막으므로(3.7MB) 각자 받아 `model/detectors/face_landmarker.task` 에 둔다. 측정은 한 번만
하면 되고, 결과 CSV 가 남아서 다시 자를 때는 모델이 없어도 된다.

```bash
curl -L -o model/detectors/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

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
