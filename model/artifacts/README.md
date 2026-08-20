# 가중치 파일

**`.keras` 는 저장소에 없다. Release 에서 받아 이 폴더에 넣어야 한다.**

    https://github.com/lcsvvo/Driver-Drowsiness-Detection/releases/tag/weights-260820

받을 파일은 두 개다. 이 폴더(`model/artifacts/`)에 그대로 놓으면 된다.

```
eye_mrl+dmd__eval-dmd__gray128.keras
yawn_face+body__eval-face__gray128.keras
```

`_metrics.json` 은 저장소에 이미 들어 있으니 따로 받을 필요 없다(전체 40KB).
노트북이 이 파일에서 입력 규격을 읽으므로 짝이 맞아야 하는데, 위 Release 의
가중치와 저장소의 metrics 는 같은 시점 산출물이다.

비교 실험 조건들은 metrics 만 남겼고, 가중치가 필요하면
`src/train_yawn.py` / `src/train_eye.py` 로 몇 분 안에 다시 만들 수 있다.

파일명 규칙: `{eye|yawn}_{학습소스}__eval-{평가대상}__{입력규격}[__변형]`
`_metrics.json` 이 입력 규격·판정 임계값·혼동행렬을 들고 있고 `02_INFER_YuNet.ipynb`
이 그 파일에서 규격을 읽는다. **`.keras` 와 `_metrics.json` 은 항상 같이 다녀야 한다.**

---

## Release 에 들어 있는 가중치 (19MB)

| 파일 | 무엇 | 성능 | 임계값 |
|---|---|---|---|
| `eye_mrl+dmd__eval-dmd__gray128.keras` | 눈 개폐 | Closed-Recall **0.914** / acc 0.962 | 0.93 |
| `yawn_face+body__eval-face__gray128.keras` | 하품 | Recall **0.807** / acc 0.630 | 0.50 |

둘 다 128x128 그레이스케일 입력, 2-class softmax 출력.

- 눈: `class 0 = Closed`
- 하품: `class 0 = yawn`

> 두 모델 다 `class 0` 이 "위험한 쪽"이다. 순서를 뒤집으면 에러 없이 조용히 반대로
> 동작하므로 새 모델을 학습할 때도 이 규칙을 지킬 것.

## metrics 만 남긴 조건들

가중치는 지웠고 평가 결과만 남겼다. 왜 위 두 개가 뽑혔는지의 근거다.

**눈** — DMD hold-out, 프레임 단위

| 조건 | Closed-Recall | acc | 지운 이유 |
|---|---|---|---|
| A `eye_mrl__` | 0.853 | 0.776 | Precision 0.317. 뜬 눈을 감았다고 하는 오경보가 너무 많다 |
| B `eye_dmd__` | 0.597 | 0.951 | 감긴 눈을 40% 놓친다. 졸음 감지에서 가장 위험한 실패 |
| C' `eye_dmd__...__ft` | 0.881 | 0.963 | 나쁘지 않다. C 보다 Recall 이 낮아서 뺐다 |

**하품** — face test 1,131장 @0.50

| 조건 | Recall | Precision | acc | 지운 이유 |
|---|---|---|---|---|
| A `yawn_face__` | 0.691 | 0.623 | 0.669 | C 와 F1 이 사실상 같다. 오경보는 더 적다 |
| B `yawn_body__` | 0.247 | 0.291 | 0.384 | 측면 학습은 정면 카메라에 안 통한다는 대조군 |

### 다시 만들려면

```bash
python src/train_eye.py  --train mrl --eval dmd            # 눈 A
python src/train_eye.py  --train dmd --eval dmd            # 눈 B
python src/train_yawn.py --train face                      # 하품 A
python src/train_yawn.py --train body                      # 하품 B
```

시드가 고정(42)돼 있지만 TF 버전·하드웨어에 따라 소수점 아래는 달라질 수 있다.

> **두 개는 "성능이 낮아서" 지운 게 아니다.** 눈 C'(0.881/0.963)와 하품 A(오경보가
> 더 적음)는 쓸 만한 대안이다. 바꿔 끼우고 싶으면 위 명령으로 다시 만든 뒤
> 노트북의 `EYE_MODEL_NAME` / `YAWN_MODEL_NAME` 만 바꾸면 된다.

## 아예 안 가져온 것

| 파일 | 왜 |
|---|---|
| `eye_model.keras`, `yawn_model.keras` | 옛 `01_TRAIN.ipynb` 산출물. 각 42MB(Adam 상태 포함), 256 RGB 규격이라 지금 파이프라인과 입력이 안 맞는다 |
| `eye_mrl+dmd__eval-dmd__rgb256.keras` | 42MB 부록 실험. Closed-Recall 0.441 로 결과도 나빴다 |
| `*__smoke.keras` | 파이프라인 점검용 스모크 잔재 |
| `*_history.json` | 옛 01_TRAIN 학습곡선. 위 모델들과 무관 |

## 주의 — 하품 모델은 아직 실사용 수준이 아니다

- 실제 하품 비율(14%)로 환산한 Precision **0.203** — 경보 5번 중 4번이 오경보
- test 피험자 4명에서 정확도가 0.333 ~ 0.864 로 널뛴다

근거와 한계는 `docs/yawn_model.md` 에 있다. **경보를 켜기 전에 반드시 읽을 것.**
