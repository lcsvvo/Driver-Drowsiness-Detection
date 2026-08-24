# Driver Drowsiness Detection - 260824 수정사항

## 1. 수정 개요

실시간 웹캠 기반 운전자 졸음 감지 과정에서 하품 감지가 정상적으로 이루어지지 않는 문제를 확인하고 수정하였다.

기존에는 하품 CNN 자체에서 높은 하품 확률이 출력되고 있었지만, 입 벌림 정도를 측정하는 `MouthGate`가 비활성화되어 `open_ratio`가 `None`으로 전달되고 있었다. 이로 인해 시간축 기반 하품 누적 판정이 정상적으로 이루어지지 않았다.

또한 실시간 측정 종료 후 운전자의 졸음 상태를 확인할 수 있도록 눈 감김, PERCLOS, 하품, 졸음 경고 등의 세션 통계와 종합 결과를 출력하는 `DRIVER MONITORING REPORT`를 추가하였다.


---

## 2. 하품 감지 문제

### 기존 문제

실시간 웹캠 테스트에서 실제로 하품을 수행하면 Yawn CNN의 출력 확률은 정상적으로 높아졌다.

예를 들어 다음과 같이 `raw_p`가 0.9 이상까지 상승하는 것을 확인하였다.

```text
raw_p=0.983
raw_p=0.978
raw_p=0.991
raw_p=0.994
```

하지만 동시에 다음과 같은 상태가 확인되었다.

```text
detector.gate = None
open_ratio = None
```

즉, **Yawn CNN은 하품 가능성을 높게 예측하고 있었지만 입 벌림 정도가 측정되지 않고 있었다.**

하품의 최종 판정에는 CNN의 하품 확률뿐만 아니라 입 벌림 정도와 시간축 누적값이 사용되기 때문에 `open_ratio=None` 상태에서는 의도한 하품 누적 판정이 정상적으로 이루어질 수 없었다.


---

## 3. 원인 확인

설정을 확인한 결과 다음과 같은 상태였다.

```python
YAWN_USE_GATE = True

YAWN_GATE = {
    "enabled": False,
    "threshold": 0.05,
    "landmarker": ".../face_landmarker.task",
    "hold": 3,
}
```

전체 설정에서는 `YAWN_USE_GATE=True`였지만 실제 게이트 설정의 `enabled` 값은 `False`였다.

따라서 `Drowsiness_Detector` 초기화 과정에서 `MouthGate` 객체가 생성되지 않았다.

```python
if YAWN_USE_GATE and self.gate_spec and self.gate_spec.get("enabled"):
    self.gate = MouthGate(...)
```

결과적으로:

```text
detector.gate = None
```

상태가 되어 입 벌림 정도를 측정하지 못하고 있었다.


---

## 4. MouthGate 활성화

입 벌림 정도를 실제로 측정할 수 있도록 게이트를 활성화하였다.

```python
YAWN_USE_GATE = True

YAWN_GATE = {
    "enabled": True,
    "threshold": 0.05,
    "landmarker": ".../face_landmarker.task",
    "hold": 3,
}
```

수정 후 다음과 같이 `MouthGate` 객체가 정상적으로 생성되는 것을 확인하였다.

```text
YAWN_USE_GATE = True
YAWN_GATE = {'enabled': True, 'threshold': 0.05, ...}
detector.gate = <mouth_gate.MouthGate object ...>
```


---

## 5. MediaPipe FaceLandmarker 적용

`MouthGate`에서는 MediaPipe FaceLandmarker를 이용해 얼굴 랜드마크를 검출하고 입 벌림 정도를 계산한다.

사용하는 주요 랜드마크는 다음과 같다.

| 위치 | Landmark Index |
|---|---:|
| 위쪽 안쪽 입술 | 13 |
| 아래쪽 안쪽 입술 | 14 |
| 왼쪽 눈 바깥점 | 33 |
| 오른쪽 눈 바깥점 | 263 |
| 입 왼쪽 | 61 |
| 입 오른쪽 | 291 |

입 벌림 정도인 `open_ratio`는 다음 기준으로 계산한다.

```text
gap = distance(upper_inner_lip, lower_inner_lip)
eye = distance(left_outer_eye, right_outer_eye)

open_ratio = gap / eye
```

단순한 입술 사이의 픽셀 거리를 사용하는 대신 눈 사이 거리로 정규화하여 얼굴이 카메라에서 가까워지거나 멀어지는 것에 따른 영향을 줄인다.


---

## 6. MediaPipe 의존성 문제 해결

MouthGate를 활성화한 이후 다음 오류가 발생하였다.

```text
ModuleNotFoundError: No module named 'mediapipe'
```

기존에는 MouthGate 자체가 비활성화되어 있었기 때문에 MediaPipe 코드가 실제로 실행되지 않아 해당 문제가 나타나지 않았다.

가상환경에 MediaPipe를 설치하여 해결하였다.

```bash
python -m pip install mediapipe
```

설치 과정에서 OpenCV의 `cv2.pyd`가 Jupyter/VS Code에서 사용 중이어서 다음 오류가 발생하기도 하였다.

```text
[WinError 5] 액세스가 거부되었습니다:
.venv\Lib\site-packages\cv2\cv2.pyd
```

실행 중인 Jupyter 커널 및 Python 프로세스를 종료한 후 가상환경에서 다시 설치하여 해결하였다.

설치 후 MediaPipe와 OpenCV가 정상적으로 import되는 것을 확인하였다.


---

## 7. MouthGate 동작 확인

디버깅을 통해 FaceLandmarker가 정상적으로 얼굴 랜드마크를 검출하고 `open_ratio`를 계산하는 것을 확인하였다.

입을 다물고 있는 상태에서는 `open_ratio`가 매우 낮게 나타났으며:

```text
open_ratio ≈ 0.003
open_ratio ≈ 0.01
open_ratio ≈ 0.03
```

입을 크게 벌릴수록 다음과 같이 값이 증가하였다.

```text
open_ratio ≈ 0.13
open_ratio ≈ 0.20
open_ratio ≈ 0.28
open_ratio ≈ 0.34
```

따라서 입 벌림 정도가 실시간으로 정상 측정되는 것을 확인하였다.

게이트 임계값은 다음과 같다.

```python
DEFAULT_THRESHOLD = 0.05
```

최근 `hold` 프레임에서 측정한 `open_ratio`의 최댓값이 임계값을 초과하면 Yawn CNN을 실행한다.

```text
open_ratio <= threshold
    → Mouth CLOSED
    → Yawn CNN 실행하지 않음
    → p(yawn) = 0

open_ratio > threshold
    → Mouth OPEN
    → Yawn CNN 실행
```


---

## 8. 하품 시간축 누적 판정

한 프레임의 CNN 출력만으로 하품을 판단하면 말하거나 잠깐 입을 벌리는 행동을 하품으로 잘못 판단할 가능성이 있다.

이를 줄이기 위해 `YawnAccumulator`를 이용하여 하품 확률과 입 벌림 정도를 시간축으로 누적한다.

실시간 처리 흐름은 다음과 같다.

```text
Face Detection
      ↓
Face Crop
      ↓
MediaPipe FaceLandmarker
      ↓
open_ratio 계산
      ↓
MouthGate
      ↓
입이 충분히 벌어졌는가?
   ├─ No  → p(yawn)=0
   └─ Yes → Yawn CNN
                ↓
            p(yawn)
                ↓
        YawnAccumulator
                ↓
        시간축 하품 판정
                ↓
          Yawn Event
```

최종적으로 `YawnAccumulator`의 상태가 `False → True`로 변경되는 순간을 하나의 하품 이벤트로 기록한다.

이를 통해 하품이 지속되는 동안 매 프레임을 각각 하나의 하품으로 세지 않고, 하나의 연속된 하품 행동을 **1회**로 집계한다.


---

## 9. 디버깅 출력 제거

문제 확인 과정에서 다음과 같은 디버깅 출력을 임시로 추가하였다.

```text
[YAWN FLOW DEBUG]
[GATE DEBUG]
[MOUTH DEBUG]
FaceLandmarker OK
MouthMeter 결과
```

이를 통해 다음 항목을 확인하였다.

- `MouthGate` 객체 생성 여부
- FaceLandmarker 실행 여부
- 얼굴 랜드마크 검출 여부
- `open_ratio` 계산 여부
- MouthGate OPEN/CLOSED 여부
- Yawn CNN 출력값

문제 해결 후에는 해당 `print()` 문이 웹캠의 매 프레임마다 실행되어 Jupyter 출력이 과도하게 증가하므로 최종 코드에서 제거하였다.

최종 코드에서는 필요한 추론 및 측정 기능만 유지한다.


---

# 10. 운전자 모니터링 지표 추가

실시간 웹캠 측정이 종료된 후 단순히 `DROWSY / NORMAL`만 출력하는 것이 아니라, 측정 세션 동안 발생한 운전자의 졸음 관련 행동을 요약하도록 수정하였다.

추가한 주요 지표는 다음과 같다.

### 졸음 의심 눈 감김 횟수

눈 감김 상태가 일정 시간 이상 지속된 경우를 하나의 이벤트로 집계한다.

현재 기준:

```python
suspicious_closure_sec = 0.15
```

따라서 **0.15초 이상 지속된 눈 감김**만 졸음 의심 눈 감김 이벤트로 기록한다.


### 최대 눈 감김 시간

전체 측정 시간 동안 발생한 눈 감김 이벤트 중 가장 길게 눈을 감고 있었던 시간을 기록한다.

예:

```text
최대 눈 감김 시간        1.24 sec
```


### PERCLOS

전체 측정 시간 중 눈이 감겨 있던 시간의 비율을 이용하여 운전자의 눈 감김 상태를 평가한다.

현재 기본 임계값:

```python
perclos_threshold = 0.15
```

즉 PERCLOS가 약 15% 이상이면 높은 PERCLOS 상태로 판단하도록 구성하였다.


### 하품 횟수

`YawnAccumulator`에서 새로운 하품 이벤트가 발생한 횟수를 집계한다.

하품 상태가 유지되는 동안에는 추가로 카운트하지 않고 새로운 하품이 시작될 때만 1회 증가한다.


### 졸음 경고 횟수

최종 결합 졸음 점수가 정상 상태에서 DROWSY 상태로 전환되는 순간을 하나의 졸음 경고 이벤트로 집계한다.


---

# 11. 주요 졸음 의심 행동

측정된 지표를 기반으로 세션 종료 후 주요 졸음 의심 행동을 함께 출력하도록 추가하였다.

현재 사용하는 기준은 다음과 같다.

| 조건 | 출력 |
|---|---|
| 졸음 의심 눈 감김 3회 이상 | 반복적인 장시간 눈 감김 |
| 최대 눈 감김 시간 1초 이상 | 1초 이상의 장시간 눈 감김 |
| PERCLOS 임계값 이상 | 높은 PERCLOS |
| 하품 2회 이상 | 반복적인 하품 |

예:

```text
주요 졸음 의심 행동

• 반복적인 장시간 눈 감김
• 높은 PERCLOS
• 반복적인 하품
```


---

# 12. DRIVER MONITORING REPORT

웹캠 실행을 종료하면 전체 측정 세션을 요약한 `DRIVER MONITORING REPORT`를 출력하도록 수정하였다.

웹캠 영상 및 얼굴 이미지는 개인정보 보호를 위해 저장하거나 저장소에 포함하지 않고, **측정된 지표와 텍스트 형태의 결과만 출력한다.**

## 출력 예시

```text
====================================
       DRIVER MONITORING REPORT
====================================

측정 시간              03:00

[눈 상태]
졸음 의심 눈 감김       7회
최대 눈 감김 시간        1.24 sec
PERCLOS                 18.3 %

[하품]
하품                     3회

[졸음 감지]
졸음 경고                4회

------------------------------------
주요 졸음 의심 행동

• 반복적인 장시간 눈 감김
• 1초 이상의 장시간 눈 감김
• 높은 PERCLOS
• 반복적인 하품

종합 상태
⚠ DROWSINESS SUSPECTED
====================================
```

정상 상태의 경우 다음과 같이 출력된다.

```text
====================================
       DRIVER MONITORING REPORT
====================================

측정 시간              03:00

[눈 상태]
졸음 의심 눈 감김       0회
최대 눈 감김 시간        0.08 sec
PERCLOS                 4.2 %

[하품]
하품                     0회

[졸음 감지]
졸음 경고                0회

------------------------------------
주요 졸음 의심 행동

• 특이사항 없음

종합 상태
✓ NORMAL
====================================
```


---

# 13. 최종 변경 사항

이번 수정의 주요 내용은 다음과 같다.

1. 비활성화되어 있던 `MouthGate` 활성화
2. MediaPipe FaceLandmarker 기반 입 벌림 측정 적용
3. `open_ratio`가 `None`으로 전달되던 문제 해결
4. 입 벌림 여부에 따른 Yawn CNN 실행 제어
5. Yawn CNN과 `YawnAccumulator` 연결 및 시간축 하품 판정
6. 연속된 하품을 하나의 하품 이벤트로 집계
7. 0.15초 이상 지속된 졸음 의심 눈 감김 횟수 추가
8. 최대 연속 눈 감김 시간 측정
9. PERCLOS 기반 졸음 지표 추가
10. 하품 횟수 및 DROWSY 경고 횟수 집계
11. 주요 졸음 의심 행동 요약 추가
12. 측정 종료 후 `DRIVER MONITORING REPORT` 출력
13. 문제 확인을 위해 추가했던 디버깅 출력 제거
14. 웹캠 얼굴 영상/이미지는 저장하지 않고 텍스트 결과만 제공