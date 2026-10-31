# Model

**실시간 졸음 감지 실행 진입점**과 모델 파일 안내.

| Path | Content |
|---|---|
| [02_INFER_YuNet.ipynb](02_INFER_YuNet.ipynb) | 카메라 입력, 눈과 하품 추론, 점수 결합과 세션 요약 |
| [Artifacts](artifacts/README.md) | CNN 가중치 다운로드와 평가 지표 |
| [Detectors](detectors/README.md) | YuNet과 FaceLandmarker 파일 |

## Setup

저장소 루트에서 의존성 설치.

```bash
python -m pip install -r requirements.txt
```

- CNN 가중치는 [Artifacts](artifacts/README.md) 안내에 따라 배치
- `model/detectors/`의 YuNet과 `face_landmarker.task` 사용
- Notebook을 열고 설치 환경을 커널로 선택한 뒤 위에서 아래로 실행
- 카메라와 가중치 경로는 Notebook의 설정 셀에서 지정

## Run

마지막 셀의 기본 실행 설정.

```python
summary = run_realtime_scores(
    max_seconds=180,
    threshold=0.6,
    perclos_window=60.0,
    perclos_threshold=0.15,
)
```

- 기본 **180초** 실행, 중단하려면 Notebook interrupt 사용
- 측정 영상과 얼굴 이미지는 저장하지 않고 지표와 세션 요약 출력
- 공개 Notebook의 저장 출력은 제거 상태
