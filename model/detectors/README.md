# Detectors

**얼굴 검출과 입 랜드마크 추정**에 사용하는 사전학습 모델.

| File | Content |
|---|---|
| `face_detection_yunet_2023mar.onnx` | YuNet 얼굴 검출 |
| `face_landmarker.task` | MediaPipe FaceLandmarker |

- 두 파일은 현재 저장소에 포함
- `face_landmarker.task`가 없으면 입 벌림 게이트가 비활성화될 수 있으므로 실행 전 파일 배치 필요
- CNN 가중치는 이 폴더가 아닌 [Artifacts](../artifacts/README.md)에 배치
- 실행 진입점은 [Model](../README.md) 참조
