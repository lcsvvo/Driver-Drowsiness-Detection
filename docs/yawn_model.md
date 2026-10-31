# Yawn Model

**MouthGate와 cnn_large를 결합한 최종 하품 경로**.

## Model

- 최종 파일: `yawn_yawn_mouthopen_v2__zoo-cnn_large__eval-face__gray128.keras`
- 입력은 **128×128 그레이스케일 얼굴 crop**
- `open_ratio > 0.05`인 입 벌림 프레임에서 CNN 실행, 최종 확률 기준은 **0.50**
- 하품 확률과 입 벌림 정도를 `YawnAccumulator`에서 시간 누적

## Saved Results

| Evaluation | Accuracy | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| 게이트 통과 Test 1,553장, 임계값 0.50 | 0.7521 | 0.6899 | 0.8118 | 0.7459 |
| DMD Test 1,131장, 게이트 포함 | 0.779 | 0.518 | 0.993 | — |

- 평가 범위와 라벨 구성이 달라 두 행의 직접 비교는 불가
- DMD의 손을 가리지 않은 하품 Recall **0.878**, 입을 가린 하품 Recall **0.162**
- 입 가림에서 게이트가 하품을 차단하는 한계 존재

## Selection

- 모델 비교와 증류 실험을 수행했으나 **최종 추론은 cnn_large 사용**
- 발표자료의 Teacher, Student 그림은 비교 실험 설계이며 현재 추론 구조 전체를 그대로 나타내지는 않음
- 입력 규격과 확률 기준은 [Artifacts](../model/artifacts/README.md) 참조
