# Artifacts

**CNN 입력 규격과 기존 평가 지표**. 가중치 `.keras`는 Release에서 별도 확보.

## Weights

[Release weights-260823](https://github.com/lcsvvo/Driver-Drowsiness-Detection/releases/tag/weights-260823)에서 아래 파일을 받아 이 폴더에 배치.

| File | Content |
|---|---|
| `eye_mrl+dmd__eval-dmd__gray128.keras` | MRL과 DMD 합본 눈 모델 |
| `yawn_yawn_mouthopen_v2__zoo-cnn_large__eval-face__gray128.keras` | 최종 하품 모델 |

```bash
gh release download weights-260823 -R lcsvvo/Driver-Drowsiness-Detection -D model/artifacts
```

## Metrics

- **`_metrics.json`은 가중치와 같은 이름의 입력 규격, 클래스, 임계값, 평가 지표** 포함
- 눈 모델의 최종 임계값은 **0.93**, 하품 실시간 추론은 **0.50** 사용
- 하품 JSON의 `test_main`은 임계값 0.01 결과. **최종 0.50 결과는 `test_main_at_050`**에서 확인
- 하품 학습과 추론은 **입 벌림 기준 0.05**를 맞춰 사용
- 비교 모델과 증류 모델은 평가 기록 보존용. 최종 실시간 추론 모델과 구분 필요

자세한 평가 조건은 [Yawn Model](../../docs/yawn_model.md), 실행은 [Model](../README.md) 참조.
