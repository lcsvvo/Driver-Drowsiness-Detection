# Source

**전처리, 학습, 시간 누적 판정** 모듈 안내.

| File or Group | Content |
|---|---|
| [eye_preprocess.py](eye_preprocess.py), [build_dmd_eye_dataset.py](build_dmd_eye_dataset.py) | 눈 crop과 공통 입력 전처리 |
| [mrl_dataset.py](mrl_dataset.py), [mrl_split.py](mrl_split.py) | MRL 로딩과 피험자 기준 분할 |
| [train_eye.py](train_eye.py) | 최종 통합 manifest 기반 눈 모델 학습 |
| [train_eye_mrl.py](train_eye_mrl.py) | 과거 MRL 단독 학습 구현 |
| [yawn_dataset.py](yawn_dataset.py), [yawn_models.py](yawn_models.py) | 하품 데이터 로딩과 모델 구조 |
| [train_yawn.py](train_yawn.py), [train_yawn_zoo.py](train_yawn_zoo.py) | 하품 CNN과 모델 비교 학습 |
| [distill_yawn.py](distill_yawn.py), [compare_yawn_zoo.py](compare_yawn_zoo.py) | 지식 증류와 비교 결과 정리 |
| [mouth_gate.py](mouth_gate.py) | 입 벌림 측정과 CNN 게이트 |
| [yawn_accumulator.py](yawn_accumulator.py) | 하품 확률과 입 벌림의 시간 누적 |
| [dmd_annotation.py](dmd_annotation.py), [dmd_crop.py](dmd_crop.py) | DMD 라벨과 crop 처리 |

## Training

저장소 루트에서 통합 manifest를 사용한 눈 모델 학습 예시.

```bash
python src/train_eye.py --train mrl dmd --eval dmd
```

- 원본 영상과 crop은 별도 확보 필요. [Data](../data/README.md) 참조
- `train_eye_mrl.py`의 과거 경로는 최종 통합 학습 경로와 구분 필요
- 실시간 실행은 [Model](../model/README.md) 참조
