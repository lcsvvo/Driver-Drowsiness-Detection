# Scripts

**DMD, YawDD의 하품 학습 자료 구성** 스크립트.

| Script | Content |
|---|---|
| [prepare_yawdd.py](prepare_yawdd.py) | YawDD 원본 준비 |
| [build_yawdd_yawn_dataset.py](build_yawdd_yawn_dataset.py) | YawDD 하품 crop과 라벨 구성 |
| [build_dmd_yawn_dataset.py](build_dmd_yawn_dataset.py) | DMD 하품 crop과 라벨 구성 |
| [build_yawn_mouthopen_dataset.py](build_yawn_mouthopen_dataset.py) | 입 벌림 프레임 기준 학습 자료 구성 |
| [dmd_yawn_gt.py](dmd_yawn_gt.py) | DMD 하품 정답 구간 처리 |

- 스크립트별 입력 경로와 옵션은 각 파일의 설명 참조
- 원본과 crop은 공개 제외. 배치 기준은 [Data](../data/README.md) 참조
- 학습 모듈은 [Source](../src/README.md), 최종 실행은 [Model](../model/README.md) 참조
