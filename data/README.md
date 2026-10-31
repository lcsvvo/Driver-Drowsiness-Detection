# Data

**DMD, MRL Eye, YawDD 원자료**의 로컬 배치 안내.

| Dataset | Use |
|---|---|
| DMD | 눈 개폐, 얼굴 및 측면 영상의 하품 라벨 |
| MRL Eye | 눈 뜸과 감김 이미지 |
| YawDD | 하품과 말하기 등 비하품 영상 |

## Paths

- 원자료와 crop은 `data/` 아래 스크립트와 manifest가 참조하는 경로에 배치
- 공개 [eye_manifest.csv](../outputs/eye_dataset/eye_manifest.csv)의 `path`는 저장소 루트 기준 상대경로
- 이미지 파일이 manifest 경로와 일치하도록 준비한 뒤 학습 코드에서 사용

## Access

- **원본 영상, crop 이미지와 직접 촬영한 얼굴 영상은 공개 제외**
- 데이터는 원제공처의 이용 조건에 따라 별도 확보 필요
- 모델 가중치는 데이터 폴더가 아닌 [Artifacts](../model/artifacts/README.md)에 배치
