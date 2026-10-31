# Eye Dataset

**통합 눈 학습 manifest와 저장된 비교 결과**.

| File | Content |
|---|---|
| [eye_manifest.csv](eye_manifest.csv) | 이미지 경로, 데이터 출처, 피험자, 라벨과 split |
| [class_balance.csv](class_balance.csv) | 출처 및 split별 클래스 분포 |
| [leakage_report.csv](leakage_report.csv) | 기존 split 점검 기록 |
| [step12_comparison.csv](step12_comparison.csv) | MRL, DMD 학습 조건별 저장 평가값 |

- manifest에는 이미지 자체가 포함되지 않음
- DMD 평가는 좌우 눈 확률을 **프레임 단위로 결합**. manifest crop 개수와 평가 프레임 수는 구분 필요
- `train_eye.py`가 이 manifest의 split을 사용. 재실행 시 [Data](../../data/README.md) 경로에 이미지 배치 필요
