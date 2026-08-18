"""
mrl_split.py — MRL Eye Dataset 을 'subject 독립'으로 재분할하는 manifest 생성기.

왜 필요한가
- Kaggle 포크(현재 data/MRL Eye)는 이미지 단위 random split 이라 37명 전원이
  train/val/test 에 동시에 들어가 있다(실측 확인). 같은 사람 눈이 학습·평가에
  섞이면 test 성능이 과대평가된다.
- 이 스크립트는 파일명의 subject ID 로 사람 단위 분할을 만들어, 새 운전자
  (DMD)에 대한 일반화를 정직하게 측정할 수 있게 한다.

무엇을 만드나
- 이미지를 물리적으로 복사하지 않고 manifest CSV 만 만든다(8만 장 복제 방지).
  컬럼: path, subject, label(Closed/Open), class_idx(0/1), eyestate, gender,
        glasses, reflect, light, sensor, split
- 두 개를 만든다:
    mrl_manifest_subject.csv  : subject 독립 (권장, 본 실험용)
    mrl_manifest_random.csv   : 이미지 단위 random (leakage 영향 비교 실험용)

파일명 규칙 (실측 확인)
    s0001_01842_0_0_1_0_0_01.png
    = subject_imageid_gender_glasses_eyestate_reflect_light_sensor
    eyestate: 0=Closed(sleepy), 1=Open(awake)
    class_idx: Closed=0, Open=1  (기존 코드 class0=Closed 와 일치)
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import random
from collections import Counter, defaultdict

CLASS_IDX = {"Closed": 0, "Open": 1}
SPLITS = ("train", "val", "test")
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}


def parse_name(path: str, rel_path: str | None = None) -> dict | None:
    """파일명에서 메타데이터를 파싱. 규칙에 안 맞으면 None.

    path      : 실제 파일 경로(파싱용)
    rel_path  : manifest 에 저장할 경로. None 이면 path 그대로.
                이식성을 위해 MRL 루트 기준 상대경로를 넣는다.
    """
    fn = os.path.basename(path)
    stem = fn[:-4] if fn.lower().endswith(".png") else fn
    parts = stem.split("_")
    if len(parts) != 8:
        return None
    subject, imgid, gender, glasses, eyestate, reflect, light, sensor = parts
    label = "Closed" if eyestate == "0" else "Open"
    return {
        "path": rel_path if rel_path is not None else path,
        "subject": subject,
        "label": label,
        "class_idx": CLASS_IDX[label],
        "eyestate": eyestate,
        "gender": gender,
        "glasses": glasses,
        "reflect": reflect,
        "light": light,
        "sensor": sensor,
    }


def collect(mrl_root: str) -> list[dict]:
    """MRL 루트(.../MRL Eye/data) 아래 모든 png 를 파싱해 레코드 리스트로."""
    rows = []
    odd = 0
    for p in glob.glob(os.path.join(mrl_root, "**", "*.png"), recursive=True):
        rel = os.path.relpath(p, mrl_root).replace("\\", "/")  # OS 무관 상대경로
        r = parse_name(p, rel_path=rel)
        if r is None:
            odd += 1
            continue
        rows.append(r)
    if odd:
        print(f"[경고] 파일명 규칙 불일치 {odd}건은 제외했습니다.")
    if not rows:
        raise SystemExit(f"png 를 찾지 못했습니다: {mrl_root}")
    return rows


def subject_split(rows: list[dict]) -> dict[str, str]:
    """subject 단위 결정적 stratified 배정 (RNG 없음, 재현 100%).

    단순 크기 배정은 subject 편중 때문에 glasses/closed 비율이 split 마다
    틀어진다(예: 안경 비율 train 31% vs test 19%). 이를 막기 위해 각 subject 를
    (glasses x eyestate) 4개 층으로 분해하고, 4개 층 이미지 수를 split 목표 쿼터에
    맞춰 채운다. 사람은 여전히 한 split 에만 들어간다(leakage 0).

    층(strata): 0=안경&Closed, 1=안경&Open, 2=비안경&Closed, 3=비안경&Open
    배정: 이미지 수 내림차순으로, 어떤 층도 목표 쿼터를 넘지 않게(=4개 층 중
          최대 채움비가 가장 작은) split 을 고른다. 크기·안경·라벨이 동시에 균형.
    """
    # subject 별 4-strata 카운트
    strat: dict[str, list[int]] = {}
    per_subj = Counter()
    for r in rows:
        su = r["subject"]
        per_subj[su] += 1
        d = strat.setdefault(su, [0, 0, 0, 0])
        glass = (r["glasses"] == "1")
        closed = (r["eyestate"] == "0")
        idx = (0 if glass else 2) + (0 if closed else 1)
        d[idx] += 1

    stot = [sum(strat[su][k] for su in strat) for k in range(4)]
    quota = {s: [stot[k] * TARGET[s] for k in range(4)] for s in SPLITS}

    order = sorted(per_subj, key=lambda s: (-per_subj[s], s))  # 완전 결정적
    cur = {s: [0, 0, 0, 0] for s in SPLITS}
    assign = {}
    for su in order:
        d = strat[su]

        def fill(s):
            return max((cur[s][k] + d[k]) / quota[s][k] if quota[s][k] > 0 else 0.0
                       for k in range(4))

        best = min(SPLITS, key=fill)
        assign[su] = best
        for k in range(4):
            cur[best][k] += d[k]
    return assign


def random_split(rows: list[dict], seed: int = 42) -> list[str]:
    """이미지 단위 random split (leakage 재현용). 반환: rows 와 같은 순서의 split 라벨."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n = len(rows)
    n_tr = int(n * TARGET["train"])
    n_va = int(n * TARGET["val"])
    split_of = [""] * n
    for rank, i in enumerate(idx):
        if rank < n_tr:
            split_of[i] = "train"
        elif rank < n_tr + n_va:
            split_of[i] = "val"
        else:
            split_of[i] = "test"
    return split_of


def report(rows: list[dict], split_key="split"):
    """split 별 subject 수 / 이미지 수 / Closed·glasses 비율 + leakage 점검."""
    by_split_subj = defaultdict(set)
    cnt = Counter()
    closed = Counter()
    glass = Counter()
    for r in rows:
        s = r[split_key]
        by_split_subj[s].add(r["subject"])
        cnt[s] += 1
        if r["label"] == "Closed":
            closed[s] += 1
        if r["glasses"] == "1":
            glass[s] += 1
    all_subj = set().union(*by_split_subj.values()) if by_split_subj else set()
    leaked = [su for su in all_subj
              if sum(su in by_split_subj[s] for s in SPLITS) > 1]
    print(f"  [{split_key}]")
    for s in SPLITS:
        n = cnt[s]
        if n == 0:
            continue
        print(f"    {s:5s}: {len(by_split_subj[s]):2d} subj | {n:6d} img "
              f"({n/len(rows)*100:4.1f}%) | Closed {closed[s]/n*100:4.1f}% | "
              f"glasses {glass[s]/n*100:4.1f}%")
    print(f"    >1 split 에 걸친 subject(leakage): {len(leaked)} / {len(all_subj)}")


def write_csv(rows: list[dict], out_path: str, split_key="split"):
    cols = ["path", "subject", "label", "class_idx", "eyestate",
            "gender", "glasses", "reflect", "light", "sensor", "split"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in cols}
            row["split"] = r[split_key]
            w.writerow(row)
    print("  저장:", out_path, f"({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrl_root", required=True,
                    help=".../data/MRL Eye/data 경로 (train/val/test 상위)")
    ap.add_argument("--out_dir", default=".",
                    help="manifest CSV 저장 폴더")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = collect(args.mrl_root)
    print(f"총 이미지: {len(rows)}")
    print(f"subject 수: {len(set(r['subject'] for r in rows))}")

    # 1) subject 독립
    assign = subject_split(rows)
    for r in rows:
        r["split_subject"] = assign[r["subject"]]
    # 2) 이미지 random
    rand = random_split(rows, seed=args.seed)
    for r, s in zip(rows, rand):
        r["split_random"] = s

    print("\n== subject-independent split ==")
    report(rows, "split_subject")
    print("\n== random split (비교용) ==")
    report(rows, "split_random")

    os.makedirs(args.out_dir, exist_ok=True)
    print("\n== 저장 ==")
    write_csv(rows, os.path.join(args.out_dir, "mrl_manifest_subject.csv"),
              "split_subject")
    write_csv(rows, os.path.join(args.out_dir, "mrl_manifest_random.csv"),
              "split_random")

    # subject 배정 목록도 출력(리뷰용)
    print("\n== subject 배정 ==")
    for s in SPLITS:
        subs = sorted(su for su, sp in assign.items() if sp == s)
        print(f"  {s}: {' '.join(subs)}")


if __name__ == "__main__":
    main()
