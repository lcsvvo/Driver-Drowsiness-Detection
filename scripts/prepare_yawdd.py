"""YawDD 원본 압축 해제 결과 -> 정규화된 raw 폴더 + 영상 목록(videos.csv).

압축을 푼 "YawDD dataset/" 은 그대로 쓰기 어렵다. 실제로 들어 있는 것:

  - 폴더 이름이 제각각      Mirror/Male_mirror Avi Videos, Dash/Female
  - 확장자가 두 번          11-FemaleGlasses.avi.avi
  - 파일명 끝에 공백        13-MaleNoGlasses .avi
  - 대소문자 흔들림         Talking&yawning / Talking&Yawning
  - 파일명에만 있는 속성    Beard, moustache, SunGlasses
  - Dash 는 라벨이 없다     한 영상 안에 normal/talking/yawning 이 섞여 있다

이 스크립트는 원본을 "옮겨서"(복사가 아니다 - 5GB를 두 벌 두지 않는다) 아래 구조로
만들고, 영상마다 실제로 열어 규격을 적어 둔다.

    raw/YawDD/
    ├── mirror/female/mF01-noglasses-normal.avi
    ├── mirror/male/  mM23-glasses-beard-talking_yawning.avi
    ├── dash/female/  dF01-noglasses-unlabeled.avi
    ├── dash/male/
    ├── docs/         Readme_YawDD.pdf, Table1.pdf, Table2.pdf
    └── metadata/
        ├── videos.csv            영상 1개 = 1행 (원본 파일명도 함께 남긴다)
        └── prepare_report.json   이상치 · 중복 · 열리지 않는 파일

파일명 규칙: <subject>-<eyewear>[-<facial_hair>]-<condition>.avi
    subject   m/d(mirror/dash) + F/M + 번호 2자리.  예: mF01, dM16
              **번호는 세트·성별 안에서만 유효하다.** Mirror 의 1번과 Dash 의 1번은
              다른 사람이고, Female 1번과 Male 1번도 다른 사람이다. 그래서 접두사를
              붙여 전역에서 유일한 ID 로 만든다.
    eyewear   noglasses / glasses / sunglasses
    condition normal / talking / yawning / talking_yawning / unlabeled(Dash)

split 도 여기서 정한다. **피험자 단위**이고 같은 사람의 다른 안경 조건은 반드시
같은 split 에 들어간다(안경만 바꾼 같은 얼굴이 train 과 test 에 나뉘면 test 가
오염된다). 2단계(build_yawdd_yawn_dataset.py)는 이 값을 그대로 읽어 쓴다.

사용
    python scripts/prepare_yawdd.py              # 정규화 + 검증 + videos.csv
    python scripts/prepare_yawdd.py --no-hash    # md5(중복 검사) 생략, 빠름
    python scripts/prepare_yawdd.py --probe-only # 파일은 건드리지 않고 다시 검사만
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

# =====================================================================
# 1. 경로
# =====================================================================
_HERE = Path(__file__).resolve().parent

#: Dataset 폴더는 저장소 밖에 있다(팀원마다 clone 위치가 다르다).
#: train_yawn.py 의 DATASET_CANDIDATES 와 같은 방식.
DATASET_CANDIDATES = (
    _HERE.parent / "data" / "Dataset",
    _HERE.parent.parent / "Dataset",
    _HERE.parent / "Dataset",
    _HERE.parent,                      # scripts/ 가 Dataset/ 바로 아래인 경우
)

#: 압축을 푼 원본 폴더 이름 후보(공백 포함이라 손으로 치기 번거롭다).
SRC_NAMES = ("YawDD dataset", "YawDD", "YawDD_dataset")

OUT_NAME = "YawDD"

SPLIT_RATIO = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_SEED = 42

VIDEO_COLUMNS = [
    "path", "dataset", "subject", "gender", "subject_no", "eyewear",
    "facial_hair", "condition", "is_yawn_video", "label_quality", "split",
    "width", "height", "fps", "frames_meta", "duration_s", "fourcc",
    "size_bytes", "md5", "readable", "original_path", "note",
]

# =====================================================================
# 2. 파일명 파싱
# =====================================================================
#: 349개 파일명을 전부 통과시킨 패턴. 조건 alternation 은 긴 것부터 둔다
#: (Talking 이 Talking&Yawning 을 먼저 먹으면 안 된다).
NAME_RE = re.compile(
    r"^(?P<no>\d+)\s*-\s*"
    r"(?P<gender>Male|Female)"
    r"(?P<eyewear>NoGlasses|SunGlasses|Glasses)"
    r"(?P<hair>Beard|Moustache)?"
    r"\s*(?:-\s*(?P<cond>Talking\s*&\s*Yawning|Normal|Talking|Yawning))?"
    r"\s*(?:\.avi)+$",
    re.IGNORECASE,
)

CONDITIONS = ("normal", "talking", "yawning", "talking_yawning", "unlabeled")

#: 영상 단위 라벨의 신뢰도. 프레임 단위로 그대로 쓸 수 있는지가 갈린다.
#:   strong_no_yawn : normal/talking - 이 영상의 **모든 프레임**이 하품이 아니다
#:   weak_yawn      : yawning - 하품이 들어 있지만 대부분의 프레임은 하품이 아니다
#:   weak_mixed     : talking&yawning - 말하기와 하품이 섞여 있다
#:   none           : Dash - 라벨 자체가 없다
LABEL_QUALITY = {
    "normal": "strong_no_yawn",
    "talking": "strong_no_yawn",
    "yawning": "weak_yawn",
    "talking_yawning": "weak_mixed",
    "unlabeled": "none",
}


def parse_name(name: str, is_mirror: bool) -> dict | None:
    """원본 파일명 -> 속성 dict. 형식에서 벗어나면 None."""
    m = NAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    cond = (g["cond"] or "").lower().replace(" ", "")
    cond = {"": "unlabeled", "talking&yawning": "talking_yawning"}.get(cond, cond)
    gender = g["gender"].capitalize()
    no = int(g["no"])
    return {
        "dataset": "mirror" if is_mirror else "dash",
        "subject": f"{'m' if is_mirror else 'd'}{gender[0]}{no:02d}",
        "gender": gender,
        "subject_no": no,
        "eyewear": g["eyewear"].lower(),
        "facial_hair": (g["hair"] or "").lower(),
        "condition": cond,
    }


def canonical_name(info: dict) -> str:
    parts = [info["subject"], info["eyewear"]]
    if info["facial_hair"]:
        parts.append(info["facial_hair"])
    parts.append(info["condition"])
    return "-".join(parts) + ".avi"


# =====================================================================
# 3. 폴더 찾기
# =====================================================================
def dataset_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "raw").is_dir():
            raise SystemExit(f"raw/ 가 없습니다: {p}")
        return p
    for c in DATASET_CANDIDATES:
        if (c / "raw").is_dir():
            return c.resolve()
    raise SystemExit(
        "Dataset 폴더를 찾지 못했습니다. 확인한 위치:\n  "
        + "\n  ".join(str(c) for c in DATASET_CANDIDATES)
        + "\n--dataset-root 로 직접 지정하세요.")


def find_src(raw: Path) -> Path | None:
    for n in SRC_NAMES:
        p = raw / n
        if p.is_dir() and p.name != OUT_NAME:
            return p
    return None


def is_mirror_path(p: Path) -> bool:
    return any(part.lower().startswith("mirror") for part in p.parts)


# =====================================================================
# 4. 정규화 - 원본을 새 구조로 옮긴다
# =====================================================================
def organize(src: Path, dst: Path) -> tuple[list[dict], list[str]]:
    """src 의 avi 를 dst 아래 정규화된 이름으로 옮긴다. (옮긴 목록, 경고)"""
    warnings: list[str] = []
    planned: dict[str, dict] = {}          # 새 상대경로 -> info

    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".pdf":
            continue
        if not p.name.lower().rstrip().endswith(".avi"):
            warnings.append(f"avi 도 pdf 도 아닌 파일을 건너뜀: {p.name}")
            continue

        mirror = is_mirror_path(p.relative_to(src))
        info = parse_name(p.name.strip(), mirror)
        if info is None:
            warnings.append(f"파일명 규칙에 맞지 않아 건너뜀: {p.relative_to(src).as_posix()}")
            continue

        rel = f"{info['dataset']}/{info['gender'].lower()}/{canonical_name(info)}"
        if rel in planned:
            # 같은 이름으로 두 파일이 겹치면 조용히 덮어쓰면 안 된다.
            warnings.append(
                f"이름 충돌: {p.relative_to(src).as_posix()} 와 "
                f"{planned[rel]['original_path']} 가 모두 {rel} 로 간다. 뒤엣것은 건너뜀")
            continue
        info["original_path"] = p.relative_to(src).as_posix()
        info["rel"] = rel
        info["_src"] = p
        planned[rel] = info

    for rel, info in planned.items():
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        shutil.move(str(info.pop("_src")), str(target))
    for info in planned.values():
        info.pop("_src", None)

    docs = dst / "docs"
    for p in sorted(src.rglob("*.pdf")):
        docs.mkdir(parents=True, exist_ok=True)
        target = docs / p.name
        if not target.exists():
            shutil.move(str(p), str(target))

    return list(planned.values()), warnings


def scan_existing(dst: Path) -> list[dict]:
    """이미 정규화된 폴더를 다시 읽는다(--probe-only / 재실행).

    original_path 는 파일명에서 되살릴 수 없다(정규화하면서 버린 정보다). 이전
    videos.csv 가 있으면 거기서 가져온다. 없으면 빈 값으로 두되, 그 사실이
    조용히 묻히지 않도록 note 에 남긴다.
    """
    previous: dict[str, str] = {}
    old_csv = dst / "metadata" / "videos.csv"
    if old_csv.exists():
        with open(old_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                previous[Path(row["path"]).name] = row.get("original_path", "")

    rows = []
    for p in sorted(dst.rglob("*.avi")):
        rel = p.relative_to(dst).as_posix()
        if len(rel.split("/")) != 3:
            continue
        info = parse_canonical(p.stem)
        if info is None:
            continue
        info["rel"] = rel
        info["original_path"] = previous.get(p.name, "")
        rows.append(info)
    return rows


def parse_canonical(stem: str) -> dict | None:
    """정규화된 이름 mF01-noglasses[-beard]-normal -> 속성 dict."""
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    subject, eyewear = parts[0], parts[1]
    hair = parts[2] if len(parts) == 4 else ""
    cond = parts[-1]
    if len(subject) < 3 or cond not in CONDITIONS:
        return None
    return {
        "dataset": "mirror" if subject[0] == "m" else "dash",
        "subject": subject,
        "gender": "Female" if subject[1] == "F" else "Male",
        "subject_no": int(subject[2:]),
        "eyewear": eyewear,
        "facial_hair": hair,
        "condition": cond,
    }


# =====================================================================
# 5. 검증 - 실제로 열어 본다
# =====================================================================
def probe(path: Path, want_hash: bool) -> dict:
    """영상을 열어 규격을 재고 첫 프레임 디코드까지 확인한다."""
    out = {
        "width": 0, "height": 0, "fps": 0.0, "frames_meta": 0,
        "duration_s": 0.0, "fourcc": "", "size_bytes": path.stat().st_size,
        "md5": "", "readable": 0, "note": "",
    }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        out["note"] = "열 수 없음"
        return out
    try:
        out["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out["fps"] = round(float(cap.get(cv2.CAP_PROP_FPS)), 3)
        out["frames_meta"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        out["fourcc"] = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip()
        ok, frame = cap.read()
        out["readable"] = int(bool(ok) and frame is not None)
        if not out["readable"]:
            out["note"] = "첫 프레임 디코드 실패"
    finally:
        cap.release()

    if out["fps"] > 0 and out["frames_meta"] > 0:
        out["duration_s"] = round(out["frames_meta"] / out["fps"], 2)
    if want_hash:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        out["md5"] = h.hexdigest()
    return out


# =====================================================================
# 6. split - 피험자 단위, 성별로 층화
# =====================================================================
def assign_splits(subjects: list[tuple[str, str]]) -> dict[str, str]:
    """[(subject, gender)] -> {subject: split}. Dash 는 라벨이 없어 'unused'."""
    out: dict[str, str] = {}
    by_gender: dict[str, list[str]] = defaultdict(list)
    for subj, gender in sorted(set(subjects)):
        if subj[0] == "d":
            out[subj] = "unused"
        else:
            by_gender[gender].append(subj)

    rng = random.Random(SPLIT_SEED)
    for gender, subs in sorted(by_gender.items()):
        subs = sorted(subs)
        rng.shuffle(subs)
        n = len(subs)
        n_tr = round(n * SPLIT_RATIO["train"])
        n_va = round(n * SPLIT_RATIO["val"])
        for i, s in enumerate(subs):
            out[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out


# =====================================================================
# 7. main
# =====================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=None, help="Dataset 폴더 경로")
    ap.add_argument("--no-hash", action="store_true", help="md5 중복 검사 생략")
    ap.add_argument("--probe-only", action="store_true",
                    help="파일을 옮기지 않고 이미 정규화된 폴더만 다시 검사")
    a = ap.parse_args()

    ds = dataset_dir(a.dataset_root)
    raw = ds / "raw"
    dst = raw / OUT_NAME
    print(f"Dataset : {ds}")

    warnings: list[str] = []
    if a.probe_only:
        if not dst.is_dir():
            raise SystemExit(f"정규화된 폴더가 없습니다: {dst}")
        rows = scan_existing(dst)
        print(f"기존 폴더에서 {len(rows)}개 영상을 읽었습니다: {dst}")
    else:
        src = find_src(raw)
        if src is None:
            if not dst.is_dir():
                raise SystemExit(
                    f"원본 폴더를 찾지 못했습니다. {raw} 아래에 "
                    f"{SRC_NAMES} 중 하나가 있어야 합니다.")
            rows = scan_existing(dst)
            print(f"원본 폴더가 없어 정규화는 건너뜁니다. 기존 {len(rows)}개를 검사합니다.")
        else:
            print(f"원본    : {src}")
            rows, warnings = organize(src, dst)
            print(f"정규화  : {len(rows)}개 영상을 {dst} 로 옮겼습니다.")
            leftovers = [p for p in src.rglob("*") if p.is_file()]
            if leftovers:
                warnings.append(
                    f"원본 폴더에 {len(leftovers)}개 파일이 남았습니다: "
                    + ", ".join(p.name for p in leftovers[:5]))
            else:
                for d in sorted(src.rglob("*"), reverse=True):
                    if d.is_dir():
                        d.rmdir()
                src.rmdir()

    if not rows:
        raise SystemExit("영상을 하나도 찾지 못했습니다.")

    splits = assign_splits([(r["subject"], r["gender"]) for r in rows])

    print(f"검증    : {len(rows)}개 영상을 열어 봅니다"
          f"{' (md5 포함)' if not a.no_hash else ''} ...")
    for i, r in enumerate(rows, 1):
        r.update(probe(dst / r["rel"], want_hash=not a.no_hash))
        r["split"] = splits[r["subject"]]
        r["is_yawn_video"] = int(r["condition"] in ("yawning", "talking_yawning"))
        r["label_quality"] = LABEL_QUALITY[r["condition"]]
        r["path"] = (dst / r["rel"]).resolve().relative_to(ds).as_posix()
        if not r["original_path"]:
            r["note"] = (r["note"] + "; " if r["note"] else "") + "원본 경로 정보 없음"
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    # 중복 - 내용이 같은 파일이 두 이름으로 들어있는지
    dups: list[list[str]] = []
    if not a.no_hash:
        by_md5: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            if r["md5"]:
                by_md5[r["md5"]].append(r["rel"])
        dups = [sorted(v) for v in by_md5.values() if len(v) > 1]

    meta = dst / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["rel"])
    with open(meta / "videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=VIDEO_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    unreadable = [r["rel"] for r in rows if not r["readable"]]
    by_cond = Counter(r["condition"] for r in rows)
    by_split = Counter(r["split"] for r in rows if r["dataset"] == "mirror")
    subjects = {r["subject"]: r["split"] for r in rows}
    sizes = Counter(f"{r['width']}x{r['height']}@{r['fps']}" for r in rows)

    report = {
        "videos": len(rows),
        "subjects": len(subjects),
        "by_condition": dict(by_cond),
        "by_split_videos": dict(by_split),
        "by_split_subjects": dict(Counter(
            s for subj, s in subjects.items() if subj[0] == "m")),
        "resolutions": dict(sizes),
        "unreadable": unreadable,
        "duplicate_groups": dups,
        "warnings": warnings,
        "split": {"by": "subject", "ratio": SPLIT_RATIO, "seed": SPLIT_SEED,
                  "map": {k: v for k, v in sorted(subjects.items())}},
    }
    with open(meta / "prepare_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n영상 {len(rows)}개 / 피험자 {len(subjects)}명")
    for k in CONDITIONS:
        if by_cond.get(k):
            print(f"  {k:<16} {by_cond[k]:4d}")
    print("  mirror split(영상):", dict(by_split))
    print("  해상도:", dict(sizes))
    if unreadable:
        print(f"  [경고] 열리지 않는 영상 {len(unreadable)}개: {unreadable[:5]}")
    if dups:
        print(f"  [경고] 내용이 같은 파일 묶음 {len(dups)}개: {dups[:3]}")
    for w_ in warnings:
        print("  [경고]", w_)
    print(f"\n기록: {(meta / 'videos.csv')}")
    print(f"      {(meta / 'prepare_report.json')}")


if __name__ == "__main__":
    main()
