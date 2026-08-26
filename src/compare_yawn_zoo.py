"""
compare_yawn_zoo.py — model zoo + distillation 결과를 한 표로 모은다.

train_yawn.py, train_yawn_zoo.py, distill_yawn.py 가 저장한 model/artifacts 의
yawn_*_metrics.json 을 전부 읽는다(train_yawn.load_all_metrics() 재사용 — 세 스크립트
모두 같은 파일명 규칙(yawn_{tag}.keras / _metrics.json)을 쓰므로 산출물을 구분 없이
다 읽는다). 이 스크립트가 그 위에 하는 일은 세 가지다.

1. "무엇을(arch) 어떻게 학습했나(scratch/distilled)"로 표를 정리
2. distilled 항목의 teacher_tags 를 역참조해서, scratch 항목 중 실제로 teacher 로
   쓰인 것을 role="teacher(used)" 로 표시(나머지 scratch 는 아직 안 쓰인 후보/기준점)
3. 같은 (dataset, arch, eval_view) 의 scratch 대비 distilled 의 성능 변화량(delta) 계산
   — "distillation이 실제로 도움이 됐는가"에 대한 답.

파라미터 수·모델 용량(MB)도 같이 낸다. params 필드가 없는 옛 train_yawn.py 산출물은
.keras 파일을 열어 count_params() 로 채운다(느리면 --no-load-missing-params 로 끈다).

사용 (저장소 루트에서)
    python src/compare_yawn_zoo.py                     # 표를 표준출력에 markdown 으로
    python src/compare_yawn_zoo.py --csv outputs/yawn_zoo_compare.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_yawn as ty


def _params(tag: str, d: dict, load_missing: bool = True):
    if d.get("params"):
        return int(d["params"])
    if not load_missing:
        return None
    model_path = ty.ARTIFACT_DIR / f"yawn_{tag}.keras"
    if not model_path.exists():
        return None
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(model_path)
        return int(m.count_params())
    except Exception as e:
        print(f"[건너뜀] {tag}: params 계산 실패 ({e})")
        return None


def _size_mb(tag: str):
    model_path = ty.ARTIFACT_DIR / f"yawn_{tag}.keras"
    if not model_path.exists():
        return None
    return round(model_path.stat().st_size / (1024 * 1024), 3)


def build_table(include_smoke: bool = False, load_missing_params: bool = True) -> pd.DataFrame:
    results = ty.load_all_metrics(include_smoke=include_smoke)

    # distillation 에 실제로 쓰인 teacher tag 집합 — scratch 모델의 역할 표시에 쓴다.
    used_as_teacher: set[str] = set()
    for d in results.values():
        for t in d.get("teacher_tags") or []:
            used_as_teacher.add(t)

    rows = []
    for tag, d in results.items():
        m = d.get("test_main")
        if not isinstance(m, dict) or "yawn_recall" not in m:
            continue
        # val 에서 고른 임계값(test_main)은 저확률 쪽으로 포화돼 낙관적인 경우가 많다
        # (docs/yawn_model.md §3 — 하품 모델은 실제로 0.50 을 쓴다). 그래서 실사용
        # 임계값인 @0.50(test_main_at_050) 도 같이 낸다 — collapse(전부 yawn/전부
        # no_yawn 로 찍는) 여부는 이 컬럼에서만 드러난다.
        m50 = d.get("test_main_at_050") or {}
        training = d.get("training", "scratch")
        arch = d.get("arch", "cnn_small")   # 옛 train_yawn.py 산출물은 build_yawn_model=cnn_small
        role = ("distilled" if training == "distilled"
                else "teacher(used)" if tag in used_as_teacher
                else "scratch")
        rows.append(dict(
            tag=tag, dataset=d.get("dataset", "processed"), arch=arch,
            zoo_kind=d.get("zoo_kind", "cnn"), role=role,
            params=_params(tag, d, load_missing_params), size_mb=_size_mb(tag),
            eval_view=d.get("eval_view"),
            yawn_recall=m.get("yawn_recall"), yawn_precision=m.get("yawn_precision"),
            yawn_f1=m.get("yawn_f1"), accuracy=m.get("accuracy"),
            precision_at_real_prior=m.get("precision_at_real_prior"),
            recall_050=m50.get("yawn_recall"), precision_050=m50.get("yawn_precision"),
            f1_050=m50.get("yawn_f1"), accuracy_050=m50.get("accuracy"),
            subj_acc_std=(d.get("subject_spread") or {}).get("accuracy_std"),
            teacher=d.get("teacher_label"),
            temperature=d.get("temperature"), alpha=d.get("alpha"),
        ))
    t = pd.DataFrame(rows)
    if t.empty:
        return t

    # scratch(role != distilled) 대비 distilled 의 변화량 — 같은 (dataset, arch, eval_view)
    # 기준. 같은 키에 scratch 가 여럿이면 마지막으로 읽힌 것을 쓴다(태그 알파벳순).
    scratch_lookup = {}
    for _, r in t[t.role != "distilled"].iterrows():
        scratch_lookup[(r.dataset, r.arch, r.eval_view)] = r

    def _delta(row, col):
        if row.role != "distilled":
            return None
        base = scratch_lookup.get((row.dataset, row.arch, row.eval_view))
        if base is None or pd.isna(row[col]) or pd.isna(base[col]):
            return None
        return round(row[col] - base[col], 4)

    for col in ("yawn_recall", "yawn_precision", "yawn_f1", "accuracy",
               "recall_050", "precision_050", "f1_050", "accuracy_050"):
        t[f"delta_{col}_vs_scratch"] = t.apply(lambda r: _delta(r, col), axis=1)

    return t.sort_values(["dataset", "role", "params"], na_position="last").reset_index(drop=True)


def _df_to_markdown(t: pd.DataFrame) -> str:
    """pandas.to_markdown() 은 선택적 의존성(tabulate)이 필요해 직접 그린다."""
    cols = list(t.columns)
    sub = t.copy()
    for c in cols:
        sub[c] = sub[c].apply(lambda v: "" if pd.isna(v) else str(v))
    widths = {c: max(len(c), int(sub[c].str.len().max()) if len(sub) else 0) for c in cols}

    def _row(vals):
        return "| " + " | ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals)) + " |"

    lines = [_row(cols), "| " + " | ".join("-" * widths[c] for c in cols) + " |"]
    for _, r in sub.iterrows():
        lines.append(_row(r.tolist()))
    return "\n".join(lines)


def to_markdown(t: pd.DataFrame) -> str:
    # @0.50(실사용 임계값)을 앞에 낸다 — val-threshold(yawn_recall 등)는 포화돼 낙관적일
    # 수 있다(docs/yawn_model.md §3). collapse(전부 yawn/전부 no_yawn) 여부는 @0.50 에서만 보인다.
    cols = ["tag", "dataset", "arch", "zoo_kind", "role", "params", "size_mb",
           "accuracy_050", "recall_050", "precision_050", "f1_050",
           "delta_f1_050_vs_scratch", "delta_accuracy_050_vs_scratch",
           "yawn_recall", "yawn_precision", "yawn_f1", "accuracy",
           "precision_at_real_prior", "subj_acc_std",
           "teacher", "temperature", "alpha"]
    cols = [c for c in cols if c in t.columns]
    return _df_to_markdown(t[cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV 로도 저장할 경로")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--no-load-missing-params", action="store_true",
                    help="params 필드가 없는 옛 산출물의 .keras 를 열어보지 않는다(더 빠름)")
    a = ap.parse_args()

    t = build_table(include_smoke=a.include_smoke,
                    load_missing_params=not a.no_load_missing_params)
    if t.empty:
        print("model/artifacts 에 metrics json 이 없습니다. 먼저 학습하세요.")
        return
    print(to_markdown(t))
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(a.csv, index=False)
        print("\n저장:", a.csv)


if __name__ == "__main__":
    main()
