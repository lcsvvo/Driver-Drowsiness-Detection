"""
yawn_mar_validate.py — 하품(MAR) 판정 검증용
목적: MAR threshold / min_seconds를 '결정'하기 위한 데이터 수집·분석.
이 단계에서는 위험도 지수와 연결하지 않는다. 하품 판정 자체만 검증한다.
설치: pip install mediapipe opencv-python numpy

============================================================
[중요] 아래 두 값은 '임시 시작값'일 뿐 검증된 기준이 아니다.
       반드시 record 로그를 analyze 로 분석해 조정해야 한다.
============================================================
"""
import argparse, csv, time, os
import numpy as np

# ---- 임시 시작값 (PROVISIONAL — 검증 후 변경 대상) --------------------
MAR_THRESHOLD_PROVISIONAL = 0.5   # TODO: analyze 결과로 조정
MIN_SECONDS_PROVISIONAL   = 1.5   # TODO: analyze 결과로 조정
EAR_THRESHOLD_PROVISIONAL = 0.20  # 참고용(눈 감김). 이번 검증 대상 아님
# ----------------------------------------------------------------------

# MediaPipe Face Mesh 468점 기준 입술/눈 인덱스 (확인된 값)
MOUTH = dict(in_top=13, in_bot=14, in_left=78, in_right=308,
             out_top=0, out_bot=17, out_left=61, out_right=291)
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]

# 검증할 6개 상황 (녹화 중 숫자키로 라벨 전환)
LABELS = {"1": "normal", "2": "yawn", "3": "brief_open",
          "4": "talking", "5": "laughing", "6": "eyes_closed"}


# ---------- 순수 계산 (mediapipe/cv2 불필요, 단독 테스트 가능) ----------
def _pt(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])

def _d(a, b):
    return float(np.linalg.norm(a - b))

def compute_mar(lm, w, h):
    p = lambda i: _pt(lm, i, w, h)
    inner = _d(p(MOUTH["in_top"]),  p(MOUTH["in_bot"]))  / _d(p(MOUTH["in_left"]),  p(MOUTH["in_right"]))
    outer = _d(p(MOUTH["out_top"]), p(MOUTH["out_bot"])) / _d(p(MOUTH["out_left"]), p(MOUTH["out_right"]))
    return 0.5 * (inner + outer)

def compute_ear(lm, idx, w, h):
    p = [_pt(lm, i, w, h) for i in idx]
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]))


class YawnDetector:
    """MAR >= threshold 가 min_seconds 이상 '연속' 지속되면 하품.
       라이브 피드백용일 뿐, 최종 기준값 결정은 analyze 로 한다."""
    def __init__(self, mar_threshold, min_seconds):
        self.mar_th = mar_threshold
        self.min_sec = min_seconds
        self.open_start = None
        self.yawn_count = 0
        self._latched = False

    def update(self, mar, now):
        if mar >= self.mar_th:
            if self.open_start is None:
                self.open_start = now
            duration = now - self.open_start
        else:
            self.open_start = None
            duration = 0.0
            self._latched = False
        is_yawn = duration >= self.min_sec
        if is_yawn and not self._latched:
            self.yawn_count += 1
            self._latched = True
        return dict(is_yawn=is_yawn, duration=duration, mar=mar, count=self.yawn_count)


# --------------------------- 녹화(웹캠) ---------------------------------
def record(out_csv, mar_th, min_sec, ear_th, start_label="normal", cam=0):
    import cv2, mediapipe as mp  # 무거운 import는 여기서만
    fm = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    det = YawnDetector(mar_th, min_sec)
    cap = cv2.VideoCapture(cam)

    label = start_label
    t0 = time.time()
    new_file = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline="")
    wr = csv.writer(f)
    if new_file:
        wr.writerow(["elapsed_s", "label", "mar", "ear",
                     "open_duration_s(prov)", "is_yawn(prov)"])

    print("[키] 1 normal | 2 yawn | 3 brief_open | 4 talking | 5 laughing | 6 eyes_closed | q 종료")
    print(f"[임시값] MAR_TH={mar_th}  MIN_SEC={min_sec}  (검증 후 조정)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time(); elapsed = now - t0

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            mar = compute_mar(lm, w, h)
            ear = 0.5 * (compute_ear(lm, L_EYE, w, h) + compute_ear(lm, R_EYE, w, h))
            r = det.update(mar, now)
            wr.writerow([f"{elapsed:.3f}", label, f"{mar:.4f}", f"{ear:.4f}",
                         f"{r['duration']:.3f}", int(r["is_yawn"])])

            color = (0, 0, 255) if r["is_yawn"] else (0, 255, 0)
            cv2.putText(frame, f"[{label}] MAR:{mar:.2f} dur:{r['duration']:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"EAR:{ear:.2f}  yawns:{r['count']}  (prov TH={mar_th}, {min_sec}s)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        else:
            cv2.putText(frame, "NO FACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("MAR validation (label with number keys)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        c = chr(k) if 0 <= k < 256 else ""
        if c in LABELS:
            label = LABELS[c]
            det = YawnDetector(mar_th, min_sec)  # 라벨 바꾸면 지속시간 리셋
            print(f"  -> label = {label}")

    f.close(); cap.release(); cv2.destroyAllWindows()
    print(f"저장됨: {out_csv}")


# --------------------------- 분석(로그) ---------------------------------
def _rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))

def _max_sustained(times, mars, th):
    """threshold th 로 연속 지속시간의 최댓값(초)."""
    best = run_start = 0.0
    open_now = False
    for t, m in zip(times, mars):
        if m >= th:
            if not open_now:
                open_now = True; run_start = t
            best = max(best, t - run_start)
        else:
            open_now = False
    return best

def analyze(csv_path):
    rows = _rows(csv_path)
    if not rows:
        print("로그가 비어있음"); return
    labels = {}
    for r in rows:
        labels.setdefault(r["label"], []).append(
            (float(r["elapsed_s"]), float(r["mar"])))

    print("\n=== 라벨별 MAR 분포 ===")
    print(f"{'label':12s}{'n':>5s}{'mean':>8s}{'p50':>8s}{'p95':>8s}{'max':>8s}")
    for lab, seq in labels.items():
        m = np.array([x[1] for x in seq])
        print(f"{lab:12s}{len(m):5d}{m.mean():8.3f}"
              f"{np.percentile(m,50):8.3f}{np.percentile(m,95):8.3f}{m.max():8.3f}")

    print("\n=== 후보 threshold별 최대 연속 지속시간(초) ===")
    cands = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
    print("th   " + "".join(f"{lab[:8]:>10s}" for lab in labels))
    for th in cands:
        line = f"{th:<5.2f}"
        for lab, seq in labels.items():
            ts = [x[0] for x in seq]; ms = [x[1] for x in seq]
            line += f"{_max_sustained(ts, ms, th):10.2f}"
        print(line)

    print("\n[해석 가이드]")
    print("- threshold: 'yawn' p50 이상은 넘되, talking/laughing/brief_open p95는 안 넘는 값(골짜기).")
    print("- min_seconds: 그 threshold에서 비하품 최대 지속시간 < min_seconds < 실제 yawn 지속시간.")


# ------------------------------ CLI -------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="MAR yawn validation")
    ap.add_argument("mode", choices=["record", "analyze"])
    ap.add_argument("--out", default="mar_log.csv")
    ap.add_argument("--mar-threshold", type=float, default=MAR_THRESHOLD_PROVISIONAL)
    ap.add_argument("--min-seconds", type=float, default=MIN_SECONDS_PROVISIONAL)
    ap.add_argument("--ear-threshold", type=float, default=EAR_THRESHOLD_PROVISIONAL)
    ap.add_argument("--label", default="normal", help="녹화 시작 라벨")
    ap.add_argument("--cam", type=int, default=0)
    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.mode == "record":
        record(args.out, args.mar_threshold, args.min_seconds,
               args.ear_threshold, args.label, args.cam)
    else:
        analyze(args.out)