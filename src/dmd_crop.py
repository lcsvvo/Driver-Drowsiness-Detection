"""DMD mosaic 좌하(bottom-left) 셀 크롭 — 눈·하품이 '같은 프레임'을 쓰게 하는 공용 함수.

팀원 DMD_Yawn/4_extract_bottomleft_avi.py, 5_extract_leftbottom_face_avi.py 의
셀 분할 방식(rows=2, cols=2, cell_row=1, cell_col=0)과 동일하다.
1280x720 mosaic 기준 frame[360:720, 0:640] = 정면 얼굴 타일.

프레임 인덱스는 mosaic = annotation = 팀원 frames_timestamps.csv 가 모두 같은 순서라,
frame_idx 하나로 눈 GT / 하품 GT / 팀원 수동 라벨을 붙일 수 있다 (정렬은 03에서 확인).
"""
from __future__ import annotations


def bottomleft_cell(frame, rows: int = 2, cols: int = 2,
                    cell_row: int = 1, cell_col: int = 0):
    """mosaic 프레임에서 좌하 셀만 잘라 반환. (팀원 4/5번과 같은 규격)"""
    h, w = frame.shape[:2]
    cw, ch = w // cols, h // rows
    x0, y0 = cell_col * cw, cell_row * ch
    return frame[y0:y0 + ch, x0:x0 + cw]
