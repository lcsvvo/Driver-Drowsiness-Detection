"""
config.py — 프로젝트 공통 경로 관리

이 모듈은 분석·탐지 로직을 담지 않는다. 팀원 4명이 각자 다른 PC에서
저장소를 clone해도 동일한 코드가 동작하도록 "경로만" 정의한다.

설계 원칙
1. PROJECT_ROOT는 이 파일의 위치(__file__)에서 결정한다.
   - Path.cwd()에 의존하지 않는다. VS Code나 노트북에서 작업 폴더가
     어디로 잡히든 결과가 달라지지 않는다.
   - 개인 PC의 특정 경로(C:\\Users\\..., Desktop, OneDrive 등)를 전혀
     참조하지 않으므로 저장소를 어디에 두든 동일하게 동작한다.
2. 지금 실제로 쓰는 경로만 정의한다.
   확정되지 않은 값(EAR/MAR 임계값, 프레임 크기, 카메라 인덱스 등)은
   넣지 않는다. 근거 없이 넣은 상수는 나중에 아무도 못 건드린다.
3. 무엇이 잡혔는지 확인할 수 있어야 한다. -> describe()

사용법:
    import config

    video_path = config.DATA_DIR / "yawdd" / "sample.avi"
    save_path  = config.OUTPUTS_DIR / "result.png"

    config.describe()      # 경로가 제대로 잡혔는지 점검

주의:
- 코드에 절대경로를 직접 쓰지 않는다. 경로는 전부 여기서 가져온다.
- 문자열을 이어붙이지 말고 pathlib의 "/" 연산자를 쓴다 (OS 호환).
"""

from pathlib import Path
import warnings

# =====================================================================
# 1. PROJECT_ROOT — 이 파일의 위치로 고정 (cwd 비의존)
# =====================================================================
#: 저장소 루트임을 확인하는 마커. 탐색 기준이 아니라 검증용이다.
ROOT_MARKERS: tuple[str, ...] = (".git", "README.md", ".gitignore")

#: resolve()로 심볼릭 링크·상대경로를 실제 경로로 정규화한다.
PROJECT_ROOT: Path = Path(__file__).resolve().parent

if not any((PROJECT_ROOT / m).exists() for m in ROOT_MARKERS):
    # config.py가 하위 폴더로 옮겨지면 DATA_DIR이 조용히 엉뚱한 곳을 가리킨다.
    # 조용히 넘어가지 않고 즉시 알린다.
    warnings.warn(
        f"PROJECT_ROOT({PROJECT_ROOT.name})에서 {ROOT_MARKERS} 중 어떤 마커도 "
        "찾지 못했습니다. config.py가 저장소 최상단에 있는지 확인하세요.",
        RuntimeWarning,
        stacklevel=2,
    )

# =====================================================================
# 2. 공통 경로
#    현재 실제로 필요한 두 개만 정의한다. 폴더가 늘어나면 그때 추가한다.
# =====================================================================

#: 데이터셋, 직접 촬영한 테스트 영상. git 추적 제외.
DATA_DIR: Path = PROJECT_ROOT / "data"

#: 실행 결과(로그, 캡처, 데모 영상). git 추적 제외.
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

_MANAGED_DIRS: tuple[Path, ...] = (DATA_DIR, OUTPUTS_DIR)


# =====================================================================
# 3. 유틸
# =====================================================================
def _rel(path: Path) -> str:
    """PROJECT_ROOT 기준 상대경로 문자열.

    출력에 개인 PC 절대경로를 남기지 않는다. 점검 결과를 PR·메신저·보고서에
    그대로 붙여도 사용자 계정명이 노출되지 않게 하기 위함이다.
    """
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    """DATA_DIR, OUTPUTS_DIR가 없으면 생성한다.

    git은 빈 폴더를 추적하지 않으므로 clone 직후에는 두 폴더가 존재하지 않는다.
    팀원마다 수동으로 만들게 하면 이름이 어긋나므로 import 시점에 자동
    생성한다. 이미 있으면 아무 일도 하지 않는다.
    """
    for directory in _MANAGED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


ensure_dirs()


# =====================================================================
# 4. 점검용
# =====================================================================
def describe() -> None:
    """경로가 제대로 잡혔는지 출력한다. 스크립트/노트북 첫 셀 점검용.

    절대경로가 필요하면 config.PROJECT_ROOT를 직접 참조한다.
    """
    print(f"PROJECT_ROOT : {PROJECT_ROOT.name}")
    for name, path in (("DATA_DIR", DATA_DIR), ("OUTPUTS_DIR", OUTPUTS_DIR)):
        mark = "OK" if path.is_dir() else "MISSING"
        print(f"  [{mark:<7}] {name:<12}: {_rel(path)}")


if __name__ == "__main__":
    describe()
