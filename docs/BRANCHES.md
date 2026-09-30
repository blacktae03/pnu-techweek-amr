# 브랜치 규칙 (2026-09-30 해커톤)

## 원칙
1. `main` 은 **항상 돌아가는 버전**. 직접 push 금지, PR 로만 합친다.
2. 브랜치 하나 = 담당 하나 = 건드리는 파일 목록 고정. 목록 밖 파일을 고쳐야 하면 먼저 팀에 말한다.
3. `webots_adapter.py` 는 공용 파일이라 **번호 붙은 블록 단위**로 소유를 나눈다 (아래 표). 다른 블록은 읽기만.
4. 합치기 전 판정: `worlds/apartment_gridnav.wbt` 로 실행해 로그의 (사과 발견 시각, 방문, 복귀 오차, 충돌) 를 PR 설명에 적는다.
5. 컨트롤러 파일(`controllers/tb3_gridnav/tb3_gridnav.py`)과 월드 파일은 통합 담당만 수정.

## 브랜치와 소유 범위

| 브랜치 | 담당 | 고칠 수 있는 파일 / 블록 | 목표 | 판정 |
|---|---|---|---|---|
| `feat/vision` | A조 | `grid_nav/vision.py`, `webots_adapter.py` **(3) 비전 블록**, `target_world_from_pixel`, `TARGET_*`/`MIN_APPLE_*` 상수 | 빨간 오탐 제거: 수평선 아래 조건, 반지름-세로위치 거리 일관성, 재탐지는 병합 | 진짜 빨간 사과 (−5.34, −10.54) 또는 (−12.02, −3.02) 에 도착 |
| `feat/exploration` | A조 | `grid_nav/frontier.py`, `grid_nav/occupancy_grid.py`, `grid_nav/astar.py`, `webots_adapter.py` **GridPlanner 클래스**, `SAFETY_MARGIN`/`RESOLUTION`/`REPLAN_PERIOD_S` | 사과 발견까지 시간 단축, 문 통과 안정 | 발견 시각(초), 정체 횟수 |
| `feat/dwa` | B조 | `grid_nav/dwa.py`, `webots_adapter.py` **(5) 속도 명령 블록**, `lookahead_control`, `to_wheel_speeds` | look-ahead 추종기를 DWA 로 교체, Pedestrian 회피 | 사람 지나가는 복도에서 충돌 0, 완주 시간 |
| `feat/localization` | B조 | `grid_nav/localization.py`, `webots_adapter.py` **(1) pose 블록**, `WheelOdometry`, `CompassHeading`, `COMPASS_SIGN` | scan matching 연결로 위치 오차 감소 | 로그 `위치오차` 평균/최대 (지금 0.02~0.12 m) |
| `integration` | 통합 담당 | 컨트롤러, 월드, README, 제출본(Supervisor 제거) | 각 feat 브랜치 합치고 전체 검증 | apartment 완주 |

## 작업 흐름
```bash
git clone https://github.com/blacktae03/pnu-techweek-amr.git && cd pnu-techweek-amr
git checkout feat/vision                 # 자기 브랜치
# ... 수정 → 실행 검증 ...
git add -A && git commit -m "vision: 수평선 아래 조건 추가"
git push
# GitHub 에서 main 으로 PR. 설명에 판정 수치 기록. 통합 담당이 확인 후 merge.
git checkout main && git pull            # 남이 합친 걸 받기
git checkout feat/vision && git merge main   # 내 브랜치에 반영 (충돌은 자기 블록만 남기면 됨)
```

## 실행 (모두 동일)
```bash
python prefetch_assets.py worlds/apartment.wbt          # 최초 1회, Webots 자산 캐시 채우기 (macOS 필수)
/Applications/Webots.app/Contents/MacOS/webots --stdout --stderr worlds/apartment_gridnav.wbt   # 개발용 (Supervisor, 지도 Display)
python grid_nav/view_live.py controllers/tb3_gridnav    # (선택) 카메라+지도 창
```
Webots Preferences > General > Python command 를 numpy/opencv/scipy 가 있는 파이썬으로 (예: conda ml_env).

## 제출본 체크리스트 (integration)
- [ ] 원본 `worlds/apartment.wbt` 의 controller 를 `tb3_gridnav` 로, `supervisor` 없음, Display 없음
- [ ] `tb3_gridnav.py` 에서 Supervisor 분기 제거, 시작 pose 상수 (−0.3, −7.5, π) 확인
- [ ] 대상 개수/색 (`TARGET_COUNT`, `TARGET_COLOR`) 을 당일 규칙에 맞춤
- [ ] 시간 예산 복귀 조건
