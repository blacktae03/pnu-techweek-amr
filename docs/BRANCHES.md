# 브랜치 규칙 (2026-09-30 해커톤)

## 원칙
1. `main` 은 **항상 돌아가는 버전**. 직접 push 금지, PR 로만 합친다.
2. 브랜치 하나 = 담당 하나 = 건드리는 파일 목록 고정. 목록 밖 파일을 고쳐야 하면 먼저 팀에 말한다.
3. `webots_adapter.py` 는 각 모듈을 부르기만 하는 조립 코드(integration 소유). 각 브랜치는 자기 모듈 파일만 고친다 → 충돌 없음.
4. 합치기 전 판정: `worlds/apartment_gridnav.wbt` 로 실행해 로그의 (사과 발견 시각, 방문, 복귀 오차, 충돌) 를 PR 설명에 적는다.
5. 컨트롤러 파일(`controllers/tb3_gridnav/tb3_gridnav.py`)과 월드 파일은 통합 담당만 수정.

## 브랜치와 소유 범위 (브랜치 = 파일. 서로 다른 파일만 고치므로 충돌이 나지 않는다)

| 브랜치 | 담당 | **소유 파일 (여기만 수정)** | 인터페이스 (바꾸지 말 것) | 목표 | 판정 |
|---|---|---|---|---|---|
| `feat/vision` | A조 | `grid_nav/target_detection.py`, `grid_nav/vision.py` | `TargetDetector(cam_w, cam_h, cam_fov, debug_dir)`, `.process(image_bytes, pose, t) -> (x,y)|None`, `.targets` | 빨간 오탐 제거 (수평선 아래 조건, 반지름-세로위치 거리 일관성, 재탐지 병합) | 진짜 빨간 사과 (−5.34, −10.54)/(−12.02, −3.02) 도착 |
| `feat/exploration` | A조 | `grid_nav/exploration.py`, `grid_nav/frontier.py`, `grid_nav/occupancy_grid.py`, `grid_nav/astar.py` | `GridPlanner.update_map / next_exploration_goal / plan / give_up_goal / get_map`, `.grid.log_odds` | 사과 발견까지 시간 단축, 문 통과 안정 | 발견 시각(초), 정체 횟수 |
| `feat/dwa` | B조 | `grid_nav/motion_control.py`, `grid_nav/dwa.py` | `MotionController().command(pose, path, ranges, angles, state) -> (v, w)`, `to_wheel_speeds(v, w)` | look-ahead 를 DWA 로 교체, Pedestrian 회피 | 사람 지나가는 복도에서 충돌 0, 완주 시간 |
| `feat/localization` | B조 | `grid_nav/pose_estimation.py`, `grid_nav/localization.py` | `PoseEstimator(start_pose, use_scan_matching, spec).update(phi_l, phi_r, compass, gyro_z, dt, ranges, angles, log_odds) -> pose` | scan matching 연결 (지금은 odometry+컴퍼스) | 로그 `위치오차` 평균/최대 (지금 0.02~0.12 m) |
| `integration` | 통합 담당 | `grid_nav/webots_adapter.py`, `grid_nav/robot_config.py`, `controllers/`, `worlds/`, `docs/`, README | 상태 기계·로그·Display·제출본 | feat 브랜치 합치고 전체 검증 | apartment 완주 |

공용(누구도 단독 수정 금지): `grid_nav/geometry.py`, `grid_nav/robot_config.py`. 바꿔야 하면 팀 채팅에 먼저.
`sim_demo.py`, `reactive.py`, `view_live.py` 는 학습/디버그용이라 자유롭게, 단 PR 은 따로.

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
