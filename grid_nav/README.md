# grid_nav — AMR Search & Rescue 해커톤 "격자 파트" 학습 코드

Webots 없이 **NumPy + matplotlib (+ scipy)** 만으로 돌아가는, 지도 작성 → frontier 탐색 → A* 경로 계획 학습용 코드입니다.
대회 조건(사전 지도 없음, 시작 pose 제공, 대상 위치 미지, 충돌 금지, 복귀)을 가짜 2D 세계에서 그대로 재현합니다.

```
grid_nav/
├── geometry.py        좌표 변환 (LiDAR→로봇→월드→격자), 각도 wrap, Bresenham
├── occupancy_grid.py  log-odds 점유 격자 지도, inf 처리, inflation, 벽 거리 지도
├── frontier.py        frontier 검출 → 클러스터 → 목표 선택 (+ BFS 경로 거리)
├── astar.py           8방향 A*, 코너커팅 금지, 벽 근접 비용, 경로 직선화, 월드 좌표 출력
├── reactive.py        강의 Algorithm 1·2·3 (반응형 회피 + MOVE/TURN FSM + 코너 탈출). "여기서 시작"
├── localization.py    (팀원 파트 1) 바퀴 odometry + scan matching (correlative 기본, ICP 참고용)
├── dwa.py             (팀원 파트 6) DWA 국소 회피: 속도 후보 → 궤적 시뮬 → critic 점수
├── sim_demo.py        가짜 세계 + 레이캐스트 LiDAR + 탐색/수색/대상방문/복귀 애니메이션
└── webots_adapter.py  Webots 컨트롤러에 끼우는 뼈대 (당일 API 에 맞춰 수정)
```

## 실행 방법

```bash
pip install numpy matplotlib scipy          # scipy 없어도 느린 대체 구현으로 동작

cd grid_nav
python geometry.py          # 각 모듈은 단독 실행하면 자체 테스트 + 그림
python occupancy_grid.py
python frontier.py
python astar.py

python sim_demo.py                        # 실시간 애니메이션 (닫으면 종료)
python sim_demo.py --speed 5              # 5스텝마다 한 번 그려서 빠르게
python sim_demo.py --headless             # 화면 없이 끝까지 돌리고 demo_result.png 저장
python sim_demo.py --seed 3 --margin 0.2 --w-prox 5   # 파라미터 실험
python sim_demo.py --dwa                  # 단순 추종기 대신 DWA 로 경로 추종 + 실시간 회피
python sim_demo.py --pedestrian --dwa     # 움직이는 사람 추가 (DWA 없으면 충돌 발생)
python sim_demo.py --localize             # 진짜 위치 대신 노이즈 odometry + scan matching 추정 위치 사용
python sim_demo.py --localize --dwa --pedestrian   # 실전에 가장 가까운 조건
python sim_demo.py --unified              # EXPLORE 와 SEARCH 를 한 단계로 합쳐 비교
python sim_demo.py --reactive             # 강의 Algorithm 3 으로 돌아다니며 지도만 만들고 1500스텝 뒤 A* 복귀 (frontier 와 비교용)
python sim_demo.py --lookahead 0.8        # 경로 추종 look-ahead 거리 실험
python sim_demo.py -h                     # 모든 옵션
```

검증 결과 (headless, seed 0):

| 조건 | 대상 | 충돌 | 복귀 오차 | 비고 |
|---|---|---|---|---|
| 기본 (진짜 위치, 단순 추종) | 3/3 | 0 | 0.18 m | |
| `--pedestrian` | 3/3 | 3 | 0.20 m | 회피 없음 → 사람이 부딛힘 |
| `--pedestrian --dwa` | 3/3 | 0 | 0.20 m | DWA 가 사람을 피함 |
| `--localize` | 3/3 | 0 | 0.34 m | 위치 오차 평균 0.18 m |
| `--localize --dwa --pedestrian` | 3/3 | 0 | 0.51 m | 위치 오차 평균 0.24 m, 최대 0.51 m |
| `--reactive` (1500스텝 반응형 후 복귀) | 0/3 | 0 | 0.18 m | 지도 10% 만 채움. frontier 가 왜 필요한지 보여주는 대조군 |

Webots 에서 바로 돌릴 첫 컨트롤러: `PNU-TECHWEEK-260930/controllers/tb3_reactive/tb3_reactive.py` (Algorithm 3 + 컴퍼스 yaw). 월드의 TurtleBot3Burger controller 를 `tb3_reactive` 로 바꾸면 됨.

검증 결과(seed 0/1/2, headless, Python 3.14 및 3.10+numpy 1.23.5): 대상 3/3 발견·방문, **충돌 0회**, 시작점 복귀(오차 ≤0.2 m), 스텝당 6~8 ms.

## 추천 공부 순서 (약 3시간 배분)

| 순서 | 파일 | 시간 | 이걸 알면 끝 |
|---|---|---|---|
| 1 | `geometry.py` | 30분 | `(row, col)` ↔ `(x, y)`, `pose` 로 회전+이동, `wrap_angle` 이 왜 필요한지 |
| 2 | `occupancy_grid.py` | 40분 | log-odds 누적, "지나간 칸 = 빈칸 / 끝점 = 벽", inf 빔 처리, inflation 이 곧 안전거리 |
| 3 | `frontier.py` | 30분 | frontier 정의 한 줄, 클러스터를 쓰는 이유, 점수 함수의 가중치 감각 |
| 4 | `astar.py` | 40분 | f = g + h, 8방향/octile, 코너커팅 금지, 벽 근접 비용, `plan_path()` 인터페이스 |
| 5 | `sim_demo.py` | 40분 | 전체 루프 순서(스캔→지도→목표→경로→추종), 상태 기계, 재계획/정체/블랙리스트 |
| 6 | `webots_adapter.py` | 20분 | `robot.step()` 루프, `getDevice/enable`, 어디에 우리 모듈을 끼우는지 |

**공부 팁**: 각 모듈 `__main__` 의 숫자를 바꿔 보세요. 예) inflation 반지름을 0 으로 → A* 경로가 벽에 붙는다.
`--w-prox 0` → 로봇이 문 모서리를 스친다. `--min-frontier 30` → 좁은 문 너머를 안 간다.

## 모듈별 핵심 개념 요약

### geometry.py
- **좌표계 4개**: LiDAR 극좌표 → 로봇(x 앞, y 왼쪽) → 월드(시작점 기준 고정) → 격자(row=y, col=x).
- `robot_to_world`: `p_w = R(θ) p_r + t`. 행벡터 배열이라 `pts @ R.T + t`. 역변환은 `(pts - t) @ R`.
- `GridSpec`: 해상도 + 원점(**(0,0) 칸의 왼쪽-아래 모서리**) + 크기. `world_to_grid` 는 **floor** (int 절삭 아님).
- `wrap_angle`: `(a + π) % 2π − π`. 각도 차이를 계산한 직후 반드시 호출.
- `bresenham_line`: 두 칸을 잇는 격자 칸 열거. 지도 업데이트(교과서 방식)와 경로 직선화의 line-of-sight 검사에 사용.

### occupancy_grid.py
- 칸마다 `L = log(p/(1−p))`. hit 이면 `+l_hit`, 통과했으면 `+l_miss`(음수). `L=0` 은 미탐색. `±l_clamp` 로 상한 → 움직이는 사람이 지나간 자리가 다시 빈칸으로 돌아올 수 있음.
- **벡터화 업데이트**: 빔마다 `t∈[0,1]` 로 점을 촘촘히 찍어 `(N빔, S점, 2)` 배열을 한 번에 격자 인덱스로. `np.unique` 로 한 스캔 내 중복 제거(안 하면 로봇 옆 칸이 한 번에 -100).
- **inf 처리**: finite 이고 `< max_range` 면 hit. 아니면 `max_range*0.98` 까지 빈칸만 표시, 끝점은 벽으로 찍지 않음.
- `to_ternary()`: 팀 형식 `0/1/-1`. 임계값은 log-odds 로 비교.
- `inflate()`: 벽을 `로봇 반지름 + 여유` 만큼 원형 커널로 팽창(binary dilation). 이후 로봇은 점으로 취급 가능. **심사 항목 "안전거리 확보" 가 여기서 결정됨.**
- `obstacle_distance_map()`: 벽까지 거리(m). A* 비용에 사용.

### frontier.py
- **frontier** = 빈칸(0)이면서 8방향 이웃에 미탐색(-1)이 있는 칸. 팽창 지도에서 1 인 칸은 제외(설 수 없는 곳).
- 8번의 슬라이스 OR 로 완전 벡터화. 지도 바깥은 미탐색으로 안 침(지도 끝으로 달려가는 것 방지).
- `cluster_frontiers`: `ndimage.label` 8-연결. `min_size` 로 노이즈 덩어리 제거. 목표 칸 = centroid 에 가장 가까운 **실제 frontier 칸**(centroid 는 벽 위일 수 있음).
- `select_frontier`: `score = w_size·min(size, cap) − w_dist·거리 − w_turn·|heading차|`. 블랙리스트(실패한 목표) 제외. `bfs_distance_map` 을 주면 벽을 고려한 경로 거리 사용.
- 반환 `None` = **탐색 완료** 신호.

### astar.py
- `f = g + h`, `h` 는 octile 거리(8방향에서 admissible). heapq + 2D 배열(`g`, `parent`, `closed`).
- 코너 커팅 금지: 대각선 이동 시 양옆 직교 칸도 통과 가능해야 함.
- 이동 비용 `= 기하거리 × (1 + cost_map)`, `cost_map = w_prox·(1 − d/safe_dist)⁺`. 벽에 붙은 칸은 `(1+w_prox)` 배 비쌈 → 복도 한가운데로.
- `nearest_passable`: 시작/목표가 팽창 벽 안이면 가장 가까운 통과 가능 칸으로 스냅(벽에 붙은 대상, 벽 옆에 선 로봇).
- `shortcut_path`: line-of-sight 로 계단 경로를 직선화. 비용 높은 칸(벽 근처)을 지나는 직선은 불허.
- `plan_path(inflated, spec, start_xy, goal_xy)` → `[(x, y), ...]` 또는 `None`. **팀원이 부르는 함수는 이것 하나.**

### reactive.py (슬라이드 110~112쪽 의사코드)
- Alg 1: `d_F > threshold` 면 직진, 아니면 `d_L > d_R` 인 쪽으로 제자리 회전. 상태 없음 → 좌우가 번갈아 뽑히면 제자리에서 떨다 멈춤 (테스트에서 1.1 m 만 이동).
- Alg 2: MOVE/TURN FSM. 막힌 순간 `target_yaw = yaw + dir·ANGLE` 을 한 번 정하고 도달까지 TURN 유지. **`|target − yaw|` 는 반드시 wrap_angle** (슬라이드엔 없음).
- Alg 3: 연속 TURN 횟수 `turn_cnt > MAX_CNT` 면 EscapeBehavior(후진 → 넓은 쪽으로 150° 회전). 슬라이드에 정의가 없어 우리가 정한 것. "직진 시 turn_cnt=0" 을 그대로 두면 코너에서 매번 리셋되어 탈출이 안 걸려서, 15스텝 이상 직진했을 때만 리셋.
- `lidar_sectors`: 정면/왼/오 ±15° 구간의 **최소** 거리. 평균은 좁은 기둥을 놓친다.

### localization.py (팀원 파트이지만 우리 지도의 입력)
- 예측: 엔코더 → `ds=(dl+dr)/2`, `dθ=(dr−dl)/L`, 회전 중간값으로 적분. 방향 오차가 가장 치명적이라 자이로/컴퍼스로 dθ 를 대체하는 옵션.
- 보정(기본) `CorrelativeMatcher`: 예측 pose 주변 (dx, dy, dθ) 후보 격자(±0.1 m/2 cm, ±3°/0.5°)에서 스캔 점이 "확실한 벽"(log-odds > 2) 위에 가장 많이 얹히는 후보 선택. odometry 에서 멀어질수록 작은 벌점 → 복도에서 미끄러짐 방지. 2 ms/회.
- 보정(참고) `ScanMatcher`(ICP): 점-점 짝짓기 + SVD. 교과서 방식이지만 긴 벽에서 미끄러져 테스트에서 오차가 오히려 커짐(0.48 m). 왜 correlative 를 쓰는지 보여주는 반례.
- **순서**: odometry 예측 → 보정 → 보정된 pose 로 지도 갱신. 지도를 먼저 갱신하면 틀린 벽에 맞추는 되먹임.

### dwa.py (팀원 파트이지만 우리 경로의 소비자)
- 가감속 한계 안의 (v, ω) 후보 7×15 → 1.5 s 등속 원호 궤적 → 충돌 탈락 → heading/거리/clearance/속도 critic 합산 최소.
- 장애물은 지도가 아니라 **지금 LiDAR 점**. 지도에 굳지 않은 점(사람 후보)은 `dyn_margin` 만큼 더 멀리 피함. 전부 충돌이면 가장 멀어지는 궤적(후진).
- 임시 목표는 A* 경로의 look-ahead 점(`path_lookahead_point`). 튠 포인트: `clearance_max` 크면 좁은 문 앞에서 진동.

### sim_demo.py
- 상태 기계: `EXPLORE`(frontier) → `SEARCH`(카메라 미확인 빈칸 수색) → `VISIT`(발견 대상 방문) → `RETURN` → `DONE`.
- **SEARCH 가 왜 필요한가**: LiDAR 는 5 m 밖 방까지 지도를 다 그리지만 카메라는 2~3 m 안만 봄. "지도 완성 ≠ 대상 다 봄". LiDAR 빔을 카메라 사거리로 잘라 `seen` 층을 만들고, 안 본 빈칸 덩어리를 frontier 와 같은 방식으로 방문.
- 루프: 매 스텝 스캔+지도 갱신 → N 스텝마다 재계획(inflate + frontier + A*) → 5 스텝마다 경로가 새 벽에 막혔는지 검사 → 정체 감지 시 목표 블랙리스트 → 정면 LiDAR 최소거리로 비상정지.
- 진짜 세계와의 충돌을 스스로 채점(`충돌 N회`). 팽창/비용이 제대로면 0 이어야 함.

## 팀 인터페이스 (약속)

| 항목 | 형식 |
|---|---|
| 로봇 자세 | `(x, y, theta)` 미터·라디안, theta 는 월드 x축 기준 반시계 + |
| 지도 | `int8` 2D NumPy, `grid[row, col]`, `0 빈칸 / 1 벽 / -1 미탐색` + `GridSpec(resolution, origin_x, origin_y, rows, cols)` |
| 경로 | `[(x, y), ...]` 월드 좌표, 첫 점 = 현재 로봇 위치 |
| 격자 파트 API | `GridPlanner.update_map / get_map / next_exploration_goal / plan / give_up_goal` (webots_adapter.py) |

## 당일 흔히 막히는 포인트 (체크리스트)

1. **row/col 과 x/y 뒤바뀜** — `grid[row, col]`, `row ↔ y`, `col ↔ x`. 지도가 대각선으로 뒤집혀 보이면 이것. `imshow(..., origin='lower', extent=spec.extent)` 로 그려야 월드축과 맞음(기본 `origin='upper'` 는 y 가 아래로 증가).
2. **격자 원점이 칸 중심인지 모서리인지** — 이 코드는 모서리. 팀원 코드와 다르면 반 칸씩 어긋남. `world_to_grid` 는 `floor` 사용.
3. **각도 wrap 누락** — 목표 방향 − 현재 방향을 뺀 뒤 `wrap_angle` 안 하면 로봇이 빙글빙글 돈다.
4. **LiDAR 빔 순서/부호** — Webots range image 는 인덱스 0 이 왼쭉 끝(각도 +fov/2 → −fov/2 감소). 지도가 **좌우 반전**이면 각도 부호를 뒤집을 것. 확인법: 로봇 왼쪽에 상자 두고 어느 인덱스가 짧아지는지 출력.
5. **LiDAR 장착 오프셋/방향** — 센서가 로봇 중심에서 떨어져 있거나 뒤를 보고 있으면 `lidar_to_robot` 결과에 오프셋/회전을 더해야 함.
6. **inf / nan / 0** — Webots 는 못 보면 `inf`. 0 이나 아주 작은 값은 로봇 몸체. `_prepare_beams` 참고. inf 를 `max_range` 로 바꿔 벽으로 찍으면 지도에 원형 가짜 벽이 생김.
7. **지도가 너무 작음** — 지도 밖 측정값은 조용히 버려지고 frontier 도 안 생겨 "탐색이 이상하게 일찍 끝남". `grid.dropped_hits` 가 늘면 `MAP_HALF_M` 키우기. 반대로 너무 크면 A*/inflate 가 느려짐.
8. **문을 못 지나감** — 문 폭 < 2×(로봇 반지름+여유) 면 팽창 후 막힘. 여유를 줄이거나 해상도를 낮추고, `min_frontier` 도 작게.
9. **로봇 자신이 팽창 벽 안에 있어 A* 실패** — `nearest_passable` 로 시작점 스냅 (이미 `plan_path` 에 포함).
10. **위치 추정이 흔들리면 지도가 겹겹이 어긋남** — 격자 파트 잘못이 아니라 pose 입력 문제. odometry 만 쓰면 회전 오차가 특히 치명적 → 자이로 적분으로 dθ 대체(webots_adapter `WheelOdometry`). 지도가 흐려지면 팀원 scan matching 과 함께 확인.
11. **재계획을 매 스텝 하면 느리고 떨림** — 1 초 주기 또는 이벤트(도달/막힘/정체) 시에만. frontier 목표가 매번 바뀌어 왔다갔다 하면 `w_turn` 을 키우거나 목표 유지 조건 추가.
12. **움직이는 사람** — log-odds `l_clamp` 덕에 지나간 자리는 회복되지만 몇 스캔 걸림. 실시간 회피는 DWA(팀원) 몫. 격자 파트는 "사람이 서 있던 자리를 영구 벽으로 만들지 않기" 만 책임.
13. **Python 3.10** — 이 코드는 3.10 문법만 사용(`match`, `Self` 미사용). `from __future__ import annotations` 포함.
14. **matplotlib 한글 깨짐** — `geometry.setup_korean_font()`. Ubuntu: `sudo apt install fonts-nanum` 후 `rm -rf ~/.cache/matplotlib`.

## 강의 레포(PNU-TECHWEEK-260930)에서 확인한 실제 값

`controllers/`, `worlds/apartment.wbt`, 노트북 5장에서 직접 읽은 값입니다. webots_adapter.py 에 이미 반영되어 있습니다.

| 항목 | 값 | 출처 |
|---|---|---|
| 로봇 | TurtleBot3 Burger, `WHEEL_RADIUS 0.033`, `WHEEL_SEPARATION 0.160`, `ROBOT_RADIUS 0.105` | 노트북 5장 |
| 디바이스 이름 | `"LDS-01"`, `"camera"`, `"gyro"`, `"compass"`, `"accelerometer"`, `"left/right wheel motor"`, 엔코더는 `motor.getPositionSensor()` | tb3_teleop_sensors.py |
| LiDAR 인덱스 | 360개. `ranges[180]`=정면, `[0]`=후방, `[90]`=왼쪽, `[270]`=오른쪽 → `angle_i = π − 2π·i/360` | tb3_lidar.py |
| LiDAR 주기 | `lidar.enable(100)` → 100 ms 마다 갱신, `basicTimeStep 64` → 같은 스캔이 반복 들어옴 | 컨트롤러/월드 |
| 카메라 | FOV 1.0472 rad(60°), 640×480, 로봇 기준 (0.05, 0, −0.08) | apartment.wbt |
| 시작 pose | apartment: translation (−0.3, −7.5), rotation z π → `(−0.3, −7.5, π)` | apartment.wbt |
| 구조 대상 | 색 사과 PROTO 4종(Red/Green/Purple/Orange), 지름 5 cm, 바닥 z=0.05, 7개 배치 | protos/, apartment.wbt |
| 대상 범위 | x∈[−12.0, −2.8], y∈[−11.7, −1.3] → 시작점에서 최대 약 13 m → 지도 반폭 ≥ 15 m | apartment.wbt |
| 동적 장애물 | Pedestrian 1명, `--speed=0.2`, 고정 궤적 순환 | apartment.wbt |
| 사과 탐지 | LAB 색공간 inRange → 최대 contour → minEnclosingCircle(반지름) + moments(중심) | tb3_teleop_cam.py |
| Ground truth | Supervisor `robot.getSelf().getPosition()`, `θ = atan2(orientation[3], orientation[0])` | tb3_ground_truth.py |

주의할 점:
- LDS-01 은 바닥에서 약 17 cm 높이. **5 cm 사과는 LiDAR 에 안 보입니다.** 거리는 카메라 외접원 반지름으로 추정(`target_world_from_pixel`)해야 합니다. 사과는 장애물로도 지도에 안 찍히니 밟고 지나갈 수 있습니다(대회 규칙 확인).
- 컴퍼스 heading 부호는 `tb3_ground_truth` 와 비교해서 `COMPASS_SIGN` 을 정합니다.
- 슬라이드 PDF(148쪽)는 이론만 있고 규칙·수치는 없습니다. 채점 기준·도착 판정 거리·제한 시간은 당일 공지로 확인.
- 강의 Costmap 관례(ROS): 0 free / 1~252 팽창 감쇠 / 253 내접 / 254 벽 / 255 unknown. 이 코드의 `build_cost_map` 은 같은 개념을 실수 비용으로 표현한 것.

## Python 3.10 커널

노트북은 Python 3.10.20 + `numpy==1.23.5` 등 옛 버전을 고정합니다. 이 Mac 의 기본 Python 3.14 로는 설치가 안 되므로 프로젝트 루트에 `.venv-py310` 을 만들고 Jupyter 커널 **"Python 3.10 (pnu-techweek)"** 으로 등록했습니다.

```bash
cd 2026_TECHWEEK_HACKATHON
.venv-py310/bin/jupyter lab PNU-TECHWEEK-260930/TECH-WEEK-26_Physical-AI.ipynb   # 커널 선택: Python 3.10 (pnu-techweek)
.venv-py310/bin/python grid_nav/sim_demo.py                                       # grid_nav 도 이 환경에서 검증됨
```
노트북의 `%%bash pip install ...` 셀은 이미 설치돼 있으니 건너뛰어도 됩니다. YOLO 셀(`ultralytics`, torch)은 미설치 상태이며 필요하면 `uv pip install --python .venv-py310/bin/python ultralytics`.

## Webots 에서 실행 (2026-09-30 오후 검증)

```bash
# 1) (최초 1회) 월드 자산을 Webots 캐시에 미리 받기 — macOS 에서 GitHub 다운로드가 자주 끊겨 벽/가구가 빠진 채 뜨는 문제 해결
cd PNU-TECHWEEK-260930 && /opt/miniconda3/envs/ml_env/bin/python prefetch_assets.py worlds/apartment.wbt

# 2) Webots 실행 (Preferences > General > Python command = /opt/miniconda3/envs/ml_env/bin/python 로 설정되어 있음)
/Applications/Webots.app/Contents/MacOS/webots --stdout --stderr worlds/apartment_gridnav.wbt   # 전체 스택
/Applications/Webots.app/Contents/MacOS/webots --stdout --stderr worlds/breakroom_reactive.wbt  # Algorithm 3 만

# 3) (선택) 카메라+지도 창을 파이썬으로 따로 보기
/opt/miniconda3/envs/ml_env/bin/python grid_nav/view_live.py PNU-TECHWEEK-260930/controllers/tb3_gridnav
```

| 컨트롤러 | 월드 | 내용 |
|---|---|---|
| `tb3_reactive` | `breakroom_reactive.wbt`, `apartment_dev.wbt` | 강의 Algorithm 3 + Supervisor 진짜 위치 로그 |
| `tb3_gridnav` | `breakroom_gridnav.wbt`, `apartment_gridnav.wbt` | 전체 스택: odometry+컴퍼스 → 지도 → frontier → A* → look-ahead, 빨간 사과 비전, Display 지도 오버레이 |

`*_gridnav`/`*_dev` 월드는 로봇에 `supervisor TRUE` 와 `map_display` 가 추가된 개발용 복사본입니다. **제출용은 원본 `apartment.wbt` + Supervisor 없는 컨트롤러**여야 합니다.

apartment 실행 결과(1회): 빨간 물체 발견 → 방문 → 시작점 복귀 완료(210 s, 복귀 오차 0.11 m, odometry+컴퍼스 위치 오차 0.02~0.12 m).
단, 발견한 것은 사과가 아니라 복도 끝의 **큰 빨간 물체(오탐)** 였음 → 아래 "남은 문제" 1번.

확인된 사실: `COMPASS_SIGN = -1` (컴퍼스 raw 각은 시계방향 +), LiDAR 인덱스 180=정면·90=왼쪽 정상, LiDAR 는 100 ms 주기라 첫 스캔 전에 계획하면 즉시 "탐색 완료" 가 됨(3스캔 대기로 해결).

### 남은 문제 (우선순위)
1. **빨간 오탐**: 크기만으로 거리를 재면 "멀리 있는 큰 빨간 것"과 "가까운 사과"를 구분 못 함. 해결안: 사과는 바닥에 있으므로 화면 중심선(수평선)보다 아래에 있어야 함(`cy > H/2 + 4`), 반지름 기반 거리와 화면 세로 위치 기반 거리가 일치해야 함, 같은 물체의 재탐지는 새 대상으로 추가하지 말고 위치를 갱신(합치기 반경 1.0 m).
2. 탐색 목표 유지(hysteresis)는 넣었으나 apartment 전체 탐색 시간 미측정.
3. scan matching(localization.py) 미연결 — 현재 odometry+컴퍼스만으로 오차 0.1 m 수준이라 우선순위 낮음.
4. DWA 미연결 — 사람(Pedestrian) 회피는 정면 비상정지뿐.

## 심사 항목과 격자 파트의 연결

- **목표 달성**: frontier 로 빠짐없이 탐색, SEARCH 로 카메라 미확인 구역까지, `None` 반환으로 탐색 완료 판단, 시작점 복귀 경로.
- **주행 안정성**: inflation(안전거리) + 벽 근접 비용(복도 중앙) + 코너커팅 금지 + 경로 막힘 재계획.
- **창의성 후보**: 카메라 커버리지 층(SEARCH), BFS 경로 거리 기반 frontier 선택, 블랙리스트/정체 복구.
