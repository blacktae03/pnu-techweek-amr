# 강의 슬라이드(148쪽) 대비 구현 감사 + 튠 대상

작성일 2026-09-30. 페이지 번호는 `부산대 TECH WEEK Physical AI.pdf` 기준.

## 1. 언급한대로 구현한 것

| 슬라이드 항목 | 쪽 | 우리 코드 |
|---|---|---|
| Perception → Planning → Action 파이프라인 | 4-8, 148 | occupancy_grid/localization → frontier/astar → dwa/reactive |
| 2D LiDAR 극좌표 → (x, y), 로봇/월드 좌표 변환 | 13 | geometry.py |
| LiDAR 데이터 구조(각도 범위, 거리 범위, 개수, inf), 샘플링 주기 | 14 | webots_adapter.lidar_angles(idx180=정면), enable(100ms), 새 스캔만 갱신 |
| 컴퍼스 heading, IMU drift | 15, 57 | webots_adapter.CompassHeading, WheelOdometry의 dθ를 자이로/컴퍼스로 대체 옵션 |
| 엔코더 odometry Δs=(dr+dl)/2, Δθ=(dr−dl)/L | 17, 58-61 | localization.WheelOdometry |
| 색 임계값 → 마스크 → 최대 contour → minEnclosingCircle → moments centroid | 31-37 | 강의 레포 tb3_teleop_cam.py 그대로 사용 |
| Occupancy Grid(Free/Occupied/Unknown), 베이지안 갱신, log 공간, Trinary | 53-55 | occupancy_grid.py (log-odds, 0/1/−1 export) |
| Bresenham 레이 처리 | 53 | geometry.bresenham_line + 벡터화 샘플링 |
| Scan matching(scan-to-map), ICP(SVD, KD-tree) | 63-72 | localization.py (ICP는 참고용, 기본은 correlative → 3표) |
| 예측 → 보정 → 지도 갱신 순서 | 74-75, 92 | localization + sim_demo.sense_and_map |
| 초기 위치 아는 상태로 시작(Kidnapped 미다룸) | 73 | 동일 전제 |
| Costmap: 로봇 크기 inflation, 벽에서 멀수록 낮은 비용, 내접 영역 진입 금지, Unknown 회피 | 96-99 | occupancy_grid.inflate, astar.build_cost_map, unknown 통행 불가 |
| Global Planner A* (f=g+h), 약 1 Hz 재계획 | 101, 103 | astar.py, sim_demo 1.5 s / webots_adapter 1 s |
| Frontier 4단계(후보 → 선택 → Global Path → Local 이동), 종료 조건 | 104-105 | frontier.py → astar → dwa/PathFollower, None=완료 |
| Local Planner DWA | 109 | dwa.py |
| cKDTree 최근접 탐색 | 70, 117 | dwa.py, localization.py, path_lookahead_point |
| Reactive Algorithm 1·2·3 | 110-112 | reactive.py (+ Webots controllers/tb3_reactive) |
| Look-ahead: 최근접 waypoint → 경로거리 기준 앞점 → 로봇 프레임 변환 → κ=2y/(x²+y²), ω=vκ, v_r/v_l 변환 | 116-125 | webots_adapter.lookahead_control(공식 그대로), dwa.path_lookahead_point, sim_demo.PathFollower(look-ahead, 비례 제어) |
| Look-ahead 거리는 튠 대상 | 126 | sim_demo --lookahead (고정 거리, 속도 연동은 안 함) |
| Decision Making FSM, 청소로봇 예제의 GO_HOME 성격 | 127-129, 146 | sim_demo Mission FSM (EXPLORE→SEARCH→VISIT→RETURN→DONE) |
| Recovery Action | 152(노트북), 142 | 정체 감지 → 블랙리스트 → 재계획, Alg3 Escape, DWA 전부충돌 시 최대 clearance |

## 2. 언급하였으나 구현하지 않은 것

| 슬라이드 항목 | 쪽 | 해커톤에서 의미 | 난이도 |
|---|---|---|---|
| **배터리/시간 예산 기반 복귀 조건** | 110-112, 145-147 | 제한 시간이 있으면 "남은 시간 = 복귀 경로 길이/속도 × 여유" 에 도달하면 무조건 RETURN. 미완주 방지에 가장 실용적 | 작음 |
| Behavior Tree(Selector/Sequence/Parallel/Fallback/Decorator/Blackboard) | 127, 130-145 | 상태가 4~5개면 FSM으로 충분. 노트북에 BT 코드 예시가 있어 붙일 수는 있음 | 중간 |
| Dijkstra | 101-102 | A*에 h=0. 여러 대상까지 비용을 한 번에 구할 때 유용하나 frontier의 BFS 거리맵이 이미 그 역할 | 작음 |
| Kalman Filter | 76-78 | odometry+scan matching 융합 시 노이즈 감소. 현재는 보정값을 그대로 채택 | 중간 |
| Particle Filter / AMCL / 전역 재위치추정 | 79-91 | 위치를 완전히 잃었을 때 복구. 시뮬레이션 + 시작 pose 제공 조건에서는 우선순위 낮음 | 큼 |
| TEB, MPPI | 109 | DWA로 충분 | 큼 |
| YOLO(COCO 47 apple) | 38-50 | 설치·모델 준비 완료, 파이프라인 미연결. 색 필터가 흔들릴 때의 대안 | 중간 |
| HSV 색공간 | 27-33 | 강의 레포 코드가 LAB을 써서 그대로 따름. 조명 문제 생기면 비교 | 작음 |
| Local Costmap(rolling window 격자) | 100, 114 | DWA가 LiDAR 점을 직접 써서 대체. LiDAR 사각의 장애물 기억은 없음 | 중간 |
| IMU 가속도계, LiDAR 강도값, 3D LiDAR, GNSS, Visual SLAM, 위상지도 | 9-16, 52 | 하드웨어 없음 또는 범위 밖 | — |

## 3. 언급하였으나 다르게(더 낫게) 구현한 것

| 항목 | 쪽 | 슬라이드 | 우리 | 이유 / 트레이드오프 |
|---|---|---|---|---|
| Scan matching | 63-72 | ICP | Correlative(후보 격자 + 우도장 + odometry prior), 확실한 벽(log-odds>2)만 앵커 | ICP는 긴 벽에서 미끄러져 테스트 오차 0.48 m, correlative 0.04 m. 탐색 범위(±0.1 m, ±3°) 밖 큰 오차는 못 잡음 |
| Odometry 적분 | 62 | 원호 모델 Rc=Δs/Δθ | 중점 적분 θ+dθ/2 | Δθ→0 특이점 없음, 정확도 동급 |
| A* 휴리스틱 | 103 | Manhattan/Euclidean | Octile + 코너커팅 금지 | 8방향에서 admissible이면서 더 타이트 → 탐색 칸 수 감소, 벽 모서리 스침 방지 |
| Costmap 값 | 96-99 | uint8 0~255 계단 | 이진 팽창 + 거리변환 연속 비용 | safe_dist, w_prox 두 숫자로 직접 조절. ROS 호환은 안 됨 |
| Local Costmap | 100, 109 | rolling window 격자 | 현재 LiDAR 점 + "지도에 굳지 않은 점"에 큰 여유 | 사람에 즉각 반응, 잔상 없음. 사각지대 기억 없음 |
| Frontier 선택 | 105 | 거리·cost·크기 | + heading 변화 벌점 + 블랙리스트 + 대표점=중심 최근접 칸 + 선택적 BFS 경로거리 | 왕복 감소, 도달불가 frontier 무한반복 방지 |
| 탐색 완료 판정 | 104-105 | frontier 없음 | + SEARCH(카메라 미확인 빈칸) 단계 | LiDAR 3.5 m vs 카메라 2.5 m 차이로 지도 완성 ≠ 대상 발견. 대상 3/3 발견의 결정적 요인 |
| 경로 추종 | 122-125 | 곡률 공식 | Webots 어댑터는 공식 그대로, 데모는 heading 비례 + 감속 | 급회전에서 κ 폭주 없음. 두 컨트롤러 불일치는 단점 |
| 제어 주기 | 75, 109 | 30/5-10/20 Hz | 지도·보정 10 Hz(LiDAR 주기), DWA 10 Hz | 센서보다 빠른 계산은 이득 없음. 고속 회피엔 20 Hz가 유리 |
| Recovery | 142 | BT Recovery 노드 | 정체 감지(목표 거리 미감소) → 블랙리스트 → 재계획 | 같은 실패 반복 안 함. 정책이 FSM 코드에 흩어짐 |

## 4. 슬라이드에 없는데 우리가 추가한 것
SEARCH 단계와 카메라 커버리지 층, VISIT/RETURN 미션 FSM, 픽셀→월드 좌표(핀홀 + 겉보기 반지름 거리), DWA 제동거리·동적점 여유·후진 복구, 정체 감지·블랙리스트, nearest_passable 스냅, line-of-sight 직선화, dropped_hits 지도 크기 경고, 움직이는 사람 시뮬레이션, log-odds 클램프로 사람 잔상 소거.

## 5. 튠 대상 (성적에 영향 큰 순)

| 우선 | 파라미터 (기본값) | 위치 | 어느 쪽으로 틀리면 어떤 증상 | 조절 방향 |
|---|---|---|---|---|
| ★★★ | 팽창 여유 margin (0.12 / Webots 0.10) | sim_demo --margin, webots_adapter.SAFETY_MARGIN | 크면 문·가구 사이 통과 불가("경로 없음", frontier 사라짐), 작으면 벽 스침 | apartment 가장 좁은 통로 폭을 재고 (폭 − 2·반지름)/2 미만으로 |
| ★★★ | 벽 근접 비용 w_prox, safe_dist (3.0, 0.5) | --w-prox --safe-dist | 크면 좁은 곳을 돌아가거나 A* 느려짐, 0이면 벽에 붙어 감 | 충돌 없으면 그대로, 복도 중앙을 못 지키면 w_prox↑ |
| ★★★ | 카메라 유효거리 cam_range (2.5) | --cam-range, SEARCH 판정 | 실제보다 크게 잡으면 "봤다"고 착각 → 대상 놓침. 작게 잡으면 수색 시간↑ | 비전 팀원이 실제로 인식되는 최대거리를 재서 그 값의 0.8배 |
| ★★★ | frontier 점수 w_dist, w_size, w_turn (1.0, 0.05, 0.3), min_frontier (6) | frontier.select_frontier, --min-frontier | 왔다갔다하면 w_turn↑ 또는 목표 유지 추가. 좁은 문 너머를 안 가면 min_frontier↓ | 총 스텝·마지막 대상 발견 스텝으로 seed 3개 비교 |
| ★★☆ | 대상 개수 N (규칙 확인) | replan EXPLORE 분기에 한 줄 | N개 다 찾으면 탐색 중단 → 시간 대폭 절약 | 당일 공지 확인 후 추가 |
| ★★☆ | 시간 예산 복귀 조건 (없음) | Mission.update | 제한시간 초과로 미복귀 = 목표 미달성 | "남은 시간 < 복귀 경로/속도 × 1.5" 면 RETURN |
| ★★☆ | DWA dyn_margin (0.30 sim / 0.40 기본), clearance_max (0.5), w_clearance (1.2) | dwa.DWAParams | 크면 사람 앞에서 무한 대기·좁은 문 앞 진동, 작으면 사람과 접촉 | 사람 속도 0.2 m/s 기준 0.3 부터. 진동하면 clearance_max↓ |
| ★★☆ | look-ahead (0.4) | --lookahead, webots lookahead_control(0.25) | 길면 커브 안쪽 가로질러 문 모서리 스침, 짧으면 지그재그 | 0.3~0.5 사이. 속도 올리면 함께 ↑ |
| ★★☆ | 위치추정 탐색범위 search_xy/th (±0.1 m/±3°), step (2 cm/0.5°), l_static (2.0) | localization.CorrelativeMatcher | 범위 작으면 큰 미끄러짐 못 잡음, 크면 엉뚱한 벽에 붙음·느려짐 | 위치 오차 로그 보며. 회전 오차 크면 search_th↑ |
| ★☆☆ | log-odds p_hit/p_miss (0.7/0.35), clamp (±4) | OccupancyGrid | 사람 잔상이 오래 남으면 clamp↓, 벽이 깜빡이면 p_hit↑ | 지도 화면으로 판단 |
| ★☆☆ | 해상도 (0.05), 지도 반폭 (15 m) | RESOLUTION, MAP_HALF_M | A*·팽창 느리면 0.075. dropped_hits 경고 나면 반폭↑ | 스텝당 시간 측정 |
| ★☆☆ | 재계획 주기 (1.5 s / 1 s), 정체 판정 (60 스텝) | --replan-every --stuck-steps | 짧으면 목표가 자꾸 바뀌어 떨림, 길면 막힌 길을 오래 감 | 1 s 전후 |
| ★☆☆ | 속도 v_max (TB3 0.22), Alg3 distance_threshold (0.45), ANGLE (60°) | DWAParams, ReactiveParams | 빠르면 충돌 위험·위치추정 악화 | 심사가 충돌에 민감하면 0.15로 낮추는 것도 전략 |

측정 방법: `python sim_demo.py --headless --seed {0,1,2} [옵션]` 로 총 스텝, 마지막 대상 발견 스텝, 충돌 수, 복귀 오차를 비교. Webots에서는 ground truth 컨트롤러로 위치 오차를 함께 기록.
