# TODO / 상태 파일 (해커톤 격자 파트 학습 코드)

## 목표
Webots 없이 NumPy+matplotlib(+scipy)로 돌아가는 학습용 격자 파트 코드 (grid_nav/ 폴더)

## 태스크 (2026-09-30 완료)
- [x] geometry.py      좌표 변환, 각도 wrap, Bresenham, 한글 폰트 헬퍼
- [x] occupancy_grid.py log-odds 지도(벡터화+Bresenham 두 버전), inf 처리, inflation, 벽 거리 지도, dropped_hits 경고
- [x] frontier.py      frontier 검출/클러스터/목표 선택, BFS 경로 거리
- [x] astar.py         8방향 A*, 코너커팅 금지, 벽 근접 비용, 직선화, plan_path()
- [x] sim_demo.py      EXPLORE→SEARCH(카메라 미확인 수색)→VISIT→RETURN 상태 기계 + 애니메이션
- [x] webots_adapter.py GridPlanner API + Webots 컨트롤러 뼈대 (당일 수정 포인트 [A]~[E] 주석)
- [x] README.md
- [x] 검증: seed 0/1/2 headless 모두 대상 3/3 발견·방문, 충돌 0, 복귀 오차 ≤0.2m, 스텝당 ~8ms

## 결정 사항
- 격자: grid[row, col], row↔y, col↔x. origin = (row0,col0) 칸의 왼쪽-아래 모서리 월드좌표
- 지도 외부 형식: int8, 0 빈칸 / 1 벽 / -1 미탐색
- 로봇 좌표계: x 전방, y 왼쪽, theta는 x축 기준 반시계(라디안)
- 지도 크기는 시작점 기준 ±11m (sim_demo --map-half). 작으면 지도 밖 구역이 조용히 사라짐

## 2026-09-30 오후 추가
- [x] 강의 레포 PNU-TECHWEEK-260930 분석 → webots_adapter.py 에 TB3/LDS-01/카메라/시작pose 반영 (README 표 참고)
- [x] Python 3.10 venv (.venv-py310) + Jupyter 커널 "Python 3.10 (pnu-techweek)" 생성, grid_nav 검증 완료
- [x] ml_env(conda, py3.10) 에 torch 2.8.0 / torchvision / ultralytics 설치, models/YOLO/yolo11n.pt 다운로드. 노트북은 ml_env 커널 사용 (pnu-py310 은 중복)
- [x] localization.py (odometry + correlative scan matching, ICP 참고) / dwa.py 작성, sim_demo 에 --localize --dwa --pedestrian --unified 옵션 통합
- [x] reactive.py (강의 Alg 1·2·3) + sim_demo --reactive + Webots 컨트롤러 controllers/tb3_reactive 추가
- [x] PathFollower 를 look-ahead 방식으로 교체 (--lookahead)
- [x] 전 시나리오 headless 검증: 충돌 0 (단순추종+사람 제외), 3/3 방문, 복귀
- [ ] Webots 설치 여부 확인 (이 Mac 에 /Applications/Webots 없음)

## 당일 할 일 (Webots 코드 받은 뒤)
- webots_adapter.py 의 [확인필요] 항목: LDS-01 maxRange, COMPASS_SIGN(ground truth 와 비교), 대회 월드/시작 pose, 도착 판정 거리
- pose 를 팀원 scan matching 으로 교체, lookahead_control 을 DWA 로 교체, 비전 (cx, radius) 연결

## 2026-09-30 Webots 연동 (오후)
- [x] Webots R2025a 설치, Python=ml_env, 자산 캐시 prefetch (prefetch_assets.py)
- [x] tb3_reactive (Alg3) breakroom 검증, COMPASS_SIGN=-1 확정, LiDAR 매핑 확정
- [x] tb3_gridnav 전체 스택 apartment 에서 발견→방문→복귀 완주 (오탐이지만 파이프라인 동작)
- [ ] 빨간 오탐 수정 (수평선 아래 조건 + 거리 일관성 + 재탐지 병합)
- [ ] apartment 전체 탐색 시간 측정, frontier 파라미터 튠
- [ ] scan matching / DWA 연결 (팀원)
- [ ] 제출용: 원본 apartment.wbt + Supervisor 제거 버전
