"""
sim_demo.py - Webots 없이 돌아가는 탐색/구조/복귀 데모

  가짜 2D 세계(벽, 방, 문, 기둥) 를 만들고, 레이캐스팅으로 가짜 LiDAR 를 만들어
  [스캔 → 지도 업데이트 → (frontier 선택 | 대상 방문 | 복귀) → A* → 경로 추종] 을 반복합니다.

상태 기계 (해커톤 미션 흐름 그대로):
  EXPLORE  : frontier 가 남아 있으면 그쪽으로. 이동 중 카메라(가짜) 로 대상을 발견하면 위치 기록
  SEARCH   : frontier 가 없어도(LiDAR 로는 지도를 다 그렸어도) 카메라가 아직 안 본 빈칸이 남았으면
             그곳을 훑음. ★ LiDAR 는 5m 밖까지 보지만 카메라는 2~3m 안만 본다는 점이 핵심.
             지도를 다 그렸다고 대상을 다 본 게 아니다 → Search and Rescue 에서 꼭 필요한 단계.
  VISIT    : 발견한 대상들을 가까운 순서로 방문
  RETURN   : 시작 지점으로 복귀
  DONE

이 데모에서 "진짜" 라고 가정한 것 (실전에서는 팀원 모듈이 대신함):
  - 로봇 pose 는 정확히 안다고 가정 (실전: scan matching / odometry 팀원)
  - 경로 추종은 단순 비례 제어 (실전: DWA 팀원). 데모에서는 LiDAR 최소거리로 비상정지만 함
  - 대상 탐지는 "거리 2.5m 이내 + 카메라 시야각 안 + 벽에 가려지지 않음" 이면 발견 (실전: 비전 팀원)

실행:
  python sim_demo.py                 # 애니메이션
  python sim_demo.py --headless      # 화면 없이 끝까지 돌리고 결과 PNG 저장 (검증용)
  python sim_demo.py --seed 3 --speed 2
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from geometry import GridSpec, wrap_angle, lidar_to_world, robot_to_world, world_to_robot, setup_korean_font
from occupancy_grid import OccupancyGrid, inflate
from frontier import frontier_mask, cluster_frontiers, select_frontier, bfs_distance_map
from astar import plan_path, build_cost_map
from localization import WheelOdometry, CorrelativeMatcher
from dwa import DWAParams, dwa_control, path_lookahead_point
from reactive import Algorithm3, ReactiveParams, lidar_sectors, wheels_to_vw


# ======================================================================
# 1. 가짜 세계
# ======================================================================
class FakeWorld:
    """True = 벽 인 불리언 격자. 지도 GridSpec 과는 별개의 '진짜 세계' 해상도를 가짐.
    (로봇은 이 배열을 절대 직접 보지 않습니다. LiDAR 레이캐스트로만 간접적으로 봅니다.)"""

    def __init__(self, width_m=10.0, height_m=8.0, resolution=0.05):
        self.spec = GridSpec(resolution=resolution, origin_x=0.0, origin_y=0.0,
                             rows=int(round(height_m / resolution)),
                             cols=int(round(width_m / resolution)))
        self.occ = np.zeros(self.spec.shape, dtype=bool)
        self.targets: List[Tuple[float, float]] = []
        # 움직이는 사람들: dict(pos, radius, wps(경유점 목록), i(다음 경유점), speed)
        self.pedestrians: List[dict] = []

    def add_pedestrian(self, waypoints, radius=0.2, speed=0.3):
        """경유점을 순환하는 사람 추가 (apartment.wbt 의 Pedestrian --trajectory 와 같은 개념)."""
        self.pedestrians.append(dict(pos=np.array(waypoints[0], dtype=float), radius=radius,
                                     wps=[np.array(w, dtype=float) for w in waypoints], i=1, speed=speed))

    def step_pedestrians(self, dt):
        for p in self.pedestrians:
            tgt = p["wps"][p["i"]]
            d = tgt - p["pos"]
            dist = np.hypot(*d)
            if dist < p["speed"] * dt:
                p["pos"] = tgt.copy(); p["i"] = (p["i"] + 1) % len(p["wps"])
            else:
                p["pos"] = p["pos"] + d / dist * p["speed"] * dt

    def occ_now(self):
        """지금 이 순간의 '진짜 세계' = 고정 벽 + 사람 원판. LiDAR 와 충돌 판정은 이걸 본다.
        (사람은 지도에 영구 벽으로 남으면 안 되므로, 지도 쪽은 log-odds 클램프로 스스로 지워지게 둔다)"""
        if not self.pedestrians:
            return self.occ
        occ = self.occ.copy()
        for p in self.pedestrians:
            r_cells = int(np.ceil(p["radius"] / self.spec.resolution))
            r0, c0 = self.spec.world_to_grid(p["pos"][0], p["pos"][1])
            rr, cc = np.ogrid[-r_cells:r_cells + 1, -r_cells:r_cells + 1]
            disk = (rr ** 2 + cc ** 2) <= r_cells ** 2
            R = np.clip(r0 + rr, 0, self.spec.rows - 1); C = np.clip(c0 + cc, 0, self.spec.cols - 1)
            occ[R, C] |= disk
        return occ

    def add_rect(self, x0, y0, x1, y1):
        """월드 좌표 사각형 [x0,x1] x [y0,y1] 을 벽으로."""
        r, c = self.spec.world_to_grid(np.array([x0, x1]), np.array([y0, y1]))
        r, c = self.spec.clip(r, c)
        self.occ[r[0]:r[1] + 1, c[0]:c[1] + 1] = True

    def clear_rect(self, x0, y0, x1, y1):
        """벽에 문(구멍) 내기."""
        r, c = self.spec.world_to_grid(np.array([x0, x1]), np.array([y0, y1]))
        r, c = self.spec.clip(r, c)
        self.occ[r[0]:r[1] + 1, c[0]:c[1] + 1] = False

    def is_wall_at(self, x, y):
        r, c = self.spec.world_to_grid(x, y)
        if not self.spec.in_bounds(r, c):
            return True
        return bool(self.occ[r, c])

    def collides(self, x, y, radius):
        """반지름 radius 원 안에 벽 칸이 있는가 (충돌 판정용). 원 둘레 16점 + 중심 검사."""
        angs = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        xs = np.append(x + radius * np.cos(angs), x)
        ys = np.append(y + radius * np.sin(angs), y)
        r, c = self.spec.world_to_grid(xs, ys)
        inb = self.spec.in_bounds(r, c)
        if not inb.all():
            return True
        return bool(self.occ_now()[r, c].any())


def build_world(seed=0):
    """벽 + 방 3개 + 문 + 기둥 + 구조 대상 3개 가 있는 10m x 8m 세계."""
    rng = np.random.default_rng(seed)
    w = FakeWorld(10.0, 8.0, 0.05)
    t = 0.15                                   # 벽 두께
    # 바깥 벽
    w.add_rect(0, 0, 10, t); w.add_rect(0, 8 - t, 10, 8)
    w.add_rect(0, 0, t, 8);  w.add_rect(10 - t, 0, 10, 8)
    # 세로 벽 x=5 (아래쪽 방/위쪽 방 나눔), 문 두 개
    w.add_rect(5 - t / 2, 0, 5 + t / 2, 8)
    w.clear_rect(5 - t, 1.5, 5 + t, 2.6)
    w.clear_rect(5 - t, 6.0, 5 + t, 7.0)
    # 가로 벽 y=4.5 왼쪽 절반, 문 하나
    w.add_rect(0, 4.5 - t / 2, 5, 4.5 + t / 2)
    w.clear_rect(3.2, 4.5 - t, 4.3, 4.5 + t)
    # 가로 벽 y=4 오른쪽 절반 (벽이 짧아 통로가 생김)
    w.add_rect(6.5, 4.0 - t / 2, 10, 4.0 + t / 2)
    # 기둥/가구 몇 개 (랜덤 위치, 시작점과 문 근처는 피함)
    boxes = [(2.0, 2.2, 2.6, 2.8), (7.5, 1.0, 8.1, 1.6), (8.0, 6.0, 8.6, 6.8), (1.0, 6.0, 1.8, 6.4)]
    for (x0, y0, x1, y1) in boxes:
        jx, jy = rng.uniform(-0.2, 0.2, 2)
        w.add_rect(x0 + jx, y0 + jy, x1 + jx, y1 + jy)
    # 구조 대상 (벽 근처에 놓음 - 실제 대상은 벽에 붙어 있기도 하니 nearest_passable 연습)
    w.targets = [(9.3, 7.3), (0.7, 3.8), (6.0, 0.7)]
    return w


# ======================================================================
# 2. 가짜 센서
# ======================================================================
def raycast_lidar(world: FakeWorld, pose, angles, max_range=6.0, noise_std=0.01, rng=None):
    """레이캐스팅 LiDAR. 각 빔을 따라 해상도/2 간격으로 점을 찍고 첫 벽까지 거리를 반환.
    벽을 못 만나면 inf (실제 LiDAR 도 inf 또는 max_range 를 줌 → occupancy_grid 가 처리).

    벡터화: (N빔, S점) 격자에서 벽 여부를 한 번에 조회하고 argmax 로 '첫 True' 인덱스를 얻음.
    (argmax 는 첫 최대값 위치를 반환하므로 불리언 배열에 쓰면 첫 True 의 위치가 됨)
    """
    step = world.spec.resolution * 0.5
    ts = np.arange(0.0, max_range, step)
    dirs = np.asarray(angles) + pose[2]
    xs = pose[0] + ts[None, :] * np.cos(dirs)[:, None]
    ys = pose[1] + ts[None, :] * np.sin(dirs)[:, None]
    r, c = world.spec.world_to_grid(xs, ys)
    inb = world.spec.in_bounds(r, c)
    hit = np.zeros(r.shape, dtype=bool)
    occ = world.occ_now()
    hit[inb] = occ[r[inb], c[inb]]
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    ranges = np.where(any_hit, ts[first], np.inf)
    if noise_std > 0:
        rng = rng or np.random.default_rng()
        ranges = ranges + rng.normal(0.0, noise_std, ranges.shape)
    return ranges


def fake_camera_detect(world: FakeWorld, pose, fov_deg=70.0, max_dist=2.5):
    """가짜 대상 탐지: 대상이 카메라 시야각 안 + 거리 내 + 벽에 가려지지 않으면 발견.
    반환: 발견한 대상들의 월드 좌표 리스트 (실전에서는 비전 결과 + 로봇 pose 로 이 좌표를 계산)."""
    found = []
    for (tx, ty) in world.targets:
        dx, dy = tx - pose[0], ty - pose[1]
        dist = np.hypot(dx, dy)
        if dist > max_dist:
            continue
        bearing = wrap_angle(np.arctan2(dy, dx) - pose[2])
        if abs(bearing) > np.deg2rad(fov_deg / 2):
            continue
        # 시선이 막혔나: 대상 방향으로 레이 하나 쏴서 대상보다 먼저 벽에 맞으면 안 보임
        r = raycast_lidar(world, pose, np.array([bearing]), max_range=max_dist + 0.1, noise_std=0.0)[0]
        if r < dist - 0.15:
            continue
        found.append((tx, ty))
    return found


# ======================================================================
# 3. 로봇 (운동학 + 단순 경로 추종)
# ======================================================================
@dataclass
class Robot:
    x: float
    y: float
    theta: float
    radius: float = 0.20
    v_max: float = 0.5           # m/s
    w_max: float = 1.8           # rad/s
    trail: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    def step(self, v, w, dt):
        """Differential drive 의 운동학 (오일러 적분). 실제 로봇이면 바퀴 속도로 변환해 명령."""
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta = wrap_angle(self.theta + w * dt)
        self.trail.append((self.x, self.y))


class PathFollower:
    """경로를 따라가는 단순 제어기 = 강의 4장 'Look-ahead' 방식. (실전에서는 DWA 팀원 모듈이 이 자리)

    규칙:
      - 경로에서 로봇에 가장 가까운 점을 찾고, 거기서 경로를 따라 lookahead [m] 앞의 점을 임시 목표로 삼는다
        (점을 하나씩 정확히 찍고 가면 좌우로 크게 흔들린다 - 슬라이드 116~126쪽)
      - 임시 목표 방향과 heading 의 차이(wrap 필수!) 에 비례해 회전
      - 차이가 크면 서서 돌고, 작으면 전진 (cos 로 부드럽게)
      - 경로 끝점에 reach_dist 안으로 들어오면 완료
    lookahead 는 정해진 값이 아니라 튠 대상: 길면 커브 안쪽을 가로지르고(벽 스침), 짧으면 지그재그.
    로봇 속도 0.2 m/s 급이면 0.3~0.5 m 가 무난. sim_demo --lookahead 로 실험.
    """

    def __init__(self, path_xy, reach_dist=0.15, lookahead=0.4):
        self.path = list(path_xy)
        self.idx = 1 if len(self.path) > 1 else 0
        self.reach_dist = reach_dist
        self.lookahead = lookahead
        self._finished = len(self.path) <= 1

    def done(self):
        return self._finished

    def command(self, robot: Robot):
        if self.done():
            return 0.0, 0.0
        pts = np.asarray(self.path)
        gx, gy = pts[-1]
        if np.hypot(gx - robot.x, gy - robot.y) < self.reach_dist:       # 끝점 도달
            self._finished = True
            return 0.0, 0.0
        near = int(np.argmin(np.hypot(pts[:, 0] - robot.x, pts[:, 1] - robot.y)))
        self.idx = max(self.idx, min(near + 1, len(pts) - 1))            # 지나온 점은 버림 (path_blocked 용)
        tx, ty = path_lookahead_point(robot.pose, self.path[near:], self.lookahead)
        dx, dy = tx - robot.x, ty - robot.y
        dist = max(np.hypot(gx - robot.x, gy - robot.y), 1e-6)           # 감속은 '끝점까지 거리' 기준
        err = wrap_angle(np.arctan2(dy, dx) - robot.theta)     # ★ wrap 안 하면 빙글빙글 돈다
        w = float(np.clip(2.5 * err, -robot.w_max, robot.w_max))
        v = robot.v_max * max(0.0, np.cos(err)) if abs(err) < np.deg2rad(60) else 0.0
        v = min(v, robot.v_max * max(0.3, dist / 0.5))      # 목표점 근처에서 감속
        return v, w


# ======================================================================
# 4. 탐색 미션 (상태 기계)
# ======================================================================
class Mission:
    EXPLORE, SEARCH, VISIT, RETURN, DONE = "EXPLORE", "SEARCH", "VISIT", "RETURN", "DONE"

    def __init__(self, world: FakeWorld, robot: Robot, spec: GridSpec, args, rng):
        self.world, self.robot, self.spec, self.args, self.rng = world, robot, spec, args, rng
        self.grid = OccupancyGrid(spec, max_range=args.lidar_range)
        self.angles = np.linspace(-np.pi, np.pi, args.beams, endpoint=False)
        self.inflate_r = robot.radius + args.margin
        # 카메라 커버리지 층: "이 칸을 카메라 시야로 본 적 있다" (SEARCH 단계에서 사용)
        self.seen = np.zeros(spec.shape, dtype=bool)
        self.cam_fov = np.deg2rad(args.cam_fov)
        self.cam_range = args.cam_range
        self.start_xy = (robot.x, robot.y)
        self.state = self.EXPLORE
        self.path: Optional[List[Tuple[float, float]]] = None
        self.follower: Optional[PathFollower] = None
        self.goal_xy: Optional[Tuple[float, float]] = None
        self.blacklist: List[Tuple[float, float]] = []
        self.found_targets: List[Tuple[float, float]] = []
        self.visited_targets: List[Tuple[float, float]] = []
        self.ternary = self.grid.to_ternary()
        self.inflated = self.ternary.copy()
        self.fmask = np.zeros(spec.shape, dtype=bool)
        self.step_count = 0
        self.last_replan = -999
        self.stuck_counter = 0
        self.last_progress_xy = self.start_xy
        self.best_goal_dist = np.inf
        self.collisions = 0
        self.emergency_stops = 0
        self.scan_pts = np.zeros((0, 2))
        self.log: List[str] = []
        self.replans = 0
        # --reactive: 강의 Algorithm 3 으로 돌아다니며 지도만 만들다가, 정해진 스텝 뒤 A* 로 복귀
        self.reactive = Algorithm3(ReactiveParams()) if args.reactive else None
        self.last_ranges = None
        self.v_cmd = self.w_cmd = 0.0          # 직전 속도 명령 (DWA 의 dynamic window 기준, odometry 시뮬 입력)
        # --- 위치 추정 (--localize) -------------------------------------------------
        # est = 로봇이 '자기라고 믿는' pose. 지도·frontier·A*·DWA 는 전부 est 만 본다.
        # robot = 진짜 pose. LiDAR 레이캐스트와 충돌 판정에만 쓴다 (실제 로봇에서는 알 수 없는 값).
        # --localize 가 없으면 est 와 robot 이 같은 객체 → 완벽한 위치 추정을 가정하는 셈.
        if args.localize:
            self.est = Robot(robot.x, robot.y, robot.theta, radius=robot.radius, v_max=robot.v_max, w_max=robot.w_max)
            self.odom = WheelOdometry(robot.x, robot.y, robot.theta, 0.033, 0.160)
            self.matcher = CorrelativeMatcher(spec, max_range=args.lidar_range - 0.1)
            self.phi_l = self.phi_r = 0.0
            self.pose_err_log: List[float] = []
        else:
            self.est = robot
        # --- DWA (--dwa) -----------------------------------------------------------------
        self.dwa_params = DWAParams(v_max=robot.v_max, w_max=robot.w_max, robot_radius=robot.radius,
                                    dt=args.dt, a_v=1.0, a_w=6.0, dyn_margin=0.30)

    # ------------------------------------------------------------------
    def say(self, msg):
        line = f"[{self.step_count:5d}] {msg}"
        self.log.append(line)
        print(line)

    # ------------------------------------------------------------------
    def sense_and_map(self, dt):
        true_pose = self.robot.pose
        ranges = raycast_lidar(self.world, true_pose, self.angles, self.args.lidar_range, rng=self.rng)
        if self.args.localize:
            # ① 엔코더 시뮬레이션: 직전 명령 (v, w) 로 바퀴가 돌았다고 치고, 바퀴 반지름 오차 + 미끄러짐 노이즈
            R, L = 0.033, 0.160
            vl, vr = self.v_cmd - self.w_cmd * L / 2, self.v_cmd + self.w_cmd * L / 2
            self.phi_l += vl * dt / (R * self.args.odom_scale) + self.rng.normal(0, self.args.odom_noise)
            self.phi_r += vr * dt / (R * self.args.odom_scale) + self.rng.normal(0, self.args.odom_noise)
            # ② odometry 예측 → ③ scan matching 보정 → ④ 보정 pose 로 지도 갱신 (순서 중요)
            pred = self.odom.update(self.phi_l, self.phi_r)
            est = self.matcher.correct(pred, self.angles, ranges, self.grid.log_odds)
            self.odom.set_pose(est)
            self.est.x, self.est.y, self.est.theta = est
            self.est.trail.append((est[0], est[1]))
            self.pose_err_log.append(float(np.hypot(est[0] - true_pose[0], est[1] - true_pose[1])))
        pose = self.est.pose
        self.last_ranges = ranges
        self.grid.update(pose, self.angles, ranges)
        ok = np.isfinite(ranges)
        self.scan_pts = lidar_to_world(self.angles[ok], ranges[ok], pose)
        # 앞쪽 ±30° 최소 거리: 비상정지용 (DWA 대신하는 최소한의 안전장치)
        front = np.abs(wrap_angle(self.angles)) < np.deg2rad(30)
        self.front_min = float(np.min(ranges[front])) if np.any(ok & front) else np.inf
        self._mark_camera_seen(pose, ranges)
        # 카메라: 탐지는 진짜 pose 기준으로 일어나지만(센서), 기록하는 좌표는 로봇이 믿는 pose 기준으로 계산
        #         (실제 로봇도 "화면 속 위치 + 자기 추정 pose" 로 대상 좌표를 만든다 → 위치 오차가 그대로 들어감)
        for t_true in fake_camera_detect(self.world, true_pose, fov_deg=self.args.cam_fov, max_dist=self.cam_range):
            rel = world_to_robot(np.array([t_true]), true_pose)
            t = tuple(float(v) for v in np.round(robot_to_world(rel, pose)[0], 2))
            if all(np.hypot(t[0] - f[0], t[1] - f[1]) > 0.6 for f in self.found_targets):   # 위치 오차 감안한 중복 제거
                self.found_targets.append(t)
                self.say(f"★ 구조 대상 발견! 위치 {t}  (지금까지 {len(self.found_targets)}개)")

    def _mark_camera_seen(self, pose, ranges):
        """카메라 시야각 안의 LiDAR 빔을 따라, 카메라 사거리까지의 칸을 '봤다' 로 표시.

        왜 LiDAR 빔을 재활용하나: 카메라가 어디까지 볼 수 있는지 = 그 방향으로 벽이 어디 있는지.
        LiDAR 가 그걸 이미 알려주니, 시야각 안 빔의 거리를 카메라 사거리로 잘라 쓰면 됩니다.
        (occupancy_grid.update 의 '빔 따라 점 찍기' 와 같은 벡터화 기법)
        """
        in_fov = np.abs(wrap_angle(self.angles)) < self.cam_fov / 2
        ang = self.angles[in_fov]
        r = np.minimum(np.nan_to_num(ranges[in_fov], nan=0.0, posinf=self.cam_range), self.cam_range)
        step = self.spec.resolution * 0.5
        ts = np.arange(0.0, self.cam_range + step, step)
        d = np.minimum(ts[None, :], r[:, None])                        # (N, S) 빔 끝에서 멈춤
        xs = pose[0] + d * np.cos(ang + pose[2])[:, None]
        ys = pose[1] + d * np.sin(ang + pose[2])[:, None]
        rr, cc = self.spec.world_to_grid(xs, ys)
        inb = self.spec.in_bounds(rr, cc)
        self.seen[rr[inb], cc[inb]] = True

    # ------------------------------------------------------------------
    def rebuild_maps(self):
        self.ternary = self.grid.to_ternary()
        self.inflated = inflate(self.ternary, self.inflate_r, self.spec.resolution)
        self.cost_map = build_cost_map(self.inflated, self.spec.resolution,
                                       safe_dist=self.args.safe_dist, w_prox=self.args.w_prox)

    def plan_to(self, goal_xy):
        return plan_path(self.inflated, self.spec, (self.est.x, self.est.y), goal_xy,
                         cost_map=self.cost_map, snap_radius_cells=int(0.6 / self.spec.resolution))

    # ------------------------------------------------------------------
    def replan(self):
        """상태에 따라 목표를 정하고 경로를 만듭니다. 몇 번 실패하면 상태를 전이."""
        self.rebuild_maps()
        self.last_replan = self.step_count
        self.replans += 1

        if self.state == self.EXPLORE and self.reactive is not None:
            self.fmask = frontier_mask(self.ternary, self.inflated)     # 표시용
            if self.step_count >= self.args.reactive_steps:
                self.say(f"반응형 탐색 {self.args.reactive_steps} 스텝 종료 → 지도 채움 {np.mean(self.ternary != -1) * 100:.0f}%, "
                         f"발견 대상 {len(self.found_targets)}개. 이제 A* 로 방문/복귀")
                self.state = self.VISIT if self.found_targets else self.RETURN
                self.replan()
            else:
                self.path, self.goal_xy, self.follower = None, None, None
            return

        if self.state == self.EXPLORE:
            self.fmask = frontier_mask(self.ternary, self.inflated)
            if self.args.unified:
                # ★ 통합 모드: LiDAR 경계(frontier) 와 '카메라로 안 본 빈칸' 을 한 마스크로 합쳐서
                #   처음부터 같은 점수 체계로 목표를 고른다. 별도 SEARCH 단계가 필요 없어진다.
                #   장점: 방을 지나갈 때 구석까지 카메라로 훑고 나가므로 나중에 되돌아오는 일이 줄어듦.
                #   단점: 초반에 자잘한 미확인 구석을 쫓아 지그재그가 늘어 지도 완성은 늦어짐.
                unseen = (self.ternary == 0) & (self.inflated != 1) & ~self.seen
                self.fmask = self.fmask | unseen
            frontiers = cluster_frontiers(self.fmask, min_size=self.args.min_frontier)
            rr, cc = self.spec.world_to_grid(self.est.x, self.est.y)
            dmap = bfs_distance_map(self.inflated == 0, (int(rr), int(cc))) if self.args.path_dist else None
            for _ in range(5):          # 최대 5개 후보까지 시도
                f = select_frontier(frontiers, self.est.pose, self.spec,
                                    blacklist_xy=self.blacklist, dist_map=dmap)
                if f is None:
                    break
                goal = f.goal_xy(self.spec)
                path = self.plan_to(goal)
                if path is not None:
                    self._set_path(path, goal)
                    return
                self.blacklist.append(goal)     # 경로가 안 나오는 frontier 는 다음부터 무시
            # 후보 없음 → LiDAR 지도 완성. 이제 카메라로 안 본 곳을 훑는다
            self.say(f"LiDAR 탐색 완료 (frontier 없음). 발견한 대상 {len(self.found_targets)}개"
                     f" → 카메라 미확인 구역 수색(SEARCH)")
            self.state = self.SEARCH if not (self.args.no_search or self.args.unified) else self.VISIT
            self.replan()
            return

        if self.state == self.SEARCH:
            # '갈 수 있는 빈칸' 중 카메라로 본 적 없는 칸 → 덩어리로 묶어 frontier 와 같은 방식으로 목표 선택
            unseen = (self.ternary == 0) & (self.inflated != 1) & ~self.seen
            self.fmask = unseen
            clusters = cluster_frontiers(unseen, min_size=self.args.min_unseen)
            for _ in range(5):
                f = select_frontier(clusters, self.est.pose, self.spec,
                                    blacklist_xy=self.blacklist, w_size=0.01, size_cap=200)
                if f is None:
                    break
                goal = f.goal_xy(self.spec)
                path = self.plan_to(goal)
                if path is not None:
                    self._set_path(path, goal)
                    return
                self.blacklist.append(goal)
            self.say(f"수색 완료 (카메라 미확인 구역 없음). 발견한 대상 {len(self.found_targets)}개")
            self.state = self.VISIT if self.found_targets else self.RETURN
            self.replan()
            return

        if self.state == self.VISIT:
            remaining = [t for t in self.found_targets if t not in self.visited_targets]
            if not remaining:
                self.say("모든 대상 방문 완료 → 시작 지점으로 복귀")
                self.state = self.RETURN
                self.replan()
                return
            # 가까운 대상부터 (실전에선 순서 최적화도 가능하지만 대상이 몇 개면 이걸로 충분)
            remaining.sort(key=lambda t: np.hypot(t[0] - self.est.x, t[1] - self.est.y))
            for t in remaining:
                path = self.plan_to(t)
                if path is not None:
                    self._set_path(path, t)
                    return
            self.say("대상까지 경로를 못 찾음 → 남은 대상 포기하고 복귀")
            self.visited_targets += remaining
            self.state = self.RETURN
            self.replan()
            return

        if self.state == self.RETURN:
            path = self.plan_to(self.start_xy)
            if path is None:
                # 팽창 때문에 막힌 경우 미탐색 통과 허용해서라도 시도
                path = plan_path(self.inflated, self.spec, (self.est.x, self.est.y),
                                 self.start_xy, unknown_passable=True, cost_map=self.cost_map)
            if path is None:
                self.say("복귀 경로 없음?! (있을 수 없는 상황) → 종료")
                self.state = self.DONE
                return
            self._set_path(path, self.start_xy)

    def _set_path(self, path, goal):
        self.path = path
        self.goal_xy = goal
        self.follower = PathFollower(path, lookahead=self.args.lookahead)
        self.stuck_counter = 0
        self.last_progress_xy = (self.est.x, self.est.y)
        self.best_goal_dist = np.hypot(goal[0] - self.est.x, goal[1] - self.est.y) + 0.2

    # ------------------------------------------------------------------
    def path_blocked(self):
        """현재 경로의 남은 구간이 새로 발견된 벽(팽창 포함) 을 지나가는가."""
        if not self.path or self.follower is None or self.follower.done():
            return False
        from geometry import bresenham_line
        pts = [(self.est.x, self.est.y)] + self.path[self.follower.idx:]
        passable = (self.inflated != 1)
        for a, b in zip(pts[:-1], pts[1:]):
            r0, c0 = self.spec.world_to_grid(a[0], a[1])
            r1, c1 = self.spec.world_to_grid(b[0], b[1])
            rr, cc = bresenham_line(r0, c0, r1, c1)
            inb = self.spec.in_bounds(rr, cc)
            if not passable[rr[inb], cc[inb]].all():
                return True
        return False

    # ------------------------------------------------------------------
    def goal_reached(self):
        if self.goal_xy is None:
            return False
        d = np.hypot(self.goal_xy[0] - self.est.x, self.goal_xy[1] - self.est.y)
        if self.state == self.VISIT:
            return d < self.inflate_r + 0.35 or (self.follower is not None and self.follower.done())
        if self.state == self.RETURN:
            return d < 0.2
        if self.state == self.SEARCH:
            # 목표 칸을 카메라로 봤으면 굳이 거기까지 갈 필요 없음
            gr, gc = self.spec.world_to_grid(self.goal_xy[0], self.goal_xy[1])
            if self.spec.in_bounds(gr, gc) and self.seen[gr, gc]:
                return True
        return d < 0.3 or (self.follower is not None and self.follower.done())

    # ------------------------------------------------------------------
    def update(self, dt):
        """한 시뮬레이션 스텝. 반환: True 면 계속, False 면 종료."""
        if self.state == self.DONE:
            return False
        self.step_count += 1
        self.world.step_pedestrians(dt)
        self.sense_and_map(dt)
        if self.step_count % 500 == 0:      # 진행 상황 로그 (headless 디버깅용)
            known = np.mean(self.ternary != -1) * 100
            self.say(f"상태 {self.state} pose=({self.est.x:.1f},{self.est.y:.1f}) 목표={None if self.goal_xy is None else tuple(round(g, 1) for g in self.goal_xy)} "
                     f"지도 {known:.0f}% 블랙리스트 {len(self.blacklist)} 재계획 {self.replans}")

        need_replan = ((self.path is None and not (self.reactive is not None and self.state == self.EXPLORE))
                       or self.step_count - self.last_replan >= self.args.replan_every)

        # 목표 도달 처리
        if self.goal_reached():
            if self.state == self.VISIT:
                self.visited_targets.append(self.goal_xy)
                self.say(f"대상 {self.goal_xy} 위치 도착 ✓")
            elif self.state == self.RETURN:
                self.say(f"시작 지점 복귀 완료 ✓  총 {self.step_count} 스텝, 충돌 {self.collisions}회")
                self.state = self.DONE
                return False
            need_replan = True

        # 주기적으로 지도만 갱신하고 경로가 막혔는지 확인 (지도는 매 스텝 바뀜)
        if not need_replan and self.step_count % 5 == 0:
            self.rebuild_maps()
            if self.state == self.EXPLORE:
                self.fmask = frontier_mask(self.ternary, self.inflated)
            if self.path_blocked():
                need_replan = True

        # 정체 감지: 일정 시간 동안 거의 안 움직였으면 목표 블랙리스트 후 재계획
        # 정체 = (a) 제자리에 있거나 (b) 움직이긴 하는데 목표에 가까워지지 않는 경우 (DWA 가 앞뒤로 진동할 때)
        moved = np.hypot(self.est.x - self.last_progress_xy[0], self.est.y - self.last_progress_xy[1])
        d_goal = np.hypot(self.goal_xy[0] - self.est.x, self.goal_xy[1] - self.est.y) if self.goal_xy else 0.0
        if moved > 0.2 and d_goal < self.best_goal_dist - 0.1:
            self.last_progress_xy = (self.est.x, self.est.y)
            self.best_goal_dist = d_goal
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1
        if self.reactive is not None and self.state == self.EXPLORE:
            self.stuck_counter = 0
        if self.stuck_counter > self.args.stuck_steps:
            self.say(f"정체 감지 → 목표 {self.goal_xy} 블랙리스트")
            if self.goal_xy is not None and self.state in (self.EXPLORE, self.SEARCH):
                self.blacklist.append(self.goal_xy)
            self.stuck_counter = 0
            need_replan = True

        if need_replan:
            self.replan()
            if self.state == self.DONE:
                return False

        # 제어
        if self.reactive is not None and self.state == self.EXPLORE:
            d_F, d_L, d_R = lidar_sectors(self.last_ranges, self.angles, self.reactive.p, self.args.lidar_range)
            vL, vR = self.reactive.step(d_F, d_L, d_R, self.est.theta)
            v, w = wheels_to_vw(vL, vR, self.reactive.p)
            v = float(np.clip(v, -self.robot.v_max, self.robot.v_max)); w = float(np.clip(w, -self.robot.w_max, self.robot.w_max))
        elif self.args.dwa and self.path and self.follower and not self.follower.done():
            # DWA: A* 경로에서 look-ahead 점을 임시 목표로, '지금 보이는' LiDAR 점들을 장애물로
            pts = np.asarray(self.path)
            near = int(np.argmin(np.hypot(pts[:, 0] - self.est.x, pts[:, 1] - self.est.y)))
            self.follower.idx = max(self.follower.idx, min(near + 1, len(self.path) - 1))   # 지나간 점 정리
            goal = path_lookahead_point(self.est.pose, self.path[near:], lookahead=self.args.lookahead)
            # 스캔 점을 '확실한 벽'(log-odds 큼) 과 '아직 모르는 것'(방금 나타남 = 사람일 수도) 으로 나눔
            rr, cc = self.spec.world_to_grid(self.scan_pts[:, 0], self.scan_pts[:, 1])
            static_mask = np.zeros(len(rr), dtype=bool)
            for dr in (-1, 0, 1):                       # 3x3 이웃 중 하나라도 확실한 벽이면 정적
                for dc in (-1, 0, 1):
                    r2, c2 = self.spec.clip(rr + dr, cc + dc)
                    static_mask |= self.grid.log_odds[r2, c2] > 1.5
            v, w = dwa_control(self.est.pose, self.v_cmd, self.w_cmd, goal, self.scan_pts[static_mask],
                               self.dwa_params, dynamic_xy=self.scan_pts[~static_mask])
        else:
            v, w = self.follower.command(self.est) if self.follower else (0.0, 0.0)
        # 비상정지: 정면 아주 가까이에 무언가 있으면 전진 금지 (DWA 가 없을 때의 최소 안전장치)
        if not self.args.dwa and self.front_min < self.robot.radius + 0.12 and v > 0:
            v = 0.0
            self.emergency_stops += 1
            if self.emergency_stops % 20 == 1:
                self.say(f"비상정지 (정면 {self.front_min:.2f} m)")
            if self.emergency_stops % 40 == 0:      # 계속 막혀 있으면 재계획
                need_replan = True
                self.last_replan = -999
        # 이동 (진짜 세계와 충돌하는지 검사 - 심사 기준 '충돌 회피' 를 우리가 스스로 채점)
        nx = self.robot.x + v * np.cos(self.robot.theta) * dt
        ny = self.robot.y + v * np.sin(self.robot.theta) * dt
        if self.world.collides(nx, ny, self.robot.radius):
            self.collisions += 1
            dped = min([np.hypot(*(p["pos"] - np.array([self.robot.x, self.robot.y]))) - p["radius"]
                        for p in self.world.pedestrians], default=np.inf)
            self.say(f"!! 충돌 발생  v={v:.2f} 상태={self.state} 사람표면까지={dped:.2f}m")
            # 이미 겹친 상태(사람이 밀고 들어옴)면 빠져나가는 이동은 허용, 아니면 그 자리에 멈춤
            if not self.world.collides(self.robot.x, self.robot.y, self.robot.radius):
                v = 0.0
        self.robot.step(v, w, dt)
        self.v_cmd, self.w_cmd = v, w
        return True


# ======================================================================
# 5. 시각화
# ======================================================================
class Viewer:
    def __init__(self, mission: Mission):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.m = mission
        setup_korean_font()
        plt.ion()
        self.fig, (self.ax_w, self.ax_m) = plt.subplots(1, 2, figsize=(15, 6.5))
        w, spec = mission.world, mission.spec
        # 왼쪽: 진짜 세계
        self.ax_w.imshow(w.occ, origin="lower", cmap="gray_r", extent=w.spec.extent)
        self.ax_w.set_title("진짜 세계 (로봇은 이걸 모름)")
        self.scan_sc = self.ax_w.scatter([], [], s=3, c="r", label="LiDAR")
        self.trail_ln, = self.ax_w.plot([], [], "-", color="orange", lw=1, label="궤적")
        self.robot_w = self.ax_w.add_patch(plt.Circle((0, 0), mission.robot.radius, color="tab:blue"))
        self.head_w, = self.ax_w.plot([], [], "k-", lw=2)
        tx = [t[0] for t in w.targets]; ty = [t[1] for t in w.targets]
        self.ax_w.plot(tx, ty, "g^", ms=12, label="구조 대상")
        self.found_sc, = self.ax_w.plot([], [], "y*", ms=16, label="발견함")
        self.ax_w.plot(*mission.start_xy, "ks", ms=8, label="시작")
        self.ped_patches = [self.ax_w.add_patch(plt.Circle(tuple(p["pos"]), p["radius"], color="red", alpha=0.6))
                            for p in w.pedestrians]
        self.est_ln, = self.ax_w.plot([], [], "--", color="tab:blue", lw=1, label="추정 위치" if mission.args.localize else None)
        self.ax_w.legend(loc="upper right", fontsize=8)
        self.ax_w.set_xlim(w.spec.extent[0], w.spec.extent[1]); self.ax_w.set_ylim(w.spec.extent[2], w.spec.extent[3])
        # 오른쪽: 만들어지는 지도
        self.map_im = self.ax_m.imshow(self._map_rgb(), origin="lower", extent=spec.extent)
        self.front_sc = self.ax_m.scatter([], [], s=4, c="cyan", label="frontier")
        self.path_ln, = self.ax_m.plot([], [], "m.-", lw=2, ms=4, label="A* 경로")
        self.goal_mk, = self.ax_m.plot([], [], "r*", ms=16, label="목표")
        self.robot_m = self.ax_m.add_patch(plt.Circle((0, 0), mission.robot.radius, color="tab:blue"))
        self.head_m, = self.ax_m.plot([], [], "k-", lw=2)
        self.ax_m.legend(loc="upper right", fontsize=8)
        self.ax_m.set_xlim(w.spec.extent[0] - 0.5, w.spec.extent[1] + 0.5)
        self.ax_m.set_ylim(w.spec.extent[2] - 0.5, w.spec.extent[3] + 0.5)
        self.txt = self.fig.text(0.5, 0.01, "", ha="center", fontsize=11)
        plt.tight_layout(rect=(0, 0.04, 1, 1))

    def _map_rgb(self):
        """삼진 지도 + 팽창을 색으로: 미탐색 회색, 빈칸 흰색, 벽 검정, 팽창 영역 연한 빨강."""
        t, inf = self.m.ternary, self.m.inflated
        rgb = np.full(t.shape + (3,), 0.6)
        rgb[t == 0] = (1.0, 1.0, 1.0)
        rgb[(t == 0) & self.m.seen] = (0.85, 1.0, 0.85)      # 카메라로 본 빈칸 = 연두
        rgb[(inf == 1) & (t != 1)] = (1.0, 0.75, 0.75)
        rgb[t == 1] = (0.0, 0.0, 0.0)
        return rgb

    def draw(self):
        m, r = self.m, self.m.robot
        hx = [r.x, r.x + 0.35 * np.cos(r.theta)]; hy = [r.y, r.y + 0.35 * np.sin(r.theta)]
        if len(m.scan_pts):
            self.scan_sc.set_offsets(m.scan_pts)
        if r.trail:
            tr = np.array(r.trail); self.trail_ln.set_data(tr[:, 0], tr[:, 1])
        self.robot_w.center = (r.x, r.y); self.head_w.set_data(hx, hy)
        for patch, p in zip(self.ped_patches, m.world.pedestrians):
            patch.center = tuple(p["pos"])
        if m.args.localize and m.est.trail:
            et = np.array(m.est.trail); self.est_ln.set_data(et[:, 0], et[:, 1])
        if m.found_targets:
            f = np.array(m.found_targets); self.found_sc.set_data(f[:, 0], f[:, 1])
        self.map_im.set_data(self._map_rgb())
        fr = np.argwhere(m.fmask) if m.state in (Mission.EXPLORE, Mission.SEARCH) else np.zeros((0, 2), int)
        if len(fr):
            fx, fy = m.spec.grid_to_world(fr[:, 0], fr[:, 1])
            self.front_sc.set_offsets(np.column_stack([fx, fy]))
        else:
            self.front_sc.set_offsets(np.zeros((0, 2)))
        if m.path:
            p = np.array(m.path); self.path_ln.set_data(p[:, 0], p[:, 1])
        if m.goal_xy:
            self.goal_mk.set_data([m.goal_xy[0]], [m.goal_xy[1]])
        e = m.est      # 오른쪽 지도에는 로봇이 '믿는' 위치를 그림
        self.robot_m.center = (e.x, e.y); self.head_m.set_data([e.x, e.x + 0.35 * np.cos(e.theta)], [e.y, e.y + 0.35 * np.sin(e.theta)])
        known = np.mean(m.ternary != -1) * 100
        extra = f" | 위치오차 {m.pose_err_log[-1]:.2f}m" if m.args.localize and m.pose_err_log else ""
        self.txt.set_text(f"step {m.step_count} | 상태 {m.state} | 지도 채움 {known:.0f}% | "
                          f"대상 {len(m.found_targets)}발견/{len(m.visited_targets)}방문 | 충돌 {m.collisions}{extra}")
        self.ax_m.set_title("작성 중인 지도 (회색 미탐색 / 흰 빈칸 / 연두 카메라확인 / 검정 벽 / 분홍 팽창)")
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def save(self, fn):
        self.fig.savefig(fn, dpi=110)


# ======================================================================
# 6. main
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--headless", action="store_true", help="화면 없이 실행 후 PNG 저장")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--dt", type=float, default=0.1, help="시뮬레이션 스텝 [s]")
    p.add_argument("--speed", type=int, default=1, help="화면 갱신 빈도: N 스텝마다 1번 그림")
    p.add_argument("--resolution", type=float, default=0.05, help="지도 해상도 [m/칸]")
    p.add_argument("--map-half", type=float, default=11.0, help="지도 반폭 [m]: 시작점 기준 ±이 값")
    p.add_argument("--robot-radius", type=float, default=0.105, help="로봇 반지름 [m] (TurtleBot3 Burger 0.105)")
    p.add_argument("--margin", type=float, default=0.12, help="팽창 안전 여유 [m] (로봇 반지름에 더함)")
    p.add_argument("--safe-dist", type=float, default=0.5, help="A* 벽 근접 비용이 붙는 거리 [m]")
    p.add_argument("--w-prox", type=float, default=3.0, help="A* 벽 근접 비용 가중치")
    p.add_argument("--beams", type=int, default=360)
    p.add_argument("--lidar-range", type=float, default=3.5, help="LiDAR 최대 사거리 [m] (LDS-01 3.5)")
    p.add_argument("--min-frontier", type=int, default=6, help="frontier 클러스터 최소 칸 수")
    p.add_argument("--path-dist", action="store_true", help="frontier 선택에 BFS 경로 거리 사용")
    p.add_argument("--cam-fov", type=float, default=60.0, help="가짜 카메라 시야각 [deg] (Webots 월드 1.0472 rad)")
    p.add_argument("--cam-range", type=float, default=2.5, help="가짜 카메라 탐지 거리 [m]")
    p.add_argument("--min-unseen", type=int, default=40, help="SEARCH 에서 수색할 미확인 덩어리 최소 칸 수")
    p.add_argument("--no-search", action="store_true", help="SEARCH 단계 생략 (LiDAR frontier 만으로 탐색)")
    p.add_argument("--unified", action="store_true", help="EXPLORE 에 카메라 미확인 빈칸을 합쳐 한 단계로 탐색 (SEARCH 생략)")
    p.add_argument("--dwa", action="store_true", help="단순 추종기 대신 DWA(dwa.py) 로 경로 추종 + 실시간 회피")
    p.add_argument("--reactive", action="store_true", help="강의 Algorithm 3(reactive.py) 으로 돌아다니며 지도만 만들고, --reactive-steps 뒤 A* 로 방문/복귀")
    p.add_argument("--reactive-steps", type=int, default=1500)
    p.add_argument("--pedestrian", action="store_true", help="움직이는 사람 추가 (동적 장애물)")
    p.add_argument("--localize", action="store_true", help="진짜 위치 대신 노이즈 odometry + scan matching(localization.py) 추정 위치 사용")
    p.add_argument("--odom-noise", type=float, default=0.02, help="--localize 시 엔코더 노이즈 [rad/스텝]")
    p.add_argument("--odom-scale", type=float, default=1.02, help="--localize 시 바퀴 반지름 오차 배율")
    p.add_argument("--replan-every", type=int, default=15, help="N 스텝마다 재계획")
    p.add_argument("--lookahead", type=float, default=0.4, help="경로 추종 look-ahead 거리 [m] (길면 커브 안쪽 가로지름, 짧으면 지그재그)")
    p.add_argument("--stuck-steps", type=int, default=60)
    p.add_argument("--save", default="demo_result.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    world = build_world(args.seed)
    if args.pedestrian:
        # 왼쪽 아래 방을 돌아다니는 사람 (0.3 m/s). apartment.wbt 의 Pedestrian 은 0.2 m/s
        world.add_pedestrian([(1.2, 3.8), (4.3, 3.8), (4.3, 1.3), (3.3, 1.3), (3.3, 3.0), (1.2, 3.8)], radius=0.2, speed=0.3)
    robot = Robot(x=1.0, y=1.0, theta=np.deg2rad(45), radius=args.robot_radius)   # 시작 pose (대회에서 주어지는 값)
    # 지도는 미지의 환경이니 실제보다 넉넉히: 시작점 기준 ±map_half m.
    # ★ 너무 작으면 지도 밖 영역은 frontier 도 안 생기고 대상도 못 찾는데 아무 에러도 안 난다.
    #   (grid.dropped_hits 가 늘어나면 그 신호. 아래 루프에서 경고 출력)
    spec = GridSpec.from_size(2 * args.map_half, 2 * args.map_half, resolution=args.resolution,
                              center=(robot.x, robot.y))
    mission = Mission(world, robot, spec, args, rng)
    print(f"지도 격자 {spec.rows}x{spec.cols} @ {spec.resolution} m,  팽창 반지름 {mission.inflate_r:.2f} m")

    viewer = None
    if not args.headless:
        try:
            viewer = Viewer(mission)
        except Exception as e:          # 디스플레이가 없는 환경 등
            print("시각화 불가, headless 로 진행:", e)
    t0 = time.perf_counter()
    running = True
    warned_small_map = False
    while running and mission.step_count < args.max_steps:
        running = mission.update(args.dt)
        if mission.grid.dropped_hits > 50 and not warned_small_map:
            print(f"!! 경고: 지도 밖에 떨어진 벽 측정값 {mission.grid.dropped_hits}개 → 지도(--map-half)가 너무 작을 수 있음")
            warned_small_map = True
        if viewer and (mission.step_count % args.speed == 0 or not running):
            viewer.draw()
    elapsed = time.perf_counter() - t0
    known = np.mean(mission.ternary != -1) * 100
    print("-" * 60)
    print(f"종료 상태: {mission.state}   스텝: {mission.step_count}   실시간 {elapsed:.1f}s "
          f"(스텝당 {1000 * elapsed / max(1, mission.step_count):.1f} ms)")
    print(f"지도 채움: {known:.1f}%   대상 발견 {len(mission.found_targets)}/{len(world.targets)}, "
          f"방문 {len(mission.visited_targets)}   충돌 {mission.collisions}회   "
          f"시작점과의 거리 {np.hypot(robot.x - mission.start_xy[0], robot.y - mission.start_xy[1]):.2f} m")
    if args.localize and mission.pose_err_log:
        print(f"위치 추정 오차: 평균 {np.mean(mission.pose_err_log):.2f} m, 최대 {np.max(mission.pose_err_log):.2f} m, "
              f"마지막 {mission.pose_err_log[-1]:.2f} m  (scan matching 수락 {mission.matcher.accepted}/{mission.matcher.calls})")
    if viewer:
        viewer.save(args.save)
        print("결과 이미지 저장:", args.save)
        viewer.plt.ioff(); viewer.plt.show()
    elif args.headless:
        import matplotlib
        matplotlib.use("Agg")                 # 화면 없는 백엔드로 마지막 장면만 PNG 저장
        v = Viewer(mission)
        v.draw(); v.save(args.save)
        print("결과 이미지 저장:", args.save)
    return mission


if __name__ == "__main__":
    main()
