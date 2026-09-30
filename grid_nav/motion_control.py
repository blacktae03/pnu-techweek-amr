"""
motion_control.py - 경로를 바퀴 속도로 바꾸는 국소 제어 (브랜치 feat/dwa 소유)

현재 LiDAR 끝점으로 DWA 회피. 경로는 0.5 m 앞 임시 목표를 정하는 데만 사용한다.

인터페이스 (webots_adapter.run 이 부름):
    mc = MotionController()
    v, w = mc.command(pose, path, ranges, angles, state)   # path 가 None 이거나 state=="DONE" 이면 (0, 0)
    wl, wr = to_wheel_speeds(v, w)
"""
from __future__ import annotations

import math

import numpy as np

from geometry import GridSpec, lidar_to_world
from dwa import DWAParams, dwa_control, path_lookahead_point
from occupancy_grid import OccupancyGrid
from robot_config import ROBOT_RADIUS, WHEEL_RADIUS, WHEEL_SEPARATION, MAX_WHEEL_SPEED, LIDAR_MAX_RANGE

LOOKAHEAD = 0.25            # [m] 길면 커브 안쪽 가로지름, 짧으면 지그재그
CRUISE_V = 0.15             # [m/s] 순항 속도 (TB3 최대 0.22)
STATIC_CONFIRM_SCANS = 8
STATIC_CONFIRM_SECONDS = 0.8  # 스캔 빈도가 높아져도 정적 판정 시간을 단축하지 않는다
STATIC_GAP_PERIODS = 5.0      # 같은 셀 재관측 사이 허용 간격(스캔 주기 배수). LDS-01 잡음 σ≈1.5 cm 로 점이 5 cm 셀
                              # 경계를 오가면 한 셀은 2~4 스캔마다 찍힌다 → 2.5 로는 이력이 계속 리셋돼 벽이 영원히 '동적'.
STATIC_NEIGHBOR = 1           # 정적 판정 때 이 반경(셀)의 이웃 이력까지 본다 (3x3). 이력 전파는 하지 않으므로 걷는 사람은
                              # 셀마다 1~2회 관측에 그쳐 정적으로 굳지 않는다.
STOP_DIST = ROBOT_RADIUS + 0.10   # 정면 이 거리 안에 무언가 있으면 전진 금지
RECOVER_V = 0.08                  # [m/s] DWA 복구모드에서 정면이 비었을 때 경로 추종 전진 속도


def lookahead_control(pose, path, lookahead=LOOKAHEAD, v=CRUISE_V, w_max=1.5):
    """κ = 2*y_LA / (x_LA² + y_LA²),  ω = v*κ.  y_LA, x_LA 는 로봇 좌표계에서 본 look-ahead 점."""
    pts = np.asarray(path, dtype=float)
    d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
    i_near = int(np.argmin(d))
    acc, i_la = 0.0, i_near
    while i_la < len(pts) - 1 and acc < lookahead:
        acc += float(np.hypot(*(pts[i_la + 1] - pts[i_la])))
        i_la += 1
    dx, dy = pts[i_la] - np.array([pose[0], pose[1]])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x_la = c * dx + s * dy                       # 로봇 좌표계로 회전
    y_la = -s * dx + c * dy
    if x_la < 0.05:                              # 목표가 뒤/옆에 있으면 제자리 회전
        return 0.0, 1.0 * math.copysign(1.0, y_la)
    kappa = 2.0 * y_la / (x_la ** 2 + y_la ** 2)
    return v, float(np.clip(v * kappa, -w_max, w_max))


def to_wheel_speeds(v, w):
    """(선속도, 각속도) → (왼바퀴, 오른바퀴) 각속도 [rad/s], 모터 한계로 비율 유지하며 축소."""
    vl = v - w * WHEEL_SEPARATION / 2.0
    vr = v + w * WHEEL_SEPARATION / 2.0
    wl, wr = vl / WHEEL_RADIUS, vr / WHEEL_RADIUS
    m = max(abs(wl), abs(wr), 1e-9)
    limit = MAX_WHEEL_SPEED * (1.0 - 1e-12)  # 부동소수점 반올림으로 maxVelocity 초과 방지
    if m > limit:
        wl, wr = wl * limit / m, wr * limit / m
    return wl, wr


class MotionController:
    def __init__(self):
        self.v_cmd = self.w_cmd = 0.0          # DWA 의 dynamic window 기준이 될 직전 명령
        self.emergency_stops = 0
        self.recoveries = 0                    # DWA 복구모드(전 후보 탈락) 진입 횟수 (진단)
        self.params = DWAParams(v_max=0.22, dyn_margin=0.20, clearance_max=0.5,
                                w_dist=2.0,
                                control_dt=0.064,  # apartment 월드 기본 64 ms. set_timing으로 설정 가능
                                wheel_v_max=WHEEL_RADIUS * MAX_WHEEL_SPEED * (1.0 - 1e-12),
                                wheel_separation=WHEEL_SEPARATION)
        # command 인터페이스에는 planner가 없다. 동일 센서로 별도 이력 지도만
        # 유지하며, 장애물 자체는 언제나 현재 스캔에서 가져온다.
        self.grid = None
        self._last_ranges = None
        self._owns_grid = True
        self._scan_index = 0
        self._last_hit_scan = None
        self._stable_hit_count = None
        self._first_hit_time = self._last_hit_time = None
        self.scan_period = 0.1
        self._elapsed = 0.0
        self._scan_timestamp = None
        self._last_scan_time = None
        self._last_scan_received = None
        self._cached_obstacles = (np.empty((0, 2)), np.empty((0, 2)))
        self._virtual_obstacles = []

    def remember_obstacle(self, x, y, radius=0.15):
        """헛바퀴로 확인한 LiDAR 아래 장애물도 국소 궤적에서 피한다. GT 미사용."""
        if not np.isfinite([x, y, radius]).all() or radius <= 0:
            raise ValueError("invalid obstacle")
        if not any(math.hypot(x - a, y - b) < radius / 2 for a, b, _ in self._virtual_obstacles):
            self._virtual_obstacles.append((x, y, radius))

    def _remembered_points(self, pose):
        points = []
        for x, y, radius in self._virtual_obstacles:
            if math.hypot(x - pose[0], y - pose[1]) < LIDAR_MAX_RANGE:
                # 원 내부도 채워서 시작 pose/궤적이 원 안에 들어가 안전하다고 오판하지 않는다.
                xx, yy = np.meshgrid(np.arange(-radius, radius + .025, .025),
                                     np.arange(-radius, radius + .025, .025))
                keep = xx ** 2 + yy ** 2 <= radius ** 2
                points.append(np.column_stack((x + xx[keep], y + yy[keep])))
        return np.vstack(points) if points else np.empty((0, 2))

    def set_timing(self, control_dt, scan_period=0.1):
        """초 단위 실제 제어/센서 주기를 지정한다. command 시그니처는 유지한다.

        통합 예: motion.set_timing(robot.getBasicTimeStep() / 1000, 0.1).
        궤적 적분 간격(params.dt)과 명령 가속도 제한 주기는 독립적이다.
        """
        if not all(np.isfinite(v) and v > 0 for v in (control_dt, scan_period)):
            raise ValueError("control_dt and scan_period must be positive and finite")
        self.params.control_dt = float(control_dt)
        self.scan_period = float(scan_period)

    def set_scan_timestamp(self, timestamp):
        """스캔 취득 시각[s]. 반복 프레임은 같은 시각, 새 프레임은 증가한 시각.

        동일 거리 배열의 새 프레임도 처리하며, timestamp가 멈추면 정지한다.
        미연결 시에는 제어 주기와 scan_period로 새 프레임을 추정한다.
        이 호환 모드로는 정상적인 고정 장면과 센서 정지를 구별할 수 없다.
        """
        if not np.isfinite(timestamp) or (self._scan_timestamp is not None
                                          and timestamp < self._scan_timestamp):
            raise ValueError("scan timestamp must be finite and monotonic")
        self._scan_timestamp = float(timestamp)

    def set_applied_velocity(self, v, w):
        """CONFIRM/후진 복구 등 외부 덮어쓰기 후 실제 적용한 명령을 알려준다."""
        if not np.isfinite([v, w]).all():
            raise ValueError("applied velocity must be finite")
        wl, wr = to_wheel_speeds(v, w)
        self.v_cmd = WHEEL_RADIUS * (wl + wr) / 2.0
        self.w_cmd = WHEEL_RADIUS * (wr - wl) / WHEEL_SEPARATION

    def guard_command(self, v, w, ranges, angles):
        """DWA 밖의 명령도 짧은 주행 궤적과 정지거리 검사. 후방 장애물도 포함."""
        ranges, angles = np.asarray(ranges, float), np.asarray(angles, float)
        if (ranges.ndim != 1 or ranges.shape != angles.shape or ranges.size == 0
                or not np.isfinite(angles).all() or not np.isfinite([v, w]).all()):
            return 0.0, 0.0
        valid = (np.isfinite(ranges) & (ranges > 0)) | np.isposinf(ranges)
        if valid.mean() < 0.9:
            return 0.0, 0.0
        finite = np.isfinite(ranges) & (ranges > 0) & (ranges < LIDAR_MAX_RANGE)
        if not finite.any():
            return float(v), float(w)
        points = np.column_stack((ranges[finite] * np.cos(angles[finite]),
                                  ranges[finite] * np.sin(angles[finite])))
        horizon = self.params.control_dt + self.scan_period + abs(v) / self.params.a_v
        ts = np.linspace(0, horizon, 20)
        if abs(w) < 1e-6:
            centers = np.column_stack((v * ts, np.zeros_like(ts)))
        else:
            centers = np.column_stack((v / w * np.sin(w * ts), v / w * (1 - np.cos(w * ts))))
        distance = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        if distance.min() < self.params.robot_radius + 0.02:
            self.emergency_stops += 1
            return 0.0, 0.0
        return float(v), float(w)

    def _reset_static_evidence(self):
        """점유 log-odds 외에 위치가 일정 시간 유지됐는지도 별도로 기록한다."""
        shape = self.grid.spec.shape
        self._last_hit_scan = np.full(shape, -1_000_000, dtype=np.int32)
        self._stable_hit_count = np.zeros(shape, dtype=np.uint16)
        self._first_hit_time = np.full(shape, -np.inf)
        self._last_hit_time = np.full(shape, -np.inf)
        self._scan_index = 0

    def set_obstacle_map(self, grid):
        """통합 시 planner.grid를 공유할 수 있다. 공유 지도는 여기서 갱신하지 않는다.

        기존 adapter는 이 메서드 없이도 작동한다. 공유할 때에는 adapter가
        planner.update_map 이후 command를 호출해야 한다.
        """
        self.grid = grid
        self._owns_grid = False
        self._last_ranges = None
        self._last_scan_time = None
        self._reset_static_evidence()

    def _obstacles(self, pose, ranges, angles):
        if self.grid is None:
            self.grid = OccupancyGrid(
                GridSpec.from_size(30.0, 30.0, 0.05, center=pose[:2]),
                max_range=LIDAR_MAX_RANGE)
            self._reset_static_evidence()
        elif self._last_hit_scan is None or self._last_hit_scan.shape != self.grid.spec.shape:
            self._reset_static_evidence()

        # 거리값이 같다는 사실만으로 중복 프레임이라고 판단하지 않는다.
        scan_time = self._elapsed if self._scan_timestamp is None else self._scan_timestamp
        if self._scan_timestamp is not None:
            new_scan = self._last_scan_time is None or scan_time > self._last_scan_time
        else:
            new_scan = (self._last_ranges is None
                        or not np.array_equal(ranges, self._last_ranges, equal_nan=True)
                        or scan_time - self._last_scan_time >= self.scan_period - 1e-9)
        if not new_scan:
            # 이미 받은 스캔은 취득 당시 pose로 변환한 점을 재사용한다.
            # 최신 pose로 다시 변환하면 정지한 벽도 로봇을 따라 움직이게 된다.
            return self._cached_obstacles

        finite = np.isfinite(ranges) & (ranges > 0.05) & (ranges < LIDAR_MAX_RANGE)
        points = lidar_to_world(angles[finite], ranges[finite], pose)
        rr, cc = self.grid.spec.world_to_grid(points[:, 0], points[:, 1])
        if new_scan:
            self._scan_index += 1
            self._last_ranges = ranges.copy()
            self._last_scan_time = scan_time
            self._last_scan_received = self._elapsed

            in_bounds = self.grid.spec.in_bounds(rr, cc)
            flat = np.unique(rr[in_bounds] * self.grid.spec.cols + cc[in_bounds])
            hit_r, hit_c = np.divmod(flat, self.grid.spec.cols)
            previous = self._last_hit_time[hit_r, hit_c]
            continuing = scan_time - previous <= STATIC_GAP_PERIODS * self.scan_period
            self._first_hit_time[hit_r, hit_c] = np.where(
                continuing, self._first_hit_time[hit_r, hit_c], scan_time)
            count = self._stable_hit_count[hit_r, hit_c]
            self._stable_hit_count[hit_r, hit_c] = np.where(
                continuing, np.minimum(count.astype(np.uint32) + 1, 65535), 1
            ).astype(np.uint16)
            self._last_hit_scan[hit_r, hit_c] = self._scan_index
            self._last_hit_time[hit_r, hit_c] = scan_time

        # log-odds만으로 분류하면 천천히 움직이는 사람도 몇 프레임 만에 정적으로
        # 굳는다. 이 점이 찍힌 같은 셀 자체가 8회 이상 지속 관측돼야 정적으로 본다.
        # 주변 셀까지 빌려오면 벽 옆을 걷는 사람을 벽으로 오인할 수 있어 쓰지 않는다.
        in_bounds = self.grid.spec.in_bounds(rr, cc)
        r, c = self.grid.spec.clip(rr, cc)
        # 판정은 3x3 이웃 중 '최근 관측된' 셀의 최장 이력으로 한다 (fix/integration).
        # 왜: 잡음으로 점이 인접 셀을 오가면 셀 하나의 이력만으로는 벽의 26~37 % 가 정적 확정을 영원히 못 받았고,
        #     그 점들이 dyn_margin(0.455 m) 을 요구해 실내 통로에서 궤적 87~96 % 가 탈락 → 후진/회전만 반복했다.
        offs = np.arange(-STATIC_NEIGHBOR, STATIC_NEIGHBOR + 1)
        nr = np.clip(r[None, :] + np.repeat(offs, len(offs))[:, None], 0, self.grid.spec.rows - 1)
        nc = np.clip(c[None, :] + np.tile(offs, len(offs))[:, None], 0, self.grid.spec.cols - 1)
        nb_recent = scan_time - self._last_hit_time[nr, nc] <= STATIC_GAP_PERIODS * self.scan_period
        nb_count = np.where(nb_recent, self._stable_hit_count[nr, nc], 0).max(axis=0)
        nb_first = np.where(nb_recent, self._first_hit_time[nr, nc], np.inf).min(axis=0)
        persistent = (in_bounds & nb_recent.any(axis=0)
                      & (scan_time - nb_first >= STATIC_CONFIRM_SECONDS - 1e-9)
                      & (nb_count >= STATIC_CONFIRM_SCANS))
        nb_logodds = self.grid.log_odds[nr, nc].max(axis=0)

        if new_scan and self._owns_grid:
            # Unconfirmed hits are the dynamic/unknown set. Skip those beams entirely
            # instead of turning them into free rays through the moving obstacle.
            # Thus a person is still present in `dynamic` for this DWA cycle, while
            # its transient position does not accumulate in the static log-odds map.
            map_ranges = ranges.copy()
            map_ranges[np.isnan(map_ranges) | np.isneginf(map_ranges)] = 0.0
            hit_indices = np.flatnonzero(finite)
            map_ranges[hit_indices[~persistent]] = 0.0
            self.grid.update(pose, angles, map_ranges)
        static = persistent & (nb_logodds > 1.5)
        self._cached_obstacles = (points[static], points[~static])
        return self._cached_obstacles

    def command(self, pose, path, ranges, angles, state):
        self._elapsed += self.params.control_dt
        if (state == "DONE" or ranges is None or angles is None
                or np.asarray(pose).shape != (3,) or not np.isfinite(pose).all()):
            self.v_cmd = self.w_cmd = 0.0
            return 0.0, 0.0
        ranges = np.asarray(ranges, dtype=float)
        angles = np.asarray(angles, dtype=float)
        if (ranges.ndim != 1 or angles.shape != ranges.shape or len(ranges) == 0
                or not np.isfinite(angles).all()
                or not (np.isposinf(ranges) | (np.isfinite(ranges) & (ranges > 0.05))).any()):
            self.v_cmd = self.w_cmd = 0.0
            self.emergency_stops += 1
            return 0.0, 0.0
        obstacles, dynamic = self._obstacles(pose, ranges, angles)
        obstacles = np.vstack((obstacles, self._remembered_points(pose)))
        if (self._scan_timestamp is not None and self._last_scan_received is not None
                and self._elapsed - self._last_scan_received > max(3 * self.scan_period, 0.3)):
            self.v_cmd = self.w_cmd = 0.0
            self.emergency_stops += 1
            return 0.0, 0.0
        if state == "DONE" or path is None or len(path) == 0:
            v, w = 0.0, 0.0
        else:
            goal = path_lookahead_point(pose, path, 0.5)
            # 0.5 m 앞점이 모서리 너머라면 그 점만 향하는 DWA는 정체될 수 있다.
            # 현재 스캔으로 연결 구간이 막혔을 때만, 그 앞의 보이는 경로점을 택한다.
            if len(obstacles):
                start = np.asarray(pose[:2], dtype=float)

                def visible(point):
                    delta = np.asarray(point) - start
                    length2 = float(delta @ delta)
                    if length2 < 1e-8:
                        return True
                    u = np.clip((obstacles - start) @ delta / length2, 0.0, 1.0)
                    distances = np.linalg.norm(obstacles - (start + u[:, None] * delta), axis=1)
                    return bool(np.all(distances >= self.params.robot_radius + self.params.safety_margin))

                if not visible(goal):
                    pts = np.asarray(path, dtype=float)
                    near = int(np.argmin(np.linalg.norm(pts - start, axis=1)))
                    distance = 0.0
                    visible_goal = None
                    for i in range(near + 1, len(pts)):
                        distance += float(np.linalg.norm(pts[i] - pts[i - 1]))
                        if not visible(pts[i]):
                            break
                        if np.linalg.norm(pts[i] - start) >= 0.08:
                            visible_goal = tuple(pts[i])
                        if distance >= 0.5:
                            break
                    if visible_goal is not None:
                        goal = visible_goal
            (v, w), dbg = dwa_control(pose, self.v_cmd, self.w_cmd, goal, obstacles,
                                      self.params, dynamic_xy=dynamic, return_debug=True)
            if dbg["best"] is None:
                # 모든 후보가 여유 기준 위반(복구 모드). 정면이 실제로 비어 있으면 후진/회전 대신
                # 경로를 따라 느리게 전진한다 (fix/integration). 벽 옆 사과·좁은 통로에서 제자리 진동 방지.
                self.recoveries += 1
                front = ranges[np.abs(angles) <= math.radians(30)]
                front = front[np.isfinite(front) & (front > 0.05)]
                if front.size == 0 or float(front.min()) > STOP_DIST:
                    v, w = lookahead_control(pose, path, v=RECOVER_V)
        # adapter의 모터 포화와 같은 속도를 다음 dynamic window의 기준으로 삼는다.
        wl, wr = to_wheel_speeds(v, w)
        v = WHEEL_RADIUS * (wl + wr) / 2.0
        w = WHEEL_RADIUS * (wr - wl) / WHEEL_SEPARATION
        self.v_cmd, self.w_cmd = v, w
        return v, w
