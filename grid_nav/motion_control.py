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
from robot_config import ROBOT_RADIUS, WHEEL_RADIUS, WHEEL_SEPARATION, MAX_WHEEL_SPEED, LIDAR_MAX_RANGE

LOOKAHEAD = 0.25            # [m] 길면 커브 안쪽 가로지름, 짧으면 지그재그
CRUISE_V = 0.15             # [m/s] 순항 속도 (TB3 최대 0.22)
STATIC_LOGODDS = 1.5          # planner 지도에서 이 값 초과 셀 = 벽(정적 장애물). p_hit 0.7 → 2회 관측이면 넘는다
STATIC_WINDOW_M = 3.0         # 로봇 주변 이 반경의 벽 셀만 DWA 에 넘김 (예측 1.5 s × 0.22 m/s ≪ 3 m)
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
        self.grid = None                       # set_obstacle_map(planner.grid) 로 공유되는 점유 격자 (costmap)
        self._last_ranges = None
        self._owns_grid = True
        self._scan_index = 0
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

    def set_obstacle_map(self, grid):
        """planner.grid(OccupancyGrid) 공유 — costmap 기반 DWA (fix/integration (i)).
        정적 장애물 = 이 지도의 벽 셀(log-odds>1.5) 중 로봇 3 m 안 셀 중심점. 동적 = 지도에 없는 현재 스캔 점(사람).
        (이전: 컨트롤러 자체 지도 + 같은 5 cm 셀 8스캔 지속 판정 → LiDAR 잡음 1.5 cm 로 벽 26~37 % 가 동적으로 남아 정체.)"""
        self.grid = grid
        self._owns_grid = False
        self._last_ranges = None
        self._last_scan_time = None

    def _obstacles(self, pose, ranges, angles):
        # 같은 스캔이면 이전 분류 재사용 (취득 당시 pose 기준 점 — 최신 pose 로 재변환하면 벽이 로봇을 따라 움직임)
        scan_time = self._elapsed if self._scan_timestamp is None else self._scan_timestamp
        if self._scan_timestamp is not None:
            new_scan = self._last_scan_time is None or scan_time > self._last_scan_time
        else:
            new_scan = (self._last_ranges is None
                        or not np.array_equal(ranges, self._last_ranges, equal_nan=True)
                        or scan_time - self._last_scan_time >= self.scan_period - 1e-9)
        if not new_scan:
            return self._cached_obstacles
        self._scan_index += 1
        self._last_ranges = ranges.copy()
        self._last_scan_time = scan_time
        self._last_scan_received = self._elapsed

        finite = np.isfinite(ranges) & (ranges > 0.05) & (ranges < LIDAR_MAX_RANGE)
        points = lidar_to_world(angles[finite], ranges[finite], pose)
        if self.grid is None:
            # 지도 공유 전(단독 사용): 모든 스캔 점을 동적으로 취급 (보수적)
            self._cached_obstacles = (np.empty((0, 2)), points)
            return self._cached_obstacles

        spec = self.grid.spec
        lo = self.grid.log_odds
        # --- 정적: 로봇 반경 STATIC_WINDOW_M 안의 벽 셀 중심점 ---
        r0, c0 = spec.world_to_grid(pose[0], pose[1]); k = int(STATIC_WINDOW_M / spec.resolution)
        ra, rb = max(0, int(r0) - k), min(spec.rows, int(r0) + k + 1); ca, cb = max(0, int(c0) - k), min(spec.cols, int(c0) + k + 1)
        wr, wc = np.nonzero(lo[ra:rb, ca:cb] > STATIC_LOGODDS)
        if wr.size:
            wx, wy = spec.grid_to_world(wr + ra, wc + ca)
            static = np.column_stack((np.asarray(wx, float), np.asarray(wy, float)))
        else:
            static = np.empty((0, 2))
        # --- 동적: 지도에 벽으로 없는 곳(3x3 이웃 최대 log-odds ≤ STATIC_LOGODDS)에 찍힌 스캔 점 = 사람/새 물체 ---
        if points.shape[0]:
            rr, cc = spec.world_to_grid(points[:, 0], points[:, 1])
            r = np.clip(np.asarray(rr, int), 1, spec.rows - 2); c = np.clip(np.asarray(cc, int), 1, spec.cols - 2)
            nb = np.max([lo[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)
            dynamic = points[nb <= STATIC_LOGODDS]
        else:
            dynamic = points
        self._cached_obstacles = (static, dynamic)
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
