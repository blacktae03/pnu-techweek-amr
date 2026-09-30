"""
webots_adapter.py - Webots(TurtleBot3 Burger) 컨트롤러 조립 코드 (브랜치 integration 소유)

역할 분담 (브랜치 = 파일). 이 파일은 아래 모듈을 순서대로 부르기만 한다.
  robot_config.py       로봇/센서 상수                       (integration)
  pose_estimation.py    (1) 위치 추정   PoseEstimator        (feat/localization)
  exploration.py        (2)(4) 지도·frontier·A*  GridPlanner (feat/exploration)
  target_detection.py   (3) 카메라 대상 탐지  TargetDetector  (feat/vision)
  motion_control.py     (5) 경로 → 바퀴 속도  MotionController (feat/dwa)
  webots_adapter.py     (4) 미션 상태 기계, (6) 로그/지도 저장/Display  (integration)

확인된 사실 [2026-09-30]:
  - 디바이스 이름: "LDS-01", "camera", "gyro", "compass", "left/right wheel motor", 엔코더는 motor.getPositionSensor()
  - LiDAR: ranges[180]=정면, [0]=후방, [90]=왼쪽, [270]=오른쪽, 100 ms 주기 → 같은 스캔이 반복 들어옴 (새 스캔만 갱신)
  - 컴퍼스 부호 -1 (robot_config.COMPASS_SIGN), 시작 pose apartment (-0.3, -7.5, pi)
  - 첫 스캔 전에 계획하면 "frontier 없음 → 탐색 완료" 가 되어 즉시 끝남 → 스캔 3개 대기 + 3회 연속 확인
실행: controllers/tb3_gridnav/tb3_gridnav.py 가 run() 을 부른다. 터미널에서 직접 실행하면 뼈대 점검만.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from geometry import wrap_angle
from robot_config import (ROBOT_RADIUS, LIDAR_NAME, LIDAR_PERIOD_MS, LIDAR_MAX_RANGE, MAX_WHEEL_SPEED,
                          COMPASS_SIGN, START_POSE_APARTMENT)
from pose_estimation import PoseEstimator, WheelOdometry, CompassHeading          # 재수출 (tb3_reactive 가 씀)
from exploration import GridPlanner, REPLAN_PERIOD_S
from astar import plan_path
from target_detection import (TargetDetector, target_world_from_pixel, distance_from_row,
                              CONSISTENCY_RATIO, HORIZON_MARGIN_PX, VISION_EVERY)
import vision
from motion_control import MotionController, lookahead_control, to_wheel_speeds

try:
    from controller import Robot          # Webots 가 제공. Webots 밖에서는 없음
    _IN_WEBOTS = True
except ImportError:
    Robot = object                        # type: ignore
    _IN_WEBOTS = False

TARGET_COUNT = 2            # 서로 다른 대상의 재확인 + 최종 접근 성공 개수
CONFIRM_DIST = 0.70         # 1차 정지·재확인 거리. 카메라(높이 8.8 cm, 수평)에 사과가 화면 안에 들어오는 거리 (0.4 m 이하는 화면 아래로 빠짐)
ARRIVE_DIST = 0.35          # 최종 "도달" 판정 거리 (로봇 중심~사과). 당일 규칙에 맞출 것
APPROACH_V = 0.12           # [m/s] 최종 접근 속도 (pure pursuit, DWA 미사용 — 벽 옆 사과는 DWA 여유 거리 안에 있음)
LOOKAROUND_W = 1.2          # [rad/s] 탐색 목표를 바꿀 때 제자리 한 바퀴 둘러보기 (카메라 60° 는 좁다: 1차 실행에서 빨간 사과 2.5 m 옆을
LOOKAROUND_MIN_UNSEEN = 120 #   137회 지나치며 한 번도 정면에 두지 않음). 주변 2 m 에 미확인 칸이 이만큼 있을 때만, 25 s 에 1회.
LOOKAROUND_COOLDOWN_S = 25.0
HOME_DIST = 0.2             # 복귀 완료 판정
CONFIRM_HOLD_S = 3.0        # 대상 도착 후 사과를 바라보며 정지·재확인하는 시간 (심사자에게 "도달" 이 보이게)
CONFIRM_SPIN_S = 4.0        # 정지 중 못 보면 제자리 회전으로 찾는 최대 시간
HOLD_SECONDS = float(os.environ.get("GRIDNAV_HOLD", "0"))   # 테스트용: 처음 N초 정지
# 헛바퀴(slip) 감지: 전진 명령 중인데 스캔이 1초 전과 거의 같으면 로봇은 안 움직인 것 (낮은 물체에 걸림)
SLIP_WINDOW_S = 1.0         # 비교할 과거 스캔의 시간 차
SLIP_SCAN_CHANGE_M = 0.04   # 이보다 스캔 변화가 작으면 '정지' (0.15 m/s × 1 s = 0.15 m 변해야 정상)
SLIP_CONFIRM_S = 2.0        # 이 시간 연속 정지 판정이면 걸린 것으로 확정
SLIP_SECTOR_DEG = 40        # 앞뒤 ±이 각도의 빔만 비교. 복도에서 옆 벽 빔은 직진해도 안 변하므로 제외 (4차 실행 오판 원인)
RECOVER_BACK_S = 3.0        # 복구: 후진 시간 (0.1 m/s → 0.3 m)
OBSTACLE_AHEAD_M = 0.25     # 복구 시 로봇 앞 이 거리에 장애물을 찍음


class MissionVisits:
    """Detector.targets는 append/동일 인덱스 갱신만 한다. 좌표 대신 인덱스로 방문 관리."""
    def __init__(self, required=TARGET_COUNT):
        self.required = required
        self.completed = set()
        self.retry_after = {}
        self.active = None

    @property
    def complete(self):
        return len(self.completed) >= self.required

    def select(self, targets, pose, now):
        remaining = [i for i in range(len(targets)) if i not in self.completed
                     and now >= self.retry_after.get(i, -np.inf)]
        self.active = min(remaining, key=lambda i: math.dist(pose[:2], targets[i])) if remaining else None
        return self.active

    def defer(self, now):
        if self.active is not None:
            self.retry_after[self.active] = now + 20.0
        self.active = None

    def finish(self, distance, confirmed):
        if self.active is None or not confirmed or not np.isfinite(distance) or distance >= ARRIVE_DIST:
            return False
        self.completed.add(self.active)
        self.active = None
        return True


class ProgressWatchdog:
    """명령 속도와 무관하게 목표를 향한 주행 중 위치 정체를 감지한다."""
    def __init__(self, timeout=15.0, distance=0.15):
        self.timeout, self.distance = timeout, distance
        self.anchor = None
        self.since = None

    def update(self, pose, now, navigating):
        if not navigating:
            self.anchor = None; self.since = None
            return False
        if self.anchor is None or math.dist(pose[:2], self.anchor) > self.distance:
            self.anchor = pose[:2]; self.since = now
            return False
        return now - self.since >= self.timeout


def confirms_active_target(detector, pose, target):
    """이번에 처리한 유효 카메라 관측만 현재 방문 대상의 재확인으로 사용."""
    if detector.step_i % VISION_EVERY or detector.consecutive < 2 or detector.last_det is None:
        return False
    cx, cy, radius = detector.last_det
    _, bearing, d_radius = target_world_from_pixel(pose, cx, radius, detector.cam_w, detector.cam_fov)
    d_row = distance_from_row(cy, detector.cam_h, detector.cam_w, detector.cam_fov)
    if cy < detector.cam_h / 2 + HORIZON_MARGIN_PX or not 1 / CONSISTENCY_RATIO < d_radius / d_row < CONSISTENCY_RATIO:
        return False
    distance = math.sqrt(d_radius * d_row)
    observed = (pose[0] + distance * math.cos(pose[2] + bearing),
                pose[1] + distance * math.sin(pose[2] + bearing))
    return distance < 1.0 and math.dist(observed, target) < 0.4


# ======================================================================
# LiDAR 각도 배열  [확인됨: LDS-01 인덱스 매핑]
# ======================================================================
def lidar_angles(n_points=360):
    """ranges[i] 의 각도 (정면 0, 왼쪽 +). angle_i = pi - 2*pi*i/N  → i=180 정면, i=90 왼쪽, i=270 오른쪽."""
    i = np.arange(n_points)
    return wrap_angle(np.pi - 2.0 * np.pi * i / n_points)


def read_lidar(lidar):
    return np.array(lidar.getRangeImage(), dtype=float)


# ======================================================================
# Display 에 지도 그리기 (개발용 월드의 map_display)
# ======================================================================
def draw_map_on_display(display, W, H, planner, pose, goal, path, targets, state):
    """지도의 '알려진 영역' 을 잘라 Display 에 확대해 그린다. Display 는 y=0 이 위라 세로를 뒤집는다."""
    import cv2
    from controller import Display
    tern, spec = planner.get_map()
    known = tern != -1
    if not known.any():
        return
    r_, c_ = np.nonzero(known)
    r0, r1 = max(0, r_.min() - 10), min(spec.rows, r_.max() + 11)
    c0, c1 = max(0, c_.min() - 10), min(spec.cols, c_.max() + 11)
    crop = tern[r0:r1, c0:c1]
    rgb = np.full(crop.shape + (3,), 150, np.uint8)
    rgb[crop == 0] = 255
    rgb[crop == 1] = 0
    rgb = rgb[::-1]
    scale = min(W / rgb.shape[1], H / rgb.shape[0])
    ow, oh = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
    canvas = np.full((H, W, 3), 40, np.uint8)
    canvas[:oh, :ow] = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_NEAREST)
    x0 = spec.origin_x + c0 * spec.resolution
    y1 = spec.origin_y + r1 * spec.resolution

    def to_px(x, y):
        return int((x - x0) / spec.resolution * scale), int((y1 - y) / spec.resolution * scale)

    if path:
        cv2.polylines(canvas, [np.array([to_px(x, y) for x, y in path], np.int32)], False, (200, 0, 200), 2)   # 자주 (RGB)
    for tx, ty in targets:
        cv2.drawMarker(canvas, to_px(tx, ty), (255, 210, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)   # 노랑 = 기록한 사과
    if goal is not None:
        cv2.drawMarker(canvas, to_px(goal[0], goal[1]), (255, 0, 0), cv2.MARKER_STAR, 16, 2)   # 빨강 = 현재 목표
    px, py = to_px(pose[0], pose[1])
    cv2.circle(canvas, (px, py), max(3, int(ROBOT_RADIUS / spec.resolution * scale)), (0, 120, 255), -1)   # 파랑 = 로봇
    cv2.line(canvas, (px, py), (int(px + 12 * math.cos(pose[2])), int(py - 12 * math.sin(pose[2]))), (0, 0, 0), 2)
    cv2.putText(canvas, state, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    im = display.imageNew(canvas.tobytes(), Display.RGB, W, H)
    display.imagePaste(im, 0, 0, False)
    display.imageDelete(im)


# ======================================================================
# 컨트롤러 본체
# ======================================================================
def main():
    if not _IN_WEBOTS:
        print("Webots 밖에서 실행됨: controller 모듈이 없어 뼈대 점검만 합니다. (import/문법 OK)")
        planner = GridPlanner(start_xy=START_POSE_APARTMENT[:2])
        tern, spec = planner.get_map()
        ang = lidar_angles(360)
        print(f"  GridPlanner 준비: 지도 {tern.shape}, 해상도 {spec.resolution} m")
        print(f"  LiDAR 각도: idx0={math.degrees(ang[0]):.0f}°(후방) idx90={math.degrees(ang[90]):.0f}°(왼쪽) "
              f"idx180={math.degrees(ang[180]):.0f}°(정면) idx270={math.degrees(ang[270]):.0f}°(오른쪽)")
        (tx, ty), b, d = target_world_from_pixel((0, 0, 0), cx_px=400, radius_px=20)
        print(f"  픽셀(400, r=20) → bearing {math.degrees(b):.1f}°, dist {d:.2f} m, world ({tx:.2f}, {ty:.2f})")
        return
    run(Robot(), start_pose=START_POSE_APARTMENT)


def run(robot, start_pose, ground_truth=None, map_save_path="map.npy", map_save_every_s=2.0,
        use_scan_matching=False):
    """robot: Robot() 또는 Supervisor().  start_pose: 대회 제공 (x, y, theta).
    ground_truth: 디버깅용 콜백 (진짜 pose). map_save_path: 지도/상태 npz 저장 위치 (view_live.py 용)."""
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    # --- 디바이스 ---
    lidar = robot.getDevice(LIDAR_NAME); lidar.enable(LIDAR_PERIOD_MS); lidar.enablePointCloud()
    left_motor = robot.getDevice("left wheel motor"); right_motor = robot.getDevice("right wheel motor")
    for m in (left_motor, right_motor):
        m.setPosition(float("inf")); m.setVelocity(0.0)
    left_enc = left_motor.getPositionSensor();   left_enc.enable(timestep)
    right_enc = right_motor.getPositionSensor(); right_enc.enable(timestep)
    gyro = robot.getDevice("gyro");       gyro.enable(timestep)
    compass = robot.getDevice("compass"); compass.enable(timestep)
    camera = robot.getDevice("camera");   camera.enable(timestep)
    try:
        display = robot.getDevice("map_display"); disp_w, disp_h = display.getWidth(), display.getHeight()
    except Exception:
        display = None
    debug_dir = os.path.dirname(os.path.abspath(map_save_path)) if map_save_path else None

    # --- 모듈 (브랜치별) ---
    planner = GridPlanner(start_xy=start_pose[:2], max_range=min(lidar.getMaxRange(), LIDAR_MAX_RANGE))
    estimator = PoseEstimator(start_pose, use_scan_matching=use_scan_matching, spec=planner.spec)
    detector = TargetDetector(camera.getWidth(), camera.getHeight(), camera.getFov(), debug_dir)
    motion = MotionController()
    motion.set_timing(dt, LIDAR_PERIOD_MS / 1000.0)
    motion.set_obstacle_map(planner.grid)      # costmap 공유: DWA 정적 장애물 = 지도 벽, 동적 = 지도에 없는 스캔 점

    # --- 미션 상태 ---
    x0, y0 = start_pose[0], start_pose[1]
    state = "EXPLORE"                   # EXPLORE → VISIT → CONFIRM → APPROACH → RETURN → DONE
    confirm_until = -1e9; confirm_target = None; confirm_seen = False; confirm_spin_until = -1e9
    path: Optional[List[Tuple[float, float]]] = None
    goal: Optional[Tuple[float, float]] = None
    visits = MissionVisits()
    angles = None; last_scan = None; scan_count = 0; no_frontier_count = 0
    scan_hist: List[Tuple[float, np.ndarray]] = []     # (t, ranges) 최근 스캔들 (slip 감지용)
    slip_time = 0.0; slip_pose_ref = None; recover_until = -1e9; slip_events = 0; slip_goal_count = {}
    last_plan_t = last_save = last_log = last_disp = -1e9
    last_map_warn = -1e9
    visit_fail = 0; return_fail = 0      # VISIT/RETURN 에서 경로 실패 연속 횟수 (무한 회전 방지)
    search_done = 0
    marked_obstacles = 0                 # detector.obstacles 중 지도에 찍은 개수
    spin_until = -1e9; last_spin = -1e9; prev_goal = None; lookarounds = 0
    trail: List[Tuple[float, float, float]] = []   # (t, x, y) 최근 10 s 궤적 — 후진 허용 판정용
    unsafe_backs = 0

    def back_is_safe(pose_):
        """로봇 뒤 0.5 m 구간이 카메라로 본 곳이거나 직전 10 s 에 지나온 궤적 위일 때만 후진 허용.
        1차 실행: LiDAR 가 못 보는 낮은 물체(사과) 쪽으로 물러날 위험. 아니면 제자리 회전으로 대체."""
        for frac in (0.25, 0.5):
            bx = pose_[0] - frac * math.cos(pose_[2]); by = pose_[1] - frac * math.sin(pose_[2])
            r_, c_ = planner.spec.world_to_grid(bx, by)
            if not planner.spec.in_bounds(r_, c_):
                return False
            seen_ok = bool(planner.seen[int(r_), int(c_)])
            trail_ok = any(math.hypot(bx - tx, by - ty) < 0.25 for _, tx, ty in trail)
            if not (seen_ok or trail_ok):
                return False
        return True
    v = w = 0.0
    last_scan_t = -np.inf
    progress_since = None; progress_xy = None
    watchdog = ProgressWatchdog()
    # 진단 전용: 바닥보다 높은 접촉점의 연속 구간 수. 제어 입력에는 쓰지 않는다.
    contact_node = None; contact_events = 0; in_contact = False
    if ground_truth is not None and hasattr(robot, "getSelf"):
        contact_node = robot.getSelf()
        contact_node.enableContactPointsTracking(timestep, True)

    def clip_to_map(g):
        """목표가 지도 밖이면 가장자리에서 0.5 m 안쪽으로 클립 (위치 오차로 목표가 지도 밖에 찍히는 경우)."""
        if g is None:
            return None
        sp = planner.spec
        gx = min(max(g[0], sp.origin_x + 0.5), sp.origin_x + sp.width_m - 0.5)
        gy = min(max(g[1], sp.origin_y + 0.5), sp.origin_y + sp.height_m - 0.5)
        if (gx, gy) != (g[0], g[1]):
            print(f"[{t:.1f}s] 목표 ({g[0]:.2f}, {g[1]:.2f}) 가 지도 밖 → ({gx:.2f}, {gy:.2f}) 로 클립")
        return (gx, gy)

    while robot.step(timestep) != -1:
        t = robot.getTime()
        ranges = read_lidar(lidar)
        planner.now = t                                    # 블랙리스트 만료 계산용
        if angles is None:
            angles = lidar_angles(len(ranges))
        # 값이 같은 새 스캔도 처리한다. 센서 API에 획득 timestamp가 없어 명목 주기 사용.
        new_scan = (last_scan is None or not np.array_equal(ranges, last_scan, equal_nan=True)
                    or t - last_scan_t >= LIDAR_PERIOD_MS / 1000.0)
        valid_scan = new_scan and np.isfinite(ranges).any() and np.nanmax(np.where(np.isfinite(ranges), ranges, 0)) > 0.0

        # (1) 위치 추정  [feat/localization]
        pose = estimator.update(left_enc.getValue(), right_enc.getValue(), compass.getValues(),
                                gyro_z=gyro.getValues()[2], dt=dt,
                                ranges=ranges if valid_scan else None, angles=angles, log_odds=planner.grid.log_odds)

        # (1b) 헛바퀴 감지  [integration]: 전진 중인데 환경(스캔)이 안 변하면 로봇은 제자리 → odometry 증가분 취소
        if valid_scan:
            scan_hist.append((t, ranges)); scan_hist = [(ts, r) for ts, r in scan_hist if t - ts <= SLIP_WINDOW_S + 0.2]
        trail.append((t, pose[0], pose[1])); trail = [p_ for p_ in trail if t - p_[0] <= 10.0]
        moving_cmd = abs(motion.v_cmd) > 0.05 and abs(motion.w_cmd) < 0.4
        if moving_cmd and state not in ("CONFIRM", "DONE", "BLOCKED") and t >= recover_until:
            if progress_xy is None or math.dist(pose[:2], progress_xy) > 0.08:
                progress_xy = pose[:2]; progress_since = t
        else:
            progress_xy = None; progress_since = None
        if moving_cmd and len(scan_hist) >= 2 and t - scan_hist[0][0] >= SLIP_WINDOW_S * 0.8 and t >= recover_until:
            old = scan_hist[0][1]
            sec = np.deg2rad(SLIP_SECTOR_DEG)
            fb = (np.abs(wrap_angle(angles)) < sec) | (np.abs(wrap_angle(angles - np.pi)) < sec)   # 앞뒤 빔만
            both = fb & np.isfinite(ranges) & np.isfinite(old)
            change = float(np.mean(np.abs(ranges[both] - old[both]))) if both.sum() > 15 else np.inf
            if change < SLIP_SCAN_CHANGE_M:
                if slip_pose_ref is None:
                    slip_pose_ref = pose
                slip_time += dt
            else:
                slip_time = 0.0; slip_pose_ref = None
        else:
            slip_time = 0.0; slip_pose_ref = None
        stalled = progress_since is not None and t - progress_since > 8.0
        if slip_time > SLIP_CONFIRM_S or stalled:
            slip_events += 1
            if not stalled and slip_pose_ref is not None:
                estimator.odom.set_pose(slip_pose_ref); pose = slip_pose_ref
            ox = pose[0] + OBSTACLE_AHEAD_M * math.cos(pose[2]); oy = pose[1] + OBSTACLE_AHEAD_M * math.sin(pose[2])
            planner.mark_obstacle(ox, oy, 0.15)
            motion.remember_obstacle(ox, oy, 0.15)
            key = None if goal is None else (round(goal[0], 1), round(goal[1], 1))
            slip_goal_count[key] = slip_goal_count.get(key, 0) + 1
            if goal is not None and slip_goal_count[key] >= 2:
                planner.give_up_goal(goal, t); print(f"[{t:.1f}s] 같은 목표에서 2번 걸림 → 목표 포기 {key}")
            print(f"[{t:.1f}s] !! 헛바퀴 감지 ({slip_events}회): 앞 {OBSTACLE_AHEAD_M} m 에 장애물 표시, {RECOVER_BACK_S} s 후진 후 재계획")
            recover_until = t + RECOVER_BACK_S; slip_time = 0.0; slip_pose_ref = None; path = None
            progress_since = None; progress_xy = None

        navigating = (goal is not None and state in ("EXPLORE", "SEARCH", "VISIT", "APPROACH", "RETURN")
                      and math.dist(pose[:2], goal) > ARRIVE_DIST and t >= recover_until and t >= spin_until)
        if watchdog.update(pose, t, navigating):
            # 1차 실행(21b4945): 전진 명령(v 0.19) 중 15 s 위치 불변 = LiDAR 아래 물체에 걸림. 헛바퀴 감지는 |w|<0.4 조건에
            # 걸려 못 잡았다. 걸린 자리를 지도에 찍지 않으면 같은 곳으로 다시 간다 → 앞 0.25 m 에 장애물 표시.
            blocked_ahead = abs(motion.v_cmd) > 0.05
            print(f"[{t:.1f}s] !! 국소 정체: 15초 위치 진행 없음 → 안전 후진·재계획" + (" (앞 0.25 m 장애물 표시)" if blocked_ahead else ""))
            if blocked_ahead:
                ox = pose[0] + OBSTACLE_AHEAD_M * math.cos(pose[2]); oy = pose[1] + OBSTACLE_AHEAD_M * math.sin(pose[2])
                planner.mark_obstacle(ox, oy, 0.15); motion.remember_obstacle(ox, oy, 0.15)
            if state in ("EXPLORE", "SEARCH"):
                planner.give_up_goal(goal, t); goal = None
            recover_until = t + RECOVER_BACK_S
            path = None
            watchdog.update(pose, t, False)

        # (2) 지도 갱신  [feat/exploration]  (새 스캔일 때만. LiDAR 100 ms 주기)
        if new_scan:
            if valid_scan:
                planner.update_map(pose, angles, ranges); scan_count += 1
                planner.mark_camera_seen(pose, angles, ranges)          # SEARCH 용: 카메라가 훑은 빈칸 기록
            last_scan = ranges
            last_scan_t = t

        # (3) 대상 탐지  [feat/vision]
        detector.process(camera.getImage(), pose, t)
        found_targets = detector.targets
        # 비대상 사과(초록·주황·보라) → LiDAR 가 못 보는 낮은 장애물로 지도·DWA 에 등록 (보라 사과 충돌 방지)
        while detector.obstacle_updates:
            _, ox_, oy_, orad = detector.obstacle_updates.pop(0); marked_obstacles += 1
            planner.mark_obstacle(ox_, oy_, orad); motion.remember_obstacle(ox_, oy_, orad)
            if detector.obstacle_ms is not None and marked_obstacles == 1:
                print(f"[{t:.1f}s] 장애물 색 3종 처리 비용 {detector.obstacle_ms:.1f} ms/프레임")

        # (4) 미션 상태 기계 + 경로 계획  [integration]  (스캔 3개 쌓인 뒤, 1 s 주기)
        if scan_count >= 3 and t >= recover_until and t - last_plan_t > REPLAN_PERIOD_S:
            last_plan_t = t
            if state == "EXPLORE" and visits.select(found_targets, pose, t) is not None:
                state = "VISIT"; print(f"[{t:.1f}s] 미방문 사과 ID={visits.active} → VISIT")
            if state == "EXPLORE":
                goal = planner.next_exploration_goal(pose, current_goal=goal)
                if goal is None and planner.blacklist:
                    # 포기한 목표들 때문에 후보가 없어진 것이면 한 번 비우고 다시 고른다 (기준선: 정체 8회 → 블랙리스트 8개 →
                    # 후보 0 → BLOCKED. 지도상 도달 가능한 frontier 는 21개 있었음).
                    n = planner.clear_blacklist(); print(f"[{t:.1f}s] 탐색 후보 없음 → 블랙리스트 {n}개 해제 후 재선택")
                    goal = planner.next_exploration_goal(pose, current_goal=None)
                if goal is None:
                    no_frontier_count += 1
                    # frontier 가 없어도 카메라가 안 본 빈칸이 남았으면 SEARCH 로 수색 (학습 코드의 SEARCH 단계).
                    # 3회 연속 없음이면 전이 (1회는 지도 갱신 타이밍 노이즈). 둘 다 없을 때만 BLOCKED.
                    if no_frontier_count >= 3:
                        sgoal = planner.next_search_goal(pose)
                        if sgoal is not None:
                            state = "SEARCH"; goal = sgoal; search_done = 0
                            print(f"[{t:.1f}s] frontier 없음 → 카메라 미확인 {planner.unseen_cells}칸 수색(SEARCH), 목표 ({goal[0]:.2f}, {goal[1]:.2f})")
                        elif no_frontier_count >= 30 and not visits.retry_after:
                            state = "BLOCKED"; print(f"[{t:.1f}s] 미션 미완료: 방문 {len(visits.completed)}/{TARGET_COUNT}, frontier·미확인 구역 모두 없음")
                else:
                    no_frontier_count = 0
            if state == "SEARCH":
                if visits.select(found_targets, pose, t) is not None:
                    state = "VISIT"; print(f"[{t:.1f}s] 수색 중 미방문 사과 ID={visits.active} → VISIT"); goal = None
                else:
                    goal = planner.next_search_goal(pose, current_goal=goal)
                    if goal is None or math.dist(pose[:2], goal) < 0.3:
                        # 수색 목표 도달/소진 → frontier 가 다시 생겼을 수 있으니 EXPLORE 로 (거기서 없으면 다시 SEARCH)
                        state = "EXPLORE"; goal = None; no_frontier_count = 0
                        print(f"[{t:.1f}s] 수색 목표 {'소진' if goal is None else '도달'} (미확인 {planner.unseen_cells}칸) → EXPLORE 재확인")
            target_goal = None
            if state == "VISIT":
                if visits.active is None:
                    visits.select(found_targets, pose, t)
                if visits.active is not None:
                    target_goal = found_targets[visits.active]
                    goal = target_goal
                    if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < CONFIRM_DIST:
                        print(f"[{t:.1f}s] 대상 {CONFIRM_DIST} m 안 도착 ({goal[0]:.2f}, {goal[1]:.2f}) → 확인 단계(CONFIRM)")
                        state = "CONFIRM"; confirm_target = goal; confirm_until = t + CONFIRM_HOLD_S
                        confirm_spin_until = -1e9; confirm_seen = False
                        goal = None; target_goal = None; visit_fail = 0
                else:
                    state = "RETURN" if visits.complete else "EXPLORE"
                    goal = None
            if state == "CONFIRM":
                # 도착 후: 사과를 바라보고 정지 → 카메라로 재확인 → (못 보면 제자리 회전) → 복귀
                if confirm_spin_until < 0 and t >= confirm_until:
                    if confirm_seen:
                        print(f"[{t:.1f}s] ✓ 구조 대상 확인 완료 ({confirm_target[0]:.2f}, {confirm_target[1]:.2f}) → 최종 접근")
                        state = "APPROACH"
                    else:
                        print(f"[{t:.1f}s] 정지 중 대상 미확인 → 제자리 회전으로 재탐색 {CONFIRM_SPIN_S:.0f} s")
                        confirm_spin_until = t + CONFIRM_SPIN_S
                elif confirm_spin_until > 0 and (t >= confirm_spin_until or confirm_seen):
                    if confirm_seen:
                        state = "APPROACH"
                    else:
                        print(f"[{t:.1f}s] 재확인 실패 → 방문 미인정, 탐색 후 재시도")
                        visits.defer(t); state = "EXPLORE"
                goal = None
            if state == "APPROACH":
                # 재확인 때 더 가까이서 본 관측으로 detector 가 좌표를 갱신했을 수 있으니 최신 좌표 사용
                near = found_targets[visits.active]
                confirm_target = near
                goal = near
            if state == "RETURN":
                goal = (x0, y0)
                if visits.complete and math.hypot(pose[0] - x0, pose[1] - y0) < HOME_DIST:
                    state = "DONE"; goal = None; print(f"[{t:.1f}s] 복귀 완료: 빨간 사과 {len(visits.completed)}개 방문 성공")
            goal = clip_to_map(goal)
            if (state in ("EXPLORE", "SEARCH") and goal is not None and prev_goal is not None
                    and math.dist(goal, prev_goal) > 0.5 and t - last_spin > LOOKAROUND_COOLDOWN_S
                    and t >= recover_until and planner.unseen_near(pose, 2.0) >= LOOKAROUND_MIN_UNSEEN):
                spin_until = t + 2 * math.pi / LOOKAROUND_W; last_spin = t; lookarounds += 1
                print(f"[{t:.1f}s] 목표 변경 → 제자리 둘러보기 {2 * math.pi / LOOKAROUND_W:.1f} s (주변 미확인 {planner.unseen_near(pose, 2.0)}칸)")
            prev_goal = goal
            path = planner.plan(pose, goal) if goal is not None else None
            if goal is not None and path is None:
                if state == "VISIT" and target_goal is not None:
                    visit_fail += 1
                    if visit_fail >= 3:                      # 3번 연속 경로 실패 → 그 대상 포기 (무한 회전 방지)
                        visits.defer(t); visit_fail = 0; state = "EXPLORE"; goal = None; path = None
                        print(f"[{t:.1f}s] 대상 경로 3회 실패 → 방문 미인정, 탐색 후 재시도")
                elif state == "RETURN":
                    return_fail += 1
                    if return_fail >= 5:                     # 5번 연속 실패 → 미탐색 칸도 지나가도록 재시도
                        path = plan_path(planner.get_inflated(), planner.spec, (pose[0], pose[1]), goal,
                                         unknown_passable=True, cost_map=planner._cost)
                        if path is None and return_fail == 5:
                            print(f"[{t:.1f}s] 복귀 경로 없음 (미탐색 통과 허용해도) → 정지")
                else:
                    planner.give_up_goal(goal)
            else:
                if state == "VISIT": visit_fail = 0
                if state == "RETURN": return_fail = 0
            if planner.grid.dropped_hits > 100 and t - last_map_warn >= 10.0:
                last_map_warn = t
                print(f"[{t:.1f}s] !! 지도 밖 측정값 {planner.grid.dropped_hits}개 → exploration.MAP_HALF_M 확인")

        # 도착 판정은 A*의 1초 주기와 분리한다. 0.22m/s에서 최대 22cm 더 진입하는 것을 방지.
        if state == "APPROACH" and visits.active is not None:
            target = found_targets[visits.active]
            distance = math.dist(pose[:2], target)
            if visits.finish(distance, confirm_seen):
                state = "RETURN" if visits.complete else "EXPLORE"
                print(f"[{t:.1f}s] ★ 방문 성공 {len(visits.completed)}/{TARGET_COUNT}: ({target[0]:.2f}, {target[1]:.2f}) 거리 {distance:.2f} m → {state}")
                # 이미 관측한 낮은 사과도 이후 궤적에서 장애물로 기억한다.
                # 사과 반경뿐 아니라 위치 추정 오차도 포함해 출발 시 재접근을 막는다.
                motion.remember_obstacle(*target, radius=0.18)
                planner.mark_obstacle(target[0], target[1], 0.20)      # 방문한 사과도 낮은 장애물: 이후 경로에서 회피 (고정 표시)
                goal = None; path = None; no_frontier_count = 0

        # (5) 속도 명령  [feat/dwa]   (환경변수 GRIDNAV_HOLD=초 를 주면 그동안 정지: 비전 테스트용)
        motion.params.v_min = -0.15 if back_is_safe(pose) else 0.0     # DWA 후진 후보도 뒤가 안전할 때만
        v, w = motion.command(pose, path, ranges, angles, state)
        if state == "APPROACH" and path:
            # 사과는 흔히 벽에서 0.1 m 에 있어 DWA 여유(정적 0.205 / 동적 0.305 m) 안이다 → 최종 0.35 m 접근은
            # pure pursuit + 정면 정지로. (story 실행: DWA 로 0.8 m 에서 350 s 진동, 오프라인 재현 3.8 s 도달)
            v, w = lookahead_control(pose, path, v=APPROACH_V)
            front = ranges[np.abs(angles) <= math.radians(30)]
            front = front[np.isfinite(front) & (front > 0.05)]
            if front.size and float(front.min()) < ROBOT_RADIUS + 0.05:
                v = 0.0                                                          # 벽 코앞: 회전만
        if state in ("EXPLORE", "SEARCH") and t < spin_until:
            v, w = 0.0, LOOKAROUND_W                                              # 둘러보기: 제자리 회전 (카메라로 주변 훑기)
        if state == "CONFIRM":
            if confirm_spin_until > 0 and t < confirm_spin_until:
                v, w = 0.0, 1.0                                                   # 못 봤으면 제자리 회전
            else:                                                                 # 사과 방향으로 몸 돌린 뒤 정지
                err = wrap_angle(math.atan2(confirm_target[1] - pose[1], confirm_target[0] - pose[0]) - pose[2])
                v, w = (0.0, float(np.clip(2.0 * err, -1.2, 1.2))) if abs(err) > math.radians(5) else (0.0, 0.0)
            if not confirm_seen and confirms_active_target(detector, pose, confirm_target):
                confirm_seen = True
                bgr = vision.webots_image_to_bgr(camera.getImage(), detector.cam_w, detector.cam_h)
                vision.save_debug_frame(vision.draw_detection(bgr, detector.last_det, "confirmed"),
                                        os.path.join(debug_dir or ".", "confirmed.jpg"))
                print(f"[{t:.1f}s] 카메라로 대상 재확인 (r={detector.last_det[2]:.0f}px) → confirmed.jpg 저장")
        if t < recover_until:                        # 복구 중: 후진 (뒤가 '본 곳/지나온 곳' 일 때만, 아니면 제자리 회전)
            if back_is_safe(pose):
                v, w = -0.10, 0.0
            else:
                v, w = 0.0, 0.8; unsafe_backs += 1
        if state == "EXPLORE" and goal is None and no_frontier_count > 0:
            v, w = 0.0, 0.5
        if state in ("DONE", "BLOCKED"):
            v, w = 0.0, 0.0
        if t < HOLD_SECONDS:
            v, w = 0.0, 0.0
        # 복구 후진/확인 회전도 동일한 LiDAR 안전 검사를 거친다.
        v, w = motion.guard_command(v, w, ranges, angles)
        motion.set_applied_velocity(v, w)
        wl, wr = to_wheel_speeds(v, w)
        left_motor.setVelocity(wl); right_motor.setVelocity(wr)

        # (6) 로그 / 지도 저장 / Display  [integration]
        if contact_node is not None:
            contacts = [cp for cp in contact_node.getContactPoints(True) if cp.point[2] > 0.02]
            touching = bool(contacts)
            if touching and not in_contact:
                contact_events += 1
                print(f"[{t:.1f}s] !! 비바닥접촉={contact_events} (진단용, z>0.02m) "
                      f"높이={[round(cp.point[2], 3) for cp in contacts]} 노드={[cp.node_id for cp in contacts]}")
            in_contact = touching
        if map_save_path and t - last_save >= map_save_every_s:
            last_save = t
            tern, spec = planner.get_map()
            gt = ground_truth() if ground_truth else None
            np.savez(map_save_path.replace(".npy", ".npz"), ternary=tern,
                     origin=np.array([spec.origin_x, spec.origin_y, spec.resolution]),
                     pose=np.array(pose), gt=np.array(gt if gt else [np.nan] * 3),
                     goal=np.array(goal if goal else [np.nan, np.nan]),
                     path=np.array(path if path else np.zeros((0, 2))),
                     targets=np.array(found_targets if found_targets else np.zeros((0, 2))), state=np.array(state))
        if display is not None and t - last_disp >= 1.0:
            last_disp = t
            draw_map_on_display(display, disp_w, disp_h, planner, pose, goal, path, found_targets, state)
        if t - last_log >= 1.0:
            last_log = t
            gt = ground_truth() if ground_truth else None
            err = f" 위치오차={math.hypot(pose[0] - gt[0], pose[1] - gt[1]):.2f}m" if gt else ""
            print(f"[{t:6.1f}s] {state:8s} pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}°){err} "
                  f"목표={None if goal is None else (round(goal[0], 2), round(goal[1], 2))} 경로점={0 if not path else len(path)} "
                  f"v={v:.2f} w={w:+.2f} 대상={len(found_targets)} 방문={len(visits.completed)}/{TARGET_COUNT} 헛바퀴={slip_events} 미확인칸={planner.unseen_cells} 둘러보기={lookarounds} DWA복구={motion.recoveries} 비상정지={motion.emergency_stops} 사과장애물={len(detector.obstacles)} 후진거부={unsafe_backs} 정적/동적점={len(motion._cached_obstacles[0])}/{len(motion._cached_obstacles[1])} 보정={estimator.corrections} 비바닥접촉={contact_events if contact_node else '미계측'} 비전거부={detector.rejected} 지도밖hit={planner.grid.dropped_hits}")


if __name__ == "__main__":
    main()
