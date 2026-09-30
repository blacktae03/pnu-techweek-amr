"""
webots_adapter.py - Webots(TurtleBot3 Burger) 컨트롤러에 격자 파트를 끼우는 뼈대

강의 레포(PNU-TECHWEEK-260930)의 controllers/, worlds/ 를 읽고 확인한 사실은 [확인됨] 으로,
당일 다시 봐야 하는 것은 [확인필요] 로 표시했습니다.

[확인됨] 로봇: TurtleBot3Burger (Webots R2025a 기본 PROTO)
         WHEEL_RADIUS 0.033, WHEEL_SEPARATION 0.160, ROBOT_RADIUS 0.105  (강의 노트북 5장 '핵심 파라미터')
[확인됨] 디바이스 이름 (controllers/tb3_teleop_sensors.py):
         "LDS-01"(LiDAR), "camera", "gyro", "compass", "accelerometer", "display",
         "left wheel motor", "right wheel motor",  엔코더는 motor.getPositionSensor()
[확인됨] LiDAR 인덱스 ↔ 방향 (controllers/tb3_lidar.py 의 print):
         ranges[180] = 정면, ranges[0] = 후방, ranges[90] = 왼쪽, ranges[270] = 오른쪽, 총 360개
         → angle_i = pi - 2*pi*i/360  (i=180 → 0 rad 정면, i=90 → +pi/2 왼쪽)   ※ lidar_angles() 참고
[확인됨] 강의 코드는 lidar.enable(100): LiDAR 는 100 ms 마다 갱신. basicTimeStep 64 ms.
         → 같은 스캔이 두 스텝 연속 들어올 수 있으니 '새 스캔일 때만' 지도 업데이트 (아래 루프 참고)
[확인됨] 카메라: fieldOfView 1.0472 rad(60°), 640x480, 로봇 기준 (0.05, 0, -0.08) 에 장착
[확인됨] apartment.wbt: 시작 translation (-0.3, -7.5), rotation z 3.14159 → 시작 pose (-0.3, -7.5, pi)
         구조 대상 = 색 사과(Red/Green/Purple/Orange Apple, 지름 5cm, 바닥 z=0.05), 7개
         사과 위치 x∈[-12.0,-2.8], y∈[-11.7,-1.3]  → 시작점 기준 최대 약 13 m → MAP_HALF_M ≥ 15
         Pedestrian 1명이 --speed=0.2 로 정해진 궤적을 순환 (움직이는 사람 = 동적 장애물)
[확인필요] LDS-01 최대 사거리(실제 제품 3.5 m; Webots PROTO 값은 lidar.getMaxRange() 로 읽기)
[확인됨] 컴퍼스 heading 부호: raw = atan2(compass[1], compass[0]) 는 시계방향이 양수라 COMPASS_SIGN = -1 (breakroom 월드에서
         Supervisor ground truth 와 대조해 확인). CompassHeading 은 시작 시 캘리브레이션하므로 절대 오프셋은 무관.
[확인필요] 대회용 월드가 apartment.wbt 인지, 시작 pose 가 같은지.

실행: Webots 안에서 controller 로 지정. 터미널에서 직접 실행하면 controller 모듈이 없어 뼈대 점검만 합니다.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from geometry import GridSpec, wrap_angle
from occupancy_grid import OccupancyGrid, inflate
from frontier import frontier_mask, cluster_frontiers, select_frontier
from astar import plan_path, build_cost_map
import vision

try:
    from controller import Robot          # Webots 가 제공. Webots 밖에서는 없음
    _IN_WEBOTS = True
except ImportError:
    Robot = object                        # type: ignore
    _IN_WEBOTS = False


# ======================================================================
# 설정값
# ======================================================================
WHEEL_RADIUS = 0.033        # [확인됨] TB3 Burger
WHEEL_SEPARATION = 0.160    # [확인됨]
ROBOT_RADIUS = 0.105        # [확인됨]
SAFETY_MARGIN = 0.10        # 팽창 여유. 문/가구 사이가 좁으면 0.06 까지 줄여 볼 것
MAP_HALF_M = 15.0           # 시작점 기준 ±15 m (apartment 기준). 작으면 dropped_hits 경고
RESOLUTION = 0.05           # 600x600 격자. A* 가 느리면 0.075 로
LIDAR_MAX_RANGE = 3.5       # [확인필요] 실행 시 lidar.getMaxRange() 로 덮어씀
LIDAR_PERIOD_MS = 100       # 강의 코드와 동일. lidar.enable(100)
REPLAN_PERIOD_S = 1.0
CAM_FOV = 1.0472            # [확인됨] 60°
CAM_W, CAM_H = 640, 480     # [확인됨]
APPLE_DIAMETER = 0.05       # [확인됨] 사과 PROTO 주석 "0.05 x 0.05 x 0.05 m"
COMPASS_SIGN = -1.0         # [확인됨 2026-09-30] breakroom 월드에서 ground truth 와 비교: atan2(c[1], c[0]) 는 시계방향이 +
                            # (로봇이 왼쪽으로 돌 때 진짜 θ 는 증가, raw 는 감소) → -1 을 곱해야 반시계 + 규약과 맞음
MAX_WHEEL_SPEED = 6.67      # TB3 Burger 모터 한계 [rad/s] (0.22 m/s). 강의 teleop 은 3.0 사용
TARGET_COLOR = "red"        # 찾을 사과 색 (vision.COLOR_RANGES 키)
TARGET_COUNT = 1            # 이 개수를 찾으면 탐색을 멈추고 방문 → 복귀. 규칙에 따라 조정
VISION_EVERY = 3            # 몇 스텝마다 카메라 처리 (640x480 색 분할 ≈ 5 ms)
MIN_APPLE_RADIUS_PX = 3.0   # 이보다 작은 덩어리는 무시 (사과 5 cm → 2.5 m 에서 약 5.5 px)


# ======================================================================
# 1. 격자 파트 묶음: 팀원이 부를 API 는 이 클래스 하나
# ======================================================================
class GridPlanner:
    """지도 작성 + frontier + A* 를 한 객체로.
         update_map(pose, angles, ranges)          pose=(x,y,theta) [m, rad]
         get_map()   -> (int8 2D 0/1/-1, GridSpec)
         next_exploration_goal(pose) -> (x, y) 또는 None(탐색 완료)
         plan(pose, goal_xy) -> [(x, y), ...] 또는 None
         give_up_goal(goal_xy)
    """

    def __init__(self, start_xy=(0.0, 0.0), max_range=LIDAR_MAX_RANGE):
        self.spec = GridSpec.from_size(2 * MAP_HALF_M, 2 * MAP_HALF_M, RESOLUTION, center=start_xy)
        self.grid = OccupancyGrid(self.spec, max_range=max_range)
        self.inflate_r = ROBOT_RADIUS + SAFETY_MARGIN
        self.blacklist: List[Tuple[float, float]] = []
        self._ternary = None
        self._inflated = None
        self._cost = None

    def update_map(self, pose, angles, ranges):
        self.grid.update(pose, angles, ranges)
        self._ternary = None                       # 캐시 무효화

    def _refresh(self):
        if self._ternary is None:
            self._ternary = self.grid.to_ternary()
            self._inflated = inflate(self._ternary, self.inflate_r, self.spec.resolution)
            self._cost = build_cost_map(self._inflated, self.spec.resolution, safe_dist=0.4, w_prox=3.0)

    def get_map(self):
        self._refresh()
        return self._ternary, self.spec

    def get_inflated(self):
        self._refresh()
        return self._inflated

    def next_exploration_goal(self, pose, current_goal=None, keep_radius=0.5) -> Optional[Tuple[float, float]]:
        """다음 탐색 목표. current_goal 을 주면 '목표 유지' 규칙을 적용한다:
        현재 목표 반경 keep_radius 안에 아직 frontier 칸이 남아 있으면 그대로 둔다.
        왜: 매초 점수 1등이 바뀌면 로봇이 두 목표 사이를 왔다갔다 한다(진동). 목표는 끝까지 가거나 사라질 때만 바꾼다."""
        self._refresh()
        mask = frontier_mask(self._ternary, self._inflated)
        if current_goal is not None:
            r, c = self.spec.world_to_grid(current_goal[0], current_goal[1])
            k = int(keep_radius / self.spec.resolution)
            r0, r1 = max(0, r - k), min(self.spec.rows, r + k + 1)
            c0, c1 = max(0, c - k), min(self.spec.cols, c + k + 1)
            still_frontier = mask[r0:r1, c0:c1].any()
            far_enough = np.hypot(current_goal[0] - pose[0], current_goal[1] - pose[1]) > 0.3
            not_blacklisted = all(np.hypot(current_goal[0] - b[0], current_goal[1] - b[1]) > 0.4 for b in self.blacklist)
            if still_frontier and far_enough and not_blacklisted:
                return tuple(current_goal)
        frontiers = cluster_frontiers(mask, min_size=6)
        f = select_frontier(frontiers, pose, self.spec, blacklist_xy=self.blacklist)
        return f.goal_xy(self.spec) if f else None

    def plan(self, pose, goal_xy):
        self._refresh()
        return plan_path(self._inflated, self.spec, (pose[0], pose[1]), goal_xy, cost_map=self._cost,
                         snap_radius_cells=int(0.6 / self.spec.resolution))

    def give_up_goal(self, goal_xy):
        self.blacklist.append(tuple(goal_xy))


# ======================================================================
# 2. LiDAR 각도 배열  [확인됨: LDS-01 인덱스 매핑]
# ======================================================================
def lidar_angles(n_points=360):
    """ranges[i] 에 대응하는 각도 (로봇 정면 = 0, 왼쪽 = +, 라디안).

    강의 코드 tb3_lidar.py 가 출력하는 대응:
        ranges[180] Front, ranges[0] Back, ranges[90] Left, ranges[270] Right
    이를 만족하는 식:  angle_i = pi - 2*pi * i / N
        i=0   → pi      (후방)      i=90  → +pi/2 (왼쪽)
        i=180 → 0       (정면)      i=270 → -pi/2 (오른쪽)
    즉 인덱스가 커질수록 시계 방향으로 돕니다. 지도가 좌우 반전으로 그려지면 이 부호를 의심할 것.
    """
    i = np.arange(n_points)
    return wrap_angle(np.pi - 2.0 * np.pi * i / n_points)


def read_lidar(lidar):
    """range image → float 배열. 못 본 빔은 inf (occupancy_grid 가 처리)."""
    return np.array(lidar.getRangeImage(), dtype=float)


# ======================================================================
# 3. 임시 위치 추정 (scan matching 팀원 모듈이 오기 전까지)
# ======================================================================
class WheelOdometry:
    """엔코더(누적 회전각 rad) 차이로 위치 적분 + 자이로/컴퍼스로 방향 보정.

      dl = Δphi_l * r,  dr = Δphi_r * r,  ds = (dl+dr)/2,  dθ = (dr-dl)/L
      x += ds cos(θ + dθ/2),  y += ds sin(θ + dθ/2),  θ = wrap(θ + dθ)

    방향(θ) 오차가 지도 품질에 가장 치명적이므로 θ 는 가능하면 바퀴가 아니라
    (1) 컴퍼스 절대각(CompassHeading) 또는 (2) 자이로 z 적분으로 잡습니다.
    """

    def __init__(self, x0, y0, theta0):
        self.x, self.y, self.theta = x0, y0, theta0
        self.prev = None

    def update(self, phi_l, phi_r, theta_external=None, gyro_z=None, dt=None):
        if self.prev is None:
            self.prev = (phi_l, phi_r)
            return self.pose
        dl = (phi_l - self.prev[0]) * WHEEL_RADIUS
        dr = (phi_r - self.prev[1]) * WHEEL_RADIUS
        self.prev = (phi_l, phi_r)
        ds = 0.5 * (dl + dr)
        if theta_external is not None:            # 컴퍼스 등 절대 heading 이 있으면 그것을 사용
            new_theta = theta_external
            dtheta = wrap_angle(new_theta - self.theta)
        elif gyro_z is not None and dt is not None:
            dtheta = gyro_z * dt
            new_theta = wrap_angle(self.theta + dtheta)
        else:
            dtheta = (dr - dl) / WHEEL_SEPARATION
            new_theta = wrap_angle(self.theta + dtheta)
        self.x += ds * math.cos(self.theta + 0.5 * dtheta)
        self.y += ds * math.sin(self.theta + 0.5 * dtheta)
        self.theta = new_theta
        return self.pose

    @property
    def pose(self):
        return (self.x, self.y, self.theta)


class CompassHeading:
    """컴퍼스 벡터 → 절대 heading. 시작 시 주어진 theta0 로 캘리브레이션하므로 월드 북쪽이 어디든 상관없음.
        raw = atan2(c[1], c[0])   (강의 코드와 동일)
        theta = wrap(theta0 + COMPASS_SIGN * (raw - raw0))
    [확인필요] COMPASS_SIGN: 로봇을 왼쪽(반시계)으로 돌렸을 때 theta 가 증가해야 함. 아니면 -1.
    """

    def __init__(self, theta0):
        self.theta0 = theta0
        self.raw0 = None

    def update(self, compass_values):
        raw = math.atan2(compass_values[1], compass_values[0])
        if self.raw0 is None:
            self.raw0 = raw
        return wrap_angle(self.theta0 + COMPASS_SIGN * wrap_angle(raw - self.raw0))


# ======================================================================
# 4. 카메라 픽셀 → 대상 월드 좌표 (비전 팀원 결과를 받는 자리)
# ======================================================================
def target_world_from_pixel(pose, cx_px, radius_px, cam_w=CAM_W, fov=CAM_FOV,
                            object_diameter=APPLE_DIAMETER):
    """색 분할로 얻은 (중심 x 픽셀, 외접원 반지름 픽셀) → 대상의 월드 (x, y).

    방향: 핀홀 모델. 초점거리 f = (W/2) / tan(fov/2) [px].  bearing = -atan((cx - W/2) / f)
          (화면 오른쪽 = 로봇 오른쪽 = 음의 각도)
    거리: 사과 지름 5 cm 를 알고 있으므로  dist = f * D / (2 * radius_px)
          LDS-01 은 바닥에서 약 17 cm 높이에 있어 5 cm 사과를 못 봅니다 → LiDAR 거리로 대체 불가.
          반지름 픽셀은 노이즈가 크므로 여러 프레임 중앙값을 쓰거나, 가까이 갈수록 다시 계산하는 게 안전.
    """
    f = (cam_w / 2.0) / math.tan(fov / 2.0)
    bearing = -math.atan((cx_px - cam_w / 2.0) / f)
    dist = f * object_diameter / max(2.0 * radius_px, 1.0)
    dist += 0.05                                  # 카메라가 로봇 중심보다 5 cm 앞에 있음 (대략 보정)
    x = pose[0] + dist * math.cos(pose[2] + bearing)
    y = pose[1] + dist * math.sin(pose[2] + bearing)
    return (x, y), bearing, dist


# ======================================================================
# 5. 경로 추종: 강의 4장 Look-ahead(pure pursuit) 공식  (DWA 팀원 모듈이 오기 전 임시)
# ======================================================================
def draw_map_on_display(display, W, H, planner, pose, goal, path, targets, state):
    """지도(0/1/-1) 의 '알려진 영역' 을 잘라 Display 크기에 맞게 확대하고 로봇/목표/경로를 그려 넣는다.
    Webots Display 는 y=0 이 위쪽이므로 세로를 뒤집는다 (지도는 row=0 이 아래)."""
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
    rgb = rgb[::-1]                                             # 세로 뒤집기 (위 = 큰 y)
    scale = min(W / rgb.shape[1], H / rgb.shape[0])
    ow, oh = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
    img = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((H, W, 3), 40, np.uint8)
    canvas[:oh, :ow] = img
    x0 = spec.origin_x + c0 * spec.resolution
    y1 = spec.origin_y + r1 * spec.resolution                   # 화면 위쪽 = y1

    def to_px(x, y):
        return int((x - x0) / spec.resolution * scale), int((y1 - y) / spec.resolution * scale)

    if path:
        pts = np.array([to_px(x, y) for x, y in path], np.int32)
        cv2.polylines(canvas, [pts], False, (255, 0, 255), 2)
    for tx, ty in targets:
        cv2.drawMarker(canvas, to_px(tx, ty), (0, 200, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)
    if goal is not None:
        cv2.drawMarker(canvas, to_px(goal[0], goal[1]), (0, 0, 255), cv2.MARKER_STAR, 16, 2)
    px, py = to_px(pose[0], pose[1])
    cv2.circle(canvas, (px, py), max(3, int(ROBOT_RADIUS / spec.resolution * scale)), (255, 80, 0), -1)
    cv2.line(canvas, (px, py), (int(px + 12 * math.cos(pose[2])), int(py - 12 * math.sin(pose[2]))), (0, 0, 0), 2)
    cv2.putText(canvas, state, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    im = display.imageNew(canvas.tobytes(), Display.RGB, W, H)
    display.imagePaste(im, 0, 0, False)
    display.imageDelete(im)


def lookahead_control(pose, path, lookahead=0.25, v=0.15):
    """κ = 2*y_LA / (x_LA² + y_LA²),  ω = v*κ.  y_LA, x_LA 는 로봇 좌표계에서 본 look-ahead 점.
    장애물 검사는 전혀 없음 → 반드시 DWA/비상정지와 함께 써야 함."""
    pts = np.asarray(path, dtype=float)
    d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
    i_near = int(np.argmin(d))
    # 경로 거리 기준으로 lookahead 만큼 앞의 점
    acc = 0.0
    i_la = i_near
    while i_la < len(pts) - 1 and acc < lookahead:
        acc += float(np.hypot(*(pts[i_la + 1] - pts[i_la])))
        i_la += 1
    dx, dy = pts[i_la] - pts[i_near] * 0 - np.array([pose[0], pose[1]])
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x_la = c * dx + s * dy                       # 로봇 좌표계로 회전 (world_to_robot 과 동일)
    y_la = -s * dx + c * dy
    if x_la < 0.05:                              # 목표가 뒤/옆에 있으면 제자리 회전
        return 0.0, 1.0 * math.copysign(1.0, y_la)
    kappa = 2.0 * y_la / (x_la ** 2 + y_la ** 2)
    return v, v * kappa


def to_wheel_speeds(v, w):
    """(선속도, 각속도) → (왼바퀴, 오른바퀴) 각속도 [rad/s], 모터 한계로 클립."""
    vl = v - w * WHEEL_SEPARATION / 2.0
    vr = v + w * WHEEL_SEPARATION / 2.0
    wl, wr = vl / WHEEL_RADIUS, vr / WHEEL_RADIUS
    m = max(abs(wl), abs(wr), 1e-9)
    if m > MAX_WHEEL_SPEED:                      # 비율 유지하며 축소 (회전 반경 유지)
        wl, wr = wl * MAX_WHEEL_SPEED / m, wr * MAX_WHEEL_SPEED / m
    return wl, wr


# ======================================================================
# 6. 컨트롤러 메인 루프
# ======================================================================
def main():
    if not _IN_WEBOTS:
        print("Webots 밖에서 실행됨: controller 모듈이 없어 뼈대 점검만 합니다. (import/문법 OK)")
        planner = GridPlanner(start_xy=(-0.3, -7.5))
        tern, spec = planner.get_map()
        ang = lidar_angles(360)
        print(f"  GridPlanner 준비: 지도 {tern.shape}, 해상도 {spec.resolution} m")
        print(f"  LiDAR 각도: idx0={math.degrees(ang[0]):.0f}°(후방) idx90={math.degrees(ang[90]):.0f}°(왼쪽) "
              f"idx180={math.degrees(ang[180]):.0f}°(정면) idx270={math.degrees(ang[270]):.0f}°(오른쪽)")
        (tx, ty), b, d = target_world_from_pixel((0, 0, 0), cx_px=400, radius_px=20)
        print(f"  픽셀(400, r=20) → bearing {math.degrees(b):.1f}°, dist {d:.2f} m, world ({tx:.2f}, {ty:.2f})")
        return

    run(Robot(), start_pose=(-0.3, -7.5, math.pi))


def run(robot, start_pose, ground_truth=None, map_save_path="map.npy", map_save_every_s=2.0):
    """실제 컨트롤러 본체. tb3_gridnav 컨트롤러가 이 함수를 부른다.
    robot        : Robot() 또는 Supervisor()
    start_pose   : 대회에서 주는 시작 (x, y, theta)
    ground_truth : 디버깅용. 호출하면 (x, y, theta) 진짜 pose 를 주는 함수 (없으면 None)
    map_save_path: 주기적으로 지도(int8)를 np.save → grid_nav/view_map.py 로 실시간 확인
    """
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    # --- 디바이스 [확인됨: 이름] -------------------------------------------------
    lidar = robot.getDevice("LDS-01")
    lidar.enable(LIDAR_PERIOD_MS)
    lidar.enablePointCloud()
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    for m in (left_motor, right_motor):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)
    left_enc = left_motor.getPositionSensor();   left_enc.enable(timestep)
    right_enc = right_motor.getPositionSensor(); right_enc.enable(timestep)
    gyro = robot.getDevice("gyro");       gyro.enable(timestep)
    compass = robot.getDevice("compass"); compass.enable(timestep)
    # 지도 표시용 Display (dev 월드에만 있음. 없으면 None)
    try:
        display = robot.getDevice("map_display")
        disp_w, disp_h = display.getWidth(), display.getHeight()
    except Exception:
        display = None
    camera = robot.getDevice("camera"); camera.enable(timestep)
    cam_w, cam_h = camera.getWidth(), camera.getHeight()
    cam_fov = camera.getFov()
    cam_dir = os.path.dirname(os.path.abspath(map_save_path)) if map_save_path else "."

    # --- 시작 pose (대회 제공값. apartment.wbt 는 (-0.3, -7.5, pi)) ------------------
    x0, y0, theta0 = start_pose
    heading = CompassHeading(theta0)
    odom = WheelOdometry(x0, y0, theta0)
    lidar_max = lidar.getMaxRange()
    planner = GridPlanner(start_xy=(x0, y0), max_range=min(lidar_max, LIDAR_MAX_RANGE))
    angles = None

    path: Optional[List[Tuple[float, float]]] = None
    goal: Optional[Tuple[float, float]] = None
    last_plan_t = -1e9
    last_scan = None
    state = "EXPLORE"                   # EXPLORE → VISIT → RETURN → DONE (sim_demo.Mission 참고)
    found_targets: List[Tuple[float, float]] = []
    visited: List[Tuple[float, float]] = []

    last_save = -1e9
    last_log = -1e9
    step_i = 0
    last_disp = -1e9
    last_cam_save = -1e9
    scan_count = 0            # 지금까지 지도에 넣은 스캔 수. 몇 개 쌓이기 전엔 계획하지 않는다
    no_frontier_count = 0     # "frontier 없음" 이 연속으로 나온 횟수 (한 번의 우연으로 탐색을 끝내지 않기 위해)
    while robot.step(timestep) != -1:
        t = robot.getTime()

        # (1) pose  ★ 실전: scan matching 팀원 모듈의 pose 로 교체
        theta = heading.update(compass.getValues())
        pose = odom.update(left_enc.getValue(), right_enc.getValue(), theta_external=theta)

        # (2) LiDAR → 지도.  100 ms 마다만 새 스캔이 오므로 바뀐 스캔에만 업데이트
        ranges = read_lidar(lidar)
        if angles is None:
            angles = lidar_angles(len(ranges))
        if last_scan is None or not np.array_equal(ranges, last_scan):
            if np.isfinite(ranges).any() and np.nanmax(np.where(np.isfinite(ranges), ranges, 0)) > 0.0:
                planner.update_map(pose, angles, ranges)
                scan_count += 1
            last_scan = ranges

        # (3) 비전: 빨간 사과 → 월드 좌표 기록 (0.6 m 안에 이미 기록된 것은 같은 사과로 봄)
        step_i += 1
        if step_i % VISION_EVERY == 0:
            bgr = vision.webots_image_to_bgr(camera.getImage(), cam_w, cam_h)
            det = vision.detect_apple(bgr, TARGET_COLOR, MIN_APPLE_RADIUS_PX)
            if det is not None:
                (tx, ty), bearing, dist = target_world_from_pixel(pose, det[0], det[2], cam_w, cam_fov)
                if dist < 3.0 and all(math.hypot(tx - f[0], ty - f[1]) > 0.6 for f in found_targets):
                    found_targets.append((tx, ty))
                    print(f"[{t:.1f}s] ★ {TARGET_COLOR} 사과 발견: 화면 x={det[0]:.0f} r={det[2]:.1f}px → 방향 {math.degrees(bearing):.0f}°, "
                          f"거리 {dist:.2f} m, 월드 ({tx:.2f}, {ty:.2f})   (지금까지 {len(found_targets)}개)")
            if map_save_path and t - last_cam_save >= 0.5:
                last_cam_save = t
                vision.save_debug_frame(vision.draw_detection(bgr, det, TARGET_COLOR), os.path.join(cam_dir, "cam.jpg"))

        # (4) 주기적 재계획 (상태 기계).  ★ 스캔이 3개는 쌓인 뒤에 시작 (LiDAR 는 100 ms 주기라 첫 스텝엔 지도가 비어 있음)
        if scan_count >= 3 and (path is None or t - last_plan_t > REPLAN_PERIOD_S):
            last_plan_t = t
            if state == "EXPLORE" and len(found_targets) >= TARGET_COUNT:
                state = "VISIT"
                print(f"[{t:.1f}s] 대상 {len(found_targets)}개 확보 → 탐색 중단, VISIT")
            if state == "EXPLORE":
                goal = planner.next_exploration_goal(pose, current_goal=goal)
                if goal is None:
                    no_frontier_count += 1
                    if no_frontier_count >= 3:          # 3번 연속 없어야 진짜 완료 (지도 갱신 타이밍 우연 방지)
                        state = "VISIT" if found_targets else "RETURN"
                        print(f"[{t:.1f}s] 탐색 완료 → {state}")
                else:
                    no_frontier_count = 0
            if state == "VISIT":
                remaining = [tg for tg in found_targets if tg not in visited]
                if remaining:
                    goal = min(remaining, key=lambda tg: math.hypot(tg[0] - pose[0], tg[1] - pose[1]))
                    if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < ROBOT_RADIUS + 0.3:
                        visited.append(goal); print(f"[{t:.1f}s] 대상 도착 {goal}")
                        goal = None
                else:
                    state = "RETURN"
            if state == "RETURN":
                goal = (x0, y0)
                if math.hypot(pose[0] - x0, pose[1] - y0) < 0.2:
                    state = "DONE"; goal = None; print(f"[{t:.1f}s] 복귀 완료")
            path = planner.plan(pose, goal) if goal is not None else None
            if goal is not None and path is None:
                planner.give_up_goal(goal)
            if planner.grid.dropped_hits > 100:
                print("!! 지도 밖 측정값 많음 → MAP_HALF_M 확인")

        # (5) 속도 명령  ★ 실전: v, w = dwa.compute(pose, ranges, angles, path)
        if state == "DONE" or path is None:
            v, w = 0.0, 0.0
        else:
            v, w = lookahead_control(pose, path)
            front = ranges[150:211]                      # 정면 ±30° (idx 180 중심)
            if np.nanmin(np.where(np.isfinite(front), front, np.inf)) < ROBOT_RADIUS + 0.10:
                v = 0.0                                  # 최소한의 비상정지
        wl, wr = to_wheel_speeds(v, w)
        left_motor.setVelocity(wl)
        right_motor.setVelocity(wr)

        # (6) 디버그: 지도/상태 저장 → 다른 터미널에서 `python grid_nav/view_map.py map.npy` 로 실시간 확인
        if map_save_path and t - last_save >= map_save_every_s:
            last_save = t
            tern, spec = planner.get_map()
            gt = ground_truth() if ground_truth else None
            np.savez(map_save_path.replace(".npy", ".npz"), ternary=tern, seen=np.zeros(1),
                     origin=np.array([spec.origin_x, spec.origin_y, spec.resolution]),
                     pose=np.array(pose), gt=np.array(gt if gt else [np.nan] * 3),
                     goal=np.array(goal if goal else [np.nan, np.nan]),
                     path=np.array(path if path else np.zeros((0, 2))),
                     targets=np.array(found_targets if found_targets else np.zeros((0, 2))),
                     state=np.array(state))
        if display is not None and t - last_disp >= 1.0:
            last_disp = t
            draw_map_on_display(display, disp_w, disp_h, planner, pose, goal, path, found_targets, state)
        if t - last_log >= 1.0:
            last_log = t
            gt = ground_truth() if ground_truth else None
            err = f" 위치오차={math.hypot(pose[0] - gt[0], pose[1] - gt[1]):.2f}m" if gt else ""
            print(f"[{t:6.1f}s] {state:8s} pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}°){err} "
                  f"목표={None if goal is None else (round(goal[0], 2), round(goal[1], 2))} 경로점={0 if not path else len(path)} "
                  f"v={v:.2f} w={w:+.2f} 지도밖hit={planner.grid.dropped_hits}")


if __name__ == "__main__":
    main()
