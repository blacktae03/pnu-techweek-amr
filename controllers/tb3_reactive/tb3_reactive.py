"""
tb3_reactive.py - 강의 Algorithm 3 (Yaw-based LiDAR Obstacle Avoidance + Corner Escape) Webots 컨트롤러

"여기서 시작하라" 는 강사님 조언에 맞춘 첫 자율주행 컨트롤러입니다. 지도/경로 없이 LiDAR 세 방향 거리와
컴퍼스 yaw 만으로 돌아다닙니다. 이게 돌면 센서 부호/모터 방향이 맞다는 뜻이고, 그 위에 grid_nav 의
지도 → frontier → A* → DWA 를 얹습니다 (webots_adapter.py 참고).

사용법:
  1) Webots 에서 월드를 열고 TurtleBot3Burger 의 controller 를 "tb3_reactive" 로 바꾼다
  2) 실행. 콘솔에 d_F/d_L/d_R, 상태(MOVE/TURN/ESCAPE) 가 1초마다 찍힌다.
  3) 확인할 것:
     - 로봇 앞에 상자를 두면 d_F 가 줄어야 한다 (아니면 LiDAR 인덱스 매핑이 틀림 → GRID_NAV/webots_adapter.lidar_angles)
     - 왼쪽에 상자를 두면 d_L 이 줄어야 한다 (반대면 각도 부호가 뒤집힘)
     - TURN 상태에서 로봇이 정한 방향으로 실제로 돌고 MOVE 로 돌아와야 한다 (아니면 COMPASS_SIGN 뒤집기)

이 파일은 grid_nav 폴더의 모듈을 import 합니다 (레포 옆 폴더). 경로가 다르면 GRID_NAV 를 고치세요.
"""
import math
import os
import sys

# grid_nav 모듈 경로: <이 레포>/../grid_nav
_HERE = os.path.dirname(os.path.abspath(__file__))
# grid_nav 위치: 팀 레포 구조(<repo>/controllers/<이름>/ → <repo>/grid_nav) 또는 강의 레포 옆 폴더
for _cand in (os.path.join(_HERE, "..", "..", "grid_nav"), os.path.join(_HERE, "..", "..", "..", "grid_nav")):
    if os.path.isdir(_cand):
        GRID_NAV = os.path.normpath(_cand)
        break
else:
    raise ImportError("grid_nav 폴더를 찾을 수 없습니다: controllers/<이름>/ 기준 ../../grid_nav")
sys.path.insert(0, GRID_NAV)

import numpy as np
from controller import Robot, Supervisor

from reactive import Algorithm3, ReactiveParams, lidar_sectors
from webots_adapter import lidar_angles, CompassHeading, COMPASS_SIGN, MAX_WHEEL_SPEED

# 월드의 로봇 노드에 supervisor TRUE 가 있으면 진짜 위치(ground truth)를 읽어 함께 출력한다 (디버깅용).
# 대회 제출 코드에서는 위치를 몰라야 하므로 Supervisor 를 쓰지 않는다.
try:
    robot = Supervisor()
    _self_node = robot.getSelf()
except Exception:
    robot = Robot()
    _self_node = None
timestep = int(robot.getBasicTimeStep())


def ground_truth_pose():
    """(x, y, theta) 진짜 값. 강의 tb3_ground_truth.py 와 같은 계산."""
    if _self_node is None:
        return None
    p = _self_node.getPosition()
    o = _self_node.getOrientation()
    return p[0], p[1], math.atan2(o[3], o[0])

# --- 디바이스 (강의 tb3_teleop_sensors.py 와 같은 이름) ---
lidar = robot.getDevice("LDS-01")
lidar.enable(100)
lidar.enablePointCloud()
compass = robot.getDevice("compass")
compass.enable(timestep)
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for m in (left_motor, right_motor):
    m.setPosition(float("inf"))
    m.setVelocity(0.0)

params = ReactiveParams(base_speed=3.0,                 # 강의 teleop 과 같은 바퀴 속도
                        distance_threshold=0.45,
                        angle=math.radians(60),
                        tolerance=math.radians(5),
                        max_cnt=4)
algo = Algorithm3(params)
heading = CompassHeading(theta0=0.0)                     # 절대각은 상관없음. 상대 회전량만 쓴다
angles = None
lidar_max = lidar.getMaxRange()
battery = 1.0                                            # Webots 에 배터리 없음 → 항상 충분
prev_print = 0.0

while robot.step(timestep) != -1:
    t = robot.getTime()
    ranges = np.array(lidar.getRangeImage(), dtype=float)
    if angles is None:
        angles = lidar_angles(len(ranges))               # idx180=정면, 90=왼쪽, 270=오른쪽
    yaw = heading.update(compass.getValues())

    d_F, d_L, d_R = lidar_sectors(ranges, angles, params, max_range=lidar_max)
    vL, vR = algo.step(d_F, d_L, d_R, yaw, battery)
    vL = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, vL))
    vR = max(-MAX_WHEEL_SPEED, min(MAX_WHEEL_SPEED, vR))
    left_motor.setVelocity(vL)
    right_motor.setVelocity(vR)

    if t - prev_print >= 1.0:
        gt = ground_truth_pose()
        gt_s = f"  진짜위치=({gt[0]:.2f}, {gt[1]:.2f}, {math.degrees(gt[2]):.0f}°)" if gt else ""
        print(f"[{t:6.1f}s] dF={d_F:.2f} dL={d_L:.2f} dR={d_R:.2f}  yaw={math.degrees(yaw):6.1f}°  "
              f"state={algo.state:6s} turn_cnt={algo.turn_cnt} escapes={algo.escapes}  vL={vL:+.1f} vR={vR:+.1f}{gt_s}")
        prev_print = t
