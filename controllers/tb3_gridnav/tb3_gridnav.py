"""
tb3_gridnav.py - grid_nav 전체 스택(지도 → frontier → A* → look-ahead 추종)을 Webots 에서 돌리는 컨트롤러

- 위치: 엔코더 odometry + 컴퍼스 heading (webots_adapter.WheelOdometry/CompassHeading). scan matching 은 아직 미연결.
- 로봇 노드에 supervisor TRUE 가 있으면 시작 pose 를 ground truth 에서 읽고(대회에서 주는 값에 해당),
  이후 진짜 위치와의 오차를 로그로 찍는다 (디버깅 전용. 제출 코드에서는 Supervisor 제거).
- 지도는 2초마다 컨트롤러 폴더의 map.npz 로 저장 → 다른 터미널에서
      python grid_nav/view_map.py PNU-TECHWEEK-260930/controllers/tb3_gridnav/map.npz
"""
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# grid_nav 위치: 팀 레포 구조(<repo>/controllers/<이름>/ → <repo>/grid_nav) 또는 강의 레포 옆 폴더
for _cand in (os.path.join(_HERE, "..", "..", "grid_nav"), os.path.join(_HERE, "..", "..", "..", "grid_nav")):
    if os.path.isdir(_cand):
        GRID_NAV = os.path.normpath(_cand)
        break
else:
    raise ImportError("grid_nav 폴더를 찾을 수 없습니다: controllers/<이름>/ 기준 ../../grid_nav")
sys.path.insert(0, GRID_NAV)

from controller import Robot, Supervisor
import webots_adapter

try:
    robot = Supervisor()
    node = robot.getSelf()
except Exception:
    robot, node = Robot(), None


def ground_truth():
    p = node.getPosition(); o = node.getOrientation()
    return (p[0], p[1], math.atan2(o[3], o[0]))


if node is not None:
    robot.step(int(robot.getBasicTimeStep()))          # 한 스텝 진행해야 위치가 유효
    start_pose = ground_truth()
    print(f"[tb3_gridnav] 시작 pose (ground truth) = ({start_pose[0]:.2f}, {start_pose[1]:.2f}, {math.degrees(start_pose[2]):.0f}°)")
else:
    start_pose = (-0.3, -7.5, math.pi)                 # apartment.wbt 기본값
    print("[tb3_gridnav] Supervisor 없음 → 시작 pose 를 상수로 사용:", start_pose)

webots_adapter.run(robot, start_pose,
                   ground_truth=ground_truth if node is not None else None,
                   map_save_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.npy"))
