"""
robot_config.py - 로봇/센서 상수 한곳 (통합 담당 소유. 당일 규칙·로봇이 바뀔 때만 수정)

강의 레포 controllers/, worlds/apartment.wbt, 노트북 5장에서 확인한 값. 근거는 grid_nav/README.md 표 참고.
"""
import math

# --- TurtleBot3 Burger [확인됨] ---
WHEEL_RADIUS = 0.033        # [m]
WHEEL_SEPARATION = 0.160    # [m]
ROBOT_RADIUS = 0.105        # [m]
MAX_WHEEL_SPEED = 6.67      # [rad/s] 모터 한계 (≈0.22 m/s). 강의 teleop 은 3.0 사용
V_MAX = 0.22                # [m/s]
W_MAX = 2.84                # [rad/s]

# --- LDS-01 LiDAR [확인됨] ---
LIDAR_NAME = "LDS-01"
LIDAR_PERIOD_MS = 100       # 강의 코드와 동일. lidar.enable(100) → 100 ms 마다 새 스캔
LIDAR_MAX_RANGE = 3.5       # 실행 시 lidar.getMaxRange() 로 덮어씀
LIDAR_FRONT_INDEX = 180     # ranges[180]=정면, [0]=후방, [90]=왼쪽, [270]=오른쪽

# --- 카메라 [확인됨: apartment.wbt] ---
CAM_FOV = 1.0472            # [rad] 60°
CAM_W, CAM_H = 640, 480
CAM_HEIGHT = 0.11           # [m] 바닥에서 카메라 중심까지 대략 (TB3 상판 0.19 - 0.08). 비전 거리 일관성 검사에 사용

# --- 컴퍼스 [확인됨 2026-09-30 breakroom 에서 ground truth 대조] ---
COMPASS_SIGN = -1.0         # atan2(c[1], c[0]) 는 시계방향 + → -1 을 곱해 반시계 + 규약으로

# --- 미션/대상 [당일 규칙 확인] ---
APPLE_DIAMETER = 0.05       # [m] 사과 PROTO "0.05 x 0.05 x 0.05"
START_POSE_APARTMENT = (-0.3, -7.5, math.pi)
