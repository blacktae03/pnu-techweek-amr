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
CAM_HEIGHT = 0.088          # [m] 바닥에서 카메라 광축까지. [확인됨 2026-09-30] 시험 사과(거리 1.45/1.53 m) 두 프레임에서 역산 (0.091, 0.085)
CAM_YAW_OFFSET = 0.0        # [rad] 카메라 광축이 로봇 정면과 이루는 각. [확인됨] 0.5 m 옆 사과의 bearing 오차 0.5° → 0

# --- 컴퍼스 [확인됨 2026-09-30 breakroom 에서 ground truth 대조] ---
COMPASS_SIGN = -1.0         # atan2(c[1], c[0]) 는 시계방향 + → -1 을 곱해 반시계 + 규약으로

# --- 미션/대상 [당일 규칙 확인] ---
APPLE_DIAMETER = 0.085      # [m] 색 분할로 잡히는 '겉보기' 지름. [확인됨] PROTO 메시는 반지름 0.05(지름 0.10)지만 그늘진 가장자리가
                            #     마스크에서 빠져 1.45 m 에서 r=16.5px, 1.53 m 에서 r=14.4px → 유효 지름 0.086/0.080
APPLE_CENTER_Z = 0.05       # [m] 사과 중심 높이 (translation z=0.05)
START_POSE_APARTMENT = (-0.3, -7.5, math.pi)
