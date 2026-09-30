"""
motion_control.py - 경로를 바퀴 속도로 바꾸는 국소 제어 (브랜치 feat/dwa 소유)

지금: 강의 4장 Look-ahead(pure pursuit) + 정면 비상정지.  다음: MotionController.command() 를 dwa.dwa_control 로 교체.

인터페이스 (webots_adapter.run 이 부름):
    mc = MotionController()
    v, w = mc.command(pose, path, ranges, angles, state)   # path 가 None 이거나 state=="DONE" 이면 (0, 0)
    wl, wr = to_wheel_speeds(v, w)
"""
from __future__ import annotations

import math

import numpy as np

from geometry import wrap_angle
from robot_config import ROBOT_RADIUS, WHEEL_RADIUS, WHEEL_SEPARATION, MAX_WHEEL_SPEED, LIDAR_FRONT_INDEX

LOOKAHEAD = 0.25            # [m] 길면 커브 안쪽 가로지름, 짧으면 지그재그
CRUISE_V = 0.15             # [m/s] 순항 속도 (TB3 최대 0.22)
STOP_DIST = ROBOT_RADIUS + 0.10   # 정면 이 거리 안에 무언가 있으면 전진 금지


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
    if m > MAX_WHEEL_SPEED:
        wl, wr = wl * MAX_WHEEL_SPEED / m, wr * MAX_WHEEL_SPEED / m
    return wl, wr


class MotionController:
    def __init__(self):
        self.v_cmd = self.w_cmd = 0.0          # DWA 의 dynamic window 기준이 될 직전 명령
        self.emergency_stops = 0

    def command(self, pose, path, ranges, angles, state):
        if state == "DONE" or not path:
            v, w = 0.0, 0.0
        else:
            # --- feat/dwa: 여기를 dwa.dwa_control(pose, self.v_cmd, self.w_cmd, goal, obstacles, params) 로 교체 ---
            v, w = lookahead_control(pose, path)
            front = ranges[LIDAR_FRONT_INDEX - 30:LIDAR_FRONT_INDEX + 31]       # 정면 ±30°
            if np.nanmin(np.where(np.isfinite(front), front, np.inf)) < STOP_DIST:
                v = 0.0
                self.emergency_stops += 1
        self.v_cmd, self.w_cmd = v, w
        return v, w
