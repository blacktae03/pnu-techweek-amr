"""
pose_estimation.py - 로봇 위치 추정 (브랜치 feat/localization 소유)

바퀴 엔코더 odometry + 컴퍼스 절대 heading + 선택적 CorrelativeMatcher 위치 보정.
webots_adapter.run(..., use_scan_matching=True)로 활성화한다. 기본 OFF는 유지한다.
예측 → 보정 → 지도 갱신 순서이며, 인자로 받은 지도는 이전 스캔까지의 지도여야 한다.

인터페이스 (webots_adapter.run 이 부름):
    est = PoseEstimator(start_pose)
    pose = est.update(phi_l, phi_r, compass_values, gyro_z=None, ranges=None, angles=None, log_odds=None)
    → (x, y, theta) [m, rad], theta 는 반시계 +
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from geometry import wrap_angle
from robot_config import WHEEL_RADIUS, WHEEL_SEPARATION, COMPASS_SIGN


class WheelOdometry:
    """엔코더(PositionSensor, 라디안 누적) 차이로 pose 적분. 실전에서는 scan matching 보정과 함께 사용.

    수식 (differential drive):
        dl = Δphi_l * r,  dr = Δphi_r * r,  ds = (dl + dr)/2,  dθ = (dr - dl)/L
        x += ds cos(θ + dθ/2),  y += ds sin(θ + dθ/2),  θ = wrap(θ + dθ)
    회전 오차가 지도 품질에 가장 치명적이므로 θ 는 가능하면 바퀴가 아니라 컴퍼스/자이로로 잡는다.
    """

    def __init__(self, x0, y0, theta0):
        self.x, self.y, self.theta = float(x0), float(y0), float(theta0)
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

    def set_pose(self, pose):
        self.x, self.y, self.theta = float(pose[0]), float(pose[1]), float(pose[2])

    @property
    def pose(self):
        return (self.x, self.y, self.theta)


class CompassHeading:
    """컴퍼스 벡터 → 절대 heading. 시작 시 주어진 theta0 로 캘리브레이션하므로 월드 북쪽이 어디든 상관없음.
        raw = atan2(c[1], c[0]),   theta = wrap(theta0 + COMPASS_SIGN * (raw - raw0))
    COMPASS_SIGN = -1 [확인됨]: 로봇이 왼쪽(반시계)으로 돌 때 raw 는 감소한다."""

    def __init__(self, theta0):
        self.theta0 = theta0
        self.raw0 = None

    def update(self, compass_values):
        raw = math.atan2(compass_values[1], compass_values[0])
        if self.raw0 is None:
            self.raw0 = raw
        return wrap_angle(self.theta0 + COMPASS_SIGN * wrap_angle(raw - self.raw0))


class PoseEstimator:
    """odometry 예측 (+ 선택: LiDAR scan matching 보정)."""

    def __init__(self, start_pose, use_scan_matching=False, spec=None):
        self.odom = WheelOdometry(*start_pose)
        self.heading = CompassHeading(start_pose[2])
        self.matcher = None
        if use_scan_matching and spec is not None:
            from localization import CorrelativeMatcher
            # 보정 후 컴퍼스 각도를 복원하는 것만으로는 충분하지 않다.
            # 후보 탐색 자체도 같은 각도로 해야 회전 오차를 x/y 이동으로 보상하지 않는다.
            self.matcher = CorrelativeMatcher(spec, search_th=0.0)
        self.corrections = 0

    def update(self, phi_l, phi_r, compass_values, gyro_z=None, dt=None,
               ranges=None, angles=None, log_odds=None):
        theta = self.heading.update(compass_values)
        pose = self.odom.update(phi_l, phi_r, theta_external=theta, gyro_z=gyro_z, dt=dt)
        # 지도 갱신은 adapter가 이 호출 이후에 수행한다. 여기서는 읽기만 한다.
        if (self.matcher is not None and ranges is not None
                and angles is not None and log_odds is not None):
            corrected = self.matcher.correct(pose, angles, ranges, log_odds)
            if corrected[:2] != pose[:2]:
                self.corrections += 1
                # 위치만 보정하고 heading 은 컴퍼스를 믿는다 (컴퍼스가 절대각이라 더 정확)
                self.odom.set_pose((corrected[0], corrected[1], pose[2]))
                pose = self.odom.pose
        return pose

    @property
    def pose(self):
        return self.odom.pose
