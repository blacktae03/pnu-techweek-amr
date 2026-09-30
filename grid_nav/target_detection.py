"""
target_detection.py - 카메라로 구조 대상(색 사과) 찾고 월드 좌표로 기록 (브랜치 feat/vision 소유)

인터페이스 (webots_adapter.run 이 부름):
    det = TargetDetector(cam_w, cam_h, cam_fov, debug_dir)
    new_xy = det.process(camera.getImage(), pose, t)     # 새 대상이 확정되면 (x, y), 아니면 None
    det.targets                                          # 지금까지 확정된 대상 [(x, y), ...]

알려진 문제 (feat/vision 의 첫 과제):
  - 크기(반지름)만으로 거리를 재면 "멀리 있는 큰 빨간 것" 을 "가까운 사과" 로 오인 → 위치가 로봇을 따라 밀림.
    해결안: ① 사과는 바닥에 있으니 화면 수평선 아래(cy > H/2) 조건  ② 반지름 기반 거리와 세로 위치 기반 거리
    (d = f * CAM_HEIGHT / (cy - H/2)) 가 비슷해야 채택  ③ 같은 물체 재탐지는 새로 추가하지 말고 위치 갱신(병합 반경 1.0 m)
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import vision
from robot_config import CAM_W, CAM_H, CAM_FOV, CAM_HEIGHT, APPLE_DIAMETER, APPLE_CENTER_Z

TARGET_COLOR = "red"        # 찾을 사과 색 (vision.COLOR_RANGES 키). 당일 규칙 확인
VISION_EVERY = 3            # 몇 스텝마다 카메라 처리 (640x480 색 분할 ≈ 5 ms)
MIN_APPLE_RADIUS_PX = 3.0   # 이보다 작은 덩어리는 무시 (사과 5 cm → 2.5 m 에서 약 5.5 px)
MAX_DETECT_DIST = 3.0       # 추정 거리가 이보다 멀면 무시 (거리 추정이 부정확해짐)
MERGE_RADIUS = 1.0          # 이 반경 안의 재탐지는 같은 대상으로 보고 위치를 갱신 (거리 오차가 크므로 넉넉히)
HORIZON_MARGIN_PX = 6       # 사과 중심은 화면 수평선(H/2) 보다 이만큼 아래여야 함 (바닥 물체 조건)
CONSISTENCY_RATIO = 2.0     # 반지름 거리 / 세로위치 거리 비율이 [1/R, R] 안이어야 채택
CONFIRM_FRAMES = 2          # 연속 프레임 수 이상 보여야 확정 (한 프레임 노이즈 배제)
# 비(非)대상 사과 = 장애물. 사과(지름 10 cm, 중심 5 cm)는 LiDAR 평면(17 cm) 아래라 지도에 안 찍혀 로봇이 치고 지나간다
# (B조 실행: 보라 사과 충돌). 같은 색·원형도 검사로 찾아 지도에 장애물로 등록한다. 빨간 사과 판정 로직은 건드리지 않는다.
OBSTACLE_COLORS = ("green", "orange", "purple")
OBSTACLE_EVERY = 1          # 대상 색 처리(VISION_EVERY) 마다 장애물 색도 처리 (3색 ≈ 3 ms)
OBSTACLE_CONFIRM_FRAMES = 1 # 비대상 사과는 1프레임 즉시 등록 (오등록은 통행 비용만 늘 뿐, 놓치면 충돌)
OBSTACLE_MIN_RADIUS = 0.20  # 등록 반경 하한
OBSTACLE_MERGE_RADIUS = 0.8 # 이 반경 안의 재탐지는 같은 장애물 (더 가까이서 보면 위치 갱신; 1차 실행 1.7 m 추정이 0.54 m 어긋남)
OBSTACLE_MAX_DIST = 2.0     # 이보다 먼 추정은 장애물 등록 안 함. 멀수록 반경을 키워(obstacle_radius) 오차를 덮는다
                            # (1차 실행: 2.2~2.4 m 추정이 진실값과 0.85 m, 1.7 m 는 0.54 m, 1.4 m 는 0.2 m, 0.45 m 는 0.02 m 어긋남)


def target_world_from_pixel(pose, cx_px, radius_px, cam_w=CAM_W, fov=CAM_FOV,
                            object_diameter=APPLE_DIAMETER):
    """색 분할로 얻은 (중심 x 픽셀, 외접원 반지름 픽셀) → 대상의 월드 (x, y), bearing, dist.

    방향: 핀홀 모델. 초점거리 f = (W/2) / tan(fov/2) [px].  bearing = -atan((cx - W/2) / f)
          (화면 오른쪽 = 로봇 오른쪽 = 음의 각도)
    거리: 사과 지름 5 cm 를 알고 있으므로  dist = f * D / (2 * radius_px)
          LDS-01 은 바닥에서 약 17 cm 높이에 있어 5 cm 사과를 못 봅니다 → LiDAR 거리로 대체 불가.
    """
    f = (cam_w / 2.0) / math.tan(fov / 2.0)
    bearing = -math.atan((cx_px - cam_w / 2.0) / f)
    dist = f * object_diameter / max(2.0 * radius_px, 1.0)
    dist += 0.05                                  # 카메라가 로봇 중심보다 5 cm 앞에 있음 (대략 보정)
    x = pose[0] + dist * math.cos(pose[2] + bearing)
    y = pose[1] + dist * math.sin(pose[2] + bearing)
    return (x, y), bearing, dist


def distance_from_row(cy_px, cam_h=CAM_H, cam_w=CAM_W, fov=CAM_FOV, cam_height=CAM_HEIGHT, object_z=APPLE_CENTER_Z):
    """화면 세로 위치로 잰 거리 (두 번째 독립 추정).
    카메라가 수평이면 바닥 물체(높이 object_z) 의 중심은 수평선(H/2) 아래  f * (cam_height - object_z) / d  픽셀에 맺힌다.
    → d = f * (cam_height - object_z) / (cy - H/2).  수평선 위(cy <= H/2)면 바닥 물체가 아니다 → inf."""
    f = (cam_w / 2.0) / math.tan(fov / 2.0)
    dy = cy_px - cam_h / 2.0
    if dy <= 0:
        return float("inf")
    return f * (cam_height - object_z) / dy


class TargetDetector:
    """탐지 채택 규칙 (모두 만족해야 대상으로 기록):
      ① 색·크기·원형도 (vision.detect_apple)
      ② 화면 수평선 아래 (바닥에 있는 물체)                          → 탁자 위 캔, 표지판 배제
      ③ 반지름 기반 거리 ≈ 세로위치 기반 거리 (비율 1/R ~ R)          → 멀리 있는 큰 빨간 것 배제
      ④ 연속 CONFIRM_FRAMES 프레임 이상                              → 한 프레임 노이즈 배제
    같은 대상 재탐지(MERGE_RADIUS 안)는 새로 추가하지 않고 더 가까이서 본 추정으로 위치를 갱신한다."""

    def __init__(self, cam_w, cam_h, cam_fov, debug_dir=None, color=TARGET_COLOR):
        self.cam_w, self.cam_h, self.cam_fov = cam_w, cam_h, cam_fov
        self.debug_dir = debug_dir
        self.color = color
        self.targets: List[Tuple[float, float]] = []
        self._best_dist: List[float] = []          # 각 대상을 가장 가까이서 본 거리 (갱신 판단용)
        self.step_i = 0
        self.last_cam_save = -1e9
        self.last_det = None
        self.consecutive = 0
        self.rejected = {"horizon": 0, "inconsistent": 0, "far": 0}
        self.n_snapshots = 0
        self.obstacles: List[Tuple[float, float, str]] = []   # 확정된 비대상 사과 (x, y, color)
        self._obs_dist: List[float] = []                       # 각 장애물을 가장 가까이서 본 거리
        self.obstacle_updates: List[Tuple[int, float, float, float]] = []   # (idx, x, y, radius) 어댑터가 지도에 찍을 큐
        self._obs_consec = {c: 0 for c in OBSTACLE_COLORS}
        self._obs_frames = 0
        self.obstacle_ms = None                                # 장애물 색 처리 1회 비용 [ms] (진단)

    def process(self, image_bytes, pose, t) -> Optional[Tuple[float, float]]:
        self.step_i += 1
        if self.step_i % VISION_EVERY != 0:
            return None
        bgr = vision.webots_image_to_bgr(image_bytes, self.cam_w, self.cam_h)
        det = vision.detect_apple(bgr, self.color, MIN_APPLE_RADIUS_PX)
        self.last_det = det
        new_target = None
        accepted = False
        if det is not None:
            cx, cy, r = det
            (tx, ty), bearing, d_r = target_world_from_pixel(pose, cx, r, self.cam_w, self.cam_fov)
            d_row = distance_from_row(cy, self.cam_h, self.cam_w, self.cam_fov)
            if cy < self.cam_h / 2 + HORIZON_MARGIN_PX:
                self.rejected["horizon"] += 1
            elif not (1.0 / CONSISTENCY_RATIO < d_r / d_row < CONSISTENCY_RATIO):
                self.rejected["inconsistent"] += 1
            elif d_r > MAX_DETECT_DIST:
                self.rejected["far"] += 1
            else:
                accepted = True
                self.consecutive += 1
                if self.consecutive >= CONFIRM_FRAMES:
                    # 두 추정의 기하평균이 한쪽 편향을 줄인다
                    dist = math.sqrt(d_r * d_row)
                    tx = pose[0] + dist * math.cos(pose[2] + bearing)
                    ty = pose[1] + dist * math.sin(pose[2] + bearing)
                    idx = next((i for i, f in enumerate(self.targets) if math.hypot(tx - f[0], ty - f[1]) < MERGE_RADIUS), None)
                    if idx is None:
                        self.targets.append((tx, ty)); self._best_dist.append(dist)
                        new_target = (tx, ty)
                        print(f"[{t:.1f}s] ★ {self.color} 사과 발견: 화면 ({cx:.0f},{cy:.0f}) r={r:.1f}px → 방향 {math.degrees(bearing):.0f}°, "
                              f"거리 반지름기준 {d_r:.2f} / 세로기준 {d_row:.2f} → {dist:.2f} m, 월드 ({tx:.2f}, {ty:.2f})  (지금까지 {len(self.targets)}개)")
                        self._snapshot(bgr, det)
                    elif dist < self._best_dist[idx]:        # 더 가까이서 봤으면 위치 갱신
                        self.targets[idx] = (tx, ty); self._best_dist[idx] = dist
        if not accepted:
            self.consecutive = 0
        self._obs_frames += 1
        if self._obs_frames % OBSTACLE_EVERY == 0:
            self._detect_obstacle_apples(bgr, pose, t)
        if self.debug_dir and t - self.last_cam_save >= 0.5:
            self.last_cam_save = t
            vision.save_debug_frame(vision.draw_detection(bgr, det, self.color), os.path.join(self.debug_dir, "cam.jpg"))
        return new_target

    def _snapshot(self, bgr, det):
        if self.debug_dir and self.n_snapshots < 20:
            self.n_snapshots += 1
            vision.save_debug_frame(vision.draw_detection(bgr, det, self.color),
                                    os.path.join(self.debug_dir, f"det_{self.n_snapshots}.jpg"))

    @staticmethod
    def obstacle_radius(dist):
        """지도에 찍을 장애물 반경: 사과 반지름 0.05 + 여유 0.10 + 거리 비례 위치 불확실성(1차 실행 실측 ≈ 0.3·거리)."""
        return max(OBSTACLE_MIN_RADIUS, 0.15 + 0.25 * max(0.0, dist - 0.5))

    def _detect_obstacle_apples(self, bgr, pose, t):
        """빨강이 아닌 사과(초록·주황·보라) 를 같은 기하 규칙(수평선 아래, 두 거리 추정 일치, 2프레임) 으로 찾아
        self.obstacles 에 기록. 어댑터가 이를 지도 장애물(mark_obstacle)로 찍는다. 대상 색 판정과는 완전히 분리."""
        import time as _time
        t0 = _time.perf_counter()
        for color in OBSTACLE_COLORS:
            det = vision.detect_apple(bgr, color, MIN_APPLE_RADIUS_PX)
            ok = False
            if det is not None:
                cx, cy, r = det
                (_, _), bearing, d_r = target_world_from_pixel(pose, cx, r, self.cam_w, self.cam_fov)
                d_row = distance_from_row(cy, self.cam_h, self.cam_w, self.cam_fov)
                if (cy >= self.cam_h / 2 + HORIZON_MARGIN_PX and 1.0 / CONSISTENCY_RATIO < d_r / d_row < CONSISTENCY_RATIO
                        and d_r <= OBSTACLE_MAX_DIST):
                    ok = True
                    self._obs_consec[color] += 1
                    if self._obs_consec[color] >= OBSTACLE_CONFIRM_FRAMES:
                        dist = math.sqrt(d_r * d_row)
                        x = pose[0] + dist * math.cos(pose[2] + bearing); y = pose[1] + dist * math.sin(pose[2] + bearing)
                        if all(math.hypot(x - tx, y - ty) >= OBSTACLE_MERGE_RADIUS for tx, ty in self.targets):
                            idx = next((i for i, (ox, oy, _) in enumerate(self.obstacles)
                                        if math.hypot(x - ox, y - oy) < OBSTACLE_MERGE_RADIUS), None)
                            if idx is None:
                                self.obstacles.append((x, y, color)); self._obs_dist.append(dist); idx = len(self.obstacles) - 1
                                print(f"[{t:.1f}s] ○ {color} 사과(장애물) 확정: r={r:.1f}px 거리 {dist:.2f} m → 월드 ({x:.2f}, {y:.2f}) "
                                      f"반경 {self.obstacle_radius(dist):.2f} m  (장애물 {len(self.obstacles)}개)")
                                self.obstacle_updates.append((idx, x, y, self.obstacle_radius(dist)))
                            elif dist < self._obs_dist[idx] - 0.2 and dist <= 2.0:   # 더 가까이서 다시 봄 → 위치 갱신
                                self.obstacles[idx] = (x, y, color); self._obs_dist[idx] = dist
                                self.obstacle_updates.append((idx, x, y, self.obstacle_radius(dist)))
            if not ok:
                self._obs_consec[color] = 0
        self.obstacle_ms = (_time.perf_counter() - t0) * 1000.0

