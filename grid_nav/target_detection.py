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
from robot_config import CAM_W, CAM_FOV, APPLE_DIAMETER

TARGET_COLOR = "red"        # 찾을 사과 색 (vision.COLOR_RANGES 키). 당일 규칙 확인
VISION_EVERY = 3            # 몇 스텝마다 카메라 처리 (640x480 색 분할 ≈ 5 ms)
MIN_APPLE_RADIUS_PX = 3.0   # 이보다 작은 덩어리는 무시 (사과 5 cm → 2.5 m 에서 약 5.5 px)
MAX_DETECT_DIST = 3.0       # 추정 거리가 이보다 멀면 무시 (거리 추정이 부정확해짐)
MERGE_RADIUS = 0.6          # 이 반경 안의 재탐지는 같은 대상으로 봄


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


class TargetDetector:
    def __init__(self, cam_w, cam_h, cam_fov, debug_dir=None, color=TARGET_COLOR):
        self.cam_w, self.cam_h, self.cam_fov = cam_w, cam_h, cam_fov
        self.debug_dir = debug_dir
        self.color = color
        self.targets: List[Tuple[float, float]] = []
        self.step_i = 0
        self.last_cam_save = -1e9
        self.last_det = None

    def process(self, image_bytes, pose, t) -> Optional[Tuple[float, float]]:
        self.step_i += 1
        if self.step_i % VISION_EVERY != 0:
            return None
        bgr = vision.webots_image_to_bgr(image_bytes, self.cam_w, self.cam_h)
        det = vision.detect_apple(bgr, self.color, MIN_APPLE_RADIUS_PX)
        self.last_det = det
        new_target = None
        if det is not None:
            (tx, ty), bearing, dist = target_world_from_pixel(pose, det[0], det[2], self.cam_w, self.cam_fov)
            if dist < MAX_DETECT_DIST and all(math.hypot(tx - f[0], ty - f[1]) > MERGE_RADIUS for f in self.targets):
                self.targets.append((tx, ty))
                new_target = (tx, ty)
                print(f"[{t:.1f}s] ★ {self.color} 사과 발견: 화면 x={det[0]:.0f} y={det[1]:.0f} r={det[2]:.1f}px → 방향 "
                      f"{math.degrees(bearing):.0f}°, 거리 {dist:.2f} m, 월드 ({tx:.2f}, {ty:.2f})   (지금까지 {len(self.targets)}개)")
        if self.debug_dir and t - self.last_cam_save >= 0.5:
            self.last_cam_save = t
            vision.save_debug_frame(vision.draw_detection(bgr, det, self.color), os.path.join(self.debug_dir, "cam.jpg"))
        return new_target
