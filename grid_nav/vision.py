"""
vision.py - 카메라 이미지에서 색 사과 찾기 (강의 레포 tb3_teleop_cam.py 방식 + 거리/방향 추정)

파이프라인 (슬라이드 31~37쪽): 색 임계값 → 이진 마스크 → 모폴로지(노이즈 제거) → 가장 큰 contour
                             → 최소 외접원(반지름 = 크기) + 무게중심(위치)
색공간: HSV. 빨강은 Hue 가 0 과 180 양 끝에 걸쳐 있어서 범위를 두 개로 나눠 OR 한다.
        (강의 레포는 초록에 LAB 을 썼다. 어느 쪽이든 임계값은 실제 화면을 보고 튠해야 한다 → save_debug_frame)

출력: (cx_px, cy_px, radius_px) 또는 None.  cx 는 왼쪽 0 → 오른쪽 W.
      webots_adapter.target_world_from_pixel(pose, cx, radius) 가 이를 월드 좌표로 바꾼다.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # OpenCV 없으면 NumPy 만으로 대충 (성능·정확도 낮음)
    cv2 = None

# HSV 임계값 (OpenCV 규약: H 0~180, S 0~255, V 0~255). Webots RedApple 기준으로 튠할 것.
COLOR_RANGES = {
    # RedApple.proto 는 baseColor 1 0 0 (순수 빨강, 텍스처 없음) → 채도가 매우 높다. 나무 바닥(붉은 갈색, S 100~150)과
    # 구분하려면 S 하한을 160 으로. 어두운 쪽(그늘)도 잡도록 V 하한은 50.
    "red":    [((0, 160, 50), (6, 255, 255)), ((174, 160, 50), (180, 255, 255))],
    "green":  [((35, 80, 50), (85, 255, 255))],
    "orange": [((9, 120, 80), (22, 255, 255))],
    "purple": [((125, 60, 40), (160, 255, 255))],
}


def webots_image_to_bgr(image_bytes, width, height):
    """camera.getImage() (BGRA 바이트열) → (H, W, 3) BGR uint8 배열."""
    arr = np.frombuffer(image_bytes, np.uint8).reshape((height, width, 4))
    return arr[:, :, :3].copy()          # BGRA 에서 A 만 버리면 BGR


def color_mask(bgr, color="red"):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in COLOR_RANGES[color]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else (mask | m)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)      # 작은 점 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)     # 구멍 메우기
    return mask


def detect_apple(bgr, color="red", min_radius_px=3.0, max_radius_px=60.0, min_circularity=0.45):
    """가장 큰 색 덩어리의 (cx, cy, r) [px]. 없거나 조건에 안 맞으면 None.
    max_radius_px : 사과(지름 5 cm)는 0.3 m 앞에서도 반지름 ≈ 46 px. 그보다 크면 바닥·벽 같은 큰 면 → 무시.
    min_circularity: contour 면적 / 외접원 면적. 사과는 둥글어 0.6~0.8, 바닥 얼룩·가구 모서리는 낮다."""
    if cv2 is None:
        return None
    mask = color_mask(bgr, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    if r < min_radius_px or r > max_radius_px:
        return None
    if cv2.contourArea(c) / (np.pi * r * r + 1e-6) < min_circularity:
        return None
    M = cv2.moments(c)
    if M["m00"] > 0:                                       # 무게중심이 외접원 중심보다 안정적
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    return float(cx), float(cy), float(r)


def draw_detection(bgr, det, label=""):
    out = bgr.copy()
    if det is not None:
        cx, cy, r = det
        cv2.circle(out, (int(cx), int(cy)), int(max(r, 2)), (255, 0, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (255, 0, 0), -1)
        cv2.putText(out, f"{label} r={r:.0f}px", (int(cx) + 8, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return out


def save_debug_frame(bgr, path):
    if cv2 is not None:
        cv2.imwrite(path, bgr)


if __name__ == "__main__":
    # 합성 이미지로 자체 테스트: 회색 배경에 빨간 원 하나
    img = np.full((480, 640, 3), 120, np.uint8)
    cv2.circle(img, (400, 300), 12, (30, 30, 200), -1)      # BGR 빨강
    det = detect_apple(img, "red")
    print("탐지:", det)
    assert det is not None and abs(det[0] - 400) < 2 and abs(det[2] - 12) < 2
    print("OK")
