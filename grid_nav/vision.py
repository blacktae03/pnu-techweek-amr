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


def object_extent(mask, cx, cy, r):
    """탐지된 덩어리가 속한 '전체 빨간 물체' 의 (폭, 높이) [px].

    왜: 소화기처럼 라벨·손잡이로 빨간 부분이 여러 조각으로 갈라진 물체는 조각 하나만 보면 사과처럼 보인다.
        마스크를 세로로 부풀려(조각 사이 틈 ≈ 반지름 크기) 조각들을 붙인 뒤, 덩어리를 품는 연결 성분의 크기를 잰다.
        사과: 높이 ≈ 폭.   소화기·병·기둥: 높이 ≫ 폭."""
    k = max(3, int(r))
    kernel = np.ones((2 * k + 1, max(3, k // 2 * 2 + 1)), np.uint8)       # 세로로 긴 커널
    grown = cv2.dilate(mask, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    lab = labels[min(max(int(cy), 0), mask.shape[0] - 1), min(max(int(cx), 0), mask.shape[1] - 1)]
    if lab == 0:
        return 2 * r, 2 * r
    x, y, w, h = stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP], stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT]
    # 부풀린 만큼 되돌림
    return max(1.0, w - (k // 2 * 2)), max(1.0, h - 2 * k)


def red_outside_blob(mask, cx, cy, r, span=6.0):
    """덩어리 위아래 세로 띠(폭 = 지름, 높이 = ±span·r) 안에서 '덩어리 자신을 뺀' 같은 색 픽셀 수 / 덩어리 면적.

    왜: 소화기·병처럼 라벨로 갈라진 물체는 조각 사이 틈이 얼마든 위아래에 같은 색이 더 있다.
        사과는 위아래에 같은 색이 없다 (0 에 가까움). 연결 여부에 기대지 않아 틈 크기와 무관하게 동작한다."""
    H, W = mask.shape
    x0, x1 = int(max(0, cx - r)), int(min(W, cx + r + 1))
    y0, y1 = int(max(0, cy - span * r)), int(min(H, cy + span * r + 1))
    strip = mask[y0:y1, x0:x1] > 0
    by0, by1 = int(max(0, cy - r - 1)), int(min(H, cy + r + 2))
    inside = strip.copy(); inside[:] = False
    inside[max(0, by0 - y0):max(0, by1 - y0), :] = True
    outside_px = int((strip & ~inside).sum())
    return outside_px / max(1.0, np.pi * r * r)


def detect_apple(bgr, color="red", min_radius_px=3.0, max_radius_px=130.0, min_circularity=0.70,
                 min_aspect=0.65, max_aspect=1.5, max_outside_ratio=0.10):
    """가장 큰 색 덩어리의 (cx, cy, r) [px]. 없거나 조건에 안 맞으면 None.
    max_radius_px : 도착 거리 0.4 m(카메라→사과 0.35 m)에서 반지름 ≈ 67 px [7차 실행에서 60 으로 두어 CONFIRM 재확인 실패].
                    0.3 m 까지 접근해도 80 px 안. 그보다 크면 바닥·벽 같은 큰 면 → 무시.
    min_circularity: contour 면적 / 외접원 면적. [실측 2026-09-30 카탈로그] 사과 0.88~0.92, 누운 캔 0.50, 상자 스티커 0.45,
                     카펫 무늬 0.45, 소화기 조각 0.36~0.40 → 0.70 이 양쪽에 여유.
    min/max_aspect: 덩어리가 속한 전체 빨간 물체의 높이/폭. 사과 0.97~1.03, 누운 캔 0.47, 카펫 0.40, 소화기 2.1 → 0.65~1.5."""
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
    w_obj, h_obj = object_extent(mask, cx, cy, r)
    if not (min_aspect <= h_obj / w_obj <= max_aspect):
        return None
    if red_outside_blob(mask, cx, cy, r) > max_outside_ratio:       # 위아래에 같은 색이 더 있음 → 키 큰 물체의 조각
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
