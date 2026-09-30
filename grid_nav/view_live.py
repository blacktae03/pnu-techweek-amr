"""
view_live.py - Webots 컨트롤러(tb3_gridnav)가 저장하는 카메라 프레임(cam.jpg)과 지도(map.npz)를 나란히 실시간으로 보여준다.

    python grid_nav/view_live.py PNU-TECHWEEK-260930/controllers/tb3_gridnav

왼쪽: 카메라 (파란 원 = 탐지된 사과)   오른쪽: 지도 (회색 미탐색 / 흰 빈칸 / 검정 벽), 파란 원 = 추정 위치,
초록 x = 진짜 위치, 빨간 별 = 목표, 자주색 = A* 경로, 노란 삼각 = 발견한 사과
"""
import os
import sys
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from geometry import setup_korean_font

d = sys.argv[1] if len(sys.argv) > 1 else "."
map_path, cam_path = os.path.join(d, "map.npz"), os.path.join(d, "cam.jpg")
setup_korean_font()
plt.ion()
fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [4, 5]})
fig.canvas.manager.set_window_title("grid_nav live: 카메라 + 지도")
while plt.fignum_exists(fig.number):
    # --- 카메라 ---
    try:
        img = plt.imread(cam_path)
        ax_c.clear(); ax_c.imshow(img[:, :, ::-1] if img.ndim == 3 else img); ax_c.set_title("카메라 (파란 원 = 탐지)"); ax_c.axis("off")
    except Exception:
        ax_c.set_title("카메라 프레임 대기 중...")
    # --- 지도 ---
    try:
        z = np.load(map_path, allow_pickle=False)
        tern = z["ternary"]; ox, oy, res = z["origin"]
        extent = [ox, ox + tern.shape[1] * res, oy, oy + tern.shape[0] * res]
        rgb = np.full(tern.shape + (3,), 0.6); rgb[tern == 0] = 1.0; rgb[tern == 1] = 0.0
        ax_m.clear(); ax_m.imshow(rgb, origin="lower", extent=extent)
        pose, gt, goal, pth, tg = z["pose"], z["gt"], z["goal"], z["path"], z["targets"]
        ax_m.plot(pose[0], pose[1], "bo", ms=8, label="추정 위치")
        ax_m.arrow(pose[0], pose[1], 0.4 * np.cos(pose[2]), 0.4 * np.sin(pose[2]), head_width=0.12, color="b")
        if np.isfinite(gt[0]):
            ax_m.plot(gt[0], gt[1], "gx", ms=10, mew=2, label=f"진짜 위치 (오차 {np.hypot(pose[0] - gt[0], pose[1] - gt[1]):.2f} m)")
        if np.isfinite(goal[0]):
            ax_m.plot(goal[0], goal[1], "r*", ms=14, label="목표")
        if len(pth):
            ax_m.plot(pth[:, 0], pth[:, 1], "m.-", lw=1.5, ms=3, label="A* 경로")
        if len(tg):
            ax_m.plot(tg[:, 0], tg[:, 1], "y^", ms=12, label="발견한 사과")
        known = tern != -1
        if known.any():
            r_, c_ = np.nonzero(known)
            ax_m.set_xlim(ox + c_.min() * res - 1, ox + c_.max() * res + 1)
            ax_m.set_ylim(oy + r_.min() * res - 1, oy + r_.max() * res + 1)
        ax_m.set_title(f"상태 {z['state']}  |  지도 채움 {known.mean() * 100:.1f}%")
        ax_m.legend(loc="upper right", fontsize=8)
    except Exception as e:
        ax_m.set_title(f"지도 대기 중... ({type(e).__name__})")
    plt.pause(0.5)
