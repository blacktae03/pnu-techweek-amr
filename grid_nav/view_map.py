"""
view_map.py - Webots 컨트롤러가 저장하는 map.npz 를 실시간으로 그려주는 뷰어 (별도 터미널에서 실행)

    python grid_nav/view_map.py PNU-TECHWEEK-260930/controllers/tb3_gridnav/map.npz

회색 미탐색 / 흰 빈칸 / 검정 벽, 파란 원 = 로봇이 믿는 위치, 초록 x = 진짜 위치(Supervisor 있을 때),
빨간 별 = 현재 목표, 자주색 선 = A* 경로, 노란 삼각 = 발견한 대상.
"""
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from geometry import setup_korean_font

path = sys.argv[1] if len(sys.argv) > 1 else "map.npz"
setup_korean_font()
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
im = None
while True:
    try:
        d = np.load(path, allow_pickle=False)
    except Exception:
        time.sleep(0.5)
        continue
    tern = d["ternary"]; ox, oy, res = d["origin"]
    extent = [ox, ox + tern.shape[1] * res, oy, oy + tern.shape[0] * res]
    rgb = np.full(tern.shape + (3,), 0.6); rgb[tern == 0] = 1.0; rgb[tern == 1] = 0.0
    ax.clear()
    ax.imshow(rgb, origin="lower", extent=extent)
    pose, gt, goal, pth, tg = d["pose"], d["gt"], d["goal"], d["path"], d["targets"]
    ax.plot(pose[0], pose[1], "bo", ms=8, label="추정 위치")
    ax.arrow(pose[0], pose[1], 0.4 * np.cos(pose[2]), 0.4 * np.sin(pose[2]), head_width=0.1, color="b")
    if np.isfinite(gt[0]):
        ax.plot(gt[0], gt[1], "gx", ms=10, mew=2, label=f"진짜 위치 (오차 {np.hypot(pose[0]-gt[0], pose[1]-gt[1]):.2f} m)")
    if np.isfinite(goal[0]):
        ax.plot(goal[0], goal[1], "r*", ms=14, label="목표")
    if len(pth):
        ax.plot(pth[:, 0], pth[:, 1], "m.-", lw=1.5, ms=3, label="A* 경로")
    if len(tg):
        ax.plot(tg[:, 0], tg[:, 1], "y^", ms=12, label="발견 대상")
    known = (tern != -1)
    if known.any():
        rows, cols = np.nonzero(known)
        ax.set_xlim(ox + cols.min() * res - 1, ox + cols.max() * res + 1)
        ax.set_ylim(oy + rows.min() * res - 1, oy + rows.max() * res + 1)
    ax.set_title(f"상태 {d['state']}  |  지도 채움 {known.mean() * 100:.1f}%")
    ax.legend(loc="upper right", fontsize=8)
    plt.pause(0.5)
