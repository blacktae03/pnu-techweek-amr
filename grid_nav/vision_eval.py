"""
vision_eval.py - 카탈로그 프레임(정답 있음)으로 vision.detect_apple 을 채점한다. 파라미터를 바꾸면 바로 재채점.

    python grid_nav/vision_eval.py controllers/tb3_catalog/catalog          # 바닥 물체 전수 (오탐 검사)
    python grid_nav/vision_eval.py controllers/tb3_catalog/catalog_apples   # 빨간 사과 거리·각도별 (재현율 검사)

objects 카탈로그: 촬영 대상 종류와 '근처 사과' 로 정답을 매긴다 (옆에 있는 사과가 찍힌 프레임은 보류).
apples 카탈로그: 거리별 탐지율(빨강 픽셀 0 = 시야 가림 은 제외), 놓친 프레임의 수치, 그리고 target_detection 단계 필터
                (수평선 아래, 반지름거리/세로거리 비율 0.5~2, 최대 거리 3.0 m) 통과 여부까지 채점.
"""
import csv
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision
import target_detection as td
from target_detection import TARGET_COLOR, MIN_APPLE_RADIUS_PX
from robot_config import CAM_H

RED_APPLES = [(-12.02, -3.02), (-5.34, -10.54)]      # apartment.wbt 기준. 다른 월드면 수정
NEAR_M = 4.0


def blob_stats(img):
    """가장 큰 빨간 덩어리의 (cx, cy, r, 원형도, 높이/폭, 위아래빨강). 없으면 None."""
    mask = vision.color_mask(img, TARGET_COLOR)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    circ = cv2.contourArea(c) / (math.pi * r * r + 1e-6)
    w, h = vision.object_extent(mask, cx, cy, r)
    return cx, cy, r, circ, h / w, vision.red_outside_blob(mask, cx, cy, r)


def stage2_ok(cx, cy, r):
    """target_detection 의 채택 규칙(프레임 연속 조건 제외)."""
    (_, _), _, d_r = td.target_world_from_pixel((0, 0, 0), cx, r)
    d_row = td.distance_from_row(cy)
    horizon = cy >= CAM_H / 2 + td.HORIZON_MARGIN_PX
    ratio = d_r / d_row if np.isfinite(d_row) and d_row > 0 else float("inf")
    consistent = 1.0 / td.CONSISTENCY_RATIO < ratio < td.CONSISTENCY_RATIO
    near = d_r <= td.MAX_DETECT_DIST
    return horizon and consistent and near, d_r, d_row


def eval_objects(d, rows):
    tp = fn = fp = tn = amb = 0
    missed, false, ambiguous = [], [], []
    for r in rows:
        img = cv2.imread(os.path.join(d, r["file"]))
        det = vision.detect_apple(img, TARGET_COLOR, MIN_APPLE_RADIUS_PX)
        apple_target = r["type"] == "RedApple"
        x, y = float(r["x"]), float(r["y"])
        apple_near = min(math.hypot(x - a, y - b) for a, b in RED_APPLES) < NEAR_M
        red_seen = int(r["red_px"]) > 0
        if apple_target:
            if det: tp += 1
            elif not red_seen: amb += 1; ambiguous.append(r["file"] + " (빨강 픽셀 0 = 시야 가림)")
            else: fn += 1; missed.append(r["file"])
        elif det:
            if apple_near: amb += 1; ambiguous.append(r["file"] + " (근처 사과가 찍혔을 가능성)")
            else: fp += 1; false.append(r["file"])
        else:
            tn += 1
    print(f"빨간 사과 프레임: 탐지 {tp} / 놓침 {fn}      비사과 프레임: 오탐 {fp} / 정상 거부 {tn}      판정 보류 {amb}")
    for f in missed: print("  놓침:", f)
    for f in false: print("  오탐:", f)
    for f in ambiguous: print("  보류:", f)


def eval_apples(d, rows):
    by_dist = {}
    misses = []
    for r in rows:
        img = cv2.imread(os.path.join(d, r["file"]))
        dist = float(r["dist"])
        b = by_dist.setdefault(dist, dict(total=0, hidden=0, det=0, stage2=0, live=0))
        b["total"] += 1
        if int(r["red_px"]) == 0:
            b["hidden"] += 1
            continue
        b["live"] += int(r["detected"])          # 컨트롤러가 원본 프레임에서 탐지한 결과 (JPEG 손실 없음)
        det = vision.detect_apple(img, TARGET_COLOR, MIN_APPLE_RADIUS_PX)
        st = blob_stats(img)
        if det:
            b["det"] += 1
            ok, d_r, d_row = stage2_ok(det[0], det[1], det[2])
            if ok:
                b["stage2"] += 1
            else:
                misses.append((r["file"], "2단계", f"d_r={d_r:.2f} d_row={d_row:.2f} cy-H/2={det[1] - CAM_H / 2:.0f}"))
        elif st is None:
            misses.append((r["file"], "1단계", f"저장 프레임에서 빨간 덩어리 없음 (촬영 시 red_px={r['red_px']}, JPEG 압축/모폴로지로 소실)"))
        else:
            cx, cy, rr, circ, asp, out = st
            misses.append((r["file"], "1단계", f"r={rr:.1f} 원형도={circ:.2f} 높이/폭={asp:.2f} 위아래빨강={out:.2f} red_px={r['red_px']}"))
    print(f"{'거리[m]':>7s} {'프레임':>5s} {'시야가림':>6s} {'라이브 탐지':>10s} {'JPEG 재판정':>10s} {'2단계 통과':>9s}")
    for dist in sorted(by_dist):
        b = by_dist[dist]; vis = b["total"] - b["hidden"]
        live = f"{b['live']}/{vis} ({100 * b['live'] / max(1, vis):.0f}%)"
        rate1 = f"{b['det']}/{vis} ({100 * b['det'] / max(1, vis):.0f}%)"
        rate2 = f"{b['stage2']}/{vis} ({100 * b['stage2'] / max(1, vis):.0f}%)"
        print(f"{dist:7.1f} {b['total']:5d} {b['hidden']:6d} {live:>12s} {rate1:>12s} {rate2:>12s}")
    print("  (라이브 = 촬영 시 원본 프레임 판정. JPEG 재판정은 압축으로 먼 사과의 작은 빨강이 사라져 낮게 나올 수 있음)")
    for f, stage, why in misses:
        print(f"  놓침({stage}): {f}  {why}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = list(csv.DictReader(open(os.path.join(d, "catalog.csv"))))
    if rows and "dist" in rows[0]:
        eval_apples(d, rows)
    else:
        eval_objects(d, rows)
