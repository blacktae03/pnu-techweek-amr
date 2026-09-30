"""
vision_eval.py - 카탈로그 프레임(정답 있음)으로 vision.detect_apple 을 채점한다. 파라미터를 바꾸면 바로 재채점.

    python grid_nav/vision_eval.py controllers/tb3_catalog/catalog

출력: 사과 프레임 중 탐지(재현율), 비사과 프레임 중 탐지(오탐), 놓친/오탐 파일 목록.
"""
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision
from target_detection import TARGET_COLOR, MIN_APPLE_RADIUS_PX

d = sys.argv[1] if len(sys.argv) > 1 else "."
import math
# 정답 기준: 촬영 대상이 무엇이든, 카메라 위치(물체에서 1.2 m) 반경 NEAR_M 안에 빨간 사과가 있으면 "사과가 보였을 수 있음".
# (촬영 대상 종류만으로 채점하면 옆에 있는 사과가 찍힌 프레임을 오탐으로 잘못 센다 — 2026-09-30 카탈로그에서 확인)
RED_APPLES = [(-12.02, -3.02), (-5.34, -10.54)]      # apartment.wbt 기준. 다른 월드면 수정
NEAR_M = 4.0
rows = list(csv.DictReader(open(os.path.join(d, "catalog.csv"))))
tp = fn = fp = tn = amb = 0
missed, false, ambiguous = [], [], []
for r in rows:
    img = cv2.imread(os.path.join(d, r["file"]))
    det = vision.detect_apple(img, TARGET_COLOR, MIN_APPLE_RADIUS_PX)   # 파란 원 오버레이는 빨강 마스크에 영향 없음
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
