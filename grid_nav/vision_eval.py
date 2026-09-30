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
rows = list(csv.DictReader(open(os.path.join(d, "catalog.csv"))))
tp = fn = fp = tn = 0
missed, false = [], []
for r in rows:
    img = cv2.imread(os.path.join(d, r["file"]))
    # 저장된 프레임에는 파란 원이 그려져 있지만 빨강 마스크에는 영향 없음
    det = vision.detect_apple(img, TARGET_COLOR, MIN_APPLE_RADIUS_PX)
    apple = r["is_apple"] == "1" and r["type"] == "RedApple"
    if apple and det: tp += 1
    elif apple and not det: fn += 1; missed.append(r["file"])
    elif not apple and det: fp += 1; false.append(r["file"])
    else: tn += 1
print(f"빨간 사과 프레임: 탐지 {tp} / 놓침 {fn}      비사과 프레임: 오탐 {fp} / 정상 거부 {tn}")
for f in missed: print("  놓침:", f)
for f in false: print("  오탐:", f)
