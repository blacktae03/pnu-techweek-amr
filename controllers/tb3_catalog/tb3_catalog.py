"""
tb3_catalog.py - 월드의 바닥 물체 앞으로 로봇을 순간이동시켜 카메라 프레임을 모으는 개발용 Supervisor 컨트롤러

목적: 로봇을 몰지 않고 "이 월드에서 카메라에 빨갛게 보이는 것" 을 전부 모아 vision.py 를 오프라인에서 튜닝·회귀검증.
결과: controllers/tb3_catalog/catalog/<번호>_<노드타입>_<이름>.jpg  +  catalog.csv (타입, 이름, 위치, 탐지 여부, r, 비율)
      → python grid_nav/vision_eval.py controllers/tb3_catalog/catalog  로 채점

방법: 최상위 노드 중 z < 1.2 m 인 것(바닥·낮은 가구 위 물체) 각각에 대해, 물체에서 DIST 만큼 떨어진 곳에 로봇을 놓고
      물체를 바라보게 한 뒤(4방향 중 벽에 안 박히는 방향 선택은 생략, 4방향 모두 촬영) 몇 스텝 후 프레임 저장.
주의: 로봇이 가구와 겹치면 물리가 튕겨낼 수 있어 resetPhysics 후 2스텝만 진행하고 찍는다. 대회 제출과 무관한 도구.
"""
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "..", "grid_nav"), os.path.join(_HERE, "..", "..", "..", "grid_nav")):
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.normpath(_cand)); break

import numpy as np
from controller import Supervisor
import vision
from target_detection import TARGET_COLOR, MIN_APPLE_RADIUS_PX

DIST = 1.2                      # 물체까지 거리 [m]
MODE = os.environ.get("CATALOG_MODE", "objects")     # objects: 바닥 물체 전수 촬영 / apples: 빨간 사과 거리·각도별 촬영
OUT = os.path.join(_HERE, "catalog" if MODE == "objects" else "catalog_apples")
os.makedirs(OUT, exist_ok=True)

sup = Supervisor()
ts = int(sup.getBasicTimeStep())
me = sup.getSelf()
tr_field = me.getField("translation"); rot_field = me.getField("rotation")
camera = sup.getDevice("camera"); camera.enable(ts)
W, H = camera.getWidth(), camera.getHeight()
sup.step(ts)

root = sup.getRoot().getField("children")
objects = []
for i in range(root.getCount()):
    n = root.getMFNode(i)
    try:
        tn = n.getTypeName()
        f = n.getField("translation")
        if f is None or tn in ("TurtleBot3Burger", "Floor", "Wall", "Ceiling", "Door", "Window", "CeilingLight",
                               "Viewpoint", "WorldInfo", "TexturedBackground", "TexturedBackgroundLight"):
            continue
        x, y, z = f.getSFVec3f()
        if z > 1.2:
            continue
        name = n.getField("name").getSFString() if n.getField("name") else tn
        objects.append((tn, name, x, y, z))
    except Exception:
        continue
print(f"[catalog] 촬영 대상 {len(objects)}개")

rows = []
k = 0


def shoot(rx, ry, yaw):
    """로봇을 (rx, ry) 에 yaw 방향으로 놓고 3스텝 뒤 프레임/탐지/빨강 픽셀 수 반환."""
    tr_field.setSFVec3f([rx, ry, 0.0]); rot_field.setSFRotation([0, 0, 1, yaw])
    me.resetPhysics()
    for _ in range(3):
        sup.step(ts)
    bgr = vision.webots_image_to_bgr(camera.getImage(), W, H)
    det = vision.detect_apple(bgr, TARGET_COLOR, MIN_APPLE_RADIUS_PX)
    red_px = int((vision.color_mask(bgr, TARGET_COLOR) > 0).sum())
    return bgr, det, red_px


if MODE == "apples":
    # 빨간 사과 노드의 '실제' 위치 (물리로 굴렀을 수 있으니 getPosition)
    apples = []
    for i in range(root.getCount()):
        n = root.getMFNode(i)
        try:
            if n.getTypeName() == "RedApple":
                px, py, pz = n.getPosition()
                nm = n.getField("name").getSFString() if n.getField("name") else "RedApple"
                apples.append((nm, px, py, pz))
        except Exception:
            continue
    print(f"[catalog] apples 모드: 빨간 사과 {len(apples)}개 × 거리 4 × 방향 8")
    for nm, ax, ay, az in apples:
        for dist in (0.8, 1.5, 2.5, 3.5):
            for j in range(8):
                ang = j * math.pi / 4
                rx, ry = ax + dist * math.cos(ang), ay + dist * math.sin(ang)
                yaw = math.atan2(ay - ry, ax - rx)
                bgr, det, red_px = shoot(rx, ry, yaw)
                k += 1
                safe = "".join(ch if ch.isalnum() else "_" for ch in nm)[:30]
                fn = f"{k:03d}_RedApple_{safe}_d{dist}_a{j * 45:03d}.jpg"
                vision.save_debug_frame(vision.draw_detection(bgr, det, TARGET_COLOR), os.path.join(OUT, fn))
                rows.append(dict(file=fn, type="RedApple", name=nm, x=round(ax, 2), y=round(ay, 2), z=round(az, 2),
                                 view=j, dist=dist, angle=j * 45, red_px=red_px, detected=int(det is not None),
                                 r=round(det[2], 1) if det else "", is_apple=1))
                print(f"[catalog] {fn}: 빨강 {red_px}px 탐지={'O' if det else 'x'}")
    objects = []          # 아래 objects 루프는 건너뜀

for tn, name, x, y, z in objects:
    for j, ang in enumerate((0.0, math.pi / 2, math.pi, -math.pi / 2)):     # 물체 기준 4방향에서 접근
        rx, ry = x + DIST * math.cos(ang), y + DIST * math.sin(ang)
        yaw = math.atan2(y - ry, x - rx)                                     # 물체를 바라보는 방향
        tr_field.setSFVec3f([rx, ry, 0.0]); rot_field.setSFRotation([0, 0, 1, yaw])
        me.resetPhysics()
        for _ in range(3):
            sup.step(ts)
        bgr = vision.webots_image_to_bgr(camera.getImage(), W, H)
        det = vision.detect_apple(bgr, TARGET_COLOR, MIN_APPLE_RADIUS_PX)
        mask = vision.color_mask(bgr, TARGET_COLOR)
        red_px = int((mask > 0).sum())
        if red_px < 20 and "Apple" not in tn:
            continue                                                         # 빨강이 거의 없는 프레임은 저장 안 함
        k += 1
        safe = "".join(ch if ch.isalnum() else "_" for ch in name)[:30]
        fn = f"{k:03d}_{tn}_{safe}_v{j}.jpg"
        vision.save_debug_frame(vision.draw_detection(bgr, det, TARGET_COLOR), os.path.join(OUT, fn))
        rows.append(dict(file=fn, type=tn, name=name, x=round(x, 2), y=round(y, 2), z=round(z, 2), view=j,
                         red_px=red_px, detected=int(det is not None), r=round(det[2], 1) if det else "",
                         is_apple=int(tn.endswith("Apple"))))
        print(f"[catalog] {fn}: 빨강 {red_px}px 탐지={'O' if det else 'x'} {'(사과)' if tn.endswith('Apple') else ''}")
with open(os.path.join(OUT, "catalog.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
    wtr.writeheader(); wtr.writerows(rows)
print(f"[catalog] 완료: {k}장 저장, {OUT}/catalog.csv")
sup.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
