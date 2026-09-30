"""
webots_adapter.py - Webots(TurtleBot3 Burger) 컨트롤러 조립 코드 (브랜치 integration 소유)

역할 분담 (브랜치 = 파일). 이 파일은 아래 모듈을 순서대로 부르기만 한다.
  robot_config.py       로봇/센서 상수                       (integration)
  pose_estimation.py    (1) 위치 추정   PoseEstimator        (feat/localization)
  exploration.py        (2)(4) 지도·frontier·A*  GridPlanner (feat/exploration)
  target_detection.py   (3) 카메라 대상 탐지  TargetDetector  (feat/vision)
  motion_control.py     (5) 경로 → 바퀴 속도  MotionController (feat/dwa)
  webots_adapter.py     (4) 미션 상태 기계, (6) 로그/지도 저장/Display  (integration)

확인된 사실 [2026-09-30]:
  - 디바이스 이름: "LDS-01", "camera", "gyro", "compass", "left/right wheel motor", 엔코더는 motor.getPositionSensor()
  - LiDAR: ranges[180]=정면, [0]=후방, [90]=왼쪽, [270]=오른쪽, 100 ms 주기 → 같은 스캔이 반복 들어옴 (새 스캔만 갱신)
  - 컴퍼스 부호 -1 (robot_config.COMPASS_SIGN), 시작 pose apartment (-0.3, -7.5, pi)
  - 첫 스캔 전에 계획하면 "frontier 없음 → 탐색 완료" 가 되어 즉시 끝남 → 스캔 3개 대기 + 3회 연속 확인
실행: controllers/tb3_gridnav/tb3_gridnav.py 가 run() 을 부른다. 터미널에서 직접 실행하면 뼈대 점검만.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np

from geometry import wrap_angle
from robot_config import (ROBOT_RADIUS, LIDAR_NAME, LIDAR_PERIOD_MS, LIDAR_MAX_RANGE, MAX_WHEEL_SPEED,
                          COMPASS_SIGN, START_POSE_APARTMENT)
from pose_estimation import PoseEstimator, WheelOdometry, CompassHeading          # 재수출 (tb3_reactive 가 씀)
from exploration import GridPlanner, REPLAN_PERIOD_S
from target_detection import TargetDetector, target_world_from_pixel
from motion_control import MotionController, lookahead_control, to_wheel_speeds

try:
    from controller import Robot          # Webots 가 제공. Webots 밖에서는 없음
    _IN_WEBOTS = True
except ImportError:
    Robot = object                        # type: ignore
    _IN_WEBOTS = False

TARGET_COUNT = 1            # 이 개수를 찾으면 탐색을 멈추고 방문 → 복귀. 당일 규칙에 맞출 것
ARRIVE_DIST = ROBOT_RADIUS + 0.3   # 대상 "도착" 판정 거리 (당일 규칙 확인)
HOME_DIST = 0.2             # 복귀 완료 판정
HOLD_SECONDS = float(os.environ.get("GRIDNAV_HOLD", "0"))   # 테스트용: 처음 N초 정지
# 헛바퀴(slip) 감지: 전진 명령 중인데 스캔이 1초 전과 거의 같으면 로봇은 안 움직인 것 (낮은 물체에 걸림)
SLIP_WINDOW_S = 1.0         # 비교할 과거 스캔의 시간 차
SLIP_SCAN_CHANGE_M = 0.04   # 이보다 스캔 변화가 작으면 '정지' (0.15 m/s × 1 s = 0.15 m 변해야 정상)
SLIP_CONFIRM_S = 2.0        # 이 시간 연속 정지 판정이면 걸린 것으로 확정
SLIP_SECTOR_DEG = 40        # 앞뒤 ±이 각도의 빔만 비교. 복도에서 옆 벽 빔은 직진해도 안 변하므로 제외 (4차 실행 오판 원인)
RECOVER_BACK_S = 3.0        # 복구: 후진 시간 (0.1 m/s → 0.3 m)
OBSTACLE_AHEAD_M = 0.25     # 복구 시 로봇 앞 이 거리에 장애물을 찍음


# ======================================================================
# LiDAR 각도 배열  [확인됨: LDS-01 인덱스 매핑]
# ======================================================================
def lidar_angles(n_points=360):
    """ranges[i] 의 각도 (정면 0, 왼쪽 +). angle_i = pi - 2*pi*i/N  → i=180 정면, i=90 왼쪽, i=270 오른쪽."""
    i = np.arange(n_points)
    return wrap_angle(np.pi - 2.0 * np.pi * i / n_points)


def read_lidar(lidar):
    return np.array(lidar.getRangeImage(), dtype=float)


# ======================================================================
# Display 에 지도 그리기 (개발용 월드의 map_display)
# ======================================================================
def draw_map_on_display(display, W, H, planner, pose, goal, path, targets, state):
    """지도의 '알려진 영역' 을 잘라 Display 에 확대해 그린다. Display 는 y=0 이 위라 세로를 뒤집는다."""
    import cv2
    from controller import Display
    tern, spec = planner.get_map()
    known = tern != -1
    if not known.any():
        return
    r_, c_ = np.nonzero(known)
    r0, r1 = max(0, r_.min() - 10), min(spec.rows, r_.max() + 11)
    c0, c1 = max(0, c_.min() - 10), min(spec.cols, c_.max() + 11)
    crop = tern[r0:r1, c0:c1]
    rgb = np.full(crop.shape + (3,), 150, np.uint8)
    rgb[crop == 0] = 255
    rgb[crop == 1] = 0
    rgb = rgb[::-1]
    scale = min(W / rgb.shape[1], H / rgb.shape[0])
    ow, oh = max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))
    canvas = np.full((H, W, 3), 40, np.uint8)
    canvas[:oh, :ow] = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_NEAREST)
    x0 = spec.origin_x + c0 * spec.resolution
    y1 = spec.origin_y + r1 * spec.resolution

    def to_px(x, y):
        return int((x - x0) / spec.resolution * scale), int((y1 - y) / spec.resolution * scale)

    if path:
        cv2.polylines(canvas, [np.array([to_px(x, y) for x, y in path], np.int32)], False, (200, 0, 200), 2)   # 자주 (RGB)
    for tx, ty in targets:
        cv2.drawMarker(canvas, to_px(tx, ty), (255, 210, 0), cv2.MARKER_TRIANGLE_UP, 14, 2)   # 노랑 = 기록한 사과
    if goal is not None:
        cv2.drawMarker(canvas, to_px(goal[0], goal[1]), (255, 0, 0), cv2.MARKER_STAR, 16, 2)   # 빨강 = 현재 목표
    px, py = to_px(pose[0], pose[1])
    cv2.circle(canvas, (px, py), max(3, int(ROBOT_RADIUS / spec.resolution * scale)), (0, 120, 255), -1)   # 파랑 = 로봇
    cv2.line(canvas, (px, py), (int(px + 12 * math.cos(pose[2])), int(py - 12 * math.sin(pose[2]))), (0, 0, 0), 2)
    cv2.putText(canvas, state, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    im = display.imageNew(canvas.tobytes(), Display.RGB, W, H)
    display.imagePaste(im, 0, 0, False)
    display.imageDelete(im)


# ======================================================================
# 컨트롤러 본체
# ======================================================================
def main():
    if not _IN_WEBOTS:
        print("Webots 밖에서 실행됨: controller 모듈이 없어 뼈대 점검만 합니다. (import/문법 OK)")
        planner = GridPlanner(start_xy=START_POSE_APARTMENT[:2])
        tern, spec = planner.get_map()
        ang = lidar_angles(360)
        print(f"  GridPlanner 준비: 지도 {tern.shape}, 해상도 {spec.resolution} m")
        print(f"  LiDAR 각도: idx0={math.degrees(ang[0]):.0f}°(후방) idx90={math.degrees(ang[90]):.0f}°(왼쪽) "
              f"idx180={math.degrees(ang[180]):.0f}°(정면) idx270={math.degrees(ang[270]):.0f}°(오른쪽)")
        (tx, ty), b, d = target_world_from_pixel((0, 0, 0), cx_px=400, radius_px=20)
        print(f"  픽셀(400, r=20) → bearing {math.degrees(b):.1f}°, dist {d:.2f} m, world ({tx:.2f}, {ty:.2f})")
        return
    run(Robot(), start_pose=START_POSE_APARTMENT)


def run(robot, start_pose, ground_truth=None, map_save_path="map.npy", map_save_every_s=2.0,
        use_scan_matching=False):
    """robot: Robot() 또는 Supervisor().  start_pose: 대회 제공 (x, y, theta).
    ground_truth: 디버깅용 콜백 (진짜 pose). map_save_path: 지도/상태 npz 저장 위치 (view_live.py 용)."""
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    # --- 디바이스 ---
    lidar = robot.getDevice(LIDAR_NAME); lidar.enable(LIDAR_PERIOD_MS); lidar.enablePointCloud()
    left_motor = robot.getDevice("left wheel motor"); right_motor = robot.getDevice("right wheel motor")
    for m in (left_motor, right_motor):
        m.setPosition(float("inf")); m.setVelocity(0.0)
    left_enc = left_motor.getPositionSensor();   left_enc.enable(timestep)
    right_enc = right_motor.getPositionSensor(); right_enc.enable(timestep)
    gyro = robot.getDevice("gyro");       gyro.enable(timestep)
    compass = robot.getDevice("compass"); compass.enable(timestep)
    camera = robot.getDevice("camera");   camera.enable(timestep)
    try:
        display = robot.getDevice("map_display"); disp_w, disp_h = display.getWidth(), display.getHeight()
    except Exception:
        display = None
    debug_dir = os.path.dirname(os.path.abspath(map_save_path)) if map_save_path else None

    # --- 모듈 (브랜치별) ---
    planner = GridPlanner(start_xy=start_pose[:2], max_range=min(lidar.getMaxRange(), LIDAR_MAX_RANGE))
    estimator = PoseEstimator(start_pose, use_scan_matching=use_scan_matching, spec=planner.spec)
    detector = TargetDetector(camera.getWidth(), camera.getHeight(), camera.getFov(), debug_dir)
    motion = MotionController()

    # --- 미션 상태 ---
    x0, y0 = start_pose[0], start_pose[1]
    state = "EXPLORE"                   # EXPLORE → VISIT → RETURN → DONE
    path: Optional[List[Tuple[float, float]]] = None
    goal: Optional[Tuple[float, float]] = None
    visited: List[Tuple[float, float]] = []
    angles = None; last_scan = None; scan_count = 0; no_frontier_count = 0
    scan_hist: List[Tuple[float, np.ndarray]] = []     # (t, ranges) 최근 스캔들 (slip 감지용)
    slip_time = 0.0; slip_pose_ref = None; recover_until = -1e9; slip_events = 0; slip_goal_count = {}
    last_plan_t = last_save = last_log = last_disp = -1e9
    v = w = 0.0

    while robot.step(timestep) != -1:
        t = robot.getTime()
        ranges = read_lidar(lidar)
        if angles is None:
            angles = lidar_angles(len(ranges))
        new_scan = last_scan is None or not np.array_equal(ranges, last_scan)
        valid_scan = new_scan and np.isfinite(ranges).any() and np.nanmax(np.where(np.isfinite(ranges), ranges, 0)) > 0.0

        # (1) 위치 추정  [feat/localization]
        pose = estimator.update(left_enc.getValue(), right_enc.getValue(), compass.getValues(),
                                gyro_z=gyro.getValues()[2], dt=dt,
                                ranges=ranges if valid_scan else None, angles=angles, log_odds=planner.grid.log_odds)

        # (1b) 헛바퀴 감지  [integration]: 전진 중인데 환경(스캔)이 안 변하면 로봇은 제자리 → odometry 증가분 취소
        if valid_scan:
            scan_hist.append((t, ranges)); scan_hist = [(ts, r) for ts, r in scan_hist if t - ts <= SLIP_WINDOW_S + 0.2]
        moving_cmd = abs(motion.v_cmd) > 0.05 and abs(motion.w_cmd) < 0.4
        if moving_cmd and len(scan_hist) >= 2 and t - scan_hist[0][0] >= SLIP_WINDOW_S * 0.8 and t >= recover_until:
            old = scan_hist[0][1]
            sec = np.deg2rad(SLIP_SECTOR_DEG)
            fb = (np.abs(wrap_angle(angles)) < sec) | (np.abs(wrap_angle(angles - np.pi)) < sec)   # 앞뒤 빔만
            both = fb & np.isfinite(ranges) & np.isfinite(old)
            change = float(np.mean(np.abs(ranges[both] - old[both]))) if both.sum() > 15 else np.inf
            if change < SLIP_SCAN_CHANGE_M:
                if slip_pose_ref is None:
                    slip_pose_ref = pose
                slip_time += dt
            else:
                slip_time = 0.0; slip_pose_ref = None
        else:
            slip_time = 0.0; slip_pose_ref = None
        if slip_time > SLIP_CONFIRM_S:
            slip_events += 1
            estimator.odom.set_pose(slip_pose_ref); pose = slip_pose_ref      # 확정된 뒤에만 가짜 전진 취소
            ox = pose[0] + OBSTACLE_AHEAD_M * math.cos(pose[2]); oy = pose[1] + OBSTACLE_AHEAD_M * math.sin(pose[2])
            planner.mark_obstacle(ox, oy, 0.15)
            key = None if goal is None else (round(goal[0], 1), round(goal[1], 1))
            slip_goal_count[key] = slip_goal_count.get(key, 0) + 1
            if goal is not None and slip_goal_count[key] >= 2:
                planner.give_up_goal(goal); print(f"[{t:.1f}s] 같은 목표에서 2번 걸림 → 목표 포기 {key}")
            print(f"[{t:.1f}s] !! 헛바퀴 감지 ({slip_events}회): 앞 {OBSTACLE_AHEAD_M} m 에 장애물 표시, {RECOVER_BACK_S} s 후진 후 재계획")
            recover_until = t + RECOVER_BACK_S; slip_time = 0.0; slip_pose_ref = None; path = None

        # (2) 지도 갱신  [feat/exploration]  (새 스캔일 때만. LiDAR 100 ms 주기)
        if new_scan:
            if valid_scan:
                planner.update_map(pose, angles, ranges); scan_count += 1
            last_scan = ranges

        # (3) 대상 탐지  [feat/vision]
        detector.process(camera.getImage(), pose, t)
        found_targets = detector.targets

        # (4) 미션 상태 기계 + 경로 계획  [integration]  (스캔 3개 쌓인 뒤, 1 s 주기)
        if scan_count >= 3 and t >= recover_until and (path is None or t - last_plan_t > REPLAN_PERIOD_S):
            last_plan_t = t
            if state == "EXPLORE" and len(found_targets) >= TARGET_COUNT:
                state = "VISIT"; print(f"[{t:.1f}s] 대상 {len(found_targets)}개 확보 → 탐색 중단, VISIT")
            if state == "EXPLORE":
                goal = planner.next_exploration_goal(pose, current_goal=goal)
                if goal is None:
                    no_frontier_count += 1
                    if no_frontier_count >= 3:
                        state = "VISIT" if found_targets else "RETURN"; print(f"[{t:.1f}s] 탐색 완료 → {state}")
                else:
                    no_frontier_count = 0
            if state == "VISIT":
                remaining = [tg for tg in found_targets if tg not in visited]
                if remaining:
                    goal = min(remaining, key=lambda tg: math.hypot(tg[0] - pose[0], tg[1] - pose[1]))
                    if math.hypot(goal[0] - pose[0], goal[1] - pose[1]) < ARRIVE_DIST:
                        visited.append(goal); print(f"[{t:.1f}s] 대상 도착 ({goal[0]:.2f}, {goal[1]:.2f})"); goal = None
                else:
                    state = "RETURN"
            if state == "RETURN":
                goal = (x0, y0)
                if math.hypot(pose[0] - x0, pose[1] - y0) < HOME_DIST:
                    state = "DONE"; goal = None; print(f"[{t:.1f}s] 복귀 완료")
            path = planner.plan(pose, goal) if goal is not None else None
            if goal is not None and path is None:
                planner.give_up_goal(goal)
            if planner.grid.dropped_hits > 100:
                print("!! 지도 밖 측정값 많음 → exploration.MAP_HALF_M 확인")

        # (5) 속도 명령  [feat/dwa]   (환경변수 GRIDNAV_HOLD=초 를 주면 그동안 정지: 비전 테스트용)
        v, w = motion.command(pose, path, ranges, angles, state)
        if t < recover_until:                        # 복구 중: 후진
            v, w = -0.10, 0.0
            motion.v_cmd, motion.w_cmd = v, w
        if t < HOLD_SECONDS:
            v, w = 0.0, 0.0
        wl, wr = to_wheel_speeds(v, w)
        left_motor.setVelocity(wl); right_motor.setVelocity(wr)

        # (6) 로그 / 지도 저장 / Display  [integration]
        if map_save_path and t - last_save >= map_save_every_s:
            last_save = t
            tern, spec = planner.get_map()
            gt = ground_truth() if ground_truth else None
            np.savez(map_save_path.replace(".npy", ".npz"), ternary=tern,
                     origin=np.array([spec.origin_x, spec.origin_y, spec.resolution]),
                     pose=np.array(pose), gt=np.array(gt if gt else [np.nan] * 3),
                     goal=np.array(goal if goal else [np.nan, np.nan]),
                     path=np.array(path if path else np.zeros((0, 2))),
                     targets=np.array(found_targets if found_targets else np.zeros((0, 2))), state=np.array(state))
        if display is not None and t - last_disp >= 1.0:
            last_disp = t
            draw_map_on_display(display, disp_w, disp_h, planner, pose, goal, path, found_targets, state)
        if t - last_log >= 1.0:
            last_log = t
            gt = ground_truth() if ground_truth else None
            err = f" 위치오차={math.hypot(pose[0] - gt[0], pose[1] - gt[1]):.2f}m" if gt else ""
            print(f"[{t:6.1f}s] {state:8s} pose=({pose[0]:.2f},{pose[1]:.2f},{math.degrees(pose[2]):.0f}°){err} "
                  f"목표={None if goal is None else (round(goal[0], 2), round(goal[1], 2))} 경로점={0 if not path else len(path)} "
                  f"v={v:.2f} w={w:+.2f} 대상={len(found_targets)} 헛바퀴={slip_events} 비전거부={detector.rejected} 지도밖hit={planner.grid.dropped_hits}")


if __name__ == "__main__":
    main()
