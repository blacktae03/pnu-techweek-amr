"""
dwa.py - Dynamic Window Approach: 실시간 장애물 회피 국소 계획기 (Local Planner)

역할: A* 가 준 경로(월드 좌표 점들)를 따라가되, "지금 LiDAR 에 보이는" 장애물(사람, 지도에 없는 물체)을
      피하는 (v, ω) 속도 명령을 매 스텝 만든다. Global Planner(A*) 는 1초에 한 번, 이건 매 스텝(10~20Hz).

아이디어 (강의 슬라이드 109쪽 'DWA, Critic 기반'):
  1) 지금 속도에서 한 스텝 안에 도달 가능한 (v, ω) 후보를 격자로 깐다  ← "Dynamic Window" (가감속 한계)
  2) 각 후보로 T 초 동안 달리면 어디를 지나는지 궤적을 시뮬레이션한다 (등속 원호)
  3) 궤적마다 점수(critic)를 매긴다:
       - 장애물에 부딛는 궤적은 탈락
       - 목표 방향을 향하는가 (heading)      - 목표에 가까워지는가 (distance)
       - 장애물에서 멀리 떨어지는가 (clearance) - 빨리 가는가 (velocity)
  4) 점수가 가장 좋은 후보의 (v, ω) 를 이번 스텝 명령으로 내보낸다. 다음 스텝에 다시 계산.

전부 NumPy 벡터화: 후보 K개 × 시간 T스텝 의 (K, T, 2) 배열을 한 번에 만들고, 장애물과의 거리는
KD-tree 한 번 질의로 끝낸다. 후보 7x15=105개, 15스텝이면 1575개 점 → 1~3 ms.

팀 인터페이스:
  dwa_control(pose, v_cur, w_cur, goal_xy, obstacles_xy, params) -> (v, w)
    goal_xy      : 경로 위 look-ahead 점 (path_lookahead_point 로 뽑음)
    obstacles_xy : LiDAR 끝점의 월드 좌표 (N, 2). 지도 벽이 아니라 '지금 보이는 것' 을 쓴다 → 사람도 포함
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import wrap_angle

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


@dataclass
class DWAParams:
    # --- 로봇 한계 (TurtleBot3 Burger 공식 스펙) ---
    v_max: float = 0.22          # m/s
    v_min: float = -0.15         # 후진 허용. 사람이 다가올 때 물러날 수단. (LiDAR 가 360° 라 뒤도 보임)
                                 # 후진이 싫으면 0.0 → 대신 사람이 정면으로 걸어오면 서서 맞는다
    w_max: float = 2.84          # rad/s
    a_v: float = 0.5             # m/s²   한 스텝에 바꿀 수 있는 속도 폭 = a_v * dt
    a_w: float = 3.2             # rad/s²
    # --- 시뮬레이션 ---
    dt: float = 0.1              # 제어 주기 [s]
    predict_time: float = 1.5    # 궤적을 몇 초 앞까지 볼지. 길면 신중, 짧으면 반응적
    n_v: int = 7                 # v 후보 개수
    n_w: int = 15                # ω 후보 개수
    # --- 안전 ---
    robot_radius: float = 0.105
    safety_margin: float = 0.10  # 이 안으로 들어오는 궤적은 탈락
    dyn_margin: float = 0.40     # '움직일 수 있는 것'(지도에 굳지 않은 점) 에 대한 안전거리. 사람은 다가오니 더 멀리서 피함
                                 # 트레이드오프: 0.40 → 길을 계속 가로막는 사람 앞에서는 안전하게 기다림(미도착 가능)
                                 #              0.30 → 지나가지만 사람이 정면으로 걸어오면 부딛힐 수 있음. sim_demo 는 0.30 사용
    clearance_max: float = 0.5   # 이보다 멀면 clearance 점수 만점. 실내 통로 폭(1 m 안팎) 을 생각하면 0.5 가 적당.
                                 # 크게 잡으면 좁은 문 앞에서 "들어가기 싫어" 진동하는 원인이 됨
    # --- critic 가중치 (튠 대상. 합이 1 일 필요 없음) ---
    #   clearance 를 dist 보다 크게 두는 이유: 심사 기준이 '충돌 회피·안전거리'. 조금 느려도 멀리 돌아가는 편이 낫다.
    w_heading: float = 0.8
    w_dist: float = 1.0
    w_clearance: float = 1.2
    w_velocity: float = 0.3
    w_reverse: float = 0.5       # 후진 후보에 추가 벌점 (필요할 때만 후진)
    wheel_v_max: float = np.inf  # 바퀴 선속도 한계 [m/s]. 유한하면 실행 가능한 후보만 평가
    wheel_separation: float = 0.160
    control_dt: float | None = None  # 실제 명령 주기. None이면 기존 dt 사용; dt는 궤적 적분 간격


def dynamic_window(v_cur, w_cur, p: DWAParams):
    """지금 속도에서 dt 안에 도달 가능한 (v, ω) 후보 격자."""
    control_dt = p.dt if p.control_dt is None else p.control_dt
    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be positive and finite")
    v_lo = max(p.v_min, v_cur - p.a_v * control_dt)
    v_hi = min(p.v_max, v_cur + p.a_v * control_dt)
    w_lo = max(-p.w_max, w_cur - p.a_w * control_dt)
    w_hi = min(p.w_max, w_cur + p.a_w * control_dt)
    vs = np.linspace(v_lo, v_hi, p.n_v)
    ws = np.linspace(w_lo, w_hi, p.n_w)
    # 균등 격자가 0을 놓쳐도 정지/직진을 선택할 수 있게 한다 (후보 수 유지).
    if v_lo <= 0.0 <= v_hi:
        vs[np.argmin(np.abs(vs))] = 0.0
    if w_lo <= 0.0 <= w_hi:
        ws[np.argmin(np.abs(ws))] = 0.0
    V, W = np.meshgrid(vs, ws, indexing="ij")
    return V.ravel(), W.ravel()                     # (K,), (K,)


def simulate_trajectories(pose, V, W, p: DWAParams):
    """각 (v, ω) 로 predict_time 동안 등속 원호 주행 → (K, T, 3) [x, y, θ]."""
    T = int(round(p.predict_time / p.dt))
    t = np.arange(1, T + 1) * p.dt                                 # (T,)
    th = pose[2] + W[:, None] * t[None, :]                         # (K, T)
    # x(t) = x0 + Σ v cos θ dt  (누적합으로 벡터화)
    dx = V[:, None] * np.cos(th) * p.dt
    dy = V[:, None] * np.sin(th) * p.dt
    xs = pose[0] + np.cumsum(dx, axis=1)
    ys = pose[1] + np.cumsum(dy, axis=1)
    return np.stack([xs, ys, th], axis=-1)


def dwa_control(pose, v_cur, w_cur, goal_xy, obstacles_xy, p: DWAParams = DWAParams(), return_debug=False,
                dynamic_xy=None):
    """(v, ω) 명령 한 개 반환.
    obstacles_xy : 정적으로 봐도 되는 점 (지도에서 여러 번 확인된 벽)
    dynamic_xy   : 움직일 수 있는 점 (지도에 아직 굳지 않은 것 = 사람일 가능성). dyn_margin 만큼 더 멀리 피함.
                   DWA 는 장애물 속도를 모르므로, '모르는 것은 더 멀리' 가 가장 단순하고 효과적인 대응.
    모든 후보가 여유 거리 기준을 위반하면 몸체 충돌이 없는 복구 궤적(보통 후진)을
    택한다. 복구 궤적도 없으면 정지한다."""
    V, W = dynamic_window(v_cur, w_cur, p)
    traj = simulate_trajectories(pose, V, W, p)                    # (K, T, 3)
    K, T, _ = traj.shape

    # --- clearance: 궤적 위 모든 점에서 가장 가까운 장애물까지의 거리, 궤적별 최소 ---
    obs = np.asarray(obstacles_xy, dtype=float).reshape(-1, 2)
    if obs.shape[0] > 0:
        tree = cKDTree(obs)
        d, _ = tree.query(traj[..., :2].reshape(-1, 2))
        clearance = d.reshape(K, T).min(axis=1)                    # (K,)
    else:
        clearance = np.full(K, np.inf)
    physical_clearance = clearance.copy()
    if dynamic_xy is not None and len(dynamic_xy) > 0:
        dtree = cKDTree(np.asarray(dynamic_xy, dtype=float).reshape(-1, 2))
        dd, _ = dtree.query(traj[..., :2].reshape(-1, 2))
        clear_dyn = dd.reshape(K, T).min(axis=1)
        physical_clearance = np.minimum(physical_clearance, clear_dyn)
        # 동적 점은 여유(dyn_margin - safety_margin) 를 뺀 '유효 거리' 로 환산해 정적 점과 같은 잣대로 비교
        clearance = np.minimum(clearance, clear_dyn - (p.dyn_margin - p.safety_margin))
    # 정지거리 v²/(2a) 만큼 여유를 더 둠: 빨리 갈수록 더 멀리서 멈춰야 한다
    brake = V ** 2 / (2.0 * p.a_v)
    collide = clearance < p.robot_radius + p.safety_margin + brake

    # --- critics (전부 0=좋음 ~ 1=나쁨 으로 정규화) ---
    gx, gy = goal_xy
    end = traj[:, -1, :]
    heading_err = np.abs(wrap_angle(np.arctan2(gy - end[:, 1], gx - end[:, 0]) - end[:, 2]))
    c_heading = heading_err / np.pi
    dist_end = np.hypot(gx - end[:, 0], gy - end[:, 1])
    dist_now = np.hypot(gx - pose[0], gy - pose[1])
    c_dist = np.clip(dist_end / max(dist_now, 1e-6), 0.0, 2.0) / 2.0      # 목표에 얼마나 가까워졌나
    c_clear = 1.0 - np.clip(clearance / p.clearance_max, 0.0, 1.0)
    c_vel = 1.0 - np.clip(V, 0.0, None) / max(p.v_max, 1e-6)
    c_rev = (V < -1e-3).astype(float)

    cost = (p.w_heading * c_heading + p.w_dist * c_dist
            + p.w_clearance * c_clear + p.w_velocity * c_vel + p.w_reverse * c_rev)
    # v와 w가 각자 한계 안이어도 바깥 바퀴는 모터 한계를 넘을 수 있다.
    # adapter에서 나중에 축소하면 평가한 궤적과 달라지므로 여기서 제외한다.
    reachable = np.abs(V) + np.abs(W) * p.wheel_separation / 2.0 <= p.wheel_v_max
    cost[collide | ~reachable] = np.inf

    if not np.isfinite(cost).any():
        # 전부 충돌: 복구 행동 = 장애물에서 가장 멀어지는 궤적 (보통 후진 또는 제자리 회전)
        # 여유 거리 안에 들어왔어도 몸체 충돌이 예측되는 궤적은 복구용으로 쓰지 않는다.
        recoverable = reachable & (physical_clearance >= p.robot_radius + 0.02 + brake)
        if recoverable.any():
            best = int(np.argmax(np.where(recoverable, clearance, -np.inf)))
            cmd = (float(V[best]), float(W[best]))
        else:
            cmd = (0.0, 0.0)
        best = None
    else:
        best = int(np.argmin(cost))
        cmd = (float(V[best]), float(W[best]))
    if return_debug:
        return cmd, dict(traj=traj, cost=cost, best=best, collide=collide,
                         reachable=reachable, clearance=clearance)
    return cmd


def path_lookahead_point(pose, path_xy, lookahead=0.5):
    """경로에서 로봇에 가장 가까운 점부터 경로 거리 lookahead 만큼 앞의 점 (DWA 의 임시 목표).
    A* 경로 끝점이 lookahead 보다 가까우면 끝점 자체를 반환."""
    pts = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 1:
        return tuple(pts[0])
    d = np.hypot(pts[:, 0] - pose[0], pts[:, 1] - pose[1])
    i = int(np.argmin(d))
    acc = 0.0
    while i < len(pts) - 1 and acc < lookahead:
        acc += float(np.hypot(*(pts[i + 1] - pts[i])))
        i += 1
    return (float(pts[i, 0]), float(pts[i, 1]))


# ======================================================================
# 단독 실행 테스트: 복도 + 길을 가로지르는 사람
# ======================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import setup_korean_font
    setup_korean_font()

    p = DWAParams()
    # 복도 벽 두 줄 (y=0, y=1.6) 을 점으로, 사람 = 반지름 0.25 원, 로봇 앞을 가로지름
    wall_x = np.linspace(0, 6, 121)
    walls = np.vstack([np.column_stack([wall_x, np.zeros_like(wall_x)]),
                       np.column_stack([wall_x, np.full_like(wall_x, 1.6)])])
    path = [(0.5, 0.8), (5.5, 0.8)]                              # A* 가 준 직선 경로
    pose = np.array([0.5, 0.8, 0.0]); v = w = 0.0
    person = np.array([3.0, 1.6])                                 # 위 벽에서 아래로 내려옴
    person_v = np.array([0.0, -0.35])

    traj_robot, traj_person, min_clear = [], [], []
    arrived = False
    for step in range(600):
        # 사람 이동 (아래 벽에 닿으면 되돌아감)
        person = person + person_v * p.dt
        if person[1] < 0.0 or person[1] > 1.6:
            person_v = -person_v
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        person_pts = person + 0.25 * np.column_stack([np.cos(ang), np.sin(ang)])
        obstacles = np.vstack([walls, person_pts])               # "지금 LiDAR 에 보이는 것" 을 흉내
        goal = path_lookahead_point(pose, path, lookahead=0.6)
        # 벽은 정적 점, 사람은 동적 점으로 (sim_demo 는 지도 log-odds 로 이 구분을 자동으로 함)
        (v, w) = dwa_control(pose, v, w, goal, walls, p, dynamic_xy=person_pts)
        pose = pose + np.array([v * np.cos(pose[2]) * p.dt, v * np.sin(pose[2]) * p.dt, w * p.dt])
        pose[2] = wrap_angle(pose[2])
        traj_robot.append(pose[:2].copy()); traj_person.append(person.copy())
        min_clear.append(np.min(np.hypot(obstacles[:, 0] - pose[0], obstacles[:, 1] - pose[1])))
        if np.hypot(pose[0] - path[-1][0], pose[1] - path[-1][1]) < 0.15:
            arrived = True
            break
    traj_robot = np.array(traj_robot); traj_person = np.array(traj_person)
    print(f"{len(traj_robot)} 스텝 {'에 도착' if arrived else '동안 미도착(사람 앞에서 대기 중)'}. 장애물과의 최소 거리 {min(min_clear):.2f} m "
          f"(로봇 반지름 {p.robot_radius} → {'충돌 없음' if min(min_clear) > p.robot_radius else '충돌!'})")
    assert min(min_clear) > p.robot_radius

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(walls[:, 0], walls[:, 1], "k.", ms=3)
    ax.plot([path[0][0], path[1][0]], [path[0][1], path[1][1]], "c--", label="A* 경로")
    ax.plot(traj_robot[:, 0], traj_robot[:, 1], "b-", lw=2, label="DWA 실제 궤적")
    ax.plot(traj_person[:, 0], traj_person[:, 1], "r:", label="사람 궤적")
    for k in range(0, len(traj_robot), 15):
        ax.add_patch(plt.Circle(traj_person[k], 0.25, color="r", alpha=0.15))
        ax.add_patch(plt.Circle(traj_robot[k], p.robot_radius, color="b", alpha=0.15))
    ax.set_aspect("equal"); ax.legend(); ax.set_title("복도에서 길을 가로지르는 사람 회피")
    # 한 순간의 후보 궤적들 시각화
    ax = axes[1]
    k = min(45, len(traj_robot) - 1)
    pose_k = np.array([*traj_robot[k], 0.0])
    person_pts = traj_person[k] + 0.25 * np.column_stack([np.cos(ang), np.sin(ang)])
    (vb, wb), dbg = dwa_control(pose_k, 0.15, 0.0, path_lookahead_point(pose_k, path, 0.6),
                                walls, p, return_debug=True, dynamic_xy=person_pts)
    for i in range(dbg["traj"].shape[0]):
        col = "r" if dbg["collide"][i] else "0.7"
        ax.plot(dbg["traj"][i, :, 0], dbg["traj"][i, :, 1], color=col, lw=0.8)
    if dbg["best"] is not None:
        ax.plot(dbg["traj"][dbg["best"], :, 0], dbg["traj"][dbg["best"], :, 1], "b", lw=3, label="선택된 궤적")
    ax.plot(walls[:, 0], walls[:, 1], "k.", ms=3); ax.plot(person_pts[:, 0], person_pts[:, 1], "r.")
    ax.set_xlim(pose_k[0] - 0.5, pose_k[0] + 1.2); ax.set_ylim(-0.2, 1.8); ax.set_aspect("equal")
    ax.legend(); ax.set_title(f"step {k}: 후보 궤적 (빨강=충돌 탈락, 회색=후보, 파랑=선택)  v={vb:.2f} ω={wb:.2f}")
    plt.tight_layout(); plt.show()
