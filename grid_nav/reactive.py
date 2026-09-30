"""
reactive.py - 강의 슬라이드 110~112쪽 "Decision Making & Reactive Control" 의사코드 구현

Algorithm 1  Simple LiDAR Obstacle Avoidance
Algorithm 2  Yaw-based LiDAR Obstacle Avoidance            (MOVE / TURN 두 상태의 FSM)
Algorithm 3  ... with Corner Escaping Behavior             (TURN 이 반복되면 EscapeBehavior)

왜 여기서 시작하나 (강사님 조언):
  지도도 경로도 없이 LiDAR 세 방향 거리(d_F, d_L, d_R)만으로 로봇이 "혼자 돌아다니게" 만드는 가장 짧은 길.
  이게 되면 (1) 센서/모터/좌표 부호가 맞는지 확인되고 (2) 그 위에 지도 → frontier → A* → DWA 를 한 층씩
  얹을 수 있다. 각 층을 얹을 때마다 이 반응형 제어는 '최후의 안전장치' 로 남겨둔다.

입력은 전부 스칼라라 Webots/시뮬레이터 어디서든 같다:
  d_F, d_L, d_R : 정면/왼쪽/오른쭉 최소 거리 [m]  (lidar_sectors 로 뽑음)
  yaw           : 현재 방향 [rad]  (컴퍼스, 자이로 적분, 또는 위치 추정기)
  battery       : 0~1  (Webots 에 배터리가 없으면 1.0 고정)
출력은 슬라이드처럼 (v_L, v_R) 바퀴 각속도 [rad/s]. (v, ω) 로 바꾸는 헬퍼도 있음.

슬라이드와 다른 점 (주석에 ★):
  ★ |target_yaw − current_yaw| 는 반드시 wrap_angle 을 거쳐야 한다. 슬라이드 의사코드에는 없지만
    실제로 이걸 빠뜨리면 target 이 +190° 같은 값이 되어 로봇이 영원히 돈다.
  ★ EscapeBehavior 와 END_CONDITION 은 슬라이드에 정의가 없어서 "잠깐 후진 → 넓은 쪽으로 크게 회전" 으로 정했다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import wrap_angle


# ======================================================================
# 0. 파라미터와 LiDAR 전처리
# ======================================================================
@dataclass
class ReactiveParams:
    base_speed: float = 5.0            # 바퀴 각속도 [rad/s] (≈0.165 m/s). 강의 teleop 은 3.0, TB3 최대 6.67
    distance_threshold: float = 0.45   # 정면 이 거리 미만이면 회전 [m]
    angle: float = np.deg2rad(60)      # 한 번에 도는 각도 (Alg 2/3 의 ANGLE)
    tolerance: float = np.deg2rad(5)   # 목표 yaw 도달 판정 (TOLERANCE)
    max_cnt: int = 4                   # 연속 TURN 횟수가 이 값을 넘으면 코너에 갇힌 것으로 판단 (MAX_CNT)
    battery_threshold: float = 0.05
    sector: float = np.deg2rad(30)     # d_F/d_L/d_R 를 계산할 각도 폭 (각 방향 ±sector/2)
    escape_back_steps: int = 8         # EscapeBehavior: 후진 스텝 수
    escape_turn: float = np.deg2rad(150)   # EscapeBehavior: 후진 뒤 회전량
    wheel_radius: float = 0.033
    wheel_sep: float = 0.160


def lidar_sectors(ranges, angles, p: ReactiveParams = ReactiveParams(), max_range=3.5):
    """LiDAR 전체 스캔 → (d_F, d_L, d_R). 각 방향 ±sector/2 안의 최소 거리. inf/nan 은 max_range 로.

    각도 규약: 정면 0, 왼쪽 +π/2, 오른쪽 −π/2 (geometry.py 와 동일).
    Webots LDS-01 이면 webots_adapter.lidar_angles() 가 이 규약의 angles 를 만들어 준다.
    최소값을 쓰는 이유: 평균은 좁은 기둥 하나를 놓친다. 안전 판단은 항상 최소로.
    """
    r = np.asarray(ranges, dtype=float)
    r = np.where(np.isfinite(r) & (r > 0.02), r, max_range)
    a = wrap_angle(np.asarray(angles))
    half = p.sector / 2

    def sector_min(center):
        m = np.abs(wrap_angle(a - center)) < half
        return float(r[m].min()) if m.any() else max_range

    return sector_min(0.0), sector_min(np.pi / 2), sector_min(-np.pi / 2)


def wheels_to_vw(v_L, v_R, p: ReactiveParams = ReactiveParams()):
    """바퀴 각속도 → (선속도, 각속도). 슬라이드 148쪽 공식의 역."""
    vl, vr = v_L * p.wheel_radius, v_R * p.wheel_radius
    return 0.5 * (vl + vr), (vr - vl) / p.wheel_sep


def vw_to_wheels(v, w, p: ReactiveParams = ReactiveParams()):
    vl, vr = v - w * p.wheel_sep / 2, v + w * p.wheel_sep / 2
    return vl / p.wheel_radius, vr / p.wheel_radius


# ======================================================================
# Algorithm 1: 정면이 막히면 넓은 쪽으로 제자리 회전
# ======================================================================
class Algorithm1:
    """슬라이드 110쪽. 상태가 없다: 매 스텝 거리만 보고 즉시 결정 (순수 reactive).
    단점: 회전 중에도 매 스텝 d_L, d_R 을 다시 비교하므로 좌/우가 번갈아 뽑히면 제자리에서 떨 수 있다."""

    def __init__(self, p: ReactiveParams = ReactiveParams()):
        self.p = p

    def step(self, d_F, d_L, d_R, battery=1.0):
        p = self.p
        if battery <= p.battery_threshold:             # while battery > BATTERY_THRESHOLD
            return 0.0, 0.0
        if d_F > p.distance_threshold:                 # 앞이 비었으면 직진
            return p.base_speed, p.base_speed
        if d_L > d_R:                                  # 넓은 쪽으로 제자리 회전
            return -p.base_speed, p.base_speed         # 왼쪽으로 (왼바퀴 뒤, 오른바퀴 앞)
        return p.base_speed, -p.base_speed


# ======================================================================
# Algorithm 2: MOVE / TURN 두 상태 FSM. 회전 방향과 목표 yaw 를 '결정한 순간' 고정
# ======================================================================
class Algorithm2:
    """슬라이드 111쪽. Alg 1 의 떨림을 없애기 위해, 막힌 순간 한 번만 방향을 정하고
    target_yaw 에 도달할 때까지 TURN 상태를 유지한다. → 가장 작은 Decision Making(FSM) 예시."""

    def __init__(self, p: ReactiveParams = ReactiveParams()):
        self.p = p
        self.state = "MOVE"
        self.turn_dir = 1
        self.target_yaw = 0.0

    def step(self, d_F, d_L, d_R, yaw, battery=1.0):
        p = self.p
        if battery <= p.battery_threshold:
            return 0.0, 0.0
        if self.state == "MOVE":
            if d_F > p.distance_threshold:
                return p.base_speed, p.base_speed
            self.state = "TURN"
            self.turn_dir = 1 if d_L > d_R else -1
            self.target_yaw = wrap_angle(yaw + self.turn_dir * p.angle)    # ★ wrap
            return 0.0, 0.0
        # TURN
        if abs(wrap_angle(self.target_yaw - yaw)) > p.tolerance:            # ★ wrap
            return -self.turn_dir * p.base_speed, self.turn_dir * p.base_speed
        self.state = "MOVE"
        return 0.0, 0.0


# ======================================================================
# Algorithm 3: Alg 2 + 코너 탈출 (TURN 이 연속 MAX_CNT 번 넘으면 EscapeBehavior)
# ======================================================================
class Algorithm3:
    """슬라이드 112쪽. 좁은 코너에서 좌→우→좌→우 회전만 반복하며 못 빠져나오는 상황(livelock)을
    'turn_cnt' 로 감지하고, 별도 행동(EscapeBehavior) 으로 벗어난다.
    → 강의 152쫙 'Recovery Action' 의 가장 단순한 형태. sim_demo 의 정체 감지/블랙리스트와 같은 역할."""

    def __init__(self, p: ReactiveParams = ReactiveParams()):
        self.p = p
        self.state = "MOVE"            # MOVE / TURN / ESCAPE
        self.turn_dir = 1
        self.target_yaw = 0.0
        self.turn_cnt = 0
        self._escape_phase = 0         # 0: 후진, 1: 회전
        self._escape_steps = 0
        self.escapes = 0               # 통계

    def step(self, d_F, d_L, d_R, yaw, battery=1.0):
        p = self.p
        if battery <= p.battery_threshold:
            return 0.0, 0.0

        if self.state == "ESCAPE":                                   # repeat EscapeBehavior until END_CONDITION
            if self._escape_phase == 0:                              # ★ (정의) 1단계: 잠깐 후진
                self._escape_steps += 1
                if self._escape_steps >= p.escape_back_steps:
                    self._escape_phase = 1
                    self.turn_dir = 1 if d_L > d_R else -1
                    self.target_yaw = wrap_angle(yaw + self.turn_dir * p.escape_turn)
                return -p.base_speed * 0.5, -p.base_speed * 0.5
            if abs(wrap_angle(self.target_yaw - yaw)) > p.tolerance:   # ★ 2단계: 넓은 쪽으로 크게 회전
                return -self.turn_dir * p.base_speed, self.turn_dir * p.base_speed
            # END_CONDITION: 회전 완료 (앞이 열렸는지는 MOVE 가 다음 스텝에 판단)
            self.state, self.turn_cnt = "MOVE", 0
            return 0.0, 0.0

        if self.turn_cnt > p.max_cnt:                                # 코너에 갇힘 → ESCAPE 진입
            self.state = "ESCAPE"
            self._escape_phase, self._escape_steps = 0, 0
            self.escapes += 1
            return 0.0, 0.0

        if self.state == "MOVE":
            if d_F > p.distance_threshold:
                # ★ 슬라이드는 직진할 때 turn_cnt ← 0 이지만, 그러면 "한 칸 전진 → 다시 막힘" 을 반복하는
                #   코너에서 카운터가 매번 리셋되어 ESCAPE 가 절대 안 걸린다. 충분히 전진했을 때만 리셋한다.
                self._free_steps = getattr(self, "_free_steps", 0) + 1
                if self._free_steps > 15:
                    self.turn_cnt = 0
                return p.base_speed, p.base_speed
            self._free_steps = 0
            self.state = "TURN"
            self.turn_cnt += 1
            self.turn_dir = 1 if d_L > d_R else -1
            self.target_yaw = wrap_angle(yaw + self.turn_dir * p.angle)
            return 0.0, 0.0

        # TURN
        if abs(wrap_angle(self.target_yaw - yaw)) > p.tolerance:
            return -self.turn_dir * p.base_speed, self.turn_dir * p.base_speed
        self.state = "MOVE"
        return 0.0, 0.0


# ======================================================================
# 단독 실행 테스트: 가짜 세계에서 세 알고리즘으로 돌아다니기
# ======================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import setup_korean_font
    from sim_demo import build_world, raycast_lidar, Robot
    setup_korean_font()

    p = ReactiveParams()
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    DT, STEPS = 0.1, 1500
    world = build_world(0)

    def run(algo, name):
        rng = np.random.default_rng(1)
        robot = Robot(1.0, 1.0, np.deg2rad(45), radius=0.105, v_max=0.22, w_max=2.84)
        collisions, visited = 0, np.zeros(world.spec.shape, dtype=bool)
        for _ in range(STEPS):
            ranges = raycast_lidar(world, robot.pose, angles, 3.5, rng=rng)
            d_F, d_L, d_R = lidar_sectors(ranges, angles, p)
            if isinstance(algo, Algorithm1):
                vL, vR = algo.step(d_F, d_L, d_R)
            else:
                vL, vR = algo.step(d_F, d_L, d_R, robot.theta)
            v, w = wheels_to_vw(vL, vR, p)
            v = float(np.clip(v, -robot.v_max, robot.v_max)); w = float(np.clip(w, -robot.w_max, robot.w_max))
            nx, ny = robot.x + v * np.cos(robot.theta) * DT, robot.y + v * np.sin(robot.theta) * DT
            if world.collides(nx, ny, robot.radius):
                collisions += 1; v = 0.0
            robot.step(v, w, DT)
            r, c = world.spec.world_to_grid(robot.x, robot.y)
            if world.spec.in_bounds(r, c):
                visited[r, c] = True
        tr = np.array(robot.trail)
        length = float(np.sum(np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1]))))
        # 0.5 m 격자로 뭉쳐서 '몇 군데를 가봤나' (제자리에서 떨면 1~2개에 머문다)
        cells = {(int(x // 0.5), int(y // 0.5)) for x, y in tr}
        esc = getattr(algo, "escapes", 0)
        print(f"{name:12s} 충돌 {collisions:3d}회  이동거리 {length:5.1f} m  가본 0.5m 칸 {len(cells):3d}개  탈출행동 {esc}회")
        return tr

    trails = [run(Algorithm1(p), "Algorithm 1"), run(Algorithm2(p), "Algorithm 2"), run(Algorithm3(p), "Algorithm 3")]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, tr, name in zip(axes, trails, ["Alg 1: 즉시 반응", "Alg 2: MOVE/TURN FSM", "Alg 3: + 코너 탈출"]):
        ax.imshow(world.occ, origin="lower", cmap="gray_r", extent=world.spec.extent)
        ax.plot(tr[:, 0], tr[:, 1], "-", color="tab:orange", lw=0.8)
        ax.plot(tr[0, 0], tr[0, 1], "ks"); ax.plot(tr[-1, 0], tr[-1, 1], "bo")
        ax.set_title(f"{name}  ({STEPS} 스텝 궤적)")
    plt.tight_layout(); plt.show()
