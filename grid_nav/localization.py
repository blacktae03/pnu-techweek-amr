"""
localization.py - 위치 추정: 바퀴 odometry (예측) + LiDAR scan-to-map ICP (보정)

문제: 로봇은 자기 위치를 모른다. 시작 pose 만 안다.
  1) 예측 (Prediction)  : 바퀴가 얼마나 돌았는지(엔코더)로 "아마 여기쯤" 을 적분한다.
                          빠르고 매 스텝 가능하지만, 미끄러짐/바퀴 반지름 오차가 계속 쌓인다(drift).
  2) 보정 (Correction)  : 지금 LiDAR 스캔을 "지금까지 만든 지도의 벽" 에 겹쳐 보고,
                          가장 잘 맞는 위치로 살짝 옮긴다 (ICP). drift 를 주기적으로 지운다.
  이 둘을 합친 것이 강의 슬라이드 74-75쪽의 Pose Prediction(30Hz) → Pose Update(5~10Hz) 사이클.

ICP (Iterative Closest Point) 한 줄 요약:
  스캔 점 각각에 대해 지도에서 가장 가까운 벽 점을 짝지은 뒤, 그 짝들을 가장 잘 겹치게 하는
  회전 R 과 이동 t 를 한 번에 구한다(SVD). 옮긴 뒤 다시 짝짓고 반복. 보통 5~20번이면 수렴.
  회전 구하는 공식(슬라이드 63-72쪽):  H = P'ᵀQ',  U S Vᵀ = svd(H),  R = V Uᵀ

보정 방법은 두 가지를 넣었다:
  ScanMatcher (ICP)            : 교과서 방식. 점-점 짝짓기 + SVD. 이해용. 긴 벽 옆에서는 벽을 따라 "미끄러지는"
                                 방향이 제약이 없어 오차가 늘 수 있다 (__main__ 비교에서 실제로 그렇게 나옴).
  CorrelativeMatcher (기본 권장): odometry 예측 주변에 (dx, dy, dθ) 후보를 격자로 깔고, 각 후보에서 스캔 점이
                                 지도의 벽 위에 얼마나 많이 얹히는지 점수를 매겨 최고점을 고른다. 완전 벡터화.
                                 후보 범위를 작게 두면 "odometry 에서 멀리 못 간다" 는 사전지식이 자동으로 들어가
                                 미끄러짐이 억제된다. Cartographer 등 실전 SLAM 의 기본 아이디어와 같다.

팀 인터페이스: 이 모듈의 출력은 pose = (x, y, theta) [m, rad]. 이것이 occupancy_grid.update() 에 들어간다.

★ 순서 주의 (슬라이드 92쪽): 스캔이 오면  ①odometry 예측 → ②ICP 로 pose 보정 → ③보정된 pose 로 지도 갱신.
  지도를 먼저 갱신하면 틀린 위치에 벽이 찍히고, 다음 ICP 가 그 틀린 벽에 맞추는 악순환이 생긴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import GridSpec, wrap_angle, lidar_to_world, rotation_matrix

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ======================================================================
# 1. 바퀴 odometry (예측)
# ======================================================================
class WheelOdometry:
    """엔코더 누적 회전각(rad) → pose 적분.  TurtleBot3 Burger: r=0.033, L=0.160

      dl = Δφ_l·r,  dr = Δφ_r·r         (각 바퀴가 굴러간 거리)
      ds = (dl+dr)/2,  dθ = (dr−dl)/L    (로봇 중심 이동, 회전)
      x += ds·cos(θ+dθ/2),  y += ds·sin(θ+dθ/2),  θ = wrap(θ+dθ)   (회전 중간값으로 적분 = 2차 정확도)

    dθ 를 바퀴 대신 자이로(z 각속도 × dt)나 컴퍼스(절대각)로 바꾸면 회전 오차가 크게 준다.
    회전 오차 1° 는 5 m 앞에서 9 cm 위치 오차가 되므로, 위치보다 방향을 먼저 잡는 게 이득이 크다.
    """

    def __init__(self, x0, y0, theta0, wheel_radius=0.033, wheel_base=0.160):
        self.x, self.y, self.theta = float(x0), float(y0), float(theta0)
        self.r, self.L = wheel_radius, wheel_base
        self.prev = None

    def update(self, phi_l, phi_r, dtheta_external=None):
        """phi_l/phi_r: 엔코더 누적 각 [rad]. dtheta_external: 자이로 적분 등 외부에서 준 회전량(선택)."""
        if self.prev is None:
            self.prev = (phi_l, phi_r)
            return self.pose
        dl = (phi_l - self.prev[0]) * self.r
        dr = (phi_r - self.prev[1]) * self.r
        self.prev = (phi_l, phi_r)
        ds = 0.5 * (dl + dr)
        dtheta = (dr - dl) / self.L if dtheta_external is None else dtheta_external
        self.x += ds * np.cos(self.theta + 0.5 * dtheta)
        self.y += ds * np.sin(self.theta + 0.5 * dtheta)
        self.theta = wrap_angle(self.theta + dtheta)
        return self.pose

    def set_pose(self, pose):
        """ICP 보정 결과를 되돌려 넣는다 (예측기의 기준점을 옮김)."""
        self.x, self.y, self.theta = float(pose[0]), float(pose[1]), float(pose[2])

    @property
    def pose(self):
        return (self.x, self.y, self.theta)


# ======================================================================
# 2. ICP (두 점집합을 겹치는 강체 변환 찾기)
# ======================================================================
@dataclass
class ICPResult:
    R: np.ndarray          # 2x2 회전
    t: np.ndarray          # (2,) 이동
    iterations: int
    inlier_ratio: float    # 짝을 찾은 스캔 점 비율
    mean_residual: float   # 짝 사이 평균 거리 [m]
    converged: bool

    @property
    def dtheta(self):
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))


def best_rigid_transform(P, Q):
    """P 의 각 점을 Q 의 같은 인덱스 점에 최대한 겹치는 (R, t) 를 닫힌 형태로 구한다 (Kabsch/SVD).
      1) 두 집합의 평균을 빼서 중심을 맞춘다
      2) H = P'ᵀ Q'  (2x2 공분산)
      3) SVD: H = U S Vᵀ,  R = V Uᵀ   (det(R)<0 이면 반사이므로 V 의 마지막 열 부호 반전)
      4) t = mean(Q) − R·mean(P)
    """
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mq - R @ mp
    return R, t


def icp_2d(src, dst, max_iter=20, max_corr_dist=0.5, tol_t=1e-3, tol_theta=1e-3, dst_tree=None):
    """src(N,2) 를 dst(M,2) 에 맞추는 누적 변환 (R, t) 반환.  src_aligned = src @ R.T + t

    max_corr_dist: 이보다 먼 짝은 무시 (스캔에는 있지만 지도에 아직 없는 새 벽, 움직이는 사람 등).
                   너무 크면 엉뚱한 벽에 끌려가고, 너무 작으면 초기 오차가 큰 경우 짝을 못 찾는다.
                   odometry drift 가 스캔 사이에 몇 cm 수준이라면 0.3~0.5 m 가 적당.
    """
    if dst_tree is None:
        dst_tree = cKDTree(dst)
    cur = np.asarray(src, dtype=float).copy()
    R_tot = np.eye(2)
    t_tot = np.zeros(2)
    inlier_ratio, resid, it = 0.0, np.inf, 0
    for it in range(1, max_iter + 1):
        d, idx = dst_tree.query(cur, distance_upper_bound=max_corr_dist)
        ok = np.isfinite(d)
        inlier_ratio = float(ok.mean())
        if ok.sum() < 10:
            return ICPResult(R_tot, t_tot, it, inlier_ratio, np.inf, False)
        R, t = best_rigid_transform(cur[ok], dst[idx[ok]])
        cur = cur @ R.T + t
        R_tot = R @ R_tot
        t_tot = R @ t_tot + t
        resid = float(d[ok].mean())
        if np.linalg.norm(t) < tol_t and abs(np.arctan2(R[1, 0], R[0, 0])) < tol_theta:
            break
    return ICPResult(R_tot, t_tot, it, inlier_ratio, resid, True)


# ======================================================================
# 3. Scan-to-map 위치 보정기
# ======================================================================
class ScanMatcher:
    """지도(0/1/-1) 의 벽 칸들을 참조 점집합으로 삼아 현재 스캔을 ICP 로 맞춘다.

    사용:
        pose_pred = odom.update(...)                       # ① 예측
        pose = matcher.correct(pose_pred, angles, ranges, ternary)   # ② 보정
        odom.set_pose(pose); grid.update(pose, angles, ranges)       # ③ 지도 갱신

    안전장치 (실전에서 이게 절반):
      - 지도에 벽 칸이 min_map_points 미만이면 보정 안 함 (초반엔 지도가 없다)
      - inlier 비율이 낮거나 잔차가 크면 보정 거부 (사람이 많이 보이거나 새 방에 들어갔을 때)
      - 한 번의 보정량이 max_jump / max_jump_theta 를 넘으면 거부 (odometry 가 그 정도로 틀릴 리 없다)
    """

    def __init__(self, spec: GridSpec, every_n=1, min_map_points=200, min_inlier=0.5,
                 max_resid=0.15, max_jump=0.3, max_jump_theta=np.deg2rad(15),
                 max_corr_dist=0.5, beam_stride=2, max_range=3.4):
        self.spec = spec
        self.every_n = every_n
        self.min_map_points = min_map_points
        self.min_inlier = min_inlier
        self.max_resid = max_resid
        self.max_jump = max_jump
        self.max_jump_theta = max_jump_theta
        self.max_corr_dist = max_corr_dist
        self.beam_stride = beam_stride
        self.max_range = max_range
        self.calls = 0
        self.accepted = 0
        self.last: ICPResult | None = None

    def map_points(self, ternary):
        """벽 칸(==1) 의 중심 월드 좌표 (M, 2)."""
        rc = np.argwhere(ternary == 1)
        if rc.shape[0] == 0:
            return np.zeros((0, 2))
        x, y = self.spec.grid_to_world(rc[:, 0], rc[:, 1])
        return np.column_stack([x, y])

    def correct(self, pose_pred, angles, ranges, ternary):
        self.calls += 1
        if self.calls % self.every_n != 0:
            return pose_pred
        dst = self.map_points(ternary)
        if dst.shape[0] < self.min_map_points:
            return pose_pred
        angles = np.asarray(angles)[::self.beam_stride]
        ranges = np.asarray(ranges, dtype=float)[::self.beam_stride]
        ok = np.isfinite(ranges) & (ranges > 0.05) & (ranges < self.max_range)
        if ok.sum() < 30:
            return pose_pred
        src = lidar_to_world(angles[ok], ranges[ok], pose_pred)
        res = icp_2d(src, dst, max_corr_dist=self.max_corr_dist)
        self.last = res
        if not res.converged or res.inlier_ratio < self.min_inlier or res.mean_residual > self.max_resid:
            return pose_pred
        dth = res.dtheta
        new_xy = res.R @ np.array(pose_pred[:2]) + res.t
        jump = np.linalg.norm(new_xy - np.array(pose_pred[:2]))
        if jump > self.max_jump or abs(dth) > self.max_jump_theta:
            return pose_pred
        self.accepted += 1
        return (float(new_xy[0]), float(new_xy[1]), wrap_angle(pose_pred[2] + dth))


# ======================================================================
# 4. Correlative scan matcher (기본 권장)
# ======================================================================
class CorrelativeMatcher:
    """odometry 예측 pose 주변의 (dx, dy, dθ) 후보 격자에서 '스캔이 벽에 가장 잘 얹히는' 후보를 고른다.

    점수: 벽 칸(==1) 을 가우시안으로 살짝 흐린 '가능도 지도(likelihood field)' 를 만들고,
          각 후보 pose 로 옮긴 스캔 점들이 떨어지는 칸의 값을 전부 더한다.
          흐리는 이유: 벽에 1칸 빗나간 점도 부분 점수를 받아 점수 표면이 매끈해진다 (안 흐리면 계단식).
    사전지식: 점수에서 |dx|+|dy| 와 |dθ| 에 비례한 작은 벌점을 뺀다 → 점수가 비슷하면 odometry 쪽을 택함.
             이것이 긴 복도에서 벽을 따라 미끄러지는 현상을 막는 핵심.
    비용: 후보 (11x11x13=1573) × 스캔점 180 ≈ 28만 번의 배열 조회. NumPy 로 수 ms~수십 ms.

    파라미터 감:
      search_xy / step_xy : 스캔 간 odometry 가 틀릴 수 있는 최대 거리. 0.1 s 마다 보정하면 ±0.1 m 로 충분.
      search_th / step_th : 회전 오차 허용. ±3° 정도.
      sigma_cells         : 흐림 정도. 1~1.5 칸.
      min_score           : 이보다 점수(평균 가능도)가 낮으면 '지도와 안 맞음' → 보정 거부.
    """

    def __init__(self, spec: GridSpec, search_xy=0.10, step_xy=0.02, search_th=np.deg2rad(3), step_th=np.deg2rad(0.5),
                 sigma_cells=1.2, prior_xy=0.02, prior_th=0.02, min_score=0.25, min_map_points=200,
                 beam_stride=2, max_range=3.4, every_n=1, l_static=2.0):
        self.spec = spec
        self.l_static = l_static     # log-odds 가 이 이상인 칸만 '확실한 벽' (기본 hit 3번 이상)
        n_xy = int(round(search_xy / step_xy))
        n_th = int(round(search_th / step_th))
        self.dxs = np.arange(-n_xy, n_xy + 1) * step_xy
        self.dths = np.arange(-n_th, n_th + 1) * step_th
        self.sigma_cells = sigma_cells
        self.prior_xy, self.prior_th = prior_xy, prior_th
        self.min_score = min_score
        self.min_map_points = min_map_points
        self.beam_stride = beam_stride
        self.max_range = max_range
        self.every_n = every_n
        self.calls = self.accepted = 0
        self.last_score = 0.0

    def likelihood_field(self, ternary_or_logodds):
        """벽 칸을 1 로 두고 가우시안 블러 → 0~1 로 정규화. 벽 근처일수록 1 에 가까움.

        log-odds 배열(float) 을 주면 L > l_static 인 '여러 번 확인된 벽' 만 기준으로 삼는다.
        왜: 방금 한 번 찍힌 벽(또는 지나가는 사람)은 지금 pose 오차를 그대로 담고 있어서, 거기에 맞추면
            "틀린 지도에 맞춰 더 틀리는" 되먹임이 생긴다. 확실한 벽만 앵커로 쓰면 drift 가 훨씬 덜 쌓인다.
        """
        from scipy import ndimage
        arr = np.asarray(ternary_or_logodds)
        if arr.dtype.kind == "f":
            occ = (arr > self.l_static).astype(np.float32)
        else:
            occ = (arr == 1).astype(np.float32)
        lf = ndimage.gaussian_filter(occ, sigma=self.sigma_cells)
        m = lf.max()
        return lf / m if m > 0 else lf

    def correct(self, pose_pred, angles, ranges, ternary):
        self.calls += 1
        if self.calls % self.every_n != 0:
            return pose_pred
        arr = np.asarray(ternary)
        n_wall = int((arr > self.l_static).sum() if arr.dtype.kind == "f" else (arr == 1).sum())
        if n_wall < self.min_map_points:
            return pose_pred
        angles = np.asarray(angles)[::self.beam_stride]
        ranges = np.asarray(ranges, dtype=float)[::self.beam_stride]
        ok = np.isfinite(ranges) & (ranges > 0.05) & (ranges < self.max_range)
        if ok.sum() < 30:
            return pose_pred
        lf = self.likelihood_field(ternary)
        x, y, th = pose_pred
        a, r = angles[ok], ranges[ok]
        # 후보 회전 θ+dθ 마다 스캔 점의 월드 좌표 (T, N, 2)
        ths = th + self.dths[:, None]                                     # (T, 1)
        px = x + r[None, :] * np.cos(ths + a[None, :])                    # (T, N)
        py = y + r[None, :] * np.sin(ths + a[None, :])
        # 후보 이동 (dx, dy) 를 더해 (T, Dx, Dy, N) 의 격자 인덱스로
        DX = self.dxs[None, :, None, None]
        DY = self.dxs[None, None, :, None]
        rows, cols = self.spec.world_to_grid(px[:, None, None, :] + DX, py[:, None, None, :] + DY)
        inb = self.spec.in_bounds(rows, cols)
        rows, cols = self.spec.clip(rows, cols)
        vals = np.where(inb, lf[rows, cols], 0.0)
        score = vals.mean(axis=-1)                                        # (T, Dx, Dy) 평균 가능도
        # 사전지식 벌점: odometry 예측에서 멀어질수록 조금씩 감점
        pen = (self.prior_xy * (np.abs(DX[..., 0]) + np.abs(DY[..., 0])) / self.dxs.max().clip(1e-9)
               + self.prior_th * np.abs(self.dths)[:, None, None] / max(self.dths.max(), 1e-9))
        total = score - pen
        ti, xi, yi = np.unravel_index(np.argmax(total), total.shape)
        self.last_score = float(score[ti, xi, yi])
        if self.last_score < self.min_score:
            return pose_pred
        self.accepted += 1
        return (float(x + self.dxs[xi]), float(y + self.dxs[yi]), wrap_angle(th + self.dths[ti]))


# ======================================================================
# 단독 실행 테스트: 노이즈 odometry 만 vs odometry + ICP
# ======================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import setup_korean_font
    from occupancy_grid import OccupancyGrid
    from sim_demo import build_world, raycast_lidar, Robot
    setup_korean_font()

    # --- ICP 자체 검증: 점집합을 알고 있는 변환으로 옮긴 뒤 되찾는지 ---
    rng = np.random.default_rng(0)
    Q = rng.uniform(0, 5, (200, 2))
    R_true = rotation_matrix(np.deg2rad(7)); t_true = np.array([0.15, -0.1])
    P = (Q - t_true) @ R_true            # Q = P @ R_true.T + t_true  의 역
    res = icp_2d(P, Q, max_corr_dist=1.0)
    print(f"ICP 검증: 참 회전 7.0°, 추정 {np.rad2deg(res.dtheta):.2f}°  참 이동 {t_true}, 추정 {np.round(res.t, 3)}  "
          f"({res.iterations}회 반복, 잔차 {res.mean_residual * 1000:.1f} mm)")
    assert abs(res.dtheta - np.deg2rad(7)) < 1e-3 and np.allclose(res.t, t_true, atol=2e-3)

    # --- 가짜 세계에서 주행하며 비교 ---
    world = build_world(0)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    spec = GridSpec.from_size(22, 22, 0.05, center=(1.0, 1.0))
    WHEEL_R, WHEEL_L, DT = 0.033, 0.160, 0.1

    # 스크립트된 조종: 직진하다 벽 가까우면 회전 (지도가 어느 정도 만들어지도록 방을 한 바퀴)
    def drive(step, front_min):
        if front_min < 0.6:
            return 0.0, 1.2
        return 0.2, 0.25 * np.sin(step / 40.0)

    true = Robot(1.0, 1.0, np.deg2rad(45))
    # 노이즈 모델: 바퀴 반지름 2% 오차 + 스텝마다 각 바퀴 회전각에 가우시안 노이즈 (미끄러짐)
    R_ERR = 1.02
    phi_l = phi_r = 0.0
    odom_only = WheelOdometry(1.0, 1.0, np.deg2rad(45), WHEEL_R, WHEEL_L)
    odom_icp = WheelOdometry(1.0, 1.0, np.deg2rad(45), WHEEL_R, WHEEL_L)
    odom_cor = WheelOdometry(1.0, 1.0, np.deg2rad(45), WHEEL_R, WHEEL_L)
    grid = OccupancyGrid(spec, max_range=3.5)          # ICP 용 지도
    grid_c = OccupancyGrid(spec, max_range=3.5)        # correlative 용 지도
    matcher = ScanMatcher(spec, every_n=2)
    cmatcher = CorrelativeMatcher(spec)
    traj_true, traj_odom, traj_icp, traj_cor = [], [], [], []
    err_odom, err_icp, err_cor = [], [], []
    import time as _time; t_cor = 0.0
    for step in range(700):
        ranges = raycast_lidar(world, true.pose, angles, 3.5, rng=rng)
        front = np.abs(wrap_angle(angles)) < np.deg2rad(25)
        v, w = drive(step, np.min(ranges[front]))
        # 진짜 이동 + 엔코더 값 생성 (진짜 바퀴 반지름은 WHEEL_R*R_ERR 라고 가정 → 로봇이 생각보다 더 감)
        true.step(v, w, DT)
        vl, vr = v - w * WHEEL_L / 2, v + w * WHEEL_L / 2
        phi_l += vl * DT / (WHEEL_R * R_ERR) + rng.normal(0, 0.02)
        phi_r += vr * DT / (WHEEL_R * R_ERR) + rng.normal(0, 0.02)
        # (a) odometry 만
        p_o = odom_only.update(phi_l, phi_r)
        # (b) odometry 예측 → ICP 보정 → 보정 pose 로 지도 갱신
        p_pred = odom_icp.update(phi_l, phi_r)
        p_c = matcher.correct(p_pred, angles, ranges, grid.to_ternary())
        odom_icp.set_pose(p_c)
        grid.update(p_c, angles, ranges)
        # (c) odometry 예측 → correlative 보정 → 지도 갱신
        p_pred2 = odom_cor.update(phi_l, phi_r)
        _t0 = _time.perf_counter()
        p_k = cmatcher.correct(p_pred2, angles, ranges, grid_c.log_odds)   # 확실한 벽만 기준
        t_cor += _time.perf_counter() - _t0
        odom_cor.set_pose(p_k)
        grid_c.update(p_k, angles, ranges)
        traj_true.append(true.pose[:2]); traj_odom.append(p_o[:2]); traj_icp.append(p_c[:2]); traj_cor.append(p_k[:2])
        err_odom.append(np.hypot(p_o[0] - true.x, p_o[1] - true.y))
        err_icp.append(np.hypot(p_c[0] - true.x, p_c[1] - true.y))
        err_cor.append(np.hypot(p_k[0] - true.x, p_k[1] - true.y))
    print(f"700 스텝 후 위치 오차  odometry만: {err_odom[-1]:.2f} m   odometry+ICP: {err_icp[-1]:.2f} m   "
          f"odometry+correlative: {err_cor[-1]:.2f} m")
    print(f"  평균 오차          odometry만: {np.mean(err_odom):.2f} m   odometry+ICP: {np.mean(err_icp):.2f} m   "
          f"odometry+correlative: {np.mean(err_cor):.2f} m")
    print(f"  ICP 수락 {matcher.accepted}/{matcher.calls // matcher.every_n},  correlative 수락 {cmatcher.accepted}/{cmatcher.calls}, "
          f"correlative 평균 {1000 * t_cor / max(1, cmatcher.calls):.1f} ms/회")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    ax = axes[0]
    ax.imshow(world.occ, origin="lower", cmap="gray_r", extent=world.spec.extent)
    for tr, c, lb in [(traj_true, "g", "진짜"), (traj_odom, "r", "odometry만"), (traj_icp, "b", "odometry+ICP"),
                      (traj_cor, "m", "odometry+correlative")]:
        tr = np.array(tr); ax.plot(tr[:, 0], tr[:, 1], c, lw=1.5, label=lb)
    ax.legend(); ax.set_title("궤적 비교")
    ax = axes[1]
    ax.plot(err_odom, "r", label="odometry만"); ax.plot(err_icp, "b", label="odometry+ICP"); ax.plot(err_cor, "m", label="odometry+correlative")
    ax.set_xlabel("step"); ax.set_ylabel("위치 오차 [m]"); ax.legend(); ax.grid(True); ax.set_title("오차 누적")
    ax = axes[2]
    ax.imshow(grid_c.to_ternary(), origin="lower", cmap="gray_r", vmin=-1, vmax=1, extent=spec.extent)
    ax.set_xlim(world.spec.extent[0] - 0.5, world.spec.extent[1] + 0.5); ax.set_ylim(world.spec.extent[2] - 0.5, world.spec.extent[3] + 0.5)
    ax.set_title("correlative 보정 pose 로 만든 지도 (벽이 겹으로 안 그려져야 정상)")
    plt.tight_layout(); plt.show()
