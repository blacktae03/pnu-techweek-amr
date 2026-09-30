"""
occupancy_grid.py - LiDAR 로 점유 격자 지도(Occupancy Grid Map) 만들기

핵심 아이디어 (log-odds 누적):
  각 칸마다 "벽일 확률" 을 들고 있되, 확률 대신 log-odds  L = log(p / (1-p)) 로 저장합니다.
  - 빔이 칸을 통과했다  → L += l_miss  (음수, 빈칸 증거)
  - 빔이 칸에서 멈췄다  → L += l_hit   (양수, 벽 증거)
  확률을 곱하는 대신 log 를 더하면 되어서 NumPy 로 한 번에 처리하기 좋고,
  센서 노이즈로 한두 번 잘못 찍혀도 여러 번 관측하면 스스로 교정됩니다.
  L = 0 은 "아무 정보 없음" = 미탐색.

외부 인터페이스 (팀 약속):
  to_ternary() -> int8 배열,  0 빈칸 / 1 벽 / -1 미탐색

이 파일의 두 가지 업데이트 구현:
  update()        : NumPy 벡터화 (빔을 따라 점을 촘촘히 찍고 칸 인덱스로 변환). 실전용.
  update_bresenham(): 빔마다 Bresenham 으로 칸을 구하는 교과서 방식. 느리지만 이해용.
  두 결과는 거의 같습니다 (__main__ 에서 비교).
"""
from __future__ import annotations

import numpy as np

from geometry import GridSpec, bresenham_line, lidar_to_robot, robot_to_world

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except ImportError:            # scipy 가 없어도 돌아가게 (느린 NumPy 대체 구현 사용)
    _HAS_SCIPY = False


def _logit(p):
    return float(np.log(p / (1.0 - p)))


class OccupancyGrid:
    def __init__(self, spec: GridSpec,
                 p_hit=0.70, p_miss=0.35,
                 occ_thresh=0.60, free_thresh=0.45,
                 l_clamp=4.0,
                 max_range=6.0, min_range=0.05):
        """
        p_hit / p_miss : 한 번의 hit / miss 관측이 주는 확률 증거. 0.5 에서 멀수록 확신이 빠름.
                         기본값은 hit 한 번이면 벽(0.70 > 0.60), miss 한 번이면 빈칸(0.35 < 0.45)
                         이 되도록 잡았습니다. 지도가 빨리 채워져야 탐색이 빨라지기 때문.
                         노이즈가 심하면 0.5 쪽으로 (예: 0.6 / 0.4) 조정.
        l_clamp        : log-odds 절대값 상한. 없으면 벽을 1000번 본 뒤 사람이 지나가도
                         절대 안 바뀌는 "고집스러운 지도" 가 됨. 움직이는 사람 대응에 중요.
        max_range      : 이 거리 이상(또는 inf)은 "벽을 못 봤다" 로 처리하고 max_range 근처까지만 빈칸.
        min_range      : 이보다 짧은 값은 로봇 몸체/노이즈로 보고 버림.
        """
        self.spec = spec
        self.log_odds = np.zeros(spec.shape, dtype=np.float32)
        self.l_hit = _logit(p_hit)
        self.l_miss = _logit(p_miss)
        self.l_occ = _logit(occ_thresh)
        self.l_free = _logit(free_thresh)
        self.l_clamp = float(l_clamp)
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        # 지도 밖으로 떨어진 벽 측정값 개수. 이게 계속 늘면 지도(GridSpec) 가 너무 작다는 신호!
        # (지도 밖 frontier 는 조용히 사라지므로, 탐색이 이상하게 일찍 끝나면 먼저 확인할 것)
        self.dropped_hits = 0

    # ------------------------------------------------------------------
    # 측정값 전처리: inf / nan / 너무 짧은 값 처리
    # ------------------------------------------------------------------
    def _prepare_beams(self, pose, angles, ranges):
        """각 빔에 대해 (월드 끝점, hit 여부) 를 구합니다.

        - range 가 finite 이고 max_range 미만  → hit 빔. 끝점 칸은 벽, 그 앞은 빈칸.
        - range 가 inf/nan 또는 >= max_range → no-hit 빔. max_range*0.98 까지 빈칸만 표시.
          (0.98 을 곱하는 이유: 정확히 max_range 위치 칸을 벽으로 오해하는 일을 막기 위해
           끝점 자체는 아무것도 표시하지 않고, 조금 안쪽까지만 빈칸으로 둠)
        - range < min_range → 버림 (로봇 몸체에 맞은 빔, 0 으로 오는 이상값 등)
        """
        angles = np.asarray(angles, dtype=float)
        ranges = np.asarray(ranges, dtype=float)
        finite = np.isfinite(ranges)
        too_close = finite & (ranges < self.min_range)
        keep = ~too_close
        angles, ranges, finite = angles[keep], ranges[keep], finite[keep]

        hit = finite & (ranges < self.max_range)
        r_eff = np.where(hit, ranges, self.max_range * 0.98)

        end_world = robot_to_world(lidar_to_robot(angles, r_eff), pose)
        return end_world, r_eff, hit

    # ------------------------------------------------------------------
    # 업데이트 (벡터화 버전) - 실전용
    # ------------------------------------------------------------------
    def update(self, pose, angles, ranges):
        """한 번의 스캔으로 지도를 갱신합니다. pose=(x,y,theta), angles/ranges 는 같은 길이.

        벡터화 전략:
          빔마다 Bresenham 을 돌리는 대신, 로봇 위치에서 끝점까지 t=0..1 로 S개의 점을 찍고
          (N빔 x S점 x 2) 배열을 한 번에 격자 인덱스로 바꿉니다. 점 간격을 해상도의 절반으로
          잡으면 지나가는 칸을 거의 빠뜨리지 않습니다. 그 뒤 np.unique 로 중복 칸을 합쳐서
          한 스캔에서 같은 칸이 여러 번 깎이지 않게 합니다 (안 그러면 로봇 바로 옆 칸은
          360개 빔이 전부 지나가므로 한 번에 -100 이 되어 버림).
        """
        spec = self.spec
        end_world, r_eff, hit = self._prepare_beams(pose, angles, ranges)
        if end_world.shape[0] == 0:
            return
        x0, y0 = float(pose[0]), float(pose[1])

        # --- 빔을 따라 점 찍기 -------------------------------------------------
        n_steps = int(np.ceil(self.max_range / (spec.resolution * 0.5))) + 1
        t = np.linspace(0.0, 1.0, n_steps)                     # (S,)
        start = np.array([x0, y0])
        pts = start + (end_world - start)[:, None, :] * t[None, :, None]   # (N, S, 2)

        # 빔 위의 각 점이 시작점에서 얼마나 떨어져 있나 (m)
        dist_along = r_eff[:, None] * t[None, :]               # (N, S)
        # hit 빔은 끝점 근처(한 칸 이내)는 빈칸으로 찍지 않음 → 그 칸은 벽 후보
        # no-hit 빔은 끝까지 전부 빈칸
        free_ok = np.where(hit[:, None],
                           dist_along < (r_eff[:, None] - spec.resolution),
                           np.ones_like(dist_along, dtype=bool))

        rows, cols = spec.world_to_grid(pts[..., 0], pts[..., 1])
        inb = spec.in_bounds(rows, cols)
        sel = free_ok & inb
        free_flat = np.unique(rows[sel] * spec.cols + cols[sel])   # 평탄 인덱스로 중복 제거

        # --- 끝점(벽) 칸 -----------------------------------------------------
        hr, hc = spec.world_to_grid(end_world[hit, 0], end_world[hit, 1])
        hin = spec.in_bounds(hr, hc)
        self.dropped_hits += int((~hin).sum())
        hit_flat = np.unique(hr[hin] * spec.cols + hc[hin])

        # 같은 스캔에서 벽으로 찍힌 칸은 빈칸 감점에서 제외 (스치는 빔이 벽을 지우는 것 방지)
        free_flat = np.setdiff1d(free_flat, hit_flat, assume_unique=True)

        L = self.log_odds.ravel()          # view: 여기 쓰면 원본 배열이 바뀜
        L[free_flat] += self.l_miss
        L[hit_flat] += self.l_hit
        np.clip(self.log_odds, -self.l_clamp, self.l_clamp, out=self.log_odds)

    # ------------------------------------------------------------------
    # 업데이트 (Bresenham 루프 버전) - 이해용
    # ------------------------------------------------------------------
    def update_bresenham(self, pose, angles, ranges):
        """update() 와 같은 일을 빔마다 Bresenham 으로 처리. 교과서 그대로의 방식.
        360빔 기준 파이썬 루프 360번 → 벡터화 버전보다 수십 배 느림 (__main__ 에서 시간 비교)."""
        spec = self.spec
        end_world, r_eff, hit = self._prepare_beams(pose, angles, ranges)
        r0, c0 = spec.world_to_grid(pose[0], pose[1])
        touched_free, touched_hit = set(), set()
        for (ex, ey), is_hit in zip(end_world, hit):
            r1, c1 = spec.world_to_grid(ex, ey)
            rr, cc = bresenham_line(r0, c0, r1, c1)
            if is_hit:
                cells_free = zip(rr[:-1].tolist(), cc[:-1].tolist())   # 마지막 칸은 벽
                if spec.in_bounds(rr[-1], cc[-1]):
                    touched_hit.add((int(rr[-1]), int(cc[-1])))
            else:
                cells_free = zip(rr.tolist(), cc.tolist())
            for rc in cells_free:
                if 0 <= rc[0] < spec.rows and 0 <= rc[1] < spec.cols:
                    touched_free.add(rc)
        touched_free -= touched_hit
        for r, c in touched_free:
            self.log_odds[r, c] += self.l_miss
        for r, c in touched_hit:
            self.log_odds[r, c] += self.l_hit
        np.clip(self.log_odds, -self.l_clamp, self.l_clamp, out=self.log_odds)

    # ------------------------------------------------------------------
    # 외부 형식으로 내보내기
    # ------------------------------------------------------------------
    def to_ternary(self):
        """팀 인터페이스 형식: int8 배열, 0 빈칸 / 1 벽 / -1 미탐색."""
        out = np.full(self.spec.shape, -1, dtype=np.int8)
        out[self.log_odds > self.l_occ] = 1
        out[self.log_odds < self.l_free] = 0
        return out

    def probability(self):
        """시각화용: 각 칸의 점유 확률 (0~1). p = 1 / (1 + exp(-L))"""
        return 1.0 / (1.0 + np.exp(-self.log_odds))


# ----------------------------------------------------------------------
# Inflation (장애물 팽창)
# ----------------------------------------------------------------------
def disk_structure(radius_cells: int):
    """반지름 radius_cells 인 원형 커널(불리언). 팽창(dilation) 의 '도장' 역할."""
    r = int(radius_cells)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def _dilate_numpy(mask, structure):
    """scipy 없을 때의 대체 팽창. 커널의 각 오프셋만큼 mask 를 밀어서 OR.
    = "벽 칸 하나를 커널 모양 도장으로 찍는다" 를 오프셋별로 한 번씩 하는 것."""
    out = np.zeros_like(mask, dtype=bool)
    R, C = mask.shape
    kr, kc = structure.shape
    cr, cc = kr // 2, kc // 2
    for i in range(kr):
        for j in range(kc):
            if not structure[i, j]:
                continue
            dr, dc = i - cr, j - cc
            # out[r, c] |= mask[r - dr, c - dc]  를 슬라이스로
            rs, re = max(0, dr), min(R, R + dr)
            cs, ce = max(0, dc), min(C, C + dc)
            out[rs:re, cs:ce] |= mask[rs - dr:re - dr, cs - dc:ce - dc]
    return out


def inflate(ternary, radius_m, resolution, unknown_as_obstacle=False):
    """벽을 반지름 radius_m 만큼 두껍게 만듭니다 (Configuration space 변환).

    왜: 경로 계획에서 로봇을 '점' 으로 취급하는 대신, 벽을 로봇 반지름(+안전 여유)만큼
        키워 두면 점이 지나갈 수 있는 곳 = 로봇 몸체가 안 부딛는 곳이 됩니다.
        radius_m = 로봇 반지름 + 안전 여유 (예: 0.2 + 0.1). 심사 항목 "안전거리 확보" 와 직결.

    반환: 입력과 같은 형식 (0/1/-1). 벽 근처 칸은 원래 값이 무엇이든 1 로 덮어씁니다.
          (미탐색 칸도 벽 근처면 1 이 되므로, 벽에 붙은 frontier 가 자동으로 걸러지는 부수 효과)
    unknown_as_obstacle=True 면 미탐색 칸도 벽처럼 팽창 (보수적으로 갈 때).
    """
    occ = (ternary == 1)
    if unknown_as_obstacle:
        occ |= (ternary == -1)
    r_cells = int(np.ceil(radius_m / resolution))
    if r_cells <= 0:
        return ternary.copy()
    st = disk_structure(r_cells)
    if _HAS_SCIPY:
        grown = _ndi.binary_dilation(occ, structure=st)
    else:
        grown = _dilate_numpy(occ, st)
    out = ternary.copy()
    out[grown] = 1
    return out


def obstacle_distance_map(ternary_or_inflated, resolution):
    """각 칸에서 가장 가까운 벽(값 1)까지의 거리 [m]. A* 에서 '벽 근처 비용' 을 줄 때 사용.
    scipy 의 distance_transform_edt: 0 이 아닌 칸에 대해 가장 가까운 0 칸까지의 유클리드 거리.
    그래서 벽이 아닌 칸을 True(≠0) 로 넘깁니다."""
    not_wall = (ternary_or_inflated != 1)
    if _HAS_SCIPY:
        d = _ndi.distance_transform_edt(not_wall)
    else:
        # 느린 대체: 반복 팽창으로 홉 거리 계산 (정확한 유클리드는 아님)
        d = np.full(not_wall.shape, np.inf)
        reached = ~not_wall
        d[reached] = 0
        st = disk_structure(1)
        k = 0
        while not reached.all() and k < max(not_wall.shape):
            k += 1
            new = _dilate_numpy(reached, st) & ~reached
            d[new] = k
            reached |= new
    return d * resolution


# ----------------------------------------------------------------------
# 단독 실행 테스트 + 시각화
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from geometry import setup_korean_font
    setup_korean_font()

    # 아주 단순한 진짜 세계: 6m x 6m 방, 가운데 기둥 하나 (격자 True = 벽)
    spec = GridSpec.from_size(8.0, 8.0, resolution=0.05, center=(3.0, 3.0))
    world = np.zeros(spec.shape, dtype=bool)
    R, C = spec.world_to_grid(np.array([0.0, 6.0]), np.array([0.0, 6.0]))
    world[R[0], C[0]:C[1] + 1] = True; world[R[1], C[0]:C[1] + 1] = True
    world[R[0]:R[1] + 1, C[0]] = True; world[R[0]:R[1] + 1, C[1]] = True
    pr, pc = spec.world_to_grid(np.array([3.5, 4.0]), np.array([2.0, 2.5]))
    world[pr[0]:pr[1], pc[0]:pc[1]] = True

    # 가짜 LiDAR: sim_demo 의 raycast 를 재사용하면 순환 의존이 되니 여기선 간단히 구현
    def fake_lidar(pose, angles, max_range=6.0):
        step = spec.resolution * 0.5
        ts = np.arange(0, max_range, step)
        dirs = angles + pose[2]
        xs = pose[0] + ts[None, :] * np.cos(dirs)[:, None]
        ys = pose[1] + ts[None, :] * np.sin(dirs)[:, None]
        rr, cc = spec.world_to_grid(xs, ys)
        inb = spec.in_bounds(rr, cc)
        hit = np.zeros(rr.shape, dtype=bool)
        hit[inb] = world[rr[inb], cc[inb]]
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        ranges = np.where(any_hit, ts[first], np.inf)
        return ranges + np.random.normal(0, 0.01, ranges.shape)

    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    poses = [(1.0, 1.0, 0.0), (1.5, 3.0, 0.5), (2.5, 4.5, -1.0), (4.5, 4.5, 2.0), (5.0, 1.0, 3.0)]

    grid_fast = OccupancyGrid(spec, max_range=6.0)
    grid_slow = OccupancyGrid(spec, max_range=6.0)
    t_fast = t_slow = 0.0
    np.random.seed(0)
    for pose in poses:
        rng = fake_lidar(pose, angles)
        t0 = time.perf_counter(); grid_fast.update(pose, angles, rng); t_fast += time.perf_counter() - t0
        t0 = time.perf_counter(); grid_slow.update_bresenham(pose, angles, rng); t_slow += time.perf_counter() - t0

    tern = grid_fast.to_ternary()
    tern_slow = grid_slow.to_ternary()
    agree = np.mean(tern == tern_slow)
    print(f"벡터화 update  : {t_fast / len(poses) * 1000:.1f} ms/스캔")
    print(f"Bresenham update: {t_slow / len(poses) * 1000:.1f} ms/스캔")
    print(f"두 방식 결과 일치율: {agree * 100:.1f}%  (샘플링 방식 차이로 100% 는 아님)")
    print("칸 통계  빈칸:", int((tern == 0).sum()), " 벽:", int((tern == 1).sum()),
          " 미탐색:", int((tern == -1).sum()))

    inflated = inflate(tern, radius_m=0.3, resolution=spec.resolution)
    dist = obstacle_distance_map(inflated, spec.resolution)

    # inf 처리 확인: 문(구멍)이 없어도 max_range 밖은 미탐색으로 남아야 함
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(world, origin="lower", cmap="gray_r", extent=spec.extent)
    axes[0].set_title("진짜 세계 (검정=벽)")
    for p in poses:
        axes[0].plot(p[0], p[1], "ro")
    axes[1].imshow(grid_fast.probability(), origin="lower", cmap="gray_r", vmin=0, vmax=1, extent=spec.extent)
    axes[1].set_title("점유 확률 (log-odds→p)")
    axes[2].imshow(tern, origin="lower", cmap="gray_r", vmin=-1, vmax=1, extent=spec.extent)
    axes[2].set_title("삼진 지도 (-1 미탐색 / 0 빈칸 / 1 벽)")
    axes[3].imshow(inflated, origin="lower", cmap="gray_r", vmin=-1, vmax=1, extent=spec.extent)
    axes[3].set_title("팽창 후 (radius 0.3m)")
    for ax in axes:
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    plt.tight_layout()
    plt.show()
