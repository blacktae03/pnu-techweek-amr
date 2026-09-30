"""
frontier.py - "다음에 어디를 탐색하러 갈까?" 를 정하는 모듈 (Frontier-based Exploration)

Frontier 의 정의:
  '빈칸(0)' 이면서 8방향 이웃 중 '미탐색(-1)' 이 하나라도 있는 칸.
  = 이미 갈 수 있다고 아는 곳과 아직 모르는 곳의 경계선.
  그 경계에 서면 LiDAR 가 미탐색 영역을 비추므로 지도가 넓어집니다. 이걸 반복하면
  갈 수 있는 모든 곳을 다 봤을 때 frontier 가 사라지고 → "탐색 완료".

파이프라인:
  ternary(+inflated) --frontier_mask--> 경계 칸 마스크 --cluster_frontiers--> 덩어리 목록
                     --select_frontier--> 목표 하나 (월드 좌표)

왜 클러스터로 묶나:
  경계선은 수백 칸짜리 선입니다. 칸 하나하나를 목표로 삼으면 로봇이 1칸 이동하고 다시 계획하는
  일을 반복하며 덜덜 떨게 됩니다. 붙어 있는 경계 칸들을 한 덩어리(=하나의 "열린 방향")로
  묶고, 덩어리마다 대표점 하나를 목표 후보로 삼습니다. 덩어리 크기 = 그쪽에 미탐색 영역이
  얼마나 넓게 열려 있는지의 대략적 지표.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from geometry import GridSpec, wrap_angle

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# 8방향 오프셋 (row, col)
_NEIGH8 = [(-1, -1), (-1, 0), (-1, 1),
           (0, -1),           (0, 1),
           (1, -1),  (1, 0),  (1, 1)]


# ----------------------------------------------------------------------
# 1. frontier 칸 찾기 (완전 벡터화)
# ----------------------------------------------------------------------
def frontier_mask(ternary, inflated=None):
    """frontier 칸을 True 로 하는 불리언 배열 반환.

    ternary  : 0/1/-1 지도
    inflated : 같은 형식의 팽창 지도. 주면 팽창된 벽(1) 위의 frontier 는 제외합니다.
               로봇이 실제로 '설 수 없는' 곳을 목표로 잡지 않게 하려는 것.
               (벽 바로 옆의 미탐색 슬리버는 대부분 여기서 걸러짐)

    구현: 미탐색 마스크를 1칸 패딩한 뒤 8방향으로 슬라이스해 OR → "이웃에 미탐색이 있나".
          파이썬 루프는 8번만 돌고 나머지는 전부 NumPy. 400x400 격자에서도 ms 단위.
          패딩값 False = 지도 배열 바깥은 미탐색으로 치지 않음 (지도 끝으로 달려가는 것 방지).
    """
    free = (ternary == 0)
    unknown = (ternary == -1)
    R, C = ternary.shape
    pad = np.pad(unknown, 1, mode="constant", constant_values=False)
    neigh_unknown = np.zeros_like(unknown)
    for dr, dc in _NEIGH8:
        neigh_unknown |= pad[1 + dr:1 + dr + R, 1 + dc:1 + dc + C]
    mask = free & neigh_unknown
    if inflated is not None:
        mask &= (inflated != 1)
    return mask


# ----------------------------------------------------------------------
# 2. 클러스터링
# ----------------------------------------------------------------------
@dataclass
class Frontier:
    cells: np.ndarray                 # (K, 2) int, 각 행 = (row, col)
    size: int                         # K
    centroid_rc: Tuple[float, float]  # 평균 위치 (실수 row, col). 벽 위일 수도 있음
    goal_rc: Tuple[int, int]          # 실제 목표로 쓸 칸 = centroid 에 가장 가까운 frontier 칸
    score: float = field(default=0.0)  # select_frontier 가 채움 (디버그/시각화용)

    def goal_xy(self, spec: GridSpec):
        x, y = spec.grid_to_world(self.goal_rc[0], self.goal_rc[1])
        return float(x), float(y)


def _label_8conn(mask):
    """8-연결 성분 라벨링. scipy 가 있으면 ndimage.label, 없으면 단순 BFS."""
    if _HAS_SCIPY:
        labels, n = _ndi.label(mask, structure=np.ones((3, 3), dtype=int))
        return labels, n
    # --- 대체 구현 (느림) ---
    labels = np.zeros(mask.shape, dtype=int)
    n = 0
    R, C = mask.shape
    for r0, c0 in zip(*np.nonzero(mask)):
        if labels[r0, c0]:
            continue
        n += 1
        stack = [(r0, c0)]
        labels[r0, c0] = n
        while stack:
            r, c = stack.pop()
            for dr, dc in _NEIGH8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < R and 0 <= cc < C and mask[rr, cc] and not labels[rr, cc]:
                    labels[rr, cc] = n
                    stack.append((rr, cc))
    return labels, n


def cluster_frontiers(mask, min_size=5) -> List[Frontier]:
    """frontier 마스크를 덩어리(Frontier) 목록으로. min_size 미만의 작은 덩어리는 버립니다.

    min_size 를 두는 이유: 센서 노이즈나 벽 모서리 샘플링 틈에서 생기는 1~3칸짜리 가짜
    frontier 가 많습니다. 거기까지 가 봐도 얻는 정보가 없으니 무시. 값은 해상도에 따라 조정
    (5cm 격자에서 5칸 = 25cm 폭). 너무 크게 잡으면 좁은 문 너머를 절대 안 탐색하게 됨.
    """
    labels, n = _label_8conn(mask)
    if n == 0:
        return []
    cells_all = np.argwhere(mask)              # (K_total, 2) = (row, col)
    lab_of_cells = labels[mask]                # 각 frontier 칸의 라벨
    out = []
    # np.unique + 정렬 인덱스로 라벨별 그룹을 한 번에 나눔 (라벨마다 mask==k 를 반복하지 않음)
    order = np.argsort(lab_of_cells, kind="stable")
    sorted_labels = lab_of_cells[order]
    boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
    for group in np.split(order, boundaries):
        if group.size < min_size:
            continue
        cells = cells_all[group]
        centroid = cells.mean(axis=0)
        # centroid 는 곡선 모양 frontier 에서 벽 위/미탐색 위에 놓일 수 있음
        # → 실제 frontier 칸 중 centroid 에 가장 가까운 칸을 목표로
        idx = np.argmin(((cells - centroid) ** 2).sum(axis=1))
        goal = (int(cells[idx, 0]), int(cells[idx, 1]))
        out.append(Frontier(cells=cells, size=int(cells.shape[0]),
                            centroid_rc=(float(centroid[0]), float(centroid[1])),
                            goal_rc=goal))
    return out


# ----------------------------------------------------------------------
# 3. 경로 기반 거리 (선택 기능)
# ----------------------------------------------------------------------
def bfs_distance_map(passable, start_rc, max_iters=None):
    """start 에서 각 칸까지의 '격자 홉 수' (8-연결). 벽 너머는 inf.

    직선거리는 "벽 하나 건너 바로 옆" 을 가깝다고 착각합니다. 실제로는 방을 돌아 나가야 하죠.
    BFS 로 실제 이동 거리를 재면 그런 실수가 없습니다.

    구현 트릭: 큐 기반 BFS 대신 "도달 집합을 한 칸씩 팽창" 을 반복. 매 반복은 NumPy 연산
    하나라서 파이썬 루프 횟수 = 최장 경로 홉 수(수백) 정도로 끝납니다.
    (대각선도 1홉으로 세므로 정확한 유클리드 거리는 아니지만 우선순위 비교용으로 충분)
    """
    R, C = passable.shape
    dist = np.full((R, C), np.inf, dtype=np.float32)
    reached = np.zeros((R, C), dtype=bool)
    r0, c0 = int(start_rc[0]), int(start_rc[1])
    if not (0 <= r0 < R and 0 <= c0 < C):
        return dist
    reached[r0, c0] = True
    dist[r0, c0] = 0.0
    if max_iters is None:
        max_iters = R + C
    for k in range(1, max_iters + 1):
        pad = np.pad(reached, 1, constant_values=False)
        grown = reached.copy()
        for dr, dc in _NEIGH8:
            grown |= pad[1 + dr:1 + dr + R, 1 + dc:1 + dc + C]
        new = grown & passable & ~reached
        if not new.any():
            break
        dist[new] = k
        reached |= new
    return dist


# ----------------------------------------------------------------------
# 4. 목표 선택
# ----------------------------------------------------------------------
def select_frontier(frontiers: Sequence[Frontier], robot_pose, spec: GridSpec,
                    blacklist_xy: Optional[Sequence[Tuple[float, float]]] = None,
                    blacklist_radius=0.4,
                    w_dist=1.0, w_size=0.05, size_cap=40, w_turn=0.3,
                    min_goal_dist=0.3,
                    dist_map=None) -> Optional[Frontier]:
    """후보 frontier 중 하나를 고릅니다. 없으면 None (= 탐색 완료 신호).

    점수 (클수록 좋음):
        score = w_size * min(size, size_cap) - w_dist * distance - w_turn * |heading 차|

      - distance   : 로봇→목표 거리 [m]. 가까운 곳부터 훑어야 왔다갔다 하지 않음 (가장 중요).
                     dist_map(bfs_distance_map 결과, 칸 단위) 을 주면 벽을 고려한 경로 거리 사용.
      - size       : 큰 frontier = 넓게 열린 방향 = 정보 이득이 큼. 다만 무한정 커지면
                     거리보다 우선하게 되므로 size_cap 으로 상한.
      - heading 차 : 지금 보고 있는 방향에 가까운 목표를 약간 선호. 회전 시간 절약 + 진동 감소.
    가중치는 실험으로 잡는 값입니다. 데모에서 로봇이 이상하게 움직이면 이 숫자들을 먼저 만져 보세요.

    blacklist_xy : 과거에 도달 실패한 목표들. 그 근처 frontier 는 건너뜀 (같은 곳에 계속 시도 방지).
    min_goal_dist: 로봇 발밑의 frontier 는 이미 보고 있는 셈이므로 제외.
    """
    if not frontiers:
        return None
    rx, ry, rth = robot_pose
    best, best_score = None, -np.inf
    for f in frontiers:
        gx, gy = f.goal_xy(spec)
        # --- 블랙리스트 ---
        if blacklist_xy:
            bl = np.asarray(blacklist_xy, dtype=float).reshape(-1, 2)
            if np.any(np.hypot(bl[:, 0] - gx, bl[:, 1] - gy) < blacklist_radius):
                continue
        # --- 거리 ---
        if dist_map is not None:
            d_cells = float(dist_map[f.goal_rc])
            if not np.isfinite(d_cells):
                continue                     # 도달 불가 (벽으로 막힘)
            dist = d_cells * spec.resolution
        else:
            dist = float(np.hypot(gx - rx, gy - ry))
        if dist < min_goal_dist:
            continue
        # --- 방향 차 ---
        heading_err = abs(wrap_angle(np.arctan2(gy - ry, gx - rx) - rth))
        score = (w_size * min(f.size, size_cap)
                 - w_dist * dist
                 - w_turn * heading_err)
        f.score = score
        if score > best_score:
            best, best_score = f, score
    return best


# ----------------------------------------------------------------------
# 단독 실행 테스트 + 시각화
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from geometry import setup_korean_font
    setup_korean_font()
    from occupancy_grid import inflate

    # 손으로 만든 부분 탐색 지도: 방 하나를 봤고, 문 두 개 너머는 미탐색
    spec = GridSpec.from_size(8.0, 6.0, resolution=0.05, center=(4.0, 3.0))
    tern = np.full(spec.shape, -1, dtype=np.int8)
    r, c = spec.world_to_grid(np.array([1.0, 5.0]), np.array([1.0, 4.0]))
    tern[r[0]:r[1], c[0]:c[1]] = 0          # 탐색된 빈 공간
    tern[r[0] - 1, c[0] - 1:c[1] + 1] = 1   # 아래 벽
    tern[r[1], c[0] - 1:c[1] + 1] = 1       # 위 벽 ...
    tern[r[0] - 1:r[1] + 1, c[0] - 1] = 1   # 왼 벽
    tern[r[0] - 1:r[1] + 1, c[1]] = 1       # 오른 벽
    # 문 두 개: 오른쪽 벽 위쪽에 큰 문, 위쪽 벽에 작은 문
    dr_ = spec.world_to_grid(0, np.array([2.8, 3.8]))[0]
    tern[dr_[0]:dr_[1], c[1]] = 0
    dc_ = spec.world_to_grid(np.array([2.0, 2.3]), 0)[1]
    tern[r[1], dc_[0]:dc_[1]] = 0
    # 노이즈: 방 안에 미탐색 점 몇 개 (작은 가짜 frontier 생성 → min_size 로 걸러져야 함)
    tern[r[0] + 10, c[0] + 10] = -1
    tern[r[0] + 30, c[0] + 50] = -1

    inflated = inflate(tern, radius_m=0.25, resolution=spec.resolution)
    mask = frontier_mask(tern, inflated)
    frontiers = cluster_frontiers(mask, min_size=5)
    pose = (2.0, 2.0, 0.0)
    print(f"frontier 칸 {int(mask.sum())}개, 클러스터 {len(frontiers)}개 (min_size 적용 후)")

    # 직선거리 기준 vs 경로거리 기준 선택 비교
    best_euclid = select_frontier(frontiers, pose, spec)
    rr0, cc0 = spec.world_to_grid(pose[0], pose[1])
    dmap = bfs_distance_map(inflated == 0, (rr0, cc0))
    best_path = select_frontier(frontiers, pose, spec, dist_map=dmap)
    for f in frontiers:
        print(f"  size={f.size:4d}  goal_xy={tuple(round(v, 2) for v in f.goal_xy(spec))}  score={f.score:+.2f}")
    print("선택(직선거리):", best_euclid.goal_xy(spec) if best_euclid else None)
    print("선택(경로거리):", best_path.goal_xy(spec) if best_path else None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.imshow(tern, origin="lower", cmap="gray_r", vmin=-1, vmax=1, extent=spec.extent)
    fr = np.argwhere(mask)
    fx, fy = spec.grid_to_world(fr[:, 0], fr[:, 1])
    ax.scatter(fx, fy, s=4, c="cyan", label="frontier 칸")
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, f in enumerate(frontiers):
        gx, gy = f.goal_xy(spec)
        ax.plot(gx, gy, "*", ms=14, color=colors[i % 10], label=f"cluster {i} (size {f.size})")
    ax.plot(pose[0], pose[1], "ro", ms=8, label="robot")
    if best_euclid:
        gx, gy = best_euclid.goal_xy(spec)
        ax.plot([pose[0], gx], [pose[1], gy], "r--", label="선택된 목표")
    ax.set_title("삼진 지도 + frontier + 클러스터 대표점")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    d_show = np.where(np.isfinite(dmap), dmap, np.nan)
    im = ax.imshow(d_show, origin="lower", cmap="viridis", extent=spec.extent)
    ax.set_title("BFS 경로 거리 (홉 수)  – 벽 너머는 공백")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
