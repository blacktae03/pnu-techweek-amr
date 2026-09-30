"""
astar.py - 팽창된 격자 위에서 A* 로 경로 찾기 (Global Path Planning)

A* 한 줄 요약:
  Dijkstra(가까운 칸부터 퍼져나감) 에 "목표까지 남은 거리 추정치 h" 를 더해서
  목표 쪽으로 먼저 뻗어 나가게 만든 것.  우선순위 f = g(지금까지 비용) + h(추정 잔여비용).
  h 가 실제 잔여비용을 절대 과대평가하지 않으면(admissible) 최단 경로를 보장합니다.
  8방향 이동에서는 'octile 거리' 가 그런 h 입니다.

이 파일의 특징:
  - 8방향 이동, 대각선 비용 sqrt(2)
  - 코너 커팅 금지: 대각선으로 갈 때 양옆 직교 칸 중 하나라도 벽이면 금지
    (점 로봇이 벽 모서리를 스치며 지나가는 걸 막음)
  - 벽 근접 비용: 벽에 가까운 칸은 통과 비용을 올려서, 갈 수 있어도 되도록 복도 한가운데로
    가게 만듦 → 심사 항목 "안전거리 확보"
  - 결과는 (row, col) 리스트 → 단순화/직선화 → 월드 (x, y) 리스트로 반환 (팀 인터페이스)
"""
from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

from geometry import GridSpec, bresenham_line
from occupancy_grid import obstacle_distance_map

SQRT2 = float(np.sqrt(2.0))
# (dr, dc, 이동 비용)
_MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
          (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


# ----------------------------------------------------------------------
# 1. 비용 지도
# ----------------------------------------------------------------------
def build_cost_map(inflated, resolution, safe_dist=0.5, w_prox=3.0):
    """벽 근처 추가 비용 (0 ~ w_prox). 팽창된 벽에서 safe_dist [m] 이상 떨어지면 0.

    형태: cost = w_prox * (1 - d / safe_dist)  (d < safe_dist 일 때),  선형으로 줄어듦.
    A* 에서 한 칸 이동 비용 = 기하 거리 * (1 + cost). 즉 벽에 딱 붙은 칸은 (1+w_prox) 배 비쌈.
    w_prox 를 키우면 더 멀리 돌아가더라도 벽을 피하고, 0 이면 순수 최단 경로.
    """
    d = obstacle_distance_map(inflated, resolution)          # [m]
    prox = np.clip(1.0 - d / safe_dist, 0.0, 1.0)
    return (w_prox * prox).astype(np.float32)


# ----------------------------------------------------------------------
# 2. A* 본체 (격자 인덱스 기준)
# ----------------------------------------------------------------------
def _octile(r0, c0, r1, c1):
    dr, dc = abs(r0 - r1), abs(c0 - c1)
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)


def astar_grid(passable, start_rc, goal_rc, cost_map=None, max_expansions=None
               ) -> Optional[List[Tuple[int, int]]]:
    """passable(불리언 2D) 위에서 start→goal 격자 경로. 못 찾으면 None.

    자료구조:
      open   : heapq (f, tie, r, c). tie 는 f 가 같을 때 비교 오류 방지용 일련번호.
      g      : 2D float 배열 (inf 초기화). 딕셔너리보다 빠르고 코드가 단순.
      parent : 2D int 배열, 평탄 인덱스로 이전 칸 저장 → 경로 복원.
      closed : 2D bool. 이미 확정된 칸을 다시 펼치지 않음.

    성능: 순수 파이썬 루프라 40만 칸 지도 전체를 탐색하면 수 초가 걸릴 수 있음.
          보통은 목표가 가까워 수천~수만 칸만 펼치므로 수십~수백 ms.
          느리면 (1) 지도 해상도를 0.1m 로 (2) max_expansions 제한 (3) 벽 근접 비용을 줄여
          h 와 g 의 스케일을 맞추기 (비용을 키우면 A* 가 Dijkstra 에 가까워져 느려짐).
    """
    R, C = passable.shape
    sr, sc = int(start_rc[0]), int(start_rc[1])
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    if not (0 <= sr < R and 0 <= sc < C and 0 <= gr < R and 0 <= gc < C):
        return None
    if not passable[sr, sc] or not passable[gr, gc]:
        return None
    if cost_map is None:
        cost_map = np.zeros((R, C), dtype=np.float32)

    g = np.full((R, C), np.inf, dtype=np.float64)
    parent = np.full((R, C), -1, dtype=np.int64)
    closed = np.zeros((R, C), dtype=bool)
    g[sr, sc] = 0.0
    tie = 0
    open_heap = [(_octile(sr, sc, gr, gc), tie, sr, sc)]
    expansions = 0

    while open_heap:
        f, _, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue                 # 오래된 중복 항목 (더 좋은 g 로 이미 처리됨)
        closed[r, c] = True
        if r == gr and c == gc:
            break
        expansions += 1
        if max_expansions is not None and expansions > max_expansions:
            return None
        for dr, dc, step in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < R and 0 <= nc < C):
                continue
            if not passable[nr, nc] or closed[nr, nc]:
                continue
            if dr != 0 and dc != 0:
                # 코너 커팅 금지: 대각선 이동 시 양옆 직교 칸이 모두 통과 가능해야 함
                if not passable[r + dr, c] or not passable[r, c + dc]:
                    continue
            ng = g[r, c] + step * (1.0 + float(cost_map[nr, nc]))
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = r * C + c
                tie += 1
                heapq.heappush(open_heap, (ng + _octile(nr, nc, gr, gc), tie, nr, nc))
    else:
        return None                  # open 이 비었는데 목표 미도달

    if not closed[gr, gc]:
        return None
    # --- 경로 복원 (goal → start 를 parent 로 거슬러 올라간 뒤 뒤집기) ---
    path = []
    cur = gr * C + gc
    while cur != -1:
        path.append((cur // C, cur % C))
        cur = parent[cur // C, cur % C]
    path.reverse()
    return path


# ----------------------------------------------------------------------
# 3. 도우미: 통과 불가한 목표/시작을 근처 통과 가능 칸으로 옮기기
# ----------------------------------------------------------------------
def nearest_passable(passable, rc, max_radius_cells=12):
    """rc 가 통과 불가면 반경을 1칸씩 늘리며 가장 가까운 통과 가능 칸을 찾음. 없으면 None.

    필요한 상황:
      - 로봇 자신이 벽에 붙어 있어서 로봇 칸이 팽창 벽 안에 들어간 경우 (시작점)
      - 구조 대상이 벽에 붙어 있어 대상 칸 자체는 못 가는 경우 (목표점)
      - frontier 대표점이 팽창 경계에 걸친 경우
    """
    R, C = passable.shape
    r0, c0 = int(rc[0]), int(rc[1])
    if 0 <= r0 < R and 0 <= c0 < C and passable[r0, c0]:
        return (r0, c0)
    for rad in range(1, max_radius_cells + 1):
        rs, re = max(0, r0 - rad), min(R, r0 + rad + 1)
        cs, ce = max(0, c0 - rad), min(C, c0 + rad + 1)
        sub = passable[rs:re, cs:ce]
        if not sub.any():
            continue
        cand = np.argwhere(sub) + np.array([rs, cs])
        d2 = ((cand - np.array([r0, c0])) ** 2).sum(axis=1)
        best = cand[np.argmin(d2)]
        return (int(best[0]), int(best[1]))
    return None


# ----------------------------------------------------------------------
# 4. 경로 후처리
# ----------------------------------------------------------------------
def simplify_collinear(path_rc):
    """같은 방향으로 계속 가는 중간 점 제거. (0,0),(0,1),(0,2),(1,3) → (0,0),(0,2),(1,3)"""
    if len(path_rc) <= 2:
        return list(path_rc)
    out = [path_rc[0]]
    for i in range(1, len(path_rc) - 1):
        d_prev = (path_rc[i][0] - path_rc[i - 1][0], path_rc[i][1] - path_rc[i - 1][1])
        d_next = (path_rc[i + 1][0] - path_rc[i][0], path_rc[i + 1][1] - path_rc[i][1])
        if d_prev != d_next:
            out.append(path_rc[i])
    out.append(path_rc[-1])
    return out


def line_of_sight(passable, a_rc, b_rc):
    """두 칸 사이를 Bresenham 으로 이었을 때 전부 통과 가능한가."""
    rr, cc = bresenham_line(a_rc[0], a_rc[1], b_rc[0], b_rc[1])
    return bool(passable[rr, cc].all())


def shortcut_path(path_rc, passable, cost_map=None, max_cost_for_shortcut=0.5):
    """'끈 당기기(string pulling)': 현재 점에서 보이는(line-of-sight) 가장 먼 점으로 바로 연결.

    격자 경로는 8방향 제약 때문에 계단 모양이 됩니다. 이를 직선으로 이으면 로봇이 부드럽게
    움직입니다. 단, 직선이 벽 근처(비용 높은 칸)를 지나면 안전거리 효과가 사라지므로
    cost_map 을 주면 비용이 max_cost_for_shortcut 이하인 칸들만 지나는 직선만 허용합니다.
    """
    if len(path_rc) <= 2:
        return list(path_rc)

    def ok(a, b):
        rr, cc = bresenham_line(a[0], a[1], b[0], b[1])
        if not passable[rr, cc].all():
            return False
        if cost_map is not None and (cost_map[rr, cc] > max_cost_for_shortcut).any():
            return False
        return True

    out = [path_rc[0]]
    i = 0
    n = len(path_rc)
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not ok(path_rc[i], path_rc[j]):
            j -= 1
        out.append(path_rc[j])
        i = j
    return out


def path_to_world(path_rc, spec: GridSpec) -> List[Tuple[float, float]]:
    """(row, col) 리스트 → 칸 중심 월드 좌표 (x, y) 리스트 (팀 인터페이스 형식)."""
    if not path_rc:
        return []
    arr = np.asarray(path_rc, dtype=int)
    xs, ys = spec.grid_to_world(arr[:, 0], arr[:, 1])
    return [(float(x), float(y)) for x, y in zip(xs, ys)]


# ----------------------------------------------------------------------
# 5. 한 번에: 월드 좌표 시작/목표 → 월드 좌표 경로
# ----------------------------------------------------------------------
def plan_path(inflated, spec: GridSpec, start_xy, goal_xy,
              safe_dist=0.5, w_prox=3.0, unknown_passable=False,
              smooth=True, snap_radius_cells=12, cost_map=None):
    """팀원들이 부를 함수. 성공 시 [(x, y), ...] (시작점 포함), 실패 시 None.

    inflated         : 팽창된 0/1/-1 지도 (occupancy_grid.inflate 결과)
    unknown_passable : 미탐색 칸을 지나가도 되는가. 탐색 중엔 False 권장 (모르는 곳으로 경로를
                       내면 실제로는 벽일 수 있음). 복귀 시엔 이미 다 알고 있으니 False 로 충분.
    cost_map         : 미리 계산한 비용 지도를 넘겨 재사용 가능 (같은 지도로 여러 번 계획할 때).
    """
    passable = (inflated == 0)
    if unknown_passable:
        passable |= (inflated == -1)

    s = spec.world_to_grid(start_xy[0], start_xy[1])
    gl = spec.world_to_grid(goal_xy[0], goal_xy[1])
    s = nearest_passable(passable, (int(s[0]), int(s[1])), snap_radius_cells)
    gl = nearest_passable(passable, (int(gl[0]), int(gl[1])), snap_radius_cells)
    if s is None or gl is None:
        return None
    if cost_map is None:
        cost_map = build_cost_map(inflated, spec.resolution, safe_dist, w_prox)
    path_rc = astar_grid(passable, s, gl, cost_map)
    if path_rc is None:
        return None
    if smooth:
        path_rc = shortcut_path(path_rc, passable, cost_map)
    else:
        path_rc = simplify_collinear(path_rc)
    pts = path_to_world(path_rc, spec)
    # 첫 점을 실제 로봇 위치로 바꿔 두면 경로 추종이 자연스러움 (칸 중심으로 되돌아가지 않음)
    pts[0] = (float(start_xy[0]), float(start_xy[1]))
    return pts


# ----------------------------------------------------------------------
# 단독 실행 테스트 + 시각화
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    from geometry import setup_korean_font
    setup_korean_font()
    from occupancy_grid import inflate

    spec = GridSpec.from_size(10.0, 8.0, resolution=0.05, center=(5.0, 4.0))
    tern = np.zeros(spec.shape, dtype=np.int8)
    # 바깥 벽
    tern[0, :] = tern[-1, :] = 1; tern[:, 0] = tern[:, -1] = 1
    # 가운데 세로 벽 (문 하나), 옆에 기둥
    r, c = spec.world_to_grid(np.array([5.0, 5.0]), np.array([0.0, 8.0]))
    tern[:, c[0]:c[0] + 3] = 1
    dr_ = spec.world_to_grid(0, np.array([5.5, 6.5]))[0]
    tern[dr_[0]:dr_[1], c[0]:c[0] + 3] = 0
    pr, pc = spec.world_to_grid(np.array([7.0, 7.6]), np.array([2.0, 2.6]))
    tern[pr[0]:pr[1], pc[0]:pc[1]] = 1
    # 오른쪽 위는 미탐색으로
    ur, uc = spec.world_to_grid(np.array([7.5, 10.0]), np.array([6.0, 8.0]))
    tern[ur[0]:ur[1], uc[0]:uc[1]] = -1

    inflated = inflate(tern, radius_m=0.3, resolution=spec.resolution)
    cost = build_cost_map(inflated, spec.resolution, safe_dist=0.6, w_prox=3.0)
    start, goal = (1.0, 1.0), (9.0, 1.0)

    t0 = time.perf_counter()
    path_prox = plan_path(inflated, spec, start, goal, safe_dist=0.6, w_prox=3.0)
    t1 = time.perf_counter()
    path_plain = plan_path(inflated, spec, start, goal, w_prox=0.0, smooth=False)
    t2 = time.perf_counter()
    print(f"벽 근접 비용 O: {len(path_prox)} 점, {1000 * (t1 - t0):.0f} ms")
    print(f"벽 근접 비용 X: {len(path_plain)} 점, {1000 * (t2 - t1):.0f} ms")
    print("첫 3점 (월드 좌표):", [tuple(round(v, 2) for v in p) for p in path_prox[:3]])

    # 벽 안에 있는 목표 → nearest_passable 로 스냅되는지
    p_snap = plan_path(inflated, spec, start, (7.3, 2.3))
    print("벽 안 목표 → 스냅 후 도착점:", tuple(round(v, 2) for v in p_snap[-1]))
    # 미탐색 영역 목표 → unknown_passable=False 면 실패(None) 해야 함
    p_unknown = plan_path(inflated, spec, start, (9.0, 7.5), snap_radius_cells=3)
    print("미탐색 영역 목표 (unknown_passable=False):", "None (예상대로)" if p_unknown is None else "경로 있음?!")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.imshow(inflated, origin="lower", cmap="gray_r", vmin=-1, vmax=1, extent=spec.extent)
    pp = np.array(path_plain); ax.plot(pp[:, 0], pp[:, 1], "b.-", label="비용 X (최단)")
    pq = np.array(path_prox);  ax.plot(pq[:, 0], pq[:, 1], "r.-", label="벽 근접 비용 O + 직선화")
    ax.plot(*start, "go", ms=10, label="start"); ax.plot(*goal, "m*", ms=14, label="goal")
    ax.legend(); ax.set_title("팽창 지도 위 A* 경로 비교")
    ax = axes[1]
    im = ax.imshow(cost, origin="lower", cmap="magma", extent=spec.extent)
    ax.plot(pq[:, 0], pq[:, 1], "c.-")
    plt.colorbar(im, ax=ax); ax.set_title("벽 근접 비용 지도 (밝을수록 비쌈)")
    plt.tight_layout()
    plt.show()
