"""
exploration.py - 지도·frontier·A* 를 묶은 GridPlanner (브랜치 feat/exploration 소유)

인터페이스 (webots_adapter.run 이 부름):
    planner = GridPlanner(start_xy, max_range)
    planner.update_map(pose, angles, ranges)
    planner.next_exploration_goal(pose, current_goal) -> (x, y) 또는 None(탐색 완료)
    planner.plan(pose, goal_xy) -> [(x, y), ...] 또는 None
    planner.give_up_goal(goal_xy)
    planner.get_map() -> (int8 0/1/-1, GridSpec),  planner.grid.log_odds (scan matching 용)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from geometry import GridSpec
from occupancy_grid import OccupancyGrid, inflate
from frontier import frontier_mask, cluster_frontiers, select_frontier, bfs_distance_map_coarse, bfs_distance_map
from astar import plan_path, build_cost_map
from robot_config import ROBOT_RADIUS, LIDAR_MAX_RANGE, CAM_FOV

SAFETY_MARGIN = 0.10        # 팽창 여유. 문/가구 사이가 좁으면 0.06 까지 줄여 볼 것
MAP_HALF_M = 20.0           # 시작점 기준 ±20 m. apartment 서쪽 끝이 시작점(-0.3)에서 -13 m 이고 위치 추정 오차가 수 m 까지
                            # 날 수 있어 15 m 로는 지도 밖으로 나감(5차 실행: 목표가 지도 밖 → 무한 회전). 800x800 격자.
RESOLUTION = 0.05           # 600x600 격자. A* 가 느리면 0.075 로
REPLAN_PERIOD_S = 1.0
MIN_FRONTIER_CELLS = 6
SNAP_RADIUS_M = 0.6         # (j) 좁은 방: frontier 군집이 전부 벽 0.2 m 안(팽창 영역)이어도 버리지 않고, 중심에서 이 반경 안의
                            #     가장 가까운 통과·도달 가능 칸을 목표로 스냅. 2차 지도: 시작점 북쪽 작은 방 frontier 166칸 중
                            #     통과 가능 5칸, 변기방 4칸 → 후보에서 걸러져 '죽어도 안 들어감'. 카메라는 2 m 밖에서도 보므로
                            #     그 칸 '앞' 에 서면 충분하다.
GOAL_KEEP_RADIUS = 0.5      # 현재 목표 이 반경 안에 frontier 가 남아 있으면 목표 유지 (진동 방지)
BLACKLIST_TTL_S = 90.0      # 포기한 목표를 이만큼만 피한다. 영구 블랙리스트는 정체가 반복되면 모든 frontier 를 지워
                            # "탐색 경로 없음"(기준선 c54afcb: 525 s BLOCKED) 으로 끝난다. 시간이 지나면 다시 시도.
# frontier 점수 가중치 (sim_demo 스윕으로 결정 — docs/WORKLOG.md feat/exploration 절 참고)
USE_PATH_DIST = True        # 직선거리 대신 BFS 경로거리 (벽 너머 "가짜로 가까운" frontier 방지)
BFS_COARSE_FACTOR = 2       # 경로거리 BFS 격자 축소 배율. 4(20 cm 칸, 18 ms) 는 팽창 뒤 폭 ~0.4 m 인 문을 막아 도달 가능 frontier 를
                            # 0 개로 판정(1차 실행 618 s "탐색 후보 없음": 전해상도 BFS 는 3개 도달). 2(10 cm, 80 ms) 는 전해상도와 일치.
W_DIST = 3.0                # 경로거리 가중치를 지배적으로(1.0→3.0): 가장 가까운 목표부터 → 같은 방을 다 보고 나감 (h)
W_SIZE = 0.02               # 크기 항 최대 0.8 (0.05→0.02)
SIZE_CAP = 40
W_TURN = 0.1                # 회전 벌점 낮게 (0.3→0.1)
# (h) 통합 탐색: 목표 후보 = LiDAR frontier ∪ 카메라 미확인 자유셀 군집. 1차 실행에서 LiDAR 지도는 다 그렸는데 카메라(60°)로
#     빨간 사과 2.5 m 옆을 137회 지나치며 한 번도 정면에 두지 않음 → "카메라로 다 봤다" 를 탐색 완료 기준으로.
UNSEEN_MIN_CELLS = 200      # 미확인 군집 최소 크기 ≈ 0.5 m² (지그재그 방지)
UNIFIED_UNSEEN = False      # EXPLORE 후보에 미확인 군집을 합칠지. 오프라인 재생(1차 지도, 700 s): 합치면 목표 변경 565회(도착 전 50 %)
                            # 로 사과를 못 포착, frontier 만이면 10회·Red2 44 s 포착 → 기본 끔. 미확인 구역은 SEARCH 폴백이 담당.
DEPTH_FIRST_RADIUS = 3.0    # 이 반경 안에 미확인 군집이 남아 있으면 그 밖 후보는 뒤로 미룸 (깊이 우선)
UNSEEN_GOAL_MIN_M = 1.0     # 미확인 군집 목표 칸은 로봇에서 최소 이만큼 떨어진 칸 (코앞 칸은 곧 '봤다' 가 되어 목표가 흔들림)
SWITCH_RATIO = 0.5          # 현재 목표가 유효하면 유지. 새 후보 경로거리가 남은 거리의 이 비율 미만일 때만 교체 (중도 이탈 방지)
W_DETOUR = 0.0              # 경로/직선 비율이 1.5 넘는 frontier 벌점 (방 먼저 끝내기)
# SEARCH (카메라 미확인 구역 수색) — 학습 코드 sim_demo.py 의 SEARCH 단계 이식.
# 왜: LiDAR(3.5 m, 360°) 로 지도가 다 그려져 frontier 가 사라져도, 카메라(60°, 사과 탐지 유효 ~3 m) 가 훑지 않은 빈칸이
#     남는다. 빨간 사과 2개를 모두 찾아야 하므로 frontier 소진 = 탐색 끝이 아니다.
CAM_SEEN_RANGE = 1.2        # [m] 카메라로 '봤다' 고 인정하는 거리. 2.0 으로 두면 2차 실행 300 s 에 미확인 5칸(=다 봤다) 인데 빨간 사과 2개
                            # 모두 미탐지 — 가구(LiDAR 는 다리만 봄)에 가려 멀리선 안 보이고, 완주 3회 모두 0.8~0.9 m 에서 탐지됨.
                            # 1.2 m 면 SEARCH·둘러보기가 그 거리까지 실제로 다가가게 된다 (느리지만 빠짐 없음).
MIN_UNSEEN_CELLS = 40       # 수색 대상 미확인 덩어리 최소 칸 수 (0.1 m² = 40칸)


class GridPlanner:
    def __init__(self, start_xy=(0.0, 0.0), max_range=LIDAR_MAX_RANGE):
        self.spec = GridSpec.from_size(2 * MAP_HALF_M, 2 * MAP_HALF_M, RESOLUTION, center=start_xy)
        self.grid = OccupancyGrid(self.spec, max_range=max_range)
        self.inflate_r = ROBOT_RADIUS + SAFETY_MARGIN
        self.blacklist: List[Tuple[float, float]] = []
        self._blacklist_t: List[float] = []       # 각 항목의 등록 시각 (give_up_goal 의 t). None 이면 만료 없음
        self.now = 0.0                            # adapter 가 매 스텝 갱신하는 시뮬 시각 (만료 계산용)
        self._ternary = None
        self._inflated = None
        self._cost = None
        self.seen = np.zeros(self.spec.shape, dtype=bool)   # 카메라 시야로 본 칸 (SEARCH 용)
        self._forced = np.zeros(self.spec.shape, dtype=bool) # mark_obstacle 로 강제한 칸. 매 스캔 뒤 다시 벽으로 고정
        self.unseen_cells = 0                                # 마지막 next_search_goal 때 '갈 수 있는데 안 본' 칸 수

    def update_map(self, pose, angles, ranges):
        self.grid.update(pose, angles, ranges)
        if self._forced.any():
            # LiDAR 가 못 보는 물체(사과 등)는 빔이 그 자리를 '빈칸' 으로 지나가며 log-odds 를 매 스캔 -0.62 씩 깎아
            # 강제 표시가 1 초 안에 지워졌다 (1차 실행: 보라 사과 등록 0.45 m 뒤 30 s 만에 그 자리로 주행해 접촉).
            self.grid.log_odds[self._forced] = self.grid.l_clamp
        self._ternary = None                       # 캐시 무효화

    def _refresh(self):
        if self._ternary is None:
            self._ternary = self.grid.to_ternary()
            self._inflated = inflate(self._ternary, self.inflate_r, self.spec.resolution)
            self._cost = build_cost_map(self._inflated, self.spec.resolution, safe_dist=0.4, w_prox=3.0)

    def get_map(self):
        self._refresh()
        return self._ternary, self.spec

    def get_inflated(self):
        self._refresh()
        return self._inflated

    def _path_dist_map(self, pose, clusters):
        """BFS 경로거리 지도. 축소 격자로 먼저 계산하고, 후보 군집이 하나도 도달 불가로 나오면(좁은 문이 축소에 막힘)
        전해상도(≈0.6 s) 로 다시 계산한다."""
        if not (USE_PATH_DIST and clusters):
            return None
        r, c = self.spec.world_to_grid(pose[0], pose[1])
        free = self._inflated == 0
        dm = bfs_distance_map_coarse(free, (int(r), int(c)), BFS_COARSE_FACTOR)
        if not any(np.isfinite(dm[f.goal_rc]) for f in clusters):
            dm = bfs_distance_map(free, (int(r), int(c)))
        return dm

    def _snap(self, centroid_rc, ok, radius_cells):
        """centroid 주변 radius_cells 창에서 ok(True) 인 가장 가까운 칸. 없으면 None."""
        r0, c0 = int(round(centroid_rc[0])), int(round(centroid_rc[1]))
        k = radius_cells
        ra, rb = max(0, r0 - k), min(self.spec.rows, r0 + k + 1)
        ca, cb = max(0, c0 - k), min(self.spec.cols, c0 + k + 1)
        win = ok[ra:rb, ca:cb]
        if not win.any():
            return None
        yy, xx = np.nonzero(win)
        d2 = (yy + ra - centroid_rc[0]) ** 2 + (xx + ca - centroid_rc[1]) ** 2
        i = int(np.argmin(d2))
        return (int(yy[i] + ra), int(xx[i] + ca))

    def _snap_clusters(self, clusters, pose):
        """(j) 각 군집의 goal_rc 를 통과 가능 칸으로, 다시 도달 가능 칸으로 스냅. 스냅 실패 군집은 제거.
        반환 (군집 목록, 경로거리 지도)."""
        passable = self._inflated == 0
        k = int(SNAP_RADIUS_M / self.spec.resolution)
        kept = []
        for f in clusters:
            if not passable[f.goal_rc]:
                g = self._snap(f.centroid_rc, passable, k)
                if g is None:
                    continue
                f.goal_rc = g
            kept.append(f)
        dm = self._path_dist_map(pose, kept)
        if dm is not None:
            reach = passable & np.isfinite(dm)
            out = []
            for f in kept:
                if np.isfinite(dm[f.goal_rc]):
                    out.append(f); continue
                g = self._snap(f.centroid_rc, reach, k)
                if g is not None:
                    f.goal_rc = g; out.append(f)
            kept = out
        return kept, dm

    def _candidates(self, pose):
        """(frontier 군집 + 미확인 군집, frontier 마스크|미확인 마스크, 경로거리 지도)
        (j) frontier 는 팽창 필터 없이 뽑고(좁은 방 안 frontier 보존) 목표 칸만 통과·도달 가능 칸으로 스냅."""
        fmask = frontier_mask(self._ternary, None)
        unseen = self.unseen_mask(); self.unseen_cells = int(unseen.sum())
        frontiers = cluster_frontiers(fmask, min_size=MIN_FRONTIER_CELLS)
        unseen_cl = cluster_frontiers(unseen, min_size=UNSEEN_MIN_CELLS) if UNIFIED_UNSEEN else []
        rr, cc = self.spec.world_to_grid(pose[0], pose[1])
        for f in unseen_cl:
            f.kind = "unseen"
            d2 = (f.cells[:, 0] - rr) ** 2 + (f.cells[:, 1] - cc) ** 2
            far = d2 >= (UNSEEN_GOAL_MIN_M / self.spec.resolution) ** 2
            idx = int(np.argmin(np.where(far, d2, np.inf))) if far.any() else int(np.argmax(d2))
            f.goal_rc = (int(f.cells[idx, 0]), int(f.cells[idx, 1]))
        cands, dm = self._snap_clusters(frontiers + unseen_cl, pose)
        return cands, (fmask | unseen) if UNIFIED_UNSEEN else fmask, dm

    def _path_len(self, pose, goal_xy, dist_map):
        r, c = self.spec.world_to_grid(goal_xy[0], goal_xy[1])
        r = int(np.clip(r, 0, self.spec.rows - 1)); c = int(np.clip(c, 0, self.spec.cols - 1))
        if dist_map is not None and np.isfinite(dist_map[r, c]):
            return float(dist_map[r, c]) * self.spec.resolution
        return float(np.hypot(goal_xy[0] - pose[0], goal_xy[1] - pose[1]))

    def next_exploration_goal(self, pose, current_goal=None, keep_radius=GOAL_KEEP_RADIUS) -> Optional[Tuple[float, float]]:
        """다음 탐색 목표 (h: frontier ∪ 카메라 미확인 군집, 경로거리 최우선, 깊이 우선, 강한 목표 유지).
        None 이면 후보 없음 (→ adapter 가 SEARCH 폴백 / BLOCKED 판단)."""
        self._refresh()
        self._expire_blacklist(self.now)
        cands, mask, dist_map = self._candidates(pose)

        def pick(pool):
            return select_frontier(pool, pose, self.spec, blacklist_xy=self.blacklist, dist_map=dist_map,
                                   w_dist=W_DIST, w_size=W_SIZE, size_cap=SIZE_CAP, w_turn=W_TURN, w_detour=W_DETOUR)

        # 깊이 우선: 반경 안에 (도달 가능한) 미확인 군집이 남아 있으면 후보를 그 반경 안으로 제한
        def near(f):
            g = f.goal_xy(self.spec)
            return (np.hypot(g[0] - pose[0], g[1] - pose[1]) <= DEPTH_FIRST_RADIUS
                    and (dist_map is None or np.isfinite(dist_map[f.goal_rc])))
        near_pool = [f for f in cands if near(f)]
        pool = near_pool if any(getattr(f, "kind", "") == "unseen" for f in near_pool) else cands

        if current_goal is not None:
            r, c = self.spec.world_to_grid(current_goal[0], current_goal[1])
            r = int(np.clip(r, 0, self.spec.rows - 1)); c = int(np.clip(c, 0, self.spec.cols - 1))
            k = int((keep_radius + SNAP_RADIUS_M) / self.spec.resolution)   # (j) 목표는 frontier 에서 최대 SNAP 만큼 떨어질 수 있음
            still_valid = mask[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1].any()
            goal_passable = self._inflated[r, c] == 0
            far_enough = np.hypot(current_goal[0] - pose[0], current_goal[1] - pose[1]) > 0.3
            not_blacklisted = all(np.hypot(current_goal[0] - b[0], current_goal[1] - b[1]) > 0.4 for b in self.blacklist)
            if not goal_passable:
                # 목표 칸이 벽/팽창 영역이 됨 (기준선: 벽 셀 목표 유지 → 정체) → 포기
                self.give_up_goal(current_goal, self.now)
            elif still_valid and far_enough and not_blacklisted:
                remaining = self._path_len(pose, current_goal, dist_map)
                f = pick(pool)
                if f is None:
                    return tuple(current_goal)
                g = f.goal_xy(self.spec)
                if np.hypot(g[0] - current_goal[0], g[1] - current_goal[1]) < keep_radius \
                        or self._path_len(pose, g, dist_map) >= SWITCH_RATIO * remaining:
                    return tuple(current_goal)               # 도착 전 교체 금지 (훨씬 가까운 새 후보만 예외)
                return g
        f = pick(pool)
        if f is None and pool is not cands:
            f = pick(cands)
        return f.goal_xy(self.spec) if f else None

    def plan(self, pose, goal_xy):
        self._refresh()
        return plan_path(self._inflated, self.spec, (pose[0], pose[1]), goal_xy, cost_map=self._cost,
                         snap_radius_cells=int(0.6 / self.spec.resolution))

    def give_up_goal(self, goal_xy, t=None):
        """목표 포기. t(시뮬 시각) 를 주면 BLACKLIST_TTL_S 뒤 자동 해제된다."""
        self._expire_blacklist(self.now if t is None else t)
        self.blacklist.append(tuple(goal_xy)); self._blacklist_t.append(self.now if t is None else t)

    def _expire_blacklist(self, t):
        keep = [i for i, t0 in enumerate(self._blacklist_t) if t0 is None or t - t0 < BLACKLIST_TTL_S]
        if len(keep) != len(self.blacklist):
            self.blacklist = [self.blacklist[i] for i in keep]; self._blacklist_t = [self._blacklist_t[i] for i in keep]

    def clear_blacklist(self):
        n = len(self.blacklist); self.blacklist = []; self._blacklist_t = []; return n

    def mark_obstacle(self, x, y, radius_m=0.15):
        """지도에 벽을 강제로 찍는다. LiDAR 가 못 보는 낮은 물체(공, 사과, 고양이)에 걸렸을 때 그 자리를 막아
        A* 가 다시는 그리로 경로를 내지 않게 한다. log-odds 를 최대치로 두어 다음 스캔의 miss 로 금방 지워지지 않게."""
        r_c = int(np.ceil(radius_m / self.spec.resolution))
        r0, c0 = self.spec.world_to_grid(x, y)
        rr, cc = np.ogrid[-r_c:r_c + 1, -r_c:r_c + 1]
        disk = (rr ** 2 + cc ** 2) <= r_c ** 2
        R = np.clip(r0 + rr, 0, self.spec.rows - 1); C = np.clip(c0 + cc, 0, self.spec.cols - 1)
        self.grid.log_odds[R, C] = np.where(disk, self.grid.l_clamp, self.grid.log_odds[R, C])
        self._forced[R, C] |= disk                 # 이후 스캔의 miss 로 지워지지 않게 고정 (update_map 에서 재적용)
        self._ternary = None

    # ------------------------------------------------------------------
    # SEARCH: 카메라 미확인 구역 수색 (sim_demo.py 이식)
    # ------------------------------------------------------------------
    def mark_camera_seen(self, pose, angles, ranges, cam_fov=CAM_FOV, cam_range=CAM_SEEN_RANGE):
        """카메라 시야각 안의 LiDAR 빔을 따라 cam_range 까지의 칸을 '봤다' 로 표시.
        왜 LiDAR 빔을 재활용하나: 카메라가 어디까지 보이는지 = 그 방향으로 벽이 어디 있는지 → LiDAR 가 이미 알려준다."""
        a = np.asarray(angles); rg = np.asarray(ranges, dtype=float)
        in_fov = np.abs(a) < cam_fov / 2
        ang = a[in_fov]
        r = np.minimum(np.nan_to_num(rg[in_fov], nan=0.0, posinf=cam_range, neginf=0.0), cam_range)
        step = self.spec.resolution * 0.5
        ts = np.arange(0.0, cam_range + step, step)
        d = np.minimum(ts[None, :], r[:, None])                       # (N, S) 빔 끝(벽)에서 멈춤
        xs = pose[0] + d * np.cos(ang + pose[2])[:, None]
        ys = pose[1] + d * np.sin(ang + pose[2])[:, None]
        rr, cc = self.spec.world_to_grid(xs, ys)
        inb = self.spec.in_bounds(rr, cc)
        self.seen[rr[inb], cc[inb]] = True

    def unseen_mask(self):
        self._refresh()
        return (self._ternary == 0) & (self._inflated == 0) & ~self.seen

    def unseen_near(self, pose, radius_m=2.0):
        """로봇 주변 radius_m 안의 '갈 수 있는데 카메라로 안 본' 칸 수 (제자리 둘러보기 판단용)."""
        unseen = self.unseen_mask()
        r, c = self.spec.world_to_grid(pose[0], pose[1]); k = int(radius_m / self.spec.resolution)
        r0, r1 = max(0, int(r) - k), min(self.spec.rows, int(r) + k + 1); c0, c1 = max(0, int(c) - k), min(self.spec.cols, int(c) + k + 1)
        sub = unseen[r0:r1, c0:c1]
        yy, xx = np.ogrid[r0:r1, c0:c1]
        return int((sub & ((yy - r) ** 2 + (xx - c) ** 2 <= k * k)).sum())

    def next_search_goal(self, pose, current_goal=None, keep_radius=GOAL_KEEP_RADIUS) -> Optional[Tuple[float, float]]:
        """'갈 수 있는 빈칸' 중 카메라로 본 적 없는 칸을 덩어리로 묶어 frontier 와 같은 점수(BFS 경로거리) 로 목표 선택.
        None 이면 수색할 곳 없음. 목표 칸을 이미 봤으면 다른 목표로 바꾼다."""
        self._expire_blacklist(self.now)
        unseen = self.unseen_mask()
        self.unseen_cells = int(unseen.sum())
        if current_goal is not None:
            r, c = self.spec.world_to_grid(current_goal[0], current_goal[1])
            r = int(np.clip(r, 0, self.spec.rows - 1)); c = int(np.clip(c, 0, self.spec.cols - 1))
            k = int(keep_radius / self.spec.resolution)
            still_unseen = unseen[max(0, r - k):r + k + 1, max(0, c - k):c + k + 1].any()
            if still_unseen and not self.seen[r, c] and self._inflated[r, c] == 0 \
                    and all(np.hypot(current_goal[0] - b[0], current_goal[1] - b[1]) > 0.4 for b in self.blacklist):
                return tuple(current_goal)
        clusters = cluster_frontiers(unseen, min_size=MIN_UNSEEN_CELLS)
        if not clusters:
            return None
        clusters, dist_map = self._snap_clusters(clusters, pose)   # (j) 도달 불가 군집은 근처 도달 가능 칸으로
        if not clusters:
            return None
        f = select_frontier(clusters, pose, self.spec, blacklist_xy=self.blacklist, dist_map=dist_map,
                            w_dist=W_DIST, w_size=0.01, size_cap=200, w_turn=W_TURN)
        return f.goal_xy(self.spec) if f else None

