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
from frontier import frontier_mask, cluster_frontiers, select_frontier
from astar import plan_path, build_cost_map
from robot_config import ROBOT_RADIUS, LIDAR_MAX_RANGE

SAFETY_MARGIN = 0.10        # 팽창 여유. 문/가구 사이가 좁으면 0.06 까지 줄여 볼 것
MAP_HALF_M = 15.0           # 시작점 기준 ±15 m (apartment 기준). 작으면 dropped_hits 경고
RESOLUTION = 0.05           # 600x600 격자. A* 가 느리면 0.075 로
REPLAN_PERIOD_S = 1.0
MIN_FRONTIER_CELLS = 6
GOAL_KEEP_RADIUS = 0.5      # 현재 목표 이 반경 안에 frontier 가 남아 있으면 목표 유지 (진동 방지)


class GridPlanner:
    def __init__(self, start_xy=(0.0, 0.0), max_range=LIDAR_MAX_RANGE):
        self.spec = GridSpec.from_size(2 * MAP_HALF_M, 2 * MAP_HALF_M, RESOLUTION, center=start_xy)
        self.grid = OccupancyGrid(self.spec, max_range=max_range)
        self.inflate_r = ROBOT_RADIUS + SAFETY_MARGIN
        self.blacklist: List[Tuple[float, float]] = []
        self._ternary = None
        self._inflated = None
        self._cost = None

    def update_map(self, pose, angles, ranges):
        self.grid.update(pose, angles, ranges)
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

    def next_exploration_goal(self, pose, current_goal=None, keep_radius=GOAL_KEEP_RADIUS) -> Optional[Tuple[float, float]]:
        """다음 탐색 목표. current_goal 을 주면 '목표 유지' 규칙: 현재 목표 반경 keep_radius 안에 아직 frontier 가
        남아 있으면 그대로 둔다 (매초 1등이 바뀌면 로봇이 두 목표 사이를 왔다갔다 한다)."""
        self._refresh()
        mask = frontier_mask(self._ternary, self._inflated)
        if current_goal is not None:
            r, c = self.spec.world_to_grid(current_goal[0], current_goal[1])
            k = int(keep_radius / self.spec.resolution)
            r0, r1 = max(0, r - k), min(self.spec.rows, r + k + 1)
            c0, c1 = max(0, c - k), min(self.spec.cols, c + k + 1)
            still_frontier = mask[r0:r1, c0:c1].any()
            far_enough = np.hypot(current_goal[0] - pose[0], current_goal[1] - pose[1]) > 0.3
            not_blacklisted = all(np.hypot(current_goal[0] - b[0], current_goal[1] - b[1]) > 0.4 for b in self.blacklist)
            if still_frontier and far_enough and not_blacklisted:
                return tuple(current_goal)
        frontiers = cluster_frontiers(mask, min_size=MIN_FRONTIER_CELLS)
        f = select_frontier(frontiers, pose, self.spec, blacklist_xy=self.blacklist)
        return f.goal_xy(self.spec) if f else None

    def plan(self, pose, goal_xy):
        self._refresh()
        return plan_path(self._inflated, self.spec, (pose[0], pose[1]), goal_xy, cost_map=self._cost,
                         snap_radius_cells=int(0.6 / self.spec.resolution))

    def give_up_goal(self, goal_xy):
        self.blacklist.append(tuple(goal_xy))
