"""
geometry.py - 좌표 변환 모음 (격자 파트의 기초)

이 파일이 다루는 좌표계는 4개입니다. 순서대로 변환합니다.

  LiDAR 측정값 (angle, range)          극좌표. 로봇에 붙어 있는 센서가 준 원시 데이터
        │ lidar_to_robot
        ▼
  로봇 좌표계 (x_r, y_r)               로봇 중심이 원점, x_r = 로봇 앞, y_r = 로봇 왼쪽
        │ robot_to_world (pose 필요)
        ▼
  월드 좌표계 (x_w, y_w)               시작 지점 기준 고정 좌표. 지도와 경로는 전부 이 좌표
        │ GridSpec.world_to_grid
        ▼
  격자 인덱스 (row, col)               NumPy 배열 인덱스. grid[row, col]

가장 많이 헷갈리는 두 가지 (README의 "막히는 포인트"에도 정리):
  1) row 는 y 방향, col 은 x 방향입니다. grid[y][x] 순서라는 뜻.
     → 함수 인자/반환값에서 항상 (row, col) 순서로 쓰고, 월드 좌표는 (x, y) 순서로 씁니다.
  2) 각도는 x축(전방)에서 반시계 방향이 양수(+)입니다. 왼쪽이 +y, +각도.
     Webots LiDAR의 빔 순서가 왼→오인지 오→왼인지는 당일 반드시 확인해야 합니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. 각도 정규화
# ---------------------------------------------------------------------------
def wrap_angle(angle):
    """각도를 [-pi, pi) 범위로 접어 넣습니다. 스칼라/배열 모두 지원.

    왜 필요한가:
      heading 오차 = 목표각 - 현재각 을 그냥 빼면 350° - 10° = 340° 처럼 나오는데,
      실제로는 -20° 만 돌면 됩니다. 각도를 뺀 뒤에는 항상 wrap 해야 합니다.
      "로봇이 목표 근처에서 갑자기 한 바퀴 빙 돈다"는 증상은 십중팔구 이걸 빠뜨린 것.

    구현: (a + pi) mod 2pi - pi.  파이썬 % 는 음수에도 양수 결과를 주므로 그대로 쓸 수 있음.
    """
    out = (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    return float(out) if np.ndim(out) == 0 else out


# ---------------------------------------------------------------------------
# 2. LiDAR (angle, range) -> 로봇 좌표
# ---------------------------------------------------------------------------
def lidar_to_robot(angles, ranges):
    """극좌표 (angle, range) 를 로봇 좌표계 점 (N, 2) 배열로 바꿉니다.

    angle 은 로봇 전방(x축) 기준 반시계 양수 [rad], range 는 미터.
    LiDAR가 로봇 중심에서 (dx, dy) 만큼 떨어져 붙어 있다면 반환값에 그 오프셋을 더해야
    합니다 (당일 로봇 모델을 보고 결정. 보통 작은 로봇은 무시해도 됨).

    inf/nan range 는 그대로 inf/nan 점이 되므로, 호출 전에 걸러내거나
    occupancy_grid 쪽에서 처리합니다.
    """
    angles = np.asarray(angles, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    return np.stack([xs, ys], axis=1)


# ---------------------------------------------------------------------------
# 3. 로봇 좌표 <-> 월드 좌표 (2D 강체 변환)
# ---------------------------------------------------------------------------
def rotation_matrix(theta):
    """2D 회전행렬 R(theta). 열벡터 v 를 R @ v 로 회전시킵니다."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def robot_to_world(points_robot, pose):
    """로봇 좌표계 점들 (N, 2) 을 월드 좌표계로. pose = (x, y, theta).

    수식:  p_world = R(theta) @ p_robot + t,   t = (x, y)
    NumPy 는 점을 행벡터 (N, 2) 로 들고 있으므로  points @ R.T  형태가 됩니다.
    (R @ p)^T = p^T @ R^T 이기 때문.
    """
    x, y, theta = pose
    pts = np.asarray(points_robot, dtype=float)
    R = rotation_matrix(theta)
    return pts @ R.T + np.array([x, y])


def world_to_robot(points_world, pose):
    """월드 좌표 점들 (N, 2) 을 로봇 좌표계로 (robot_to_world 의 역변환).

    수식:  p_robot = R(theta)^T @ (p_world - t)
    회전행렬의 역행렬은 전치(R^T)라는 점을 이용. 행벡터 형태에서는 (p - t) @ R 이 됩니다.
    용도: 목표점이 로봇 기준으로 앞/뒤/왼/오 어디에 있는지 볼 때 (예: heading 오차 계산).
    """
    x, y, theta = pose
    pts = np.asarray(points_world, dtype=float)
    R = rotation_matrix(theta)
    return (pts - np.array([x, y])) @ R


def lidar_to_world(angles, ranges, pose):
    """편의 함수: LiDAR 극좌표 -> 월드 좌표 한 번에."""
    return robot_to_world(lidar_to_robot(angles, ranges), pose)


# ---------------------------------------------------------------------------
# 4. 월드 좌표 <-> 격자 인덱스
# ---------------------------------------------------------------------------
@dataclass
class GridSpec:
    """격자 지도의 '틀' 정보. 지도 배열 자체는 들고 있지 않습니다.

    resolution : 한 칸의 한 변 길이 [m/칸]. 0.05 면 5cm.
    origin_x/y : grid[0, 0] 칸의 **왼쪽-아래 모서리**의 월드 좌표.
                 (칸 중심이 아님! 중심으로 잡으면 변환마다 0.5칸씩 어긋나기 쉬움)
    rows, cols : 배열 크기. rows 는 y 방향 칸 수, cols 는 x 방향 칸 수.

    이 값들은 팀 전체가 같은 것을 써야 합니다 (지도를 주고받는 인터페이스).
    """
    resolution: float = 0.05
    origin_x: float = -5.0
    origin_y: float = -5.0
    rows: int = 200
    cols: int = 200

    # --- 생성 도우미 -------------------------------------------------------
    @classmethod
    def from_size(cls, width_m, height_m, resolution=0.05, center=(0.0, 0.0)):
        """'가로 width_m, 세로 height_m 인 지도를 center 중심으로' 라는 식으로 만들기.

        미지의 환경이므로 실제 크기를 모릅니다. 넉넉히 잡되, 칸 수가 너무 많으면
        A*/frontier 계산이 느려집니다. 20m x 20m @ 5cm = 400x400 = 16만 칸 정도가 적당.
        """
        cols = int(np.ceil(width_m / resolution))
        rows = int(np.ceil(height_m / resolution))
        ox = center[0] - cols * resolution / 2.0
        oy = center[1] - rows * resolution / 2.0
        return cls(resolution=resolution, origin_x=ox, origin_y=oy, rows=rows, cols=cols)

    @property
    def shape(self):
        return (self.rows, self.cols)

    @property
    def width_m(self):
        return self.cols * self.resolution

    @property
    def height_m(self):
        return self.rows * self.resolution

    @property
    def extent(self):
        """matplotlib imshow(extent=...) 에 넘길 [xmin, xmax, ymin, ymax].
        imshow(..., origin='lower', extent=spec.extent) 로 그리면 월드 좌표축과 일치합니다.
        """
        return [self.origin_x, self.origin_x + self.width_m,
                self.origin_y, self.origin_y + self.height_m]

    # --- 변환 --------------------------------------------------------------
    def world_to_grid(self, x, y):
        """월드 (x, y) -> 격자 (row, col). 정수 배열 반환. 범위 밖이어도 그냥 계산함
        (범위 확인은 in_bounds 로 따로).

        floor 를 쓰는 이유: origin 이 칸의 모서리이므로 [origin, origin+res) 구간이 0번 칸.
        int() 로 자르면 음수에서 0 쪽으로 잘려 -0.3칸이 0번 칸이 되는 버그가 생김 → floor 사용.
        """
        col = np.floor((np.asarray(x, dtype=float) - self.origin_x) / self.resolution).astype(int)
        row = np.floor((np.asarray(y, dtype=float) - self.origin_y) / self.resolution).astype(int)
        return row, col

    def grid_to_world(self, row, col):
        """격자 (row, col) -> 그 칸의 **중심** 월드 좌표 (x, y).
        경로를 월드 좌표로 내보낼 때 칸 중심을 쓰면 로봇이 칸 모서리를 따라가지 않아 자연스럽습니다.
        """
        x = self.origin_x + (np.asarray(col, dtype=float) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(row, dtype=float) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        """(row, col) 이 배열 안에 있는지. 배열 입력이면 불리언 배열."""
        row = np.asarray(row)
        col = np.asarray(col)
        return (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)

    def clip(self, row, col):
        """범위 밖 인덱스를 가장자리로 잘라 넣기 (인덱싱 에러 방지용)."""
        return (np.clip(row, 0, self.rows - 1), np.clip(col, 0, self.cols - 1))


# ---------------------------------------------------------------------------
# 5. Bresenham 직선 (격자 위에서 두 칸을 잇는 칸들)
# ---------------------------------------------------------------------------
def bresenham_line(r0, c0, r1, c1):
    """(r0,c0) 에서 (r1,c1) 까지 지나는 격자 칸들을 (rows, cols) 정수 배열로 반환.
    양 끝점 포함, 시작점부터 순서대로.

    왜 쓰나: LiDAR 빔이 "지나간" 칸들은 비어 있다고 표시해야 하는데, 그 칸들이 정확히
    어디인지 정수 연산만으로 빠르게 구하는 고전 알고리즘. 픽셀 아트에서 선 긋는 것과 같음.
    A* 결과 경로를 직선으로 단축(line-of-sight 검사)할 때도 씁니다.

    NumPy 벡터화가 어려운 순수 루프이므로, 360개 빔을 매 스텝 파이썬 루프로 돌리면
    느릴 수 있음 → occupancy_grid.py 에는 벡터화된 샘플링 방식도 같이 넣어 두었습니다.
    """
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dc - dr
    rows, cols = [], []
    r, c = r0, c0
    while True:
        rows.append(r)
        cols.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:      # 열(col) 방향으로 한 칸
            err -= dr
            c += sc
        if e2 < dc:       # 행(row) 방향으로 한 칸
            err += dc
            r += sr
    return np.array(rows, dtype=int), np.array(cols, dtype=int)



# ---------------------------------------------------------------------------
# 6. (부록) matplotlib 한글 폰트 설정 - 그래프 제목의 한글이 □ 로 나오는 것 방지
# ---------------------------------------------------------------------------
def setup_korean_font():
    """OS 별로 흔한 한글 폰트를 찾아 matplotlib 에 설정. 없으면 조용히 넘어감.
    Ubuntu 에 폰트가 없으면:  sudo apt install fonts-nanum  후 ~/.cache/matplotlib 삭제."""
    import warnings
    import matplotlib
    from matplotlib import font_manager
    candidates = ["AppleGothic", "Apple SD Gothic Neo",          # macOS
                  "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",  # Ubuntu
                  "Malgun Gothic"]                                 # Windows
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            break
    matplotlib.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")

# ---------------------------------------------------------------------------
# 단독 실행 테스트 + 시각화
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    setup_korean_font()

    # --- (1) wrap_angle 확인 -----------------------------------------------
    print("wrap_angle 테스트")
    for a in [0.0, np.pi, -np.pi, 3 * np.pi, np.deg2rad(350) - np.deg2rad(10)]:
        print(f"  {a:+.3f} rad -> {wrap_angle(a):+.3f} rad ({np.rad2deg(wrap_angle(a)):+.1f} deg)")
    assert np.allclose(wrap_angle(np.deg2rad(350) - np.deg2rad(10)), np.deg2rad(-20))

    # --- (2) 로봇<->월드 왕복 변환 확인 -------------------------------------
    pose = (2.0, 1.0, np.deg2rad(30))
    pts_r = np.array([[1.0, 0.0],    # 로봇 정면 1m
                      [0.0, 1.0],    # 로봇 왼쪽 1m
                      [-0.5, -0.5]])
    pts_w = robot_to_world(pts_r, pose)
    back = world_to_robot(pts_w, pose)
    assert np.allclose(pts_r, back), "왕복 변환이 원래 점으로 돌아와야 함"
    print("로봇->월드->로봇 왕복 OK")

    # --- (3) 월드<->격자 왕복 확인 ----------------------------------------
    spec = GridSpec.from_size(10.0, 8.0, resolution=0.05, center=(0.0, 0.0))
    xs = np.array([0.0, -4.99, 4.99, 0.026, -0.026])
    ys = np.array([0.0, -3.99, 3.99, 0.026, -0.026])
    r, c = spec.world_to_grid(xs, ys)
    x2, y2 = spec.grid_to_world(r, c)
    # 격자로 갔다 오면 오차는 최대 반 칸
    assert np.all(np.abs(x2 - xs) <= spec.resolution / 2 + 1e-9)
    assert np.all(np.abs(y2 - ys) <= spec.resolution / 2 + 1e-9)
    print("월드->격자->월드 왕복 OK (오차 <= 반 칸)")
    print("  (x=-0.026, y=-0.026) ->", (int(r[4]), int(c[4])),
          "  <- floor 덕분에 0이 아닌 음수쪽 칸으로 감 (int() 였으면 틀림)")

    # --- (4) Bresenham 확인 ------------------------------------------------
    rr, cc = bresenham_line(0, 0, 3, 7)
    print("bresenham (0,0)->(3,7):", list(zip(rr.tolist(), cc.tolist())))
    assert (rr[0], cc[0]) == (0, 0) and (rr[-1], cc[-1]) == (3, 7)

    # --- 시각화: 가짜 LiDAR 점을 세 좌표계로 그려보기 ---------------------------
    angles = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    ranges = 1.5 + 0.5 * np.cos(3 * angles)     # 그냥 예쁜 모양의 가짜 거리
    pts_robot = lidar_to_robot(angles, ranges)
    pts_world = robot_to_world(pts_robot, pose)
    rows, cols = spec.world_to_grid(pts_world[:, 0], pts_world[:, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    ax.set_title("로봇 좌표계 (x=앞, y=왼쪽)")
    ax.scatter(pts_robot[:, 0], pts_robot[:, 1], c=angles, cmap="hsv")
    ax.arrow(0, 0, 0.5, 0, head_width=0.1, color="k")
    ax.set_aspect("equal"); ax.grid(True); ax.set_xlabel("x_r"); ax.set_ylabel("y_r")

    ax = axes[1]
    ax.set_title(f"월드 좌표계  pose=({pose[0]}, {pose[1]}, {np.rad2deg(pose[2]):.0f}°)")
    ax.scatter(pts_world[:, 0], pts_world[:, 1], c=angles, cmap="hsv")
    ax.arrow(pose[0], pose[1], 0.5 * np.cos(pose[2]), 0.5 * np.sin(pose[2]),
             head_width=0.1, color="k")
    ax.set_aspect("equal"); ax.grid(True); ax.set_xlabel("x_w"); ax.set_ylabel("y_w")

    ax = axes[2]
    ax.set_title("격자 인덱스 (row=y, col=x)  ※ imshow origin='lower'")
    grid = np.zeros(spec.shape)
    grid[rows, cols] = 1
    rr_, cc_ = spec.world_to_grid(pose[0], pose[1])
    grid[rr_, cc_] = 2
    ax.imshow(grid, origin="lower", cmap="viridis", extent=spec.extent)
    ax.set_xlabel("x_w  (col 방향)"); ax.set_ylabel("y_w  (row 방향)")
    plt.tight_layout()
    plt.show()
