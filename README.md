# pnu-techweek-amr — 부산대 TECH WEEK 해커톤 AMR Search & Rescue

TurtleBot3 Burger (Webots R2025a) 로 미지의 아파트를 탐색해 빨간 사과를 찾아 도달한 뒤 시작점으로 복귀하는 자율주행 스택.

```
grid_nav/        알고리즘 (지도·frontier·A*·DWA·위치추정·비전·반응형). 각 파일 단독 실행 시 자체 테스트+그림
controllers/     Webots 컨트롤러. tb3_gridnav = 전체 스택, tb3_reactive = 강의 Algorithm 3, 나머지 = 강의 예제
worlds/          apartment.wbt(대회 원본), *_gridnav.wbt(개발용: Supervisor+지도 Display), breakroom_*(작은 테스트 방)
protos/          색 사과
docs/            BRANCHES.md(브랜치 규칙·소유), AUDIT.md(강의 대비 구현 감사·튠 대상), TODO.md
prefetch_assets.py   Webots 자산 캐시 미리 채우기 (macOS 필수)
```

- 알고리즘 설명·공부 순서·막히는 포인트: `grid_nav/README.md`
- 브랜치 규칙과 누가 어떤 파일을 고치는지: `docs/BRANCHES.md`
- Webots 없이 시뮬레이션: `python grid_nav/sim_demo.py --dwa --pedestrian --localize`

## 빠른 시작
```bash
pip install numpy scipy matplotlib opencv-python        # Python 3.10 권장
python prefetch_assets.py worlds/apartment.wbt
/Applications/Webots.app/Contents/MacOS/webots --stdout --stderr worlds/apartment_gridnav.wbt
```
Webots Preferences > General > Python command 에 위 패키지가 있는 파이썬 경로를 넣을 것.
