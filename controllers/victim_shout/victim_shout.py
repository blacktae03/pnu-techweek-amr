"""
victim_shout.py - 구조 대상(사람) 이 일정 시간 뒤부터 2 초마다 "HELP" 를 외치는 컨트롤러 (스토리 기능)

Webots 에 마이크/음파는 없으므로 Emitter(radio 채널 1) 발신을 소리로 간주한다. 로봇의 Receiver 가 방향과 세기를 받는다.
SHOUT_START_S 뒤부터 발신하는 이유: 로봇이 조금 탐색한 뒤 노래를 듣고 사람이 소리치는 연출.
Pedestrian PROTO 의 bodySlot 에 같은 Emitter 를 넣으면 움직이는 사람이 소리치게 할 수도 있다 (이번엔 고정 노드).
"""
import os
from controller import Robot

SHOUT_START_S = float(os.environ.get("SHOUT_START_S", "20"))
SHOUT_PERIOD_S = 2.0

robot = Robot()
ts = int(robot.getBasicTimeStep())
emitter = robot.getDevice("mouth")
last = -1e9
while robot.step(ts) != -1:
    t = robot.getTime()
    if t >= SHOUT_START_S and t - last >= SHOUT_PERIOD_S:
        emitter.send(b"HELP")
        last = t
        print(f"[victim {t:.1f}s] HELP! (emitter send)")
