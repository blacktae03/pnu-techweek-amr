"""
sound_localization.py - "소리" 로 구조 대상 방향·거리 추정 (스토리 기능, 브랜치 feat/sound)

Webots 에는 마이크가 없다. 그래서 사람의 외침을 Emitter(전파, radio 채널 1) 발신으로, 로봇의 귀를 Receiver 로 흉내낸다.
Receiver 가 주는 두 값이 소리의 두 단서에 대응한다:
  getEmitterDirection() : 수신기 좌표계에서 본 발신원 방향 벡터  → 소리가 들리는 방향(bearing)
  getSignalStrength()   : 1 / 거리²  (Webots 문서)                   → 소리 크기 = 거리
발신원 월드 좌표 = 로봇 pose + 거리·방향. 한 번의 수신은 노이즈가 있으니 최근 N 개의 중앙값을 쓴다.

주의: radio 신호는 벽을 통과한다(소리도 어느 정도 그렇다). infra-red 타입으로 바꾸면 벽에 막힌다.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from geometry import wrap_angle

STRENGTH_TO_DIST_K = 1.0     # dist = sqrt(K / strength). Webots 기본은 K=1 (strength = 1/d²). 실측으로 보정.


class SoundListener:
    def __init__(self, receiver, timestep_ms, history=5, k=STRENGTH_TO_DIST_K):
        self.rx = receiver
        self.rx.enable(timestep_ms)
        self.k = k
        self.hist = deque(maxlen=history)      # (x, y) 추정들
        self.estimate: Optional[Tuple[float, float]] = None
        self.last_t = -1e9
        self.count = 0
        self.last_bearing = 0.0
        self.last_dist = 0.0

    def update(self, pose, t) -> Optional[Tuple[float, float]]:
        """큐에 쌓인 패킷을 모두 읽고 추정을 갱신. 새 패킷이 있었으면 최신 추정을 반환, 없으면 None."""
        got = False
        while self.rx.getQueueLength() > 0:
            strength = self.rx.getSignalStrength()
            d = self.rx.getEmitterDirection()          # 수신기(=로봇) 좌표계 단위 벡터
            self.rx.nextPacket()
            if strength <= 0:
                continue
            dist = math.sqrt(self.k / strength)
            bearing = math.atan2(d[1], d[0])           # 로봇 정면 기준, 왼쪽 +
            x = pose[0] + dist * math.cos(pose[2] + bearing)
            y = pose[1] + dist * math.sin(pose[2] + bearing)
            self.hist.append((x, y))
            self.last_bearing, self.last_dist = bearing, dist
            self.last_t = t
            self.count += 1
            got = True
        if got:
            arr = np.array(self.hist)
            self.estimate = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
            return self.estimate
        return None


def make_song_wav(path, seconds=8.0, rate=16000):
    """저작권 없는 짧은 멜로디(단순 음계 아르페지오) 를 16-bit PCM wav 로 생성. Speaker.playSound 용."""
    import wave
    notes = [262, 330, 392, 523, 392, 330, 262, 196]        # C4 E4 G4 C5 G4 E4 C4 G3
    per = seconds / len(notes)
    t = np.arange(int(rate * per)) / rate
    env = np.minimum(1.0, np.minimum(t / 0.02, (per - t) / 0.05)).clip(0, 1)     # 클릭 방지 페이드
    sig = np.concatenate([0.4 * np.sin(2 * np.pi * f * t) * env for f in notes])
    pcm = (sig * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    return path
