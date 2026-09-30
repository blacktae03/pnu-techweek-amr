"""
prefetch_assets.py - 월드가 참조하는 GitHub 자산(PROTO, 텍스처, 메시)을 Webots 캐시에 미리 채워 넣는다.

왜: macOS Webots 가 raw.githubusercontent.com 에서 파일을 받을 때 HTTP/2 오류("Connection closed",
    "header compression context")로 자주 실패해 벽/가구가 통째로 빠진 채 월드가 뜬다.
    Webots 캐시(~/Library/Caches/Cyberbotics/Webots/assets/<sha1(url)>)에 파일이 있으면 다운로드하지 않으므로,
    curl 같은 안정적인 방법으로 미리 채워 두면 오프라인에서도 월드가 완전하게 뜬다. (대회장 네트워크 대비)

사용:  python prefetch_assets.py worlds/apartment.wbt [worlds/breakroom_teleop.wbt ...]
"""
import hashlib
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CACHE = os.path.expanduser("~/Library/Caches/Cyberbotics/Webots/assets")
os.makedirs(CACHE, exist_ok=True)
URL_RE = re.compile(r'"((?:https?://[^"]+)|(?:(?:\.\./|\./)?[\w./-]+\.(?:proto|jpg|jpeg|png|hdr|obj|dae|stl|wav|mp3|wbo)))"')
TEXT_EXT = (".proto", ".wbt", ".wbo")
seen, ok, fail = set(), [], []


def cache_path(url):
    return os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest())


def fetch(url):
    path = cache_path(url)
    if os.path.exists(path):
        return open(path, "rb").read()
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    data = urllib.request.urlopen(req, timeout=30).read()
    with open(path, "wb") as f:
        f.write(data)
    return data


def crawl(url):
    if url in seen or "raw.githubusercontent.com" not in url:
        return
    seen.add(url)
    try:
        data = fetch(url)
        ok.append(url)
    except Exception as e:
        fail.append((url, str(e)))
        return
    if url.endswith(TEXT_EXT):
        base = url.rsplit("/", 1)[0] + "/"
        for ref in URL_RE.findall(data.decode("utf-8", "ignore")):
            if ref.startswith("http"):
                child = ref
            elif ref.startswith("webots://"):
                continue
            else:
                child = urllib.parse.urljoin(base, ref)
            crawl(child)


import urllib.parse
for wbt in sys.argv[1:] or ["worlds/apartment.wbt"]:
    text = open(wbt, encoding="utf-8").read()
    roots = [u for u in re.findall(r'"(https://raw\.githubusercontent\.com[^"]+)"', text)]
    print(f"{wbt}: 최상위 URL {len(roots)}개")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(crawl, roots))
print(f"캐시 확보 {len(ok)}개, 실패 {len(fail)}개")
for u, e in fail[:10]:
    print("  실패:", u, e)
