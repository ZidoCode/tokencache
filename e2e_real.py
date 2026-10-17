"""Real end-to-end: proxy in front of the actual 9router gateway.

Proves the cache works against a live provider, not just the stub.
Run: python3 e2e_real.py
"""
import argparse
import json
import os
import re
import tempfile
import threading
import time
import urllib.request

import tokencache as tc

MODEL = "ag/gpt-oss-120b-medium"
REQ = {"model": MODEL, "max_tokens": 400,
       "messages": [{"role": "user", "content": "Reply with exactly: PONG"}]}


def api_key():
    p = os.path.expanduser("~/.hermes/.env")
    m = re.search(r'NINE_ROUTER_API_KEY\s*=\s*["\']?([^"\'\n]+)', open(p).read())
    return m.group(1) if m else ""


def post(port, payload, key):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    try:
        with op.open(req, timeout=120) as r:
            return r.status, r.headers.get("X-Tokencache"), r.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("X-Tokencache"), e.read().decode(), time.time() - t0


def get(port, path):
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(f"http://127.0.0.1:{port}{path}", timeout=30) as r:
        return json.loads(r.read())


def main():
    tmp = tempfile.TemporaryDirectory()
    cfg = argparse.Namespace(
        host="127.0.0.1", port=0, upstream="http://127.0.0.1:20128/v1",
        db=os.path.join(tmp.name, "real.sqlite"), ttl_days=30, max_entries=1000,
        mode="cache-first", fallback=True, breaker_threshold=3, breaker_cooldown=300,
        timeout=180, api_key="", price_per_mtok=1.0, verbose=False)
    httpd = tc.build_server(cfg)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    key = api_key()
    print(f"proxy on 127.0.0.1:{port} -> {cfg.upstream}   model={MODEL}\n")

    try:
        s1, c1, b1, d1 = post(port, REQ, key)
        print(f"call 1: status={s1} cache={c1} {d1:.2f}s  body={(b1 or '')[:90]!r}")
        s2, c2, b2, d2 = post(port, REQ, key)
        print(f"call 2: status={s2} cache={c2} {d2:.2f}s  body={(b2 or '')[:90]!r}")
        print(f"call 3 (stream): ", end="")
        s3, c3, b3, d3 = post(port, dict(REQ, stream=True), key)
        print(f"status={s3} cache={c3} {d3:.2f}s  sse={('data: ' in (b3 or ''))} "
              f"done={'[DONE]' in (b3 or '')}")

        st = get(port, "/_stats")
        print("\n--- /_stats ---")
        print(json.dumps({k: st[k] for k in
                          ("requests", "hits", "misses", "hit_rate", "tokens_saved",
                           "upstream_errors", "served_by_fallback")}, indent=2))
        print("cache:", st["cache"])

        ok = (s1 == 200 and c1 in ("MISS", "MISS-STREAMED") and s2 == 200 and c2 == "HIT"
              and st["hits"] >= 2)
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.store.close()
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
