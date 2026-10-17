#!/usr/bin/env python3
"""tokencache — caching proxy for OpenAI-compatible AI providers.

Sits in front of an upstream (default: the local 9router gateway) and:

  * caches completions on disk (SQLite) so identical requests never re-spend tokens
  * survives provider death: on upstream failure it serves the cached answer instead
  * trips a circuit breaker per model so a dead provider is not hammered
  * de-duplicates identical in-flight requests (one upstream call, N waiters)
  * reports token and cost savings

Stdlib only. Point your agent at http://127.0.0.1:20129/v1 instead of the upstream.

    python3 tokencache.py --upstream http://127.0.0.1:20128 --port 20129

Then:  export OPENAI_BASE_URL=http://127.0.0.1:20129/v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "1.0.0"

# Request fields that define a distinct completion. Anything not listed here
# (stream, user, request ids) must NOT change the cache key.
KEY_FIELDS = ("model", "messages", "temperature", "max_tokens", "top_p",
              "top_k", "frequency_penalty", "presence_penalty", "stop",
              "tools", "tool_choice", "response_format", "seed", "n")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    fp          TEXT,
    model       TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    last_hit    REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0,
    ptok        INTEGER NOT NULL DEFAULT 0,
    ctok        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS cache_model  ON cache(model);
CREATE INDEX IF NOT EXISTS cache_last   ON cache(last_hit);
CREATE INDEX IF NOT EXISTS cache_fp     ON cache(fp);

CREATE TABLE IF NOT EXISTS breaker (
    model        TEXT PRIMARY KEY,
    fails        INTEGER NOT NULL DEFAULT 0,
    opened_until REAL NOT NULL DEFAULT 0,
    last_error   TEXT
);

CREATE TABLE IF NOT EXISTS stats (
    k TEXT PRIMARY KEY,
    v REAL NOT NULL DEFAULT 0
);
"""


def log(*a):
    print(f"[tokencache {time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)


class Store:
    """SQLite-backed cache. WAL + a lock; the workload is read-heavy."""

    def __init__(self, path: str, ttl_days: float, max_entries: int):
        self.path = path
        self.ttl = ttl_days * 86400 if ttl_days else 0
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        try:  # migrate pre-fp databases in place
            self.db.execute("ALTER TABLE cache ADD COLUMN fp TEXT")
        except sqlite3.OperationalError:
            pass
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.commit()

    # ---- cache ----------------------------------------------------------
    def get(self, key: str):
        with self._lock:
            row = self.db.execute(
                "SELECT key, model, response, created_at FROM cache WHERE key = ?", (key,)).fetchone()
            if not row:
                return None
            k, model, resp, created = row
            if self.ttl and time.time() - created > self.ttl:
                self.db.execute("DELETE FROM cache WHERE key = ?", (k,))
                self.db.commit()
                return None
            self.db.execute("UPDATE cache SET hits = hits + 1, last_hit = ? WHERE key = ?",
                            (time.time(), k))
            self.db.commit()
            return model, resp

    def get_by_fingerprint(self, fp: str):
        """Cross-model fallback: any cached answer for the same message set."""
        with self._lock:
            row = self.db.execute(
                "SELECT key, model, response, created_at FROM cache WHERE fp = ? "
                "ORDER BY last_hit DESC LIMIT 1", (fp,)).fetchone()
            if not row:
                return None
            k, model, resp, created = row
            if self.ttl and time.time() - created > self.ttl:
                return None
            self.db.execute("UPDATE cache SET hits = hits + 1, last_hit = ? WHERE key = ?",
                            (time.time(), k))
            self.db.commit()
            return model, resp

    def put(self, key: str, fp: str, model: str, response: str, ptok: int, ctok: int):
        with self._lock:
            now = time.time()
            self.db.execute(
                "INSERT INTO cache(key, fp, model, response, created_at, last_hit, ptok, ctok) "
                "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "fp=excluded.fp, response=excluded.response, created_at=excluded.created_at, "
                "ptok=excluded.ptok, ctok=excluded.ctok",
                (key, fp, model, response, now, now, ptok, ctok))
            self.db.commit()
            self._evict()

    def _evict(self):
        n = self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if n > self.max_entries:
            self.db.execute(
                "DELETE FROM cache WHERE key IN (SELECT key FROM cache "
                "ORDER BY last_hit ASC LIMIT ?)", (n - self.max_entries,))
            self.db.commit()

    # ---- circuit breaker -------------------------------------------------
    def breaker_state(self, model: str, cooldown: float, threshold: int):
        with self._lock:
            row = self.db.execute("SELECT fails, opened_until FROM breaker WHERE model = ?",
                                  (model,)).fetchone()
            if not row:
                return False, 0
            fails, until = row
            if until and time.time() < until:
                return True, until - time.time()
            return False, 0

    def breaker_fail(self, model: str, cooldown: float, threshold: int, err: str):
        with self._lock:
            row = self.db.execute("SELECT fails FROM breaker WHERE model = ?", (model,)).fetchone()
            fails = (row[0] if row else 0) + 1
            until = time.time() + cooldown if fails >= threshold else 0
            self.db.execute(
                "INSERT INTO breaker(model, fails, opened_until, last_error) VALUES(?,?,?,?) "
                "ON CONFLICT(model) DO UPDATE SET fails=excluded.fails, "
                "opened_until=excluded.opened_until, last_error=excluded.last_error",
                (model, fails, until, err[:300]))
            self.db.commit()
            return fails >= threshold

    def breaker_ok(self, model: str):
        with self._lock:
            self.db.execute("DELETE FROM breaker WHERE model = ?", (model,))
            self.db.commit()

    def breaker_table(self):
        with self._lock:
            return self.db.execute(
                "SELECT model, fails, opened_until, last_error FROM breaker "
                "ORDER BY fails DESC").fetchall()

    # ---- counters --------------------------------------------------------
    def bump(self, k: str, n: float = 1):
        with self._lock:
            self.db.execute(
                "INSERT INTO stats(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = v + ?",
                (k, n, n))
            self.db.commit()

    def counters(self):
        with self._lock:
            return dict(self.db.execute("SELECT k, v FROM stats").fetchall())

    def cache_info(self):
        with self._lock:
            n, hits, pt, ct = self.db.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits),0), COALESCE(SUM(ptok),0), "
                "COALESCE(SUM(ctok),0) FROM cache").fetchone()
            return {"entries": n, "served_hits": hits,
                    "ptok_stored": pt, "ctok_stored": ct,
                    "db_bytes": os.path.getsize(self.path) if os.path.exists(self.path) else 0}

    def close(self):
        with self._lock:
            try:
                self.db.close()
            except Exception:
                pass

    def purge(self, model: str | None = None):
        with self._lock:
            if model:
                cur = self.db.execute("DELETE FROM cache WHERE model = ?", (model,))
            else:
                cur = self.db.execute("DELETE FROM cache")
            self.db.commit()
            return cur.rowcount


def endpoint(base: str, path: str) -> str:
    """Join an upstream base with an API path.

    Tolerates a base that already carries the full endpoint, so
    `--upstream http://host:20128/v1` and `.../v1/chat/completions` both work.
    """
    b = base.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
            break
    return b + path


def sse_delta(line: str):
    """Parse one SSE line -> (model, usage, content, reasoning) or None.

    Reasoning models (gpt-oss, DeepSeek-R1 style) stream their thinking in
    `reasoning_content` and only later emit `content`. Both are collected.
    """
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        j = json.loads(body)
    except Exception:
        return None
    content, reasoning = "", ""
    for ch in j.get("choices", []):
        d = ch.get("delta") or ch.get("message") or {}
        content += d.get("content") or ""
        reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
    return j.get("model"), j.get("usage"), content, reasoning


def parse_sse(body: str):
    """Collapse a whole SSE body into (text, usage, model).

    Prefers the final answer; falls back to the reasoning text when the model
    was cut off mid-thought and never produced `content`.
    """
    parts, reason, usage, model = [], [], {}, None
    for line in (body or "").splitlines():
        r = sse_delta(line)
        if not r:
            continue
        m, u, c, rc = r
        model = m or model
        if u:
            usage = u
        if c:
            parts.append(c)
        if rc:
            reason.append(rc)
    return ("".join(parts) or "".join(reason)), usage, model


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"tokencache/{VERSION}"

    # injected by serve()
    store: Store
    upstream: str
    cfg: argparse.Namespace
    inflight: dict
    inflight_lock = threading.Lock()

    def log_message(self, fmt, *args):
        if self.cfg.verbose:
            log(fmt % args)

    # ---- helpers ---------------------------------------------------------
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, code, payload: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, code, obj, extra=None):
        self._send(code, json.dumps(obj).encode(), extra=extra)

    # ---- upstream --------------------------------------------------------
    def _upstream(self, payload: dict, stream: bool):
        url = endpoint(self.upstream, "/chat/completions")
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        auth = self.headers.get("Authorization")
        if auth:
            req.add_header("Authorization", auth)
        elif self.cfg.api_key:
            req.add_header("Authorization", "Bearer " + self.cfg.api_key)
        # localhost upstreams must never go through the ambient egress proxy
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=self.cfg.timeout)

    def _passthrough_stream(self, resp):
        """Relay SSE to the client and return the aggregated completion."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        parts, reason, usage, model = [], [], {}, None
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace")
                self._chunk(line.encode())
                r = sse_delta(line.strip())
                if not r:
                    continue
                m, u, c, rc = r
                model = m or model
                if u:
                    usage = u
                if c:
                    parts.append(c)
                if rc:
                    reason.append(rc)
        finally:
            self._chunk(b"")
        return ("".join(parts) or "".join(reason)), usage, model

    def _chunk(self, data: bytes):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")

    def _serve_cached(self, key, model, resp_json, stream, source):
        """Re-render a cached non-stream response, as SSE if the client asked."""
        try:
            obj = json.loads(resp_json)
        except Exception:
            return False
        obj["model"] = obj.get("model") or model
        hdrs = {"X-Tokencache": source, "X-Tokencache-Model": model}
        if not stream:
            self._send(200, json.dumps(obj).encode(), extra=hdrs)
            return True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.end_headers()
        msg = (obj.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        base = {"id": obj.get("id", "cache"), "object": "chat.completion.chunk",
                "created": int(obj.get("created") or time.time()),
                "model": obj.get("model")}
        def emit(delta, finish=None):
            j = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
            self._chunk(("data: " + json.dumps(j) + "\n\n").encode())
        emit({"role": "assistant"})
        for i in range(0, len(content), 512):
            emit({"content": content[i:i + 512]})
        emit({}, "stop")
        if obj.get("usage"):
            j = dict(base, choices=[], usage=obj["usage"])
            self._chunk(("data: " + json.dumps(j) + "\n\n").encode())
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")
        return True

    # ---- routes ----------------------------------------------------------
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/_health", "/health"):
            return self._json(200, {"ok": True, "version": VERSION, "upstream": self.upstream})
        if p in ("/_stats", "/stats"):
            return self._json(200, self.report())
        if p.startswith("/v1/models") or p == "/models":
            try:
                req = urllib.request.Request(
                    endpoint(self.upstream, "/models"))
                auth = self.headers.get("Authorization") or (
                    "Bearer " + self.cfg.api_key if self.cfg.api_key else None)
                if auth:
                    req.add_header("Authorization", auth)
                op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with op.open(req, timeout=self.cfg.timeout) as r:
                    return self._send(200, r.read())
            except Exception as e:
                return self._json(502, {"error": {"message": f"upstream models failed: {e}"}})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        p = self.path.split("?")[0]
        if p in ("/_purge", "/purge"):
            try:
                body = json.loads(self._body() or b"{}")
            except Exception:
                body = {}
            n = self.store.purge(body.get("model"))
            return self._json(200, {"purged": n})
        if p not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})

        try:
            payload = json.loads(self._body() or b"{}")
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})

        stream = bool(payload.get("stream"))
        model = payload.get("model") or ""
        key = cache_key(payload)
        fp = fingerprint(payload)

        # 1. exact cache hit
        hit = self.store.get(key)
        if hit and self.cfg.mode in ("cache-first", "cache-only"):
            self.store.bump("hits")
            self.store.bump("tokens_saved", tokens_of(hit[1]))
            return self._serve_cached(key, hit[0], hit[1], stream, "HIT")
        self.store.bump("requests")

        # 2. circuit breaker: skip a provider known to be down
        open_, left = self.store.breaker_state(model, self.cfg.breaker_cooldown,
                                               self.cfg.breaker_threshold)
        if open_ and self.cfg.mode != "passthrough":
            fb = self.store.get_by_fingerprint(fp) if self.cfg.fallback else None
            if fb:
                self.store.bump("hits")
                self.store.bump("served_by_fallback")
                self.store.bump("tokens_saved", tokens_of(fb[1]))
                return self._serve_cached(key, fb[0], fb[1], stream, "FALLBACK-BREAKER")
            self.store.bump("breaker_rejected")
            return self._json(503, {"error": {
                "message": f"circuit open for {model}, retry in {left:.0f}s "
                           f"and no cached answer available"}})

        # 3. de-duplicate identical in-flight requests: one upstream call, N waiters
        is_leader = False
        with self.inflight_lock:
            ev = self.inflight.get(key)
            if ev is None:
                ev = self.inflight[key] = threading.Event()
                is_leader = True
        if not is_leader:
            ev.wait(timeout=self.cfg.timeout)
            self.store.bump("dedup_waiters")
            hit = self.store.get(key)
            if hit:
                self.store.bump("hits")
                self.store.bump("tokens_saved", tokens_of(hit[1]))
                return self._serve_cached(key, hit[0], hit[1], stream, "HIT-DEDUP")

        if self.cfg.mode == "cache-only":
            self.store.bump("misses")
            self._release(key, ev, is_leader)
            return self._json(503, {"error": {"message": "cache-only mode: no entry"}})

        try:
            return self._upstream_path(payload, key, fp, model, stream)
        finally:
            self._release(key, ev, is_leader)

    def _release(self, key, ev, is_leader):
        if is_leader:
            with self.inflight_lock:
                self.inflight.pop(key, None)
            ev.set()

    def _upstream_path(self, payload, key, fp, model, stream):
        # 4. upstream
        try:
            resp = self._upstream(payload, stream)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            tripped = self.store.breaker_fail(model, self.cfg.breaker_cooldown,
                                              self.cfg.breaker_threshold,
                                              f"{e.code}: {body[:200]}")
            self.store.bump("upstream_errors")
            fb = self.store.get_by_fingerprint(fp) if self.cfg.fallback else None
            if fb:
                self.store.bump("hits")
                self.store.bump("served_by_fallback")
                self.store.bump("tokens_saved", tokens_of(fb[1]))
                return self._serve_cached(key, fb[0], fb[1], stream, "FALLBACK-ERROR")
            extra = {"X-Tokencache": "MISS", "X-Tokencache-Breaker": "OPEN" if tripped else "closed"}
            return self._send(e.code, body.encode(), extra=extra)
        except Exception as e:
            self.store.breaker_fail(model, self.cfg.breaker_cooldown,
                                    self.cfg.breaker_threshold, str(e))
            self.store.bump("upstream_errors")
            fb = self.store.get_by_fingerprint(fp) if self.cfg.fallback else None
            if fb:
                self.store.bump("hits")
                self.store.bump("served_by_fallback")
                return self._serve_cached(key, fb[0], fb[1], stream, "FALLBACK-ERROR")
            return self._json(502, {"error": {"message": f"upstream unreachable: {e}"}})

        self.store.breaker_ok(model)

        if stream:
            text, usage, m = self._passthrough_stream(resp)
            resp.close()
            if text:
                self._store_from_text(key, fp, m or model, text, usage)
            return
        raw = resp.read()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        resp.close()
        # Some upstreams stream even when stream=false was requested; others
        # lie about the content-type (text/event-stream with a plain JSON
        # body, sometimes with stray whitespace/HTML noise prepended).
        # Trust the body: find the first real JSON/SSE payload.
        ls = raw.lstrip()
        cut = min((i for i in (ls.find(b"{"), ls.find(b"["),
                               ls.find(b"data:")) if i != -1), default=-1)
        payload = ls[cut:] if cut > 0 else ls
        if payload.startswith(b"data:"):
            body = payload.decode("utf-8", "replace")
            text, usage, m = parse_sse(body)
            if text:
                self._store_from_text(key, fp, m or model, text, usage)
                self.store.bump("misses")
                obj = {"id": "streamed", "object": "chat.completion",
                       "created": int(time.time()), "model": m or model,
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": text}}],
                       "usage": usage}
                return self._send(200, json.dumps(obj).encode(),
                                  extra={"X-Tokencache": "MISS-STREAMED"})
            return self._send(200, raw)
        self.store.bump("misses")
        # some gateways prepend stray whitespace/HTML noise before the JSON
        # body; find the first "{" or "[" and parse from there
        try:
            obj = json.loads(payload)
        except Exception:
            # some gateways append an SSE tail ("data: [DONE]") after the
            # JSON body — decode the first JSON value and ignore the rest
            try:
                obj, _ = json.JSONDecoder().raw_decode(payload.decode("utf-8", "replace").lstrip())
            except Exception:
                return self._send(200, raw)
        self._store_from_obj(key, fp, model, obj)
        self._send(200, raw, extra={"X-Tokencache": "MISS"})

    def _store_from_obj(self, key, fp, model, obj):
        try:
            msg = (obj.get("choices") or [{}])[0].get("message") or {}
            txt = msg.get("content") or msg.get("reasoning_content") or ""
        except Exception:
            return
        usage = obj.get("usage") or {}
        if txt:
            self.store.put(key, fp, obj.get("model") or model, json.dumps(obj),
                           int(usage.get("prompt_tokens") or 0),
                           int(usage.get("completion_tokens") or 0))

    def _store_from_text(self, key, fp, model, text, usage):
        obj = {"id": "streamed", "object": "chat.completion", "created": int(time.time()),
               "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": text}}],
               "usage": usage}
        self.store.put(key, fp, model, json.dumps(obj),
                       int((usage or {}).get("prompt_tokens") or 0),
                       int((usage or {}).get("completion_tokens") or 0))

    # ---- report ----------------------------------------------------------
    def report(self):
        c = self.store.counters()
        req = c.get("requests", 0)
        hits = c.get("hits", 0)
        total = req + hits
        info = self.store.cache_info()
        saved = c.get("tokens_saved", 0)
        return {
            "version": VERSION,
            "upstream": self.upstream,
            "mode": self.cfg.mode,
            "requests": total,
            "hits": hits,
            "misses": c.get("misses", 0),
            "upstream_errors": c.get("upstream_errors", 0),
            "served_by_fallback": c.get("served_by_fallback", 0),
            "breaker_rejected": c.get("breaker_rejected", 0),
            "hit_rate": round(100 * hits / total, 1) if total else 0.0,
            "tokens_saved": saved,
            "cost_saved_usd_est": round(saved / 1_000_000 * self.cfg.price_per_mtok, 4),
            "cache": info,
            "breakers": [
                {"model": m, "fails": f,
                 "open_for_s": max(0, round(u - time.time())) if u else 0,
                 "last_error": e} for m, f, u, e in self.store.breaker_table()
            ],
        }


def canonical(payload: dict) -> dict:
    return {k: payload[k] for k in KEY_FIELDS if k in payload}


def cache_key(payload: dict) -> str:
    blob = json.dumps(canonical(payload), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def fingerprint(payload: dict) -> str:
    """Key ignoring the model, so any model's answer can serve as a fallback."""
    p = canonical(payload)
    p.pop("model", None)
    blob = json.dumps(p, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def tokens_of(resp_json: str) -> int:
    """Total tokens a cached response cost, for the savings counter."""
    try:
        u = json.loads(resp_json).get("usage") or {}
        return int(u.get("prompt_tokens") or 0) + int(u.get("completion_tokens") or 0)
    except Exception:
        return 0


def build_server(cfg):
    """Construct the proxy without entering its serve loop (used by tests)."""
    store = Store(cfg.db, cfg.ttl_days, cfg.max_entries)
    handler = type("BoundProxy", (Proxy,), {
        "store": store, "upstream": cfg.upstream, "cfg": cfg,
        "inflight": {}, "inflight_lock": threading.Lock(),
    })
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
    httpd.daemon_threads = True
    httpd.store = store
    return httpd


def serve(cfg):
    httpd = build_server(cfg)
    cfg.port = httpd.server_address[1]
    log(f"v{VERSION} listening on http://{cfg.host}:{cfg.port} -> {cfg.upstream}")
    log(f"mode={cfg.mode} ttl={cfg.ttl_days}d max_entries={cfg.max_entries} "
        f"fallback={cfg.fallback} db={cfg.db}")
    log(f"point your agent at http://{cfg.host}:{cfg.port}/v1")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        httpd.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Caching proxy for OpenAI-compatible providers")
    ap.add_argument("--host", default=os.environ.get("TOKENCACHE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("TOKENCACHE_PORT", 20129)))
    ap.add_argument("--upstream", default=os.environ.get("TOKENCACHE_UPSTREAM",
                                                         "http://127.0.0.1:20128/v1"),
                    help="OpenAI-compatible base URL, including the version segment")
    ap.add_argument("--db", default=os.environ.get("TOKENCACHE_DB", "tokencache.sqlite"))
    ap.add_argument("--ttl-days", type=float, default=float(os.environ.get("TOKENCACHE_TTL_DAYS", 30)))
    ap.add_argument("--max-entries", type=int, default=int(os.environ.get("TOKENCACHE_MAX", 20000)))
    ap.add_argument("--mode", choices=["cache-first", "cache-only", "passthrough"],
                    default=os.environ.get("TOKENCACHE_MODE", "cache-first"),
                    help="cache-first: serve hits, else upstream. cache-only: never call upstream "
                         "(survive total provider outage). passthrough: always call upstream, still store.")
    ap.add_argument("--no-fallback", dest="fallback", action="store_false", default=True,
                    help="disable cross-model fallback to cached answers")
    ap.add_argument("--breaker-threshold", type=int, default=3,
                    help="consecutive upstream failures before a model's circuit opens")
    ap.add_argument("--breaker-cooldown", type=float, default=300,
                    help="seconds a circuit stays open")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--api-key", default=os.environ.get("TOKENCACHE_API_KEY", ""))
    ap.add_argument("--price-per-mtok", type=float,
                    default=float(os.environ.get("TOKENCACHE_PRICE_PER_MTOK", 1.0)),
                    help="USD per 1M tokens, only used for the savings estimate")
    ap.add_argument("--stats", action="store_true", help="print stats and exit")
    ap.add_argument("--purge", action="store_true", help="clear the cache and exit")
    ap.add_argument("-v", "--verbose", action="store_true")
    cfg = ap.parse_args(argv)

    if cfg.stats or cfg.purge:
        store = Store(cfg.db, cfg.ttl_days, cfg.max_entries)
        if cfg.purge:
            print(f"purged {store.purge()} entries")
        c = store.counters()
        hits, req = c.get("hits", 0), c.get("requests", 0)
        tot = hits + req
        print(json.dumps({
            "entries": store.cache_info()["entries"],
            "requests": tot, "hits": hits,
            "hit_rate": round(100 * hits / tot, 1) if tot else 0.0,
            "tokens_saved": c.get("tokens_saved", 0),
            "upstream_errors": c.get("upstream_errors", 0),
            "served_by_fallback": c.get("served_by_fallback", 0),
        }, indent=2))
        return 0
    serve(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
