"""Run: python3 -m unittest test_tokencache -v

Covers the cache store, key derivation, circuit breaker, and a full end-to-end
pass through the proxy against a stub upstream (no real provider is called).
"""
import argparse
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tokencache as tc


# --------------------------------------------------------------------------
# stub upstream
# --------------------------------------------------------------------------
class Upstream:
    def __init__(self):
        self.calls = 0
        self.fail_code = 0
        self.delay = 0.0
        self.force_stream = False
        self.last_payload = None
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "stub-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse(self, msg):
                def chunk(delta, finish=None):
                    return ("data: " + json.dumps({
                        "id": "cmpl-stub", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": "stub-model",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }) + "\n\n").encode()
                body = chunk({"role": "assistant"})
                body += chunk({"content": "echo:" + msg})
                body += chunk({}, "stop")
                body += ("data: " + json.dumps({
                    "id": "cmpl-stub", "object": "chat.completion.chunk",
                    "model": "stub-model", "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }) + "\n\n").encode()
                body += b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                outer.calls += 1
                outer.last_payload = payload
                if outer.delay:
                    time.sleep(outer.delay)
                msg = (payload.get("messages") or [{}])[-1].get("content", "")
                if outer.force_stream:
                    self._sse(msg)
                    return
                if outer.fail_code:
                    body = json.dumps({"error": {"message": "stub failure"}}).encode()
                    self.send_response(outer.fail_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps({
                    "id": "cmpl-stub", "object": "chat.completion",
                    "created": int(time.time()), "model": payload.get("model", "stub-model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "echo:" + msg}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_cfg(**kw):
    cfg = argparse.Namespace(
        host="127.0.0.1", port=0, upstream="", db="", ttl_days=30, max_entries=1000,
        mode="cache-first", fallback=True, breaker_threshold=3, breaker_cooldown=300,
        timeout=30, api_key="", price_per_mtok=1.0, verbose=False)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class Harness:
    """Stub upstream + proxy on ephemeral ports, torn down on exit."""

    def __init__(self, **cfgkw):
        self.tmp = tempfile.TemporaryDirectory()
        self.up = Upstream()
        cfg = make_cfg(upstream=f"http://127.0.0.1:{self.up.port}/v1",
                       db=os.path.join(self.tmp.name, "c.sqlite"), **cfgkw)
        self.cfg = cfg
        self.httpd = tc.build_server(cfg)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def post(self, payload, timeout=30):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer test"})
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with op.open(req, timeout=timeout) as r:
                return r.status, dict(r.headers), r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read().decode()

    def get(self, path, timeout=30):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}")
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.httpd.store.close()
        self.up.stop()
        self.tmp.cleanup()


REQ = {"model": "m1", "messages": [{"role": "user", "content": "halo"}]}


# --------------------------------------------------------------------------
class TestKeys(unittest.TestCase):
    def test_same_request_same_key(self):
        self.assertEqual(tc.cache_key(REQ), tc.cache_key(dict(REQ)))

    def test_stream_flag_does_not_change_key(self):
        a = dict(REQ, stream=True)
        self.assertEqual(tc.cache_key(REQ), tc.cache_key(a))

    def test_volatile_fields_do_not_change_key(self):
        self.assertEqual(tc.cache_key(REQ), tc.cache_key(dict(REQ, user="abc", request_id="xyz")))

    def test_different_model_changes_key(self):
        self.assertNotEqual(tc.cache_key(REQ), tc.cache_key(dict(REQ, model="m2")))

    def test_different_messages_change_key(self):
        other = dict(REQ, messages=[{"role": "user", "content": "beda"}])
        self.assertNotEqual(tc.cache_key(REQ), tc.cache_key(other))

    def test_fingerprint_ignores_model(self):
        self.assertEqual(tc.fingerprint(REQ), tc.fingerprint(dict(REQ, model="other")))

    def test_fingerprint_tracks_messages(self):
        other = dict(REQ, messages=[{"role": "user", "content": "beda"}])
        self.assertNotEqual(tc.fingerprint(REQ), tc.fingerprint(other))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = tc.Store(os.path.join(self.tmp.name, "s.sqlite"), 30, 3)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_put_get_and_hit_counter(self):
        self.s.put("k", "fp", "m", '{"a":1}', 10, 5)
        self.assertEqual(self.s.get("k")[1], '{"a":1}')
        self.assertEqual(self.s.cache_info()["served_hits"], 1)

    def test_miss_returns_none(self):
        self.assertIsNone(self.s.get("nope"))

    def test_ttl_expiry(self):
        self.s.ttl = -1
        self.s.put("k", "fp", "m", "{}", 0, 0)
        self.assertIsNone(self.s.get("k"))

    def test_eviction_respects_max_entries(self):
        for i in range(6):
            self.s.put(f"k{i}", "fp", "m", "{}", 0, 0)
            self.s.db.execute("UPDATE cache SET last_hit = ? WHERE key = ?", (i, f"k{i}"))
            self.s.db.commit()
        self.assertLessEqual(self.s.cache_info()["entries"], 3)

    def test_purge_all_and_by_model(self):
        self.s.put("a", "fp", "m1", "{}", 0, 0)
        self.s.put("b", "fp", "m2", "{}", 0, 0)
        self.assertEqual(self.s.purge("m1"), 1)
        self.assertEqual(self.s.cache_info()["entries"], 1)
        self.assertEqual(self.s.purge(), 1)

    def test_fingerprint_lookup_finds_other_model(self):
        fp = tc.fingerprint(REQ)
        self.s.put("key-for-m1", fp, "m1", '{"ok":1}', 0, 0)
        self.assertEqual(self.s.get_by_fingerprint(fp)[0], "m1")

    def test_fingerprint_lookup_misses_other_messages(self):
        self.s.put("key-for-m1", tc.fingerprint(REQ), "m1", '{"ok":1}', 0, 0)
        other = dict(REQ, messages=[{"role": "user", "content": "lain"}])
        self.assertIsNone(self.s.get_by_fingerprint(tc.fingerprint(other)))


class TestBreaker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = tc.Store(os.path.join(self.tmp.name, "b.sqlite"), 30, 100)

    def tearDown(self):
        self.s.close()
        self.tmp.cleanup()

    def test_closed_initially(self):
        self.assertEqual(self.s.breaker_state("m", 300, 3), (False, 0))

    def test_opens_at_threshold(self):
        for _ in range(3):
            tripped = self.s.breaker_fail("m", 300, 3, "boom")
        self.assertTrue(tripped)
        open_, left = self.s.breaker_state("m", 300, 3)
        self.assertTrue(open_)
        self.assertGreater(left, 0)

    def test_cooldown_expiry_reopens_circuit(self):
        for _ in range(3):
            self.s.breaker_fail("m", -1, 3, "boom")
        self.assertEqual(self.s.breaker_state("m", -1, 3), (False, 0))

    def test_success_resets(self):
        for _ in range(3):
            self.s.breaker_fail("m", 300, 3, "boom")
        self.s.breaker_ok("m")
        self.assertEqual(self.s.breaker_state("m", 300, 3), (False, 0))


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_second_request_is_a_cache_hit(self):
        s1, h1, b1 = self.h.post(REQ)
        self.assertEqual(s1, 200)
        self.assertEqual(h1.get("X-Tokencache"), "MISS")
        self.assertEqual(self.h.up.calls, 1)

        s2, h2, b2 = self.h.post(REQ)
        self.assertEqual(s2, 200)
        self.assertEqual(h2.get("X-Tokencache"), "HIT")
        self.assertEqual(b2, b1, "cached body must be byte-identical")
        self.assertEqual(self.h.up.calls, 1, "upstream must not be called twice")

    def test_different_prompt_is_a_miss(self):
        self.h.post(REQ)
        self.h.post(dict(REQ, messages=[{"role": "user", "content": "lain"}]))
        self.assertEqual(self.h.up.calls, 2)

    def test_stats_reports_savings(self):
        self.h.post(REQ)
        self.h.post(REQ)
        _, st = self.h.get("/_stats")
        self.assertEqual(st["hits"], 1)
        self.assertEqual(st["tokens_saved"], 15)
        self.assertEqual(st["hit_rate"], 50.0)

    def test_fallback_serves_cache_when_upstream_dies(self):
        self.h.post(REQ)                       # prime the cache
        self.h.up.fail_code = 503
        self.h.post(dict(REQ, model="m2"))     # different model -> different key
        s, h, b = self.h.post(dict(REQ, model="m2"))
        self.assertEqual(s, 200)
        self.assertEqual(h.get("X-Tokencache"), "FALLBACK-ERROR")

    def test_breaker_opens_and_rejects_when_no_cache(self):
        self.h.up.fail_code = 503
        codes = [self.h.post(dict(REQ, messages=[{"role": "user", "content": f"q{i}"}]))[0]
                 for i in range(4)]
        self.assertIn(503, codes)
        _, st = self.h.get("/_stats")
        self.assertTrue(st["breakers"], "a breaker row should exist")
        self.assertGreaterEqual(st["breakers"][0]["fails"], 3)

    def test_purge_endpoint(self):
        self.h.post(REQ)
        req = urllib.request.Request(f"http://127.0.0.1:{self.h.port}/_purge",
                                     data=b"{}", headers={"Content-Type": "application/json"})
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(req, timeout=10) as r:
            self.assertEqual(json.loads(r.read())["purged"], 1)

    def test_stream_hit_is_rendered_as_sse(self):
        self.h.post(REQ)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.h.port}/v1/chat/completions",
            data=json.dumps(dict(REQ, stream=True)).encode(),
            headers={"Content-Type": "application/json"})
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(req, timeout=15) as r:
            body = r.read().decode()
            self.assertEqual(r.headers.get("X-Tokencache"), "HIT")
        self.assertIn("data: ", body)
        self.assertIn("[DONE]", body)
        self.assertIn("echo:halo", body)

    def test_concurrent_identical_requests_hit_upstream_once(self):
        self.h.up.delay = 0.4
        results = []

        def go():
            results.append(self.h.post(REQ)[0])

        ts = [threading.Thread(target=go) for _ in range(5)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(results, [200] * 5)
        self.assertEqual(self.h.up.calls, 1, "dedup must collapse 5 identical calls into 1")

    def test_health(self):
        s, j = self.h.get("/_health")
        self.assertEqual(s, 200)
        self.assertTrue(j["ok"])


class TestModeCacheOnly(unittest.TestCase):
    def setUp(self):
        self.h = Harness(mode="cache-only", fallback=False)

    def tearDown(self):
        self.h.close()

    def test_never_calls_upstream(self):
        s, _, _ = self.h.post(REQ)
        self.assertEqual(s, 503)
        self.assertEqual(self.h.up.calls, 0)


class TestSSEParsing(unittest.TestCase):
    LINE = ('data: {"model":"m","choices":[{"delta":{"content":"hi"}}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":2}}')

    def test_sse_delta_extracts_content(self):
        self.assertEqual(tc.sse_delta(self.LINE),
                         ("m", {"prompt_tokens": 3, "completion_tokens": 2}, "hi", ""))

    def test_sse_delta_extracts_reasoning(self):
        line = 'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}'
        self.assertEqual(tc.sse_delta(line)[3], "think")

    def test_sse_delta_ignores_done(self):
        self.assertIsNone(tc.sse_delta("data: [DONE]"))

    def test_sse_delta_ignores_non_data(self):
        self.assertIsNone(tc.sse_delta(": keep-alive"))

    def test_sse_delta_ignores_bad_json(self):
        self.assertIsNone(tc.sse_delta("data: {not json"))

    def test_parse_sse_joins_content_and_keeps_usage(self):
        body = ("data: {\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"PO\"}}]}\n\n"
                "data: {\"choices\":[{\"delta\":{\"content\":\"NG\"}}]}\n\n"
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\n"
                "data: [DONE]\n\n")
        text, usage, model = tc.parse_sse(body)
        self.assertEqual(text, "PONG")
        self.assertEqual(usage["prompt_tokens"], 7)
        self.assertEqual(model, "m")

    def test_parse_sse_reads_message_form_too(self):
        body = 'data: {"choices":[{"message":{"content":"full"}}]}\n\n'
        self.assertEqual(tc.parse_sse(body)[0], "full")

    def test_parse_sse_falls_back_to_reasoning_when_cut_off(self):
        body = ('data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning_content":"ing"}}]}\n\n')
        self.assertEqual(tc.parse_sse(body)[0], "thinking")

    def test_parse_sse_prefers_content_over_reasoning(self):
        body = ('data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
        self.assertEqual(tc.parse_sse(body)[0], "answer")


class TestUpstreamStreamsAnyway(unittest.TestCase):
    """The real 9router streams even when stream=false; the cache must still fill."""

    def setUp(self):
        self.h = Harness()

    def tearDown(self):
        self.h.close()

    def test_streaming_upstream_is_cached_and_replayed(self):
        self.h.up.force_stream = True
        s1, c1, b1 = self.h.post(REQ)
        self.assertEqual(s1, 200)
        self.assertEqual(c1.get("X-Tokencache"), "MISS-STREAMED")
        self.assertIn("echo:halo", b1)

        s2, c2, b2 = self.h.post(REQ)
        self.assertEqual(s2, 200)
        self.assertEqual(c2.get("X-Tokencache"), "HIT", "second call must be served from cache")
        self.assertEqual(self.h.up.calls, 1)

    def test_streaming_upstream_hit_serves_sse_when_asked(self):
        self.h.up.force_stream = True
        self.h.post(REQ)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.h.port}/v1/chat/completions",
            data=json.dumps(dict(REQ, stream=True)).encode(),
            headers={"Content-Type": "application/json"})
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with op.open(req, timeout=15) as r:
            self.assertEqual(r.headers.get("X-Tokencache"), "HIT")
            body = r.read().decode()
        self.assertIn("echo:halo", body)
        self.assertIn("[DONE]", body)
        self.assertEqual(self.h.up.calls, 1)


class TestEndpointJoin(unittest.TestCase):
    def test_base_with_version(self):
        self.assertEqual(tc.endpoint("http://h:1/v1", "/chat/completions"),
                         "http://h:1/v1/chat/completions")

    def test_base_without_version(self):
        self.assertEqual(tc.endpoint("http://h:1", "/models"), "http://h:1/models")

    def test_base_already_has_endpoint(self):
        self.assertEqual(tc.endpoint("http://h:1/v1/chat/completions", "/chat/completions"),
                         "http://h:1/v1/chat/completions")

    def test_trailing_slash(self):
        self.assertEqual(tc.endpoint("http://h:1/v1/", "/models"), "http://h:1/v1/models")


class TestTokens(unittest.TestCase):
    def test_sums_prompt_and_completion(self):
        self.assertEqual(tc.tokens_of('{"usage":{"prompt_tokens":10,"completion_tokens":5}}'), 15)

    def test_missing_usage_is_zero(self):
        self.assertEqual(tc.tokens_of('{"choices":[]}'), 0)

    def test_bad_json_is_zero(self):
        self.assertEqual(tc.tokens_of("not json"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
