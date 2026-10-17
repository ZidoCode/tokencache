# tokencache

Caching proxy for OpenAI-compatible providers. Sits in front of the 9router
gateway, stores completions on disk, and keeps your agent answering when a
provider dies.

Stdlib only. No dependencies, no build step.

## Why

Every identical request to a provider is paid for again. And when a provider's
credentials or quota run out, the agent stops working entirely. This fixes both:
identical requests are served from disk, and a dead provider falls back to a
cached answer instead of returning an error.

## Run

```bash
python3 tokencache.py                    # listens on 127.0.0.1:20129
```

Then point the agent at the proxy instead of the gateway:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:20129/v1
export OPENAI_API_KEY=$NINEROUTER_API_KEY     # forwarded upstream as-is
```

In Hermes `config.yaml`, set the model's `base_url` to
`http://127.0.0.1:20129/v1`. The proxy forwards the `Authorization` header it
receives, so the same key keeps working.

## Modes

| Mode | Behaviour |
|---|---|
| `cache-first` (default) | serve a hit, otherwise call upstream and store the result |
| `cache-only` | never call upstream. Survives a total provider outage on cached traffic only |
| `passthrough` | always call upstream, still store (use to warm the cache) |

## What it does when a provider dies

1. **Circuit breaker** — after N consecutive upstream failures a model's circuit
   opens for a cooldown, so a dead provider is not hammered on every request.
2. **Cross-model fallback** — while a circuit is open, a request for model B that
   matches the message set of a cached answer from model A is served from that
   cache, tagged `X-Tokencache: FALLBACK-*`.

## Observability

Every response carries `X-Tokencache: HIT | MISS | MISS-STREAMED | FALLBACK-*`.

```bash
curl -s localhost:20129/_stats | python3 -m json.tool
```

Reports hit rate, tokens saved, upstream errors, breaker state, and estimated
cost saved (`--price-per-mtok`).

```bash
python3 tokencache.py --stats     # counters from the db, no server needed
python3 tokencache.py --purge     # wipe the cache
```

## Options

```
--upstream URL        base URL incl. version segment (default http://127.0.0.1:20128/v1)
--port N              listen port (default 20129)
--db PATH             sqlite file (default tokencache.sqlite)
--ttl-days N          entry lifetime (default 30)
--max-entries N       LRU eviction ceiling (default 20000)
--mode MODE           cache-first | cache-only | passthrough
--no-fallback         disable cross-model fallback
--breaker-threshold N consecutive failures before opening (default 3)
--breaker-cooldown N  seconds a circuit stays open (default 300)
--price-per-mtok N    USD per 1M tokens, for the savings estimate only
```

All of these read from `TOKENCACHE_*` environment variables too.

## Correctness notes

- The cache key covers `model`, `messages`, `temperature`, `max_tokens`, `top_p`,
  `tools`, `response_format`, `seed`, and friends. It deliberately ignores
  `stream`, `user`, and request ids, so a streamed and a non-streamed call for the
  same prompt share one entry. A hit is re-rendered as SSE when the client asks
  for a stream.
- This gateway streams even when `stream: false` is requested, and reasoning
  models put their output in `delta.reasoning_content`. Both are handled: the
  response is reassembled and stored either way.
- Only responses with non-empty text are cached, so a failed or empty completion
  never poisons the cache.

## Tests

```bash
make test     # 46 tests: store, keys, breaker, SSE parsing, full proxy round-trip
make e2e      # live run against the real gateway
```

`make test` runs entirely against a stub upstream — no provider quota is spent.
