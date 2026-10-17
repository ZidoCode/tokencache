PY := python3

.PHONY: test e2e run stats purge clean

test:            ## unit + integration tests (stub upstream, no real provider)
	$(PY) -m unittest test_tokencache -v

e2e:             ## live end-to-end against the real 9router gateway
	$(PY) e2e_real.py

run:             ## start the caching proxy
	$(PY) tokencache.py

stats:           ## print savings counters
	$(PY) tokencache.py --stats

purge:           ## clear the cache
	$(PY) tokencache.py --purge

clean:
	rm -f tokencache.sqlite tokencache.sqlite-wal tokencache.sqlite-shm
