#!/bin/bash
# Functional tests, then the same batch of scanner-sized pages three ways:
# plugin without local OCR, plugin with local OCR, and no plugin at all.
set -u
cd "$(dirname "$0")"
mkdir -p results && chmod 777 results
COUNT="${COUNT:-24}"

up() {
  docker compose up -d webserver >/dev/null 2>&1
  for _ in $(seq 1 90); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' integration-webserver-1 2>/dev/null)" = healthy ] && return 0
    sleep 5
  done
  echo "webserver did not become healthy"; return 1
}
run() { docker compose exec -T webserver python3 /it/run.py "$@"; }

docker compose up -d >/dev/null 2>&1
IT_LOCAL_OCR=fallback up && run functional | tee results/functional.txt
[ -n "${ONLY_FUNCTIONAL:-}" ] && { echo "functional phase done"; exit 0; }
IT_LOCAL_OCR=fallback up && run load --count "$COUNT" --label fallback | tail -16
IT_LOCAL_OCR=always up && run load --count "$COUNT" --label always | tail -16
IT_API_KEY= up && run load --count "$COUNT" --label no-plugin | tail -16
docker compose exec -T webserver grep "Gemini OCR:" /usr/src/paperless/data/log/paperless.log > results/plugin-log.txt 2>/dev/null
echo "all phases done"
