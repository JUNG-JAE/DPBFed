#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

cleanup() {
    pkill -P $$ 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_round() {
    local round="$1"

    echo "========================================"
    echo "Round ${round} start"
    echo "========================================"

    # 1) model_server는 model_server 디렉터리에서 실행
    (
        cd "$ROOT_DIR/model_server" || exit 1
        python3 handler.py
    ) 2>&1 | tee "$LOG_DIR/model_server_round${round}.log" &
    SERVER_PID=$!

    # 서버가 listen 상태가 될 때까지 잠깐 대기
    sleep 3

    # 2) shard1은 shard1 디렉터리에서 실행
    (
        cd "$ROOT_DIR/shard1" || exit 1
        python3 main.py
    ) 2>&1 | tee "$LOG_DIR/shard1_round${round}.log" &
    SHARD1_PID=$!

    # 3) shard2는 shard2 디렉터리에서 실행
    (
        cd "$ROOT_DIR/shard2" || exit 1
        python3 main.py
    ) 2>&1 | tee "$LOG_DIR/shard2_round${round}.log" &
    SHARD2_PID=$!

    wait "$SHARD1_PID"
    wait "$SHARD2_PID"
    wait "$SERVER_PID"

    echo "========================================"
    echo "Round ${round} done"
    echo "========================================"
    echo
}

run_round 1
sleep 2
run_round 2

echo "All rounds finished."