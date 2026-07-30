#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Manage the shared text encoder and one GPU-backed ARDY frontend:
#   1. The Viser browser demo.
#   2. The stateful live-motion API that drives Blender through an SSH tunnel.
#
# Processes are launched in their own sessions with nohup, so they survive an
# SSH disconnect. Runtime state and logs are kept outside the repository.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ARDY_PYTHON:-}" ]]; then
    : # Use the explicit override.
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    ARDY_PYTHON="${CONDA_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
    ARDY_PYTHON="$(command -v python)"
else
    ARDY_PYTHON="python"
fi

ARDY_STATE_DIR="${ARDY_STATE_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/ardy-server}"
ARDY_STARTUP_TIMEOUT="${ARDY_STARTUP_TIMEOUT:-1800}"
ARDY_STOP_TIMEOUT="${ARDY_STOP_TIMEOUT:-30}"
ARDY_TEXT_HOST="${ARDY_TEXT_HOST:-127.0.0.1}"
ARDY_TEXT_PORT="${ARDY_TEXT_PORT:-9550}"
ARDY_TEXT_DEVICE="${ARDY_TEXT_DEVICE:-cpu}"
ARDY_TEXT_FP32="${ARDY_TEXT_FP32:-1}"
ARDY_DEMO_COMPILE="${ARDY_DEMO_COMPILE:-0}"
ARDY_LIVE_HOST="${ARDY_LIVE_HOST:-127.0.0.1}"
ARDY_LIVE_PORT="${ARDY_LIVE_PORT:-8766}"
ARDY_LIVE_BLENDER_URL="${ARDY_LIVE_BLENDER_URL:-http://127.0.0.1:9876}"
ARDY_LIVE_OUTPUT_DIR="${ARDY_LIVE_OUTPUT_DIR:-${REPO_ROOT}/outputs/live_api}"
ARDY_LIVE_RENDER_MODE="${ARDY_LIVE_RENDER_MODE:-skin}"
ARDY_LIVE_DEVICE="${ARDY_LIVE_DEVICE:-}"
ARDY_LIVE_MODEL="${ARDY_LIVE_MODEL:-}"
ARDY_LIVE_LAZY_LOAD="${ARDY_LIVE_LAZY_LOAD:-0}"

# The interactive demo currently fixes these values in scripts/run_demo.py.
ARDY_DEMO_HOST="127.0.0.1"
ARDY_DEMO_PORT="2333"

TEXT_PID_FILE="${ARDY_STATE_DIR}/text-encoder.pid"
DEMO_PID_FILE="${ARDY_STATE_DIR}/demo.pid"
LIVE_PID_FILE="${ARDY_STATE_DIR}/live.pid"
TEXT_LOG="${ARDY_STATE_DIR}/text-encoder.log"
DEMO_LOG="${ARDY_STATE_DIR}/demo.log"
LIVE_LOG="${ARDY_STATE_DIR}/live.log"
TEXT_URL="http://${ARDY_TEXT_HOST}:${ARDY_TEXT_PORT}/"
DEMO_URL="http://${ARDY_DEMO_HOST}:${ARDY_DEMO_PORT}/"
LIVE_URL="http://${ARDY_LIVE_HOST}:${ARDY_LIVE_PORT}/health"

START_IN_PROGRESS=0
STARTED_TEXT=0
STARTED_DEMO=0
STARTED_LIVE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND [MODE]

Commands:
  start [MODE]       Start the text encoder and selected frontend. MODE is demo
                     (default) or live. Starting one frontend stops the other.
  stop               Stop all managed processes.
  restart [MODE]     Stop all processes, then start MODE.
  status             Show process and health status.
  logs [TARGET]      Follow logs for "text", "demo", "live", or "all" (default).
  help               Show this help.

Environment overrides:
  ARDY_PYTHON           Python executable (default: ${ARDY_PYTHON})
  ARDY_STATE_DIR        PID/log directory (default: ${ARDY_STATE_DIR})
  ARDY_STARTUP_TIMEOUT  Per-service startup timeout in seconds (default: ${ARDY_STARTUP_TIMEOUT})
  ARDY_STOP_TIMEOUT     Graceful shutdown timeout in seconds (default: ${ARDY_STOP_TIMEOUT})
  ARDY_TEXT_DEVICE      Text encoder device (default: ${ARDY_TEXT_DEVICE})
  ARDY_TEXT_FP32        Set to 1 for fp32 or 0 for bfloat16 (default: ${ARDY_TEXT_FP32})
  ARDY_DEMO_COMPILE     Set to 1 to enable the demo's default compilation mode (default: ${ARDY_DEMO_COMPILE})
  ARDY_LIVE_BLENDER_URL Blender control URL reached through reverse SSH forwarding
                        (default: ${ARDY_LIVE_BLENDER_URL})
  ARDY_LIVE_OUTPUT_DIR  Remote generated-motion directory
                        (default: ${ARDY_LIVE_OUTPUT_DIR})
  ARDY_LIVE_DEVICE      Device override such as cuda:0 (default: automatic)
  ARDY_LIVE_MODEL       Model nickname or full name (default: API default)
  ARDY_LIVE_RENDER_MODE Blender avatar mode: auto, skin, skeleton, or both
                        (default: ${ARDY_LIVE_RENDER_MODE})
  ARDY_LIVE_LAZY_LOAD   Set to 1 to defer model loading until the first prompt
                        (default: ${ARDY_LIVE_LAZY_LOAD})
EOF
}

log() {
    printf '[ardy-server] %s\n' "$*"
}

die() {
    log "ERROR: $*" >&2
    exit 1
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer (got: ${value})"
}

component_pid_file() {
    case "$1" in
        text) printf '%s\n' "${TEXT_PID_FILE}" ;;
        demo) printf '%s\n' "${DEMO_PID_FILE}" ;;
        live) printf '%s\n' "${LIVE_PID_FILE}" ;;
        *) return 1 ;;
    esac
}

component_log_file() {
    case "$1" in
        text) printf '%s\n' "${TEXT_LOG}" ;;
        demo) printf '%s\n' "${DEMO_LOG}" ;;
        live) printf '%s\n' "${LIVE_LOG}" ;;
        *) return 1 ;;
    esac
}

component_command_marker() {
    case "$1" in
        text) printf '%s\n' 'scripts/run_text_encoder_server.py' ;;
        demo) printf '%s\n' 'scripts/run_demo.py' ;;
        live) printf '%s\n' 'scripts/run_live_motion_api.py' ;;
        *) return 1 ;;
    esac
}

read_component_pid() {
    local component="$1"
    local pid_file
    local pid

    pid_file="$(component_pid_file "${component}")"
    [[ -r "${pid_file}" ]] || return 1
    IFS= read -r pid <"${pid_file}" || return 1
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    printf '%s\n' "${pid}"
}

component_is_running() {
    local component="$1"
    local pid
    local marker
    local state
    local command_line

    pid="$(read_component_pid "${component}")" || return 1
    kill -0 "${pid}" 2>/dev/null || return 1

    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "${state}" && "${state:0:1}" != "Z" ]] || return 1

    marker="$(component_command_marker "${component}")"
    [[ -r "/proc/${pid}/cmdline" ]] || return 1
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    [[ "${command_line}" == *"${marker}"* ]]
}

port_is_open() {
    local host="$1"
    local port="$2"

    "${ARDY_PYTHON}" - "${host}" "${port}" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.settimeout(1.0)
    raise SystemExit(sock.connect_ex((sys.argv[1], int(sys.argv[2]))) != 0)
PY
}

http_is_ready() {
    local url="$1"
    curl --fail --silent --show-error --max-time 3 --output /dev/null "${url}" 2>/dev/null
}

tail_failure_log() {
    local component="$1"
    local log_file

    log_file="$(component_log_file "${component}")"
    if [[ -s "${log_file}" ]]; then
        log "Last 40 lines from ${log_file}:"
        tail -n 40 "${log_file}" >&2
    fi
}

wait_for_component() {
    local component="$1"
    local url="$2"
    local deadline=$((SECONDS + ARDY_STARTUP_TIMEOUT))

    while ((SECONDS < deadline)); do
        if ! component_is_running "${component}"; then
            log "${component} exited before becoming ready." >&2
            tail_failure_log "${component}"
            return 1
        fi
        if http_is_ready "${url}"; then
            return 0
        fi
        sleep 2
    done

    log "Timed out after ${ARDY_STARTUP_TIMEOUT}s waiting for ${component} at ${url}." >&2
    tail_failure_log "${component}"
    return 1
}

write_pid_file() {
    local pid_file="$1"
    local pid="$2"
    local tmp_file="${pid_file}.tmp.$$"

    printf '%s\n' "${pid}" >"${tmp_file}"
    mv -f -- "${tmp_file}" "${pid_file}"
}

start_text_encoder() {
    local pid
    local -a command

    if component_is_running text; then
        pid="$(read_component_pid text)"
        log "Text encoder is already running (PID ${pid})."
        wait_for_component text "${TEXT_URL}" || die "Running text encoder did not become ready."
        log "Text encoder is ready at ${TEXT_URL}"
        return 0
    fi

    rm -f -- "${TEXT_PID_FILE}"
    if port_is_open "${ARDY_TEXT_HOST}" "${ARDY_TEXT_PORT}"; then
        die "Port ${ARDY_TEXT_PORT} is occupied by a process not managed by this script."
    fi

    command=(
        env
        "PYTHONUNBUFFERED=1"
        "HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}"
    )
    if [[ "${ARDY_TEXT_DEVICE}" == "cpu" ]]; then
        command+=("CUDA_VISIBLE_DEVICES=")
    fi
    command+=(
        "${ARDY_PYTHON}"
        "${REPO_ROOT}/scripts/run_text_encoder_server.py"
        --host "${ARDY_TEXT_HOST}"
        --port "${ARDY_TEXT_PORT}"
        --device "${ARDY_TEXT_DEVICE}"
    )
    if [[ "${ARDY_TEXT_FP32}" == "1" ]]; then
        command+=(--fp32)
    fi

    printf '\n=== Starting text encoder at %s ===\n' "$(date --iso-8601=seconds)" >>"${TEXT_LOG}"
    nohup setsid "${command[@]}" >>"${TEXT_LOG}" 2>&1 </dev/null &
    pid=$!
    write_pid_file "${TEXT_PID_FILE}" "${pid}"
    STARTED_TEXT=1

    sleep 1
    component_is_running text || {
        tail_failure_log text
        die "Text encoder failed to launch."
    }

    log "Waiting for the text encoder (PID ${pid}); first launch may download model weights."
    wait_for_component text "${TEXT_URL}" || die "Text encoder did not become ready."
    log "Text encoder is ready at ${TEXT_URL}"
}

start_demo() {
    local pid
    local -a command

    if component_is_running demo; then
        pid="$(read_component_pid demo)"
        log "ARDY demo is already running (PID ${pid})."
        wait_for_component demo "${DEMO_URL}" || die "Running ARDY demo did not become ready."
        log "ARDY demo is ready at ${DEMO_URL}"
        return 0
    fi

    rm -f -- "${DEMO_PID_FILE}"
    if port_is_open "${ARDY_DEMO_HOST}" "${ARDY_DEMO_PORT}"; then
        die "Port ${ARDY_DEMO_PORT} is occupied by a process not managed by this script."
    fi

    command=(
        env
        "PYTHONUNBUFFERED=1"
        "HF_ENABLE_PARALLEL_LOADING=YES"
        "TEXT_ENCODER_MODE=api"
        "TEXT_ENCODER_URL=${TEXT_URL}"
        "${ARDY_PYTHON}"
        "${REPO_ROOT}/scripts/run_demo.py"
    )
    if [[ "${ARDY_DEMO_COMPILE}" == "0" ]]; then
        command+=(--no-compile)
    fi

    printf '\n=== Starting ARDY demo at %s ===\n' "$(date --iso-8601=seconds)" >>"${DEMO_LOG}"
    nohup setsid "${command[@]}" >>"${DEMO_LOG}" 2>&1 </dev/null &
    pid=$!
    write_pid_file "${DEMO_PID_FILE}" "${pid}"
    STARTED_DEMO=1

    sleep 1
    component_is_running demo || {
        tail_failure_log demo
        die "ARDY demo failed to launch."
    }

    log "Waiting for the ARDY demo (PID ${pid})."
    wait_for_component demo "${DEMO_URL}" || die "ARDY demo did not become ready."
    log "ARDY demo is ready at ${DEMO_URL}"
}

start_live() {
    local pid
    local -a command

    if component_is_running live; then
        pid="$(read_component_pid live)"
        log "ARDY live-motion API is already running (PID ${pid})."
        wait_for_component live "${LIVE_URL}" || die "Running live-motion API did not become ready."
        log "ARDY live-motion API is ready at ${LIVE_URL}"
        return 0
    fi

    rm -f -- "${LIVE_PID_FILE}"
    if port_is_open "${ARDY_LIVE_HOST}" "${ARDY_LIVE_PORT}"; then
        die "Port ${ARDY_LIVE_PORT} is occupied by a process not managed by this script."
    fi

    command=(
        env
        "PYTHONUNBUFFERED=1"
        "HF_ENABLE_PARALLEL_LOADING=YES"
        "TEXT_ENCODER_MODE=api"
        "TEXT_ENCODER_URL=${TEXT_URL}"
        "${ARDY_PYTHON}"
        "${REPO_ROOT}/scripts/run_live_motion_api.py"
        --host "${ARDY_LIVE_HOST}"
        --port "${ARDY_LIVE_PORT}"
        --blender-url "${ARDY_LIVE_BLENDER_URL}"
        --output-dir "${ARDY_LIVE_OUTPUT_DIR}"
        --render-mode "${ARDY_LIVE_RENDER_MODE}"
        --text-encoder-mode api
        --text-encoder-url "${TEXT_URL}"
    )
    if [[ -n "${ARDY_LIVE_DEVICE}" ]]; then
        command+=(--device "${ARDY_LIVE_DEVICE}")
    fi
    if [[ -n "${ARDY_LIVE_MODEL}" ]]; then
        command+=(--model "${ARDY_LIVE_MODEL}")
    fi
    if [[ "${ARDY_LIVE_LAZY_LOAD}" == "1" ]]; then
        command+=(--lazy-load)
    fi

    printf '\n=== Starting ARDY live-motion API at %s ===\n' "$(date --iso-8601=seconds)" >>"${LIVE_LOG}"
    nohup setsid "${command[@]}" >>"${LIVE_LOG}" 2>&1 </dev/null &
    pid=$!
    write_pid_file "${LIVE_PID_FILE}" "${pid}"
    STARTED_LIVE=1

    sleep 1
    component_is_running live || {
        tail_failure_log live
        die "ARDY live-motion API failed to launch."
    }

    log "Waiting for the live-motion API (PID ${pid}); first launch may load model weights."
    wait_for_component live "${LIVE_URL}" || die "ARDY live-motion API did not become ready."
    log "ARDY live-motion API is ready at ${LIVE_URL}"
    if live_blender_is_connected; then
        log "Live Blender is connected at ${ARDY_LIVE_BLENDER_URL}"
    else
        log "Live Blender is not connected yet; establish the reverse SSH tunnel before prompting."
    fi
}

signal_component_group() {
    local pid="$1"
    local signal="$2"
    local process_group

    process_group="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${process_group}" == "${pid}" ]]; then
        kill "-${signal}" -- "-${pid}" 2>/dev/null
    else
        kill "-${signal}" "${pid}" 2>/dev/null
    fi
}

stop_component() {
    local component="$1"
    local label="$2"
    local pid_file
    local pid
    local deadline

    pid_file="$(component_pid_file "${component}")"
    if ! component_is_running "${component}"; then
        rm -f -- "${pid_file}"
        log "${label} is not running."
        return 0
    fi

    pid="$(read_component_pid "${component}")"
    log "Stopping ${label} (PID ${pid})..."
    signal_component_group "${pid}" TERM || true
    deadline=$((SECONDS + ARDY_STOP_TIMEOUT))

    while component_is_running "${component}" && ((SECONDS < deadline)); do
        sleep 1
    done

    if component_is_running "${component}"; then
        log "${label} did not stop gracefully; sending SIGKILL."
        signal_component_group "${pid}" KILL || true
    fi

    rm -f -- "${pid_file}"
}

cleanup_failed_start() {
    local exit_code=$?

    if ((exit_code != 0 && START_IN_PROGRESS == 1)); then
        log "Start failed; cleaning up processes launched by this attempt." >&2
        if ((STARTED_LIVE == 1)); then
            stop_component live "ARDY live-motion API" || true
        fi
        if ((STARTED_DEMO == 1)); then
            stop_component demo "ARDY demo" || true
        fi
        if ((STARTED_TEXT == 1)); then
            stop_component text "text encoder" || true
        fi
    fi
}

preflight() {
    if [[ "${ARDY_PYTHON}" == */* ]]; then
        [[ -x "${ARDY_PYTHON}" ]] ||
            die "Python executable not found: ${ARDY_PYTHON}. Activate the ARDY environment or set ARDY_PYTHON."
    else
        command -v "${ARDY_PYTHON}" >/dev/null 2>&1 ||
            die "Python executable not found. Activate the ARDY environment or set ARDY_PYTHON."
        ARDY_PYTHON="$(command -v "${ARDY_PYTHON}")"
    fi

    command -v curl >/dev/null 2>&1 || die "curl is required."
    command -v setsid >/dev/null 2>&1 || die "setsid is required."
    require_positive_integer ARDY_STARTUP_TIMEOUT "${ARDY_STARTUP_TIMEOUT}"
    require_positive_integer ARDY_STOP_TIMEOUT "${ARDY_STOP_TIMEOUT}"
    require_positive_integer ARDY_LIVE_PORT "${ARDY_LIVE_PORT}"
    [[ "${ARDY_TEXT_FP32}" == "0" || "${ARDY_TEXT_FP32}" == "1" ]] ||
        die "ARDY_TEXT_FP32 must be 0 or 1."
    [[ "${ARDY_DEMO_COMPILE}" == "0" || "${ARDY_DEMO_COMPILE}" == "1" ]] ||
        die "ARDY_DEMO_COMPILE must be 0 or 1."
    [[ "${ARDY_LIVE_LAZY_LOAD}" == "0" || "${ARDY_LIVE_LAZY_LOAD}" == "1" ]] ||
        die "ARDY_LIVE_LAZY_LOAD must be 0 or 1."
    case "${ARDY_LIVE_RENDER_MODE}" in
        auto|skin|skeleton|both) ;;
        *) die "ARDY_LIVE_RENDER_MODE must be auto, skin, skeleton, or both." ;;
    esac

    mkdir -p -- "${ARDY_STATE_DIR}"
    chmod 700 -- "${ARDY_STATE_DIR}"
    touch -- "${TEXT_LOG}" "${DEMO_LOG}" "${LIVE_LOG}"
    chmod 600 -- "${TEXT_LOG}" "${DEMO_LOG}" "${LIVE_LOG}"

    "${ARDY_PYTHON}" -c \
        'import ardy, gradio, motion_correction, torch, transformers, viser' ||
        die "ARDY dependencies are not importable with ${ARDY_PYTHON}."
}

live_blender_is_connected() {
    curl --fail --silent --show-error --max-time 3 "${LIVE_URL}" 2>/dev/null |
        "${ARDY_PYTHON}" -c \
            'import json, sys; b=json.load(sys.stdin).get("blender", {}); raise SystemExit(b.get("status") != "ok" or b.get("capabilities", {}).get("motion_file_transfer") != 1)'
}

start_mode() {
    local mode="$1"

    preflight
    START_IN_PROGRESS=1
    case "${mode}" in
        demo)
            stop_component live "ARDY live-motion API"
            start_text_encoder
            start_demo
            ;;
        live)
            stop_component demo "ARDY demo"
            start_text_encoder
            start_live
            ;;
        *)
            die 'MODE must be "demo" or "live".'
            ;;
    esac
    START_IN_PROGRESS=0

    log "Text encoder and ${mode} frontend are running."
    if [[ "${mode}" == "demo" ]]; then
        log "Forward the browser demo with:"
        printf '  ssh -N -L 8080:127.0.0.1:%s user@server\n' "${ARDY_DEMO_PORT}"
    else
        log "Forward the API and reverse-forward local Blender with:"
        printf '  ssh -N -L %s:127.0.0.1:%s -R 9876:127.0.0.1:9876 user@server\n' \
            "${ARDY_LIVE_PORT}" "${ARDY_LIVE_PORT}"
    fi
}

stop_all() {
    mkdir -p -- "${ARDY_STATE_DIR}"
    stop_component live "ARDY live-motion API"
    stop_component demo "ARDY demo"
    stop_component text "text encoder"
}

component_status() {
    local component="$1"
    local label="$2"
    local url="$3"
    local pid

    if component_is_running "${component}"; then
        pid="$(read_component_pid "${component}")"
        if http_is_ready "${url}"; then
            printf '%-14s RUNNING  PID %-8s %s\n' "${label}" "${pid}" "${url}"
        else
            printf '%-14s STARTING PID %-8s %s\n' "${label}" "${pid}" "${url}"
        fi
        return 0
    fi

    printf '%-14s STOPPED\n' "${label}"
    return 1
}

show_status() {
    local text_running=0
    local demo_running=0
    local live_running=0

    component_status text "Text encoder" "${TEXT_URL}" && text_running=1
    component_status demo "ARDY demo" "${DEMO_URL}" && demo_running=1
    component_status live "ARDY live API" "${LIVE_URL}" && live_running=1
    if ((live_running == 1)); then
        if live_blender_is_connected; then
            printf '%-14s CONNECTED %s\n' "Blender tunnel" "${ARDY_LIVE_BLENDER_URL}"
        else
            printf '%-14s DISCONNECTED %s\n' "Blender tunnel" "${ARDY_LIVE_BLENDER_URL}"
        fi
    fi
    printf 'Logs: %s\n' "${ARDY_STATE_DIR}"
    ((text_running == 1 && (demo_running == 1 || live_running == 1)))
}

follow_logs() {
    local target="${1:-all}"

    mkdir -p -- "${ARDY_STATE_DIR}"
    touch -- "${TEXT_LOG}" "${DEMO_LOG}" "${LIVE_LOG}"
    case "${target}" in
        text) tail -n 100 -F "${TEXT_LOG}" ;;
        demo) tail -n 100 -F "${DEMO_LOG}" ;;
        live) tail -n 100 -F "${LIVE_LOG}" ;;
        all) tail -n 100 -F "${TEXT_LOG}" "${DEMO_LOG}" "${LIVE_LOG}" ;;
        *) die 'logs target must be "text", "demo", "live", or "all".' ;;
    esac
}

trap cleanup_failed_start EXIT

command="${1:-help}"
case "${command}" in
    start)
        [[ $# -le 2 ]] || die "start accepts at most one MODE."
        start_mode "${2:-demo}"
        ;;
    stop)
        [[ $# -eq 1 ]] || die "stop takes no additional arguments."
        stop_all
        ;;
    restart)
        [[ $# -le 2 ]] || die "restart accepts at most one MODE."
        stop_all
        start_mode "${2:-demo}"
        ;;
    status)
        [[ $# -eq 1 ]] || die "status takes no additional arguments."
        show_status
        ;;
    logs)
        [[ $# -le 2 ]] || die "logs accepts at most one target."
        follow_logs "${2:-all}"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        die "Unknown command: ${command}"
        ;;
esac
