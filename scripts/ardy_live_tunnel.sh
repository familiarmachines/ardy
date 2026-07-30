#!/usr/bin/env bash
# Manage the bidirectional SSH tunnel for remote ARDY and local Blender.

set -Eeuo pipefail

ARDY_REMOTE_HOST="${ARDY_REMOTE_HOST:-}"
ARDY_SSH_KEY="${ARDY_SSH_KEY:-}"
ARDY_TUNNEL_STATE_DIR="${ARDY_TUNNEL_STATE_DIR:-${XDG_STATE_HOME:-${HOME}/.local/state}/ardy-live-tunnel}"
ARDY_TUNNEL_SOCKET="${ARDY_TUNNEL_SOCKET:-${ARDY_TUNNEL_STATE_DIR}/control.sock}"
ARDY_LOCAL_API_PORT="${ARDY_LOCAL_API_PORT:-8766}"
ARDY_REMOTE_API_PORT="${ARDY_REMOTE_API_PORT:-8766}"
ARDY_LOCAL_BLENDER_PORT="${ARDY_LOCAL_BLENDER_PORT:-9876}"
ARDY_REMOTE_BLENDER_PORT="${ARDY_REMOTE_BLENDER_PORT:-9876}"


usage() {
    cat <<EOF
Usage: $(basename "$0") COMMAND

Commands:
  start     Start a detached SSH tunnel.
  stop      Stop the managed tunnel.
  restart   Restart the managed tunnel.
  status    Check the tunnel and both HTTP endpoints.
  help      Show this help.

Required environment:
  ARDY_REMOTE_HOST       SSH destination, for example azureuser@example.com

Optional environment:
  ARDY_SSH_KEY           SSH private key path
  ARDY_LOCAL_API_PORT    Local port forwarded to remote ARDY (default: ${ARDY_LOCAL_API_PORT})
  ARDY_REMOTE_API_PORT   Remote ARDY live API port (default: ${ARDY_REMOTE_API_PORT})
  ARDY_LOCAL_BLENDER_PORT
                         Local Blender control port (default: ${ARDY_LOCAL_BLENDER_PORT})
  ARDY_REMOTE_BLENDER_PORT
                         Remote reverse-forward port (default: ${ARDY_REMOTE_BLENDER_PORT})
EOF
}


log() {
    printf '[ardy-live-tunnel] %s\n' "$*"
}


die() {
    log "ERROR: $*" >&2
    exit 1
}


require_remote() {
    [[ -n "${ARDY_REMOTE_HOST}" ]] ||
        die "Set ARDY_REMOTE_HOST to the SSH destination."
}


tunnel_is_running() {
    require_remote
    [[ -S "${ARDY_TUNNEL_SOCKET}" ]] || return 1
    ssh -S "${ARDY_TUNNEL_SOCKET}" -O check "${ARDY_REMOTE_HOST}" >/dev/null 2>&1
}


start_tunnel() {
    local -a identity_args=()

    require_remote
    mkdir -p -- "${ARDY_TUNNEL_STATE_DIR}"
    chmod 700 -- "${ARDY_TUNNEL_STATE_DIR}"

    if tunnel_is_running; then
        log "Tunnel is already running."
        return 0
    fi
    rm -f -- "${ARDY_TUNNEL_SOCKET}"

    if [[ -n "${ARDY_SSH_KEY}" ]]; then
        local key_path="${ARDY_SSH_KEY/#\~/${HOME}}"
        [[ -r "${key_path}" ]] || die "SSH private key is not readable: ${key_path}"
        identity_args=(-i "${key_path}")
    fi

    ssh "${identity_args[@]}" \
        -M \
        -S "${ARDY_TUNNEL_SOCKET}" \
        -fN \
        -o BatchMode=yes \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "127.0.0.1:${ARDY_LOCAL_API_PORT}:127.0.0.1:${ARDY_REMOTE_API_PORT}" \
        -R "127.0.0.1:${ARDY_REMOTE_BLENDER_PORT}:127.0.0.1:${ARDY_LOCAL_BLENDER_PORT}" \
        "${ARDY_REMOTE_HOST}"

    tunnel_is_running || die "SSH exited without creating a working control socket."
    log "Tunnel started."
    log "Local ARDY API: http://127.0.0.1:${ARDY_LOCAL_API_PORT}"
    log "Remote Blender callback: http://127.0.0.1:${ARDY_REMOTE_BLENDER_PORT}"
}


stop_tunnel() {
    require_remote
    if tunnel_is_running; then
        ssh -S "${ARDY_TUNNEL_SOCKET}" -O exit "${ARDY_REMOTE_HOST}" >/dev/null
        log "Tunnel stopped."
    else
        log "Tunnel is not running."
    fi
    rm -f -- "${ARDY_TUNNEL_SOCKET}"
}


show_status() {
    local api_health=""
    local result=0

    if tunnel_is_running; then
        log "Tunnel is running."
    else
        log "Tunnel is stopped."
        return 1
    fi

    if api_health="$(
        curl --fail --silent --show-error --max-time 3 \
            "http://127.0.0.1:${ARDY_LOCAL_API_PORT}/health" 2>/dev/null
    )"; then
        printf '%-16s CONNECTED http://127.0.0.1:%s/health\n' "ARDY API" "${ARDY_LOCAL_API_PORT}"
        if printf '%s' "${api_health}" |
            python3 -c \
                'import json, sys; b=json.load(sys.stdin).get("blender", {}); raise SystemExit(b.get("status") != "ok" or b.get("capabilities", {}).get("motion_file_transfer") != 1)'
        then
            printf '%-16s CONNECTED remote :%s -> local :%s\n' \
                "Blender callback" "${ARDY_REMOTE_BLENDER_PORT}" "${ARDY_LOCAL_BLENDER_PORT}"
        else
            printf '%-16s UNAVAILABLE remote :%s -> local :%s\n' \
                "Blender callback" "${ARDY_REMOTE_BLENDER_PORT}" "${ARDY_LOCAL_BLENDER_PORT}"
            result=1
        fi
    else
        printf '%-16s UNAVAILABLE http://127.0.0.1:%s/health\n' "ARDY API" "${ARDY_LOCAL_API_PORT}"
        result=1
    fi

    if curl --fail --silent --show-error --max-time 3 \
        "http://127.0.0.1:${ARDY_LOCAL_BLENDER_PORT}/health" >/dev/null 2>&1; then
        printf '%-16s CONNECTED http://127.0.0.1:%s/health\n' "Local Blender" "${ARDY_LOCAL_BLENDER_PORT}"
    else
        printf '%-16s UNAVAILABLE http://127.0.0.1:%s/health\n' "Local Blender" "${ARDY_LOCAL_BLENDER_PORT}"
        result=1
    fi

    return "${result}"
}


command="${1:-help}"
case "${command}" in
    start)
        [[ $# -eq 1 ]] || die "start takes no arguments."
        start_tunnel
        ;;
    stop)
        [[ $# -eq 1 ]] || die "stop takes no arguments."
        stop_tunnel
        ;;
    restart)
        [[ $# -eq 1 ]] || die "restart takes no arguments."
        stop_tunnel
        start_tunnel
        ;;
    status)
        [[ $# -eq 1 ]] || die "status takes no arguments."
        show_status
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        die "Unknown command: ${command}"
        ;;
esac
