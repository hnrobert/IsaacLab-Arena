#!/usr/bin/env bash
# G1D warehouse pick-and-place inside the IsaacLab-Arena container.
# tmux-managed; the Isaac Lab GUI renders on the local desktop (:1).
#
# Modes:
#   sim       compose up + GUI run in tmux window (default)
#   headless  compose up + headless smoke run in tmux window
#   stop      kill the tmux session (container stays up)
#   down      kill the session AND compose-down the container
#   status / attach
set -e

SESSION=${SESSION:-g1d_arena}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO="${REPO_DIR}/g1d_demo/g1d_pick_place_demo.py"

DESKTOP_DISPLAY=${DESKTOP_DISPLAY:-:1}
DESKTOP_XAUTH=${DESKTOP_XAUTH:-/run/user/$(id -u)/gdm/Xauthority}
COMPOSE="$(command -v docker-compose || echo "$HOME/.local/bin/docker-compose")"

# let container clients open windows on the desktop X server
DISPLAY="${DESKTOP_DISPLAY}" XAUTHORITY="${DESKTOP_XAUTH}" \
  xhost +local:docker >/dev/null 2>&1 || true

usage() { grep '^#' "$0" | sed 's/^# \?//'; }

session_exists() { tmux has-session -t "=${SESSION}" 2>/dev/null; }

run_in_container() {
  local extra="$1"
  cd "${REPO_DIR}"
  sg docker -c "env DISPLAY=${DESKTOP_DISPLAY} \
    XAUTHORITY=${DESKTOP_XAUTH} \
    ${COMPOSE} exec -T isaaclab-arena bash -lc \
    'cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
     g1d_demo/g1d_pick_place_demo.py ${extra}'"
}

mode=${1:-sim}
case "$mode" in
sim | headless)
  # bring the stack up (idempotent)
  cd "${REPO_DIR}"
  sg docker -c "env DISPLAY=${DESKTOP_DISPLAY} \
    XAUTHORITY=${DESKTOP_XAUTH} ${COMPOSE} up -d" >/dev/null

  if session_exists; then
    echo "Restarting tmux session '${SESSION}'..."
    tmux kill-session -t "${SESSION}"
  fi

  if [ "$mode" = headless ]; then
    tmux new-session -d -s "${SESSION}" -n sim \
      "bash '${REPO_DIR}/run.sh' _sim --headless; echo '== exited, press Enter to close'; read"
  else
    tmux new-session -d -s "${SESSION}" -n sim \
      "bash '${REPO_DIR}/run.sh' _sim; echo '== exited, press Enter to close'; read"
  fi
  tmux set-window-option -t "${SESSION}:sim" remain-on-exit on >/dev/null
  if [ -t 0 ] && [ -t 1 ] && [ -z "${TMUX:-}" ]; then
    exec tmux attach -t "${SESSION}"
  else
    echo "tmux attach -t ${SESSION}"
  fi
  ;;
_sim)
  run_in_container "${@:2}"
  ;;
stop)
  tmux kill-session -t "${SESSION}" 2>/dev/null \
    && echo "Session '${SESSION}' stopped." \
    || echo "No session '${SESSION}'."
  ;;
down)
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  cd "${REPO_DIR}"
  sg docker -c "${COMPOSE} down"
  ;;
status)
  tmux list-windows -t "${SESSION}" -F \
    '#{window_name}  #{pane_current_command}  #{?pane_dead,DEAD,live}' 2>/dev/null \
    || echo "No session '${SESSION}'."
  ;;
attach)
  exec tmux attach -t "${SESSION}"
  ;;
*)
  usage
  exit 1
  ;;
esac
