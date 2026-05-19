#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="wyoming-hid"
SERVICE_FILE="$SCRIPT_DIR/${SERVICE_NAME}.service"
CONFIG_EXAMPLE="$SCRIPT_DIR/config.yaml.example"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

check_deps() {
    local missing=()
    for cmd in python3 xdotool; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done
    for pkg in aiohttp pyaudio PyYAML; do
        if ! python3 -c "import ${pkg,,}" 2>/dev/null; then
            missing+=("python3:$pkg")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "Missing dependencies: ${missing[*]}"
        echo "Install: pip3 install aiohttp pyaudio PyYAML && apt install xdotool"
        exit 1
    fi
    echo "All dependencies OK"
}

install_service() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config not found. Creating from example..."
        cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
        echo "EDIT $CONFIG_FILE with your credentials, then re-run."
        exit 1
    fi

    mkdir -p "$SYSTEMD_DIR"
    cp "$SERVICE_FILE" "$SYSTEMD_DIR/${SERVICE_NAME}.service"

    systemctl --user daemon-reload
    echo "Service installed to $SYSTEMD_DIR/${SERVICE_NAME}.service"
}

enable_service() {
    systemctl --user enable "${SERVICE_NAME}.service"
    echo "Service enabled (will start with user session)"
}

start_service() {
    systemctl --user start "${SERVICE_NAME}.service"
    echo "Service started"
}

stop_service() {
    systemctl --user stop "${SERVICE_NAME}.service" 2>/dev/null || true
    echo "Service stopped"
}

status_service() {
    systemctl --user status "${SERVICE_NAME}.service" 2>/dev/null || true
}

uninstall_service() {
    stop_service
    systemctl --user disable "${SERVICE_NAME}.service" 2>/dev/null || true
    rm -f "$SYSTEMD_DIR/${SERVICE_NAME}.service"
    systemctl --user daemon-reload
    echo "Service uninstalled"
}

case "${1:-}" in
    install)
        check_deps
        install_service
        enable_service
        start_service
        status_service
        ;;
    start)
        start_service
        status_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        status_service
        ;;
    status)
        status_service
        ;;
    uninstall)
        uninstall_service
        ;;
    check)
        check_deps
        ;;
    *)
        echo "Usage: $0 {install|start|stop|restart|status|uninstall|check}"
        echo ""
        echo "  install   - Check deps, install config, enable & start service"
        echo "  start     - Start the service"
        echo "  stop      - Stop the service"
        echo "  restart   - Restart the service"
        echo "  status    - Show service status"
        echo "  uninstall - Remove the service"
        echo "  check     - Check dependencies only"
        exit 1
        ;;
esac
