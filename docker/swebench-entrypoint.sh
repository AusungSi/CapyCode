#!/bin/sh
set -eu

VENV=/workspace/.capy/docker-venv
if [ ! -x "$VENV/bin/python" ]; then
    # A regular venv copies pip and setuptools into every Windows bind-mounted
    # instance, creating hundreds of small files before the first tool can run.
    # Keep the venv lightweight and reuse the compatible build stack baked into
    # the image through --system-site-packages.
    python -m venv --without-pip --system-site-packages "$VENV"
    printf '%s\n' \
        '#!/bin/sh' \
        'exec /workspace/.capy/docker-venv/bin/python -m pip "$@"' \
        > "$VENV/bin/pip"
    chmod +x "$VENV/bin/pip"
    cp "$VENV/bin/pip" "$VENV/bin/pip3"
fi

# Old setup.py projects (notably Astropy) are incompatible with modern
# isolated build environments. Keep installs in the workspace venv; common
# requirements come from the image through --system-site-packages.
export PIP_NO_BUILD_ISOLATION=1
export PATH="$VENV/bin:$PATH"

# pip does not consistently honor the environment variable across versions.
# Inject the explicit flag for agent-issued installs so legacy projects use the
# compatible build dependencies already baked into the image.
if { [ "${1:-}" = "pip" ] || [ "${1:-}" = "pip3" ]; } && [ "${2:-}" = "install" ]; then
    case " ${*} " in
        *" --no-build-isolation "*) ;;
        *)
            _pip="$1"
            shift 2
            set -- "$_pip" install --no-build-isolation "$@"
            ;;
    esac
elif [ "${1:-}" = "python" ] && [ "${2:-}" = "-m" ] && [ "${3:-}" = "pip" ] && [ "${4:-}" = "install" ]; then
    case " ${*} " in
        *" --no-build-isolation "*) ;;
        *)
            shift 4
            set -- python -m pip install --no-build-isolation "$@"
            ;;
    esac
fi
exec "$@"
