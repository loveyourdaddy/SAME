#!/bin/bash
# Render all BVH files in a directory to MP4 using BVHView, on a headless machine.
#
# BVHView captures frames with LoadImageFromScreen(), i.e. it reads the back buffer
# of a real window -- there is no offscreen render path. So it needs an X display
# even when nobody is watching. This script supplies a virtual one via Xvfb.
#
# Differences from render_bvhs.sh:
#   - re-execs itself under xvfb-run, so no monitor / no DISPLAY is required
#   - passes --drawUI=false so the GUI overlay is not baked into the video
#   - keeps the Xvfb screen size in sync with bvhview's window size
#   - preflights bvhview / ffmpeg / xvfb-run and reports what is missing
#   - keeps bvhview output in a per-run log instead of discarding stderr
#   - applies a per-file timeout so one bad file cannot hang the whole batch
#
# Usage:
#   ./render_bvhs_headless.sh <bvh_dir> [output_dir]
#
# Environment overrides:
#   WIDTH=1920 HEIGHT=1080   render resolution           (default 1280x720)
#   DRAW_UI=true             keep the GUI overlay        (default false)
#   TIMEOUT=300              per-file seconds, 0=off     (default 600)
#   SOFTWARE_GL=1            force Mesa llvmpipe (no GPU in the container)
#   FORCE_XVFB=1             use Xvfb even if DISPLAY is already set
#   EXTRA_ARGS="--drawGrid=false"   extra flags passed through to bvhview

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
BVHVIEW="$SCRIPT_DIR/bvhview"

WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
DRAW_UI="${DRAW_UI:-false}"
TIMEOUT="${TIMEOUT:-600}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Mesa llvmpipe fallback for containers without a GPU. Applies to both the
# Xvfb re-exec and the already-have-a-display path.
if [ "${SOFTWARE_GL:-0}" = "1" ]; then
    export LIBGL_ALWAYS_SOFTWARE=1
fi

# ---------------------------------------------------------------------------
# Stage 1: preflight (runs before we bother spawning Xvfb)
# ---------------------------------------------------------------------------

BVH_DIR="${1:-}"
if [ -z "$BVH_DIR" ]; then
    echo "Usage: $0 <bvh_dir> [output_dir]" >&2
    exit 2
fi
OUTPUT_DIR="${2:-$BVH_DIR/mp4}"

fail() { echo "ERROR: $*" >&2; exit 1; }

if [ ! -x "$BVHVIEW" ]; then
    if [ -f "$BVHVIEW" ]; then
        fail "bvhview at $BVHVIEW is not executable. Run: chmod +x '$BVHVIEW'"
    fi
    cat >&2 <<MSG
ERROR: bvhview not found at $BVHVIEW

The binary is gitignored, so a fresh checkout has the source but no executable.
Build it on this machine:

  # one-time: raylib + raygui where the Makefile expects them (RAYLIB_DIR = ~/raylib)
  git clone --depth 1 https://github.com/raysan5/raylib.git ~/raylib/raylib
  make -C ~/raylib/raylib/src PLATFORM=PLATFORM_DESKTOP
  git clone --depth 1 https://github.com/raysan5/raygui.git ~/raylib/raygui

  make -C "$SCRIPT_DIR" BUILD_MODE=RELEASE
MSG
    exit 1
fi

[ -d "$BVH_DIR" ] || fail "BVH directory not found: $BVH_DIR"

command -v ffmpeg >/dev/null 2>&1 || fail \
    "ffmpeg not found. bvhview pipes raw frames to it via popen() and will fail silently without it. Install: apt install -y ffmpeg"

TIMEOUT_CMD=()
if [ "$TIMEOUT" != "0" ]; then
    if command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD=(timeout --signal=KILL "$TIMEOUT")
    else
        echo "WARN: 'timeout' not available, per-file timeout disabled" >&2
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2: make sure we have a display, re-execing under Xvfb if we do not
# ---------------------------------------------------------------------------

if [ -z "${BVHVIEW_HEADLESS_REEXEC:-}" ]; then
    if [ -z "${DISPLAY:-}" ] || [ "${FORCE_XVFB:-0}" = "1" ]; then
        command -v xvfb-run >/dev/null 2>&1 || fail \
            "no DISPLAY and xvfb-run not found. Install: apt install -y xvfb"

        echo "INFO: no display available, re-running under Xvfb (${WIDTH}x${HEIGHT}x24)"
        export BVHVIEW_HEADLESS_REEXEC=1
        export WIDTH HEIGHT DRAW_UI TIMEOUT EXTRA_ARGS
        exec xvfb-run -a -s "-screen 0 ${WIDTH}x${HEIGHT}x24" \
            "$SCRIPT_PATH" "$BVH_DIR" "$OUTPUT_DIR"
    fi
    echo "INFO: using existing DISPLAY=$DISPLAY (set FORCE_XVFB=1 to use Xvfb instead)"
fi

# ---------------------------------------------------------------------------
# Stage 3: render
# ---------------------------------------------------------------------------

mkdir -p "$OUTPUT_DIR"
DONE_LOG="$OUTPUT_DIR/completed.txt"
RUN_LOG="$OUTPUT_DIR/render_$(date +%Y%m%d_%H%M%S).log"

mapfile -t BVH_FILES < <(find "$BVH_DIR" -maxdepth 1 -name '*.bvh' -type f | sort)
TOTAL=${#BVH_FILES[@]}
[ "$TOTAL" -gt 0 ] || fail "No BVH files found in $BVH_DIR"

echo "============================================"
echo "BVH Dir   : $BVH_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Total     : $TOTAL files"
echo "Resolution: ${WIDTH}x${HEIGHT}   drawUI=$DRAW_UI"
echo "Display   : ${DISPLAY:-<none>}"
echo "Log       : $RUN_LOG"
echo "============================================"

COUNT=0
SKIPPED=0
ERRORS=0

for BVH_FILE in "${BVH_FILES[@]}"; do
    BASENAME="$(basename "$BVH_FILE" .bvh)"
    OUTPUT_MP4="$OUTPUT_DIR/${BASENAME}.mp4"

    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] $BASENAME"

    if [ -f "$OUTPUT_MP4" ]; then
        echo "$BASENAME" >> "$DONE_LOG"
        echo "  SKIP: already exists -> $OUTPUT_MP4"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # bvhview writes the mp4 next to the BVH file, replacing the extension
    TEMP_MP4="${BVH_FILE%.bvh}.mp4"

    # NOTE: { ...; } > file is a command group, not a subshell, so STATUS
    # assigned in here survives -- but it must be captured before the echo.
    {
        echo "--- [$COUNT/$TOTAL] $BASENAME ---"
        ${TIMEOUT_CMD[@]+"${TIMEOUT_CMD[@]}"} "$BVHVIEW" "$BVH_FILE" \
            --render=true \
            --drawUI="$DRAW_UI" \
            --screenWidth="$WIDTH" \
            --screenHeight="$HEIGHT" \
            $EXTRA_ARGS
        STATUS=$?
        echo "--- exit=$STATUS ---"
    } >> "$RUN_LOG" 2>&1

    if [ "$STATUS" -eq 137 ] || [ "$STATUS" -eq 124 ]; then
        echo "  ERROR: timed out after ${TIMEOUT}s (see $RUN_LOG)"
        ERRORS=$((ERRORS + 1))
        rm -f "$TEMP_MP4"
        continue
    fi

    if [ "$STATUS" -ne 0 ]; then
        echo "  ERROR: bvhview exited with status $STATUS (see $RUN_LOG)"
        ERRORS=$((ERRORS + 1))
        rm -f "$TEMP_MP4"
        continue
    fi

    if [ ! -f "$TEMP_MP4" ]; then
        echo "  ERROR: no output produced (see $RUN_LOG)"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    if [ "$TEMP_MP4" != "$OUTPUT_MP4" ]; then
        mv "$TEMP_MP4" "$OUTPUT_MP4"
    fi
    echo "  OK -> $OUTPUT_MP4"
    echo "$BASENAME" >> "$DONE_LOG"
done

echo ""
echo "============================================"
echo "Done."
echo "  Rendered : $((COUNT - SKIPPED - ERRORS))"
echo "  Skipped  : $SKIPPED"
echo "  Errors   : $ERRORS"
[ "$ERRORS" -gt 0 ] && echo "  Log      : $RUN_LOG"
echo "============================================"

[ "$ERRORS" -eq 0 ]
