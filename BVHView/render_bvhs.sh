#!/bin/bash
# Render all BVH files in a directory to MP4 using BVHView
#
# Usage:
#   ./render_bvhs.sh 
#   ./render_bvhs.sh <bvh_dir>
#   ./render_bvhs.sh <bvh_dir> <output_dir>
#
# Examples:
#   ./render_bvhs.sh /path/to/bvhs/BrownBear
#   ./render_bvhs.sh /path/to/bvhs/BrownBear /path/to/output

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BVHVIEW="$SCRIPT_DIR/bvhview"

BVH_DIR="${1:?Usage: $0 <bvh_dir> [output_dir]}"
OUTPUT_DIR="${2:-$BVH_DIR/mp4}"

# Validate
if [ ! -f "$BVHVIEW" ]; then
    echo "ERROR: bvhview not found at $BVHVIEW"
    exit 1
fi

if [ ! -d "$BVH_DIR" ]; then
    echo "ERROR: BVH directory not found: $BVH_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

DONE_LOG="$OUTPUT_DIR/completed.txt"

BVH_FILES=($(ls "$BVH_DIR"/*.bvh 2>/dev/null | sort))
TOTAL=${#BVH_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "ERROR: No BVH files found in $BVH_DIR"
    exit 1
fi

echo "============================================"
echo "BVH Dir   : $BVH_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Total     : $TOTAL files"
echo "============================================"

COUNT=0
SKIPPED=0
ERRORS=0

for BVH_FILE in "${BVH_FILES[@]}"; do
    BASENAME=$(basename "$BVH_FILE" .bvh)
    OUTPUT_MP4="$OUTPUT_DIR/${BASENAME}.mp4"

    COUNT=$((COUNT + 1))
    echo ""
    echo "[$COUNT/$TOTAL] $BASENAME"

    # Skip if already rendered
    if [ -f "$OUTPUT_MP4" ]; then
        echo "$BASENAME" >> "$DONE_LOG"
        echo "  SKIP: already exists -> $OUTPUT_MP4"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Render via bvhview --render
    # bvhview renders to the same path as the BVH file (replacing .bvh with .mp4)
    # so we render in-place, then move to output_dir if different
    TEMP_MP4="${BVH_FILE%.bvh}.mp4"

    if "$BVHVIEW" "$BVH_FILE" --render=true 2>/dev/null; then
        if [ -f "$TEMP_MP4" ]; then
            if [ "$TEMP_MP4" != "$OUTPUT_MP4" ]; then
                mv "$TEMP_MP4" "$OUTPUT_MP4"
            fi
            echo "  OK -> $OUTPUT_MP4"
            echo "$BASENAME" >> "$DONE_LOG"
        else
            echo "  ERROR: output file not found after render"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo "  ERROR: bvhview failed"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""
echo "============================================"
echo "Done."
echo "  Rendered : $((COUNT - SKIPPED - ERRORS))"
echo "  Skipped  : $SKIPPED"
echo "  Errors   : $ERRORS"
echo "============================================"
